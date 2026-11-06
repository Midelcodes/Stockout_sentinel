"""Facts JSON and precomputed facility briefs. The deployed app never calls an LLM."""
import hashlib
import json
import os
import sys
import time

import httpx2
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from src.guardrail import allowed_numbers, find_problems, guarded_brief, local_month, template_brief
from src.io import ROOT, SERIES_KEYS, load_raw
from src.stock import NOT_STOCKED, ORDERED_NEVER_RECEIVED

PROMPT_VERSION = "5"
LANGUAGES = {"en": "English", "fr": "French"}
SYSTEM_PROMPT = """You write short stock briefs for a district pharmacist in Côte d'Ivoire. \
FACTS is your only source; it describes one health facility at a historical date.

Write the whole brief in the requested language, including month names and stock labels. Only product names and site codes stay exactly as they appear in FACTS.

Structure:
- First, 3 to 5 sentences on the facility's situation, as one paragraph. Name at most 3 of the most urgent products and give each one's units in stock and its next-month p50 to p90 range. Do not list later months.
- Then a blank line and up to 3 suggested actions as bullets starting with "- ". Do not number them.

The p50 and p90 lists give the months ahead in the order of horizons.

Rules for numbers:
- Copy every number exactly from FACTS. Do no arithmetic, add no new numbers, and give no percentages unless FACTS has them.
- Write every quantity as digits, not words.
- Express uncertainty as the p50 to p90 range.
- Refer to products by name, and to sites by site code.
- Null means the value is unknown. Do not estimate it.
- Never show FACTS field names such as stock_end or p50 list names as words in a sentence; say "units in stock" and "p50 to p90" instead.

Always:
- If there are transfers, say in a sentence of the paragraph that they are suggestions for human review and that their expiry dates are simulated.
- If low_data_quality is true for any product, say in the paragraph that the site's data quality is low.
- If a product you name has data_age_months above 0, say how many months old its stock figure is.
- An "Unverified zero" product reported zero, but that may not be a real stockout. Say it needs checking; do not call it a stockout.

Reply with the brief only, with no heading or preamble."""
FRENCH_GUIDE = """French wording to use:
- the date: {month}
- site or facility: site (never "formation")
- units in stock: unités en stock
- human review: validation humaine
- expiry dates: dates de péremption
- simulated: simulées
- stockout risk: risque de rupture
- transfer: transfert
No English words may remain, apart from product names and "p50" and "p90"."""


_next_request = 0.0


class LLMUnavailable(Exception):
    """Quota, overload or network trouble. No brief is lost: the next run resumes."""


def _whole(value):
    return None if pd.isna(value) else int(round(value))


def _product_facts(row, forecast, name):
    no_forecast = row.label in (NOT_STOCKED, ORDERED_NEVER_RECEIVED)
    ordered = forecast.sort_values("horizon")
    return {
        "product_name": name,
        "mean": [None if no_forecast else _whole(v) for v in ordered["mean"]],
        "p50": [None if no_forecast else _whole(v) for v in ordered["p50"]],
        "p90": [None if no_forecast else _whole(v) for v in ordered["p90"]],
        "stock_end": _whole(row.stock_end),
        "data_age_months": _whole(row.data_age_months),
        "months_of_stock": None if pd.isna(row.mos) else round(float(row.mos), 1),
        "days_to_stockout": _whole(row.days_to_stockout),
        "label": row.label,
        "low_data_quality": bool(row.low_data_quality),
    }


def _transfer_facts(rows, other_column, names):
    return [
        {
            "product_name": names.get(t.product_code, t.product_code),
            "quantity": int(t.quantity),
            "other_site": getattr(t, other_column),
            "SYNTHETIC_expiry_date": t.synthetic_expiry_date.strftime("%Y-%m-%d"),
        }
        for t in rows.itertuples()
    ]


def build_facts(status, forecasts, transfers, names):
    """One FACTS dict per site. This is all the LLM ever sees."""
    forecasts_by_series = dict(tuple(forecasts.groupby(SERIES_KEYS)))
    facts = {}
    for site, rows in status.groupby("site_code"):
        rows = rows.assign(product_name=rows["product_code"].map(lambda c: names.get(c, c)))
        rows = rows.sort_values("product_name")
        products = [
            _product_facts(r, forecasts_by_series[(site, r.product_code)], r.product_name)
            for r in rows.itertuples()
        ]
        facts[site] = {
            "site_code": site,
            "district": rows["district"].iloc[0],
            "as_of_label": rows["as_of"].iloc[0].strftime("%B %Y"),
            "horizons": sorted(int(h) for h in forecasts["horizon"].unique()),
            "label_counts": {k: int(v) for k, v in rows["label"].value_counts().sort_index().items()},
            "products": products,
            "transfers_in": _transfer_facts(transfers[transfers["to_site"] == site], "from_site", names),
            "transfers_out": _transfer_facts(transfers[transfers["from_site"] == site], "to_site", names),
        }
    return facts


def cache_key(facts, lang, model):
    payload = json.dumps(
        {"facts": facts, "lang": lang, "prompt": PROMPT_VERSION, "model": model},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def user_message(facts, lang, feedback=None):
    message = f"Language: {LANGUAGES[lang]}\n\nFACTS:\n{json.dumps(facts, ensure_ascii=False, indent=1)}"
    if lang == "fr":
        message += "\n\n" + FRENCH_GUIDE.format(month=local_month(facts["as_of_label"], lang))
    if feedback:
        draft, problems = feedback
        reasons = "\n".join(f"- {problem}" for problem in problems)
        message += (
            f"\n\nYour previous draft was rejected for these reasons:\n{reasons}\n"
            f"Rewrite it and fix all of them.\n\nPrevious draft:\n{draft}"
        )
    return message


def _pace(cfg):
    """Space request starts evenly, counting time already spent waiting on the last reply."""
    global _next_request
    time.sleep(max(0, _next_request - time.monotonic()))
    _next_request = time.monotonic() + 60 / cfg["llm_requests_per_minute"]


def _send(client, model, cfg, message):
    _pace(cfg)
    return client.models.generate_content(
        model=model,
        contents=message,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=cfg["llm_temperature"],
            max_output_tokens=cfg["llm_max_tokens"],
            seed=cfg["seed"],
        ),
    )


def _is_temporary(error):
    return isinstance(error, httpx2.HTTPError) or error.code == 429 or error.code >= 500


def call_model(client, model, cfg, message):
    """Return (text, model that answered). Text is None if the attempt is unusable.

    Quota and server errors are not the brief's fault, so they raise instead of
    using up a guardrail attempt.
    """
    for wait_first in (False, True):
        if wait_first:
            time.sleep(cfg["llm_rate_limit_wait_seconds"])
        try:
            response = _send(client, model, cfg, message)
            break
        except (errors.APIError, httpx2.HTTPError) as error:
            if not _is_temporary(error):
                raise
            problem = error
    else:
        raise LLMUnavailable(str(problem))

    answered_by = response.model_version or model
    candidates = response.candidates or []
    if not candidates or candidates[0].finish_reason != types.FinishReason.STOP:
        return None, answered_by
    return (response.text or "").strip() or None, answered_by


def write_brief(facts, lang, client, model, cfg):
    answered_by = model

    def generate(feedback):
        nonlocal answered_by
        text, answered_by = call_model(client, model, cfg, user_message(facts, lang, feedback))
        return text

    brief = guarded_brief(facts, lang, generate, cfg["llm_max_attempts"])
    used_model = brief["guardrail_status"] != "template_fallback"
    return brief | {"model_id": answered_by if used_model else None}


def offline_brief(facts, lang):
    """A template brief for a run with no API access. attempts=0 marks it as never sent."""
    return {
        "text": template_brief(facts, lang), "guardrail_status": "template_fallback",
        "attempts": 0, "model_id": None,
    }


def _usable_cached(facts, lang, models, previous):
    """The cached brief from any listed model, if it still passes today's checks."""
    for model in models:
        key = cache_key(facts, lang, model)
        cached = previous.get(key)
        if cached and cached["attempts"] > 0 and not find_problems(
            cached["text"], facts, allowed_numbers(facts), lang
        ):
            return key, cached
    return None


def generate_briefs(all_facts, cfg, client, models, previous, save=None):
    """Reuse briefs the model was asked for and write the rest, saving after each.

    Cached briefs are rechecked, so a tighter guardrail also reaches them.
    Offline templates cost nothing, so they are rebuilt each run. When a model
    stops answering the next one takes over, because the free quota is per
    model. If none is left, the remaining sites get offline templates and a
    later run picks them up.
    """
    briefs, pending = {}, []
    for site, facts in all_facts.items():
        for lang in LANGUAGES:
            found = _usable_cached(facts, lang, models, previous)
            if found:
                briefs[found[0]] = found[1]
            else:
                pending.append((site, lang))

    def record(site, lang, key, brief):
        briefs[key] = {"site_code": site, "lang": lang, **brief, "cache_key": key}

    for model in models if client is not None else []:
        try:
            while pending:
                site, lang = pending[0]
                brief = write_brief(all_facts[site], lang, client, model, cfg)
                record(site, lang, cache_key(all_facts[site], lang, model), brief)
                pending.pop(0)
                if save:
                    save(list(briefs.values()))
        except LLMUnavailable as error:
            print(f"  {model} paused, {len(pending)} briefs left: {error}", file=sys.stderr)
        except errors.APIError as error:
            print(f"  {model} stopped: {error}", file=sys.stderr)

    for site, lang in pending:
        record(site, lang, cache_key(all_facts[site], lang, models[0]), offline_brief(all_facts[site], lang))
    return sorted(briefs.values(), key=lambda b: (b["site_code"], b["lang"]))


def load_previous(path):
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {brief["cache_key"]: brief for brief in json.load(f)}


def make_client():
    key = os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=key) if key else None


def print_summary(briefs, cfg):
    unsent = sum(b["attempts"] == 0 for b in briefs)
    if unsent:
        print(f"  {unsent} of {len(briefs)} briefs are offline templates, not yet sent to the model")
    statuses = pd.Series([b["guardrail_status"] for b in briefs])
    for status, share in statuses.value_counts(normalize=True).items():
        print(f"  {status}: {share:.1%} ({int((statuses == status).sum())} of {len(briefs)})")
    sites = sorted({b["site_code"] for b in briefs})
    sample = np.random.default_rng(cfg["seed"]).choice(sites, size=min(5, len(sites)), replace=False)
    for site in sample:
        for brief in briefs:
            if brief["site_code"] == site:
                print(f"\n--- {site} [{brief['lang']}] {brief['guardrail_status']}\n{brief['text']}")


def run(cfg):
    load_dotenv(ROOT / ".env")
    main_model = os.getenv("LLM_MODEL") or cfg["llm_default_model"]
    models = list(dict.fromkeys([main_model, *cfg["llm_backup_models"]]))
    outputs = cfg["paths"]["outputs"]
    path = outputs / "briefs.json"

    tables = load_raw(cfg)
    products = tables["products"]
    names = dict(zip(products["product_code"], products["product_name"].str.strip()))
    all_facts = build_facts(
        pd.read_parquet(outputs / "stock_status.parquet"),
        pd.read_parquet(outputs / "forecasts.parquet"),
        pd.read_parquet(outputs / "transfers.parquet"),
        names,
    )

    client = make_client()
    if client is None:
        print("  WARNING: GEMINI_API_KEY is not set. Uncached briefs use the template.", file=sys.stderr)

    def save(briefs):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(briefs, f, ensure_ascii=False, indent=1)

    briefs = generate_briefs(all_facts, cfg, client, models, load_previous(path), save)
    save(briefs)
    print_summary(briefs, cfg)
    return "done"
