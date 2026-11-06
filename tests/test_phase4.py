import json
from decimal import Decimal
from types import SimpleNamespace

import httpx2
import pandas as pd
import pytest
from google.genai import errors, types

from src import llm_brief
from src.guardrail import (
    allowed_numbers,
    extract_numbers,
    find_offending,
    find_problems,
    find_range_problems,
    find_shape_problems,
    guarded_brief,
    parse_number,
    sentence_count,
    template_brief,
    template_sentences,
)

GOOD = "Stock is 120 pieces. Demand is 135 pieces. Ranges are uncertain.\n\n- Check stock."
GOOD_2 = GOOD.replace("135", "150")
BAD_NUMBER = GOOD.replace("135", "9999")

CFG = {
    "llm_max_attempts": 3, "llm_max_tokens": 1000, "llm_temperature": 0, "seed": 42,
    "llm_requests_per_minute": 10**9, "llm_rate_limit_wait_seconds": 0,
}


def product(name="NORISTERAT 200 MG", **overrides):
    return {
        "product_name": name, "p50": [120, 135, 150], "p90": [180, 190, 210], "stock_end": 1250,
        "months_of_stock": 4.4, "days_to_stockout": 311, "label": "Stockout risk",
        "low_data_quality": False,
    } | overrides


def facts(**overrides):
    """SYNTHETIC FACTS for one facility."""
    return {
        "site_code": "C1399", "district": "ABENGOUROU", "as_of_label": "June 2019",
        "horizons": [1, 2, 3], "label_counts": {"Overstocked": 7, "Stockout risk": 1},
        "products": [product()], "transfers_in": [], "transfers_out": [],
    } | overrides


TRANSFER = {
    "product_name": "NORISTERAT 200 MG", "quantity": 44, "other_site": "C2001",
    "SYNTHETIC_expiry_date": "2019-12-03",
}


def values(text, lang):
    return [value for _, value in extract_numbers(text, lang)]


# Number parsing

@pytest.mark.parametrize("token, lang, expected", [
    ("1250", "en", "1250"),
    ("1,250", "en", "1250"),
    ("1,250.5", "en", "1250.5"),
    ("1 250", "en", "1250"),
    ("4.4", "en", "4.4"),
    ("1 250,5", "fr", "1250.5"),
    ("1 250,5", "fr", "1250.5"),
    ("1 250,5", "fr", "1250.5"),
    ("1.250,5", "fr", "1250.5"),
    ("1.250", "fr", "1250"),
    ("4,4", "fr", "4.4"),
    ("1.5", "fr", "1.5"),
])
def test_parse_number(token, lang, expected):
    assert parse_number(token, lang) == Decimal(expected)


def test_extracts_french_grouped_number_as_one_value():
    assert values("Stock de 1 250,5 unités", "fr") == [Decimal("1250.5")]


def test_extracts_english_list_with_commas_as_separate_values():
    assert values("Demand of 120, 135, and 150.", "en") == [120, 135, 150]


def test_sentence_final_period_is_not_a_decimal():
    assert values("There are 12.", "en") == [12]


def test_four_digit_year_does_not_merge_with_the_next_number():
    assert values("June 2019 120 units", "en") == [2019, 120]


def test_range_splits_into_two_numbers():
    assert values("120–180 units", "en") == [120, 180]


# Allowed set

def test_allowed_set_covers_values_names_and_dates():
    allowed = allowed_numbers(facts(transfers_in=[TRANSFER]))
    for number in ["1250", "4.4", "200", "120", "1399", "2019", "12", "3"]:
        assert Decimal(number) in allowed


def test_allowed_set_does_not_merge_json_list_values():
    assert Decimal("120135150") not in allowed_numbers(facts())


# Guardrail: invented numbers

def test_invented_number_is_rejected():
    assert find_offending("Stock is 9999 units.", allowed_numbers(facts()), "en") == ["9999"]


def test_arithmetic_result_is_rejected():
    assert find_offending("Total demand is 405.", allowed_numbers(facts()), "en") == ["405"]


def test_invented_percentage_is_rejected():
    assert find_offending("Stock fell by 37%.", allowed_numbers(facts()), "en") == ["37"]


def test_invented_french_number_is_rejected():
    assert find_offending("Stock de 2 500,5 unités.", allowed_numbers(facts()), "fr") == ["2 500,5"]


def test_rounded_decimal_is_rejected():
    assert find_offending("About 4.5 months of stock.", allowed_numbers(facts()), "en") == ["4.5"]


def test_product_code_digits_are_not_allowed():
    assert find_offending("Product AS27134 is low.", allowed_numbers(facts()), "en") == ["27134"]


def test_repeated_offenders_are_listed_once():
    assert find_offending("9999 and 9999.", allowed_numbers(facts()), "en") == ["9999"]


# Guardrail: numbers in the allowed set

def test_numbers_copied_from_facts_pass():
    text = "C1399 holds 1,250 units, 4.4 months, against 120 to 180 (p50 to p90) as of June 2019."
    assert find_offending(text, allowed_numbers(facts()), "en") == []


def test_french_formatting_of_allowed_numbers_passes():
    text = "C1399 détient 1 250 unités, soit 4,4 mois, pour une demande de 120 à 180."
    assert find_offending(text, allowed_numbers(facts()), "fr") == []


def test_trailing_zero_counts_as_the_same_number():
    assert find_offending("Stock is 120.0 units.", allowed_numbers(facts()), "en") == []


def test_dates_rewritten_from_facts_pass():
    allowed = allowed_numbers(facts(transfers_in=[TRANSFER]))
    assert find_offending("Expires 3 December 2019.", allowed, "en") == []
    assert find_offending("Péremption le 03/12/2019.", allowed, "fr") == []


def test_product_name_decimal_passes_in_french():
    allowed = allowed_numbers(facts(products=[product("LEVONORGESTREL 1.5 MG")]))
    assert find_offending("LEVONORGESTREL 1.5 MG est en risque.", allowed, "fr") == []


# Retry and fallback

def scripted(*drafts):
    calls = []

    def generate(feedback):
        calls.append(feedback)
        return drafts[len(calls) - 1]

    return generate, calls


def test_clean_first_draft_is_passed():
    generate, calls = scripted(GOOD)
    brief = guarded_brief(facts(), "en", generate, 3)
    assert brief == {"text": GOOD, "guardrail_status": "passed", "attempts": 1}
    assert calls == [None]


def test_retry_sends_offending_numbers_back_and_passes():
    generate, calls = scripted(BAD_NUMBER, GOOD)
    brief = guarded_brief(facts(), "en", generate, 3)
    assert brief["guardrail_status"] == "passed_after_retry"
    assert brief["attempts"] == 2
    assert calls[1] == (BAD_NUMBER, ["number not in FACTS: 9999"])


def test_fallback_after_the_last_failed_attempt():
    generate, calls = scripted("9991", "9992", "9993", "never used")
    brief = guarded_brief(facts(), "en", generate, 3)
    assert brief["guardrail_status"] == "template_fallback"
    assert brief["attempts"] == 3
    assert len(calls) == 3
    assert brief["text"] == template_brief(facts(), "en")


def test_unusable_attempts_count_towards_the_limit():
    generate, calls = scripted(None, None, None)
    assert guarded_brief(facts(), "en", generate, 3)["guardrail_status"] == "template_fallback"
    assert len(calls) == 3


def test_unusable_attempt_can_still_be_followed_by_a_pass():
    generate, _ = scripted(None, GOOD)
    assert guarded_brief(facts(), "en", generate, 3)["guardrail_status"] == "passed_after_retry"


# Brief shape

def shape(text, lang="en", f=None):
    return find_shape_problems(text, f or facts(), lang)


def test_a_well_formed_brief_has_no_shape_problems():
    assert shape(GOOD) == []


def test_too_few_sentences_are_rejected():
    assert shape("One sentence only. Two now.") == ["the paragraph has 2 sentences; it must have 3 to 5"]


def test_too_many_sentences_are_rejected():
    assert shape("A. B. C. D. E. F.") == ["the paragraph has 6 sentences; it must have 3 to 5"]


def test_periods_in_product_names_and_decimals_do_not_count_as_sentences():
    f = facts(products=[product("LEVONORGESTREL 1.5 MG CP. BOITE")])
    text = "LEVONORGESTREL 1.5 MG CP. BOITE is low. It has 4.4 months. Ranges vary."
    assert sentence_count(text, f) == 3


def test_too_many_bullets_are_rejected():
    text = GOOD + "\n- Two.\n- Three.\n- Four."
    assert shape(text) == ["there are 4 bullets; the limit is 3"]


def test_bullets_do_not_count_as_sentences():
    assert sentence_count("One. Two. Three.", facts()) == 3
    assert shape(GOOD + "\n- Second bullet.\n- Third bullet.") == []


def test_field_names_are_rejected():
    assert shape(GOOD.replace("Stock is", "The stock_end is")) == ["field name in the text: stock_end"]


def test_the_word_null_is_rejected():
    text = "Le site a null unités. Le stock est suivi. Les risques sont notés.\n\n- Vérifier."
    assert shape(text, "fr") == ["the word null appears; leave unknown values out instead"]


def test_transfers_at_a_site_with_none_are_rejected():
    text = "Stock is low. Transfers are suggestions. Ranges vary.\n\n- Check stock."
    assert shape(text) == ["this site has no transfers, so do not mention any"]
    assert shape(text.replace("Transfers", "Transferts"), "fr") == [
        "this site has no transfers, so do not mention any"
    ]


def test_transfers_are_fine_where_they_exist():
    text = "Stock is low. Transfers are suggestions. Ranges vary.\n\n- Check stock."
    assert shape(text, f=facts(transfers_in=[TRANSFER])) == []


def test_wrong_range_for_a_product_is_rejected():
    text = "NORISTERAT 200 MG has a range of 135 to 180. Stock is low. Ranges vary."
    assert find_range_problems(text, facts()) == [
        "wrong range for NORISTERAT 200 MG: 135 to 180 is not its p50 to p90"
    ]


def test_wrong_french_range_is_rejected():
    assert find_range_problems("NORISTERAT 200 MG a une plage de 180 à 180.", facts()) != []


def test_correct_ranges_for_any_horizon_pass():
    text = "NORISTERAT 200 MG has 120 to 180, then 135 to 190, and 150–210."
    assert find_range_problems(text, facts()) == []


def test_range_is_checked_against_the_product_it_follows():
    f = facts(products=[
        product("FIRST"), product("SECOND", p50=[1, 1, 1], p90=[9, 9, 9]),
    ])
    assert find_range_problems("FIRST has 120 to 180. SECOND has 1 to 9.", f) == []
    assert find_range_problems("FIRST has 1 to 9. SECOND has 120 to 180.", f) != []


def test_dates_and_the_words_p50_to_p90_are_not_ranges():
    f = facts(transfers_in=[TRANSFER])
    text = "NORISTERAT 200 MG (p50 to p90 is 120 to 180) expires 2019-12-03."
    assert find_range_problems(text, f) == []


def test_a_wrong_range_triggers_a_retry():
    bad = "NORISTERAT 200 MG has 135 to 180. Demand varies. Ranges are uncertain.\n\n- Check stock."
    generate, calls = scripted(bad, GOOD)
    assert guarded_brief(facts(), "en", generate, 3)["guardrail_status"] == "passed_after_retry"
    assert "wrong range" in calls[1][1][0]


def test_a_range_for_a_product_with_no_forecast_is_rejected():
    f = facts(products=[product(p50=[None] * 3, p90=[None] * 3)])
    assert find_range_problems("NORISTERAT 200 MG has 120 to 180.", f) != []


def test_english_left_in_french_is_rejected():
    text = "Le site est suivi. Le p50 to p90 est de 120 à 180. Une human review est requise.\n\n- Vérifier."
    assert shape(text, "fr") == [
        "English left in the French text: human review",
        "English left in the French text: p50 to p90",
    ]


def test_english_wording_is_fine_in_english():
    assert shape("Units are low. The p50 to p90 range is wide. Human review is needed.", "en") == []


def test_shape_problem_triggers_a_retry_with_the_reason():
    generate, calls = scripted("Too short.", GOOD)
    brief = guarded_brief(facts(), "en", generate, 3)
    assert brief["guardrail_status"] == "passed_after_retry"
    assert calls[1] == ("Too short.", ["the paragraph has 1 sentences; it must have 3 to 5"])


def test_lasting_shape_problem_falls_back_to_the_template():
    generate, _ = scripted("Too short.", "Too short.", "Too short.")
    brief = guarded_brief(facts(), "en", generate, 3)
    assert brief["guardrail_status"] == "template_fallback"


# Template

FACT_VARIANTS = {
    "plain": facts(),
    "transfers": facts(transfers_in=[TRANSFER], transfers_out=[TRANSFER | {"quantity": 9}]),
    "low_quality": facts(products=[product(low_data_quality=True)]),
    "everything_at_once": facts(
        products=[product("A", low_data_quality=True), product("B"), product("C")],
        transfers_in=[TRANSFER] * 3, transfers_out=[TRANSFER] * 3,
    ),
    "no_current_data": facts(
        label_counts={"No current data": 1},
        products=[product(label="No current data", stock_end=None, months_of_stock=None, days_to_stockout=None)],
    ),
    "never_stocked": facts(
        label_counts={"Not stocked": 1},
        products=[product(
            label="Not stocked", p50=[None] * 3, p90=[None] * 3, stock_end=0,
            months_of_stock=None, days_to_stockout=None,
        )],
    ),
}


@pytest.mark.parametrize("lang", ["en", "fr"])
@pytest.mark.parametrize("variant", FACT_VARIANTS)
def test_template_passes_its_own_guardrail(variant, lang):
    f = FACT_VARIANTS[variant]
    assert find_offending(template_brief(f, lang), allowed_numbers(f), lang) == []


@pytest.mark.parametrize("lang", ["en", "fr"])
@pytest.mark.parametrize("variant", FACT_VARIANTS)
def test_template_passes_the_shape_check(variant, lang):
    f = FACT_VARIANTS[variant]
    assert find_shape_problems(template_brief(f, lang), f, lang) == []


@pytest.mark.parametrize("lang", ["en", "fr"])
@pytest.mark.parametrize("variant", FACT_VARIANTS)
def test_template_shape(variant, lang):
    assert 3 <= len(template_sentences(FACT_VARIANTS[variant], lang)) <= 5
    bullets = template_brief(FACT_VARIANTS[variant], lang).split("\n\n")[1]
    assert 1 <= len(bullets.split("\n")) <= 3
    assert all(line.startswith("- ") for line in bullets.split("\n"))


def test_template_keeps_required_notes_when_a_site_has_everything():
    text = template_brief(FACT_VARIANTS["everything_at_once"], "en")
    assert "suggestions for human review" in text
    assert "lowest tier" in text
    assert "ranges, not certainties" in text


def test_template_counts_agree_with_their_nouns():
    one = facts(products=[product(stock_end=1, days_to_stockout=1)], transfers_in=[TRANSFER | {"quantity": 1}])
    zero = facts(products=[product(stock_end=0, days_to_stockout=0)])
    assert "1 unit on hand" in template_brief(one, "en")
    assert "about 1 day of stock" in template_brief(one, "en")
    assert "receive 1 unit of" in template_brief(one, "en")
    assert "0 units on hand" in template_brief(zero, "en")
    assert "0 unité en stock" in template_brief(zero, "fr")
    assert "0 jour de stock" in template_brief(zero, "fr")
    assert "1250 unités en stock" in template_brief(facts(), "fr")


def test_template_is_deterministic():
    assert template_brief(facts(), "fr") == template_brief(facts(), "fr")


def test_template_flags_transfers_as_simulated_suggestions():
    text = template_brief(facts(transfers_in=[TRANSFER]), "en")
    assert "suggestions for human review" in text
    assert "simulated" in text
    assert "simulées" in template_brief(facts(transfers_in=[TRANSFER]), "fr")


def test_template_mentions_low_data_quality():
    assert "lowest tier" in template_brief(FACT_VARIANTS["low_quality"], "en")
    assert "lowest tier" not in template_brief(facts(), "en")


def test_french_template_translates_the_month_and_labels():
    text = template_brief(facts(), "fr")
    assert "juin 2019" in text
    assert "Risque de rupture 1" in text


# Facts

def status_rows():
    return pd.DataFrame({
        "site_code": ["S1", "S1", "S2"], "product_code": ["P1", "P2", "P1"], "district": "D",
        "as_of": pd.Timestamp("2019-06-01"), "stock_end": [100.0, 0.0, float("nan")],
        "mos": [4.44, float("nan"), float("nan")], "days_to_stockout": [133.2, float("nan"), float("nan")],
        "label": ["Overstocked", "Not stocked", "No current data"],
        "low_data_quality": [True, True, False],
        "secret_raw_column": [1, 2, 3],
    })


def forecast_rows():
    rows = []
    for site, code in [("S1", "P1"), ("S1", "P2"), ("S2", "P1")]:
        for horizon in (1, 2, 3):
            rows.append({"site_code": site, "product_code": code, "horizon": horizon,
                         "p50": 22.4 + horizon, "p90": 30.6 + horizon})
    return pd.DataFrame(rows)


def transfer_rows():
    return pd.DataFrame({
        "product_code": ["P1"], "from_site": ["S1"], "to_site": ["S2"], "quantity": [5],
        "synthetic_expiry_date": [pd.Timestamp("2019-12-03")],
    })


def built():
    names = {"P1": "NAME ONE", "P2": "NAME TWO"}
    return llm_brief.build_facts(status_rows(), forecast_rows(), transfer_rows(), names)


def test_facts_round_units_and_mos():
    p = built()["S1"]["products"][0]
    assert p["p50"] == [23, 24, 25]
    assert p["p90"] == [32, 33, 34]
    assert p["stock_end"] == 100
    assert p["months_of_stock"] == 4.4
    assert p["days_to_stockout"] == 133
    assert p["low_data_quality"] is True


def test_facts_header():
    f = built()["S1"]
    assert (f["site_code"], f["district"], f["as_of_label"]) == ("S1", "D", "June 2019")
    assert f["horizons"] == [1, 2, 3]
    assert f["label_counts"] == {"Not stocked": 1, "Overstocked": 1}


def test_facts_hide_forecasts_for_series_that_never_held_stock():
    p = built()["S1"]["products"][1]
    assert p["p50"] == [None, None, None]
    assert p["p90"] == [None, None, None]


def test_facts_keep_unknowns_null():
    p = built()["S2"]["products"][0]
    assert p["stock_end"] is None
    assert p["months_of_stock"] is None


def test_facts_list_transfers_both_ways_and_mark_expiry_synthetic():
    f = built()
    assert f["S1"]["transfers_out"] == [{
        "product_name": "NAME ONE", "quantity": 5, "other_site": "S2",
        "SYNTHETIC_expiry_date": "2019-12-03",
    }]
    assert f["S2"]["transfers_in"][0]["other_site"] == "S1"
    assert f["S1"]["transfers_in"] == []


def test_facts_contain_nothing_but_the_facts():
    text = json.dumps(built())
    assert "secret_raw_column" not in text
    assert "P1" not in text


def test_facts_are_json_serializable_with_template_for_every_site():
    for f in built().values():
        for lang in ("en", "fr"):
            assert template_brief(f, lang)


# Prompt and API call

def reply(text=GOOD, finish=types.FinishReason.STOP, model="fake-model-001"):
    candidate = SimpleNamespace(finish_reason=finish)
    return SimpleNamespace(candidates=[candidate], text=text, model_version=model)


def api_error(code):
    kind = errors.ClientError if code < 500 else errors.ServerError
    return kind(code, {"error": {"code": code, "message": "fake", "status": "FAKE"}})


class FakeClient:
    """Stands in for the Gemini client. Records requests, makes no network call."""

    def __init__(self, *replies):
        self.requests = []
        self.replies = list(replies)
        self.models = SimpleNamespace(generate_content=self.generate_content)

    def generate_content(self, **request):
        self.requests.append(request)
        answer = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(answer, Exception):
            raise answer
        return reply(answer) if isinstance(answer, str) else answer


def test_prompt_states_the_required_rules():
    prompt = llm_brief.SYSTEM_PROMPT
    for phrase in ["3 to 5 sentences", "up to 3 suggested actions", "no arithmetic",
                   "p50 to p90", "human review", "simulated", "low_data_quality"]:
        assert phrase in prompt


def test_user_message_carries_facts_and_retry_feedback():
    message = llm_brief.user_message(facts(), "fr", ("Draft", ["9999"]))
    assert "Language: French" in message
    assert '"site_code": "C1399"' in message
    assert "9999" in message
    assert "Draft" in message


def test_french_message_carries_the_glossary_and_translated_month():
    message = llm_brief.user_message(facts(), "fr")
    assert "the date: juin 2019" in message
    assert "dates de péremption" in message
    assert "dates de péremption" not in llm_brief.user_message(facts(), "en")


def test_request_uses_the_configured_model_and_temperature_zero():
    client = FakeClient(GOOD)
    text, model = llm_brief.call_model(client, "some-model", CFG, "hello")
    request = client.requests[0]
    assert (text, model) == (GOOD, "fake-model-001")
    assert request["model"] == "some-model"
    assert request["contents"] == "hello"
    assert request["config"].temperature == 0
    assert request["config"].seed == 42
    assert request["config"].system_instruction == llm_brief.SYSTEM_PROMPT


def test_blocked_reply_is_an_unusable_attempt():
    blocked = reply(None, finish=types.FinishReason.SAFETY)
    assert llm_brief.call_model(FakeClient(blocked), "m", CFG, "x")[0] is None


def test_prompt_blocked_with_no_candidates_is_an_unusable_attempt():
    empty = SimpleNamespace(candidates=None, text=None, model_version=None)
    assert llm_brief.call_model(FakeClient(empty), "m", CFG, "x") == (None, "m")


def test_truncated_reply_is_an_unusable_attempt():
    cut = reply("Stock is", finish=types.FinishReason.MAX_TOKENS)
    assert llm_brief.call_model(FakeClient(cut), "m", CFG, "x")[0] is None


def test_temporary_error_gets_one_retry_after_a_wait():
    client = FakeClient(api_error(429), GOOD)
    assert llm_brief.call_model(client, "m", CFG, "x")[0] == GOOD
    assert len(client.requests) == 2


@pytest.mark.parametrize("failure", [api_error(429), api_error(503), httpx2.ConnectError("down")])
def test_lasting_trouble_raises_instead_of_using_up_attempts(failure):
    client = FakeClient(failure)
    with pytest.raises(llm_brief.LLMUnavailable):
        llm_brief.call_model(client, "m", CFG, "x")
    assert len(client.requests) == 2


def test_a_bad_request_is_not_retried():
    client = FakeClient(api_error(400))
    with pytest.raises(errors.ClientError):
        llm_brief.call_model(client, "m", CFG, "x")
    assert len(client.requests) == 1


def test_answering_model_is_recorded():
    brief = llm_brief.write_brief(facts(), "en", FakeClient(reply(model="other-model")), "m", CFG)
    assert brief["model_id"] == "other-model"


def test_template_fallback_records_no_model():
    brief = llm_brief.write_brief(facts(), "en", FakeClient("Stock is 9999."), "m", CFG)
    assert brief["guardrail_status"] == "template_fallback"
    assert brief["model_id"] is None


# Cache and precompute

def test_cache_key_changes_with_each_input(monkeypatch):
    base = llm_brief.cache_key(facts(), "en", "m")
    assert base == llm_brief.cache_key(facts(), "en", "m")
    assert base != llm_brief.cache_key(facts(site_code="C2"), "en", "m")
    assert base != llm_brief.cache_key(facts(), "fr", "m")
    assert base != llm_brief.cache_key(facts(), "en", "other")
    monkeypatch.setattr(llm_brief, "PROMPT_VERSION", "another-version")
    assert base != llm_brief.cache_key(facts(), "en", "m")


def test_every_site_gets_both_languages_with_required_fields():
    briefs = llm_brief.generate_briefs({"C1399": facts()}, CFG, FakeClient(GOOD), ["m"], {})
    assert [b["lang"] for b in briefs] == ["en", "fr"]
    assert set(briefs[0]) == {
        "site_code", "lang", "text", "guardrail_status", "attempts", "model_id", "cache_key",
    }


def test_cached_briefs_make_no_calls():
    all_facts = {"C1399": facts()}
    first = llm_brief.generate_briefs(all_facts, CFG, FakeClient(GOOD), ["m"], {})
    previous = {b["cache_key"]: b for b in first}
    client = FakeClient(GOOD)
    assert llm_brief.generate_briefs(all_facts, CFG, client, ["m"], previous) == first
    assert client.requests == []


def test_a_cached_brief_that_fails_a_tighter_guardrail_is_regenerated():
    first = llm_brief.generate_briefs({"C1399": facts()}, CFG, FakeClient(GOOD), ["m"], {})
    previous = {b["cache_key"]: b | {"text": "Too short."} for b in first}
    client = FakeClient(GOOD_2)
    briefs = llm_brief.generate_briefs({"C1399": facts()}, CFG, client, ["m"], previous)
    assert len(client.requests) == 2
    assert {b["text"] for b in briefs} == {GOOD_2}


def test_changed_facts_are_regenerated():
    first = llm_brief.generate_briefs({"C1399": facts()}, CFG, FakeClient(GOOD), ["m"], {})
    previous = {b["cache_key"]: b for b in first}
    client = FakeClient(GOOD_2)
    changed = {"C1399": facts(district="OTHER")}
    llm_brief.generate_briefs(changed, CFG, client, ["m"], previous)
    assert len(client.requests) == 2


def test_no_client_uses_the_template_and_marks_it_unsent():
    briefs = llm_brief.generate_briefs({"C1399": facts()}, CFG, None, ["m"], {})
    assert {b["guardrail_status"] for b in briefs} == {"template_fallback"}
    assert {b["attempts"] for b in briefs} == {0}
    assert {b["model_id"] for b in briefs} == {None}


def test_offline_templates_are_replaced_once_a_client_exists():
    offline = llm_brief.generate_briefs({"C1399": facts()}, CFG, None, ["m"], {})
    previous = {b["cache_key"]: b for b in offline}
    briefs = llm_brief.generate_briefs({"C1399": facts()}, CFG, FakeClient(GOOD), ["m"], previous)
    assert {b["guardrail_status"] for b in briefs} == {"passed"}


def test_a_rejected_key_falls_back_to_templates_instead_of_crashing():
    briefs = llm_brief.generate_briefs({"C1399": facts()}, CFG, FakeClient(api_error(403)), ["m"], {})
    assert len(briefs) == 2
    assert {b["attempts"] for b in briefs} == {0}


def two_sites():
    return {"C1399": facts(), "C2": facts(site_code="C2")}


def test_quota_stop_keeps_finished_briefs_and_leaves_the_rest_unsent():
    client = FakeClient(GOOD, api_error(429), api_error(429))
    briefs = llm_brief.generate_briefs(two_sites(), CFG, client, ["m"], {})
    assert len(briefs) == 4
    assert sorted(b["attempts"] for b in briefs) == [0, 0, 0, 1]
    assert len(client.requests) == 3


def test_quota_stop_never_caches_a_template_as_final():
    client = FakeClient(api_error(429))
    briefs = llm_brief.generate_briefs(two_sites(), CFG, client, ["m"], {})
    assert {b["attempts"] for b in briefs} == {0}


def test_a_later_run_resumes_where_the_quota_stopped():
    first = llm_brief.generate_briefs(
        two_sites(), CFG, FakeClient(GOOD, api_error(429), api_error(429)), ["m"], {},
    )
    previous = {b["cache_key"]: b for b in first}
    client = FakeClient(GOOD_2)
    second = llm_brief.generate_briefs(two_sites(), CFG, client, ["m"], previous)
    assert len(client.requests) == 3
    assert {b["guardrail_status"] for b in second} == {"passed"}


def test_each_finished_brief_is_saved_immediately():
    saves = []
    client = FakeClient(GOOD, api_error(429), api_error(429))
    llm_brief.generate_briefs(two_sites(), CFG, client, ["m"], {}, save=saves.append)
    assert [len(s) for s in saves] == [1]


def test_offline_templates_are_rebuilt_when_the_template_changes():
    offline = llm_brief.generate_briefs({"C1399": facts()}, CFG, None, ["m"], {})
    stale = {b["cache_key"]: b | {"text": "old template"} for b in offline}
    briefs = llm_brief.generate_briefs({"C1399": facts()}, CFG, None, ["m"], stale)
    assert "old template" not in {b["text"] for b in briefs}


def test_model_briefs_survive_a_run_without_a_key():
    first = llm_brief.generate_briefs({"C1399": facts()}, CFG, FakeClient(GOOD), ["m"], {})
    previous = {b["cache_key"]: b for b in first}
    assert llm_brief.generate_briefs({"C1399": facts()}, CFG, None, ["m"], previous) == first


class QuotaByModel(FakeClient):
    """SYNTHETIC client whose first model is out of quota."""

    def generate_content(self, **request):
        self.requests.append(request)
        if request["model"] == "spent":
            raise api_error(429)
        return reply(GOOD, model=request["model"])


def test_a_spent_model_hands_over_to_the_backup():
    client = QuotaByModel()
    briefs = llm_brief.generate_briefs(two_sites(), CFG, client, ["spent", "backup"], {})
    assert {b["model_id"] for b in briefs} == {"backup"}
    assert {b["attempts"] for b in briefs} == {1}


def test_briefs_cached_from_either_model_are_reused():
    client = QuotaByModel()
    first = llm_brief.generate_briefs(two_sites(), CFG, client, ["spent", "backup"], {})
    previous = {b["cache_key"]: b for b in first}
    again = QuotaByModel()
    assert llm_brief.generate_briefs(two_sites(), CFG, again, ["spent", "backup"], previous) == first
    assert again.requests == []


def test_a_retired_model_also_hands_over():
    class Retired(FakeClient):
        def generate_content(self, **request):
            self.requests.append(request)
            if request["model"] == "gone":
                raise api_error(404)
            return reply(GOOD, model=request["model"])

    briefs = llm_brief.generate_briefs(two_sites(), CFG, Retired(), ["gone", "backup"], {})
    assert {b["model_id"] for b in briefs} == {"backup"}


def test_all_models_spent_leaves_offline_templates():
    client = FakeClient(api_error(429))
    briefs = llm_brief.generate_briefs(two_sites(), CFG, client, ["a", "b"], {})
    assert {b["attempts"] for b in briefs} == {0}
    assert len(briefs) == 4


def test_no_key_means_no_client(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert llm_brief.make_client() is None
