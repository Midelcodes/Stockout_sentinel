"""Data profile report, the app's slim data files and the auto-filled README."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.io import (
    EXPECTED_COLUMNS,
    OPTIONAL_FILES,
    SERIES_KEYS,
    SchemaMismatchError,
    compare_schema,
    load_raw,
)
from src.model import MODELS
from src.panel import STOCK_FIELDS

MODEL_NAMES = {
    "lightgbm": "**LightGBM**", "amc_3": "AMC-3", "elmis_amc": "eLMIS AMC",
    "naive": "Naive", "seasonal_naive": "Seasonal naive",
}
TEMPLATE = Path(__file__).with_name("readme_template.md")
BOOTSTRAP_DRAWS = 2000


def _numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _month_index(df):
    """Months since year 0, NaN where year or month cannot be read."""
    year, month = _numeric(df["year"]), _numeric(df["month"])
    return (year * 12 + month - 1).where(month.between(1, 12))


def _month_label(index):
    if pd.isna(index):
        return "n/a"
    return f"{int(index) // 12}-{int(index) % 12 + 1:02d}"


def _share(part, whole):
    return part / whole if whole else float("nan")


def _balance_stats(df, tolerance):
    fields = df[STOCK_FIELDS].apply(_numeric)
    usable = fields.notna().all(axis=1)
    f = fields[usable]
    before_adjustment = f["stock_initial"] + f["stock_received"] - f["stock_distributed"]
    residual = f["stock_end"] - (before_adjustment + f["stock_adjustment"])
    residual_flipped = f["stock_end"] - (before_adjustment - f["stock_adjustment"])
    return {
        "rows_checked": int(usable.sum()),
        "rows_not_checked": int((~usable).sum()),
        "breaks": int((residual.abs() > tolerance).sum()),
        "breaks_exact": int((residual != 0).sum()),
        "breaks_if_adjustment_subtracted": int((residual_flipped.abs() > tolerance).sum()),
    }


def _amc_match(df, index):
    """How often the recorded AMC is the plain mean of the current and prior 2 months."""
    d = df.assign(month_index=index).sort_values(SERIES_KEYS + ["month_index"])
    groups = d.groupby(SERIES_KEYS)
    three_in_a_row = groups["month_index"].shift(2) == d["month_index"] - 2
    mean_3 = groups["stock_distributed"].transform(lambda s: s.rolling(3).mean())
    close = (_numeric(d["average_monthly_consumption"]) - mean_3).abs() <= 1
    return int(close[three_in_a_row].sum()), int(three_in_a_row.sum())


def profile_stats(logistics, cfg):
    """Everything Phase 1 reports about the logistics table, as plain numbers."""
    df = logistics
    index = _month_index(df)
    last_month = index.max()

    keyed = df[SERIES_KEYS].assign(month_index=index).dropna(subset=["month_index"])
    duplicates = int(keyed.duplicated().sum())
    series = keyed.groupby(SERIES_KEYS)["month_index"].agg(["min", "max", "nunique"])
    internal_gaps = series["max"] - series["min"] + 1 - series["nunique"]
    trailing_gaps = last_month - series["max"]

    distributed = _numeric(df["stock_distributed"])
    zeros = distributed == 0
    stockout_days = _numeric(df["stock_stockout_days"])
    all_zero = (df[STOCK_FIELDS].apply(_numeric) == 0).all(axis=1)
    sites_per_district = df.groupby("district")["site_code"].nunique()
    amc_close, amc_checked = _amc_match(df, index)

    return {
        "rows": len(df),
        "sites": df["site_code"].nunique(),
        "products": df["product_code"].nunique(),
        "regions": df["region"].nunique(),
        "districts": df["district"].nunique(),
        "first_month": _month_label(index.min()),
        "last_month": _month_label(last_month),
        "unreadable_month_rows": int(index.isna().sum()),
        "duplicate_rows": duplicates,
        "sites_in_multiple_districts": int(
            (df.groupby("site_code")["district"].nunique() > 1).sum()
        ),
        "single_site_districts": int((sites_per_district == 1).sum()),
        "all_zero_rows": int(all_zero.sum()),
        "series_never_active": int(all_zero.groupby([df[k] for k in SERIES_KEYS]).all().sum()),
        "amc_close": amc_close,
        "amc_checked": amc_checked,
        "series": len(series),
        "series_length": series["nunique"].describe(percentiles=[0.25, 0.5, 0.75]).to_dict(),
        "series_with_internal_gaps": int((internal_gaps > 0).sum()),
        "internal_missing_months": int(internal_gaps.sum()),
        "internal_gaps_per_series": internal_gaps.describe(percentiles=[0.5, 0.9]).to_dict(),
        "series_ending_early": int((trailing_gaps > 0).sum()),
        "trailing_missing_months": int(trailing_gaps.sum()),
        "distributed_non_null": int(distributed.notna().sum()),
        "distributed_zeros": int(zeros.sum()),
        "distributed_zeros_with_stock": int(
            (zeros & (_numeric(df["stock_initial"]) > 0)).sum()
        ),
        "stockout_rows": int((stockout_days > 0).sum()),
        "stockout_days_null": int(stockout_days.isna().sum()),
        "balance": _balance_stats(df, cfg["balance_tolerance"]),
    }


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _pct(part, whole):
    return f"{part:,} of {whole:,} ({_share(part, whole):.1%})"


def _file_section(name, df):
    rows = []
    for column in df.columns:
        values = df[column]
        rows.append((column, values.dtype, int(values.isna().sum()), values.nunique()))
    return "\n\n".join([
        f"### {df.attrs['filename']}",
        f"{len(df):,} rows, encoding `{df.attrs.get('encoding', 'unknown')}`",
        _table(["column", "dtype", "nulls", "distinct"], rows),
    ])


def _undocumented_columns(tables):
    """Columns that appear in no cell of data_dictionary.csv."""
    dictionary = tables["dictionary"]
    documented = {str(v).strip().rstrip("*") for v in dictionary.to_numpy().ravel()}
    return [
        f"{df.attrs['filename']}: `{column}`"
        for name, df in tables.items() if name != "dictionary"
        for column in df.columns if column not in documented
    ]


def _summary_section(stats, cfg):
    expected = cfg["expected"]
    rows = [
        ("rows", f"{stats['rows']:,}", ""),
        ("sites", stats["sites"], expected["sites"]),
        ("products", stats["products"], expected["products"]),
        ("regions", stats["regions"], ""),
        ("districts", stats["districts"], ""),
        ("first month", stats["first_month"], expected["first_month"]),
        ("last month", stats["last_month"], expected["last_month"]),
        ("rows with unreadable year or month", stats["unreadable_month_rows"], ""),
        ("duplicate site x product x month rows", stats["duplicate_rows"], ""),
        ("sites listed under more than one district", stats["sites_in_multiple_districts"], ""),
        ("districts with a single site", stats["single_site_districts"], ""),
    ]
    return "## Summary\n\n" + _table(["metric", "observed", "spec"], rows)


def _series_section(stats):
    length = stats["series_length"]
    gaps = stats["internal_gaps_per_series"]
    return "\n\n".join([
        "## Series (site x product)",
        f"Series count: {stats['series']:,}",
        _table(
            ["distribution", "min", "p25", "median", "p75", "max", "mean"],
            [("reported months per series", length.get("min"), length.get("25%"),
              length.get("50%"), length.get("75%"), length.get("max"),
              f"{length.get('mean', float('nan')):.1f}")],
        ),
        "Missing months are counted apart: internal gaps fall between a series' "
        "first and last report; trailing months fall after its last report, up to "
        "the dataset's last month.",
        _table(
            ["metric", "value"],
            [
                ("series with internal gaps", _pct(stats["series_with_internal_gaps"], stats["series"])),
                ("internal missing months, total", f"{stats['internal_missing_months']:,}"),
                ("internal missing months per series, median", gaps.get("50%")),
                ("internal missing months per series, p90", gaps.get("90%")),
                ("internal missing months per series, max", gaps.get("max")),
                ("series ending before the last month", _pct(stats["series_ending_early"], stats["series"])),
                ("trailing missing months, total", f"{stats['trailing_missing_months']:,}"),
            ],
        ),
    ])


def _quality_section(stats, cfg):
    balance = stats["balance"]
    tolerance = cfg["balance_tolerance"]
    rows = [
        ("`stock_distributed` = 0", _pct(stats["distributed_zeros"], stats["distributed_non_null"])),
        ("of those, `stock_initial` > 0",
         _pct(stats["distributed_zeros_with_stock"], stats["distributed_zeros"])),
        ("`stock_stockout_days` > 0", _pct(stats["stockout_rows"], stats["rows"])),
        ("`stock_stockout_days` null", _pct(stats["stockout_days_null"], stats["rows"])),
        (f"stock balance off by more than {tolerance}",
         _pct(balance["breaks"], balance["rows_checked"])),
        ("stock balance off by any amount", _pct(balance["breaks_exact"], balance["rows_checked"])),
        (f"off by more than {tolerance} if adjustment were subtracted instead",
         _pct(balance["breaks_if_adjustment_subtracted"], balance["rows_checked"])),
        ("rows skipped by the balance check (a stock field is null)",
         f"{balance['rows_not_checked']:,}"),
        ("rows with every stock field 0", _pct(stats["all_zero_rows"], stats["rows"])),
        ("series with no stock activity in any month", _pct(stats["series_never_active"], stats["series"])),
        ("recorded AMC within 1 unit of the plain mean of months t-2 to t",
         _pct(stats["amc_close"], stats["amc_checked"])),
    ]
    return "## Data quality\n\n" + _table(["check", "value"], rows)


def _join_section(tables):
    logistics = tables["logistics"]
    lines = ["## Joins"]
    if "product_code" in tables["products"]:
        unmatched = set(logistics["product_code"]) - set(tables["products"]["product_code"])
        lines.append(f"Product codes with no row in {tables['products'].attrs['filename']}: {len(unmatched)}")
    if "sites" in tables and "site_code" in tables["sites"]:
        filename = tables["sites"].attrs["filename"]
        unmatched = set(logistics["site_code"]) - set(tables["sites"]["site_code"])
        unused = sorted(set(tables["sites"]["site_code"]) - set(logistics["site_code"]))
        lines.append(f"Site codes with no row in {filename}: {len(unmatched)}")
        lines.append(
            f"Sites in {filename} with no logistics rows: {len(unused)}"
            + (f" ({', '.join(unused)})" if unused else "")
        )
    else:
        lines.append("The site file is absent, so there are no site types or coordinates.")
    return "\n\n".join(lines)


def build_profile(tables, cfg):
    mismatches = compare_schema(tables)
    sections = ["# Data profile"]

    schema = "## Schema check\n\n"
    schema += "\n".join(f"- {m}" for m in mismatches) if mismatches else "All headers match the spec."
    absent = [cfg["raw_files"][n] for n in OPTIONAL_FILES if n not in tables]
    if absent:
        schema += f"\n\nOptional files absent: {', '.join(absent)}"
    sections.append(schema)

    sections.append("## Files")
    sections += [_file_section(name, df) for name, df in tables.items()]

    dictionary = tables["dictionary"]
    undocumented = _undocumented_columns(tables)
    sections.append(
        "## Data dictionary\n\n"
        + _table(list(dictionary.columns), dictionary.fillna("").itertuples(index=False))
        + "\n\nColumns not named anywhere in the dictionary: "
        + (", ".join(undocumented) if undocumented else "none")
    )

    missing_needed = set(EXPECTED_COLUMNS["logistics"]) - set(tables["logistics"].columns)
    if missing_needed:
        sections.append("## Summary\n\nSkipped: the logistics file lacks expected columns.")
    else:
        stats = profile_stats(tables["logistics"], cfg)
        sections += [
            _summary_section(stats, cfg),
            _series_section(stats),
            _quality_section(stats, cfg),
            _join_section(tables),
        ]
    return "\n\n".join(sections) + "\n", mismatches


def run_profile(cfg):
    tables = load_raw(cfg)
    text, mismatches = build_profile(tables, cfg)
    path = cfg["paths"]["outputs"] / "profile.md"
    path.write_text(text, encoding="utf-8")
    print(text)
    if mismatches:
        raise SchemaMismatchError(
            f"{len(mismatches)} schema mismatch(es), listed in {path}. "
            "Approve how to handle them before continuing."
        )
    return "done"


def overall_metrics(metrics):
    rows = metrics[(metrics["scope"] == "common") & (metrics["slice"] == "overall")]
    return rows.set_index("model")


def wape_change_interval(predictions, baseline, seed):
    """95% interval for the model's WAPE change against a baseline, resampling series."""
    rows = predictions[predictions[MODELS].notna().all(axis=1)]
    codes, _ = pd.factorize(rows["site_code"] + "|" + rows["product_code"])
    n = codes.max() + 1
    model_error = np.bincount(codes, (rows["y"] - rows["lightgbm"]).abs(), n)
    baseline_error = np.bincount(codes, (rows["y"] - rows[baseline]).abs(), n)
    draws = np.random.default_rng(seed).integers(0, n, size=(BOOTSTRAP_DRAWS, n))
    counts = np.stack([np.bincount(draw, minlength=n) for draw in draws])
    with np.errstate(divide="ignore", invalid="ignore"):
        change = (counts @ model_error) / (counts @ baseline_error) - 1
    return [float(q) for q in np.nanquantile(change, [0.025, 0.975])]


def equal_stock_interval(predictions, seed):
    """95% interval for p90's shortfall minus AMC-3's at the same total stock, in points."""
    rows = predictions[predictions[MODELS].notna().all(axis=1)]
    codes, _ = pd.factorize(rows["site_code"] + "|" + rows["product_code"])
    y, p90, amc = (rows[c].to_numpy() for c in ("y", "lightgbm_p90", "amc_3"))
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(BOOTSTRAP_DRAWS):
        weights = np.bincount(rng.integers(0, codes.max() + 1, codes.max() + 1), minlength=codes.max() + 1)[codes]
        demand = (weights * y).sum()
        scale = (weights * p90).sum() / (weights * amc).sum()
        model = (weights * np.clip(y - p90, 0, None)).sum() / demand
        baseline = (weights * np.clip(y - amc * scale, 0, None)).sum() / demand
        gaps.append(model - baseline)
    return [float(q) for q in np.nanquantile(gaps, [0.025, 0.975])]


def headline_numbers(metrics, predictions, stocking, cfg):
    overall = overall_metrics(metrics)
    model = overall.loc["lightgbm"]
    result = {
        "rows_scored": int(model["rows_scored"]),
        "p90_coverage": float(model["p90_coverage"]),
        "model_wape": float(model["wape"]),
        "model_shortfall": float(model["shortfall_share"]),
        "p90_coverage_raw": float(
            (predictions["y"] <= predictions["lightgbm_p90_raw"])[predictions[MODELS].notna().all(axis=1)].mean()
        ),
        "stocking": stocking.set_index("policy").to_dict("index"),
        "equal_stock_interval": equal_stock_interval(predictions, cfg["seed"]),
    }
    for baseline in ["amc_3", "elmis_amc"]:
        row = overall.loc[baseline]
        result[baseline] = {
            "wape": float(row["wape"]),
            "shortfall": float(row["shortfall_share"]),
            "wape_change": float(model["wape"] / row["wape"] - 1),
            "interval": wape_change_interval(predictions, baseline, cfg["seed"]),
        }
    return result


def headline_sentence(h):
    amc = h["amc_3"]
    verdict = "beats" if amc["wape_change"] < 0 else "does not beat"
    shortfall = "lower" if h["model_shortfall"] < amc["shortfall"] else "higher"
    low, high = amc["interval"]
    stocking = h["stocking"]
    p90 = stocking["Stock to calibrated p90"]
    plain = stocking["Stock to AMC-3"]
    matched = stocking["Stock to AMC-3, same total stock as p90"]
    fair = "lower" if p90["shortfall_share"] < matched["shortfall_share"] else "higher"
    gap_low, gap_high = (100 * value for value in h["equal_stock_interval"])
    return (
        f"LightGBM {verdict} AMC-3 on forecast error: WAPE {amc['wape']:.3f} to {h['model_wape']:.3f} "
        f"({amc['wape_change']:+.1%}, 95% interval {low:+.1%} to {high:+.1%}). As a point forecast its shortfall "
        f"is {shortfall} ({h['model_shortfall']:.1%} against {amc['shortfall']:.1%}). Stocking to the calibrated "
        f"p90 ({h['p90_coverage']:.1%} coverage) leaves {p90['shortfall_share']:.1%} short at "
        f"{p90['stock_vs_demand']:.2f}x demand, against {plain['shortfall_share']:.1%} for AMC-3 at "
        f"{plain['stock_vs_demand']:.2f}x. Given the same total stock, AMC-3 leaves "
        f"{matched['shortfall_share']:.1%}, so the model's shortfall is {fair} at equal stock "
        f"(95% interval {gap_low:+.1f} to {gap_high:+.1f} points)."
    )


def results_table(metrics):
    overall = overall_metrics(metrics)
    rows = []
    for model, name in MODEL_NAMES.items():
        r = overall.loc[model]
        coverage = f"{r['p90_coverage']:.1%}" if pd.notna(r["p90_coverage"]) else "n/a"
        rows.append((
            name, f"{r['wape']:.3f}", f"{r['mase']:.3f}", f"{r['under_forecast_rate']:.1%}",
            f"{r['shortfall_share']:.1%}", coverage, f"{int(r['rows_scored']):,}",
        ))
    return _table(
        ["Model", "WAPE", "MASE", "Under-forecast", "Shortfall", "p90 coverage", "Rows scored"], rows
    )


def bias_text(metrics, cfg):
    """Model WAPE by volume tier, reporting-quality tier and region, as one sentence."""
    rows = metrics[(metrics["scope"] == "common") & (metrics["model"] == "lightgbm")]

    def tiers(column):
        values = rows[rows["slice"] == column].sort_values("slice_value")["wape"]
        return ", ".join(f"{value:.3f}" for value in values)

    regions = rows[(rows["slice"] == "region") & (rows["rows_scored"] >= cfg["region_min_rows"])]["wape"]
    region_part = f"{regions.min():.3f} to {regions.max():.3f}" if len(regions) else "n/a"
    return (
        f"LightGBM WAPE by site volume, lowest to highest: {tiers('volume_tier')}. "
        f"By reporting quality, best to worst: {tiers('quality_tier')}. "
        f"By region ({cfg['region_min_rows']}+ scored rows): {region_part}."
    )


def brief_status_text(briefs):
    statuses = pd.Series([b["guardrail_status"] for b in briefs]).value_counts(normalize=True)
    return ", ".join(f"{share:.1%} {status}" for status, share in statuses.items())


def llm_models(briefs):
    models = sorted({b["model_id"] for b in briefs if b["model_id"]})
    return ", ".join(f"`{m}`" for m in models) or "a template (no LLM run yet)"


def write_app_data(cfg, panel, metrics, predictions, stocking, products):
    """The only files the app needs beyond the stage outputs. No raw columns."""
    outputs = cfg["paths"]["outputs"]
    actuals = panel.loc[panel["y"].notna(), SERIES_KEYS + ["month_start", "y"]]
    actuals.to_parquet(outputs / "actuals.parquet", index=False)
    last = panel.loc[panel["month_index"].idxmax(), "month_start"]
    meta = {
        "as_of_label": last.strftime("%B %Y"),
        "products": dict(zip(products["product_code"], products["product_name"].str.strip())),
        "headline": headline_numbers(metrics, predictions, stocking, cfg),
    }
    meta["headline_sentence"] = headline_sentence(meta["headline"])
    with open(outputs / "app_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return meta


def render_readme(cfg, meta, panel, metrics, status, transfers, briefs, counts):
    reported = panel[panel["reported"]]
    sites_per_district = status.groupby("district")["site_code"].nunique()
    recipients = int((status["mos"] < cfg["mos_min"]).sum())
    served = transfers.groupby(["to_site", "product_code"]).ngroups
    url = cfg.get("app_url")
    template = TEMPLATE.read_text(encoding="utf-8")
    return template.format(
        as_of=meta["as_of_label"],
        live_line=f"[Live demo]({url})." if url else "",
        headline=meta["headline_sentence"],
        products=reported["product_code"].nunique(),
        sites=reported["site_code"].nunique(),
        first_month=reported["month_start"].min().strftime("%B %Y"),
        last_month=reported["month_start"].max().strftime("%B %Y"),
        rows=f"{len(reported):,}",
        fake_zero=f"{counts['flag_fake_zero']:,}",
        all_zero=f"{counts['flag_all_zero']:,}",
        censored=f"{counts['flag_censored']:,}",
        balance=f"{counts['flag_balance']:,}",
        valid_y=f"{counts['valid_y']:,}",
        llm_model=llm_models(briefs),
        brief_status=brief_status_text(briefs),
        results_table=results_table(metrics),
        single_site=int((sites_per_district == 1).sum()),
        districts=len(sites_per_district),
        served=served,
        recipients=recipients,
        bias_text=bias_text(metrics, cfg),
    )


def run(cfg):
    outputs = cfg["paths"]["outputs"]
    panel = pd.read_parquet(outputs / "panel_clean.parquet")
    metrics = pd.read_parquet(outputs / "backtest_metrics.parquet")
    predictions = pd.read_parquet(outputs / "backtest_predictions.parquet")
    stocking = pd.read_parquet(outputs / "stocking_decision.parquet")
    status = pd.read_parquet(outputs / "stock_status.parquet")
    transfers = pd.read_parquet(outputs / "transfers.parquet")
    with open(outputs / "briefs.json", encoding="utf-8") as f:
        briefs = json.load(f)
    with open(outputs / "flag_counts.json", encoding="utf-8") as f:
        counts = json.load(f)

    meta = write_app_data(cfg, panel, metrics, predictions, stocking, load_raw(cfg)["products"])
    text = render_readme(cfg, meta, panel, metrics, status, transfers, briefs, counts)
    cfg["readme_path"].write_text(text, encoding="utf-8")
    return "done"
