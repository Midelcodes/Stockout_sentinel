"""Rolling-origin backtest and metrics."""
import numpy as np
import pandas as pd

from src import baselines
from src.io import SERIES_KEYS

SLICES = ["horizon", "origin_month", "product_code", "region", "volume_tier", "quality_tier"]


def wape(y, forecast):
    total = y.sum()
    return (y - forecast).abs().sum() / total if total else np.nan


def under_forecast_rate(y, forecast):
    return (forecast < y).mean() if len(y) else np.nan


def shortfall_share(y, forecast):
    total = y.sum()
    return (y - forecast).clip(lower=0).sum() / total if total else np.nan


def mase_scales(panel, origin, min_points):
    """Mean absolute one-step naive error per series, on valid y up to the origin.

    Steps run between consecutive valid values, so they can span a gap. Series
    with too few valid points get NaN.
    """
    history = baselines._valid_history(panel, origin)
    groups = history.groupby(SERIES_KEYS)["y"]
    scale = groups.diff().abs().groupby([history[k] for k in SERIES_KEYS]).mean()
    return scale.where(groups.count() >= min_points).rename("mase_scale")


def assign_tiers(values, n):
    """Equal-sized tiers 1..n from lowest to highest value, NaN stays NaN."""
    valid = values.dropna()
    tiers = np.ceil(valid.rank(method="first") * n / len(valid))
    return tiers.reindex(values.index)


def _site_tiers(panel, origin, n):
    """Site volume and reporting-quality tiers from data up to the origin."""
    history = panel[panel["month_index"] <= origin]
    volume = history.groupby("site_code")["y"].mean()
    at_origin = history[history["month_index"] == origin]
    quality = at_origin.groupby("site_code")["site_reporting_quality"].first()
    return pd.DataFrame({
        "volume_tier": assign_tiers(volume, n),
        "quality_tier": assign_tiers(quality, n),
    }).astype("Int64")


def origin_indexes(panel, cfg):
    last = panel["month_index"].max()
    return [last - back for back in cfg["backtest_origins_back"]]


def baseline_predictions(panel, cfg):
    """One row per scorable target (valid y), one column per baseline."""
    frames = []
    for origin in origin_indexes(panel, cfg):
        scales = mase_scales(panel, origin, cfg["mase_min_points"])
        tiers = _site_tiers(panel, origin, cfg["volume_tiers"])
        flat = {
            "naive": baselines.naive(panel, origin),
            "amc_3": baselines.amc_3(panel, origin),
            "elmis_amc": baselines.elmis_amc(panel, origin),
        }
        for horizon in cfg["horizons"]:
            target = origin + horizon
            rows = panel[(panel["month_index"] == target) & panel["y"].notna()]
            rows = rows[SERIES_KEYS + ["region", "district", "month_start", "y"]]
            rows = rows.rename(columns={"month_start": "target_month"})
            origin_month = f"{origin // 12}-{origin % 12 + 1:02d}"
            rows = rows.assign(origin=origin, origin_month=origin_month, horizon=horizon)
            rows = rows.set_index(SERIES_KEYS)
            forecasts = dict(flat, seasonal_naive=baselines.seasonal_naive(panel, origin, target))
            for name in baselines.BASELINES:
                rows[name] = forecasts[name].reindex(rows.index)
            rows["mase_scale"] = scales.reindex(rows.index)
            rows = rows.reset_index().merge(tiers, how="left", left_on="site_code", right_index=True)
            frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def score(rows, model, upper=None):
    """All metrics for one model on rows where it has a forecast.

    upper names the column holding the model's p90, if it has one.
    """
    scored = rows[rows[model].notna()]
    y, forecast = scored["y"], scored[model]
    usable = scored["mase_scale"] > 0
    scaled_error = (y - forecast).abs()[usable] / scored["mase_scale"][usable]
    series = SERIES_KEYS + ["origin"]
    return {
        "rows_scored": len(scored),
        "wape": wape(y, forecast),
        "mase": scaled_error.mean(),
        "mase_rows": int(usable.sum()),
        "mase_series_skipped": len(scored.loc[~usable, series].drop_duplicates()),
        "under_forecast_rate": under_forecast_rate(y, forecast),
        "shortfall_share": shortfall_share(y, forecast),
        "bias": forecast.sum() / y.sum() - 1 if y.sum() else np.nan,
        "p90_coverage": (y <= scored[upper]).mean() if upper else np.nan,
    }


def metrics_table(predictions, models, upper=None):
    """Every model on every slice, twice.

    scope=common scores only rows where all models have a forecast, so models
    are compared on identical rows. scope=available scores each model on all of
    its own rows. upper maps a model to its p90 column.
    """
    upper = upper or {}
    common = predictions[predictions[models].notna().all(axis=1)]
    records = []
    for scope, rows in [("common", common), ("available", predictions)]:
        groups = [("overall", "all", rows)]
        for column in SLICES:
            groups += [(column, str(value), part) for value, part in rows.groupby(column)]
        for slice_name, slice_value, part in groups:
            for model in models:
                records.append({
                    "scope": scope, "slice": slice_name, "slice_value": slice_value,
                    "model": model, **score(part, model, upper.get(model)),
                })
    return pd.DataFrame(records)


def print_overall(metrics):
    columns = [
        "model", "rows_scored", "wape", "mase", "mase_rows", "mase_series_skipped",
        "under_forecast_rate", "shortfall_share", "bias", "p90_coverage",
    ]
    for scope in ["common", "available"]:
        table = metrics[(metrics["scope"] == scope) & (metrics["slice"] == "overall")]
        print(f"\n  scope={scope}")
        print(table[columns].round(3).to_string(index=False))


def run(cfg):
    outputs = cfg["paths"]["outputs"]
    panel = pd.read_parquet(outputs / "panel_clean.parquet")
    predictions = baseline_predictions(panel, cfg)
    predictions.to_parquet(outputs / "backtest_predictions.parquet", index=False)
    metrics = metrics_table(predictions, baselines.BASELINES)
    metrics.to_parquet(outputs / "backtest_metrics.parquet", index=False)
    print_overall(metrics)
    return "done"
