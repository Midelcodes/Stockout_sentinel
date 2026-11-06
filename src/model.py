"""LightGBM point and quantile models, production forecast."""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import baselines
from src.backtest import metrics_table, origin_indexes, print_overall, stocking_decision, wape
from src.features import FEATURES, origin_features, prediction_rows, training_table
from src.io import SERIES_KEYS
from src.panel import month_start

MODELS = baselines.BASELINES + ["lightgbm", "lightgbm_p50"]


def _fit(rows, cfg, **objective):
    model = lgb.LGBMRegressor(
        **cfg["lightgbm"], **objective, random_state=cfg["seed"], verbosity=-1
    )
    return model.fit(rows[FEATURES], rows["target"])


def fit_point(rows, cfg, power):
    return _fit(rows, cfg, objective="tweedie", tweedie_variance_power=power)


def fit_models(rows, cfg, power):
    low, high = cfg["quantiles"]
    return {
        "mean": fit_point(rows, cfg, power),
        "p50": _fit(rows, cfg, objective="quantile", alpha=low),
        "p90": _fit(rows, cfg, objective="quantile", alpha=high),
    }


def post_process(forecasts, columns=("mean", "p50", "p90")):
    """Clip every prediction at 0 and enforce p90 >= p50."""
    out = forecasts.copy()
    for column in columns:
        out[column] = out[column].clip(lower=0)
    if "p90" in columns:
        out["p90"] = out[["p50", "p90"]].max(axis=1)
    return out


def predict(models, rows, columns=("mean", "p50", "p90")):
    out = rows[SERIES_KEYS + ["month_index", "horizon", "target_index"]].copy()
    out["site_code"] = out["site_code"].astype(str)
    out["product_code"] = out["product_code"].astype(str)
    for name, model in models.items():
        out[name] = model.predict(rows[FEATURES])
    return post_process(out, columns)


def choose_power(table, features, predictions, cfg, origin):
    """Pick the Tweedie power with the lowest WAPE on the earliest fold only."""
    train = table[table["target_index"] <= origin]
    rows = prediction_rows(features, origin, cfg)
    actuals = predictions.loc[predictions["origin"] == origin, SERIES_KEYS + ["horizon", "y"]]
    scores = {}
    for power in cfg["tweedie_variance_powers"]:
        forecast = predict({"mean": fit_point(train, cfg, power)}, rows, ["mean"])
        scored = forecast.merge(actuals, on=SERIES_KEYS + ["horizon"])
        scores[power] = wape(scored["y"], scored["mean"])
    return min(scores, key=scores.get), scores


def calibration_factor(rows, coverage):
    """How far p90 must widen to cover the target share of these rows.

    Errors are scaled by p90 + 1, so one factor suits small and large sites alike.
    """
    if rows.empty:
        return 0.0
    scores = ((rows["y"] - rows["p90"]) / (rows["p90"] + 1)).to_numpy()
    level = min(1.0, np.ceil((len(scores) + 1) * coverage) / len(scores))
    return max(0.0, float(np.quantile(scores, level, method="higher")))


def widen_p90(p90, factor):
    return p90 + factor * (p90 + 1)


def fold_factors(folds, origins, coverage):
    """Each origin's factor, from earlier folds whose targets were known by then."""
    factors = {}
    for origin in origins:
        earlier = [rows[rows["target_index"] <= origin] for fold, rows in folds.items() if fold < origin]
        rows = pd.concat(earlier) if earlier else pd.DataFrame(columns=["y", "p90"])
        factors[origin] = calibration_factor(rows, coverage)
    return factors


def fold_forecasts(table, features, cfg, origins, power):
    folds = {}
    for origin in origins:
        models = fit_models(table[table["target_index"] <= origin], cfg, power)
        folds[origin] = predict(models, prediction_rows(features, origin, cfg))
    return folds


def with_actuals(forecasts, panel):
    actuals = panel.loc[panel["y"].notna(), SERIES_KEYS + ["month_index", "y"]]
    actuals = actuals.rename(columns={"month_index": "target_index"})
    return forecasts.merge(actuals, on=SERIES_KEYS + ["target_index"])


def backtest_forecasts(folds, origins, factors):
    frames = [
        folds[origin].assign(p90_raw=folds[origin]["p90"], p90=widen_p90(folds[origin]["p90"], factors[origin]))
        for origin in origins
    ]
    forecasts = pd.concat(frames, ignore_index=True).rename(columns={
        "month_index": "origin", "mean": "lightgbm", "p50": "lightgbm_p50",
        "p90": "lightgbm_p90", "p90_raw": "lightgbm_p90_raw",
    })
    return forecasts.drop(columns="target_index")


def production_forecast(table, features, cfg, last, power, factor=0.0):
    """Retrain on everything and forecast each horizon from the last month."""
    forecasts = predict(fit_models(table, cfg, power), prediction_rows(features, last, cfg))
    forecasts["p90"] = widen_p90(forecasts["p90"], factor)
    forecasts["origin"] = month_start(forecasts["month_index"])
    forecasts["target_month"] = month_start(forecasts["target_index"])
    columns = SERIES_KEYS + ["origin", "horizon", "target_month", "mean", "p50", "p90"]
    return forecasts[columns].sort_values(SERIES_KEYS + ["horizon"]).reset_index(drop=True)


def print_headline(metrics):
    overall = metrics[(metrics["scope"] == "common") & (metrics["slice"] == "overall")]
    overall = overall.set_index("model")
    model = overall.loc["lightgbm"]
    for baseline in ["amc_3", "elmis_amc"]:
        change = model["wape"] / overall.loc[baseline, "wape"] - 1
        print(f"  WAPE vs {baseline}: {overall.loc[baseline, 'wape']:.3f} -> {model['wape']:.3f} ({change:+.1%})")
    print(f"  p90 coverage after widening: {model['p90_coverage']:.1%}")


def run(cfg):
    outputs = cfg["paths"]["outputs"]
    panel = pd.read_parquet(outputs / "panel_clean.parquet")
    predictions = pd.read_parquet(outputs / "backtest_predictions.parquet")
    predictions = predictions.drop(
        columns=["lightgbm", "lightgbm_p50", "lightgbm_p90", "lightgbm_p90_raw"], errors="ignore"
    )

    features = origin_features(panel)
    table = training_table(panel, features, cfg)
    origins = origin_indexes(panel, cfg)
    power, power_scores = choose_power(table, features, predictions, cfg, origins[0])

    last = panel["month_index"].max()
    calibration_origins = [last - back for back in cfg["calibration_origins_back"]]
    folds = fold_forecasts(table, features, cfg, calibration_origins + origins, power)
    scored = {origin: with_actuals(rows, panel) for origin, rows in folds.items()}
    coverage = cfg["quantiles"][1]
    factors = fold_factors(scored, origins + [last], coverage)
    with open(outputs / "model_selection.json", "w", encoding="utf-8") as f:
        json.dump({
            "tweedie_variance_power": power, "earliest_fold_wape": power_scores,
            "p90_widening_by_origin": {str(k): v for k, v in factors.items()},
        }, f, indent=2)

    forecasts = backtest_forecasts(folds, origins, factors)
    predictions = predictions.merge(forecasts, how="left", on=SERIES_KEYS + ["origin", "horizon"])
    predictions.to_parquet(outputs / "backtest_predictions.parquet", index=False)
    metrics = metrics_table(predictions, MODELS, upper={"lightgbm": "lightgbm_p90"})
    metrics.to_parquet(outputs / "backtest_metrics.parquet", index=False)
    common = predictions[predictions[MODELS].notna().all(axis=1)]
    stocking = stocking_decision(common, "lightgbm_p90")
    stocking.to_parquet(outputs / "stocking_decision.parquet", index=False)

    production_forecast(table, features, cfg, last, power, factors[last]).to_parquet(
        outputs / "forecasts.parquet", index=False
    )
    print(f"  training rows: {len(table):,}; Tweedie power {power} {power_scores}")
    print_overall(metrics)
    print_headline(metrics)
    print(f"  p90 coverage before widening: {(common['y'] <= common['lightgbm_p90_raw']).mean():.1%}")
    print(stocking.round(3).to_string(index=False))
    return "done"
