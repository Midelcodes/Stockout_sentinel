"""Leakage-safe features for the direct multi-horizon training table.

A row's features describe origin month t and use only months up to t. Lags count
back from the first forecast month, so lag 1 is the value at t itself.
"""
import pandas as pd

from src.io import SERIES_KEYS

Y_LAGS = [1, 2, 3, 6, 12]
RECEIVED_LAGS = [1, 2]
ROLLING_WINDOWS = [3, 6]
CATEGORICAL = ["site_code", "product_code", "district", "region"]
FEATURES = (
    [f"y_lag_{k}" for k in Y_LAGS]
    + [f"y_{stat}_{w}" for w in ROLLING_WINDOWS for stat in ("mean", "std")]
    + [f"received_lag_{k}" for k in RECEIVED_LAGS]
    + ["censored_at_t", "stock_end_at_t", "distributed_raw_at_t", "site_reporting_quality"]
    + ["target_month_of_year", "horizon"]
    + CATEGORICAL
)


def origin_features(panel):
    """One row per series and month. The panel has no calendar gaps, so shift(k) is k months."""
    p = panel.sort_values(SERIES_KEYS + ["month_index"])
    out = p[SERIES_KEYS + ["month_index", "district", "region", "site_reporting_quality"]].copy()
    y = p.groupby(SERIES_KEYS)["y"]
    received = p.groupby(SERIES_KEYS)["stock_received"]

    for k in Y_LAGS:
        out[f"y_lag_{k}"] = y.shift(k - 1)
    for w in ROLLING_WINDOWS:
        out[f"y_mean_{w}"] = y.transform(lambda s: s.rolling(w, min_periods=1).mean())
        out[f"y_std_{w}"] = y.transform(lambda s: s.rolling(w, min_periods=2).std())
    for k in RECEIVED_LAGS:
        out[f"received_lag_{k}"] = received.shift(k - 1)

    out["censored_at_t"] = p["flag_censored"].astype(float).where(p["reported"])
    out["stock_end_at_t"] = p["stock_end"]
    out["distributed_raw_at_t"] = p["stock_distributed"]
    for column in CATEGORICAL:
        out[column] = out[column].astype("category")
    return out


def _for_horizon(features, horizon):
    target_index = features["month_index"] + horizon
    return features.assign(
        horizon=horizon,
        target_index=target_index,
        target_month_of_year=target_index % 12 + 1,
    )


def training_table(panel, features, cfg):
    """One row per series, origin t and horizon h whose target y at t+h is valid."""
    targets = panel.loc[panel["y"].notna(), SERIES_KEYS + ["month_index", "y"]]
    targets = targets.rename(columns={"month_index": "target_index", "y": "target"})
    frames = [
        _for_horizon(features, horizon).merge(targets, on=SERIES_KEYS + ["target_index"])
        for horizon in cfg["horizons"]
    ]
    table = pd.concat(frames, ignore_index=True)
    return table.astype({column: features[column].dtype for column in CATEGORICAL})


def prediction_rows(features, origin, cfg):
    """Feature rows for every series with a panel row at the origin, one per horizon."""
    at_origin = features[features["month_index"] == origin]
    frames = [_for_horizon(at_origin, horizon) for horizon in cfg["horizons"]]
    return pd.concat(frames, ignore_index=True)
