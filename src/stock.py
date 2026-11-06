"""Months of stock, days to stockout and stock labels."""
import numpy as np
import pandas as pd

from src.backtest import assign_tiers
from src.io import SERIES_KEYS

NO_CURRENT_DATA = "No current data"
NOT_STOCKED = "Not stocked"
ORDERED_NEVER_RECEIVED = "Ordered, never received"
NO_EXPECTED_DEMAND = "No expected demand"
STOCKOUT_RISK = "Stockout risk"
UNDERSTOCKED = "Understocked"
OVERSTOCKED = "Overstocked"
ADEQUATE = "Adequate"


def _history_flags(panel):
    """Whether each series has ever shown stock activity or placed an order."""
    reported = panel[panel["reported"]]
    return pd.DataFrame({
        "ever_active": (~reported["flag_all_zero"]).groupby(
            [reported[k] for k in SERIES_KEYS]).any(),
        "ever_ordered": (reported["stock_ordered"] > 0).groupby(
            [reported[k] for k in SERIES_KEYS]).any(),
    })


def assign_labels(s, cfg):
    """First match wins."""
    no_demand = s["p50_h1"] <= cfg["no_demand_p50"]
    never_active = ~s["ever_active"]
    return np.select(
        [
            s["stock_end"].isna(),
            never_active & s["ever_ordered"],
            never_active,
            no_demand,
            s["p90_lead"] > s["stock_end"],
            s["mos"] < cfg["mos_min"],
            s["mos"] > cfg["mos_max"],
        ],
        [
            NO_CURRENT_DATA, ORDERED_NEVER_RECEIVED, NOT_STOCKED, NO_EXPECTED_DEMAND,
            STOCKOUT_RISK, UNDERSTOCKED, OVERSTOCKED,
        ],
        default=ADEQUATE,
    )


def stock_status(panel, forecasts, cfg):
    last = panel[panel["month_index"] == panel["month_index"].max()]
    columns = SERIES_KEYS + ["district", "region", "month_start", "stock_end", "site_reporting_quality"]
    s = last[columns].rename(columns={"month_start": "as_of"})

    h1 = forecasts[forecasts["horizon"] == 1].set_index(SERIES_KEYS)
    lead = forecasts[forecasts["horizon"] <= cfg["lead_time_months"]]
    s = s.merge(h1["p50"].rename("p50_h1"), how="left", left_on=SERIES_KEYS, right_index=True)
    s = s.merge(h1["p90"].rename("p90_h1"), how="left", left_on=SERIES_KEYS, right_index=True)
    s = s.merge(
        lead.groupby(SERIES_KEYS)["p90"].sum().rename("p90_lead"),
        how="left", left_on=SERIES_KEYS, right_index=True,
    )
    s = s.merge(_history_flags(panel), how="left", left_on=SERIES_KEYS, right_index=True)

    measurable = (s["p50_h1"] > cfg["no_demand_p50"]) & s["ever_active"]
    s["mos"] = (s["stock_end"] / s["p50_h1"]).where(measurable)
    s["days_to_stockout"] = (s["stock_end"] / (s["p50_h1"] / cfg["days_per_month"])).where(measurable)
    s["label"] = assign_labels(s, cfg)

    quality = s.groupby("site_code")["site_reporting_quality"].first()
    worst = assign_tiers(quality, cfg["volume_tiers"]) == cfg["volume_tiers"]
    s["low_data_quality"] = s["site_code"].map(worst).fillna(False).astype(bool)
    return s.reset_index(drop=True)


def run(cfg):
    outputs = cfg["paths"]["outputs"]
    panel = pd.read_parquet(outputs / "panel_clean.parquet")
    forecasts = pd.read_parquet(outputs / "forecasts.parquet")
    status = stock_status(panel, forecasts, cfg)
    status.to_parquet(outputs / "stock_status.parquet", index=False)
    print(status["label"].value_counts().to_string())
    print(f"  low_data_quality series: {int(status['low_data_quality'].sum()):,}")
    return "done"
