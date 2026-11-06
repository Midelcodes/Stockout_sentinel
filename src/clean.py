"""Cleaning flags, the target y and the site reporting-quality score."""
import json

import pandas as pd

from src.io import SERIES_KEYS
from src.panel import STOCK_FIELDS


def flag_fake_zero(panel, cfg):
    """Zero distributed with stock on hand, after a series that was moving stock.

    The median uses raw stock_distributed over prior reported months, so a long
    run of zeros pulls it down and later zeros in that run are not flagged.
    """
    reported = panel[panel["reported"]]
    prior_median = reported.groupby(SERIES_KEYS)["stock_distributed"].transform(
        lambda s: s.shift().rolling(
            cfg["fake_zero_window"], min_periods=cfg["fake_zero_min_history"]
        ).median()
    )
    flag = (
        (reported["stock_distributed"] == 0)
        & (reported["stock_initial"] > 0)
        & (prior_median >= cfg["fake_zero_min_median"])
    )
    return flag.reindex(panel.index, fill_value=False)


def flag_all_zero(panel):
    return (panel[STOCK_FIELDS] == 0).all(axis=1)


def flag_censored(panel):
    ran_out = (panel["stock_end"] == 0) & (panel["stock_distributed"] > 0)
    return (panel["stock_stockout_days"] > 0) | ran_out


def flag_balance(panel, cfg):
    expected = (
        panel["stock_initial"] + panel["stock_received"]
        - panel["stock_distributed"] + panel["stock_adjustment"]
    )
    return (panel["stock_end"] - expected).abs() > cfg["balance_tolerance"]


def site_reporting_quality(panel):
    """Each site's flagged share at each month, from months up to that month only.

    A series counts once it has shown stock activity, so sites are not marked
    down for products they have never carried. An all-zero month with an order
    placed reads as a stockout, not bad reporting, so it is not flagged here.
    Higher is worse.
    """
    silent_zero = panel["flag_all_zero"] & ~(panel["stock_ordered"] > 0)
    p = panel[SERIES_KEYS + ["month_index"]].assign(
        reported=panel["reported"].astype(int),
        flagged=(panel["flag_fake_zero"] | silent_zero).astype(int),
        active=(panel["reported"] & ~panel["flag_all_zero"]).astype(int),
    )
    groups = p.groupby(SERIES_KEYS)
    started = groups["active"].cummax()
    p["reported_so_far"] = groups["reported"].cumsum() * started
    p["flagged_so_far"] = groups["flagged"].cumsum() * started
    totals = p.groupby(["site_code", "month_index"])[["reported_so_far", "flagged_so_far"]].sum()
    score = totals["flagged_so_far"] / totals["reported_so_far"]
    return score.rename("site_reporting_quality").reset_index()


def add_flags(panel, cfg):
    p = panel.copy()
    p["flag_fake_zero"] = flag_fake_zero(p, cfg)
    p["flag_all_zero"] = flag_all_zero(p)
    p["flag_censored"] = flag_censored(p)
    p["flag_balance"] = flag_balance(p, cfg)
    invalid = ~p["reported"] | p["flag_fake_zero"] | p["flag_all_zero"] | p["flag_censored"]
    p["y"] = p["stock_distributed"].astype(float).where(~invalid)
    return p.merge(site_reporting_quality(p), how="left", on=["site_code", "month_index"])


def flag_counts(p, cfg):
    reported = p[p["reported"]]
    prior_months = reported.groupby(SERIES_KEYS).cumcount()
    fake = reported["flag_fake_zero"]
    all_zero = reported[reported["flag_all_zero"]]
    return {
        "panel_rows": len(p),
        "reported": len(reported),
        "unreported": int((~p["reported"]).sum()),
        "flag_fake_zero": int(fake.sum()),
        "flag_fake_zero_short_history": int((fake & (prior_months < cfg["fake_zero_window"])).sum()),
        "flag_all_zero": len(all_zero),
        "flag_all_zero_with_recorded_amc": int((all_zero["average_monthly_consumption"] > 0).sum()),
        "flag_all_zero_with_order": int((all_zero["stock_ordered"] > 0).sum()),
        "flag_censored": int(p["flag_censored"].sum()),
        "flag_balance": int(p["flag_balance"].sum()),
        "valid_y": int(p["y"].notna().sum()),
        "valid_y_zero": int((p["y"] == 0).sum()),
        "valid_y_positive": int((p["y"] > 0).sum()),
    }


def run(cfg):
    panel = pd.read_parquet(cfg["paths"]["interim"] / "panel.parquet")
    clean = add_flags(panel, cfg)
    clean.to_parquet(cfg["paths"]["outputs"] / "panel_clean.parquet", index=False)
    counts = flag_counts(clean, cfg)
    with open(cfg["paths"]["outputs"] / "flag_counts.json", "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)
    for name, count in counts.items():
        print(f"  {name}: {count:,}")
    return "done"
