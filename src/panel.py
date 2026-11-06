"""Panel: one row per site x product x month, gaps kept as NaN."""
import pandas as pd

from src.io import SERIES_KEYS, load_raw

STOCK_FIELDS = [
    "stock_initial", "stock_received", "stock_distributed",
    "stock_adjustment", "stock_end",
]


def month_start(index):
    """Convert a months-since-year-0 index to the first day of that month."""
    return pd.to_datetime({"year": index // 12, "month": index % 12 + 1, "day": 1})


def build_panel(logistics):
    """Extend each series from its first month to the dataset's last month.

    Months with no row get reported=False and NaN measurements, never 0.
    """
    df = logistics.assign(month_index=logistics["year"] * 12 + logistics["month"] - 1)
    if df.duplicated(SERIES_KEYS + ["month_index"]).any():
        raise ValueError("Duplicate site x product x month rows in the logistics data")

    last = df["month_index"].max()
    grid = df.groupby(SERIES_KEYS)["month_index"].min().reset_index(name="start")
    grid["month_index"] = [range(start, last + 1) for start in grid["start"]]
    grid = grid.drop(columns="start").explode("month_index")
    grid["month_index"] = grid["month_index"].astype("int64")

    locations = df.groupby("site_code")[["region", "district"]].first().reset_index()
    measured = df.drop(columns=["year", "month", "region", "district"])
    panel = grid.merge(measured, how="left", on=SERIES_KEYS + ["month_index"], indicator=True)
    panel["reported"] = panel.pop("_merge") == "both"
    panel = panel.merge(locations, how="left", on="site_code")

    panel["year"] = panel["month_index"] // 12
    panel["month"] = panel["month_index"] % 12 + 1
    panel["month_start"] = month_start(panel["month_index"])
    return panel.sort_values(SERIES_KEYS + ["month_index"]).reset_index(drop=True)


def run(cfg):
    panel = build_panel(load_raw(cfg)["logistics"])
    panel.to_parquet(cfg["paths"]["interim"] / "panel.parquet", index=False)
    added = int((~panel["reported"]).sum())
    return f"done: {len(panel):,} rows, {added:,} unreported months added"
