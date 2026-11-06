"""Greedy within-district transfer suggestions. Expiry dates are SYNTHETIC."""
import math

import numpy as np
import pandas as pd

from src.io import load_raw
from src.stock import STOCKOUT_RISK

COLUMNS = [
    "district", "product_code", "from_site", "to_site",
    "quantity", "reason", "synthetic_expiry_date",
]


def recipients_and_donors(status, cfg):
    recipients = status[status["mos"] < cfg["mos_min"]].copy()
    recipients["need"] = cfg["mos_min"] * recipients["mean_h1"] - recipients["stock_end"]
    donors = status[(status["mos"] > cfg["mos_max"]) & (status["label"] != STOCKOUT_RISK)].copy()
    donors["available"] = donors["stock_end"] - cfg["mos_max"] * donors["mean_h1"]
    return recipients, donors


def add_synthetic_expiry(donors, as_of, cfg):
    """One SYNTHETIC batch per donor, expiring uniformly within the configured months."""
    rng = np.random.default_rng(cfg["seed"])
    low, high = cfg["synthetic_expiry_months"]
    donors = donors.sort_values(["district", "product_code", "site_code"]).copy()
    earliest = as_of + pd.DateOffset(months=low)
    latest = as_of + pd.DateOffset(months=high)
    days = rng.integers(0, (latest - earliest).days + 1, size=len(donors))
    donors["synthetic_expiry_date"] = earliest + pd.to_timedelta(days, unit="D")
    return donors


def distance_km(a, b):
    """Great-circle distance between two (latitude, longitude) pairs."""
    if a is None or b is None:
        return math.inf
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + (
        math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371 * math.asin(math.sqrt(h))


def _plan_group(recipients, donors, coordinates, cfg):
    transfers = []
    supply = [
        {"site": d.site_code, "available": d.available, "expiry": d.synthetic_expiry_date}
        for d in donors.itertuples()
    ]
    for r in recipients.sort_values(["days_to_stockout", "site_code"]).itertuples():
        need = r.need
        here = coordinates.get(r.site_code)
        while need >= 1:
            usable = [d for d in supply if d["available"] >= 1]
            if not usable:
                return transfers
            donor = min(usable, key=lambda d: (
                d["expiry"], -d["available"], distance_km(here, coordinates.get(d["site"])),
            ))
            quantity = math.floor(min(need, donor["available"]))
            need -= quantity
            donor["available"] -= quantity
            transfers.append({
                "district": r.district, "product_code": r.product_code,
                "from_site": donor["site"], "to_site": r.site_code, "quantity": quantity,
                "reason": (
                    f"{r.site_code} is below {cfg['mos_min']} months of stock; "
                    f"{donor['site']} holds more than {cfg['mos_max']}"
                ),
                "synthetic_expiry_date": donor["expiry"],
            })
    return transfers


def plan_transfers(status, coordinates, cfg):
    recipients, donors = recipients_and_donors(status, cfg)
    donors = add_synthetic_expiry(donors, status["as_of"].iloc[0], cfg)
    transfers = []
    for key, group in recipients.groupby(["district", "product_code"]):
        supply = donors[(donors["district"] == key[0]) & (donors["product_code"] == key[1])]
        transfers += _plan_group(group, supply, coordinates, cfg)
    return pd.DataFrame(transfers, columns=COLUMNS)


def site_coordinates(tables):
    if "sites" not in tables:
        return {}
    sites = tables["sites"].dropna(subset=["site_latitude", "site_longitude"])
    return {
        s.site_code: (s.site_latitude, s.site_longitude) for s in sites.itertuples()
    }


def run(cfg):
    outputs = cfg["paths"]["outputs"]
    status = pd.read_parquet(outputs / "stock_status.parquet")
    transfers = plan_transfers(status, site_coordinates(load_raw(cfg)), cfg)
    transfers.to_parquet(outputs / "transfers.parquet", index=False)

    recipients, donors = recipients_and_donors(status, cfg)
    print(f"  recipients {len(recipients):,}, donors {len(donors):,}")
    print(f"  transfers {len(transfers):,}, units {int(transfers['quantity'].sum()):,}")
    return "done"
