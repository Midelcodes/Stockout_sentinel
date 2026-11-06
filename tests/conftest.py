"""SYNTHETIC test fixtures. Tiny hand-built tables, written only to pytest's tmp_path."""
import pandas as pd
import pytest
import yaml

from src.io import EXPECTED_COLUMNS, ROOT, load_config


def _row(site, year, month, initial, received, distributed, adjustment, end, stockout_days=0):
    return {
        "year": year, "month": month, "region": "R1", "district": "D1",
        "site_code": site, "product_code": "P1",
        "stock_initial": initial, "stock_received": received,
        "stock_distributed": distributed, "stock_adjustment": adjustment,
        "stock_end": end, "average_monthly_consumption": 10,
        "stock_stockout_days": stockout_days, "stock_ordered": 0,
    }


def synthetic_logistics():
    """9 rows. S1 skips March; S2 stops in April, two months before the June end."""
    return pd.DataFrame([
        _row("S1", 2016, 1, 100, 0, 10, 0, 90),
        _row("S1", 2016, 2, 90, 0, 10, 0, 85),    # off by 5: breaks the identity
        _row("S1", 2016, 4, 85, 0, 0, 0, 85),     # zero distributed with stock on hand
        _row("S1", 2016, 5, 85, 0, 10, 0, 76),    # off by 1: within tolerance
        _row("S1", 2016, 6, 76, 0, 76, 0, 0, 3),  # stockout
        _row("S2", 2016, 1, 0, 0, 0, 0, 0),       # zero distributed, no stock on hand
        _row("S2", 2016, 2, 0, 50, 5, 0, 45),
        _row("S2", 2016, 3, 45, 0, 5, 0, 40),
        _row("S2", 2016, 4, 40, 0, 5, 0, 35),
    ], columns=EXPECTED_COLUMNS["logistics"])


@pytest.fixture
def config_path(tmp_path):
    """A SYNTHETIC raw directory and a config file pointing every path into tmp_path."""
    raw = tmp_path / "raw"
    raw.mkdir()
    synthetic_logistics().to_csv(raw / "Train.csv", index=False)
    pd.DataFrame({
        "site_code": ["S1", "S2"], "site_type": ["Hospital", "Clinic"],
        "site_region": ["R1", "R1"], "site_district": ["D1", "D1"],
        "site_latitude": [5.3, 5.4], "site_longitude": [-4.0, -4.1],
    }).to_csv(raw / "service_delivery_site_data.csv", index=False)
    pd.DataFrame({
        "product_type": ["Injectable"], "product_name": ["Product one"], "product_code": ["P1"],
    }).to_csv(raw / "product.csv", index=False)
    columns = [c for names in EXPECTED_COLUMNS.values() for c in names]
    pd.DataFrame({"variable": columns}).to_csv(raw / "data_dictionary.csv", index=False)

    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {
        "raw": str(raw),
        "interim": str(tmp_path / "interim"),
        "outputs": str(tmp_path / "outputs"),
    }
    cfg["readme_path"] = str(tmp_path / "README.md")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.fixture
def long_config_path(config_path):
    """The same SYNTHETIC setup with 30 months of 4 series, enough to train on."""
    rows = []
    for site, product, level in [("S1", "P1", 40), ("S1", "P2", 25), ("S2", "P1", 12), ("S2", "P2", 60)]:
        stock = 5000
        for i in range(30):
            distributed = level + i % 5
            row = _row(site, 2017 + i // 12, i % 12 + 1, stock, 0, distributed, 0, stock - distributed)
            rows.append(row | {"product_code": product})
            stock -= distributed
    frame = pd.DataFrame(rows, columns=EXPECTED_COLUMNS["logistics"])
    frame.to_csv(config_path.parent / "raw" / "Train.csv", index=False)
    return config_path


@pytest.fixture
def cfg(config_path):
    return load_config(config_path)
