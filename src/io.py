"""Config loading, raw CSV loading and schema verification."""
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

SERIES_KEYS = ["site_code", "product_code"]
OPTIONAL_FILES = {"sites"}

# Unverified expectations from the build spec. data_dictionary.csv has no
# expected layout, so it is reported verbatim rather than compared.
EXPECTED_COLUMNS = {
    "logistics": [
        "year", "month", "region", "district", "site_code", "product_code",
        "stock_initial", "stock_received", "stock_distributed", "stock_adjustment",
        "stock_end", "average_monthly_consumption", "stock_stockout_days",
        "stock_ordered",
    ],
    "sites": [
        "site_code", "site_type", "site_region", "site_district",
        "site_latitude", "site_longitude",
    ],
    "products": ["product_type", "product_name", "product_code"],
}


class MissingDataError(Exception):
    """A required raw file is not in the raw data directory."""


class SchemaMismatchError(Exception):
    """The raw files differ from the expected schema."""


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: ROOT / v for k, v in cfg["paths"].items()}
    cfg["readme_path"] = ROOT / cfg["readme_path"]
    return cfg


def ensure_dirs(cfg):
    for path in cfg["paths"].values():
        path.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    """Read a CSV with inferred dtypes, recording the encoding that worked."""
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        df.attrs["encoding"] = encoding
        return df
    raise ValueError(f"Could not decode {path}")


def load_raw(cfg):
    """Load every raw file that exists. Raises if a required file is missing."""
    raw_dir = cfg["paths"]["raw"]
    files = cfg["raw_files"]
    missing = [
        filename for name, filename in files.items()
        if name not in OPTIONAL_FILES and not (raw_dir / filename).exists()
    ]
    if missing:
        raise MissingDataError(
            f"Missing from {raw_dir}: {', '.join(missing)}. "
            "Place the raw CSVs there and run again."
        )
    present = {
        name: raw_dir / filename
        for name, filename in files.items()
        if (raw_dir / filename).exists()
    }
    empty = [path.name for path in present.values() if path.stat().st_size == 0]
    if empty:
        raise MissingDataError(
            f"Empty (0 bytes) in {raw_dir}: {', '.join(empty)}. "
            "Replace them with the real CSVs and run again."
        )
    tables = {name: read_csv(path) for name, path in present.items()}
    for name, df in tables.items():
        df.attrs["filename"] = files[name]
    return tables


def compare_schema(tables):
    """List every difference between the real headers and the expected columns."""
    mismatches = []
    for name, expected in EXPECTED_COLUMNS.items():
        if name not in tables:
            continue
        actual = list(tables[name].columns)
        filename = tables[name].attrs["filename"]
        mismatches += [
            f"{filename}: expected column `{c}` is missing"
            for c in expected if c not in actual
        ]
        mismatches += [
            f"{filename}: unexpected column `{c}`"
            for c in actual if c not in expected
        ]
    return mismatches
