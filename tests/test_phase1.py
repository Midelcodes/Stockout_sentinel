import pandas as pd
import pytest

from conftest import synthetic_logistics
from src import llm_brief
from src.io import MissingDataError, compare_schema, load_config, load_raw, read_csv
from src.pipeline import STAGES, main
from src.report import build_profile, profile_stats


def test_real_config_has_spec_defaults():
    cfg = load_config()
    assert cfg["seed"] == 42
    assert cfg["horizons"] == [1, 2, 3]
    assert cfg["backtest_origins_back"] == [9, 6, 3]
    assert cfg["fake_zero_min_median"] == 5
    assert (cfg["mos_min"], cfg["mos_max"]) == (2, 4)
    assert cfg["lead_time_months"] == 1
    assert cfg["quantiles"] == [0.5, 0.9]
    assert cfg["volume_tiers"] == 3
    assert cfg["llm_max_attempts"] == 3
    assert cfg["synthetic_expiry_months"] == [2, 18]


def test_schema_matches(cfg):
    assert compare_schema(load_raw(cfg)) == []


def test_schema_lists_missing_and_extra_columns(cfg):
    tables = load_raw(cfg)
    tables["products"] = tables["products"].rename(columns={"product_name": "name"})
    assert compare_schema(tables) == [
        "product.csv: expected column `product_name` is missing",
        "product.csv: unexpected column `name`",
    ]


def test_site_file_is_optional(cfg):
    (cfg["paths"]["raw"] / "service_delivery_site_data.csv").unlink()
    tables = load_raw(cfg)
    assert "sites" not in tables
    assert compare_schema(tables) == []


def test_missing_required_file_is_named(cfg):
    (cfg["paths"]["raw"] / "product.csv").unlink()
    with pytest.raises(MissingDataError, match="product.csv"):
        load_raw(cfg)


def test_empty_file_is_named(cfg):
    (cfg["paths"]["raw"] / "service_delivery_site_data.csv").write_bytes(b"")
    with pytest.raises(MissingDataError, match="service_delivery_site_data.csv"):
        load_raw(cfg)


def test_read_csv_falls_back_to_latin1(tmp_path):
    path = tmp_path / "accents.csv"
    path.write_bytes("district\nGagnoa Côte\n".encode("latin-1"))
    df = read_csv(path)
    assert df.attrs["encoding"] == "latin-1"
    assert df["district"].iloc[0] == "Gagnoa Côte"


def test_profile_counts(cfg):
    stats = profile_stats(synthetic_logistics(), cfg)
    assert stats["rows"] == 9
    assert (stats["sites"], stats["products"], stats["series"]) == (2, 1, 2)
    assert (stats["first_month"], stats["last_month"]) == ("2016-01", "2016-06")
    assert stats["series_length"]["min"] == 4
    assert stats["series_length"]["max"] == 5
    assert stats["series_with_internal_gaps"] == 1
    assert stats["internal_missing_months"] == 1
    assert stats["series_ending_early"] == 1
    assert stats["trailing_missing_months"] == 2
    assert stats["distributed_zeros"] == 2
    assert stats["distributed_zeros_with_stock"] == 1
    assert stats["stockout_rows"] == 1
    assert stats["duplicate_rows"] == 0
    assert stats["all_zero_rows"] == 1
    assert stats["series_never_active"] == 0
    assert stats["single_site_districts"] == 0


def test_site_without_logistics_rows_is_named(cfg):
    tables = load_raw(cfg)
    tables["logistics"] = tables["logistics"][lambda d: d["site_code"] == "S1"]
    text, _ = build_profile(tables, cfg)
    assert "with no logistics rows: 1 (S2)" in text


def test_balance_uses_tolerance(cfg):
    balance = profile_stats(synthetic_logistics(), cfg)["balance"]
    assert balance["rows_checked"] == 9
    assert balance["breaks"] == 1
    assert balance["breaks_exact"] == 2


def test_null_stock_field_is_skipped_by_balance_check(cfg):
    df = synthetic_logistics()
    df["stock_end"] = df["stock_end"].astype(float)
    df.loc[1, "stock_end"] = None
    balance = profile_stats(df, cfg)["balance"]
    assert balance["rows_not_checked"] == 1
    assert balance["breaks"] == 0


def test_duplicates_are_counted_once_per_series_month(cfg):
    df = synthetic_logistics()
    stats = profile_stats(pd.concat([df, df.iloc[[0]]]), cfg)
    assert stats["duplicate_rows"] == 1
    assert stats["series_length"]["max"] == 5


def test_undocumented_columns_are_listed(cfg):
    tables = load_raw(cfg)
    tables["dictionary"] = tables["dictionary"].iloc[1:]
    text, _ = build_profile(tables, cfg)
    assert "Train.csv: `year`" in text


def test_footnote_asterisk_counts_as_documented(cfg):
    tables = load_raw(cfg)
    tables["dictionary"] = tables["dictionary"].replace("year", "year*")
    text, _ = build_profile(tables, cfg)
    assert "Train.csv: `year`" not in text


def test_pipeline_runs_every_stage(long_config_path, capsys, monkeypatch):
    monkeypatch.setattr(llm_brief, "make_client", lambda: None)
    assert main(["--all", "--config", str(long_config_path)]) == 0
    out = capsys.readouterr().out
    for stage in STAGES:
        assert f"[{stage}]" in out


def test_pipeline_writes_profile(cfg, config_path):
    main(["--stage", "profile", "--config", str(config_path)])
    text = (cfg["paths"]["outputs"] / "profile.md").read_text(encoding="utf-8")
    assert "All headers match the spec." in text
    assert "Series count: 2" in text


def test_pipeline_stops_on_schema_mismatch(cfg, config_path, capsys):
    path = cfg["paths"]["raw"] / "Train.csv"
    synthetic_logistics().rename(columns={"stock_end": "closing"}).to_csv(path, index=False)
    assert main(["--all", "--config", str(config_path)]) == 2
    captured = capsys.readouterr()
    assert "[panel]" not in captured.out
    assert "2 schema mismatch(es)" in captured.err
    report = (cfg["paths"]["outputs"] / "profile.md").read_text(encoding="utf-8")
    assert "expected column `stock_end` is missing" in report


def test_pipeline_stops_when_data_is_missing(cfg, config_path, capsys):
    (cfg["paths"]["raw"] / "Train.csv").unlink()
    assert main(["--all", "--config", str(config_path)]) == 1
    assert "Train.csv" in capsys.readouterr().err
