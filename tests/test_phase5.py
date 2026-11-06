import json
import re
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import llm_brief, report
from src.io import ROOT, load_config
from src.pipeline import main

APP = ROOT / "app"
PAGES = ["Home.py", "pages/1_Facility.py", "pages/2_Transfers.py", "pages/3_Model_card.py"]
BANNER = re.compile(r"^Historical data as of \w+ \d{4}\. Demonstration only; not for operational use\.$")


def metric_rows():
    """SYNTHETIC overall metrics."""
    rows = []
    for model, wape, shortfall, coverage in [
        ("lightgbm", 0.5, 0.33, 0.86), ("amc_3", 0.6, 0.30, None), ("elmis_amc", 0.55, 0.36, None),
        ("naive", 0.7, 0.35, None), ("seasonal_naive", 0.8, 0.4, None),
    ]:
        rows.append({
            "scope": "common", "slice": "overall", "slice_value": "all", "model": model,
            "rows_scored": 100, "wape": wape, "mase": 1.0, "under_forecast_rate": 0.4,
            "shortfall_share": shortfall, "p90_coverage": coverage,
        })
    return pd.DataFrame(rows)


def headline(amc_wape=0.6, model_wape=0.5, model_shortfall=0.33, amc_shortfall=0.30):
    return {
        "rows_scored": 100, "p90_coverage": 0.86, "model_wape": model_wape,
        "model_shortfall": model_shortfall,
        "amc_3": {
            "wape": amc_wape, "shortfall": amc_shortfall, "wape_change": model_wape / amc_wape - 1,
            "interval": [-0.2, -0.1],
        },
    }


def test_headline_reports_a_win_and_a_worse_shortfall():
    sentence = report.headline_sentence(headline())
    assert "LightGBM beats AMC-3" in sentence
    assert "-16.7%" in sentence
    assert "higher than AMC-3's" in sentence


def test_headline_says_so_when_the_model_does_not_beat_amc():
    sentence = report.headline_sentence(headline(amc_wape=0.5, model_wape=0.6))
    assert "does not beat AMC-3" in sentence


def test_results_table_is_filled_from_the_metrics():
    table = report.results_table(metric_rows())
    assert "| **LightGBM** | 0.500 | 1.000 | 40.0% | 33.0% | 86.0% | 100 |" in table
    assert "| AMC-3 | 0.600 | 1.000 | 40.0% | 30.0% | n/a | 100 |" in table
    assert len(table.splitlines()) == 7


def test_bias_text_orders_tiers_and_limits_regions():
    rows = [
        ("volume_tier", "1", 0.7, 100), ("volume_tier", "2", 0.6, 100), ("volume_tier", "3", 0.5, 100),
        ("quality_tier", "1", 0.4, 100), ("quality_tier", "2", 0.5, 100), ("quality_tier", "3", 0.6, 100),
        ("region", "A", 0.3, 100), ("region", "B", 0.9, 100), ("region", "TINY", 5.0, 3),
    ]
    metrics = pd.DataFrame([
        {"scope": "common", "model": "lightgbm", "slice": s, "slice_value": v, "wape": w, "rows_scored": n}
        for s, v, w, n in rows
    ])
    text = report.bias_text(metrics, {"region_min_rows": 50})
    assert "lowest to highest: 0.700, 0.600, 0.500." in text
    assert "best to worst: 0.400, 0.500, 0.600." in text
    assert "(50+ scored rows): 0.300 to 0.900." in text


def test_interval_brackets_a_clear_improvement():
    rows = pd.DataFrame({
        "site_code": [f"S{i}" for i in range(40)], "product_code": "P", "y": 10.0,
        "lightgbm": 9.0, "lightgbm_p50": 9.0, "naive": 5.0, "seasonal_naive": 5.0, "amc_3": 5.0, "elmis_amc": 5.0,
    })
    low, high = report.wape_change_interval(rows, "amc_3", 42)
    assert low == pytest.approx(-0.8)
    assert high == pytest.approx(-0.8)


def test_brief_status_text():
    briefs = [{"guardrail_status": "passed"}] * 3 + [{"guardrail_status": "template_fallback"}]
    assert report.brief_status_text(briefs) == "75.0% passed, 25.0% template_fallback"


def test_llm_models_reports_a_template_only_run():
    assert report.llm_models([{"model_id": None}]) == "a template (no LLM run yet)"
    assert report.llm_models([{"model_id": "m"}, {"model_id": None}]) == "`m`"


def test_pipeline_rebuilds_readme_and_app_files(long_config_path, monkeypatch):
    monkeypatch.setattr(llm_brief, "make_client", lambda: None)
    assert main(["--all", "--config", str(long_config_path)]) == 0
    cfg = load_config(long_config_path)
    outputs = cfg["paths"]["outputs"]
    text = cfg["readme_path"].read_text(encoding="utf-8")
    for heading in ["Problem", "Data", "Method", "Results", "Limitations", "Bias", "Next", "Run it"]:
        assert f"## {heading}\n" in text
    assert "SYNTHETIC" in text
    assert "{" not in text
    for name in ["actuals.parquet", "app_meta.json", "briefs.json", "forecasts.parquet",
                 "stock_status.parquet", "transfers.parquet", "backtest_metrics.parquet"]:
        assert (outputs / name).exists()


def test_actuals_file_has_no_raw_stock_columns(long_config_path, monkeypatch):
    monkeypatch.setattr(llm_brief, "make_client", lambda: None)
    main(["--all", "--config", str(long_config_path)])
    outputs = load_config(long_config_path)["paths"]["outputs"]
    columns = set(pd.read_parquet(outputs / "actuals.parquet").columns)
    assert columns == {"site_code", "product_code", "month_start", "y"}


def test_full_panel_is_kept_out_of_git():
    assert "outputs/panel_clean.parquet" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_app_never_imports_an_llm_client_or_raw_data():
    code = "\n".join(path.read_text(encoding="utf-8") for path in APP.rglob("*.py"))
    for forbidden in ["genai", "anthropic", "dotenv", "API_KEY", "data/raw", "panel_clean"]:
        assert forbidden not in code


def run_page(page):
    app = AppTest.from_file(str(APP / page), default_timeout=60)
    app.run()
    return app


needs_outputs = pytest.mark.skipif(
    not (ROOT / "outputs" / "app_meta.json").exists(), reason="run the pipeline first"
)


@needs_outputs
@pytest.mark.parametrize("page", PAGES)
def test_page_loads_and_carries_the_banner(page):
    app = run_page(page)
    assert not app.exception
    assert any(BANNER.match(w.value) for w in app.warning)


@needs_outputs
def test_transfers_page_warns_that_expiry_is_synthetic():
    app = run_page("pages/2_Transfers.py")
    assert any("SYNTHETIC" in e.value for e in app.error)


@needs_outputs
def test_facility_page_works_for_every_site_in_both_languages():
    app = run_page("pages/1_Facility.py")
    for site in app.selectbox[0].options[:1]:
        assert site
    sites = list(pd.read_parquet(ROOT / "outputs" / "stock_status.parquet")["site_code"].unique())
    for site in sites[:: max(1, len(sites) // 12)]:
        app.selectbox[0].set_value(site).run()
        assert not app.exception
        app.radio[0].set_value("FR").run()
        assert not app.exception
        app.radio[0].set_value("EN").run()


@needs_outputs
def test_model_card_slices_all_render():
    app = run_page("pages/3_Model_card.py")
    for value in ["volume_tier", "quality_tier", "region", "product_code", "horizon", "origin_month"]:
        app.selectbox[0].set_value(value).run()
        assert not app.exception
