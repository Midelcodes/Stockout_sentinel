import numpy as np
import pandas as pd
import pytest

from src import baselines
from src.backtest import (
    assign_tiers,
    baseline_predictions,
    mase_scales,
    metrics_table,
    origin_indexes,
    score,
    shortfall_share,
    under_forecast_rate,
    wape,
)
from src.clean import add_flags, flag_all_zero, flag_balance, flag_censored, flag_counts, flag_fake_zero
from src.panel import build_panel

CFG = {
    "fake_zero_min_median": 5, "fake_zero_window": 6, "fake_zero_min_history": 1,
    "balance_tolerance": 1, "backtest_origins_back": [2], "horizons": [1, 2], "volume_tiers": 3,
    "mase_min_points": 6,
}


def logistics(rows):
    """SYNTHETIC rows: (site, product, year, month, initial, received, distributed, adjustment, end)."""
    columns = ["site_code", "product_code", "year", "month", "stock_initial", "stock_received",
               "stock_distributed", "stock_adjustment", "stock_end"]
    df = pd.DataFrame(rows, columns=columns)
    return df.assign(region="R", district="D", average_monthly_consumption=7,
                     stock_stockout_days=0, stock_ordered=0.0)


def balanced(distributed, site="S", product="P", start_stock=1000, year=2018):
    """SYNTHETIC series with one row per month, in stock throughout."""
    rows, stock = [], start_stock
    for i, d in enumerate(distributed):
        rows.append((site, product, year + i // 12, i % 12 + 1, stock, 0, d, 0, stock - d))
        stock -= d
    return rows


def clean_panel(rows, **overrides):
    df = logistics(rows)
    return add_flags(build_panel(df.assign(**overrides)), CFG)


# Panel

def test_panel_fills_gaps_with_nan_not_zero():
    rows = balanced([10, 10, 10])
    panel = build_panel(logistics([rows[0], rows[2]]))
    assert list(panel["reported"]) == [True, False, True]
    assert panel["stock_distributed"].isna().tolist() == [False, True, False]
    assert panel["district"].tolist() == ["D", "D", "D"]


def test_panel_runs_each_series_to_the_dataset_last_month():
    rows = balanced([10, 10, 10], site="A") + balanced([10], site="B")
    panel = build_panel(logistics(rows))
    assert panel.groupby("site_code").size().to_dict() == {"A": 3, "B": 3}
    assert panel[panel["site_code"] == "B"]["reported"].tolist() == [True, False, False]


def test_panel_starts_each_series_at_its_first_month():
    late = [("B", "P", 2018, 3, 50, 0, 5, 0, 45)]
    panel = build_panel(logistics(balanced([10, 10, 10], site="A") + late))
    assert len(panel[panel["site_code"] == "B"]) == 1


def test_panel_rejects_duplicates():
    rows = balanced([10])
    with pytest.raises(ValueError, match="Duplicate"):
        build_panel(logistics(rows + rows))


def test_panel_crosses_year_end():
    rows = balanced([10] * 14)
    panel = build_panel(logistics(rows))
    assert panel["month_start"].iloc[-1] == pd.Timestamp("2019-02-01")
    assert (panel["year"].iloc[-1], panel["month"].iloc[-1]) == (2019, 2)


# Flag rules

def test_fake_zero_needs_history_at_or_above_the_threshold():
    flagged = clean_panel(balanced([5, 5, 5, 5, 5, 5, 0]))
    assert flagged["flag_fake_zero"].tolist() == [False] * 6 + [True]
    assert np.isnan(flagged["y"].iloc[-1])
    below = clean_panel(balanced([4, 4, 4, 4, 4, 4, 0]))
    assert not below["flag_fake_zero"].any()
    assert below["y"].iloc[-1] == 0


def test_fake_zero_needs_stock_on_hand():
    rows = balanced([10, 10, 10]) + [("S", "P", 2018, 4, 0, 0, 0, 0, 0)]
    panel = build_panel(logistics(rows))
    assert not flag_fake_zero(panel, CFG).any()


def test_fake_zero_ignores_rows_with_positive_distribution():
    assert not clean_panel(balanced([10, 10, 10, 3]))["flag_fake_zero"].any()


def test_fake_zero_window_uses_reported_months_only():
    rows = balanced([10] * 6 + [0])
    panel = build_panel(logistics(rows[:3] + rows[3:6] + rows[6:]))
    gapped = rows[:3] + [(s, p, y + 1, m, *rest) for s, p, y, m, *rest in rows[3:]]
    assert flag_fake_zero(build_panel(logistics(gapped)), CFG).sum() == 1
    assert flag_fake_zero(panel, CFG).sum() == 1


def test_fake_zero_window_excludes_the_current_month_and_older_months():
    assert clean_panel(balanced([50] * 3 + [1] * 6 + [0]))["flag_fake_zero"].sum() == 0


def test_all_zero_flag_and_target():
    rows = balanced([10]) + [("S", "P", 2018, 2, 0, 0, 0, 0, 0)]
    flagged = clean_panel(rows)
    assert flagged["flag_all_zero"].tolist() == [False, True]
    assert np.isnan(flagged["y"].iloc[1])


def test_all_zero_needs_every_stock_field_zero():
    row = ("S", "P", 2018, 1, 0, 5, 0, -5, 0)
    assert not flag_all_zero(build_panel(logistics([row]))).any()


def test_censored_by_stockout_days():
    panel = build_panel(logistics(balanced([10])).assign(stock_stockout_days=3))
    assert flag_censored(panel).all()


def test_censored_by_running_out():
    rows = [("S", "P", 2018, 1, 10, 0, 10, 0, 0)]
    flagged = clean_panel(rows)
    assert flagged["flag_censored"].all()
    assert flagged["y"].isna().all()
    assert flagged["stock_distributed"].iloc[0] == 10


def test_zero_stock_end_with_nothing_distributed_is_not_censored():
    panel = build_panel(logistics([("S", "P", 2018, 1, 5, 0, 0, -5, 0)]))
    assert not flag_censored(panel).any()


def test_balance_flag_tolerance_and_no_correction():
    rows = [("S", "P", 2018, 1, 100, 0, 10, 0, 91), ("S", "P", 2018, 2, 91, 0, 10, 0, 83)]
    flagged = clean_panel(rows)
    assert flagged["flag_balance"].tolist() == [False, True]
    assert flagged["stock_end"].tolist() == [91, 83]
    assert flagged["y"].tolist() == [10, 10]


def test_balance_flag_skips_unreported_months():
    rows = balanced([10, 10, 10])
    panel = build_panel(logistics([rows[0], rows[2]]))
    assert not flag_balance(panel, CFG).any()


def test_target_is_nan_for_unreported_months():
    rows = balanced([10, 10, 10])
    assert clean_panel([rows[0], rows[2]])["y"].isna().tolist() == [False, True, False]


def test_flag_counts_add_up():
    rows = balanced([10, 10, 0]) + [("S", "P", 2018, 4, 0, 0, 0, 0, 0)]
    counts = flag_counts(clean_panel(rows), CFG)
    assert counts["flag_fake_zero"] == 1
    assert counts["flag_fake_zero_short_history"] == 1
    assert counts["flag_all_zero"] == 1
    assert counts["valid_y"] == 2


# Reporting quality

def test_quality_score_uses_only_months_up_to_each_month():
    rows = balanced([10, 10, 10, 10]) + [("S", "P", 2018, 5, 0, 0, 0, 0, 0)]
    panel = clean_panel(rows)
    assert panel["site_reporting_quality"].tolist() == [0, 0, 0, 0, 0.2]


def test_quality_score_at_t_ignores_later_data():
    rows = balanced([10, 10, 10, 10])
    bad_future = rows[:3] + [("S", "P", 2018, 4, 0, 0, 0, 0, 0)]
    before = clean_panel(rows)["site_reporting_quality"].iloc[:3]
    after = clean_panel(bad_future)["site_reporting_quality"].iloc[:3]
    assert before.tolist() == after.tolist()


def test_quality_score_skips_series_never_stocked():
    never = [("S", "Q", 2018, m, 0, 0, 0, 0, 0) for m in (1, 2, 3)]
    panel = clean_panel(balanced([10, 10, 10]) + never)
    assert panel["site_reporting_quality"].eq(0).all()


def test_quality_score_treats_an_all_zero_month_with_an_order_as_a_stockout():
    rows = balanced([10, 10, 10]) + [("S", "P", 2018, 4, 0, 0, 0, 0, 0)]
    df = logistics(rows)
    df.loc[3, "stock_ordered"] = 50.0
    panel = add_flags(build_panel(df), CFG)
    assert panel["flag_all_zero"].iloc[3]
    assert np.isnan(panel["y"].iloc[3])
    assert panel["site_reporting_quality"].iloc[3] == 0


def test_quality_score_counts_a_series_from_its_first_activity():
    rows = [("S", "Q", 2018, 1, 0, 0, 0, 0, 0), ("S", "Q", 2018, 2, 0, 50, 5, 0, 45)]
    panel = clean_panel(rows)
    assert np.isnan(panel["site_reporting_quality"].iloc[0])
    assert panel["site_reporting_quality"].iloc[1] == 0.5


# Metrics

def test_wape():
    assert wape(pd.Series([10.0, 30.0]), pd.Series([20.0, 10.0])) == pytest.approx(0.75)


def test_wape_with_zero_total_is_nan():
    assert np.isnan(wape(pd.Series([0.0]), pd.Series([1.0])))


def test_under_forecast_rate_is_strict():
    y, forecast = pd.Series([10.0, 10.0, 10.0, 10.0]), pd.Series([9.0, 10.0, 11.0, 8.0])
    assert under_forecast_rate(y, forecast) == 0.5


def test_shortfall_share_counts_only_shortfalls():
    y, forecast = pd.Series([10.0, 30.0]), pd.Series([20.0, 10.0])
    assert shortfall_share(y, forecast) == pytest.approx(0.5)


def test_mase_scale_is_mean_absolute_step_up_to_origin():
    panel = clean_panel(balanced([10, 12, 16, 16, 18, 24, 500]))
    assert mase_scales(panel, panel["month_index"].iloc[5], 6).iloc[0] == pytest.approx(14 / 5)


def test_mase_scale_needs_six_valid_points():
    panel = clean_panel(balanced([10, 12, 16, 16, 18]))
    assert mase_scales(panel, panel["month_index"].iloc[-1], 6).isna().all()


def scored_rows(**columns):
    defaults = {"site_code": "S", "product_code": "P", "origin": 1}
    return pd.DataFrame({**defaults, **columns})


def test_score_mase_and_row_counts():
    rows = scored_rows(y=[10.0, 20.0], m=[8.0, 26.0], mase_scale=[2.0, 2.0])
    result = score(rows, "m")
    assert result["mase"] == pytest.approx(2.0)
    assert result["rows_scored"] == 2
    assert result["mase_rows"] == 2
    assert result["mase_series_skipped"] == 0


def test_score_skips_nan_and_zero_scales_and_counts_them():
    rows = pd.DataFrame({
        "site_code": ["A", "B", "C"], "product_code": "P", "origin": 1,
        "y": [10.0, 10.0, 10.0], "m": [8.0, 8.0, 8.0], "mase_scale": [2.0, 0.0, np.nan],
    })
    result = score(rows, "m")
    assert result["mase"] == pytest.approx(1.0)
    assert result["rows_scored"] == 3
    assert result["mase_rows"] == 1
    assert result["mase_series_skipped"] == 2


def test_score_ignores_rows_without_a_forecast():
    rows = scored_rows(y=[10.0, 20.0], m=[10.0, np.nan], mase_scale=[1.0, 1.0])
    assert score(rows, "m")["rows_scored"] == 1


def test_assign_tiers_are_ordered_and_keep_nan():
    tiers = assign_tiers(pd.Series([1.0, 2.0, 3.0, np.nan, 4.0, 5.0, 6.0]), 3)
    assert tiers.tolist()[:3] == [1, 1, 2]
    assert np.isnan(tiers.iloc[3])
    assert tiers.iloc[-1] == 3


# Baselines

def ten_months():
    return clean_panel(balanced([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))


def test_naive_is_last_valid_value():
    rows = balanced([10, 20, 30])
    panel = clean_panel([rows[0], rows[1]] + [("S", "P", 2018, 4, 970, 0, 30, 0, 940)])
    assert baselines.naive(panel, panel["month_index"].iloc[2]).iloc[0] == 20


def test_amc_3_is_mean_of_last_three_valid_months():
    panel = ten_months()
    assert baselines.amc_3(panel, panel["month_index"].iloc[4]).iloc[0] == 40


def test_amc_3_skips_invalid_months():
    rows = balanced([10, 20, 30, 40])
    panel = clean_panel(rows[:2] + rows[3:])
    assert baselines.amc_3(panel, panel["month_index"].iloc[-1]).iloc[0] == pytest.approx(70 / 3)


def test_amc_3_with_short_history_uses_what_exists():
    panel = clean_panel(balanced([10, 20]))
    assert baselines.amc_3(panel, panel["month_index"].iloc[-1]).iloc[0] == 15


def test_elmis_amc_is_the_value_recorded_at_the_origin():
    panel = ten_months()
    assert baselines.elmis_amc(panel, panel["month_index"].iloc[4]).iloc[0] == 7


def test_elmis_amc_is_absent_when_the_origin_is_unreported():
    rows = balanced([10, 20, 30])
    panel = clean_panel([rows[0], rows[2]])
    assert baselines.elmis_amc(panel, panel["month_index"].iloc[1]).empty


def test_seasonal_naive_uses_the_target_month_a_year_earlier():
    panel = clean_panel(balanced(list(range(10, 150, 10))))
    origin = panel["month_index"].iloc[12]
    assert baselines.seasonal_naive(panel, origin, origin + 1).iloc[0] == 20
    assert baselines.seasonal_naive(panel, origin, origin + 2).iloc[0] == 30


def test_seasonal_naive_falls_back_to_naive():
    panel = ten_months()
    origin = panel["month_index"].iloc[5]
    assert baselines.seasonal_naive(panel, origin, origin + 1).iloc[0] == 60


@pytest.mark.parametrize("name", ["naive", "amc_3", "elmis_amc"])
def test_baselines_ignore_data_after_the_origin(name):
    rows = balanced([10, 20, 30, 40, 50])
    changed = rows[:3] + [(s, p, y, m, a, b, 999, d, e) for s, p, y, m, a, b, _, d, e in rows[3:]]
    origin = clean_panel(rows)["month_index"].iloc[2]
    forecast = getattr(baselines, name)
    assert forecast(clean_panel(rows), origin).iloc[0] == forecast(clean_panel(changed), origin).iloc[0]


# Backtest

def test_origins_count_back_from_the_last_month():
    panel = ten_months()
    last = panel["month_index"].max()
    assert origin_indexes(panel, {"backtest_origins_back": [9, 6, 3]}) == [last - 9, last - 6, last - 3]


def test_predictions_cover_valid_targets_only():
    rows = balanced([10, 20, 30, 40, 50, 60])
    invalid = ("S", "P", 2018, 6, 850, 0, 0, 0, 850)
    predictions = baseline_predictions(clean_panel(rows[:5] + [invalid]), CFG | {"backtest_origins_back": [2]})
    assert predictions["horizon"].tolist() == [1]
    assert predictions["amc_3"].iloc[0] == 30


def test_flat_baselines_repeat_across_horizons():
    predictions = baseline_predictions(ten_months(), CFG)
    assert predictions["horizon"].tolist() == [1, 2]
    assert predictions["naive"].nunique() == 1
    assert predictions["amc_3"].nunique() == 1


def test_series_with_no_history_at_origin_has_no_baseline():
    late = [("B", "P", 2018, 10, 100, 0, 5, 0, 95)]
    predictions = baseline_predictions(clean_panel(balanced([10] * 10, site="A") + late), CFG)
    row = predictions[predictions["site_code"] == "B"].iloc[0]
    assert np.isnan(row["amc_3"])


def test_common_scope_uses_identical_rows_for_every_model():
    late = [("B", "P", 2018, 10, 100, 0, 5, 0, 95)]
    predictions = baseline_predictions(clean_panel(balanced([10] * 10, site="A") + late), CFG)
    metrics = metrics_table(predictions, baselines.BASELINES)
    overall = metrics[metrics["slice"] == "overall"]
    common = overall[overall["scope"] == "common"]["rows_scored"]
    assert common.nunique() == 1
    assert common.iloc[0] == 2
    assert overall[overall["scope"] == "available"]["rows_scored"].max() == 2


def test_slices_use_readable_labels():
    predictions = baseline_predictions(ten_months(), CFG)
    metrics = metrics_table(predictions, baselines.BASELINES)
    assert set(metrics[metrics["slice"] == "origin_month"]["slice_value"]) == {"2018-08"}
    assert set(metrics[metrics["slice"] == "volume_tier"]["slice_value"]) == {"3"}
