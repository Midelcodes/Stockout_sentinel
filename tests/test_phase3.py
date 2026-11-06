import numpy as np
import pandas as pd
import pytest

from src import stock
from src.clean import add_flags
from src.features import FEATURES, origin_features, prediction_rows, training_table
from src.backtest import stocking_decision
from src.model import (
    MODELS,
    calibration_factor,
    fit_models,
    fold_factors,
    post_process,
    predict,
    production_forecast,
    widen_p90,
)
from src.panel import build_panel
from src.redistribute import (
    add_synthetic_expiry,
    distance_km,
    plan_transfers,
    recipients_and_donors,
)

CFG = {
    "seed": 42, "horizons": [1, 2, 3], "fake_zero_min_median": 5, "fake_zero_window": 6,
    "fake_zero_min_history": 1, "balance_tolerance": 1, "quantiles": [0.5, 0.9],
    "mos_min": 2, "mos_max": 4, "lead_time_months": 1, "no_demand_p50": 0.5, "days_per_month": 30,
    "volume_tiers": 3, "synthetic_expiry_months": [2, 18],
    "stock_max_age_months": 3, "unverified_zero_if_ordered": True,
    "lightgbm": {
        "n_estimators": 10, "learning_rate": 0.1, "num_leaves": 4, "min_child_samples": 2,
        "n_jobs": 1, "deterministic": True, "force_row_wise": True,
    },
}


def logistics(distributed, site="S", product="P", stock_start=5000):
    """SYNTHETIC monthly series from January 2018, in stock throughout."""
    rows, stock_level = [], stock_start
    for i, d in enumerate(distributed):
        rows.append({
            "site_code": site, "product_code": product, "year": 2018 + i // 12, "month": i % 12 + 1,
            "region": "R", "district": "D", "stock_initial": stock_level, "stock_received": 7 * i,
            "stock_distributed": d, "stock_adjustment": 0,
            "stock_end": stock_level + 7 * i - d, "average_monthly_consumption": 10,
            "stock_stockout_days": 0, "stock_ordered": 0.0,
        })
        stock_level += 7 * i - d
    return pd.DataFrame(rows)


def clean(df):
    return add_flags(build_panel(df), CFG)


VALUES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]


# Features

def test_lags_count_back_from_the_first_forecast_month():
    features = origin_features(clean(logistics(VALUES)))
    row = features.iloc[12]
    assert row["y_lag_1"] == 130
    assert row["y_lag_2"] == 120
    assert row["y_lag_3"] == 110
    assert row["y_lag_6"] == 80
    assert row["y_lag_12"] == 20


def test_rolling_windows_end_at_the_origin():
    features = origin_features(clean(logistics(VALUES)))
    assert features["y_mean_3"].iloc[5] == 50
    assert features["y_mean_6"].iloc[5] == 35
    assert features["y_std_3"].iloc[5] == pytest.approx(10)


def test_gap_months_count_as_calendar_months_in_lags():
    df = logistics([10, 20, 30, 40]).drop(index=2)
    features = origin_features(clean(df))
    assert np.isnan(features["y_lag_2"].iloc[3])
    assert features["y_lag_3"].iloc[3] == 20


def test_received_lags_and_stock_at_origin():
    df = logistics(VALUES)
    features = origin_features(clean(df))
    assert features["received_lag_1"].iloc[4] == df["stock_received"].iloc[4]
    assert features["received_lag_2"].iloc[4] == df["stock_received"].iloc[3]
    assert features["stock_end_at_t"].iloc[4] == df["stock_end"].iloc[4]


def test_censored_row_keeps_its_raw_value_as_a_feature_but_not_in_y():
    df = logistics([10, 10, 10])
    df.loc[1, ["stock_end", "stock_stockout_days"]] = [0, 4]
    panel = clean(df)
    features = origin_features(panel)
    assert np.isnan(panel["y"].iloc[1])
    assert features["distributed_raw_at_t"].iloc[1] == 10
    assert features["censored_at_t"].iloc[1] == 1


def test_no_leakage_changing_future_data_does_not_change_features_at_t():
    cut = 8
    original = logistics(VALUES)
    changed = original.copy()
    future = changed.index > cut
    for column in ["stock_distributed", "stock_received", "stock_end", "stock_initial"]:
        changed.loc[future, column] = 999
    changed.loc[future, "stock_stockout_days"] = 9

    a = origin_features(clean(original)).iloc[: cut + 1]
    b = origin_features(clean(changed)).iloc[: cut + 1]
    pd.testing.assert_frame_equal(a, b)


def test_no_leakage_truncating_future_data_does_not_change_features_at_t():
    cut = 8
    original = logistics(VALUES)
    a = origin_features(clean(original)).iloc[: cut + 1].reset_index(drop=True)
    b = origin_features(clean(original.iloc[: cut + 1])).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_training_target_is_y_at_t_plus_h():
    panel = clean(logistics(VALUES))
    features = origin_features(panel)
    table = training_table(panel, features, CFG)
    row = table[(table["month_index"] == panel["month_index"].iloc[4]) & (table["horizon"] == 3)]
    assert row["target"].iloc[0] == 80
    assert row["y_lag_1"].iloc[0] == 50


def test_training_table_drops_nan_targets():
    df = logistics(VALUES).drop(index=5)
    panel = clean(df)
    table = training_table(panel, origin_features(panel), CFG)
    assert table["target"].notna().all()
    assert panel["month_index"].iloc[5] not in set(table["target_index"])


def test_training_rows_up_to_an_origin_never_see_later_targets():
    panel = clean(logistics(VALUES))
    table = training_table(panel, origin_features(panel), CFG)
    origin = panel["month_index"].iloc[10]
    train = table[table["target_index"] <= origin]
    assert train["target"].max() == 110
    assert (train["month_index"] + train["horizon"]).max() == origin


def test_target_month_of_year_crosses_the_year_end():
    panel = clean(logistics(VALUES))
    rows = prediction_rows(origin_features(panel), panel["month_index"].iloc[10], CFG)
    assert rows["target_month_of_year"].tolist() == [12, 1, 2]


def test_every_feature_exists():
    panel = clean(logistics(VALUES))
    rows = prediction_rows(origin_features(panel), panel["month_index"].iloc[10], CFG)
    assert set(FEATURES) <= set(rows.columns)


# Models

def test_post_process_clips_at_zero_and_orders_quantiles():
    out = post_process(pd.DataFrame({"mean": [-1.0, 3.0], "p50": [-2.0, 8.0], "p90": [-1.0, 5.0]}))
    assert out["mean"].tolist() == [0, 3]
    assert out["p50"].tolist() == [0, 8]
    assert out["p90"].tolist() == [0, 8]


def trained():
    df = pd.concat([logistics(VALUES, site=s, stock_start=9000) for s in ("A", "B", "C")])
    panel = clean(df)
    features = origin_features(panel)
    return panel, features, training_table(panel, features, CFG)


def test_models_are_reproducible_with_a_fixed_seed():
    panel, features, table = trained()
    rows = prediction_rows(features, panel["month_index"].max(), CFG)
    first = predict(fit_models(table, CFG, 1.5), rows)
    second = predict(fit_models(table, CFG, 1.5), rows)
    pd.testing.assert_frame_equal(first, second)


def test_production_forecast_columns_and_horizons():
    panel, features, table = trained()
    forecast = production_forecast(table, features, CFG, panel["month_index"].max(), 1.5)
    assert list(forecast.columns) == [
        "site_code", "product_code", "origin", "horizon", "target_month", "mean", "p50", "p90",
    ]
    assert len(forecast) == 9
    assert forecast["target_month"].max() == pd.Timestamp("2019-06-01")
    assert (forecast["p90"] >= forecast["p50"]).all()
    assert (forecast[["mean", "p50", "p90"]] >= 0).all().all()


def covered_rows(y, p90, target_index=1):
    return pd.DataFrame({"y": y, "p90": p90, "target_index": target_index})


def test_calibration_needs_no_widening_when_coverage_is_already_met():
    rows = covered_rows([1.0] * 100, [5.0] * 100)
    assert calibration_factor(rows, 0.9) == 0


def test_widened_p90_reaches_the_target_on_the_calibration_rows():
    y = np.arange(1, 201, dtype=float)
    rows = covered_rows(y, [50.0] * 200)
    wide = widen_p90(rows["p90"], calibration_factor(rows, 0.9))
    assert (rows["y"] <= wide).mean() >= 0.9
    assert (rows["y"] <= rows["p90"]).mean() < 0.9


def test_widening_scales_with_the_forecast():
    assert widen_p90(pd.Series([0.0, 99.0]), 0.5).tolist() == [0.5, 149.0]


def test_widening_never_narrows():
    rows = covered_rows([1.0] * 10, [100.0] * 10)
    assert widen_p90(rows["p90"], calibration_factor(rows, 0.9)).tolist() == rows["p90"].tolist()


def test_no_calibration_rows_means_no_widening():
    assert calibration_factor(covered_rows([], []), 0.9) == 0


def test_each_origin_is_calibrated_only_on_what_was_known_by_then():
    folds = {
        10: covered_rows(np.arange(1, 101, dtype=float), [5.0] * 100, target_index=12),
        13: covered_rows([9999.0] * 100, [5.0] * 100, target_index=15),
    }
    factors = fold_factors(folds, [10, 13, 16], 0.9)
    assert factors[10] == 0
    early = calibration_factor(folds[10], 0.9)
    assert factors[13] == pytest.approx(early)
    assert factors[16] > factors[13]


def test_a_fold_with_targets_after_the_origin_is_not_used():
    folds = {10: covered_rows([9999.0] * 50, [5.0] * 50, target_index=14)}
    assert fold_factors(folds, [13], 0.9)[13] == 0


def stocking_rows():
    return pd.DataFrame({"y": [10.0, 10.0, 10.0, 10.0], "p90": [20.0, 20.0, 5.0, 15.0], "amc_3": [8.0, 8.0, 8.0, 8.0]})


def test_stocking_decision_measures_shortfall_and_excess():
    table = stocking_decision(stocking_rows(), "p90").set_index("policy")
    p90 = table.loc["Stock to calibrated p90"]
    assert p90["shortfall_share"] == pytest.approx(5 / 40)
    assert p90["excess_share"] == pytest.approx(25 / 40)
    assert p90["stock_vs_demand"] == pytest.approx(60 / 40)
    assert p90["months_short"] == 0.25


def test_stocking_decision_matches_total_stock_for_the_fair_comparison():
    table = stocking_decision(stocking_rows(), "p90").set_index("policy")
    same = table.loc["Stock to AMC-3, same total stock as p90"]
    assert same["stock_vs_demand"] == pytest.approx(table.loc["Stock to calibrated p90", "stock_vs_demand"])
    assert same["shortfall_share"] == 0


def test_model_table_compares_against_all_four_baselines():
    assert {"naive", "seasonal_naive", "amc_3", "elmis_amc", "lightgbm"} <= set(MODELS)


# Stock labels

def status_row(**overrides):
    row = {
        "stock_end": 30.0, "mean_h1": 10.0, "p90_lead": 15.0, "mos": 3.0,
        "ever_active": True, "ever_ordered": True, "unverified": False,
    }
    return row | overrides


def label(**overrides):
    return stock.assign_labels(pd.DataFrame([status_row(**overrides)]), CFG)[0]


def test_adequate_when_nothing_else_matches():
    assert label() == stock.ADEQUATE


def test_mos_boundaries_are_adequate():
    assert label(mos=2.0) == stock.ADEQUATE
    assert label(mos=4.0) == stock.ADEQUATE


def test_understocked_and_overstocked():
    assert label(mos=1.9) == stock.UNDERSTOCKED
    assert label(mos=4.1) == stock.OVERSTOCKED


def test_stockout_risk_beats_understocked():
    assert label(mos=1.0, stock_end=10.0, p90_lead=20.0) == stock.STOCKOUT_RISK


def test_stockout_risk_beats_overstocked():
    assert label(mos=5.0, stock_end=10.0, p90_lead=20.0) == stock.STOCKOUT_RISK


def test_stockout_risk_is_strictly_greater():
    assert label(stock_end=15.0, p90_lead=15.0) == stock.ADEQUATE


def test_no_expected_demand_beats_stockout_risk():
    assert label(mean_h1=0.5, mos=np.nan, stock_end=0.0, p90_lead=3.0) == stock.NO_EXPECTED_DEMAND


def test_never_active_labels_beat_demand_rules():
    assert label(ever_active=False, ever_ordered=False, stock_end=0.0) == stock.NOT_STOCKED
    assert label(ever_active=False, ever_ordered=True, stock_end=0.0) == stock.ORDERED_NEVER_RECEIVED


def test_no_current_data_beats_everything():
    assert label(stock_end=np.nan, ever_active=False) == stock.NO_CURRENT_DATA


def forecast_frame(site, mean, p90, p50=None):
    return pd.DataFrame({
        "site_code": site, "product_code": "P", "horizon": [1, 2, 3],
        "mean": mean, "p50": mean if p50 is None else p50, "p90": p90,
    })


def test_unverified_zero_beats_stockout_risk_and_no_demand():
    assert label(unverified=True, stock_end=0.0, p90_lead=9.0) == stock.UNVERIFIED_ZERO
    assert label(unverified=True, mean_h1=0.1) == stock.UNVERIFIED_ZERO


def test_never_active_and_no_current_data_beat_unverified_zero():
    assert label(unverified=True, ever_active=False, ever_ordered=False) == stock.NOT_STOCKED
    assert label(unverified=True, stock_end=np.nan) == stock.NO_CURRENT_DATA


def zero_last_month(ordered=0.0):
    df = logistics([10, 10, 10, 10], stock_start=100)
    df.loc[3, ["stock_initial", "stock_received", "stock_distributed", "stock_end"]] = 0
    df.loc[3, "stock_ordered"] = ordered
    return df


def test_all_zero_last_report_is_unverified_with_no_mos():
    status = stock.stock_status(clean(zero_last_month()), forecast_frame("S", 7.0, 9.0), CFG)
    assert status["label"].iloc[0] == stock.UNVERIFIED_ZERO
    assert np.isnan(status["mos"].iloc[0])


def test_fake_zero_last_report_is_unverified():
    df = logistics([10] * 7, stock_start=500)
    df.loc[6, "stock_distributed"] = 0
    df.loc[6, "stock_end"] = df.loc[6, "stock_initial"] + df.loc[6, "stock_received"]
    status = stock.stock_status(clean(df), forecast_frame("S", 7.0, 9.0), CFG)
    assert status["label"].iloc[0] == stock.UNVERIFIED_ZERO


def test_an_order_is_unverified_by_default_and_a_stockout_with_the_switch():
    panel = clean(zero_last_month(ordered=40.0))
    forecasts = forecast_frame("S", 7.0, 9.0)
    assert stock.stock_status(panel, forecasts, CFG)["label"].iloc[0] == stock.UNVERIFIED_ZERO
    keep = CFG | {"unverified_zero_if_ordered": False}
    assert stock.stock_status(panel, forecasts, keep)["label"].iloc[0] == stock.STOCKOUT_RISK


def test_running_out_with_a_real_report_stays_stockout_risk():
    df = logistics([10, 10, 10], stock_start=25)
    df.loc[2, ["stock_distributed", "stock_end"]] = [df.loc[2, "stock_initial"] + df.loc[2, "stock_received"], 0]
    status = stock.stock_status(clean(df), forecast_frame("S", 7.0, 9.0), CFG)
    assert status["label"].iloc[0] == stock.STOCKOUT_RISK


def test_months_of_stock_uses_the_mean_not_the_median():
    panel = clean(logistics([10, 10, 10], stock_start=100))
    status = stock.stock_status(panel, forecast_frame("S", 8.0, 12.0, p50=2.0), CFG)
    assert status["mos"].iloc[0] == pytest.approx(panel["stock_end"].iloc[-1] / 8)


def age_panel(months_silent):
    df = pd.concat([
        logistics([10] * 10, site="A"),
        logistics([10] * (10 - months_silent), site="B"),
    ])
    return clean(df)


def age_status(months_silent):
    forecasts = pd.concat([forecast_frame("A", 5.0, 6.0), forecast_frame("B", 5.0, 6.0)])
    return stock.stock_status(age_panel(months_silent), forecasts, CFG).set_index("site_code")


def test_a_report_within_the_limit_counts_and_shows_its_age():
    status = age_status(3)
    assert status.loc["A", "data_age_months"] == 0
    assert status.loc["B", "data_age_months"] == 3
    assert status.loc["B", "label"] != stock.NO_CURRENT_DATA
    assert status.loc["B", "stock_end"] > 0


def test_a_report_past_the_limit_is_no_current_data():
    status = age_status(4)
    assert status.loc["B", "label"] == stock.NO_CURRENT_DATA
    assert np.isnan(status.loc["B", "stock_end"])


def test_status_computes_mos_and_days():
    panel = clean(logistics([10, 10, 10], stock_start=100))
    status = stock.stock_status(panel, forecast_frame("S", 7.0, 9.0), CFG)
    end = panel["stock_end"].iloc[-1]
    assert status["mos"].iloc[0] == pytest.approx(end / 7)
    assert status["days_to_stockout"].iloc[0] == pytest.approx(end / (7 / 30))
    assert status["label"].iloc[0] == stock.OVERSTOCKED


def test_status_nulls_mos_when_demand_is_negligible():
    panel = clean(logistics([10, 10, 10], stock_start=100))
    status = stock.stock_status(panel, forecast_frame("S", 0.5, 2.0), CFG)
    assert np.isnan(status["mos"].iloc[0])
    assert np.isnan(status["days_to_stockout"].iloc[0])
    assert status["label"].iloc[0] == stock.NO_EXPECTED_DEMAND


def never_active_panel():
    df = logistics([0, 0, 0], stock_start=0)
    df[["stock_received", "stock_initial", "stock_end"]] = 0
    return df


def test_never_active_series_has_no_mos_and_gets_no_transfer():
    df = pd.concat([never_active_panel(), logistics([10, 10, 10], site="DONOR", stock_start=900)])
    forecasts = pd.concat([forecast_frame("S", 10.0, 12.0), forecast_frame("DONOR", 10.0, 12.0)])
    status = stock.stock_status(clean(df), forecasts, CFG).set_index("site_code")
    assert status.loc["S", "label"] == stock.NOT_STOCKED
    assert np.isnan(status.loc["S", "mos"])
    assert np.isnan(status.loc["S", "days_to_stockout"])
    assert status.loc["DONOR", "label"] == stock.OVERSTOCKED
    assert plan_transfers(status.reset_index(), {}, CFG).empty


def test_ordered_never_received_gets_no_transfer():
    never = never_active_panel().assign(stock_ordered=40.0)
    df = pd.concat([never, logistics([10, 10, 10], site="DONOR", stock_start=900)])
    forecasts = pd.concat([forecast_frame("S", 10.0, 12.0), forecast_frame("DONOR", 10.0, 12.0)])
    status = stock.stock_status(clean(df), forecasts, CFG).set_index("site_code")
    assert status.loc["S", "label"] == stock.ORDERED_NEVER_RECEIVED
    assert plan_transfers(status.reset_index(), {}, CFG).empty


def test_status_lead_time_sums_p90_over_the_lead_time():
    panel = clean(logistics([10, 10, 10], stock_start=100))
    end = panel["stock_end"].iloc[-1]
    two_months = CFG | {"lead_time_months": 2}
    status = stock.stock_status(panel, forecast_frame("S", 1.0, end / 2 + 1), two_months)
    assert status["label"].iloc[0] == stock.STOCKOUT_RISK


def test_low_data_quality_marks_the_worst_tier_beside_the_label():
    frames = [logistics([10, 10, 10], site=s) for s in ("A", "B", "C")]
    bad = logistics([10, 10, 0], site="C", stock_start=20)
    bad.loc[2, ["stock_initial", "stock_received", "stock_end"]] = 0
    panel = clean(pd.concat(frames[:2] + [bad]))
    forecasts = pd.concat([forecast_frame(s, 1.0, 2.0) for s in ("A", "B", "C")])
    status = stock.stock_status(panel, forecasts, CFG).set_index("site_code")
    assert status["low_data_quality"].to_dict() == {"A": False, "B": False, "C": True}
    assert status.loc["C", "label"] != "low_data_quality"


# Redistribution

def status_table(rows):
    """SYNTHETIC rows: (site, district, product, stock_end, p50)."""
    s = pd.DataFrame(rows, columns=["site_code", "district", "product_code", "stock_end", "mean_h1"])
    s["mos"] = s["stock_end"] / s["mean_h1"]
    s["days_to_stockout"] = s["stock_end"] / (s["mean_h1"] / 30)
    s["as_of"] = pd.Timestamp("2019-06-01")
    s["label"] = stock.ADEQUATE
    return s


def test_transfer_conservation_and_donor_floor():
    rng = np.random.default_rng(1)
    rows = [
        (f"S{i}", f"D{i % 3}", f"P{i % 2}", float(rng.integers(0, 400)), float(rng.integers(5, 40)))
        for i in range(60)
    ]
    status = status_table(rows)
    transfers = plan_transfers(status, {}, CFG)
    assert len(transfers) > 0

    sent = transfers.groupby(["from_site", "product_code"])["quantity"].sum()
    received = transfers.groupby(["to_site", "product_code"])["quantity"].sum()
    assert sent.sum() == received.sum()

    keyed = status.set_index(["site_code", "product_code"])
    after_donors = keyed.loc[sent.index, "stock_end"] - sent
    assert (after_donors >= CFG["mos_max"] * keyed.loc[sent.index, "mean_h1"]).all()
    after_recipients = keyed.loc[received.index, "stock_end"] + received
    assert (after_recipients <= CFG["mos_min"] * keyed.loc[received.index, "mean_h1"]).all()


def test_transfers_stay_within_district_and_product():
    status = status_table([
        ("A", "D1", "P", 0.0, 10.0), ("B", "D2", "P", 500.0, 10.0),
        ("C", "D1", "Q", 500.0, 10.0),
    ])
    assert plan_transfers(status, {}, CFG).empty


def test_quantities_are_whole_units_rounded_down():
    status = status_table([("A", "D", "P", 0.0, 2.75), ("B", "D", "P", 100.0, 2.75)])
    transfers = plan_transfers(status, {}, CFG)
    assert transfers["quantity"].tolist() == [5]
    assert transfers["quantity"].dtype.kind == "i"


def test_lowest_days_to_stockout_is_served_first():
    status = status_table([
        ("HIGHER", "D", "P", 15.0, 10.0), ("LOWER", "D", "P", 5.0, 10.0),
        ("DONOR", "D", "P", 47.0, 10.0),
    ])
    transfers = plan_transfers(status, {}, CFG)
    assert transfers["to_site"].tolist() == ["LOWER"]
    assert transfers["quantity"].tolist() == [7]


def test_soonest_expiry_donor_goes_first():
    status = status_table([
        ("R", "D", "P", 0.0, 10.0), ("D1", "D", "P", 100.0, 10.0),
        ("D2", "D", "P", 100.0, 10.0), ("D3", "D", "P", 100.0, 10.0),
    ])
    transfers = plan_transfers(status, {}, CFG)
    _, donors = recipients_and_donors(status, CFG)
    donors = add_synthetic_expiry(donors, status["as_of"].iloc[0], CFG)
    soonest = donors.sort_values("synthetic_expiry_date")["site_code"].iloc[0]
    assert transfers["from_site"].iloc[0] == soonest


def test_one_recipient_can_draw_on_several_donors():
    status = status_table([
        ("R", "D", "P", 0.0, 10.0), ("D1", "D", "P", 45.0, 10.0), ("D2", "D", "P", 45.0, 10.0),
        ("D3", "D", "P", 45.0, 10.0), ("D4", "D", "P", 45.0, 10.0),
    ])
    transfers = plan_transfers(status, {}, CFG)
    assert transfers["quantity"].sum() == 20
    assert len(transfers) == 4


def test_synthetic_expiry_is_seeded_and_in_range():
    status = status_table([(f"S{i}", "D", "P", 500.0, 10.0) for i in range(30)])
    as_of = status["as_of"].iloc[0]
    _, donors = recipients_and_donors(status, CFG)
    first = add_synthetic_expiry(donors, as_of, CFG)["synthetic_expiry_date"]
    second = add_synthetic_expiry(donors, as_of, CFG)["synthetic_expiry_date"]
    assert first.tolist() == second.tolist()
    assert first.min() >= pd.Timestamp("2019-08-01")
    assert first.max() <= pd.Timestamp("2020-12-01")


def test_no_donors_or_no_recipients_gives_an_empty_table():
    status = status_table([("A", "D", "P", 30.0, 10.0)])
    transfers = plan_transfers(status, {}, CFG)
    assert transfers.empty
    assert "synthetic_expiry_date" in transfers.columns


def test_null_mos_sites_are_neither_recipient_nor_donor():
    status = status_table([("A", "D", "P", 0.0, 10.0), ("B", "D", "P", 500.0, 10.0)])
    status.loc[1, "mos"] = np.nan
    assert plan_transfers(status, {}, CFG).empty


def test_stockout_risk_site_cannot_donate():
    status = status_table([("A", "D", "P", 0.0, 10.0), ("B", "D", "P", 500.0, 10.0)])
    status.loc[1, "label"] = stock.STOCKOUT_RISK
    assert plan_transfers(status, {}, CFG).empty


def test_distance_km():
    assert distance_km((5.0, -4.0), (5.0, -4.0)) == 0
    assert distance_km((0.0, 0.0), (0.0, 1.0)) == pytest.approx(111.2, abs=0.1)
    assert distance_km(None, (0.0, 1.0)) == float("inf")
