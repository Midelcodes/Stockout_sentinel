"""SYNTHETIC tests for budget-ceiling prioritisation."""
import numpy as np
import pandas as pd
import pytest

from src import budget
from src.stock import ADEQUATE, STOCKOUT_RISK

CFG = {"mos_min": 2, "default_unit_cost": 1.0, "unit_costs": {}}


def status_rows(rows):
    base = {
        "district": "D1", "region": "R1", "low_data_quality": False,
        "label": STOCKOUT_RISK, "days_to_stockout": 10.0,
    }
    return pd.DataFrame([base | r for r in rows])


def test_order_covers_the_larger_of_p90_and_min_months():
    status = status_rows([
        {"site_code": "S1", "product_code": "P1", "stock_end": 5, "mean_h1": 10.0, "p90_lead": 30.0},
        {"site_code": "S2", "product_code": "P1", "stock_end": 5, "mean_h1": 10.0, "p90_lead": 12.0},
    ])
    cands = budget.candidates(status, CFG).set_index("site_code")
    assert cands.loc["S1", "order_units"] == 25   # p90 30 beats 2 x 10
    assert cands.loc["S2", "order_units"] == 15   # 2 x 10 beats p90 12


def test_only_stockout_risk_series_with_a_forecast_are_candidates():
    status = status_rows([
        {"site_code": "S1", "product_code": "P1", "stock_end": 5, "mean_h1": 10.0, "p90_lead": 30.0},
        {"site_code": "S2", "product_code": "P1", "stock_end": 5, "mean_h1": 10.0, "p90_lead": 30.0,
         "label": ADEQUATE},
        {"site_code": "S3", "product_code": "P1", "stock_end": 5, "mean_h1": np.nan, "p90_lead": np.nan},
    ])
    assert list(budget.candidates(status, CFG)["site_code"]) == ["S1"]


def test_unit_costs_come_from_config():
    status = status_rows([
        {"site_code": "S1", "product_code": "P1", "stock_end": 0, "mean_h1": 5.0, "p90_lead": 10.0},
    ])
    cands = budget.candidates(status, CFG | {"unit_costs": {"P1": 2.5}})
    assert cands["cost"].iloc[0] == pytest.approx(25.0)


def cands_table(costs, demand=None, urgency=None):
    n = len(costs)
    return pd.DataFrame({
        "site_code": [f"S{i}" for i in range(n)], "cost": costs,
        "order_units": costs, "mean_h1": demand or [1.0] * n,
        "days_to_stockout": urgency or list(range(n)), "low_data_quality": [False] * n,
    })


def test_optimiser_beats_most_urgent_first_when_the_urgent_order_is_expensive():
    cands = cands_table([10, 3, 3, 3], urgency=[1, 2, 3, 4])
    urgent = budget.allocate(cands, 10, "Most urgent first")
    best = budget.allocate(cands, 10, "Optimised (MILP)")
    assert urgent.sum() == 1
    assert best.sum() == 3


def test_optimiser_never_exceeds_the_budget_and_is_optimal_on_a_small_case():
    rng = np.random.default_rng(0)
    costs = rng.integers(1, 20, size=12).astype(float)
    cands = cands_table(list(costs))
    for share in (0.2, 0.5, 0.8):
        b = share * costs.sum()
        funded = budget.allocate(cands, b, "Optimised (MILP)")
        assert costs[funded].sum() <= b + 1e-9
        # Fewest-cost-first is optimal for counting risks resolved.
        assert funded.sum() == budget.fund_in_order(cands, np.argsort(costs), b).sum()


def test_ties_go_to_more_demand():
    cands = cands_table([5, 5], demand=[1.0, 50.0])
    funded = budget.allocate(cands, 5, "Optimised (MILP)")
    assert list(funded) == [False, True]


def test_zero_budget_funds_nothing_and_full_budget_funds_everything():
    cands = cands_table([4, 6, 8])
    for policy in budget.POLICIES:
        assert budget.allocate(cands, 0, policy).sum() == 0
        assert budget.allocate(cands, 18, policy).sum() == 3


def test_curve_has_every_policy_at_every_share():
    curve = budget.budget_curve(cands_table([4, 6, 8]), [0.5, 1.0])
    assert len(curve) == 2 * len(budget.POLICIES)
    full = curve[curve["budget_share"] == 1.0]
    assert (full["risks_resolved"] == 3).all()
