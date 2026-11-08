"""Budget-ceiling prioritisation of resupply orders.

Question it answers: with a fixed budget that cannot cover every order, which
site x product orders should be funded first?

Each candidate is one series labelled "Stockout risk". Its order brings stock up
to the larger of (a) the p90 demand over the lead time, so the site is covered
with about 90% confidence until the next delivery, and (b) `mos_min` months of
mean demand. Funding an order resolves that risk.

The optimiser solves a 0/1 knapsack as a mixed-integer linear program (SciPy's
HiGHS solver): maximise the number of stockout risks resolved within the budget,
breaking ties toward orders that serve more monthly demand. It is compared with
two rules people use by hand: "most urgent first" and "largest demand first".

The dataset has no prices. Costs use `unit_costs` from config.yaml, which
defaults to 1 cost unit per stock unit until a national price list is added.
"""
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from src.stock import STOCKOUT_RISK

POLICIES = ["Optimised (MILP)", "Most urgent first", "Largest demand first"]
CANDIDATE_COLUMNS = [
    "site_code", "product_code", "district", "region", "stock_end", "mean_h1",
    "p90_lead", "days_to_stockout", "low_data_quality", "order_units", "unit_cost", "cost",
]


def candidates(status, cfg):
    """Stockout-risk series with a measurable forecast, and the order that resolves each."""
    risk = status[(status["label"] == STOCKOUT_RISK) & status["mean_h1"].notna()].copy()
    target = np.maximum(risk["p90_lead"], cfg["mos_min"] * risk["mean_h1"])
    risk["order_units"] = np.ceil((target - risk["stock_end"]).clip(lower=0)).astype(int)
    risk = risk[risk["order_units"] > 0]
    prices = cfg.get("unit_costs") or {}
    default = cfg.get("default_unit_cost", 1.0)
    risk["unit_cost"] = risk["product_code"].map(lambda code: prices.get(code, default)).astype(float)
    risk["cost"] = risk["order_units"] * risk["unit_cost"]
    return risk[CANDIDATE_COLUMNS].sort_values(["district", "site_code", "product_code"]).reset_index(drop=True)


def optimise(cands, budget):
    """Boolean mask of funded orders that maximises risks resolved within the budget."""
    n = len(cands)
    if n == 0 or budget <= 0:
        return np.zeros(n, dtype=bool)
    demand = cands["mean_h1"].to_numpy()
    # One point per risk resolved; the demand term sums to less than one point,
    # so it only breaks ties between equally many resolved risks.
    value = 1.0 + demand / (demand.sum() + 1.0)
    result = milp(
        c=-value,
        constraints=LinearConstraint(cands["cost"].to_numpy()[None, :], -np.inf, budget),
        integrality=np.ones(n),
        bounds=Bounds(0, 1),
    )
    if not result.success:
        raise RuntimeError(f"budget optimiser failed: {result.message}")
    return result.x > 0.5


def fund_in_order(cands, order, budget):
    """Fund orders in the given order, skipping any that no longer fit the budget."""
    funded = np.zeros(len(cands), dtype=bool)
    remaining = budget
    costs = cands["cost"].to_numpy()
    for i in order:
        if costs[i] <= remaining:
            funded[i] = True
            remaining -= costs[i]
    return funded


def allocate(cands, budget, policy):
    if policy == "Optimised (MILP)":
        return optimise(cands, budget)
    if policy == "Most urgent first":
        order = np.lexsort((cands["site_code"], cands["days_to_stockout"].to_numpy()))
        return fund_in_order(cands, order, budget)
    if policy == "Largest demand first":
        order = np.lexsort((cands["site_code"], -cands["mean_h1"].to_numpy()))
        return fund_in_order(cands, order, budget)
    raise ValueError(f"unknown policy: {policy}")


def summarise(cands, funded):
    return {
        "risks_resolved": int(funded.sum()),
        "cost_used": float(cands.loc[funded, "cost"].sum()),
        "units_ordered": int(cands.loc[funded, "order_units"].sum()),
        "monthly_demand_covered": float(cands.loc[funded, "mean_h1"].sum()),
        "low_quality_sites_funded": int((funded & cands["low_data_quality"].to_numpy()).sum()),
    }


def budget_curve(cands, shares):
    """Each policy's result at several budgets, as shares of the cost of funding everything."""
    total = float(cands["cost"].sum())
    rows = []
    for share in shares:
        budget = share * total
        for policy in POLICIES:
            rows.append({"budget_share": share, "budget": budget, "policy": policy}
                        | summarise(cands, allocate(cands, budget, policy)))
    return pd.DataFrame(rows)


def run(cfg):
    outputs = cfg["paths"]["outputs"]
    status = pd.read_parquet(outputs / "stock_status.parquet")
    cands = candidates(status, cfg)
    cands.to_parquet(outputs / "budget_candidates.parquet", index=False)
    curve = budget_curve(cands, cfg.get("budget_shares", [0.1, 0.25, 0.5, 0.75, 1.0]))
    curve.to_parquet(outputs / "budget_curve.parquet", index=False)
    half = curve[curve["budget_share"] == 0.5].set_index("policy")["risks_resolved"]
    print(f"  candidates {len(cands):,}, total cost {cands['cost'].sum():,.0f}")
    if not half.empty:
        print("  at 50% budget: " + ", ".join(f"{p} {n}" for p, n in half.items()))
    return "done"
