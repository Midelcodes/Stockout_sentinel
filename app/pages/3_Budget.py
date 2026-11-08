import plotly.graph_objects as go
import streamlit as st

from common import INK_SECONDARY, banner, chart_layout, load_parquet, product_names
from src.budget import POLICIES, allocate, summarise

st.set_page_config(page_title="Budget", layout="wide")
st.title("Budget prioritisation")
banner()
st.info(
    "When the budget cannot fund every resupply order, which ones first? Each candidate is a site and "
    "product labelled Stockout risk. Its order covers the larger of p90 demand over the lead time and "
    "2 months of mean demand. The optimiser (a mixed-integer linear program) resolves as many risks as the "
    "budget allows, breaking ties toward more demand. "
    "**The data has no prices:** costs are in cost units (1 per stock unit) until a national price list "
    "is added in `config.yaml`."
)

names = product_names()
cands = load_parquet("budget_candidates")
total = float(cands["cost"].sum())

share = st.slider("Budget, as a share of funding every order", 5, 100, 50, step=5, format="%d%%") / 100
budget = share * total
st.caption(f"{len(cands):,} candidate orders cost {total:,.0f} cost units in total; this budget is {budget:,.0f}.")

results = {policy: allocate(cands, budget, policy) for policy in POLICIES}
summaries = {policy: summarise(cands, funded) for policy, funded in results.items()}
best = summaries["Optimised (MILP)"]
for column, policy in zip(st.columns(3), POLICIES):
    s = summaries[policy]
    delta = None if policy == POLICIES[0] else f"{s['risks_resolved'] - best['risks_resolved']:+,} vs optimised"
    column.metric(f"{policy}: risks resolved", f"{s['risks_resolved']:,} of {len(cands):,}", delta)

st.dataframe(
    {
        "Policy": POLICIES,
        "Risks resolved": [summaries[p]["risks_resolved"] for p in POLICIES],
        "Monthly demand covered (units)": [f"{summaries[p]['monthly_demand_covered']:,.0f}" for p in POLICIES],
        "Units ordered": [f"{summaries[p]['units_ordered']:,}" for p in POLICIES],
        "Low-data-quality sites funded": [summaries[p]["low_quality_sites_funded"] for p in POLICIES],
        "Cost used": [f"{summaries[p]['cost_used']:,.0f}" for p in POLICIES],
    },
    hide_index=True, width="stretch",
)

st.subheader("Risks resolved as the budget grows")
curve = load_parquet("budget_curve")
colors = {"Optimised (MILP)": "#2a78d6", "Most urgent first": "#d03b3b", "Largest demand first": "#a8a79f"}
figure = go.Figure()
for policy in POLICIES:
    line = curve[curve["policy"] == policy]
    figure.add_trace(go.Scatter(
        x=line["budget_share"] * 100, y=line["risks_resolved"], name=policy, mode="lines+markers",
        line=dict(color=colors[policy], width=2.5 if policy == POLICIES[0] else 1.5),
        hovertemplate="%{x:.0f}% budget: %{y} risks resolved<extra>" + policy + "</extra>",
    ))
figure.update_layout(**chart_layout(height=340, legend=dict(orientation="h", y=-0.2)))
figure.update_xaxes(title_text="Budget (% of funding every order)", ticksuffix="%")
figure.update_yaxes(title_text="Stockout risks resolved", title_font=dict(color=INK_SECONDARY))
st.plotly_chart(figure, theme=None, width="stretch", config={"displayModeBar": False})
st.caption(
    "Resolving the most risks means funding many smaller orders, so read it beside monthly demand covered "
    "in the table above. A ministry can change the objective, for example to weight by product or region."
)

st.subheader("Funded orders (optimised)")
funded = cands[results["Optimised (MILP)"]].assign(product=lambda d: d["product_code"].map(lambda c: names.get(c, c)))
table = funded[["district", "site_code", "product", "stock_end", "order_units", "cost", "days_to_stockout"]]
table = table.assign(days_to_stockout=table["days_to_stockout"].round(0)).sort_values(["district", "site_code"])
st.dataframe(table, hide_index=True, width="stretch")
st.download_button(
    "Download CSV", table.to_csv(index=False).encode("utf-8"), "funded_orders_optimised.csv", "text/csv",
)
st.caption("Every order is a suggestion for human review. Next step: trace funded orders through procurement and cash release.")
