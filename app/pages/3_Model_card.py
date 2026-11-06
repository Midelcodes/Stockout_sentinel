import pandas as pd
import streamlit as st

from common import banner, load_json, load_parquet, product_names

MODELS = {
    "lightgbm": "LightGBM", "amc_3": "AMC-3", "elmis_amc": "eLMIS AMC",
    "naive": "Naive", "seasonal_naive": "Seasonal naive",
}
SLICES = {
    "volume_tier": "Site volume tier (1 lowest, 3 highest)",
    "quality_tier": "Reporting-quality tier (1 best, 3 worst)",
    "region": "Region",
    "product_code": "Product",
    "horizon": "Horizon (months ahead)",
    "origin_month": "Backtest origin",
}

st.set_page_config(page_title="Model card", layout="wide")
st.title("Model card")
banner()

meta = load_json("app_meta")
h = meta["headline"]
st.markdown(f"**{meta['headline_sentence']}**")

amc, elmis = h["amc_3"], h["elmis_amc"]
for column, label, value in zip(st.columns(4), [
    "WAPE change vs AMC-3", "WAPE change vs eLMIS AMC", "p90 coverage (target about 90%)", "Rows scored",
], [
    f"{amc['wape_change']:+.1%}", f"{elmis['wape_change']:+.1%}", f"{h['p90_coverage']:.1%}", f"{h['rows_scored']:,}",
]):
    column.metric(label, value)

metrics = load_parquet("backtest_metrics")
metrics = metrics[metrics["scope"] == "common"]


def formatted(rows):
    return pd.DataFrame({
        "WAPE": rows["wape"].map("{:.3f}".format),
        "MASE": rows["mase"].map("{:.3f}".format),
        "Under-forecast": rows["under_forecast_rate"].map("{:.1%}".format),
        "Shortfall": rows["shortfall_share"].map("{:.1%}".format),
        "p90 coverage": rows["p90_coverage"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.1%}"),
        "Rows scored": rows["rows_scored"].map("{:,}".format),
    })


st.subheader("Model against baselines")
overall = metrics[(metrics["slice"] == "overall") & metrics["model"].isin(MODELS)]
overall = overall.set_index("model").loc[list(MODELS)]
table = formatted(overall)
table.insert(0, "Model", [MODELS[m] for m in overall.index])
st.dataframe(table, hide_index=True, width="stretch")
st.caption(
    "All models are scored on the same rows. WAPE rewards forecasting low on skewed demand, so read "
    "it beside shortfall: the volume a facility would have been short."
)

st.subheader("Error by slice")
slice_name = st.selectbox("Slice", list(SLICES), format_func=SLICES.get)
names = product_names()
rows = metrics[(metrics["slice"] == slice_name) & metrics["model"].isin(["lightgbm", "amc_3"])]
wide = rows.pivot(index="slice_value", columns="model", values=["wape", "shortfall_share"])
model_rows = rows[rows["model"] == "lightgbm"].set_index("slice_value")
sliced = pd.DataFrame({
    "Slice": [names.get(v, v) for v in wide.index],
    "LightGBM WAPE": wide[("wape", "lightgbm")].map("{:.3f}".format).values,
    "AMC-3 WAPE": wide[("wape", "amc_3")].map("{:.3f}".format).values,
    "LightGBM shortfall": wide[("shortfall_share", "lightgbm")].map("{:.1%}".format).values,
    "AMC-3 shortfall": wide[("shortfall_share", "amc_3")].map("{:.1%}".format).values,
    "p90 coverage": model_rows.loc[wide.index, "p90_coverage"].map("{:.1%}".format).values,
    "Rows": model_rows.loc[wide.index, "rows_scored"].values,
})
st.dataframe(sliced, hide_index=True, width="stretch")
st.caption("Slices with few rows or very low volumes are unstable. Check the row count before reading them.")

st.subheader("Limitations")
st.markdown("""
- Censored demand understates true need. `stock_stockout_days` is 0 in every row, so only months ending at zero stock are detected.
- Fake-zero rules are heuristics.
- Expiry data is SYNTHETIC.
- There are no supplier lead times or order cycles.
- The data covers 2016 to 2019, one country and one commodity class: contraceptives, not all essential medicines.
- Months of stock uses p50, a median that runs below mean demand, so "Overstocked" may be overstated.
- p90 coverage is below its target, so some stockout risks will be missed.
- `site_type` does not support a rural or urban comparison, so none is made.
- Briefs are LLM-written. The guardrail stops invented numbers, not a real number beside the wrong product.
- Every transfer needs human review.
""")

st.subheader("Data source and citation")
st.markdown("""
- Data: [USAID's Intelligent Forecasting Challenge](https://zindi.africa/competitions/usaids-intelligent-forecasting-challenge-model-future-contraceptive-use/data), an extract from Côte d'Ivoire's eLMIS. July to September 2019 were withheld by the competition.
- Context: a 2023 malaria-commodity proof of concept in Côte d'Ivoire [reported](https://community.dhis2.org/t/leveraging-dhis2-data-to-create-ai-driven-health-facility-forecasts-in-cote-d-ivoire/58136) an average 37% reduction in forecast error (DHIS2 2024 conference abstract). This is a reported result, not an independent finding.
""")
