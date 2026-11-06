import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (
    BAND,
    FORECAST,
    INK_SECONDARY,
    LABELS,
    banner,
    chart_layout,
    load_json,
    load_parquet,
    short_name,
    status_with_names,
)

BADGES = {
    "passed": ("Guardrail: passed", "green"),
    "passed_after_retry": ("Guardrail: passed after retry", "orange"),
    "template_fallback": ("Guardrail: template fallback", "gray"),
}

st.set_page_config(page_title="Facility", layout="wide")
st.title("Facility")
banner()

status = status_with_names()
at_risk = status[status["label"] == "Stockout risk"].groupby("site_code").size()
sites = (
    status[["site_code", "district"]].drop_duplicates()
    .assign(risk=lambda d: d["site_code"].map(at_risk).fillna(0))
    .sort_values(["risk", "site_code"], ascending=[False, True])
)
titles = dict(zip(sites["site_code"], sites["site_code"] + " · " + sites["district"]))
site = st.selectbox("Site, most stockout risks first", list(titles), format_func=titles.get)

rows = status[status["site_code"] == site]
rows = rows.assign(rank=rows["label"].map({label: i for i, label in enumerate(LABELS)}))
rows = rows.sort_values(["rank", "days_to_stockout"])

st.subheader("Brief")
language = st.radio("Language", ["EN", "FR"], horizontal=True, label_visibility="collapsed")
briefs = load_json("briefs")
brief = next(
    (b for b in briefs if b["site_code"] == site and b["lang"] == language.lower()), None
)
if brief:
    text, color = BADGES[brief["guardrail_status"]]
    st.badge(text, color=color)
    st.markdown(brief["text"])
    st.caption(
        "The guardrail checks that every number comes from the site's facts, that the format is right, "
        "and that next-month ranges match their product. It does not check every other number's placement."
    )
else:
    st.info("No brief is available for this site.")

st.subheader("Status")
table = rows.assign(
    **{
        "Product": rows["product"],
        "Label": rows["label"],
        "Stock": rows["stock_end"],
        "Age": rows["data_age_months"],
        "p50": rows["p50_h1"].round(),
        "p90": rows["p90_h1"].round(),
        "MOS": rows["mos"].round(1),
        "Days": rows["days_to_stockout"].round(),
        "Low quality": rows["low_data_quality"],
    }
)[["Product", "Label", "Age", "Stock", "p50", "p90", "MOS", "Days", "Low quality"]]
no_forecast = rows["label"].isin(["Not stocked", "Ordered, never received"]).values
table.loc[no_forecast, ["p50", "p90"]] = None
st.dataframe(table, hide_index=True, width="stretch")
st.caption(
    "Age is how many months old the stock figure is. p50 and p90 are next month's forecast range. "
    "MOS (months of stock) and Days (to stockout) use the mean forecast."
)

st.subheader("Demand and forecast")
st.caption("Gaps are months with no valid value: unreported, flagged, or a stockout month.")
actuals = load_parquet("actuals")
actuals = actuals[actuals["site_code"] == site]
forecasts = load_parquet("forecasts")
forecasts = forecasts[forecasts["site_code"] == site]
months = pd.date_range(actuals["month_start"].min(), forecasts["origin"].max(), freq="MS") if len(actuals) else []


def demand_chart(code):
    history = actuals[actuals["product_code"] == code].set_index("month_start")["y"].reindex(months)
    ahead = forecasts[forecasts["product_code"] == code].sort_values("target_month")
    figure = go.Figure()
    figure.add_scatter(
        x=ahead["target_month"], y=ahead["p90"], name="p90", mode="lines",
        line=dict(width=0), showlegend=False, hovertemplate="%{y:.0f}",
    )
    figure.add_scatter(
        x=ahead["target_month"], y=ahead["p50"], name="p50–p90 range", mode="lines",
        line=dict(width=0), fill="tonexty", fillcolor=BAND, hoverinfo="skip",
    )
    figure.add_scatter(
        x=history.index, y=history.values, name="Valid demand", mode="lines+markers",
        line=dict(color=INK_SECONDARY, width=2), marker=dict(size=5), connectgaps=False,
        hovertemplate="%{y:.0f}",
    )
    figure.add_scatter(
        x=ahead["target_month"], y=ahead["p50"], name="p50", mode="lines+markers",
        line=dict(color=FORECAST, width=2), marker=dict(size=8), hovertemplate="%{y:.0f}",
    )
    figure.update_layout(**chart_layout(
        height=280,
        legend=dict(orientation="h", y=1.14, x=0, font=dict(size=11)),
        margin=dict(l=8, r=8, t=34, b=8),
    ))
    figure.update_yaxes(rangemode="tozero", title_text="Units per month", title_font=dict(size=11))
    return figure


charted = rows[~rows["label"].isin(["Not stocked", "Ordered, never received"])]
charted = charted[charted["product_code"].isin(actuals["product_code"])]
if charted.empty:
    st.info("This site has no valid monthly demand to chart.")
for start in range(0, len(charted), 2):
    for column, (_, row) in zip(st.columns(2), charted.iloc[start:start + 2].iterrows()):
        column.markdown(f"**{short_name(row['product'])}** · {LABELS[row['label']][0]} {row['label']}")
        column.plotly_chart(
            demand_chart(row["product_code"]), theme=None, width="stretch",
            config={"displayModeBar": False}, key=f"chart-{row['product_code']}",
        )
