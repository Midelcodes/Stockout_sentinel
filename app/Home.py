import plotly.graph_objects as go
import streamlit as st

from common import INK_SECONDARY, LABELS, banner, chart_layout, label_key, short_name, status_with_names

st.set_page_config(page_title="Stockout Sentinel", layout="wide")
st.title("Stockout Sentinel")
banner()

status = status_with_names()
districts = sorted(status["district"].unique())
chosen = st.multiselect("Districts", districts, placeholder="All districts")
view = status[status["district"].isin(chosen)] if chosen else status

st.subheader("Series by label")
counts = view["label"].value_counts()
labels = list(LABELS)
for row in (labels[:4], labels[4:]):
    for column, label in zip(st.columns(4), row):
        column.metric(f"{LABELS[label][0]} · {label}", f"{int(counts.get(label, 0)):,}")

st.subheader("District by product")
st.caption("Each cell shows the most urgent label among the district's sites. Hover for the breakdown.")
label_key()

rank = {label: i for i, label in enumerate(labels)}
view = view.assign(rank=view["label"].map(rank), short=view["product"].map(short_name))
cells = view.groupby(["district", "short"])
worst = cells["rank"].min().unstack()
risk_by_district = view[view["label"] == "Stockout risk"].groupby("district").size()
order = risk_by_district.reindex(worst.index).fillna(0).sort_values(kind="stable").index
worst = worst.loc[order]

breakdown = cells["label"].agg(
    lambda s: "<br>".join(f"{label}: {count}" for label, count in s.value_counts().items())
).unstack().reindex_like(worst)
codes = worst.map(lambda r: "" if r != r else LABELS[labels[int(r)]][0])

steps = len(labels)
scale = []
for i, label in enumerate(labels):
    color = LABELS[label][1]
    scale += [[i / steps, color], [(i + 1) / steps, color]]

heatmap = go.Figure(go.Heatmap(
    z=worst.values, x=list(worst.columns), y=list(worst.index),
    text=codes.values, texttemplate="%{text}", textfont=dict(size=10),
    customdata=breakdown.fillna("No series").values,
    hovertemplate="<b>%{y}</b><br>%{x}<br>%{customdata}<extra></extra>",
    colorscale=scale, zmin=-0.5, zmax=steps - 0.5, showscale=False, xgap=2, ygap=2,
    hoverongaps=False,
))
heatmap.update_layout(**chart_layout(
    height=max(260, 24 * len(worst) + 150), hovermode="closest",
))
heatmap.update_xaxes(side="top", tickangle=-40, showgrid=False)
heatmap.update_yaxes(showgrid=False, ticks="", tickfont=dict(size=11, color=INK_SECONDARY))
st.plotly_chart(heatmap, theme=None, width="stretch", config={"displayModeBar": False})

with st.expander("Table view"):
    table = view.groupby(["district", "product", "label"]).size().rename("series").reset_index()
    st.dataframe(table, hide_index=True, width="stretch")
