"""Shared loading, labels and chart settings. The app reads only from outputs/."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"

# Most to least urgent. Colors are state colors, and every use carries a text code too.
LABELS = {
    "Stockout risk": ("SR", "#d03b3b"),
    "Ordered, never received": ("OR", "#ec835a"),
    "Understocked": ("US", "#fab219"),
    "Overstocked": ("OS", "#2a78d6"),
    "Adequate": ("OK", "#0ca30c"),
    "No expected demand": ("ND", "#c3c2b7"),
    "Not stocked": ("NS", "#dddcd5"),
    "No current data": ("NC", "#f0efec"),
}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5df"
FORECAST = "#2a78d6"
BAND = "rgba(42, 120, 214, 0.18)"


@st.cache_data
def load_parquet(name):
    return pd.read_parquet(OUTPUTS / f"{name}.parquet")


@st.cache_data
def load_json(name):
    with open(OUTPUTS / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def product_names():
    return load_json("app_meta")["products"]


def short_name(name):
    return " ".join(name.replace("(", "").split()[:2])


def banner():
    month = load_json("app_meta")["as_of_label"]
    st.warning(f"Historical data as of {month}. Demonstration only; not for operational use.")


def status_with_names():
    status = load_parquet("stock_status")
    names = product_names()
    return status.assign(product=status["product_code"].map(lambda c: names.get(c, c)))


def chart_layout(**overrides):
    axis = dict(
        gridcolor=GRID, linecolor=GRID, zeroline=False, automargin=True,
        tickfont=dict(color=INK_SECONDARY, size=11),
    )
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=12), xaxis=axis, yaxis=axis,
        margin=dict(l=8, r=8, t=8, b=8), hovermode="x unified",
    ) | overrides


def label_key():
    """A legend of every label's code and color, so the heatmap needs no color reading."""
    chips = "".join(
        f'<span style="display:inline-block;margin:0 14px 6px 0;font-size:0.85rem;white-space:nowrap">'
        f'<span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:{color};'
        f'border:1px solid {GRID};margin-right:5px;vertical-align:-1px"></span>'
        f"<b>{code}</b> {label}</span>"
        for label, (code, color) in LABELS.items()
    )
    st.markdown(chips, unsafe_allow_html=True)
