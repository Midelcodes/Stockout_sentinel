import streamlit as st

from common import banner, load_parquet, product_names

st.set_page_config(page_title="Transfers", layout="wide")
st.title("Transfers")
banner()
st.error(
    "SYNTHETIC expiry dates: the data has no batch or expiry records, so these dates are simulated. "
    "Every transfer is a suggestion that needs human review."
)

names = product_names()
transfers = load_parquet("transfers")
transfers = transfers.assign(product=transfers["product_code"].map(lambda c: names.get(c, c)))

left, right = st.columns(2)
districts = left.multiselect("Districts", sorted(transfers["district"].unique()), placeholder="All districts")
products = right.multiselect("Products", sorted(transfers["product"].unique()), placeholder="All products")
view = transfers
if districts:
    view = view[view["district"].isin(districts)]
if products:
    view = view[view["product"].isin(products)]

first, second = st.columns(2)
first.metric("Suggested transfers", f"{len(view):,}")
second.metric("Units moved", f"{int(view['quantity'].sum()):,}")

columns = ["district", "product", "from_site", "to_site", "quantity", "synthetic_expiry_date", "reason"]
table = view[columns].assign(synthetic_expiry_date=view["synthetic_expiry_date"].dt.strftime("%Y-%m-%d"))
table = table.rename(columns={"synthetic_expiry_date": "SYNTHETIC_expiry_date"})
st.dataframe(table, hide_index=True, width="stretch")
st.download_button(
    "Download CSV", table.to_csv(index=False).encode("utf-8"),
    "transfers_SYNTHETIC_expiry.csv", "text/csv",
)
