import streamlit as st

from common import banner
from src.retrieval import Index, load_passages

st.set_page_config(page_title="Ask the docs", layout="wide")
st.title("Ask the docs")
banner()
st.info(
    "Search this project's own documentation: method, results, limitations, the budget optimiser, the "
    "guardrail and the LLM playbook. Answers are the passages themselves, each with its source, so "
    "nothing is generated and everything can be checked. This is the retrieval step of a "
    "retrieval-augmented setup."
)


@st.cache_resource
def index():
    return Index(load_passages())


examples = [
    "Are the expiry dates real?",
    "How does the model compare with AMC?",
    "What does the guardrail check?",
    "How should a programme team use an LLM safely?",
]
picked = st.pills("Try", examples, selection_mode="single")
query = st.text_input("Your question", value=picked or "")

if query:
    results = index().search(query, k=3)
    if not results:
        st.warning("No passage in the documentation answers this closely enough. Try other words.")
    for score, passage in results:
        with st.container(border=True):
            st.markdown(f"**{passage.heading}** · `{passage.source}` · relevance {score:.2f}")
            st.markdown(passage.text)
