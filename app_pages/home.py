import streamlit as st

from src.config import PAPER_URL, SOURCES
from src.data_catalog import SAMPLE_CATALOG

st.html('<div class="gb-kicker">Interactive tutorial · AutoGluon</div>')
st.html(
    '<div class="gb-hero"><h1>See what Mitra-v2 does to a table.</h1><p class="gb-muted">Choose a sample, inspect every assumption, then run the real model only when you are ready.</p></div>'
)
st.space("small")

st.subheader("Featured tables")
featured = st.columns(2)
for column, name in zip(featured, ["Houses", "Machines"]):
    with column:
        spec = SAMPLE_CATALOG[name]
        st.html(
            f'<div class="gb-card"><div class="gb-kicker">{spec.task}</div><h3>{name}</h3><p class="gb-muted">{spec.story}</p></div>'
        )
        if st.button(f"Open {name}", key=f"open_{name}", width="stretch"):
            st.session_state.selected_dataset = name
            st.switch_page("app_pages/data.py")

st.subheader("More samples")
extras = st.columns(3)
for column, name in zip(extras, ["Adult income", "Credit-g", "Wine quality"]):
    with column:
        spec = SAMPLE_CATALOG[name]
        st.html(
            f'<div class="gb-card"><h3>{name}</h3><p class="gb-muted">{spec.story}</p></div>'
        )
        if st.button("Explore", key=f"open_{name}", width="stretch"):
            st.session_state.selected_dataset = name
            st.switch_page("app_pages/data.py")

st.subheader("Mitra-v2 at a glance")
facts = [
    ("77M", "parameters"),
    ("0", "real tables used for pretraining"),
    ("300+", "evaluation datasets in the paper"),
    ("Apache-2.0", "released weights and code"),
]
for column, (value, label) in zip(st.columns(4), facts):
    with column:
        st.html(
            f'<div class="gb-fact"><strong>{value}</strong><p class="gb-muted">{label}</p></div>'
        )

st.caption(f"Claims above come from the [Mitra-v2 Technical Report]({PAPER_URL}).")
with st.expander("Primary sources"):
    st.markdown("\n".join(f"- [{label}]({url})" for label, url in SOURCES.items()))
