"""Streamlit entry point that initializes shared UI state and page navigation."""

from __future__ import annotations

import streamlit as st

from src.state import initialize_state
from src.theme import apply_theme, render_sidebar, render_top_bar

st.set_page_config(page_title="Glass Box · Mitra-v2", page_icon="◇", layout="wide")
initialize_state()
apply_theme()
render_sidebar()
render_top_bar()

pages = [
    st.Page("app_pages/home.py", title="Home", icon=":material/home:", default=True),
    st.Page("app_pages/data.py", title="Data", icon=":material/table_view:"),
    st.Page("app_pages/eda.py", title="EDA", icon=":material/query_stats:"),
    st.Page("app_pages/compare.py", title="Old way vs Mitra-v2", icon=":material/compare_arrows:"),
    st.Page("app_pages/inside.py", title="Inside Mitra-v2", icon=":material/hub:"),
    st.Page("app_pages/train.py", title="Train & Predict", icon=":material/model_training:"),
    st.Page("app_pages/results.py", title="Results", icon=":material/analytics:"),
    st.Page("app_pages/python_script.py", title="Python Script (CLI)", icon=":material/code:"),
]

st.navigation(pages, position="sidebar").run()
