"""Render the shared dark Glass Box theme and navigation."""

import os

import streamlit as st

from src.data_catalog import SAMPLE_CATALOG


def apply_theme() -> None:
    """Inject the app-wide dark navy and purple Streamlit CSS theme."""
    st.html("""
    <style>
    :root { --ink:#eef0ff; --muted:#9ca6c6; --card:#111a2e; --line:#293654; --purple:#8064f4; }
    .stApp { background:radial-gradient(circle at 78% 0,#21194a 0,#0b1220 34%); color:var(--ink); }
    [data-testid="stSidebar"] { background:#080e1a; border-right:1px solid #202b44; }
    .block-container { max-width:1440px; padding-top:1.1rem; }
    h1,h2,h3 { letter-spacing:-.025em; }
    .gb-top { display:flex; align-items:center; gap:.55rem; min-height:2.2rem; }
    .gb-ready { color:#baf7d0; background:#123222; border:1px solid #2f7650; border-radius:999px; padding:.22rem .58rem; font:700 .72rem/1.3 monospace; }
    .gb-card,.gb-hero,.gb-fact,.gb-stage { background:rgba(17,26,46,.94); border:1px solid var(--line); border-radius:16px; padding:1.2rem; box-shadow:0 16px 44px #02040b45; }
    .gb-hero { padding:2rem; }
    .gb-card { min-height:140px; }
    .gb-card h3,.gb-fact h3 { margin:0 0 .4rem; }
    .gb-muted { color:var(--muted); }
    .gb-kicker { color:#ad98ff; font:700 .74rem/1.4 monospace; letter-spacing:.13em; text-transform:uppercase; }
    .gb-fact strong { color:#c4b6ff; font-size:1.7rem; }
    .gb-stage { border-top:3px solid var(--purple); margin-bottom:.55rem; }
    .gb-board { background:linear-gradient(145deg,#111a2e,#0c1426); border:1px solid var(--line); border-radius:18px; padding:1.1rem; }
    .gb-board-title { display:flex; justify-content:space-between; gap:1rem; align-items:center; margin-bottom:.8rem; }
    .gb-runtime { color:#d7ccff; background:#28204e; border:1px solid #5e4aae; border-radius:999px; padding:.28rem .62rem; font:700 .72rem/1.3 monospace; white-space:nowrap; }
    .gb-flow { display:grid; grid-template-columns:34px 1fr; gap:.7rem; align-items:center; background:#0d1629; border:1px solid #293654; border-radius:12px; padding:.72rem; margin:.5rem 0; }
    .gb-flow-number { display:grid; place-items:center; width:30px; height:30px; border-radius:50%; background:#6f55df; color:white; font:700 .74rem/1 monospace; }
    .gb-generated { color:#c6b8ff; font:700 .68rem/1.4 monospace; letter-spacing:.08em; text-transform:uppercase; }
    .gb-warning { background:#241f17; border:1px solid #715d2d; border-radius:13px; padding:1rem; }
    .gb-row-inspector { background:#0d1629; border:1px solid #35456a; border-radius:14px; padding:1rem; }
    .gb-status-known,.gb-status-hidden { display:inline-block; border-radius:999px; padding:.18rem .55rem; font:700 .72rem/1.4 monospace; }
    .gb-status-known { background:#123222; color:#baf7d0; border:1px solid #2f7650; }
    .gb-status-hidden { background:#31245e; color:#d7ccff; border:1px solid #765be0; }
    div[data-testid="stMetric"] { background:#111a2e; border:1px solid var(--line); border-radius:14px; padding:1rem; }
    .stButton>button { border-radius:10px; font-weight:700; }
    footer { visibility:hidden; }
    </style>
    """)


def render_sidebar() -> None:
    """Render shared dataset, fine-tune-step, and time-limit controls in the sidebar."""
    with st.sidebar:
        st.markdown("# ◇ GLASS BOX")
        st.caption("Mitra-v2, explained by doing")
        st.selectbox("Dataset", list(SAMPLE_CATALOG), key="selected_dataset")
        st.slider(
            "Fine-tune steps",
            min_value=0,
            max_value=100,
            step=5,
            key="fine_tune_steps",
            help="The installed MITRA default is 50 update steps.",
        )
        st.number_input(
            "Time limit · seconds",
            min_value=60,
            max_value=7200,
            step=60,
            key="time_limit",
            help="Applies to AutoGluon fitting. Downloads, prediction, and baseline time are additional.",
        )


def render_top_bar() -> None:
    """Render the readiness badge and shared fine-tune, bagging, and data controls."""
    ready, fine, copies, view = st.columns([1.25, 1, 1, 1.1], vertical_alignment="center")
    with ready:
        token_state = "HF token ready" if os.environ.get("HF_TOKEN") else "Local data ready"
        st.html(
            f'<div class="gb-top"><span class="gb-ready">● READY</span><span>{token_state}</span></div>'
        )
    with fine:
        st.toggle("Fine-tune", key="fine_tune", help="50 update steps. Requires CUDA.")
    with copies:
        st.toggle("Eight copies", key="eight_copies", help="Eight-fold bagging.")
    with view:
        if st.button("View Data", icon=":material/table_view:", width="stretch"):
            st.switch_page("app_pages/data.py")
