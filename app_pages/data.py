"""Render uploads, deterministic split details, and row-level data inspection."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.load import display_frame, split_from_metadata
from src.state import clear_upload, current_data, set_upload

st.html('<div class="gb-kicker">01 · Data</div>')
st.title("Open the table")

uploaded = st.file_uploader("Upload CSV or Parquet", type=["csv", "parquet"])
if uploaded is not None and uploaded.name != st.session_state.uploaded_name:
    try:
        set_upload(uploaded.name, uploaded.getvalue())
        st.rerun()
    except Exception as exc:
        st.error(f"Could not read upload: {exc}")

if st.session_state.uploaded_data is not None:
    controls, action = st.columns([3, 1], vertical_alignment="bottom")
    with controls:
        target_col, problem_col = st.columns(2)
        columns = list(st.session_state.uploaded_data.columns)
        target_col.selectbox("Target column", columns, key="target")
        problem_col.selectbox(
            "Problem type",
            ["Auto", "Binary", "Multiclass", "Regression"],
            key="problem_choice",
        )
    with action:
        if st.button("Use bundled sample", width="stretch"):
            clear_upload()
            st.rerun()

try:
    frame, target, problem, source, metadata = current_data()
except Exception as exc:
    st.error(f"Could not load the selected table: {type(exc).__name__}: {exc}")
    st.stop()

st.caption(f"{source} · {metadata['story']}")

metrics = st.columns(5)
for column, label, value in zip(
    metrics,
    ["Total rows", "Known train", "Hidden test", "Columns", "Target"],
    [
        f"{len(frame):,}",
        f"{metadata['known_rows']:,}",
        f"{metadata['hidden_rows']:,}",
        len(frame.columns),
        target,
    ],
):
    column.metric(label, value)

chart_col, inspector_col = st.columns([1.55, 1])
with chart_col:
    st.subheader("Feature spread")
    feature = st.selectbox("Column", frame.columns, key="data_feature")
    figure = px.histogram(frame, x=feature, color_discrete_sequence=["#8064f4"])
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111a2e",
        plot_bgcolor="#111a2e",
        margin={"l": 24, "r": 16, "t": 24, "b": 24},
        yaxis_title="Rows",
    )
    st.plotly_chart(figure, width="stretch")

with inspector_col:
    st.subheader("Hidden row inspector")
    hidden_indices = metadata["hidden_indices"]
    selected_row = st.selectbox(
        "Hidden test row", hidden_indices, format_func=lambda value: f"Row {value:,}"
    )
    row = frame.loc[selected_row].astype("string").copy()
    row[target] = "••• hidden •••"
    st.html('<span class="gb-status-hidden">● Hidden</span>')
    st.dataframe(row.rename("Value").to_frame(), width="stretch", height=360)

st.subheader("Rows")
shown = display_frame(frame, metadata)
known, hidden = split_from_metadata(shown, metadata)
tabs = st.tabs(
    [
        f"All · {len(shown):,}",
        f"Known · {len(known):,}",
        f"Hidden · {len(hidden):,}",
    ]
)
views = [("all", shown), ("known", known), ("hidden", hidden)]
for tab, (label, view) in zip(tabs, views):
    with tab:
        st.dataframe(
            view,
            width="stretch",
            height=430,
            column_config={"Status": st.column_config.TextColumn("Status", width="small")},
        )
        st.download_button(
            "Download CSV",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name=f"{metadata['id']}-{label}.csv",
            mime="text/csv",
            key=f"download_{metadata['id']}_{label}",
            width="stretch",
        )
