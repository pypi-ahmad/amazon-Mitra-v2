"""Render the classic EDA workflow for the current table."""

from __future__ import annotations

import streamlit as st

from src.eda import (
    correlation_figure,
    distribution_figure,
    dtype_summary,
    format_bytes,
    missing_figure,
    pairplot_figure,
    profile_table,
    scatter_figure,
    target_figure,
)
from src.state import current_data
from src.training import split_data

st.html('<div class="gb-kicker">02 · EDA</div>')
st.title("Check the table before modeling")
frame, target, problem, source, metadata = current_data()
profile = profile_table(frame, target)
st.caption(f"{source} · target `{target}` · {problem}")

tabs = st.tabs(
    [
        "Overview",
        "Missing & quality",
        "Describe",
        "Distributions",
        "Relationships",
        "Pairplot",
        "Split",
    ]
)

with tabs[0]:
    metrics = st.columns(4)
    metrics[0].metric("Rows", f"{profile['rows']:,}")
    metrics[1].metric("Columns", profile["columns"])
    metrics[2].metric("Shape", f"{profile['rows']:,} × {profile['columns']}")
    metrics[3].metric("Deep memory", format_bytes(profile["memory_bytes"]))
    st.subheader("Dtypes and memory")
    st.dataframe(dtype_summary(frame), width="stretch")
    head_tab, tail_tab = st.tabs(["Head · 20 rows", "Tail · 10 rows"])
    head_tab.dataframe(frame.head(20), width="stretch")
    tail_tab.dataframe(frame.tail(10), width="stretch")

with tabs[1]:
    st.plotly_chart(missing_figure(frame), width="stretch")
    notes = st.columns(3)
    with notes[0]:
        st.markdown("#### Constant columns")
        if profile["constant_columns"]:
            st.warning(", ".join(f"`{name}`" for name in profile["constant_columns"]))
        else:
            st.success("None found")
    with notes[1]:
        st.markdown("#### High-cardinality categories")
        if profile["high_cardinality_categoricals"]:
            st.warning(", ".join(f"`{name}`" for name in profile["high_cardinality_categoricals"]))
        else:
            st.success("None found")
    with notes[2]:
        st.markdown("#### Leakage boundary")
        st.info(f"`{target}` is the declared target and must stay out of model features.")
        if profile["leakage_suspects"]:
            st.warning(
                "Matching feature names: "
                + ", ".join(f"`{name}`" for name in profile["leakage_suspects"])
            )

with tabs[2]:
    numeric = frame.select_dtypes("number")
    categorical_columns = frame.select_dtypes(exclude="number").columns.tolist()
    st.subheader("Numeric describe()")
    if numeric.empty:
        st.info("No numeric columns are available.")
    else:
        st.dataframe(numeric.describe().T, width="stretch")
    st.subheader("Categorical value counts · top 15")
    if categorical_columns:
        category = st.selectbox("Categorical column", categorical_columns, key="eda_category")
        counts = frame[category].value_counts(dropna=False).head(15).rename("Rows").to_frame()
        counts["Percent"] = (counts["Rows"] / len(frame) * 100).round(2)
        st.dataframe(counts, width="stretch")
    else:
        st.info("No categorical columns are available.")

with tabs[3]:
    numeric_columns = frame.select_dtypes("number").columns.tolist()
    if numeric_columns:
        control_col, kind_col = st.columns(2)
        selected = control_col.selectbox("Numeric column", numeric_columns, key="eda_dist_col")
        kind = kind_col.selectbox("Plot", ["Histogram", "Box", "Violin"], key="eda_dist_kind")
        st.plotly_chart(distribution_figure(frame, selected, kind), width="stretch")
    else:
        st.info("Histogram, box, and violin plots need a numeric column.")
    st.subheader("Target")
    st.plotly_chart(target_figure(frame, target, problem), width="stretch")

with tabs[4]:
    correlation = correlation_figure(frame)
    if correlation is None:
        st.info("Correlation and scatter plots need at least two numeric columns.")
    else:
        st.plotly_chart(correlation, width="stretch")
        numeric_columns = frame.select_dtypes("number").columns.tolist()
        x_default = 0
        y_default = numeric_columns.index(target) if target in numeric_columns else 1
        x_col, y_col = st.columns(2)
        x = x_col.selectbox("Scatter X", numeric_columns, index=x_default, key="eda_scatter_x")
        y_options = [column for column in numeric_columns if column != x]
        preferred_y = target if target in y_options else numeric_columns[y_default]
        y = y_col.selectbox(
            "Scatter Y",
            y_options,
            index=y_options.index(preferred_y) if preferred_y in y_options else 0,
            key="eda_scatter_y",
        )
        st.plotly_chart(scatter_figure(frame, x, y, target, problem), width="stretch")

with tabs[5]:
    numeric_columns = frame.select_dtypes("number").columns.tolist()
    if len(numeric_columns) < 2:
        st.info("Pairplot needs at least two numeric columns.")
    else:
        defaults = [column for column in numeric_columns if column != target][:4]
        if target in numeric_columns:
            defaults.append(target)
        defaults = defaults[:5]
        selected = st.multiselect(
            "Numeric columns · maximum 5",
            numeric_columns,
            default=defaults,
            max_selections=5,
            key="eda_pair_columns",
        )
        if len(selected) >= 2:
            with st.spinner("Drawing a deterministic sample of up to 400 rows…"):
                figure = pairplot_figure(frame, selected, target, problem)
                st.pyplot(figure, clear_figure=True, width="stretch")
        else:
            st.info("Choose at least two columns.")

with tabs[6]:
    known, hidden = split_data(frame, metadata)
    left, right = st.columns(2)
    left.metric("Known train rows", len(known), "90%")
    right.metric("Hidden test rows", len(hidden), "10% held out")
    left.dataframe(known.head(10), width="stretch")
    right.dataframe(hidden.head(10), width="stretch")
