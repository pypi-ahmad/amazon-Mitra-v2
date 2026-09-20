from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.mitra_run import load_last_result

st.html('<div class="gb-kicker">06 · Evaluate</div>')
st.title("Results")
st.html(
    '<div class="gb-warning"><b>Trust boundary.</b> The model does not certify that the table is accurate, representative, fair, causally valid, or trustworthy.</div>'
)

if st.session_state.run_error:
    st.error(st.session_state.run_error)

result = st.session_state.result or load_last_result()
if result:
    st.caption(
        f"Run `{result['run_id']}` · {result['mode']} · {result['copies']} "
        f"{'copy' if result['copies'] == 1 else 'copies'} · {result['runtime_seconds']:.1f}s · {result['device']}"
    )

    with st.container(horizontal=True):
        for name, value in result["metrics"].items():
            shown = "—" if value is None else f"{value:.4f}"
            st.metric(name, shown, border=True)

    metric_names = list(dict.fromkeys([*result["metrics"], *result["baseline_metrics"]]))
    comparison = pd.DataFrame(
        {
            "MITRA": {name: result["metrics"].get(name) for name in metric_names},
            "HistGradientBoosting": {
                name: result["baseline_metrics"].get(name) for name in metric_names
            },
        }
    )
    st.subheader("Same-split comparison")
    st.caption("The reference is one sklearn baseline on the same known and hidden rows.")
    st.dataframe(comparison.style.format("{:.4f}", na_rep="—"), width="stretch")

    prediction_path = Path(result["prediction_path"])
    predictions = pd.read_parquet(prediction_path) if prediction_path.exists() else None

    if predictions is not None and result["problem"] == "regression":
        st.subheader("Prediction versus actual")
        chart_data = predictions.rename(
            columns={"actual": "Actual", "mitra_prediction": "Prediction"}
        )
        figure = px.scatter(
            chart_data,
            x="Actual",
            y="Prediction",
            opacity=0.62,
            color_discrete_sequence=["#9b87f5"],
        )
        lower = float(min(chart_data["Actual"].min(), chart_data["Prediction"].min()))
        upper = float(max(chart_data["Actual"].max(), chart_data["Prediction"].max()))
        figure.add_shape(
            type="line",
            x0=lower,
            y0=lower,
            x1=upper,
            y1=upper,
            line={"color": "#46c2b3", "dash": "dash"},
        )
        figure.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0b1220",
            font={"color": "#dbe2ff"},
        )
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    if matrix := result.get("confusion_matrix"):
        st.subheader("MITRA confusion matrix")
        labels = [str(label) for label in matrix["labels"]]
        figure = go.Figure(
            data=go.Heatmap(
                z=matrix["values"],
                x=[f"Predicted · {label}" for label in labels],
                y=[f"Actual · {label}" for label in labels],
                colorscale=[[0, "#111a2e"], [1, "#8064f4"]],
                text=matrix["values"],
                texttemplate="%{text}",
                showscale=False,
            )
        )
        figure.update_layout(
            height=380,
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#dbe2ff"},
        )
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    if predictions is not None:
        st.subheader("Per-row predictions")
        rows = pd.DataFrame(
            {"y_true": predictions["actual"], "y_pred": predictions["mitra_prediction"]}
        )
        if result["problem"] == "regression":
            rows["error"] = rows["y_true"] - rows["y_pred"]
        else:
            rows["error"] = rows["y_true"] != rows["y_pred"]

        spread_column = next(
            (name for name in ("spread", "prediction_spread", "variance") if name in predictions),
            None,
        )
        if spread_column:
            rows["spread"] = predictions[spread_column]
        else:
            st.caption(
                "Prediction spread is unavailable for this run because the current AutoGluon "
                "predictor output did not expose per-copy or per-fold variance."
            )
        st.dataframe(rows, hide_index=True, width="stretch")

        st.download_button(
            "Download predictions",
            prediction_path.read_bytes(),
            file_name=f"{result['run_id']}-predictions.parquet",
            mime="application/octet-stream",
            icon=":material/download:",
            width="stretch",
        )
    else:
        st.warning("The run metadata is available, but its prediction artifact is missing.")

    details = {
        key: result[key]
        for key in [
            "checkpoint", "target", "problem", "train_rows", "hidden_rows",
            "fine_tune_steps", "time_limit", "run_dir",
        ]
    }
    with st.expander("Run details"):
        st.json(details)
else:
    st.info("No result yet. Open Train & Predict and start an explicit run.")
    if st.button("Open Train & Predict", icon=":material/model_training:"):
        st.switch_page("app_pages/train.py")
