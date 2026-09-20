import streamlit as st
import pandas as pd

from src.mitra_run import load_last_result

st.html('<div class="gb-kicker">03 · Workflow</div>')
st.title("Old way vs Mitra-v2")
traditional = [
    "Audit schema",
    "Clean values",
    "Encode categories",
    "Impute gaps",
    "Engineer features",
    "Tune models",
    "Build ensemble",
    "Validate",
]
mitra = [
    "Practised model (synthetic priors)",
    "Read table",
    "Fine-tune",
    "Eight copies vote",
    "Answer with spread",
]
result = st.session_state.result or load_last_result()

left, right = st.columns(2)
with left:
    st.html('<div class="gb-board-title"><h3>Traditional · 8 stages</h3><span class="gb-runtime">a person\'s week</span></div>')
    st.caption("Pedagogical runtime label · not a measured claim")
    for number, name in enumerate(traditional, 1):
        st.html(
            f'<div class="gb-flow"><span class="gb-flow-number">{number:02}</span><strong>{name}</strong></div>'
        )
with right:
    runtime = f"{result['runtime_seconds']:.1f}s measured" if result else "run to measure"
    st.html(f'<div class="gb-board-title"><h3>Mitra-v2 · 5 stages</h3><span class="gb-runtime">{runtime}</span></div>')
    st.caption("Measured on the latest local run" if result else "No completed local run found")
    for number, name in enumerate(mitra, 1):
        st.html(
            f'<div class="gb-flow"><span class="gb-flow-number">{number:02}</span><strong>{name}</strong></div>'
        )
    st.info(
        "Mitra-v2 shortens model preparation. Data quality, leakage checks, domain review, and validation remain required."
    )

st.subheader("Measured on the same hidden rows")
st.caption(
    "The traditional reference is one sklearn HistGradientBoosting pipeline with basic imputation "
    "and category encoding. It is not a claim about a week-long AutoML project."
)
if result:
    names = list(dict.fromkeys([*result["metrics"], *result["baseline_metrics"]]))
    table = pd.DataFrame(
        {
            "MITRA": {name: result["metrics"].get(name) for name in names},
            "HistGradientBoosting": {
                name: result["baseline_metrics"].get(name) for name in names
            },
        }
    )
    st.dataframe(table.style.format("{:.4f}", na_rep="—"), width="stretch")
else:
    st.info("Run Train & Predict to populate a real same-split comparison.")
