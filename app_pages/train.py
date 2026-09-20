from __future__ import annotations

import streamlit as st

from src.config import CLASSIFIER_ID, REGRESSOR_ID
from src.load import split_from_metadata
from src.mitra_run import device_info, run_mitra
from src.state import current_data

st.html('<div class="gb-kicker">05 · Execute</div>')
st.title("Train & Predict")
frame, target, problem, source, metadata = current_data()
cuda, device = device_info()
hf_repo = REGRESSOR_ID if problem == "regression" else CLASSIFIER_ID

last_runtime = None
if st.session_state.result:
    last_runtime = st.session_state.result.get("runtime_seconds")

dataset_col, target_col, runtime_col, device_col = st.columns(4)
dataset_col.metric("Dataset", source)
target_col.metric("Target", target)
runtime_col.metric("Last runtime", f"{last_runtime:.1f}s" if last_runtime else "—")
device_col.metric("Device", device)

if not cuda:
    st.warning(
        "CUDA is unavailable. The installed MITRA documentation says CPU inference is substantially "
        "slower, and CPU fine-tuning can be very slow. Your Fine-tune choice will still be respected."
    )

st.markdown(
    f"**Mode:** {'fine-tune' if st.session_state.fine_tune else 'zero-shot'} · "
    f"**Steps:** {st.session_state.fine_tune_steps} · "
    f"**Copies:** {8 if st.session_state.eight_copies else 1} · "
    f"**Fit limit:** {st.session_state.time_limit}s"
)
st.info(f"Hugging Face repository for this run: `{hf_repo}`")
st.caption(
    "Both classification and regression use AutoGluon TabularPredictor with the installed native MITRA model."
)
st.markdown(
    '<div class="gb-warning"><b>Explicit action.</b> Run can download about 300 MB of model weights. '
    "The sklearn baseline uses the same known/hidden rows and is not a full AutoML benchmark.</div>"
)

st.subheader("Run log")
log_area = st.empty()
if st.session_state.run_error:
    st.error(st.session_state.run_error)

if st.button("Run Mitra-v2", type="primary", width="stretch"):
    st.session_state.run_error = None
    with st.status("Running MITRA and the sklearn baseline…", expanded=True) as status:
        try:
            train, test = split_from_metadata(frame, metadata)
            st.session_state.result = run_mitra(
                train,
                test,
                target,
                problem,
                st.session_state.fine_tune,
                st.session_state.eight_copies,
                fine_tune_steps=st.session_state.fine_tune_steps,
                time_limit=int(st.session_state.time_limit),
            )
            run_log = st.session_state.result.get("run_log", "")
            if run_log:
                log_area.code(run_log, language="text")
            st.success(f"Used `{st.session_state.result['hf_repo']}`")
            status.update(label="Run complete", state="complete")
            st.switch_page("app_pages/results.py")
        except Exception as exc:
            st.session_state.run_error = f"{type(exc).__name__}: {exc}"
            status.update(label="Run failed", state="error")
            st.error(st.session_state.run_error)
