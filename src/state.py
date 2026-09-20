"""Own Streamlit session state for selected samples, uploads, and model settings."""

import pandas as pd
import streamlit as st

from src.load import build_upload_metadata, load_sample, read_upload


def initialize_state() -> None:
    """Populate missing Streamlit session-state defaults without overwriting choices."""
    defaults = {
        "selected_dataset": "Houses",
        "uploaded_data": None,
        "uploaded_name": None,
        "fine_tune": True,
        "eight_copies": True,
        "result": None,
        "run_error": None,
        "data_error": None,
        "fine_tune_steps": 50,
        "time_limit": 600,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def set_upload(name: str, content: bytes) -> None:
    """Parse an upload and make it the current table for this browser session.

    Args:
        name: Original upload filename.
        content: Complete uploaded file content.

    Raises:
        ValueError: If the uploaded CSV or Parquet table is not usable.
    """
    frame = read_upload(name, content)
    st.session_state.uploaded_data = frame
    st.session_state.uploaded_name = name
    st.session_state.target = frame.columns[-1]
    st.session_state.problem_choice = "Auto"


def clear_upload() -> None:
    """Discard the in-session upload and return to the selected bundled sample."""
    st.session_state.uploaded_data = None
    st.session_state.uploaded_name = None
    st.session_state.pop("target", None)
    st.session_state.pop("problem_choice", None)


def current_data() -> tuple[pd.DataFrame, str, str, str, dict]:
    """Resolve the current table, target, problem type, source label, and metadata.

    Returns:
        A ``(frame, target, problem, source, metadata)`` tuple for page rendering or training.

    Raises:
        ValueError: If an uploaded target choice is invalid or cannot be split.
        RuntimeError: If the selected catalog sample and all of its sources fail.
    """
    if st.session_state.uploaded_data is not None:
        frame = st.session_state.uploaded_data
        target = st.session_state.get("target", frame.columns[-1])
        choice = st.session_state.get("problem_choice", "Auto")
        problem = infer_problem(frame[target]) if choice == "Auto" else choice.lower()
        metadata = build_upload_metadata(frame, st.session_state.uploaded_name, target, problem)
        return frame, target, problem, st.session_state.uploaded_name, metadata
    name = st.session_state.selected_dataset
    frame, metadata = load_sample(name)
    return frame, metadata["target"], metadata["task"], f"Bundled · {name}", metadata


def infer_problem(series: pd.Series) -> str:
    """Infer binary, multiclass, or regression from dtype and target cardinality.

    Args:
        series: Candidate target column.

    Returns:
        ``binary`` or ``multiclass`` for categorical or low-cardinality targets; otherwise
        ``regression``.
    """
    unique = series.nunique(dropna=True)
    if not pd.api.types.is_numeric_dtype(series) or unique <= 20:
        return "binary" if unique == 2 else "multiclass"
    return "regression"
