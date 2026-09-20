"""Render a downloadable standalone Python reproduction script for the current table."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.config import CLASSIFIER_ID, REGRESSOR_ID
from src.state import current_data

st.html('<div class="gb-kicker">07 · Reproduce</div>')
st.title("Python Script (CLI)")
_, target, problem, _, metadata = current_data()
checkpoint = REGRESSOR_ID if problem == "regression" else CLASSIFIER_ID

if metadata["id"] == "upload":
    dataset_path = Path(st.session_state.uploaded_name or "table.csv")
    path_note = "Save this script beside your uploaded data file, or edit DATA_PATH."
else:
    dataset_path = Path("data") / "hf" / metadata["id"] / "table.parquet"
    path_note = "This path points to the bundled dataset cached by Glass Box."

loader = (
    "pd.read_csv(DATA_PATH)"
    if dataset_path.suffix.lower() == ".csv"
    else "pd.read_parquet(DATA_PATH)"
)
regression_patch = ""
if problem == "regression":
    regression_patch = f"""from huggingface_hub import hf_hub_download
from mitra_finetune.patches import install_reg_ce_patches

# mitra-regressor-2 uses the official 1,000-bin distributional regression head.
checkpoint_config = json.loads(
    Path(hf_hub_download({checkpoint!r}, "config.json", token=_hf_token)).read_text(
        encoding="utf-8"
    )
)
install_reg_ce_patches(int(checkpoint_config["dim_output"]))

"""

stratify = "frame[TARGET]" if problem in {"binary", "multiclass"} else "None"
mitra_hyperparameters = {"hf_model": checkpoint, "fine_tune": st.session_state.fine_tune}
if st.session_state.fine_tune:
    mitra_hyperparameters["fine_tune_steps"] = st.session_state.fine_tune_steps
code = f"""import json
import os
from pathlib import Path

import pandas as pd
import torch
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split

# Fail clearly if the Hugging Face token is missing; never print its value.
_hf_token = os.environ["HF_TOKEN"]
{regression_patch}
DATA_PATH = Path({str(dataset_path)!r})
TARGET = {target!r}

frame = {loader}
train, hidden = train_test_split(
    frame,
    test_size={float(metadata.get("test_size", 0.1))!r},
    random_state={int(metadata.get("random_state", 42))},
    stratify={stratify},
)

predictor = TabularPredictor(
    label=TARGET,
    problem_type={problem!r},
    path="data/runs/my-run/predictor",
)
predictor.fit(
    train_data=train,
    time_limit={int(st.session_state.time_limit)},
    hyperparameters={{"MITRA": {mitra_hyperparameters!r}}},
    num_bag_folds={8 if st.session_state.eight_copies else 0},
    num_bag_sets=1,
    num_stack_levels=0,
    dynamic_stacking=False,
    fit_weighted_ensemble=False,
    num_gpus=1 if torch.cuda.is_available() else 0,
    ag_args_fit={{"ag.max_memory_usage_ratio": 1.2}},
)
predictions = predictor.predict(hidden.drop(columns=[TARGET]))
print(predictions.head())
"""

st.code(code, language="python")
st.download_button(
    "Download Python script",
    code.encode("utf-8"),
    file_name="glass-box-mitra-run.py",
    mime="text/x-python",
    icon=":material/download:",
    width="stretch",
)
st.caption(path_note)
st.caption(
    "The script reads HF_TOKEN from the Windows process environment and does not display or write its value."
)
