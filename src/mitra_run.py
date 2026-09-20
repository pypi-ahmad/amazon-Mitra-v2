from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.config import CLASSIFIER_ID, REGRESSOR_ID, ROOT
from src.load import split_from_metadata

RUNS_DIR = ROOT / "data" / "runs"
CACHE_DIR = ROOT / "data" / "cache"
MAX_MITRA_TRAIN_ROWS = 10_000
LogCallback = Callable[[str], None]


def load_last_result() -> dict[str, Any] | None:
    """Return the latest complete local run without trusting cached absolute paths."""
    pointer_path = CACHE_DIR / "last_run.json"
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        run_id = str(pointer["run_id"])
        run_dir = RUNS_DIR / run_id
        result = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        return None

    if result.get("run_id") != run_id:
        return None
    result["run_dir"] = str(run_dir)
    prediction_path = run_dir / "predictions.parquet"
    if prediction_path.exists():
        result["prediction_path"] = str(prediction_path)
    return result


@dataclass(frozen=True, slots=True)
class RunConfig:
    fine_tune: bool = True
    fine_tune_steps: int = 50
    eight_copies: bool = True
    time_limit: int = 600

    def __post_init__(self) -> None:
        if not 0 <= self.fine_tune_steps <= 100:
            raise ValueError("fine_tune_steps must be between 0 and 100")
        if self.time_limit < 1:
            raise ValueError("time_limit must be positive")


def device_info() -> tuple[bool, str]:
    try:
        import torch

        cuda = torch.cuda.is_available()
        return cuda, torch.cuda.get_device_name(0) if cuda else "CPU"
    except ImportError:
        return False, "CPU"


def run_mitra(
    frame: pd.DataFrame,
    target: str,
    problem: str,
    metadata: dict[str, Any],
    config: RunConfig,
    log_callback: LogCallback | None = None,
) -> dict[str, Any]:
    from autogluon.tabular import TabularPredictor

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = RUNS_DIR / run_id
    predictor_path = run_dir / "predictor"
    run_dir.mkdir(parents=True, exist_ok=False)
    log_lines: list[str] = []

    def log(message: str) -> None:
        stamped = f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {message}"
        log_lines.append(stamped)
        (run_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        if log_callback:
            log_callback(stamped)

    cuda, device = device_info()
    checkpoint = REGRESSOR_ID if problem == "regression" else CLASSIFIER_ID
    settings = {
        **asdict(config),
        "run_id": run_id,
        "dataset_id": metadata.get("id"),
        "target": target,
        "problem": problem,
        "checkpoint": checkpoint,
        "device": device,
        "num_bag_folds": 8 if config.eight_copies else 0,
    }
    (run_dir / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    started = time.perf_counter()
    try:
        log("Preparing the known training rows and hidden test rows.")
        train, hidden = split_from_metadata(frame, metadata)
        train = train.dropna(subset=[target]).copy()
        hidden = hidden.dropna(subset=[target]).copy()
        if train.empty or hidden.empty:
            raise ValueError("Both known training rows and hidden test rows are required")
        if len(train) > MAX_MITRA_TRAIN_ROWS:
            train = train.sample(MAX_MITRA_TRAIN_ROWS, random_state=42)
            log(f"Capped training data at MITRA's installed {MAX_MITRA_TRAIN_ROWS:,}-row limit.")
        train = _to_autogluon_dtypes(train)
        hidden = _to_autogluon_dtypes(hidden)

        x_hidden = hidden.drop(columns=[target])
        y_hidden = hidden[target]
        if problem == "regression":
            _install_official_regression_patch(checkpoint)
        log(
            f"Configuring MITRA on {device}: fine_tune={config.fine_tune}, "
            f"steps={config.fine_tune_steps}, folds={settings['num_bag_folds']}."
        )
        predictor = TabularPredictor(
            label=target,
            problem_type=problem,
            path=str(predictor_path),
            verbosity=2,
        )
        predictor.fit(
            train_data=train,
            time_limit=config.time_limit,
            hyperparameters={
                "MITRA": {
                    "hf_model": checkpoint,
                    "fine_tune": config.fine_tune,
                    "fine_tune_steps": config.fine_tune_steps,
                }
            },
            num_bag_folds=settings["num_bag_folds"],
            num_bag_sets=1,
            num_stack_levels=0,
            dynamic_stacking=False,
            fit_weighted_ensemble=False,
            num_gpus=1 if cuda else 0,
            ag_args_fit={"ag.max_memory_usage_ratio": 1.2},
        )

        log("Predicting the hidden test rows with MITRA.")
        predictions = predictor.predict(x_hidden)
        probabilities = predictor.predict_proba(x_hidden) if problem != "regression" else None
        metrics, matrix = calculate_metrics(
            y_hidden,
            predictions,
            problem,
            probabilities=probabilities,
            positive_class=predictor.positive_class if problem == "binary" else None,
        )

        log("Fitting one sklearn HistGradientBoosting baseline on the same rows and features.")
        baseline_predictions, baseline_probabilities, baseline_positive = run_baseline(
            train, hidden, target, problem
        )
        baseline_metrics, baseline_matrix = calculate_metrics(
            y_hidden,
            baseline_predictions,
            problem,
            probabilities=baseline_probabilities,
            positive_class=baseline_positive,
        )

        prediction_table = hidden[[target]].rename(columns={target: "actual"})
        prediction_table["mitra_prediction"] = predictions.to_numpy()
        prediction_table["baseline_prediction"] = np.asarray(baseline_predictions)
        if problem == "regression":
            prediction_table["mitra_residual"] = (
                prediction_table["actual"] - prediction_table["mitra_prediction"]
            )
        prediction_path = run_dir / "predictions.parquet"
        prediction_table.to_parquet(prediction_path, index=True)

        runtime = time.perf_counter() - started
        result = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "prediction_path": str(prediction_path),
            "metrics": metrics,
            "baseline_metrics": baseline_metrics,
            "confusion_matrix": matrix,
            "baseline_confusion_matrix": baseline_matrix,
            "runtime_seconds": runtime,
            "device": device,
            "checkpoint": checkpoint,
            "mode": "fine-tuned" if config.fine_tune and config.fine_tune_steps > 0 else "zero-shot",
            "copies": 8 if config.eight_copies else 1,
            "fine_tune_steps": config.fine_tune_steps,
            "time_limit": config.time_limit,
            "train_rows": len(train),
            "hidden_rows": len(hidden),
            "problem": problem,
            "target": target,
        }
        (run_dir / "metrics.json").write_text(
            json.dumps(_jsonable(result), indent=2), encoding="utf-8"
        )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / "last_run.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "metrics": metrics,
                    "baseline_metrics": baseline_metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        log(f"Run complete in {runtime:.1f} seconds.")
        return _jsonable(result)
    except Exception as exc:
        runtime = time.perf_counter() - started
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "runtime_seconds": runtime,
        }
        (run_dir / "error.json").write_text(json.dumps(error, indent=2), encoding="utf-8")
        log(f"Run failed: {type(exc).__name__}: {exc}")
        raise


def run_baseline(
    train: pd.DataFrame,
    hidden: pd.DataFrame,
    target: str,
    problem: str,
) -> tuple[np.ndarray, np.ndarray | None, Any | None]:
    x_train = train.drop(columns=[target])
    y_train = train[target]
    x_hidden = hidden.drop(columns=[target])
    numeric = x_train.select_dtypes("number").columns.tolist()
    categorical = [column for column in x_train.columns if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                    ]
                ),
                categorical,
            )
        )
    estimator = (
        HistGradientBoostingRegressor(random_state=42)
        if problem == "regression"
        else HistGradientBoostingClassifier(random_state=42)
    )
    model = Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers, remainder="drop")),
            ("model", estimator),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_hidden)
    if problem == "regression":
        return predictions, None, None
    probabilities = model.predict_proba(x_hidden)
    classes = model.named_steps["model"].classes_
    positive = classes[1] if len(classes) == 2 else None
    return predictions, probabilities, positive


def calculate_metrics(
    actual: pd.Series,
    predicted: pd.Series | np.ndarray,
    problem: str,
    *,
    probabilities: pd.DataFrame | pd.Series | np.ndarray | None = None,
    positive_class: Any | None = None,
) -> tuple[dict[str, float], dict[str, Any] | None]:
    y_true = np.asarray(actual)
    y_pred = np.asarray(predicted)
    if problem == "regression":
        metrics = {
            "RMSE": float(root_mean_squared_error(y_true, y_pred)),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)),
        }
        if np.all(y_true > 0):
            metrics["MAPE (%)"] = float(mean_absolute_percentage_error(y_true, y_pred) * 100)
        return metrics, None

    labels = list(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)], ignore_index=True)))
    binary = problem == "binary" and len(labels) == 2
    if binary:
        positive = positive_class if positive_class is not None else labels[1]
        f1 = f1_score(y_true, y_pred, pos_label=positive)
    else:
        positive = None
        f1 = f1_score(y_true, y_pred, average="weighted")
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1),
    }
    if binary and probabilities is not None:
        scores = _positive_scores(probabilities, positive)
        binary_actual = (y_true == positive).astype(int)
        metrics["ROC-AUC"] = float(roc_auc_score(binary_actual, scores))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return metrics, {"labels": [_jsonable(value) for value in labels], "values": matrix.tolist()}


def _positive_scores(probabilities: Any, positive_class: Any) -> np.ndarray:
    if isinstance(probabilities, pd.DataFrame):
        return probabilities[positive_class].to_numpy()
    if isinstance(probabilities, pd.Series):
        return probabilities.to_numpy()
    values = np.asarray(probabilities)
    return values[:, 1] if values.ndim == 2 else values


def _to_autogluon_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    converted = frame.copy()
    for column in converted.columns:
        series = converted[column]
        if pd.api.types.is_float_dtype(series.dtype):
            converted[column] = pd.to_numeric(series, errors="coerce").astype("float64")
        elif pd.api.types.is_integer_dtype(series.dtype):
            numeric = pd.to_numeric(series, errors="coerce")
            converted[column] = numeric.astype("float64" if numeric.isna().any() else "int64")
        elif pd.api.types.is_bool_dtype(series.dtype):
            converted[column] = series.astype("object" if series.isna().any() else "bool")
        elif isinstance(series.dtype, pd.StringDtype):
            converted[column] = series.astype("object")
    return converted


def _install_official_regression_patch(checkpoint: str) -> None:
    from huggingface_hub import hf_hub_download
    from src.mitra_regression_patch import install_regression_patch

    config_path = hf_hub_download(
        repo_id=checkpoint,
        filename="config.json",
        token=os.environ["HF_TOKEN"],
    )
    checkpoint_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    install_regression_patch(int(checkpoint_config["dim_output"]))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
