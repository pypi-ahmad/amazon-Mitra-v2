"""Run released Mitra-v2 heads and a sklearn reference baseline on the same split."""

from __future__ import annotations

import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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

RUNS_DIR = ROOT / "data" / "runs"
CACHE_DIR = ROOT / "data" / "cache"
MAX_MITRA_TRAIN_ROWS = 10_000


RESULT_POINTERS = {
    "last_run.json",
    "last_run_reg.json",
    "last_run_clf.json",
    "last_run_reg_ft.json",
    "last_run_clf_ft.json",
}


def load_cached_result(pointer_name: str) -> dict[str, Any] | None:
    """Load one complete run through an approved generated cache pointer.

    Args:
        pointer_name: One filename from :data:`RESULT_POINTERS`.

    Returns:
        Complete run metadata plus local artifact paths, or ``None`` when the pointer or
        referenced artifact is absent, malformed, or inconsistent.

    Raises:
        ValueError: If ``pointer_name`` is not an approved result pointer.
    """
    if pointer_name not in RESULT_POINTERS:
        raise ValueError(f"Unsupported result pointer: {pointer_name}")
    pointer_path = CACHE_DIR / pointer_name
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
    prediction_path = run_dir / "predictions.csv"
    legacy_prediction_path = run_dir / "predictions.parquet"
    if prediction_path.exists():
        result["prediction_path"] = str(prediction_path)
    elif legacy_prediction_path.exists():
        result["prediction_path"] = str(legacy_prediction_path)
    return result


def load_last_result() -> dict[str, Any] | None:
    """Return metadata for the latest complete local MITRA run, if one exists.

    Returns:
        The result addressed by ``last_run.json``, or ``None`` when no valid run is cached.
    """
    return load_cached_result("last_run.json")


def device_info() -> tuple[bool, str]:
    """Detect whether PyTorch can use CUDA and return a display-ready device name.

    Returns:
        A ``(cuda_available, device_name)`` pair. Missing PyTorch is treated as CPU-only.
    """
    try:
        import torch

        cuda = torch.cuda.is_available()
        return cuda, torch.cuda.get_device_name(0) if cuda else "CPU"
    except ImportError:
        return False, "CPU"


def run_mitra(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target: str,
    problem_type: str,
    fine_tune: bool,
    eight_copies: bool,
    fine_tune_steps: int = 50,
    time_limit: int | None = None,
) -> dict[str, Any]:
    """Fit the correct released Mitra-v2 head and evaluate an explicit test table.

    Args:
        df_train: Known rows containing features and ``target``.
        df_test: Hidden rows with the same ordered feature columns and ``target`` retained
            only for local evaluation.
        target: Column to predict.
        problem_type: ``regression``, ``binary``, ``multiclass``, or generic
            ``classification``.
        fine_tune: Whether MITRA receives the official fine-tuning setting.
        eight_copies: Whether AutoGluon creates eight bagging folds instead of one model.
        fine_tune_steps: Fine-tuning step count from 0 through 100 when enabled.
        time_limit: Optional positive AutoGluon fitting limit in seconds.

    Returns:
        Serializable run metadata, current-split metrics, baseline metrics, model details,
        runtime, and paths below ``data/runs/<UTC timestamp>/``.

    Raises:
        ValueError: If the tables, target, problem type, fine-tune steps, or time limit are
            incompatible with the installed MITRA runner.
        RuntimeError: If ``HF_TOKEN`` is missing or the selected v2 checkpoint cannot be
            authenticated and downloaded.

    Side Effects:
        Writes flags, logs, metrics, predictions, leaderboard, a persisted AutoGluon
        predictor, and a ``last_run.json`` cache pointer. Failed runs retain ``error.json``.
    """
    from autogluon.tabular import TabularPredictor

    if target not in df_train or target not in df_test:
        raise ValueError(f"Target column is missing: {target}")
    problem = _normalize_problem_type(problem_type, df_train[target])
    _validate_run_inputs(df_train, df_test, target, problem, fine_tune_steps, time_limit)
    checkpoint = REGRESSOR_ID if problem == "regression" else CLASSIFIER_ID
    checkpoint_config = _preflight_checkpoint(checkpoint)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = RUNS_DIR / run_id
    predictor_path = run_dir / "predictor"
    run_dir.mkdir(parents=True, exist_ok=False)
    log_lines: list[str] = []

    def log(message: str) -> None:
        stamped = f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {message}"
        log_lines.append(stamped)
        (run_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    cuda, device = device_info()
    settings = {
        "run_id": run_id,
        "target": target,
        "problem_type": problem,
        "checkpoint": checkpoint,
        "device": device,
        "fine_tune": bool(fine_tune),
        "fine_tune_steps": fine_tune_steps if fine_tune else 0,
        "eight_copies": bool(eight_copies),
        "num_bag_folds": 8 if eight_copies else 0,
        "time_limit": time_limit,
    }
    (run_dir / "flags.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    started = time.perf_counter()
    try:
        log("Preparing the training and test rows.")
        train = df_train.dropna(subset=[target]).copy()
        hidden = df_test.dropna(subset=[target]).copy()
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
            _install_official_regression_patch(checkpoint_config)
        log(f"Using Hugging Face repository: {checkpoint}.")
        log(
            f"Configuring MITRA on {device}: fine_tune={fine_tune}, "
            f"steps={settings['fine_tune_steps']}, folds={settings['num_bag_folds']}."
        )
        predictor = TabularPredictor(
            label=target,
            problem_type=problem,
            path=str(predictor_path),
            verbosity=2,
        )
        mitra_hyperparameters: dict[str, Any] = {
            "hf_model": checkpoint,
            "fine_tune": fine_tune,
        }
        if fine_tune:
            mitra_hyperparameters["fine_tune_steps"] = fine_tune_steps
        predictor.fit(
            train_data=train,
            time_limit=time_limit,
            hyperparameters={"MITRA": mitra_hyperparameters},
            num_bag_folds=settings["num_bag_folds"],
            num_bag_sets=1,
            num_stack_levels=0,
            dynamic_stacking=False,
            fit_weighted_ensemble=False,
            num_gpus=1 if cuda else 0,
            ag_args_fit={"ag.max_memory_usage_ratio": 1.2},
        )
        model_names = predictor.model_names()
        log(
            f"Predictor class: {predictor.__class__.__module__}.{predictor.__class__.__qualname__}."
        )
        log(f"Fitted AutoGluon model(s): {', '.join(model_names)}.")
        settings["predictor_class"] = (
            f"{predictor.__class__.__module__}.{predictor.__class__.__qualname__}"
        )
        settings["model_names"] = model_names
        (run_dir / "flags.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

        leaderboard_path = run_dir / "leaderboard.csv"
        predictor.leaderboard(display=False).to_csv(leaderboard_path, index=False)

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
        prediction_path = run_dir / "predictions.csv"
        prediction_table.to_csv(prediction_path, index=True, index_label="row_index")

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
            "hf_repo": checkpoint,
            "predictor_class": f"{predictor.__class__.__module__}.{predictor.__class__.__qualname__}",
            "model_names": model_names,
            "leaderboard_path": str(leaderboard_path),
            "mode": "fine-tuned" if fine_tune and fine_tune_steps > 0 else "zero-shot",
            "copies": 8 if eight_copies else 1,
            "fine_tune_steps": settings["fine_tune_steps"],
            "time_limit": time_limit,
            "train_rows": len(train),
            "hidden_rows": len(hidden),
            "problem": problem,
            "target": target,
        }
        log(f"Run complete in {runtime:.1f} seconds.")
        result["run_log"] = (run_dir / "run.log").read_text(encoding="utf-8")
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
    """Fit the same-split HistGradientBoosting reference baseline.

    Args:
        train: Known rows containing features and target.
        hidden: Held-out rows with matching columns.
        target: Column to predict.
        problem: Normalized AutoGluon task type.

    Returns:
        Predictions, optional class probabilities, and the binary positive class. Regression
        returns ``None`` for the last two values.
    """
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
    """Calculate task-appropriate metrics and an optional classification matrix.

    Args:
        actual: Ground-truth target values.
        predicted: Predicted labels or numeric values.
        problem: Normalized problem type.
        probabilities: Optional class probabilities used for binary ROC-AUC.
        positive_class: Positive label corresponding to ``probabilities`` for binary tasks.

    Returns:
        A metric dictionary and a labeled confusion-matrix payload for classification.
        Regression returns ``None`` for the matrix.
    """
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


def _preflight_checkpoint(checkpoint: str) -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("set HF_TOKEN")

    from huggingface_hub import hf_hub_download

    try:
        config_path = hf_hub_download(
            repo_id=checkpoint,
            filename="config.json",
            token=token,
        )
    except Exception as exc:
        raise RuntimeError("set HF_TOKEN") from exc
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def _install_official_regression_patch(checkpoint_config: dict[str, Any]) -> None:
    # The released regressor's 1,000-bin output head needs this companion-package patch
    # before AutoGluon constructs the model. The model card documents this requirement.
    from mitra_finetune.patches import install_reg_ce_patches

    install_reg_ce_patches(int(checkpoint_config["dim_output"]))


def _normalize_problem_type(problem_type: str, target: pd.Series) -> str:
    normalized = problem_type.strip().lower()
    if normalized == "classification":
        return "binary" if target.nunique(dropna=True) == 2 else "multiclass"
    return normalized


def _validate_run_inputs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    problem: str,
    fine_tune_steps: int,
    time_limit: int | None,
) -> None:
    if train.empty or test.empty:
        raise ValueError("Both training and test tables must contain rows")
    if target not in train or target not in test:
        raise ValueError(f"Target column is missing: {target}")
    train_features = list(train.drop(columns=[target]).columns)
    test_features = list(test.drop(columns=[target]).columns)
    if train_features != test_features:
        raise ValueError("Training and test feature columns must match in the same order")
    if not 0 <= fine_tune_steps <= 100:
        raise ValueError("fine_tune_steps must be between 0 and 100")
    if time_limit is not None and time_limit < 1:
        raise ValueError("time_limit must be positive or None")
    _validate_problem_target(train[target], problem)


def _validate_problem_target(target: pd.Series, problem: str) -> None:
    unique = target.nunique(dropna=True)
    if problem == "regression":
        if not pd.api.types.is_numeric_dtype(target):
            raise ValueError("Regression requires a numeric target")
        return
    if problem == "binary":
        if unique != 2:
            raise ValueError(
                f"Binary classification requires exactly 2 target classes; found {unique}"
            )
        return
    if problem == "multiclass":
        if not 3 <= unique <= 20:
            raise ValueError(
                "Multiclass classification requires between 3 and 20 target classes; "
                f"found {unique}"
            )
        return
    raise ValueError(f"Unsupported problem type: {problem!r}")


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
