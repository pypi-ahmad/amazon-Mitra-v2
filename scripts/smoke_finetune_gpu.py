"""Run CUDA-gated fine-tuning and eight-copy MITRA smoke tests for both heads."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CLASSIFIER_ID, REGRESSOR_ID
from src.load import load_sample
from src.mitra_run import run_mitra

CACHE_DIR = ROOT / "data" / "cache"
GPU_STATUS = CACHE_DIR / "gpu_status.json"
TIME_LIMIT = 1800
SAMPLE_ROWS = 400


def main() -> None:
    """Skip on CPU or record successful fine-tuned regression and classification runs."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        GPU_STATUS.write_text(json.dumps({"cuda": False}), encoding="utf-8")
        print(f"CUDA unavailable; fine-tune smokes skipped: {GPU_STATUS}")
        return

    GPU_STATUS.write_text(
        json.dumps({"cuda": True, "device": torch.cuda.get_device_name(0)}, indent=2),
        encoding="utf-8",
    )
    _run_regression()
    _run_classification()


def _run_regression() -> None:
    frame, metadata = load_sample("Houses")
    sample, _ = train_test_split(frame, train_size=SAMPLE_ROWS, random_state=42)
    train, test = train_test_split(sample, test_size=0.1, random_state=42)
    result, wall_clock = _run(train, test, metadata["target"], "regression")
    assert result["checkpoint"] == REGRESSOR_ID
    assert math.isfinite(result["metrics"]["RMSE"])
    assert math.isfinite(result["metrics"]["R2"])
    _write_pointer("last_run_reg_ft.json", result, wall_clock)


def _run_classification() -> None:
    frame, metadata = load_sample("Machines")
    target = metadata["target"]
    sample, _ = train_test_split(
        frame,
        train_size=SAMPLE_ROWS,
        random_state=42,
        stratify=frame[target],
    )
    train, test = train_test_split(
        sample,
        test_size=0.1,
        random_state=42,
        stratify=sample[target],
    )
    result, wall_clock = _run(train, test, target, "classification")
    assert result["checkpoint"] == CLASSIFIER_ID
    assert math.isfinite(result["metrics"]["Accuracy"])
    matrix = result["confusion_matrix"]
    assert sum(sum(row) for row in matrix["values"]) == len(test)
    _write_pointer("last_run_clf_ft.json", result, wall_clock)


def _run(train, test, target: str, problem_type: str) -> tuple[dict, float]:
    started = time.perf_counter()
    result = run_mitra(
        train.reset_index(drop=True),
        test.reset_index(drop=True),
        target,
        problem_type,
        True,
        True,
        fine_tune_steps=50,
        time_limit=TIME_LIMIT,
    )
    wall_clock = time.perf_counter() - started
    assert result["mode"] == "fine-tuned"
    assert result["copies"] == 8
    assert result["fine_tune_steps"] == 50
    flags = json.loads((Path(result["run_dir"]) / "flags.json").read_text(encoding="utf-8"))
    assert flags["num_bag_folds"] == 8
    for relative in (
        "flags.json",
        "metrics.json",
        "predictions.csv",
        "leaderboard.csv",
        "run.log",
        "predictor",
    ):
        assert (Path(result["run_dir"]) / relative).exists(), f"Missing run artifact: {relative}"
    return result, wall_clock


def _write_pointer(filename: str, result: dict, wall_clock: float) -> None:
    payload = {
        "run_id": result["run_id"],
        "run_dir": result["run_dir"],
        "checkpoint": result["checkpoint"],
        "problem_type": result["problem"],
        "mode": result["mode"],
        "copies": result["copies"],
        "fine_tune_steps": result["fine_tune_steps"],
        "wall_clock_seconds": wall_clock,
        "metrics": result["metrics"],
        "confusion_matrix": result.get("confusion_matrix"),
    }
    output = CACHE_DIR / filename
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"GPU fine-tune smoke passed in {wall_clock:.1f}s: {output}")


if __name__ == "__main__":
    main()
