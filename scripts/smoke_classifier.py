from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CLASSIFIER_ID
from src.load import load_sample
from src.mitra_run import run_mitra

OUTPUT = ROOT / "data" / "cache" / "last_run_clf.json"


def main() -> None:
    machines, metadata = load_sample("Machines")
    target = metadata["target"]
    sample, _ = train_test_split(
        machines,
        train_size=400,
        random_state=42,
        stratify=machines[target],
    )
    train, test = train_test_split(
        sample,
        test_size=0.1,
        random_state=42,
        stratify=sample[target],
    )

    result = run_mitra(
        train.reset_index(drop=True),
        test.reset_index(drop=True),
        target,
        "classification",
        False,
        False,
        fine_tune_steps=50,
        time_limit=180,
    )

    assert result["problem"] == "binary"
    assert result["checkpoint"] == CLASSIFIER_ID
    assert math.isfinite(result["metrics"]["Accuracy"])
    matrix = result["confusion_matrix"]
    assert sum(sum(row) for row in matrix["values"]) == len(test)
    _assert_artifacts(result)
    payload = {
        "run_id": result["run_id"],
        "run_dir": result["run_dir"],
        "checkpoint": result["checkpoint"],
        "problem_type": result["problem"],
        "metrics": {"Accuracy": result["metrics"]["Accuracy"]},
        "confusion_matrix": matrix,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"MITRA classifier smoke passed: {OUTPUT}")


def _assert_artifacts(result: dict) -> None:
    run_dir = Path(result["run_dir"])
    for relative in (
        "flags.json",
        "metrics.json",
        "predictions.csv",
        "leaderboard.csv",
        "run.log",
        "predictor",
    ):
        assert (run_dir / relative).exists(), f"Missing run artifact: {relative}"


if __name__ == "__main__":
    main()
