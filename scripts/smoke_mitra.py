from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.load import build_upload_metadata, load_sample
from src.mitra_run import RunConfig, run_mitra

OUTPUT = ROOT / "data" / "cache" / "last_run.json"


def main() -> None:
    houses, source_metadata = load_sample("Houses")
    sample = houses.sample(n=400, random_state=42).reset_index(drop=True)
    target = source_metadata["target"]
    metadata = build_upload_metadata(sample, "houses-smoke", target, "regression")
    result = run_mitra(
        sample,
        target,
        "regression",
        metadata,
        RunConfig(
            fine_tune=False,
            fine_tune_steps=50,
            eight_copies=False,
            time_limit=180,
        ),
        log_callback=print,
    )

    assert result["mode"] == "zero-shot"
    assert result["copies"] == 1
    assert result["train_rows"] == 360
    assert result["hidden_rows"] == 40
    assert all(value is not None and math.isfinite(value) for value in result["metrics"].values())
    assert all(
        value is not None and math.isfinite(value)
        for value in result["baseline_metrics"].values()
    )
    run_dir = Path(result["run_dir"])
    for relative in ("config.json", "metrics.json", "predictions.parquet", "run.log"):
        assert (run_dir / relative).exists(), f"Missing run artifact: {relative}"
    assert OUTPUT.exists(), f"Missing cache file: {OUTPUT}"
    payload = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert payload["metrics"] == result["metrics"]
    print(f"mitra smoke ok: {result['runtime_seconds']:.1f}s")
    print(OUTPUT)


if __name__ == "__main__":
    main()
