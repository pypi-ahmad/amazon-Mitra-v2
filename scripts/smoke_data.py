from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_catalog import SAMPLE_CATALOG
from src.load import load_sample, split_from_metadata

OUTPUT = ROOT / "data" / "cache" / "catalog.json"


def main() -> None:
    results: dict[str, dict] = {}
    for name, spec in SAMPLE_CATALOG.items():
        try:
            frame, metadata = load_sample(name)
            known, hidden = split_from_metadata(frame, metadata)
            assert spec.target in frame, f"{spec.target} missing"
            assert not frame.empty, "table is empty"
            assert set(known.index).isdisjoint(hidden.index), "known and hidden rows overlap"
            assert len(known) + len(hidden) == frame[spec.target].notna().sum(), (
                "split does not cover every labeled row"
            )
            table_path = ROOT / "data" / "hf" / spec.id / "table.parquet"
            assert table_path.exists(), f"cache missing: {table_path}"
            results[name] = {
                "status": "ok",
                "id": metadata["source_id"],
                "task": metadata["task"],
                "target": metadata["target"],
                "shape": [len(frame), len(frame.columns)],
                "known_rows": len(known),
                "hidden_rows": len(hidden),
                "cache": str(table_path.relative_to(ROOT)),
                "error": None,
            }
        except Exception as exc:
            results[name] = {
                "status": "error",
                "id": spec.source.source_id,
                "task": spec.task,
                "target": spec.target,
                "shape": None,
                "known_rows": None,
                "hidden_rows": None,
                "cache": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "samples": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    assert results["Houses"]["status"] == "ok", results["Houses"]["error"]
    classification_ok = any(
        item["status"] == "ok" and item["task"] in {"binary", "multiclass"}
        for item in results.values()
    )
    assert classification_ok, "No classification dataset loaded"
    print(f"data smoke ok: {sum(item['status'] == 'ok' for item in results.values())}/5")
    print(OUTPUT)


if __name__ == "__main__":
    main()
