"""Import the app and verify its deterministic synthetic-gallery inputs."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

importlib.import_module("app")

from src.synthetic_gallery import synthetic_prior_shapes

shapes = synthetic_prior_shapes()
expected = {"Clusters", "Hybrid SCM", "Two moons", "Rings", "Spirals"}
assert set(shapes) == expected
assert all(not frame.empty and {"x", "y", "pattern"} <= set(frame) for frame in shapes.values())

cache_path = ROOT / "data" / "cache" / "ui_import_ok.json"
cache_path.parent.mkdir(parents=True, exist_ok=True)
cache_path.write_text(
    json.dumps({"app_imported": True, "synthetic_plots": len(shapes)}, indent=2),
    encoding="utf-8",
)
print(f"UI import smoke passed: {cache_path}")
