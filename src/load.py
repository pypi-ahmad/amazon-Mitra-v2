from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ROOT
from src.data_catalog import SAMPLE_CATALOG, SampleSpec, SourceSpec

HF_DATA_DIR = ROOT / "data" / "hf"
FALLBACK_DIR = ROOT / "data" / "fallback"
CACHE_DIR = ROOT / "data" / "cache"


def load_sample(name: str, *, force: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = SAMPLE_CATALOG[name]
    sample_dir = HF_DATA_DIR / spec.id
    table_path = sample_dir / "table.parquet"
    metadata_path = sample_dir / "metadata.json"
    if not force and table_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        used_source = SourceSpec(metadata["source_kind"], metadata["source_id"])
        _record_source_status(name, spec, used_source, metadata.get("source_failures", []))
        return pd.read_parquet(table_path), metadata

    sample_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    used_source: SourceSpec | None = None
    frame: pd.DataFrame | None = None
    for source in (spec.source, *spec.fallbacks):
        try:
            frame = _read_source(source, sample_dir)
            used_source = source
            break
        except Exception as exc:  # every source gets an independent fallback attempt
            failures.append(
                {
                    "source": source.source_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if frame is None or used_source is None:
        _record_source_status(name, spec, None, failures)
        raise RuntimeError("; ".join(item["error"] for item in failures))

    frame = _normalize_frame(frame, spec)
    metadata = build_metadata(frame, spec, used_source, failures)
    frame.to_parquet(table_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _record_source_status(name, spec, used_source, failures)
    return frame, metadata


def read_upload(filename: str, content: bytes) -> pd.DataFrame:
    buffer = BytesIO(content)
    if filename.lower().endswith(".csv"):
        frame = pd.read_csv(buffer, low_memory=False)
    elif filename.lower().endswith(".parquet"):
        frame = pd.read_parquet(buffer)
    else:
        raise ValueError("Upload must be a CSV or Parquet file")
    if frame.empty:
        raise ValueError("Uploaded table has no rows")
    if frame.shape[1] < 2:
        raise ValueError("Uploaded table needs at least one feature and one target column")
    return frame.convert_dtypes()


def build_upload_metadata(
    frame: pd.DataFrame,
    filename: str,
    target: str,
    task: str,
) -> dict[str, Any]:
    if target not in frame:
        raise ValueError(f"Target column is missing: {target}")
    spec = SampleSpec(
        id="upload",
        task=task,
        target=target,
        source_target=target,
        story=f"Uploaded table: {filename}",
        source=SourceSpec("upload", filename),
        fallbacks=(),
    )
    return build_metadata(frame, spec, spec.source, [])


def split_from_metadata(
    frame: pd.DataFrame, metadata: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hidden = set(metadata["hidden_indices"])
    hidden_mask = frame.index.to_series().isin(hidden)
    return frame.loc[~hidden_mask].copy(), frame.loc[hidden_mask].copy()


def display_frame(frame: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    shown = frame.copy()
    hidden = set(metadata["hidden_indices"])
    hidden_mask = shown.index.to_series().isin(hidden)
    shown.insert(0, "Status", hidden_mask.map({True: "● Hidden", False: "● Known"}))
    shown[metadata["target"]] = shown[metadata["target"]].astype("object")
    shown.loc[hidden_mask, metadata["target"]] = pd.NA
    return shown


def build_metadata(
    frame: pd.DataFrame,
    spec: SampleSpec,
    used_source: SourceSpec,
    failures: list[dict[str, str]],
) -> dict[str, Any]:
    clean_indices = frame.index[frame[spec.target].notna()].tolist()
    if len(clean_indices) < 2:
        raise ValueError(f"Target {spec.target!r} has fewer than two usable rows")
    stratify = None
    if spec.task in {"binary", "multiclass"}:
        labels = frame.loc[clean_indices, spec.target]
        class_count = labels.nunique(dropna=True)
        hidden_count = math.ceil(len(labels) * spec.test_size)
        known_count = len(labels) - hidden_count
        if (
            class_count > 1
            and labels.value_counts().min() >= 2
            and hidden_count >= class_count
            and known_count >= class_count
        ):
            stratify = labels
    known, hidden = train_test_split(
        clean_indices,
        test_size=spec.test_size,
        random_state=spec.random_state,
        stratify=stratify,
    )
    return {
        "id": spec.id,
        "source_id": used_source.source_id,
        "source_kind": used_source.kind,
        "task": spec.task,
        "target": spec.target,
        "story": spec.story,
        "rows": len(frame),
        "columns": len(frame.columns),
        "known_rows": len(known),
        "hidden_rows": len(hidden),
        "hidden_indices": sorted(hidden),
        "test_size": spec.test_size,
        "random_state": spec.random_state,
        "row_cap": spec.row_cap,
        "source_failures": failures,
    }


def _read_source(source: SourceSpec, sample_dir: Path) -> pd.DataFrame:
    if source.kind == "huggingface":
        from datasets import load_dataset

        dataset = load_dataset(
            source.source_id,
            token=os.environ["HF_TOKEN"],
            cache_dir=str(sample_dir / "datasets"),
        )
        splits = source.splits or tuple(dataset.keys())
        return pd.concat([dataset[split].to_pandas() for split in splits], ignore_index=True)
    if source.kind == "openml":
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml(data_id=int(source.source_id), as_frame=True, parser="auto")
        frame = bunch.data.copy()
        frame[bunch.target.name] = bunch.target
        return frame
    if source.kind == "sklearn":
        from sklearn.datasets import fetch_california_housing

        return fetch_california_housing(as_frame=True).frame
    if source.kind == "url":
        return pd.read_csv(source.source_id)
    if source.kind == "url_semicolon":
        return pd.read_csv(source.source_id, sep=";")
    if source.kind == "local":
        return pd.read_csv(FALLBACK_DIR / source.source_id)
    raise ValueError(f"Unsupported source kind: {source.kind}")


def _normalize_frame(frame: pd.DataFrame, spec: SampleSpec) -> pd.DataFrame:
    normalized = frame.copy()
    aliases = dict(spec.column_aliases)
    normalized = normalized.rename(columns=aliases)
    normalized = normalized.drop(columns=list(spec.drop_columns), errors="ignore")
    if spec.target not in normalized:
        raise ValueError(
            f"Target {spec.target!r} was not found. Columns: {list(normalized.columns)!r}"
        )
    if spec.row_cap and len(normalized) > spec.row_cap:
        normalized = normalized.sample(spec.row_cap, random_state=spec.random_state)
    return normalized.reset_index(drop=True).convert_dtypes()


def _record_source_status(
    name: str,
    spec: SampleSpec,
    used_source: SourceSpec | None,
    failures: list[dict[str, str]],
) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / "hf_status.json"
    try:
        status = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except json.JSONDecodeError:
        status = {}
    status = {key: value for key, value in status.items() if key in SAMPLE_CATALOG}
    status[name] = {
        "catalog": asdict(spec),
        "source_used": used_source.source_id if used_source else None,
        "failures": failures,
    }
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")
