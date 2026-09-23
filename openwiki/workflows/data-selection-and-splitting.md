---
type: Data Workflow
title: Data Selection and Splitting
description: Follows bundled tables and uploads through loading, fallback and cache handling, normalization, shared page state, and the hidden-row split.
tags: [data, datasets, uploads, splitting]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:16:42.939Z
sources:
  - id: openwiki-source-59fe4926a594e465cfde0858
    resource: repo://app_pages/data.py
  - id: openwiki-source-a0e9f5d2bd8f15a70d93d634
    resource: repo://app_pages/home.py
  - id: openwiki-source-8b9e30a670716b39995d2de0
    resource: repo://docs/DATASETS.md
  - id: openwiki-source-65741d003485f1a410e67e25
    resource: repo://scripts/smoke_data.py
  - id: openwiki-source-d142615b2faeb5e851e19033
    resource: repo://src/data_catalog.py
  - id: openwiki-source-86ef16584268767232778345
    resource: repo://src/load.py
generated: { by: "codex", at: "2026-09-23T14:16:42.939Z" }
---

# Data Selection and Splitting

The Data page resolves one active table for EDA and training. It can load a bundled catalog sample or parse a user upload; both paths produce task metadata and a deterministic set of known and hidden row indices.

## Selecting a bundled table

Home promotes Houses and Machines and lists Adult income, Credit-g, and Wine quality as additional samples. Choosing a table sets the selected dataset and opens the Data page. The catalog declares each table's stable cache ID, task, target, primary source, ordered fallbacks, normalization rules, and split policy.

The five examples cover regression and classification: Houses (`MedHouseVal`), Machines (`Machine failure`), Adult income (`income`), Credit-g (`Risk`), and Wine quality (`quality`). Primary sources and licenses are listed in [Bundled datasets](../../docs/DATASETS.md).

## Loading, fallback, and cache behavior

For a bundled sample, `load_sample` reuses a local Parquet table and metadata when the cached catalog source ID still matches. Otherwise it tries the primary source followed by each configured fallback. Each failed source is recorded; the first successful frame is normalized and stored with metadata under `data/hf/<sample>/`. A changed catalog source invalidates the matching cached table so the new source is read.

Fallbacks are ordered by sample: Houses uses scikit-learn California housing then its bundled CSV; Machines uses the UCI AI4I CSV then its bundled CSV; Adult income uses OpenML 1590 then its bundled CSV; Credit-g uses OpenML 31 then its bundled CSV; Wine quality uses the UCI red-wine CSV then its bundled CSV. Source failures are recorded in `data/cache/hf_status.json` for diagnosis.

## Upload path

The Data page accepts CSV and Parquet files. The loader rejects unsupported extensions, empty tables, and tables with fewer than two columns. For an accepted upload, the page lets the user choose a target column and problem type, or infer the type automatically. The loader builds catalog-compatible metadata for that target and applies the same split policy. Selecting **Use bundled sample** clears the upload and returns to the selected catalog table.

The shared state module resolves the active sample or upload for Data, EDA, the generated Python script, and training pages. This keeps table selection and task details consistent across those views.

## Known and hidden rows

Split metadata is built from rows whose target is present. By default, ten percent of labeled rows are held out with random seed 42; classification is stratified when the class counts and both partition sizes allow it. Metadata stores the hidden indices, allowing each page to reconstruct known and hidden copies from the same full frame.

The Data page displays row counts and lets users inspect all rows, known rows, and hidden rows. In the single-row hidden inspector, the target value is replaced with a hidden marker. Training receives separate known and hidden tables; the hidden rows are used to evaluate predictions. See [Evaluation and Trust Boundary](../concepts/evaluation-and-trust-boundary.md).

## Verification

[`scripts/smoke_data.py`](../../scripts/smoke_data.py) checks that all five configured primary Hugging Face sources load, the expected target exists, and the known/hidden indices are disjoint and cover every labeled row. Its strict source assertion means that a successful fallback alone does not satisfy the check. See [Smoke Check Map](../testing/smoke-checks.md) for its cache and access requirements.
