---
type: Testing Guide
title: Smoke Check Map
description: Maps the repository's focused smoke scripts to their assertions, external requirements, and skip behavior.
tags: [testing, smoke-checks, verification]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:16:42.939Z
sources:
  - id: openwiki-source-f317ee207e1653d2033c81a4
    resource: repo://CONTRIBUTING.md
  - id: openwiki-source-68b3e60cd84c1d81b154ac7d
    resource: repo://scripts/smoke_classifier.py
  - id: openwiki-source-65741d003485f1a410e67e25
    resource: repo://scripts/smoke_data.py
  - id: openwiki-source-f00e9e7d33c636e0412083ed
    resource: repo://scripts/smoke_docs.py
  - id: openwiki-source-0bf35dadf0a528e8adb7374a
    resource: repo://scripts/smoke_eda.py
  - id: openwiki-source-5645a35bab6737d87506acc2
    resource: repo://scripts/smoke_finetune_gpu.py
  - id: openwiki-source-a020c242805174ed7a866bae
    resource: repo://scripts/smoke_launcher.py
  - id: openwiki-source-f7b35ebfe24709de43487921
    resource: repo://scripts/smoke_regressor.py
  - id: openwiki-source-938708c80146e63a16f37374
    resource: repo://scripts/smoke_ui_import.py
generated: { by: "codex", at: "2026-09-23T14:16:42.939Z" }
---

# Smoke Check Map

The repository uses focused smoke scripts for behavior checks. They are bounded implementation checks, not a benchmark suite. `CONTRIBUTING.md` asks maintainers to run the documentation smoke, Ruff checks, formatting check, and `git diff --check` for each change, then choose the smoke that covers the changed surface.

## Static and local checks

| Command | What it checks | Boundary |
| --- | --- | --- |
| `\.venv\Scripts\python scripts/smoke_docs.py` | Required document headings, repository-relative Markdown links, and public `src/` module/function/class docstrings. | Static repository checks; no model or dataset download. |
| `\.venv\Scripts\ruff check .` | Ruff lint rules. | Does not execute the application. |
| `\.venv\Scripts\ruff format --check .` | Ruff formatting. | Reports formatting drift without rewriting files. |
| `git diff --check` | Whitespace errors in the current diff. | Not a behavior check. |
| `\.venv\Scripts\python scripts/smoke_launcher.py` | Expected launcher commands, first-run Notepad/exit/token-check ordering, and port 8541. | Static inspection only; it does not start Streamlit. |
| `\.venv\Scripts\python scripts/smoke_ui_import.py` | Imports `app` and checks the five deterministic synthetic-gallery frame names, columns, and nonempty rows. | Import and data-shape check; not a browser interaction test. |
| `\.venv\Scripts\python scripts/smoke.py` | Required installed imports and the five bundled fallback CSV files, including row-count and target-column checks. | Local environment and fallback data only. |

## Data and EDA checks

| Command | What it checks | Boundary |
| --- | --- | --- |
| `\.venv\Scripts\python scripts/smoke_data.py` | Loads all five catalog samples, requires each recorded source to be the configured primary Hugging Face source, checks target columns, cache artifacts, and nonoverlapping split coverage. | Uses cached data when available; fresh retrieval needs Hub access and `HF_TOKEN`. A fallback alone does not pass this strict check. |
| `\.venv\Scripts\python scripts/smoke_eda.py` | Loads Houses and builds seven Plotly figures plus one pairplot, asserting that all eight output files are nonempty. | Exercises figure generation on one sample; it does not evaluate model quality. |

## Model checks

| Command | What it checks | Boundary |
| --- | --- | --- |
| `\.venv\Scripts\python scripts/smoke_regressor.py` | A zero-shot, single-copy run on a 400-row Houses sample; checks the regression checkpoint, finite RMSE/R², and required run artifacts. | Requires data and authenticated model access; this is a bounded run, not a benchmark. |
| `\.venv\Scripts\python scripts/smoke_classifier.py` | A zero-shot, single-copy run on a stratified 400-row Machines sample; checks the classifier checkpoint, finite accuracy, confusion-matrix coverage, and run artifacts. | Requires data and authenticated model access; this is a bounded run, not a benchmark. |
| `\.venv\Scripts\python scripts/smoke_mitra.py` | Runs the regressor smoke and then the classifier smoke. | Convenience wrapper for both zero-shot checks; it inherits their prerequisites. |
| `\.venv\Scripts\python scripts/smoke_finetune_gpu.py` | When CUDA is available, runs fine-tuned eight-copy regression and classification smokes and checks their flags and artifacts. | Without CUDA it records `cuda: false` and skips model work. A skip is not a pass. |

Model runs need `HF_TOKEN`, data access, and time. Report credential- or hardware-dependent skips as skips. The classifier/regressor smokes show that the selected path can complete for these bounded samples; they do not establish general model quality, dataset validity, or a comparison against a broad AutoML field.

For current instructions and per-change selection, see [`CONTRIBUTING.md`](../../CONTRIBUTING.md). For interpretation limits see [Evaluation and Trust Boundary](../concepts/evaluation-and-trust-boundary.md); for the modeled workflow see [Model Training and Results](../workflows/model-training-and-results.md).
