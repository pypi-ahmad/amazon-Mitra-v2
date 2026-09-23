---
type: Quickstart
title: Quickstart and Task Map
description: Provides the supported Windows first-run path and routes maintainers to the right architecture, data, model, operations, and verification pages.
tags: [quickstart, windows, navigation]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:16:42.939Z
sources:
  - id: openwiki-source-a0e9f5d2bd8f15a70d93d634
    resource: repo://app_pages/home.py
  - id: openwiki-source-94bf2714adac6970c5021bb3
    resource: repo://app_pages/train.py
  - id: openwiki-source-17dded95897e01ee430228e3
    resource: repo://app.py
  - id: openwiki-source-f317ee207e1653d2033c81a4
    resource: repo://CONTRIBUTING.md
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
  - id: openwiki-source-0b361e5538aca06f3052fb20
    resource: repo://run.cmd
generated: { by: "codex", at: "2026-09-23T14:16:42.939Z" }
---

# Quickstart and Task Map

Glass Box is a Windows 11 Streamlit tutorial for inspecting a table and evaluating the released Mitra-v2 classifier or regressor on hidden rows.

## First run

1. Set `HF_TOKEN` as a Windows user environment variable. Keep the value out of `.env`, source files, docs, and Git.
2. Open a new terminal and run `run.cmd`. The first run creates `.env` from `.env.example`, opens it in Notepad, and exits.
3. Close Notepad and run `run.cmd` again. The launcher prepares `.venv`, installs the dependencies and pinned MITRA helper, and starts the app at `http://localhost:8541`.
4. Choose Houses or Machines on Home, inspect the table and split on Data, and explore it on EDA. Model weights are downloaded only after you explicitly start a run on Train & Predict.

```powershell
[Environment]::SetEnvironmentVariable("HF_TOKEN", "<your-token>", "User")
```

For launcher and credential details see [Windows Setup and Credentials](operations/windows-setup-and-credentials.md).

## Find the right guide

| If you need to understand… | Start here |
| --- | --- |
| How pages, state, loaders, model code, and result files fit together | [System Architecture](architecture/system-overview.md) |
| Catalog samples, uploads, fallbacks, caches, or hidden rows | [Data Selection and Splitting](workflows/data-selection-and-splitting.md) |
| The explicit model-run action, checkpoints, fit settings, or artifacts | [Model Training and Results](workflows/model-training-and-results.md) |
| What hidden-row metrics do and do not demonstrate | [Evaluation and Trust Boundary](concepts/evaluation-and-trust-boundary.md) |
| Which local, data, model, and CUDA checks apply | [Smoke Check Map](testing/smoke-checks.md) |

For the product overview and dataset references, use [`README.md`](../README.md), [`docs/DATASETS.md`](../docs/DATASETS.md), and [`docs/ONBOARDING.md`](../docs/ONBOARDING.md). For the exact model flags see [`docs/MITRA_FLAGS.md`](../docs/MITRA_FLAGS.md).

## Maintainer checks

The contributor runbook requires the static documentation smoke, Ruff lint, Ruff formatting check, and `git diff --check`; add the focused smoke for the affected area. Model checks need authenticated model access, data access, and time. A hardware-dependent check that skips should be reported as skipped.

```powershell
.\.venv\Scripts\python scripts\smoke_docs.py
.\.venv\Scripts\ruff check .
.\.venv\Scripts\ruff format --check .
git diff --check
```

See [Smoke Check Map](testing/smoke-checks.md) for what those checks assert and their limits. This page does not record a claim that any check or model run has passed.
