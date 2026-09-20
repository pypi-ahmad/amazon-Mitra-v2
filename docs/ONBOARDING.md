# Onboarding

This guide helps a new maintainer build a working mental model of Glass Box.

## First 15 minutes: launch and navigate

1. Set `HF_TOKEN` as described in the [README](../README.md), then open a new
   PowerShell window.
2. Run `./run.cmd`. On first use it creates `.env`, opens Notepad, and exits;
   close Notepad and run it again.
3. Open `http://localhost:8541`. Choose **Houses** or **Machines** from Home.
4. Visit **Data**. The app loads or reuses a cached table and marks 90% of
   labeled rows as known and 10% as hidden.
5. Visit **EDA**. Confirm that this page shows ordinary data checks before any
   model run: shape, dtypes, missingness, distributions, relationships, and the
   split view.

No model weights download during this tour. That only happens after the
explicit button on **Train & Predict**.

## First hour: understand the execution path

Start at [Architecture](ARCHITECTURE.md), then trace this path in code:

```text
app.py → app_pages/data.py → src/state.py → src/load.py
      → app_pages/train.py → src/mitra_run.py → app_pages/results.py
```

`src/data_catalog.py` specifies the five tutorial datasets. `src/load.py`
normalizes a source, caches it, and stores the hidden indices in metadata.
`src/state.py` resolves whether the current table is a catalog item or upload.
`src/mitra_run.py` validates the known and hidden tables, selects the
released head, fits it, and persists artifacts. The pages render these results;
they do not own model-selection rules.

Read the [Python reference](PYTHON_REFERENCE.md) beside the source. It records
the public contracts and generated artifact schema.

## First contribution

Choose a small, observable improvement such as a clearer EDA label, an
additional source-backed note, or a focused documentation correction. Read the
[contributor runbook](../CONTRIBUTING.md), make the smallest change, then run:

```powershell
.\.venv\Scripts\python scripts\smoke_docs.py
.\.venv\Scripts\python scripts\smoke_ui_import.py
.\.venv\Scripts\ruff check .
```

For a data or model change, use the more specific smoke listed in the runbook.
Do not commit caches, checkpoints, the local helper clone, or credentials.

## Useful orientation questions

| Question | Where to answer it |
| --- | --- |
| Which dataset source and license apply? | [Datasets](DATASETS.md) |
| Which checkpoint and AutoGluon keys are used? | [MITRA flags](MITRA_FLAGS.md) |
| Where do model outputs live? | [Developer guide](DEVELOPER_GUIDE.md) |
| What has been run locally? | [Status](../STATUS.md) |
| How do I learn the product workflow end to end? | [Zero to mastery](ZERO_TO_MASTERY.md) |
