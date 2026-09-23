---
type: Model Workflow
title: Model Training and Results
description: Traces the explicit training action through input checks, released-head selection, MITRA and baseline fitting, run artifacts, failures, and saved-result retrieval.
tags: [mitra, training, results, artifacts]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:16:42.939Z
sources:
  - id: openwiki-source-98a331068a0dc759cb5dd224
    resource: repo://app_pages/results.py
  - id: openwiki-source-94bf2714adac6970c5021bb3
    resource: repo://app_pages/train.py
  - id: openwiki-source-e8e61d605125cac4d909755e
    resource: repo://docs/ARCHITECTURE.md
  - id: openwiki-source-31c664748c33b2d0021564a5
    resource: repo://docs/MITRA_FLAGS.md
  - id: openwiki-source-68b3e60cd84c1d81b154ac7d
    resource: repo://scripts/smoke_classifier.py
  - id: openwiki-source-5645a35bab6737d87506acc2
    resource: repo://scripts/smoke_finetune_gpu.py
  - id: openwiki-source-f7b35ebfe24709de43487921
    resource: repo://scripts/smoke_regressor.py
  - id: openwiki-source-ec840809b2d30144c3aaebd1
    resource: repo://src/mitra_run.py
generated: { by: "codex", at: "2026-09-23T14:16:42.939Z" }
---

# Model Training and Results

Model execution starts only when the user presses **Run Mitra-v2** on Train & Predict. The page resolves the active table, reconstructs known and hidden rows from its metadata, and passes both tables and the selected run controls to `src/mitra_run.py`.

## Run request and validation

The training page shows the selected dataset, target, device, fit limit, fine-tuning mode, step count, and copy count. If CUDA is unavailable it warns about slower CPU inference and fine-tuning, while still respecting the selected fine-tuning setting. Pressing the run button invokes the runner and shows its run log or error in the page.

The runner checks that both tables contain the target, their feature columns match in the same order, the time limit and fine-tuning steps are valid, and the target matches the requested task. Generic `classification` is resolved to `binary` or `multiclass`. The selected checkpoint is preflighted with `HF_TOKEN` before fitting; an authentication or access failure stops that request rather than switching to an older checkpoint.

## Model and baseline fit

Binary and multiclass tasks use `autogluon/mitra-classifier-2`; regression uses `autogluon/mitra-regressor-2`. Regression installs the released 1,000-bin head compatibility patch before constructing the predictor. Fine-tuning controls the `fine_tune` flag and, when enabled, `fine_tune_steps`. The separate eight-copy control sets AutoGluon `num_bag_folds` to 8 or 0; it is a bagging setting, not a MITRA hyperparameter.

The runner fits MITRA on the known rows and predicts the hidden rows. It then fits one scikit-learn HistGradientBoosting model on the same known rows and evaluates it on the same hidden rows. The UI labels this a same-split reference, not a full AutoML benchmark.

## Persisted evidence and failures

Each run gets a UTC timestamp directory under `data/runs/`. A successful run writes:

- `flags.json` with the selected head, controls, device, and fitted model names;
- `metrics.json` with MITRA and baseline metrics plus run metadata;
- `predictions.csv` with actual and predicted values;
- `leaderboard.csv`, `run.log`, and the persisted AutoGluon `predictor/`.

If fitting fails after the run directory is created, the runner writes `error.json`, appends the failure to `run.log`, and raises the exception. The training page displays that error and retains it in session state for Results to show.

## Results retrieval

Results offers the current in-session run and saved pointers for regression/classification, zero-shot/fine-tuned variants. A saved pointer identifies a run directory; the loader checks that the run's metrics refer to the same run ID and attaches a local prediction artifact when present. The page compares MITRA and HistGradientBoosting metrics on the same split and displays predictions, a confusion matrix where applicable, and run details. If prediction files are missing, it reports that the metadata exists but the artifact is unavailable.

The classifier and regressor smoke scripts exercise zero-shot, single-copy runs on bounded samples and assert expected metrics and artifacts. The separate fine-tuning smoke requires CUDA; it runs both heads with eight copies when available and explicitly skips on CPU. These are focused path checks, not broad benchmarks. See [Evaluation and Trust Boundary](../concepts/evaluation-and-trust-boundary.md), [Windows Setup and Credentials](../operations/windows-setup-and-credentials.md), and [Smoke Check Map](../testing/smoke-checks.md).
