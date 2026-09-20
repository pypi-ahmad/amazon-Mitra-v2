# Glass Box status

## Current implementation

- Eight-page dark Streamlit tutorial for native Windows 11.
- Both released heads run through AutoGluon `TabularPredictor` after the user
  starts a run.
- Classification uses `autogluon/mitra-classifier-2`; regression uses
  `autogluon/mitra-regressor-2` with the official 1,000-bin compatibility
  patch from `mitra-finetune` v0.3.0.
- Five primary Hugging Face tables and CSV/Parquet upload are supported.
- No model loads during startup, navigation, data inspection, or EDA.
- Ollama, Agnes, WSL, and Docker are not used.

## Dataset smoke: `catalog.json`

`scripts/smoke_data.py` completed 5/5 primary Hugging Face sources and wrote
`data/cache/catalog.json`.

| Table | Source used | Shape | Target |
| --- | --- | --- | --- |
| Houses | `gvlassis/california_housing` | 20,640 × 9 | `MedHouseVal` |
| Machines | `EddyGiusepe/Modified_dataset_for_predictive_maintenance` | 10,000 × 7 | `Machine failure` |
| Adult income | `scikit-learn/adult-census-income` | 20,000 × 15 | `income` |
| Credit-g | `AiresPucrs/german-credit-data` | 1,000 × 10 | `Risk` |
| Wine quality | `codesignal/wine-quality` | 1,599 × 12 | `quality` |

## Model smokes: both released heads

`scripts/smoke_regressor.py` and `scripts/smoke_classifier.py` each use 400
rows, a deterministic 90/10 split, zero-shot mode, one model, CPU, and a
180-second fit limit. `scripts/smoke_mitra.py` remains an orchestrator for both.

| Head | Runtime | Current-split metrics |
| --- | ---: | --- |
| `autogluon/mitra-regressor-2` | 59.2 s | RMSE 0.4065; MAE 0.2671; R² 0.7798; MAPE 16.93% |
| `autogluon/mitra-classifier-2` | 47.9 s | Accuracy 0.975; F1 0.000; ROC-AUC 1.0000 |

The classification smoke has one positive row in its hidden 40-row sample.
The model predicted no positive label, so F1 is zero even though ROC-AUC is
finite. These are smoke metrics on the current split, not benchmark claims.

`data/cache/last_run.json` points to the classification run because it finished
last. The required task-specific summaries are `last_run_reg.json` and
`last_run_clf.json`. Each successful run contains `flags.json`, `metrics.json`,
`predictions.csv`, `leaderboard.csv`, `run.log`, and the persisted predictor.
Cache JSON, checkpoints, predictors, and run directories are generated locally
and ignored by Git.

## GPU fine-tune smoke

`scripts/smoke_finetune_gpu.py` checks CUDA before loading a dataset or model.
With Torch `2.14.0+cu132`, it detected the NVIDIA GeForce RTX 4060 Laptop GPU
and completed both deterministic 400-row jobs with 50 fine-tune steps, eight
bag children, and a 1,800-second limit per task.

| Head | Wall clock | Current-split metrics |
| --- | ---: | --- |
| `autogluon/mitra-regressor-2` | 162.2 s | RMSE 0.4165; MAE 0.2749; R² 0.7687; MAPE 17.54% |
| `autogluon/mitra-classifier-2` | 190.5 s | Accuracy 1.000; F1 1.000; ROC-AUC 1.000 |

`last_run_reg_ft.json` and `last_run_clf_ft.json` contain the measured wall
clock and point to completed runs. Each run records `num_bag_folds=8`,
`fine_tune_steps=50`, and model `Mitra_BAG_L1`; the classifier confusion counts
are `[[39, 0], [0, 1]]`. These are smoke metrics on one 40-row hidden split,
not benchmark claims. The Results page lists all four zero-shot/fine-tuned task
variants.

## Installed environment

- AutoGluon Tabular 1.6.3
- Streamlit 1.64.0
- Torch 2.14.0+cu132; CUDA 13.2 on NVIDIA GeForce RTX 4060 Laptop GPU
- datasets 4.8.5
- huggingface_hub 1.32.0
- mitra-finetune 0.3.0 at revision
  `b4701e8148dc33b00ed15d7086ff59816957cde4`

## Installation issue and resolution

Direct helper-package installation previously failed verbatim:

```text
fatal: expected 'packfile'
fatal: could not fetch 9693d97910b8aeb3dbc15a7f5c84712402462129 from promisor remote
ERROR: Failed to build 'mitra-finetune' when git clone --filter=blob:none
```

The current model card documents this Hub partial-clone limitation. `run.cmd`
now performs a normal clone, checks out the verified revision, and installs the
local directory. `requirements.txt` retains `autogluon.tabular[mitra]` and
`tabarena`; Torch and CUDA wheels are not pinned.

## Other validation

- Launcher smoke checks the Windows venv commands, first-run exit, local helper
  install, token guard, port cleanup, and Streamlit port 8541.
- Classic EDA smoke renders eight figures and writes `eda_ok.json`.
- UI import smoke imports the app and generates five synthetic teaching plots.
- Streamlit AppTest executes every page without a browser.

## Documentation audit

The developer guide, contributor runbook, onboarding path, zero-to-mastery
tutorial, and Python API reference sit beside the source. The documentation
smoke checks required guides, repository-relative Markdown links, and top-level
public `src` docstrings without downloading data or model weights. The CUDA
figures above describe the validated local environment. The default launcher
does not pin a CUDA Torch wheel, so it remains CPU-compatible.
