# Developer guide

Glass Box runs as a single Windows-native Streamlit process. It loads a
table, creates a deterministic known/hidden split, runs the released Mitra-v2
head selected by task type, and shows current-split results beside one small
scikit-learn reference model. It does not run a model server, Ollama, Agnes,
WSL, or Docker.

## Local workflow

The supported first-run path is [run.cmd](../run.cmd). It creates `.env` from
`.env.example`, opens it in Notepad, and exits. The file is only a first-run
acknowledgement; the application reads `HF_TOKEN` from the Windows user
environment, never from `.env`.

```powershell
[Environment]::SetEnvironmentVariable("HF_TOKEN", "<your-token>", "User")
```

Open a new PowerShell window and run:

```powershell
.\run.cmd
```

The launcher creates `.venv`, installs `requirements.txt`, clones and pins the
official `mitra-finetune` helper, stops a listener on port 8541, and starts
Streamlit. It does not download model weights until a run begins.

Use the created environment for checks:

```powershell
.\.venv\Scripts\python scripts\smoke_docs.py
.\.venv\Scripts\python scripts\smoke_launcher.py
.\.venv\Scripts\python scripts\smoke_ui_import.py
.\.venv\Scripts\ruff check .
.\.venv\Scripts\ruff format --check .
```

The model smokes require a valid token, cached or downloadable data, and can
take minutes. Run the narrowest relevant command:

```powershell
.\.venv\Scripts\python scripts\smoke_data.py
.\.venv\Scripts\python scripts\smoke_eda.py
.\.venv\Scripts\python scripts\smoke_regressor.py
.\.venv\Scripts\python scripts\smoke_classifier.py
.\.venv\Scripts\python scripts\smoke_finetune_gpu.py
```

`smoke_finetune_gpu.py` records `{"cuda": false}` and skips model work when
CUDA is unavailable. A CPU skip is reported as a skip, not a completed
fine-tune run.

## Source map

| Area | Responsibility |
| --- | --- |
| `app.py` and `app_pages/` | Streamlit navigation and tutorial UI. |
| `src/data_catalog.py` | Five source specifications, targets, and fallbacks. |
| `src/load.py` | Downloads, normalization, upload parsing, cache files, and deterministic splits. |
| `src/eda.py` | Explicit EDA summaries and Plotly/seaborn figure factories. |
| `src/state.py` | Session-level selected dataset, upload, target, and run settings. |
| `src/mitra_run.py` | Head selection, AutoGluon fitting, baseline, metrics, and persisted artifacts. |
| `src/theme.py` | Shared dark theme and header/sidebar controls. |
| `scripts/` | Focused smoke checks, not a full benchmark suite. |

Read [Architecture](ARCHITECTURE.md) before changing data flow or model
execution. Read [MITRA flags](MITRA_FLAGS.md) before changing any AutoGluon or
fine-tuning setting.

## Data, models, and artifacts

Bundled tables cache under `data/hf/<sample>/`; generated source diagnostics
go to `data/cache/hf_status.json`. A catalog source change invalidates the
matching cache. [Dataset documentation](DATASETS.md) is the source for IDs,
licenses, normalization, and fallback order.

Every successful model run creates `data/runs/<UTC timestamp>/` with
`flags.json`, `metrics.json`, `predictions.csv`, `leaderboard.csv`, `run.log`,
and a persisted `predictor/`. Failed runs preserve `error.json` and `run.log`.
Generated data, models, cache pointers, and the helper checkout are ignored by
Git. Do not add them to commits.

Classification always uses `autogluon/mitra-classifier-2`; regression always
uses `autogluon/mitra-regressor-2`. The runner preflights the selected
checkpoint with `HF_TOKEN` and fails with `set HF_TOKEN` rather than falling
back to a v1 model.

## Working on the app

Keep the model boundary clear. Data and EDA pages must not trigger model
downloads. `run_mitra` receives separate known and hidden tables and is the
only production path that fits Mitra. The baseline compares the same split and
does not stand in for a full AutoML benchmark.

Preserve the trust boundary in user-facing content: a model result does not
certify that a table is accurate, representative, fair, causally valid, or
trustworthy. Do not add invented Elo, cost, or benchmark claims.

The default launcher is CPU-compatible because it does not pin a CUDA wheel.
The application uses one GPU only when `torch.cuda.is_available()` is true.
Recorded CUDA evidence in [STATUS.md](../STATUS.md) describes a validated local
environment and is not a guarantee for every installation.
