# Contributing to Glass Box

Glass Box is a Windows-native application for inspecting tabular data and
running the released AutoGluon Mitra-v2 heads. Contributions should preserve
its teaching purpose, reproducibility, and data-quality limits.

## Before changing code

1. Read the [developer guide](docs/DEVELOPER_GUIDE.md),
   [architecture](docs/ARCHITECTURE.md), and [MITRA flags](docs/MITRA_FLAGS.md).
2. Start the application with `run.cmd` so the supported Windows environment is
   in place.
3. Keep `HF_TOKEN` in the Windows user environment. Never paste it into source,
   `.env`, documentation examples, logs, screenshots, or Git history.
4. Keep generated paths out of commits: `.venv/`, `vendor/`, `data/hf/`,
   `data/runs/`, and generated JSON below `data/cache/` are intentionally
   ignored.

## Change workflow

- Make the smallest change that resolves the issue or documentation gap.
- Keep classification on `autogluon/mitra-classifier-2` and regression on
  `autogluon/mitra-regressor-2`. Do not silently substitute a v1 checkpoint.
- Preserve the deterministic split seed and same-hidden-row baseline unless the
  change explicitly concerns split policy.
- Treat the Home facts, dataset details, model flags, and status evidence as
  factual documentation. Update them in the same change when their source
  behavior changes.
- Use Google-style docstrings for new or changed public functions and classes
  under `src/`. Describe arguments, return values, observable side effects, and
  real error conditions.

## Verify the affected surface

Run the static checks for every change:

```powershell
.\.venv\Scripts\python scripts\smoke_docs.py
.\.venv\Scripts\ruff check .
.\.venv\Scripts\ruff format --check .
git diff --check
```

Then choose the relevant smoke checks:

| Change | Required smoke |
| --- | --- |
| Launcher, credentials, or port behavior | `scripts/smoke_launcher.py` |
| Page imports, theme, or gallery | `scripts/smoke_ui_import.py` |
| Dataset catalog, normalization, or split metadata | `scripts/smoke_data.py` |
| EDA helper or page | `scripts/smoke_eda.py` |
| MITRA runner, checkpoint choice, or artifacts | `scripts/smoke_regressor.py` and `scripts/smoke_classifier.py` |
| Fine-tuning or bagging | `scripts/smoke_finetune_gpu.py` when CUDA is available |

Model smokes need `HF_TOKEN`, data access, and enough time. Report any skipped
hardware- or credential-dependent check plainly. Do not claim that a skipped
model run completed.

## Documentation policy

Use the right document for the reader:

- `README.md` is the product overview and quickest supported launch path.
- `docs/DEVELOPER_GUIDE.md` explains local development and project mechanics.
- `docs/ONBOARDING.md` gets a new maintainer oriented.
- `docs/ZERO_TO_MASTERY.md` is a guided user tutorial.
- `docs/PYTHON_REFERENCE.md` is the reusable-library reference.
- `docs/DATASETS.md`, `docs/MITRA_FLAGS.md`, and `STATUS.md` record source-backed
  implementation facts and validation evidence.

Use GitHub-flavored Markdown, descriptive link labels, and repository-relative
links for local files. Run `smoke_docs.py` after documentation edits. Do not
describe an unverified runtime result as completed.
