---
type: Operations Guide
title: Windows Setup and Credentials
description: Covers the supported Windows launcher, Python and dependency setup, first-run acknowledgement, Hugging Face credential source, and CPU/GPU behavior.
tags: [windows, setup, credentials, operations]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:16:42.939Z
sources:
  - id: openwiki-source-e8e61d605125cac4d909755e
    resource: repo://docs/ARCHITECTURE.md
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-0b361e5538aca06f3052fb20
    resource: repo://run.cmd
  - id: openwiki-source-5645a35bab6737d87506acc2
    resource: repo://scripts/smoke_finetune_gpu.py
  - id: openwiki-source-a020c242805174ed7a866bae
    resource: repo://scripts/smoke_launcher.py
  - id: openwiki-source-ec840809b2d30144c3aaebd1
    resource: repo://src/mitra_run.py
generated: { by: "codex", at: "2026-09-23T14:16:42.939Z" }
---

# Windows Setup and Credentials

The supported launch path is `run.cmd` on Windows 11. It prepares a local Python environment, installs the application dependencies and pinned MITRA helper, clears a listener already using port 8541, and starts Streamlit.

## First launch and later launches

On the first launch, `run.cmd` copies `.env.example` to `.env`, opens the new file in Notepad, and exits. This is a setup acknowledgement; the application does not load `HF_TOKEN` from `.env`. Set `HF_TOKEN` in the Windows user environment, open a new terminal so the process inherits it, and start `run.cmd` again. The launcher checks for the environment variable before continuing.

After the credential check, the launcher creates `.venv` with `py -3`, installs `requirements.txt`, fetches and checks out the pinned `autogluon/mitra-finetune` revision, installs that local package, and starts the app. It terminates an existing listener on port 8541 before running Streamlit at `http://localhost:8541`.

The project metadata declares Python `>=3.11,<3.14`. Check the active Python version against that range when a local installation fails; `py -3` selects the registered Python 3 runtime.

## Credential and model access boundary

The model runner reads `HF_TOKEN` from the process environment when it preflights the selected Hugging Face checkpoint. Keep the token out of `.env`, source files, documentation, logs, screenshots, and Git history. A missing or rejected token prevents checkpoint preflight; the runner does not switch to a v1 checkpoint.

The app uses one GPU when `torch.cuda.is_available()` is true and configures zero GPUs otherwise. CPU zero-shot runs are supported. The fine-tuning smoke records a skip when CUDA is unavailable rather than running that GPU-specific workflow on CPU.

## Operational checks

[`scripts/smoke_launcher.py`](../../scripts/smoke_launcher.py) statically checks the launcher commands, first-run ordering, token guard, port, and Streamlit command without starting the app. For the full smoke-test map see [Smoke Check Map](../testing/smoke-checks.md); for model-run behavior see [Model Training and Results](../workflows/model-training-and-results.md).
