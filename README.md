# Glass Box

Windows-native Streamlit tutorial for seeing how Amazon AutoGluon Mitra-v2 works with tabular data.

## Run

1. Set `HF_TOKEN` as a Windows user environment variable.
2. Double-click `run.cmd`.

The launcher closes any process already listening on port `8541`, then opens Glass Box at `http://localhost:8541`.
3. If Notepad opens, close it to continue setup.

The launcher creates a Python 3.13 `.venv` with uv, installs the official Mitra fine-tuning package,
and starts Streamlit. No model weights download until **Run Mitra-v2** is clicked.

## Tutorial

Choose Houses, Machines, Adult income, Credit-g, Wine quality, or upload CSV/Parquet. Inspect the data and EDA, compare the
traditional workflow with Mitra-v2, study the synthetic-prior idea, then explicitly train and review
held-out metrics. Model output does not certify that a table is trustworthy.

Sources: [technical report](https://arxiv.org/abs/2609.04540),
[classifier](https://huggingface.co/autogluon/mitra-classifier-2),
[regressor](https://huggingface.co/autogluon/mitra-regressor-2), and
[fine-tuning recipe](https://huggingface.co/autogluon/mitra-finetune).

The Glass Box app is available under the [Apache-2.0](LICENSE) and [MIT](LICENSE-MIT)
licenses. Vendored MITRA components retain their upstream license notices.
