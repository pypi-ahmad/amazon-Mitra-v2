# Glass Box

Glass Box is a Windows-native Streamlit tutorial that shows how Amazon
AutoGluon Mitra-v2 reads a table, creates a deterministic train/test split,
runs the released classifier or regressor, and evaluates predictions on the
hidden rows. It includes classic EDA and an explicit comparison with one
scikit-learn baseline.

## Run on Windows 11

Set `HF_TOKEN` as a Windows user environment variable. Do not put the token in
Git or source files.

```powershell
[Environment]::SetEnvironmentVariable("HF_TOKEN", "<your-token>", "User")
```

Open a new terminal after setting the variable, then double-click `run.cmd`.
The launcher:

1. Copies `.env.example` to `.env`, opens it in Notepad, and exits on the first
   launch. The runtime still reads `HF_TOKEN` from the Windows environment.
2. Creates `.venv` with `py -3 -m venv .venv`.
3. Installs `requirements.txt` and the pinned official `mitra-finetune` clone.
4. Stops an existing listener on port 8541.
5. Opens Glass Box at `http://localhost:8541`.

Model weights download when a run starts.

## Bundled tables

All primary tables are loaded with Hugging Face `datasets` and `HF_TOKEN`.

| Tutorial name | Source ID | Task | Target |
| --- | --- | --- | --- |
| Houses | `gvlassis/california_housing` | Regression | `MedHouseVal` |
| Machines | `EddyGiusepe/Modified_dataset_for_predictive_maintenance` | Binary classification | `Machine failure` |
| Adult income | `scikit-learn/adult-census-income` | Binary classification | `income` |
| Credit-g | `AiresPucrs/german-credit-data` | Binary classification | `Risk` |
| Wine quality | `codesignal/wine-quality` | Regression | `quality` |

See [dataset sources and licenses](docs/DATASETS.md) for splits, fallbacks,
licenses, and cached shapes. CSV and Parquet uploads use the same data and
model flow.

## MITRA controls

The declared problem type selects the released head:

- Classification: `autogluon/mitra-classifier-2`
- Regression: `autogluon/mitra-regressor-2`

| UI control | AutoGluon behavior |
| --- | --- |
| Fine-tune on | `MITRA.fine_tune=True` and `MITRA.fine_tune_steps=50` by default |
| Fine-tune off | `MITRA.fine_tune=False`; `fine_tune_steps` is omitted |
| Eight copies on | `num_bag_folds=8` |
| Eight copies off | `num_bag_folds=0` for one model |

The app sets `num_gpus=1` when `torch.cuda.is_available()` is true and uses
`num_gpus=0` otherwise. CPU zero-shot runs are supported. CPU fine-tuning and
eight-fold prediction can be substantially slower, so the UI shows a warning.

See [exact MITRA flags](docs/MITRA_FLAGS.md) and the
[architecture](docs/ARCHITECTURE.md) for implementation details.

## Documentation

- [Developer guide](docs/DEVELOPER_GUIDE.md): architecture, local workflow, artifacts, and checks.
- [Contributor runbook](CONTRIBUTING.md): safe changes, verification, and documentation policy.
- [Onboarding](docs/ONBOARDING.md): a first-session tour of the app and source tree.
- [Zero to mastery](docs/ZERO_TO_MASTERY.md): a guided path from first launch to custom-table runs.
- [Python reference](docs/PYTHON_REFERENCE.md): reusable `src` interfaces and generated artifacts.
- [Dataset sources and licenses](docs/DATASETS.md): primary sources, fallbacks, and splits.

## Sources

- [Mitra-v2 technical report](https://arxiv.org/abs/2609.04540)
- [Classifier weights](https://huggingface.co/autogluon/mitra-classifier-2)
- [Regressor weights](https://huggingface.co/autogluon/mitra-regressor-2)
- [Fine-tuning and bagging code](https://huggingface.co/autogluon/mitra-finetune)
- [AutoGluon](https://github.com/autogluon/autogluon)

[STATUS.md](STATUS.md) records the current validation evidence.

## License

The Glass Box app is available under the [Apache-2.0](LICENSE) and
[MIT](LICENSE-MIT) licenses. Vendored or downloaded components retain their
upstream licenses.
