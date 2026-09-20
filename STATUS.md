# Glass Box status

- Eight-page `st.navigation` tutorial implemented.
- Five bundled datasets and CSV/Parquet upload supported.
- No model loads or downloads during app startup, navigation, data, or EDA checks. The dedicated
  MITRA smoke intentionally loads the official regression checkpoint.
- Classification and regression run through AutoGluon `TabularPredictor` after explicit user action.
- `mitra-regressor-2` uses a narrow compatibility module adapted from the official
  `mitra-finetune.install_reg_ce_patches` implementation at revision
  `b4701e8148dc33b00ed15d7086ff59816957cde4`. Its released checkpoint has a 1,000-bin
  distributional head while stock AutoGluon 1.6.3 expects one scalar regression output.

The direct helper-package install was removed after the Windows pip dry-run failed verbatim:

```text
fatal: expected 'packfile'
fatal: could not fetch 9693d97910b8aeb3dbc15a7f5c84712402462129 from promisor remote
ERROR: Failed to build 'mitra-finetune' when git clone --filter=blob:none
```
- `requirements.txt` has no explicit Torch or CUDA wheel pin.
- Existing artifacts and caches were preserved.

## Dataset sources

- Houses: `gvlassis/california_housing` (`train`, `validation`, and `test`).
- Machines: `EddyGiusepe/Modified_dataset_for_predictive_maintenance` (`train` and `test`).
- Adult income: `scikit-learn/adult-census-income` (`train`, capped at 20,000 rows).
- Credit-g: OpenML dataset `31`.
- Wine quality: `codesignal/wine-quality` (`red`).

Every sample uses a deterministic 90/10 known/hidden split with random seed 42. Classification
samples are stratified when their class counts allow it.

## Validation

- `scripts/smoke_data.py`: 5/5 sources loaded and `data/cache/catalog.json` written.
- Houses: 20,640 rows; Machines: 10,000; Adult income: 20,000; Credit-g: 1,000;
  Wine quality: 1,599.
- `scripts/smoke.py`: dependencies and five local fallback datasets passed.
- Streamlit AppTest: Home and all seven remaining pages executed without exceptions.
- Classic EDA smoke: eight plots rendered from Houses and `data/cache/eda_ok.json` written.
- Installed API audit: AutoGluon Tabular 1.6.3 exposes MITRA for binary, multiclass, and
  regression tasks; eight-copy mode uses native predictor-level `num_bag_folds=8`.
- MITRA CPU smoke: 400 Houses rows, zero-shot, one copy, 180-second fit limit; real predictions
  completed in 50.7 seconds and `data/cache/last_run.json` was written.
