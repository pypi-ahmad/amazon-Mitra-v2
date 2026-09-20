# Bundled datasets

All five primary sources are loaded through Hugging Face `datasets` with the
user's `HF_TOKEN`. Each table is normalized, cached under `data/hf/<slot>/`,
and split deterministically with seed 42. Ten percent of labeled rows are
hidden for evaluation; classification splits are stratified when possible.

| Table | Primary source | License reported by source | Split(s) | Task and target | Cached shape |
| --- | --- | --- | --- | --- | --- |
| Houses | [`gvlassis/california_housing`](https://huggingface.co/datasets/gvlassis/california_housing) | MIT | `train`, `validation`, `test` | Regression · `MedHouseVal` | 20,640 × 9 |
| Machines | [`EddyGiusepe/Modified_dataset_for_predictive_maintenance`](https://huggingface.co/datasets/EddyGiusepe/Modified_dataset_for_predictive_maintenance) | Apache-2.0 | `train`, `test` | Binary · `Machine failure` | 10,000 × 7 |
| Adult income | [`scikit-learn/adult-census-income`](https://huggingface.co/datasets/scikit-learn/adult-census-income) | CC0-1.0 | `train` | Binary · `income` | 20,000 × 15 |
| Credit-g | [`AiresPucrs/german-credit-data`](https://huggingface.co/datasets/AiresPucrs/german-credit-data) | CC0-1.0 | `train` | Binary · `Risk` | 1,000 × 10 |
| Wine quality | [`codesignal/wine-quality`](https://huggingface.co/datasets/codesignal/wine-quality) | CC-BY-4.0 | `red` | Regression · `quality` | 1,599 × 12 |

Licenses above are the metadata declared by the source repositories. They do
not change the Glass Box application license.

## Normalization

- Houses aligns the source target to `MedHouseVal` when needed.
- Machines removes identifiers and the post-failure `Failure Type` field, then
  renames the source target to `Machine failure`.
- Adult income is deterministically capped at 20,000 rows.
- Credit-g uses `Risk`; its OpenML fallback target `class` is renamed to match.
- Wine quality uses only the red-wine split.

## Fallbacks

The loader tries the primary Hugging Face source first. If it fails, it records
the verbatim error in `data/cache/hf_status.json` and then tries:

| Table | Fallback order |
| --- | --- |
| Houses | scikit-learn California housing, bundled CSV |
| Machines | UCI AI4I CSV, bundled CSV |
| Adult income | OpenML 1590, bundled CSV |
| Credit-g | OpenML 31, bundled CSV |
| Wine quality | UCI red-wine CSV, bundled CSV |

The strict data smoke requires all five primary Hugging Face sources to work;
fallback success alone does not satisfy that smoke.
