# Glass Box: zero to mastery

Follow one table through inspection, evaluation, and reproduction. The model
can help with prediction, but it cannot certify that the table is trustworthy.

## 1. Start Glass Box

Set `HF_TOKEN` in your Windows user environment and start `run.cmd` as shown in
the [README](../README.md). Open `http://localhost:8541` after the launcher
starts. The first launch creates `.env` and opens it in Notepad; that file is
not a token store.

At this point, Home shows the Houses and Machines cards, plus Adult income,
Credit-g, and Wine quality.

## 2. Follow a regression table

Choose **Houses** and open **Data**. The table target is `MedHouseVal`, so this
is a regression lesson. The page shows total rows, known training rows, hidden
test rows, columns, and target. The hidden-row inspector masks the target on a
row reserved for evaluation.

Open the table tabs. **Known** is what a run may learn from. **Hidden** remains
held out until the run evaluates it. You can download each view without exposing
the masked target in the display table.

You should now be able to explain why the app uses a fixed known/hidden split
instead of reporting training-set performance.

## 3. Do classic EDA before modeling

Open **EDA**. Work through the tabs in order:

1. Check table size, dtypes, memory, head, and tail.
2. Inspect missing values, constant columns, high-cardinality categories, and
   the target boundary.
3. Review numeric `describe()` output and categorical value counts.
4. Use histogram, box, and violin views; then inspect the target distribution.
5. Compare numeric correlations and a selected scatter plot.
6. Draw a pairplot from at most five numeric columns and 400 sampled rows.
7. Revisit the known/hidden split.

You still own the decisions about leakage, data quality, sampling, domain
meaning, and evaluation design.

## 4. Run a low-cost zero-shot experiment

Use the header toggles to turn **Fine-tune** off and **Eight copies** off. On
**Train & Predict**, confirm the displayed Hugging Face repository is
`autogluon/mitra-regressor-2`, then select **Run Mitra-v2**.

The app first preflights the v2 checkpoint with `HF_TOKEN`. It fits the model
on known rows, predicts hidden rows, fits a HistGradientBoosting reference
baseline on the same split, and writes a timestamped run folder.

The run log names the Hugging Face repository, device, model, and selected
mode. A missing or inaccessible token fails with `set HF_TOKEN`; the app does
not silently use a v1 checkpoint.

## 5. Read results as current-split evidence

Open **Results** and select the saved regression run. For Houses, review RMSE,
MAE, R², and MAPE when all held-out target values are positive. Compare them
with the same-split baseline, inspect prediction versus actual, then read the
per-row residuals.

These values describe this particular hidden split. They are not paper
benchmarks, Elo scores, or a certification of data quality.

## 6. Explore fine-tuning and bagging

If the device label reports CUDA, turn on **Fine-tune** and **Eight copies**.
The default is 50 fine-tune steps and eight AutoGluon bagging folds. A
fine-tuned eight-copy run is substantially more expensive than zero-shot single
model mode. CPU runs remain permitted but the UI warns that fine-tuning can be
slow.

Read [MITRA flags](MITRA_FLAGS.md) for the exact mapping. The app records the
chosen flags in each run's `flags.json`.

## 7. Repeat the lesson with classification

Choose **Machines**. Its target is `Machine failure`, so the runner selects
`autogluon/mitra-classifier-2`. Use the same EDA workflow, then run zero-shot
single-model mode before trying fine-tuning.

Classification results show accuracy and F1, plus ROC-AUC when the problem is
binary and probabilities are available. The confusion matrix and per-row error
table identify the current-split mistakes. A small or imbalanced hidden split
can make one metric unstable, so interpret it with the row counts.

## 8. Bring your own table

Open **Data** and upload CSV or Parquet. Choose the target column and either
let the app infer the problem type or select binary, multiclass, or regression.
The uploaded table uses the same deterministic split and execution path as a
bundled sample.

Before running, ensure that regression targets are numeric, binary targets have
exactly two classes, and multiclass targets have three to twenty classes. Use
EDA to look for missing values, leakage, identifiers, and data-quality issues.

## 9. Reproduce a run outside the UI

Open **Python Script (CLI)** and download the generated script. It contains the
selected dataset path, target, checkpoint, mode, bagging setting, and official
regression compatibility patch when applicable. It reads `HF_TOKEN` from the
process environment and does not print or write the token.

For a source-level reference to the same execution path, read
[Python reference](PYTHON_REFERENCE.md#model-execution-and-results).

## 10. Know where to look next

- [Datasets](DATASETS.md) explains every primary source, fallback, and license.
- [Developer guide](DEVELOPER_GUIDE.md) explains the local artifacts and smoke checks.
- [Architecture](ARCHITECTURE.md) explains component boundaries.
- [Status](../STATUS.md) records actual local validation evidence and its limits.
