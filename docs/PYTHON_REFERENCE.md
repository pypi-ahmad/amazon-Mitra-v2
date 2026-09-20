# Python reference

Reusable application code lives in the `src` package. Streamlit page files
compose these helpers. This package is not a general public SDK. The signatures
and behavior below match the repository source.

## Dataset catalog and loading

`SourceSpec` describes one loader kind, source ID, and optional Hugging Face
splits. `SampleSpec` adds normalized target, task, narrative, fallback order,
split policy, row cap, column aliases, and dropped columns. The five entries
live in `SAMPLE_CATALOG`.

| Interface | Purpose |
| --- | --- |
| `load_sample(name, *, force=False)` | Loads a normalized catalog DataFrame and metadata. Reuses a matching Parquet cache unless forced. |
| `read_upload(filename, content)` | Parses a CSV or Parquet upload and requires a nonempty table with at least two columns. |
| `build_upload_metadata(frame, filename, target, task)` | Produces catalog-compatible metadata for an upload. |
| `build_metadata(frame, spec, used_source, failures)` | Creates the deterministic split payload. |
| `split_from_metadata(frame, metadata)` | Returns copied known and hidden DataFrames. |
| `display_frame(frame, metadata)` | Adds row status and masks hidden targets for the UI. |

Metadata includes `id`, `source_id`, `source_kind`, `task`, `target`, story,
row counts, `hidden_indices`, `test_size`, `random_state`, and recorded source
failures. Classification is stratified only when the labels and split sizes make
that possible. A failure of every source raises `RuntimeError`; source attempts
are retained in `data/cache/hf_status.json`.

## EDA and session state

`profile_table`, `dtype_summary`, and `format_bytes` return display data.
`missing_figure`, `correlation_figure`, `distribution_figure`, `scatter_figure`,
and `target_figure` return Plotly figures. `correlation_figure` returns `None`
when fewer than two numeric columns are present. `pairplot_figure` returns the
matplotlib figure from a deterministic 400-row seaborn sample.

`initialize_state()` supplies default Streamlit values without overwriting
existing choices. `set_upload()` and `clear_upload()` manage the in-session
upload. `current_data()` returns:

```python
frame, target, problem, source, metadata = current_data()
```

`infer_problem()` chooses binary or multiclass for nonnumeric or
low-cardinality targets and regression otherwise. A user can override that
choice in the Data page.

## Model execution and results

```python
run_mitra(
    df_train,
    df_test,
    target,
    problem_type,
    fine_tune,
    eight_copies,
    fine_tune_steps=50,
    time_limit=None,
)
```

`run_mitra` requires the target in both inputs and identical ordered feature
columns. It accepts `regression`, `binary`, `multiclass`, or `classification`;
the last form resolves from the training target. Regression requires a numeric
target, binary requires exactly two classes, and multiclass requires three to
twenty classes.

Regression selects `autogluon/mitra-regressor-2`; classification selects
`autogluon/mitra-classifier-2`. It preflights that exact repository with
`HF_TOKEN`, so a missing or inaccessible token raises
`RuntimeError("set HF_TOKEN")`. The function applies the official regression-head patch when
needed, calls `TabularPredictor.fit`, and chooses `num_bag_folds=8` only when
`eight_copies=True`.

The returned dictionary includes run identity, paths, MITRA and baseline
metrics, confusion matrices when applicable, device, checkpoint, fitted model
names, mode, copy count, runtime, task, target, and row counts. See
[MITRA flags](MITRA_FLAGS.md) for the exact fit configuration.

`run_baseline(train, hidden, target, problem)` returns predictions, optional
probabilities, and optional binary positive class from one deterministic
HistGradientBoosting pipeline. `calculate_metrics(...)` returns RMSE, MAE, R²,
and conditionally MAPE for regression; it returns accuracy, F1, conditionally
ROC-AUC, and a labeled confusion-matrix payload for classification.

`device_info()` returns a `(cuda_available, device_name)` pair. Missing PyTorch
is treated as CPU-only. `load_cached_result(pointer_name)` reads only one of
the approved pointer files, while `load_last_result()` resolves `last_run.json`.

## Generated artifacts

Each run creates `data/runs/<UTC timestamp>/`:

| Artifact | Contents |
| --- | --- |
| `flags.json` | Checkpoint, device, toggles, bag folds, fitted model names, and time limit. |
| `metrics.json` | Complete serializable run result. |
| `predictions.csv` | Actual values plus MITRA and baseline predictions. |
| `leaderboard.csv` | AutoGluon validation leaderboard. |
| `run.log` | Timestamped runner messages. |
| `predictor/` | Persisted AutoGluon predictor. |
| `error.json` | Error type, message, and runtime when a run fails. |

`data/cache/last_run.json` points at the last completed run. Task-specific
smokes additionally create `last_run_reg.json`, `last_run_clf.json`,
`last_run_reg_ft.json`, and `last_run_clf_ft.json`. These are local generated
files, not versioned records.

## Synthetic gallery and presentation

`synthetic_prior_shapes(seed=42, n_samples=320)` returns DataFrames for the
Clusters, Hybrid SCM, Two moons, Rings, and Spirals illustrations. They are
deterministic generated examples, not training artifacts or internal Amazon
figures.

`apply_theme()`, `render_sidebar()`, and `render_top_bar()` own shared
presentation controls. `split_data()` is a small compatibility wrapper around
`split_from_metadata()` used by the EDA page.
