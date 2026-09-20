# MITRA installation and flags

This project follows the released Mitra-v2 model cards and AutoGluon's
foundation-model API. The official cards currently require AutoGluon with the
Mitra extra, TabArena, and the separately installed `mitra-finetune` package.

```text
autogluon.tabular[mitra]>=1.6,<1.7
tabarena>=0.1
```

The Hub Git server can reject pip's partial clone. `run.cmd` therefore clones
`autogluon/mitra-finetune`, checks out revision
`b4701e8148dc33b00ed15d7086ff59816957cde4` (v0.3.0), and installs that local
checkout. Flash attention remains optional and no CUDA Torch wheel is pinned.

## Head selection

| Problem type | Checkpoint |
| --- | --- |
| `binary`, `multiclass` | `autogluon/mitra-classifier-2` |
| `regression` | `autogluon/mitra-regressor-2` |

The regressor has a 1,000-bin distributional head. Before constructing its
predictor, Glass Box reads the checkpoint `dim_output` and calls the installed
official `mitra_finetune.patches.install_reg_ce_patches` implementation.

## Exact AutoGluon configuration

The public runner accepts already separated tables:

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

The generic `classification` problem type is resolved to `binary` or
`multiclass` from the training target. Both tables must include the target.

The MITRA hyperparameter dictionary is built as follows:

```python
mitra_hyperparameters = {
    "hf_model": checkpoint,
    "fine_tune": config.fine_tune,
}
if config.fine_tune:
    mitra_hyperparameters["fine_tune_steps"] = config.fine_tune_steps
```

It is passed to `TabularPredictor.fit` with these predictor-level controls:

```python
predictor.fit(
    train_data=train,
    time_limit=config.time_limit,
    hyperparameters={"MITRA": mitra_hyperparameters},
    num_bag_folds=8 if config.eight_copies else 0,
    num_bag_sets=1,
    num_stack_levels=0,
    dynamic_stacking=False,
    fit_weighted_ensemble=False,
    num_gpus=1 if torch.cuda.is_available() else 0,
    ag_args_fit={"ag.max_memory_usage_ratio": 1.2},
)
```

`num_bag_folds` belongs to AutoGluon bagging; it is not a MITRA model
hyperparameter.

## Toggle matrix

| Fine-tune | Eight copies | Effective behavior |
| --- | --- | --- |
| Off | Off | Zero-shot, one model, no `fine_tune_steps` key |
| Off | On | Zero-shot with eight AutoGluon bag children |
| On | Off | One fine-tuned model, default 50 steps |
| On | On | Eight fine-tuned bag children, default 50 steps each |

The released benchmark recipe assumes CUDA and eight children. Glass Box also
allows CPU and single-model runs for tutorial access, while warning that CPU
fine-tuning can be slow. The app does not set the optional benchmark protocol
environment variables `MITRA_FT_BUDGET_S` or `MITRA_BAG_SALVAGE`.

Every run preflights the selected v2 repository with `HF_TOKEN`. A missing or
authentication-dependent checkpoint download fails with `set HF_TOKEN`; there
is no fallback to a v1 checkpoint.

## Primary references

- [Classifier model card](https://huggingface.co/autogluon/mitra-classifier-2)
- [Regressor model card](https://huggingface.co/autogluon/mitra-regressor-2)
- [Fine-tune package card](https://huggingface.co/autogluon/mitra-finetune)
- [AutoGluon foundation-model tutorial](https://auto.gluon.ai/stable/tutorials/tabular/tabular-foundational-models.html)
