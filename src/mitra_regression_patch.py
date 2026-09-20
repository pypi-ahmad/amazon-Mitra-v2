"""Compatibility for the official MITRA-v2 1,000-bin regression checkpoint.

Adapted from ``mitra_finetune.patches.install_reg_ce_patches`` at official
revision b4701e8148dc33b00ed15d7086ff59816957cde4. It activates AutoGluon's
existing cross-entropy regression path and changes only the installed
argmax bin decode to the official softmax-weighted mean.
"""

from __future__ import annotations

import inspect
import textwrap


def install_regression_patch(n_bins: int) -> None:
    import autogluon.tabular.models.mitra._internal.core.trainer_finetune as trainer_module
    import autogluon.tabular.models.mitra.sklearn_interface as sklearn_interface
    from autogluon.tabular.models.mitra._internal.config.enums import LossName, Task

    if getattr(sklearn_interface, "_glass_box_reg_ce_patched", False):
        return

    original_create_config = sklearn_interface.MitraBase._create_config

    def create_config(self, task, dim_output, time_limit=None):
        config, model_class = original_create_config(self, task, dim_output, time_limit)
        is_regression = config.task == Task.REGRESSION or str(task).lower().endswith(
            "regression"
        )
        if is_regression:
            config.task = Task.REGRESSION
            config.hyperparams["regression_loss"] = LossName.CROSS_ENTROPY
            config.hyperparams["dim_output"] = n_bins
        return config, model_class

    sklearn_interface.MitraBase._create_config = create_config
    trainer_module.TrainerFinetune.evaluate = _mean_decode_source(
        trainer_module.TrainerFinetune.evaluate,
        numpy_variant=False,
    )
    trainer_module._mitra_finetune_distribution_sink = None
    trainer_module.TrainerFinetune.predict = _mean_decode_source(
        trainer_module.TrainerFinetune.predict,
        numpy_variant=True,
    )
    sklearn_interface._glass_box_reg_ce_patched = True


def _mean_decode_source(function, *, numpy_variant: bool):
    source = textwrap.dedent(inspect.getsource(function))
    lines = source.split("\n")

    if numpy_variant:
        argmax_line = "y_hat = np.argmax(y_hat, axis=-1)"
        bins_line = "y_hat = (self.bins[y_hat] + self.bin_width / 2).cpu().numpy()"
        replacement = (
            "_logits = torch.as_tensor(y_hat).float()\n"
            "if not torch.isfinite(_logits).all():\n"
            "    _logits = torch.nan_to_num(_logits, nan=0.0, posinf=1e4, neginf=-1e4)\n"
            "_centers = (self.bins[:-1] + self.bin_width / 2).cpu()\n"
            "_probs = torch.softmax(_logits, dim=-1)\n"
            "if _mitra_finetune_distribution_sink is not None:\n"
            "    _mitra_finetune_distribution_sink(self, _probs)\n"
            "y_hat = (_probs * _centers).sum(dim=-1).numpy()"
        )
    else:
        argmax_line = "y_hat = torch.argmax(y_hat, dim=-1)"
        bins_line = "y_hat = self.bins[y_hat] + self.bin_width / 2"
        replacement = (
            "_values = y_hat.float()\n"
            "if not torch.isfinite(_values).all():\n"
            "    _values = torch.nan_to_num(_values, nan=0.0, posinf=1e4, neginf=-1e4)\n"
            "_centers = self.bins[:-1] + self.bin_width / 2\n"
            "y_hat = (torch.softmax(_values, dim=-1) * _centers.to(_values.device)).sum(dim=-1)"
        )

    index = next(
        (position for position, line in enumerate(lines) if line.strip() == argmax_line),
        None,
    )
    if index is None or lines[index + 1].strip() != bins_line:
        raise RuntimeError(
            "AutoGluon MITRA regression bin decoding changed; "
            f"the compatibility patch cannot safely update {function.__qualname__}"
        )
    indentation = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
    replacement_lines = [
        indentation + line if line else line for line in replacement.split("\n")
    ]
    lines[index : index + 2] = replacement_lines

    namespace: dict = {}
    exec(
        compile(
            "\n".join(lines),
            f"<glass-box mitra regression decode {function.__name__}>",
            "exec",
        ),
        function.__globals__,
        namespace,
    )
    return namespace[function.__name__]
