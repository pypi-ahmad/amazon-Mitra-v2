from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from src.eda import (
    correlation_figure,
    distribution_figure,
    missing_figure,
    pairplot_figure,
    scatter_figure,
    target_figure,
)
from src.load import load_sample

OUTPUT = ROOT / "data" / "cache" / "eda_ok.json"


def main() -> None:
    frame, metadata = load_sample("Houses")
    target = metadata["target"]
    numeric = frame.select_dtypes("number").columns.tolist()
    feature = next(column for column in numeric if column != target)
    saved = 0

    with tempfile.TemporaryDirectory(prefix="glass-box-eda-") as temp:
        plot_dir = Path(temp)
        plotly_figures = {
            "missing": missing_figure(frame),
            "correlation": correlation_figure(frame),
            "histogram": distribution_figure(frame, feature, "Histogram"),
            "box": distribution_figure(frame, feature, "Box"),
            "violin": distribution_figure(frame, feature, "Violin"),
            "scatter": scatter_figure(frame, feature, target, target, metadata["task"]),
            "target": target_figure(frame, target, metadata["task"]),
        }
        for name, figure in plotly_figures.items():
            assert figure is not None, f"{name} figure was not created"
            path = plot_dir / f"{name}.html"
            figure.write_html(path, include_plotlyjs="cdn")
            assert path.stat().st_size > 0, f"{name} plot is empty"
            saved += 1

        pair_columns = [column for column in numeric if column != target][:4] + [target]
        pair_figure = pairplot_figure(frame, pair_columns, target, metadata["task"])
        pair_path = plot_dir / "pairplot.png"
        pair_figure.savefig(pair_path, dpi=100, bbox_inches="tight")
        assert pair_path.stat().st_size > 0, "pairplot is empty"
        saved += 1
        plt.close(pair_figure)

    assert saved == 8, f"Expected 8 plots, saved {saved}"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"n_plots_saved": saved}, indent=2), encoding="utf-8")
    print(f"eda smoke ok: {saved} plots")
    print(OUTPUT)


if __name__ == "__main__":
    main()
