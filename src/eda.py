from __future__ import annotations

import re
from typing import Any

import pandas as pd
import plotly.express as px
import seaborn as sns

PLOTLY_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "#111a2e",
    "plot_bgcolor": "#111a2e",
    "margin": {"l": 24, "r": 16, "t": 36, "b": 24},
}


def profile_table(frame: pd.DataFrame, target: str) -> dict[str, Any]:
    categorical = frame.select_dtypes(exclude="number").columns.tolist()
    high_cardinality = []
    for column in categorical:
        unique = frame[column].nunique(dropna=True)
        ratio = unique / max(len(frame), 1)
        if unique >= 50 or ratio >= 0.5:
            high_cardinality.append(column)

    normalized_target = _normalize_name(target)
    leakage = [
        str(column)
        for column in frame.columns
        if str(column) != target and _normalize_name(str(column)) == normalized_target
    ]
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "memory_bytes": int(frame.memory_usage(index=True, deep=True).sum()),
        "constant_columns": [
            str(column) for column in frame.columns if frame[column].nunique(dropna=False) <= 1
        ],
        "high_cardinality_categoricals": high_cardinality,
        "leakage_suspects": leakage,
    }


def dtype_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = frame.dtypes.astype(str).rename("dtype").to_frame()
    summary["non-null"] = frame.notna().sum()
    summary["unique"] = frame.nunique(dropna=True)
    summary["memory"] = [
        _format_bytes(frame[column].memory_usage(index=False, deep=True)) for column in frame
    ]
    return summary


def missing_figure(frame: pd.DataFrame):
    missing = frame.isna().sum().sort_values(ascending=False).rename("Missing").reset_index()
    missing.columns = ["Column", "Missing"]
    figure = px.bar(
        missing,
        x="Column",
        y="Missing",
        color="Missing",
        color_continuous_scale=["#51448a", "#ad98ff"],
        title="Missing values by column",
    )
    figure.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
    return figure


def correlation_figure(frame: pd.DataFrame):
    numeric = frame.select_dtypes("number")
    if numeric.shape[1] < 2:
        return None
    figure = px.imshow(
        numeric.corr(),
        text_auto=".2f",
        color_continuous_scale="Purples",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Numeric correlation",
    )
    figure.update_layout(**PLOTLY_LAYOUT)
    return figure


def distribution_figure(frame: pd.DataFrame, column: str, kind: str):
    if kind == "Histogram":
        figure = px.histogram(frame, x=column, color_discrete_sequence=["#8064f4"])
    elif kind == "Box":
        figure = px.box(frame, y=column, color_discrete_sequence=["#8064f4"])
    elif kind == "Violin":
        figure = px.violin(
            frame,
            y=column,
            box=True,
            points=False,
            color_discrete_sequence=["#8064f4"],
        )
    else:
        raise ValueError(f"Unknown distribution plot: {kind}")
    figure.update_layout(**PLOTLY_LAYOUT, title=f"{kind}: {column}")
    return figure


def scatter_figure(frame: pd.DataFrame, x: str, y: str, target: str, task: str):
    plotted = frame if len(frame) <= 5_000 else frame.sample(5_000, random_state=42)
    color = target if task in {"binary", "multiclass"} and target not in {x, y} else None
    figure = px.scatter(
        plotted,
        x=x,
        y=y,
        color=color,
        opacity=0.65,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title=f"{x} vs {y}",
    )
    figure.update_layout(**PLOTLY_LAYOUT)
    return figure


def target_figure(frame: pd.DataFrame, target: str, task: str):
    if task in {"binary", "multiclass"}:
        counts = frame[target].value_counts(dropna=False).rename("Rows").reset_index()
        counts.columns = [target, "Rows"]
        figure = px.bar(
            counts,
            x=target,
            y="Rows",
            color=target,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            title=f"Target balance: {target}",
        )
        figure.update_layout(showlegend=False)
    else:
        figure = px.histogram(
            frame,
            x=target,
            marginal="box",
            color_discrete_sequence=["#8064f4"],
            title=f"Target distribution: {target}",
        )
    figure.update_layout(**PLOTLY_LAYOUT)
    return figure


def pairplot_figure(
    frame: pd.DataFrame,
    columns: list[str],
    target: str,
    task: str,
):
    selected = list(dict.fromkeys(columns))[:5]
    if len(selected) < 2:
        raise ValueError("Pairplot needs at least two numeric columns")
    plotted = frame[selected].dropna()
    if len(plotted) > 400:
        plotted = plotted.sample(400, random_state=42)
    hue = target if task in {"binary", "multiclass"} and target in selected else None
    grid = sns.pairplot(
        plotted,
        vars=[column for column in selected if column != hue],
        hue=hue,
        corner=True,
        diag_kind="hist",
        plot_kws={"alpha": 0.55, "s": 24},
    )
    grid.fig.patch.set_facecolor("#111a2e")
    grid.fig.suptitle("Pairplot sample · max 400 rows", y=1.02, color="#eef0ff")
    return grid.fig


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def format_bytes(value: int) -> str:
    return _format_bytes(value)
