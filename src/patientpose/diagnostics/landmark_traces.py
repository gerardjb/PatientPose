from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class TraceSpec:
    label: str
    df: pd.DataFrame
    y_column: str = "value"
    x_column: str = "timestamp_ms"
    color: Any | None = None
    linestyle: str = "-"
    marker: str | None = None
    alpha: float = 1.0


def _resolve_x(trace: TraceSpec) -> tuple[pd.Series, str]:
    if trace.x_column not in trace.df.columns:
        raise KeyError(f"Trace '{trace.label}' does not contain x column '{trace.x_column}'.")
    x = trace.df[trace.x_column]
    if trace.x_column == "timestamp_ms":
        return x.astype(float) / 1000.0, "Time (s)"
    return x, trace.x_column


def _plot_trace(axis: plt.Axes, trace: TraceSpec) -> str:
    if trace.y_column not in trace.df.columns:
        raise KeyError(f"Trace '{trace.label}' does not contain y column '{trace.y_column}'.")
    x, x_label = _resolve_x(trace)
    axis.plot(
        x,
        trace.df[trace.y_column],
        label=trace.label,
        color=trace.color,
        linestyle=trace.linestyle,
        marker=trace.marker,
        alpha=trace.alpha,
    )
    return x_label


def plot_landmark_components(
    series_map: dict[str, pd.DataFrame],
    output_path: Path,
    *,
    title: str,
    components: Sequence[str] = ("x", "y"),
    invert_components: Sequence[str] = (),
) -> None:
    if not series_map:
        raise ValueError("series_map must contain at least one landmark series.")

    component_names = [str(component) for component in components]
    fig, axes = plt.subplots(len(component_names), 1, figsize=(10, 2.8 * len(component_names)), sharex=True)
    if len(component_names) == 1:
        axes = [axes]

    resolved_x_label = "Time (s)"
    for axis, component in zip(axes, component_names):
        for label, series_df in series_map.items():
            trace = TraceSpec(label=label, df=series_df, y_column=component)
            resolved_x_label = _plot_trace(axis, trace)
        axis.set_ylabel(component)
        axis.grid(alpha=0.3)
        axis.legend(loc="upper right", fontsize=8, frameon=False)
        if component in invert_components:
            axis.invert_yaxis()
    axes[-1].set_xlabel(resolved_x_label)
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_trace(
    traces: Sequence[TraceSpec],
    output_path: Path,
    *,
    title: str,
    y_label: str = "value",
    x_label: str | None = None,
    invert_y: bool = False,
) -> None:
    if not traces:
        raise ValueError("traces must contain at least one TraceSpec.")

    fig, ax = plt.subplots(figsize=(10, 3.5))
    resolved_x_label = "Time (s)"
    for trace in traces:
        resolved_x_label = _plot_trace(ax, trace)
    ax.set_xlabel(x_label if x_label is not None else resolved_x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    if invert_y:
        ax.invert_yaxis()
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_panels(
    panels: Sequence[tuple[str, Sequence[TraceSpec]]],
    output_path: Path,
    *,
    title: str,
    y_labels: Sequence[str] | None = None,
    x_label: str | None = None,
) -> None:
    if not panels:
        raise ValueError("panels must contain at least one panel definition.")

    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.8 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]

    resolved_x_label = "Time (s)"
    for idx, ((panel_title, traces), axis) in enumerate(zip(panels, axes)):
        if not traces:
            raise ValueError(f"Panel '{panel_title}' must contain at least one TraceSpec.")
        for trace in traces:
            resolved_x_label = _plot_trace(axis, trace)
        axis.set_title(panel_title, fontsize=10)
        axis.set_ylabel(y_labels[idx] if y_labels is not None and idx < len(y_labels) else "value")
        axis.grid(alpha=0.3)
        axis.legend(loc="upper right", fontsize=8, frameon=False)
    axes[-1].set_xlabel(x_label if x_label is not None else resolved_x_label)
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pairwise_landmark_comparison(
    series_a: pd.DataFrame,
    series_b: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    components: Sequence[str] = ("x", "y"),
    label_a: str = "A",
    label_b: str = "B",
    invert_components: Sequence[str] = (),
) -> None:
    plot_landmark_components(
        {
            label_a: series_a,
            label_b: series_b,
        },
        output_path,
        title=title,
        components=components,
        invert_components=invert_components,
    )


__all__ = [
    "TraceSpec",
    "plot_landmark_components",
    "plot_metric_panels",
    "plot_metric_trace",
    "plot_pairwise_landmark_comparison",
]
