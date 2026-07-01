"""Reusable plotting helpers for the Credit Risk MVP."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.container import BarContainer


def annotate_bars(
    ax: Axes,
    bars: BarContainer | Sequence,
    *,
    fmt: str = "{:.3f}",
    offset_factor: float = 0.015,
    fontsize: int = 8,
    fontweight: str = "bold",
    min_height: float | None = None,
    ha: str = "center",
    va: str = "bottom",
) -> None:
    """Add value labels on top of vertical bars.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes containing the bars.
    bars : BarContainer or iterable of patches
        The bar patches to annotate.
    fmt : str
        Format string for the label value.
    offset_factor : float
        Vertical offset as a fraction of the axes y-range.
    fontsize : int
        Font size for annotations.
    fontweight : str
        Font weight for annotations.
    min_height : float or None
        If set, only annotate bars taller than this value.
    ha : str
        Horizontal alignment of text.
    va : str
        Vertical alignment of text.
    """
    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * offset_factor

    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        if min_height is not None and height <= min_height:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + offset,
            fmt.format(height),
            ha=ha,
            va=va,
            fontsize=fontsize,
            fontweight=fontweight,
        )


def annotate_hbars(
    ax: Axes,
    values: Sequence[float],
    *,
    fmt: str = "{:.3f}",
    offset: float = 0.008,
    fontsize: int = 9,
    fontweight: str = "bold",
) -> None:
    """Add value labels to the right of horizontal bar entries.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes.
    values : sequence of float
        The numeric values associated with each bar position.
    fmt : str
        Format string for the label.
    offset : float
        Horizontal offset from the bar end.
    fontsize : int
        Font size.
    fontweight : str
        Font weight.
    """
    for i, v in enumerate(values):
        ha_align = "left" if v >= 0 else "right"
        sign = 1 if v >= 0 else -1
        ax.text(
            v + sign * offset,
            i,
            fmt.format(v),
            va="center",
            ha=ha_align,
            fontsize=fontsize,
            fontweight=fontweight,
        )


def style_axis(
    ax: Axes,
    *,
    grid_axis: str = "y",
    grid_linestyle: str = "--",
    grid_alpha: float = 0.5,
    grid_zorder: int = 0,
    despine: bool = True,
) -> None:
    """Apply consistent grid and despine styling to an axis.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes to style.
    grid_axis : str
        Which axis to add gridlines to ('x', 'y', or 'both').
    grid_linestyle : str
        Linestyle for gridlines.
    grid_alpha : float
        Alpha for gridlines.
    grid_zorder : int
        Zorder for gridlines.
    despine : bool
        Whether to call sns.despine.
    """
    if grid_axis in ("y", "both"):
        ax.yaxis.grid(True, linestyle=grid_linestyle, alpha=grid_alpha, zorder=grid_zorder)
    if grid_axis in ("x", "both"):
        ax.xaxis.grid(True, linestyle=grid_linestyle, alpha=grid_alpha, zorder=grid_zorder)
    if despine:
        sns.despine(ax=ax)


def plot_grouped_bar_comparison(
    labels: Sequence[str],
    group_a_values: Sequence[float],
    group_b_values: Sequence[float],
    *,
    group_a_label: str = "Group A",
    group_b_label: str = "Group B",
    color_a: str = "#0173B2",
    color_b: str = "#DE8F05",
    ylabel: str = "Value",
    title: str = "",
    figsize: tuple[float, float] = (10, 6),
    ylim: tuple[float, float] = (0, 1.05),
    rotate_labels: float = 0,
    ha_labels: str = "center",
    annotate: bool = True,
    fmt: str = "{:.3f}",
) -> tuple[plt.Figure, Axes]:
    """Create a grouped bar chart comparing two groups across categories.

    Parameters
    ----------
    labels : sequence of str
        Category labels for the x-axis.
    group_a_values, group_b_values : sequence of float
        Values for each group.
    group_a_label, group_b_label : str
        Legend labels for each group.
    color_a, color_b : str
        Bar colors.
    ylabel, title : str
        Axis labels and title.
    figsize : tuple
        Figure size.
    ylim : tuple
        Y-axis limits.
    rotate_labels : float
        X-tick label rotation in degrees.
    ha_labels : str
        Horizontal alignment of x-tick labels.
    annotate : bool
        Whether to add value annotations on bars.
    fmt : str
        Format string for annotations.

    Returns
    -------
    fig, ax : Figure and Axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    width = 0.35

    bars_a = ax.bar(
        x - width / 2,
        group_a_values,
        width,
        label=group_a_label,
        color=color_a,
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )
    bars_b = ax.bar(
        x + width / 2,
        group_b_values,
        width,
        label=group_b_label,
        color=color_b,
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )

    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate_labels, ha=ha_labels)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(*ylim)
    style_axis(ax)

    if annotate:
        annotate_bars(ax, bars_a, fmt=fmt)
        annotate_bars(ax, bars_b, fmt=fmt)

    plt.tight_layout()
    return fig, ax


def annotate_score_box(
    ax: Axes,
    label: str,
    value: float,
    *,
    xy: tuple[float, float] = (0.95, 0.05),
    fontsize: int = 11,
    fmt: str = "{:.4f}",
) -> None:
    """Add an annotated score box to a plot (e.g., AUC value).

    Parameters
    ----------
    ax : Axes
        The axes to annotate.
    label : str
        Metric label (e.g., "AUC").
    value : float
        The numeric value to display.
    xy : tuple
        Position in axes fraction coordinates.
    fontsize : int
        Font size.
    fmt : str
        Format string for the value.
    """
    ax.annotate(
        f"{label} = {fmt.format(value)}",
        xy=xy,
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="#333333",
            alpha=0.9,
        ),
    )
