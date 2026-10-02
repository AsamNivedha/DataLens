"""Matplotlib / Seaborn chart builders for DataLens."""

import io

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, safe for servers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.analysis import pretty

PRIMARY = "#2F5D9E"
ACCENT = "#C0392B"


def describe_bar(cat: str, num, agg: str):
    """Return (chart title, value-axis label) for the chosen selections."""
    if agg == "Count":
        return f"Number of Records by {pretty(cat)}", "Number of records"
    prefix = {"Mean": "Average", "Median": "Median", "Sum": "Total"}[agg]
    label = f"{prefix} {pretty(num)}"
    return f"{label} by {pretty(cat)}", label


def bar_chart(values: pd.Series, title: str, x_label: str, y_label: str):
    labels = [str(i) for i in values.index]
    horizontal = len(labels) > 6 or max(len(s) for s in labels) > 12
    whole_numbers = bool((values % 1 == 0).all())
    fmt = "%.0f" if whole_numbers else "%.2f"

    if horizontal:
        fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45 * len(labels) + 1.5)))
        bars = ax.barh(labels[::-1], values.values[::-1], color=PRIMARY)  # largest on top
        ax.set_xlabel(y_label)
        ax.set_ylabel(x_label)
        ax.margins(x=0.12)
        ax.grid(axis="x", alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(labels, values.values, color=PRIMARY)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.margins(y=0.12)
        ax.grid(axis="y", alpha=0.3)
        if max(len(s) for s in labels) > 8:
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    ax.bar_label(bars, fmt=fmt, padding=3, fontsize=9)
    ax.set_title(title, fontweight="bold")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def scatter_plot(df: pd.DataFrame, x: str, y: str, group=None, trend: bool = True):
    columns = [x, y] + ([group] if group else [])
    data = df[columns].dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    if group:
        for name, subset in data.groupby(group, observed=True):
            ax.scatter(subset[x], subset[y], alpha=0.7, s=45, label=str(name),
                       edgecolor="white", linewidth=0.5)
    else:
        ax.scatter(data[x], data[y], alpha=0.7, s=45, color=PRIMARY,
                   edgecolor="white", linewidth=0.5)

    draw_legend = bool(group)
    if trend and len(data) >= 3 and data[x].nunique() > 1:
        slope, intercept = np.polyfit(data[x], data[y], 1)
        xs = np.linspace(data[x].min(), data[x].max(), 100)
        ax.plot(xs, slope * xs + intercept, color=ACCENT, linestyle="--",
                linewidth=1.8, label="Trend line")
        draw_legend = True

    if draw_legend:
        ax.legend(title=pretty(group) if group else None, frameon=False)
    ax.set_title(f"{pretty(x)} vs {pretty(y)}", fontweight="bold")
    ax.set_xlabel(pretty(x))
    ax.set_ylabel(pretty(y))
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def heatmap(corr: pd.DataFrame):
    n = len(corr)
    size = max(5.0, min(0.9 * n + 2, 12.0))
    fig, ax = plt.subplots(figsize=(size * 1.1, size))
    labels = [pretty(c) for c in corr.columns]
    sns.heatmap(
        corr,
        annot=n <= 15,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson correlation"},
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 9 if n <= 8 else 7},
        ax=ax,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    ax.set_title("Correlation Heatmap of Numerical Variables", fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def fig_to_png(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    return buffer.getvalue()
