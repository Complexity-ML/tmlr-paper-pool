#!/usr/bin/env python3
"""Render the matched-token 300M training and diagnostic evaluation figure."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "supplementary_code/results/corrected_300m_scaling.csv"
OUTPUT_PDF = REPO_ROOT / "supplementary_code/figures/fig_300m_loss_curves.pdf"
OUTPUT_PNG = REPO_ROOT / "supplementary_code/figures/fig_300m_loss_curves.png"

DENSE = "Dense SwiGLU"
ROUTED = "Residual Token-Routed"
COLORS = {DENSE: "#2474A6", ROUTED: "#D55E00"}
LABELS = {DENSE: "Dense", ROUTED: "Token-identity residual"}
INK = "#18212F"
MUTED = "#5F6B7A"
GRID = "#D9E0E8"
TEAL = "#16847A"
TEAL_LIGHT = "#E4F3F0"
WARM_LIGHT = "#FBEEE8"


def read_rows() -> dict[str, list[dict[str, float]]]:
    series: dict[str, list[dict[str, float]]] = {DENSE: [], ROUTED: []}
    with DATA_PATH.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["model"] not in series or not row["step"].isdigit():
                continue
            series[row["model"]].append(
                {
                    "tokens_b": float(row["tokens_seen_m"]) / 1000.0,
                    "train_loss": float(row["train_loss"]),
                    "eval_loss": float(row["eval_loss"]) if row["eval_loss"] else float("nan"),
                }
            )
    return series


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="both", color=GRID, linewidth=0.7, alpha=0.8)
    axis.tick_params(colors=MUTED, labelsize=7.2, length=3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#9AA7B5")
    axis.spines["bottom"].set_color("#9AA7B5")
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.xaxis.label.set_color(INK)
    axis.yaxis.label.set_color(INK)


def plot_training(axis: plt.Axes, series: dict[str, list[dict[str, float]]]) -> None:
    for model in (DENSE, ROUTED):
        rows = series[model]
        axis.plot(
            [row["tokens_b"] for row in rows],
            [row["train_loss"] for row in rows],
            color=COLORS[model],
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.1,
            markersize=3.5,
            linewidth=1.9,
            label=LABELS[model],
            zorder=3,
        )

    axis.set_title("Training trajectory", loc="left", color=INK, pad=8, fontweight="bold")
    axis.set_xlabel("Training tokens (billions)")
    axis.set_ylabel("Training NLL")
    axis.set_xlim(0, 8.15)
    axis.set_ylim(2.65, 7.05)
    axis.xaxis.set_major_locator(MultipleLocator(2))
    style_axis(axis)
    axis.legend(
        frameon=False,
        loc="upper right",
        handlelength=2.2,
        borderaxespad=0.5,
        labelcolor=INK,
    )

    axis.axvline(0.776, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.9, zorder=1)
    axis.text(
        0.88,
        4.25,
        "first recorded\ncrossover\n0.776B tokens",
        color=MUTED,
        fontsize=6.6,
        va="center",
        ha="left",
    )

    inset = axis.inset_axes([0.50, 0.31, 0.46, 0.34])
    inset.set_facecolor("#F8FAFC")
    for model in (DENSE, ROUTED):
        rows = [row for row in series[model] if row["tokens_b"] >= 0.734]
        inset.plot(
            [row["tokens_b"] for row in rows],
            [row["train_loss"] for row in rows],
            color=COLORS[model],
            marker="o",
            markerfacecolor="white",
            markeredgewidth=0.9,
            markersize=2.6,
            linewidth=1.35,
            zorder=3,
        )
    inset.axvline(0.776, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.7)
    inset.set_xlim(0.7, 8.1)
    inset.set_ylim(2.83, 3.92)
    inset.set_xticks([1, 4, 8])
    inset.set_yticks([3.0, 3.4, 3.8])
    inset.tick_params(labelsize=5.8, colors=MUTED, length=2)
    inset.grid(color=GRID, linewidth=0.55)
    for spine in inset.spines.values():
        spine.set_color("#B8C3CF")
        spine.set_linewidth(0.65)
    inset.text(
        0.03,
        0.92,
        "after crossover",
        transform=inset.transAxes,
        color=INK,
        fontsize=6.1,
        fontweight="bold",
        va="top",
    )


def plot_gap(axis: plt.Axes, series: dict[str, list[dict[str, float]]]) -> None:
    dense_eval = {
        row["tokens_b"]: row["eval_loss"]
        for row in series[DENSE]
        if row["eval_loss"] == row["eval_loss"]
    }
    routed_eval = {
        row["tokens_b"]: row["eval_loss"]
        for row in series[ROUTED]
        if row["eval_loss"] == row["eval_loss"]
    }
    common_tokens = sorted(dense_eval.keys() & routed_eval.keys())
    gaps = [routed_eval[token] - dense_eval[token] for token in common_tokens]

    axis.axhspan(-0.06, 0, color=TEAL_LIGHT, zorder=0)
    axis.axhspan(0, 0.18, color=WARM_LIGHT, zorder=0)
    axis.axhline(0, color=INK, linewidth=1.0, zorder=2)
    axis.plot(
        common_tokens,
        gaps,
        color=TEAL,
        marker="o",
        markerfacecolor="white",
        markeredgecolor=TEAL,
        markeredgewidth=1.2,
        markersize=4.2,
        linewidth=2.0,
        zorder=3,
    )
    axis.scatter(
        [common_tokens[-1]],
        [gaps[-1]],
        s=35,
        color=TEAL,
        edgecolor="white",
        linewidth=0.9,
        zorder=4,
    )

    axis.set_title(
        "Fixed diagnostic stream",
        loc="left",
        color=INK,
        pad=8,
        fontweight="bold",
    )
    axis.set_xlabel("Training tokens (billions)")
    axis.set_ylabel("NLL difference (token-routed - dense)")
    axis.set_xlim(0, 8.15)
    axis.set_ylim(-0.06, 0.18)
    axis.xaxis.set_major_locator(MultipleLocator(2))
    axis.yaxis.set_major_locator(MultipleLocator(0.04))
    style_axis(axis)

    axis.text(
        7.82,
        0.155,
        "dense lower",
        ha="right",
        va="center",
        color="#9A4A2C",
        fontsize=6.8,
        fontweight="bold",
    )
    axis.text(
        7.82,
        -0.047,
        "token-routed lower",
        ha="right",
        va="center",
        color=TEAL,
        fontsize=6.8,
        fontweight="bold",
    )
    axis.annotate(
        "-0.0153 NLL\nat 7.864B tokens",
        xy=(common_tokens[-1], gaps[-1]),
        xytext=(4.9, 0.035),
        color=INK,
        fontsize=7.2,
        fontweight="bold",
        ha="left",
        va="center",
        arrowprops={
            "arrowstyle": "-",
            "color": TEAL,
            "linewidth": 1.0,
            "connectionstyle": "arc3,rad=-0.16",
        },
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#B8C3CF",
            "linewidth": 0.8,
        },
        zorder=5,
    )


def main() -> None:
    series = read_rows()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.25),
        gridspec_kw={"width_ratios": [1.2, 1]},
        constrained_layout=True,
    )
    figure.patch.set_facecolor("white")
    figure.suptitle(
        "Matched 306.5M models at the same 8B-token training budget",
        x=0.055,
        y=1.02,
        ha="left",
        color=INK,
        fontsize=10.2,
        fontweight="bold",
    )

    plot_training(axes[0], series)
    plot_gap(axes[1], series)
    axes[0].text(
        -0.10,
        1.04,
        "a",
        transform=axes[0].transAxes,
        color=INK,
        fontsize=10,
        fontweight="bold",
        va="bottom",
    )
    axes[1].text(
        -0.12,
        1.04,
        "b",
        transform=axes[1].transAxes,
        color=INK,
        fontsize=10,
        fontweight="bold",
        va="bottom",
    )

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    figure.savefig(
        OUTPUT_PNG,
        dpi=260,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
