#!/usr/bin/env python3
"""Render the matched-token 300M training and evaluation figure."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "supplementary_code/results/corrected_300m_scaling.csv"
OUTPUT_PATH = REPO_ROOT / "supplementary_code/figures/fig_300m_loss_curves.png"

DENSE = "Dense SwiGLU"
ROUTED = "Residual Token-Routed"
COLORS = {DENSE: "#2563EB", ROUTED: "#DC2626"}
LABELS = {DENSE: "Dense", ROUTED: "Token-identity residual"}


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


def main() -> None:
    series = read_rows()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8.5,
        }
    )
    figure, (train_axis, gap_axis) = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.6),
        gridspec_kw={"width_ratios": [1.35, 1]},
        constrained_layout=True,
    )

    for model in (DENSE, ROUTED):
        rows = series[model]
        train_axis.plot(
            [row["tokens_b"] for row in rows],
            [row["train_loss"] for row in rows],
            color=COLORS[model],
            marker="o",
            markersize=3.8,
            linewidth=1.8,
            label=LABELS[model],
        )

    train_axis.axvline(0.776, color="#64748B", linestyle="--", linewidth=1)
    train_axis.annotate(
        "first recorded crossover\n0.776B tokens",
        xy=(0.776, 3.78),
        xytext=(1.25, 4.55),
        arrowprops={"arrowstyle": "-", "color": "#64748B", "linewidth": 0.8},
        color="#475569",
        fontsize=7.5,
    )
    train_axis.set_title("(a) Training stream")
    train_axis.set_xlabel("Tokens seen (billions)")
    train_axis.set_ylabel("Training NLL")
    train_axis.set_xlim(0, 8.1)
    train_axis.set_ylim(2.65, 7.05)
    train_axis.xaxis.set_major_locator(MultipleLocator(2))
    train_axis.grid(True, color="#E2E8F0", linewidth=0.7)
    train_axis.legend(frameon=False, loc="upper right")

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
    eval_gaps = [routed_eval[token] - dense_eval[token] for token in common_tokens]
    gap_axis.axhline(0, color="#334155", linewidth=1)
    gap_axis.plot(
        common_tokens,
        eval_gaps,
        color="#7C3AED",
        marker="o",
        markersize=4.5,
        linewidth=1.8,
    )
    gap_axis.fill_between(
        common_tokens,
        eval_gaps,
        0,
        where=[gap <= 0 for gap in eval_gaps],
        color="#DCFCE7",
        alpha=0.9,
        interpolate=True,
    )
    gap_axis.fill_between(
        common_tokens,
        eval_gaps,
        0,
        where=[gap > 0 for gap in eval_gaps],
        color="#FEE2E2",
        alpha=0.75,
        interpolate=True,
    )
    gap_axis.text(7.85, 0.025, "Dense lower", ha="right", va="center", color="#991B1B", fontsize=7.5)
    gap_axis.text(
        7.85,
        -0.025,
        "Token-identity residual lower",
        ha="right",
        va="center",
        color="#166534",
        fontsize=7.5,
    )
    gap_axis.set_title("(b) Fixed evaluation stream")
    gap_axis.set_xlabel("Tokens seen (billions)")
    gap_axis.set_ylabel("NLL gap (token-identity residual - dense)")
    gap_axis.set_xlim(0, 8.1)
    gap_axis.set_ylim(-0.06, 0.18)
    gap_axis.xaxis.set_major_locator(MultipleLocator(2))
    gap_axis.yaxis.set_major_locator(MultipleLocator(0.04))
    gap_axis.grid(True, color="#E2E8F0", linewidth=0.7)

    for axis in (train_axis, gap_axis):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#94A3B8")
        axis.spines["bottom"].set_color("#94A3B8")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
