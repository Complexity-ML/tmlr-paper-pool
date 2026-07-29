#!/usr/bin/env python3
"""Render the checkpoint-verified architecture as a publication-quality figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR.parent / "figures"
PDF_PATH = FIGURE_DIR / "architecture_complexity_deep.pdf"
PNG_PATH = FIGURE_DIR / "architecture_complexity_deep.png"

INK = "#18212F"
MUTED = "#5F6B7A"
LINE = "#C8D1DC"
PANEL = "#F7F9FC"
BLUE = "#2474A6"
BLUE_LIGHT = "#E8F2F8"
TEAL = "#16847A"
TEAL_LIGHT = "#E4F3F0"
PURPLE = "#7052B8"
PURPLE_LIGHT = "#EEE9F8"
AMBER = "#B87412"
AMBER_LIGHT = "#FBF1DE"


def rounded_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str = "white",
    edgecolor: str = LINE,
    linewidth: float = 1.0,
    radius: float = 0.018,
    zorder: int = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    axis.add_patch(patch)
    return patch


def arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    linewidth: float = 1.2,
    dashed: bool = False,
    connectionstyle: str = "arc3,rad=0",
    zorder: int = 1,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=linewidth,
            linestyle=(0, (3, 2)) if dashed else "solid",
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=1,
            shrinkB=1,
            zorder=zorder,
        )
    )


def panel_label(axis: plt.Axes, x: float, y: float, label: str, title: str) -> None:
    axis.text(x, y, label, color=INK, fontsize=10, fontweight="bold", va="top")
    axis.text(x + 0.025, y, title, color=INK, fontsize=10, fontweight="bold", va="top")


def label_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str,
    *,
    facecolor: str = "white",
    edgecolor: str = LINE,
    title_color: str = INK,
    align: str = "left",
) -> None:
    rounded_box(
        axis,
        x,
        y,
        width,
        height,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.05,
    )
    text_x = x + width / 2 if align == "center" else x + 0.014
    horizontal = "center" if align == "center" else "left"
    axis.text(
        text_x,
        y + height * 0.62,
        title,
        ha=horizontal,
        va="center",
        color=title_color,
        fontsize=8.3,
        fontweight="bold",
        zorder=3,
    )
    axis.text(
        text_x,
        y + height * 0.29,
        subtitle,
        ha=horizontal,
        va="center",
        color=MUTED,
        fontsize=6.8,
        zorder=3,
    )


def draw_architecture(axis: plt.Axes) -> None:
    panel_label(axis, 0.015, 0.97, "a", "GQA transformer with fixed residual routing")

    label_box(
        axis,
        0.145,
        0.78,
        0.25,
        0.115,
        r"Contextual hidden state  $\mathbf{x}_l$",
        "decoder-only backbone  |  causal GQA",
        facecolor=PANEL,
        align="center",
    )
    label_box(
        axis,
        0.505,
        0.78,
        0.12,
        0.115,
        r"Token ID  $t$",
        "lexical identity",
        facecolor=PURPLE_LIGHT,
        edgecolor=PURPLE,
        title_color=PURPLE,
        align="center",
    )

    label_box(
        axis,
        0.055,
        0.47,
        0.22,
        0.15,
        "Shared SwiGLU",
        r"width 3,840  |  gate $g_l^s$",
        facecolor=BLUE_LIGHT,
        edgecolor=BLUE,
        title_color=BLUE,
        align="center",
    )

    rounded_box(
        axis,
        0.335,
        0.41,
        0.29,
        0.265,
        facecolor=TEAL_LIGHT,
        edgecolor=TEAL,
        linewidth=1.1,
    )
    axis.text(
        0.352,
        0.63,
        "Residual expert bank",
        color=TEAL,
        fontsize=8.3,
        fontweight="bold",
        va="center",
        zorder=3,
    )
    axis.text(
        0.352,
        0.595,
        r"4 stored  |  2 active  |  width 64 each  |  gate $g_l^r$",
        color=MUTED,
        fontsize=6.6,
        va="center",
        zorder=3,
    )
    expert_positions = [
        (0.355, 0.49, "E0", False),
        (0.485, 0.49, "E1", True),
        (0.355, 0.425, "E2", True),
        (0.485, 0.425, "E3", False),
    ]
    for x, y, label, selected in expert_positions:
        rounded_box(
            axis,
            x,
            y,
            0.105,
            0.045,
            facecolor=TEAL if selected else "white",
            edgecolor=TEAL if selected else LINE,
            linewidth=1.0,
            radius=0.01,
            zorder=3,
        )
        axis.text(
            x + 0.0525,
            y + 0.0225,
            label,
            ha="center",
            va="center",
            fontsize=7.4,
            fontweight="bold",
            color="white" if selected else MUTED,
            zorder=4,
        )
    arrow(axis, (0.27, 0.78), (0.165, 0.62), color=BLUE)
    arrow(axis, (0.30, 0.78), (0.44, 0.675), color=TEAL)
    arrow(
        axis,
        (0.565, 0.78),
        (0.555, 0.675),
        color=PURPLE,
        dashed=True,
    )
    axis.text(
        0.585,
        0.72,
        "primary + cyclic successor",
        ha="right",
        va="center",
        fontsize=6.3,
        color=PURPLE,
        fontweight="bold",
    )

    rounded_box(
        axis,
        0.245,
        0.205,
        0.075,
        0.075,
        facecolor="white",
        edgecolor=INK,
        linewidth=1.0,
        radius=0.038,
    )
    axis.text(0.2825, 0.2425, "+", ha="center", va="center", fontsize=14, color=INK)
    label_box(
        axis,
        0.355,
        0.185,
        0.22,
        0.115,
        "Feed-forward output",
        "shared path + selected residuals",
        facecolor=PANEL,
        align="center",
    )
    arrow(axis, (0.165, 0.47), (0.27, 0.28), color=BLUE)
    arrow(axis, (0.44, 0.41), (0.30, 0.28), color=TEAL)
    arrow(axis, (0.32, 0.2425), (0.355, 0.2425), color=INK)

    rounded_box(
        axis,
        0.02,
        0.06,
        0.61,
        0.075,
        facecolor="#F2F5F8",
        edgecolor="none",
        linewidth=0,
        radius=0.012,
    )
    axis.text(
        0.325,
        0.0975,
        "GQA is shared across models; token ID changes only residual parameter selection.",
        ha="center",
        va="center",
        color=INK,
        fontsize=6.7,
        fontweight="bold",
    )


def draw_mechanism(axis: plt.Axes) -> None:
    panel_label(axis, 0.675, 0.97, "b", "Fixed route, contextual computation")

    rounded_box(
        axis,
        0.70,
        0.78,
        0.265,
        0.115,
        facecolor=PURPLE_LIGHT,
        edgecolor=PURPLE,
        linewidth=1.05,
    )
    axis.text(
        0.8325,
        0.850,
        r"Same token ID  $t$",
        ha="center",
        va="center",
        color=PURPLE,
        fontsize=8.3,
        fontweight="bold",
    )
    axis.text(
        0.8325,
        0.810,
        "same layer route: primary E1 + successor E2",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=6.8,
    )

    rows = [
        (0.59, "context A", r"$\mathbf{x}_l^{A}$", BLUE, BLUE_LIGHT, r"$\Delta\mathbf{x}^{A}$"),
        (0.34, "context B", r"$\mathbf{x}_l^{B}$", AMBER, AMBER_LIGHT, r"$\Delta\mathbf{x}^{B}$"),
    ]
    for y, context, symbol, color, light, output in rows:
        label_box(
            axis,
            0.69,
            y,
            0.105,
            0.12,
            context,
            symbol,
            facecolor=light,
            edgecolor=color,
            title_color=color,
            align="center",
        )
        label_box(
            axis,
            0.825,
            y,
            0.07,
            0.12,
            "E1 + E2",
            "same weights",
            facecolor=TEAL_LIGHT,
            edgecolor=TEAL,
            title_color=TEAL,
            align="center",
        )
        label_box(
            axis,
            0.925,
            y,
            0.055,
            0.12,
            output,
            "different",
            facecolor=PANEL,
            edgecolor=LINE,
            align="center",
        )
        arrow(axis, (0.795, y + 0.06), (0.825, y + 0.06), color=color)
        arrow(axis, (0.895, y + 0.06), (0.925, y + 0.06), color=TEAL)

    arrow(
        axis,
        (0.8325, 0.78),
        (0.86, 0.71),
        color=PURPLE,
        dashed=True,
        connectionstyle="arc3,rad=-0.18",
    )
    arrow(
        axis,
        (0.86, 0.59),
        (0.86, 0.46),
        color=PURPLE,
        dashed=True,
    )

    rounded_box(
        axis,
        0.69,
        0.035,
        0.29,
        0.205,
        facecolor="#F2F5F8",
        edgecolor="none",
        linewidth=0,
        radius=0.012,
    )
    axis.text(
        0.835,
        0.190,
        "Fixed",
        ha="center",
        va="center",
        color=PURPLE,
        fontsize=7.2,
        fontweight="bold",
    )
    axis.text(
        0.835,
        0.158,
        "which residual parameters are selected",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=6.6,
    )
    axis.text(
        0.835,
        0.105,
        "Contextual",
        ha="center",
        va="center",
        color=TEAL,
        fontsize=7.2,
        fontweight="bold",
    )
    axis.text(
        0.835,
        0.073,
        "the hidden state and the resulting computation",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=6.6,
    )


def render() -> Path:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(10.8, 4.45))
    figure.patch.set_facecolor("white")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    rounded_box(
        axis,
        0.0,
        0.0,
        0.65,
        1.0,
        facecolor="white",
        edgecolor=LINE,
        linewidth=0.8,
        radius=0.012,
        zorder=0,
    )
    rounded_box(
        axis,
        0.66,
        0.0,
        0.34,
        1.0,
        facecolor="white",
        edgecolor=LINE,
        linewidth=0.8,
        radius=0.012,
        zorder=0,
    )
    draw_architecture(axis)
    draw_mechanism(axis)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(PDF_PATH, bbox_inches="tight", pad_inches=0.025, facecolor="white")
    figure.savefig(
        PNG_PATH,
        dpi=260,
        bbox_inches="tight",
        pad_inches=0.025,
        facecolor="white",
    )
    plt.close(figure)
    return PDF_PATH


if __name__ == "__main__":
    print(render())
