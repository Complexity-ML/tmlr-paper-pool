"""Render the checkpoint-verified 300M COMPLEXITY-DEEP architecture diagram."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath

OUT = Path(__file__).resolve().parents[1] / "figures" / "architecture_complexity_deep.png"

fig, ax = plt.subplots(figsize=(12, 5.4))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5.4)
ax.axis("off")


def box(x, y, w, h, text, fc="#f8f8f8", ec="#666", fontsize: float = 12, lw=1.8):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.16",
        facecolor=fc, edgecolor=ec, linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(x1, y1, x2, y2, color="#666", lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=14, color=color, linewidth=lw))


def routed_arrow(points, color="#666", lw=1.6):
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="->", mutation_scale=14, color=color, linewidth=lw))

ax.text(6, 5.08, "Token identity as a residual routing signal", ha="center", va="center", fontsize=20, fontweight="bold")

box(0.35, 3.55, 2.2, 0.78, "Contextual state\n$h_t$", fc="#e8f3ff", ec="#2585d5", fontsize=13)
box(0.35, 1.22, 2.2, 0.78, "Token identity\n$t$", fc="#e8e2ff", ec="#7159b8", fontsize=13)

box(3.15, 3.55, 3.25, 0.78, "Shared dense SwiGLU\nwidth 3,840 · all tokens", fc="#fff4c9", ec="#ed8b00", fontsize=12.5)
box(3.15, 1.22, 3.25, 0.78, "Fixed top-2 lookup\npermuted modulo + balanced secondary", fc="#f1eaff", ec="#7159b8", fontsize=11.5)
box(7.0, 2.35, 3.25, 0.98, "Selected residual experts\n2 of 4 SwiGLU experts · width 64 each\nfixed top-2 weights: 0.5 / 0.5", fc="#eaf8ee", ec="#28944c", fontsize=11.5)
box(10.72, 2.35, 1.0, 0.98, "$+$", fc="#f8f8f8", ec="#666", fontsize=20)

arrow(2.55, 3.94, 3.15, 3.94, color="#2585d5")
routed_arrow([(2.55, 3.76), (2.82, 3.42), (2.82, 2.84), (7.0, 2.84)], color="#2585d5")
arrow(2.55, 1.61, 3.15, 1.61, color="#7159b8")
arrow(6.4, 1.61, 7.0, 2.62, color="#7159b8")
routed_arrow([(6.4, 3.94), (10.48, 3.94), (10.48, 3.05), (10.72, 3.05)], color="#ed8b00")
arrow(10.25, 2.84, 10.72, 2.84, color="#28944c")

ax.text(4.78, 4.55, "shared branch gate $g^s_l$ (init 1.0)", ha="center", fontsize=10.5, color="#8a5100")
ax.text(8.62, 2.02, "routed branch gate $g^r_l$ (init 0.1)", ha="center", fontsize=10.5, color="#216f3c")
ax.text(6, 0.48, "The token ID selects parameters; both branches transform the same contextual hidden state.", ha="center", fontsize=12, color="#444")

fig.tight_layout(pad=0.5)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(OUT)
