"""Render the checkpoint-verified 300M COMPLEXITY-DEEP architecture diagram."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "figures" / "architecture_complexity_deep.png"

fig, ax = plt.subplots(figsize=(12, 17))
ax.set_xlim(0, 12)
ax.set_ylim(0, 17)
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

ax.text(6, 16.55, "COMPLEXITY-DEEP", ha="center", va="center", fontsize=27, fontweight="bold")
ax.text(6, 16.12, "300M shared dense backbone + deterministic lexical residual", ha="center", color="#666", fontsize=14)

box(3.0, 15.2, 6.0, 0.62, "Tokenized input · vocabulary 32,000", fc="#e8e2ff", fontsize=13)
arrow(6, 15.2, 6, 14.82)
box(3.0, 14.2, 6.0, 0.62, "Tied token embedding · d_model = 1024", fc="#e8e2ff", fontsize=13)
arrow(6, 14.2, 6, 13.72)

outer = FancyBboxPatch((0.65, 3.65), 10.7, 10.05, boxstyle="round,pad=0.06,rounding_size=0.25", facecolor="#ffffff", edgecolor="#c8c8c8", linewidth=1.7, linestyle="--")
ax.add_patch(outer)
ax.text(6, 13.42, "decoder block ×18 · context 2048", ha="center", color="#e87800", fontsize=15, fontweight="bold")

box(3.8, 12.45, 4.4, 0.56, "RMSNorm")
arrow(6, 12.45, 6, 11.98)
box(1.6, 10.65, 8.8, 1.3, "Grouped-Query Attention\n16 Q heads · 4 KV heads · QK-Norm · RoPE · SDPA", fc="#e8f3ff", ec="#2585d5", fontsize=14)
arrow(6, 10.65, 6, 10.25)
box(4.55, 9.65, 2.9, 0.55, "Residual add")
arrow(6, 9.65, 6, 9.18)
box(3.8, 8.62, 4.4, 0.56, "RMSNorm")
arrow(6, 8.62, 6, 8.15)

mlp = FancyBboxPatch((1.35, 5.0), 9.3, 3.15, boxstyle="round,pad=0.05,rounding_size=0.22", facecolor="#eefaf1", edgecolor="#28944c", linewidth=1.8)
ax.add_patch(mlp)
ax.text(6, 7.85, "Residual Token-Routed MLP", ha="center", fontsize=17, fontweight="bold", color="#238d48")
box(2.0, 7.05, 8.0, 0.55, "Fixed top-2 lookup: layer permutation of token_id mod 4", fc="#fff0df", ec="#ed7d00", fontsize=12.5)
box(1.9, 5.72, 3.7, 0.92, "Shared dense branch\nSwiGLU · d_ff = 3840", fc="#fff4c9", ec="#ed8b00", fontsize=12.5)
box(6.4, 5.72, 3.7, 0.92, "Routed lexical branch\n4 × SwiGLU · d_ff = 64 each", fc="#eaf8ee", ec="#28944c", fontsize=12.5)
arrow(6, 7.05, 3.75, 6.66, color="#ed7d00")
arrow(6, 7.05, 8.25, 6.66, color="#28944c")
ax.text(6, 5.42, "top-2 weights: 0.5 / 0.5 · branch gates initialized: shared 1.0 / routed 0.1", ha="center", fontsize=10.5, color="#555")
arrow(3.75, 5.72, 5.45, 4.72, color="#ed7d00")
arrow(8.25, 5.72, 6.55, 4.72, color="#28944c")
box(3.1, 4.22, 5.8, 0.55, "shared output + routed lexical residual")
arrow(6, 4.22, 6, 3.82)
box(4.55, 3.22, 2.9, 0.55, "Residual add")

arrow(6, 3.22, 6, 2.72)
box(3.8, 2.15, 4.4, 0.56, "Final RMSNorm")
arrow(6, 2.15, 6, 1.7)
box(3.15, 1.02, 5.7, 0.65, "Tied LM head · 1024 → 32,000", fc="#e8f0ff", fontsize=13)
arrow(6, 1.02, 6, 0.68)
ax.text(6, 0.42, "output logits", ha="center", fontsize=12, color="#666")

ax.text(6, -0.02, "Matched-token comparison: Token-Routed 306.5M vs dense 306.5M · no wall-clock efficiency claim", ha="center", va="bottom", fontsize=11.5, fontweight="bold", color="#b3264b")

fig.tight_layout(pad=0.5)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(OUT)
