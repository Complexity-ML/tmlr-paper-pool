"""Render corrected 100M ablation figures from the archived metric CSVs."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

RUNS = [
    ("100m_modulo_balanced_secondary_shared_metrics.csv", "Modulo-primary / balanced-secondary + shared"),
    ("100m_dense_residual_metrics.csv", "Dense residual"),
    ("100m_modulo_shared_metrics.csv", "Modulo-adjacent top-2 + shared"),
    ("100m_round_robin_shared_metrics.csv", "Round-robin top-2 + shared"),
    ("100m_random_shared_metrics.csv", "Random top-2 + shared"),
    ("100m_shared_only_metrics.csv", "Shared-only"),
    ("100m_modulo_balanced_secondary_no_shared_metrics.csv", "Modulo-primary / balanced-secondary, no shared"),
]


def rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    series = []
    for filename, label in RUNS:
        data = rows(args.metrics_dir / filename)
        steps = [int(row["step"]) for row in data]
        train = [float(row["train_loss"]) for row in data]
        valid = [(int(row["step"]), float(row["eval_loss"])) for row in data if row["eval_loss"].lower() != "nan"]
        series.append((label, steps, train, valid))

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for label, steps, train, _ in series:
        ax.plot(steps, train, linewidth=1.8, label=label)
    ax.set_title("Exploratory 100M ablations — matched 1.0003B-token budget")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(args.out_dir / "fig_100m_b200_ablation_train_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    ranking = []
    for label, _, _, valid in series:
        step, loss = min(valid, key=lambda item: item[1])
        ranking.append((label, loss, step))
    ranking.sort(key=lambda item: item[1])

    fig, ax = plt.subplots(figsize=(11, 6.5))
    labels = [item[0] for item in ranking]
    losses = [item[1] for item in ranking]
    bars = ax.barh(labels, losses, color="#4c91c7")
    ax.invert_yaxis()
    ax.set_title("Exploratory 100M validation ranking\n(fixed top-2 lookup; no learned expert router)")
    ax.set_xlabel("Best logged validation loss (lower is better)")
    lo = min(losses) - 0.02
    hi = max(losses) + 0.02
    ax.set_xlim(lo, hi)
    ax.grid(axis="x", alpha=0.25)
    for bar, (_, loss, step) in zip(bars, ranking):
        ax.text(loss + 0.001, bar.get_y() + bar.get_height() / 2, f"{loss:.4f} @ {step}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_dir / "fig_100m_b200_ablation_eval_ranking.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
