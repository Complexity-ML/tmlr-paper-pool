#!/usr/bin/env python3
"""Reconstruct the short-budget learned-router table from raw artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

VARIANTS = (
    ("Learned contextual top-2 + auxiliary balancing", "100m_learned_aux_shared"),
    ("Dense residual", "100m_dense_residual"),
    ("Learned contextual top-2 + loss-free balancing", "100m_learned_loss_free_shared"),
    (
        "Modulo-primary/corpus-balanced secondary + shared",
        "100m_modulo_balanced_secondary_shared",
    ),
)
EXPECTED_TOTAL_TOKENS = 99_614_720


def metric_at(rows: list[dict[str, str]], step: int, field: str) -> float:
    for row in rows:
        if int(row["step"]) == step:
            value = float(row[field])
            if value != value:  # NaN
                raise ValueError(f"{field} is NaN at step {step}")
            return value
    raise ValueError(f"missing step {step}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "100m_router_short",
    )
    args = parser.parse_args()

    print("| Variant | Params | Train @95 | Eval @75 |")
    print("|---|---:|---:|---:|")
    for label, run_name in VARIANTS:
        run_dir = args.results_dir / run_name
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        if config["total_tokens"] != EXPECTED_TOTAL_TOKENS:
            raise ValueError(
                f"{run_name}: expected {EXPECTED_TOTAL_TOKENS} tokens, "
                f"got {config['total_tokens']}"
            )
        with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        train = metric_at(rows, 95, "train_loss")
        evaluation = metric_at(rows, 75, "eval_loss")
        print(
            f"| {label} | {config['params'] / 1e6:.3f}M | "
            f"{train:.4f} | {evaluation:.4f} |"
        )


if __name__ == "__main__":
    main()
