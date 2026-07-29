#!/usr/bin/env python3
"""Independent MLX evaluation of Dense-306 and TR-MOE-306.

This script uses Apple MLX/Metal for all model computation. It loads the public
Hugging Face safetensors directly through ``mlx_lm.models.complexity`` and
shares only deterministic corpus packing and statistical helpers with the
PyTorch reference script.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

from mlx_complexity_306 import Model, ModelArgs
from run_heldout_nll import (
    DEFAULT_DATASET,
    DEFAULT_DATASET_REVISION,
    DEFAULT_SPLIT,
    compact_model_result,
    paired_bootstrap,
    sha256_file,
    stream_packed_blocks,
)


def load_mlx_model(model_dir: Path):
    from mlx_lm.utils import load_model

    model, config = load_model(
        model_dir,
        strict=True,
        model_config={"model_type": "complexity"},
        get_model_classes=lambda config: (Model, ModelArgs),
    )
    return model, config


def score_model(
    label: str,
    model_dir: Path,
    blocks,
    *,
    progress_every: int,
) -> dict:
    started = time.perf_counter()
    model, config = load_mlx_model(model_dir)
    load_seconds = time.perf_counter() - started

    block_nll: list[float] = []
    block_tokens: list[int] = []
    total_nll = 0.0
    total_tokens = 0
    evaluation_started = time.perf_counter()

    for index, block in enumerate(blocks, start=1):
        input_ids = mx.array([block.input_ids], dtype=mx.int32)
        labels = mx.array([block.labels], dtype=mx.int32)
        logits = model(input_ids).astype(mx.float32)
        losses = nn.losses.cross_entropy(logits, labels, reduction="none")
        loss_sum = losses.sum()
        mx.eval(loss_sum)

        token_count = block.scored_tokens
        nll_sum = float(loss_sum.item())
        block_nll.append(nll_sum / token_count)
        block_tokens.append(token_count)
        total_nll += nll_sum
        total_tokens += token_count

        del input_ids, labels, logits, losses, loss_sum
        if index % progress_every == 0 or index == len(blocks):
            elapsed = time.perf_counter() - evaluation_started
            print(
                f"{label}: {index}/{len(blocks)} blocks, "
                f"{total_tokens:,} tokens, {total_tokens / elapsed:,.1f} tok/s",
                flush=True,
            )

    evaluation_seconds = time.perf_counter() - evaluation_started
    mean_nll = total_nll / total_tokens
    result = {
        "model_name": label,
        "model_directory": str(model_dir.resolve()),
        "model_sha256": sha256_file(model_dir / "model.safetensors"),
        "config_sha256": sha256_file(model_dir / "config.json"),
        "load_seconds": load_seconds,
        "evaluation_seconds": evaluation_seconds,
        "tokens_per_second": total_tokens / evaluation_seconds,
        "scored_tokens": total_tokens,
        "negative_log_likelihood": mean_nll,
        "perplexity": math.exp(mean_nll),
        "per_block_nll": block_nll,
        "per_block_tokens": block_tokens,
        "loaded_model_type": config.get("model_type"),
        "checkpoint_mlp_type": config.get("mlp_type"),
    }

    del model
    gc.collect()
    mx.clear_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLX paired held-out NLL evaluation for the 306.5M pair"
    )
    parser.add_argument("--dense-model", type=Path, required=True)
    parser.add_argument("--routed-model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Optional Hugging Face dataset configuration name (for example, 'en' for C4).",
    )
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--target-tokens", type=int, default=262_144)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    parser.add_argument("--progress-every", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.bootstrap_seed)
    mx.random.seed(args.bootstrap_seed)

    mlx_lm_version = getattr(mlx_lm, "__version__", "unknown")
    print(f"runtime: mlx-lm {mlx_lm_version} on Apple MLX/Metal", flush=True)
    print(
        f"corpus: {args.dataset}@{args.dataset_revision} split={args.split}",
        flush=True,
    )
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
        revision=args.dataset_revision,
        streaming=True,
    )
    blocks, documents, sources = stream_packed_blocks(
        dataset,
        tokenizer,
        eos_token_id=0,
        block_size=args.block_size,
        target_tokens=args.target_tokens,
    )
    print(
        f"packed: {documents:,} documents -> {len(blocks):,} blocks -> "
        f"{sum(block.scored_tokens for block in blocks):,} scored tokens",
        flush=True,
    )

    dense = score_model(
        "dense-306",
        args.dense_model,
        blocks,
        progress_every=args.progress_every,
    )
    routed = score_model(
        "tr-moe-306",
        args.routed_model,
        blocks,
        progress_every=args.progress_every,
    )
    paired = paired_bootstrap(
        routed["per_block_nll"],
        dense["per_block_nll"],
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )

    output = {
        "protocol": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inference_backend": "Apple MLX/Metal",
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "dataset_revision": args.dataset_revision,
            "split": args.split,
            "independent_of_declared_training_corpus": True,
            "decontamination_against_fineweb_edu": False,
            "selection": "first documents in the pinned split",
            "documents_consumed": documents,
            "source_document_counts": dict(sorted(sources.items())),
            "tokenizer": str(args.tokenizer.resolve()),
            "tokenizer_sha256": sha256_file(args.tokenizer),
            "eos_token_id": 0,
            "packing": (
                "documents concatenated with EOS; contiguous blocks preserve one "
                "context token so every scored target appears exactly once"
            ),
            "block_size": args.block_size,
            "target_tokens": args.target_tokens,
            "mlx_version": getattr(mx, "__version__", "unknown"),
            "mlx_lm_version": mlx_lm_version,
            "mlx_model_implementation": str(
                Path(__file__).with_name("mlx_complexity_306.py").resolve()
            ),
            "mlx_model_implementation_sha256": sha256_file(
                Path(__file__).with_name("mlx_complexity_306.py")
            ),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "models": {
            "dense": compact_model_result(dense),
            "token_identity_residual": compact_model_result(routed),
        },
        "paired_comparison": paired,
        "per_block": {
            "dense_nll": dense["per_block_nll"],
            "token_identity_residual_nll": routed["per_block_nll"],
            "scored_tokens": dense["per_block_tokens"],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        "result: "
        f"dense={dense['negative_log_likelihood']:.6f} "
        f"(ppl={dense['perplexity']:.3f}), "
        f"routed={routed['negative_log_likelihood']:.6f} "
        f"(ppl={routed['perplexity']:.3f}), "
        f"delta={paired['mean']:+.6f}, "
        f"95% CI=[{paired['confidence_interval'][0]:+.6f}, "
        f"{paired['confidence_interval'][1]:+.6f}]",
        flush=True,
    )
    print(f"saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
