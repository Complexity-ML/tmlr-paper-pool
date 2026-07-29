#!/usr/bin/env python3
"""Evaluate the matched 306.5M checkpoints on an independent held-out corpus.

The script streams the pinned test split of EleutherAI/pile_val_test, tokenizes
documents with the checkpoint's exact 32k tokenizer, packs non-overlapping
causal-LM targets, and scores the same token blocks with both models.

No model weights are updated. The output records aggregate NLL/perplexity,
per-block paired differences, a deterministic paired-bootstrap confidence
interval, package versions, and hashes of every local model artifact.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer


DEFAULT_DATASET = "EleutherAI/pile_val_test"
DEFAULT_DATASET_REVISION = "05b327037e6301f256d8df32193756edc4c8e3bd"
DEFAULT_SPLIT = "test"


@dataclass(frozen=True)
class PackedBlock:
    input_ids: list[int]
    labels: list[int]

    @property
    def scored_tokens(self) -> int:
        return len(self.labels)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested: str) -> tuple[str, torch.dtype]:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda", torch.bfloat16
        if torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return "cuda", torch.bfloat16
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return "mps", torch.float16
    return "cpu", torch.float32


def stream_packed_blocks(
    rows: Iterable[dict],
    tokenizer: Tokenizer,
    *,
    eos_token_id: int,
    block_size: int,
    target_tokens: int,
) -> tuple[list[PackedBlock], int, Counter[str]]:
    """Pack documents while scoring each target token exactly once."""

    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    if target_tokens < block_size:
        raise ValueError("target_tokens must be at least one complete block")

    blocks: list[PackedBlock] = []
    buffer: list[int] = []
    documents = 0
    sources: Counter[str] = Counter()
    scored = 0

    for row in rows:
        text = row.get("text", "")
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if not ids:
            continue

        documents += 1
        metadata = row.get("meta") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        sources[str(metadata.get("pile_set_name", "unknown"))] += 1
        buffer.extend(ids)
        buffer.append(eos_token_id)

        while len(buffer) >= block_size + 1 and scored < target_tokens:
            remaining = target_tokens - scored
            current_targets = min(block_size, remaining)
            window = buffer[: current_targets + 1]
            blocks.append(PackedBlock(window[:-1], window[1:]))
            scored += current_targets
            # Preserve the final token as the next block's first-token context.
            buffer = buffer[current_targets:]

        if scored >= target_tokens:
            break

    if scored != target_tokens:
        raise RuntimeError(
            f"Corpus ended after {scored:,} scored tokens; "
            f"{target_tokens:,} were requested"
        )
    return blocks, documents, sources


def load_runtime_model(
    model_name: str,
    model_dir: Path,
    *,
    dtype: torch.dtype,
    device: str,
) -> torch.nn.Module:
    try:
        from vllm_i64.core.loader import load_model_by_name
    except ImportError as exc:
        raise RuntimeError(
            "vllm-i64 is required. Install it or prepend its repository to "
            "PYTHONPATH before running this script."
        ) from exc

    model = load_model_by_name(
        model_name,
        dtype=dtype,
        device=device,
        checkpoint_override=str(model_dir),
    )
    model.eval()
    return model


@torch.inference_mode()
def score_model(
    model_name: str,
    model_dir: Path,
    blocks: list[PackedBlock],
    *,
    dtype: torch.dtype,
    device: str,
    progress_every: int,
) -> dict:
    started = time.perf_counter()
    model = load_runtime_model(
        model_name,
        model_dir,
        dtype=dtype,
        device=device,
    )
    load_seconds = time.perf_counter() - started

    block_nll: list[float] = []
    block_tokens: list[int] = []
    total_nll = 0.0
    total_tokens = 0
    evaluation_started = time.perf_counter()

    for index, block in enumerate(blocks, start=1):
        input_ids = torch.tensor(
            block.input_ids,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        labels = torch.tensor(
            block.labels,
            dtype=torch.long,
            device=device,
        )
        logits = model(input_ids)
        loss_sum = F.cross_entropy(
            logits.float(),
            labels,
            reduction="sum",
        )
        token_count = block.scored_tokens
        nll_sum = float(loss_sum.item())
        block_nll.append(nll_sum / token_count)
        block_tokens.append(token_count)
        total_nll += nll_sum
        total_tokens += token_count

        del input_ids, labels, logits, loss_sum
        if index % progress_every == 0 or index == len(blocks):
            elapsed = time.perf_counter() - evaluation_started
            print(
                f"{model_name}: {index}/{len(blocks)} blocks, "
                f"{total_tokens:,} tokens, {total_tokens / elapsed:,.1f} tok/s",
                flush=True,
            )

    evaluation_seconds = time.perf_counter() - evaluation_started
    mean_nll = total_nll / total_tokens
    result = {
        "model_name": model_name,
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
    }

    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    return result


def paired_bootstrap(
    routed_block_nll: list[float],
    dense_block_nll: list[float],
    *,
    samples: int,
    seed: int,
) -> dict:
    if len(routed_block_nll) != len(dense_block_nll):
        raise ValueError("Paired results must have identical block counts")

    paired = np.asarray(routed_block_nll) - np.asarray(dense_block_nll)
    rng = np.random.default_rng(seed)
    n = paired.size
    bootstrap_means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1_000):
        count = min(1_000, samples - start)
        indices = rng.integers(0, n, size=(count, n))
        bootstrap_means[start : start + count] = paired[indices].mean(axis=1)

    return {
        "definition": "token_identity_residual_minus_dense_nll",
        "block_count": int(n),
        "mean": float(paired.mean()),
        "standard_deviation": float(paired.std(ddof=1)),
        "standard_error": float(paired.std(ddof=1) / math.sqrt(n)),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "confidence_level": 0.95,
        "confidence_interval": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
        "fraction_of_blocks_favouring_routed": float((paired < 0).mean()),
    }


def compact_model_result(result: dict) -> dict:
    return {
        key: value
        for key, value in result.items()
        if key not in {"per_block_nll", "per_block_tokens"}
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired held-out NLL evaluation for Dense-306 and TR-MOE-306"
    )
    parser.add_argument("--dense-model", type=Path, required=True)
    parser.add_argument("--routed-model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--target-tokens", type=int, default=262_144)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    parser.add_argument("--progress-every", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    random.seed(args.bootstrap_seed)
    np.random.seed(args.bootstrap_seed)
    torch.manual_seed(args.bootstrap_seed)

    device, dtype = resolve_device(args.device)
    print(
        f"runtime: device={device}, dtype={dtype}, threads={args.threads}",
        flush=True,
    )
    print(
        f"corpus: {args.dataset}@{args.dataset_revision} split={args.split}",
        flush=True,
    )

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    dataset = load_dataset(
        args.dataset,
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
        dtype=dtype,
        device=device,
        progress_every=args.progress_every,
    )
    routed = score_model(
        "tr-moe-306",
        args.routed_model,
        blocks,
        dtype=dtype,
        device=device,
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
            "dataset": args.dataset,
            "dataset_revision": args.dataset_revision,
            "split": args.split,
            "independent_of_declared_training_corpus": True,
            "decontamination_against_fineweb_edu": False,
            "selection": "first documents in the pinned test split",
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
            "device": device,
            "dtype": str(dtype),
            "threads": args.threads,
            "torch_version": torch.__version__,
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
