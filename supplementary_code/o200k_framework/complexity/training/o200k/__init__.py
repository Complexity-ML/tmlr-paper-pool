"""Utilities for the o200k Token-Routed pretraining runner."""

from .data import (
    FineWebDataset,
    LocalTextDataset,
    RandomTokenDataset,
    batch_expert_counts,
    build_loaders,
    infer_vocab_size,
    text_token_frequencies,
    token_shard_frequencies,
    tokenizer_token_classes,
)
from .cli import build_parser
from .checkpointing import load_checkpoint, save_checkpoint
from .optimizer import build_optimizer
from .profiles import PROFILES, make_config
from .runtime import (
    apply_topk_primary_weight,
    apply_shared_routed_gates,
    evaluate,
    expert_diversity_loss,
    learned_router_auxiliary_loss,
    init_distributed,
    reduce_average,
    reduce_average_tensor,
    scheduled_value,
    scheduled_topk_primary_weight,
    update_loss_free_router_biases,
)

__all__ = [
    "PROFILES",
    "make_config",
    "build_parser",
    "build_optimizer",
    "save_checkpoint",
    "load_checkpoint",
    "evaluate",
    "init_distributed",
    "reduce_average",
    "reduce_average_tensor",
    "scheduled_topk_primary_weight",
    "scheduled_value",
    "apply_topk_primary_weight",
    "apply_shared_routed_gates",
    "expert_diversity_loss",
    "learned_router_auxiliary_loss",
    "update_loss_free_router_biases",
    "RandomTokenDataset",
    "LocalTextDataset",
    "FineWebDataset",
    "batch_expert_counts",
    "build_loaders",
    "infer_vocab_size",
    "text_token_frequencies",
    "token_shard_frequencies",
    "tokenizer_token_classes",
]
