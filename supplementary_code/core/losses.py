"""Loss helpers for local training scripts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossMetrics:
    ce: float
    z_loss: float = 0.0


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
    z_loss_coef: float = 0.0,
) -> tuple[torch.Tensor, LossMetrics]:
    """Cross-entropy on next-token logits with optional PaLM-style z-loss."""

    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    ce = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=-100,
        label_smoothing=label_smoothing,
    )

    z_loss = flat_logits.logsumexp(dim=-1).square().mean()
    loss = ce + z_loss_coef * z_loss
    return loss, LossMetrics(ce=float(ce.detach().item()), z_loss=float(z_loss.detach().item()))
