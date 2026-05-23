"""Training helpers shared by supplementary scripts."""

from __future__ import annotations

import torch


def global_expert_shares(model, num_experts: int) -> tuple[list[float], int]:
    """Estimate global routed-expert shares from token routing tables."""

    counts = torch.zeros(num_experts, dtype=torch.float64)
    for module in model.modules():
        table = getattr(module, "token_to_expert", None)
        if table is None:
            continue
        counts += torch.bincount(table.detach().cpu().reshape(-1), minlength=num_experts)[:num_experts]

    total = float(counts.sum().item())
    if total == 0:
        return [], num_experts

    shares = (counts / total).tolist()
    dead = int((counts == 0).sum().item())
    return [float(x) for x in shares], dead
