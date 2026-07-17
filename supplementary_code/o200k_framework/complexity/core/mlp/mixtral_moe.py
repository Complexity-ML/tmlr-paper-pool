"""Contextual top-k residual MoE baselines.

The module keeps the shared-plus-residual architecture of TokenRoutedMLP and
changes only the routing signal. Two balancing modes are supported:

* ``aux_loss``: softmax routing with the standard differentiable load loss.
* ``loss_free_bias``: sigmoid scores with a non-gradient selection bias that
  is updated from observed expert loads.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_mlp
from .base import MLPBase, MLPConfig


@register_mlp("mixtral")
@register_mlp("learned_router")
@register_mlp("standard_moe")
class MixtralMoE(MLPBase):
    """Learned top-k router over the same residual experts as TokenRoutedMLP."""

    def __init__(self, config: MLPConfig):
        super().__init__(config)
        self.num_experts = int(config.num_experts)
        self.top_k = int(config.top_k)
        self.expert_intermediate_size = self.intermediate_size // self.num_experts
        self.router_balance_mode = config.router_balance_mode
        self.router_aux_loss_coef = float(config.router_aux_loss_coef)
        self.router_bias_update_rate = float(config.router_bias_update_rate)

        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.gate_proj_w = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, self.expert_intermediate_size)
        )
        self.up_proj_w = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, self.expert_intermediate_size)
        )
        self.down_proj_w = nn.Parameter(
            torch.empty(self.num_experts, self.expert_intermediate_size, self.hidden_size)
        )

        shared_size = config.shared_intermediate_size or self.intermediate_size
        self.shared_expert = bool(config.shared_expert)
        if self.shared_expert:
            self.shared_gate = nn.Linear(self.hidden_size, shared_size, bias=False)
            self.shared_up = nn.Linear(self.hidden_size, shared_size, bias=False)
            self.shared_down = nn.Linear(shared_size, self.hidden_size, bias=False)

        if config.use_shared_routed_gates:
            self.shared_output_gate = nn.Parameter(torch.tensor(float(config.shared_gate_init)))
            self.routed_output_gate = nn.Parameter(torch.tensor(float(config.routed_gate_init)))
        else:
            self.register_buffer(
                "shared_output_gate", torch.tensor(float(config.shared_gate_init)), persistent=False
            )
            self.register_buffer(
                "routed_output_gate", torch.tensor(float(config.routed_gate_init)), persistent=False
            )

        self.register_buffer("router_selection_bias", torch.zeros(self.num_experts))
        self.register_buffer("expert_counts", torch.zeros(self.num_experts, dtype=torch.long))
        self.register_buffer("last_expert_counts", torch.zeros(self.num_experts))
        self.last_topk_expert_ids = torch.empty(0, self.top_k, dtype=torch.long)
        self._last_router_aux_loss: Optional[torch.Tensor] = None

        self._init_weights()

    def _init_weights(self) -> None:
        for expert_idx in range(self.num_experts):
            nn.init.kaiming_uniform_(self.gate_proj_w[expert_idx], a=5**0.5)
            nn.init.kaiming_uniform_(self.up_proj_w[expert_idx], a=5**0.5)
            nn.init.kaiming_uniform_(self.down_proj_w[expert_idx], a=5**0.5)
        nn.init.kaiming_uniform_(self.router.weight, a=5**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del token_ids, kwargs
        batch_size, seq_len, _ = hidden_states.shape
        flat_x = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.router(flat_x).float()

        if self.router_balance_mode == "loss_free_bias":
            gate_scores = torch.sigmoid(router_logits)
            selection_scores = gate_scores + self.router_selection_bias
        else:
            gate_scores = F.softmax(router_logits, dim=-1)
            selection_scores = gate_scores

        topk_expert_ids = selection_scores.topk(self.top_k, dim=-1).indices
        selected_weights = gate_scores.gather(1, topk_expert_ids)
        selected_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        flat_expert_ids = topk_expert_ids.reshape(-1)
        flat_route_weights = selected_weights.reshape(-1).to(flat_x.dtype)
        token_indices = (
            torch.arange(flat_x.shape[0], device=flat_x.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
            .reshape(-1)
        )
        routed_out = torch.zeros_like(flat_x)
        for expert_idx in range(self.num_experts):
            assignment_mask = flat_expert_ids == expert_idx
            if not bool(assignment_mask.any()):
                continue
            assigned_tokens = token_indices[assignment_mask]
            expert_x = flat_x.index_select(0, assigned_tokens)
            gate = expert_x @ self.gate_proj_w[expert_idx]
            up = expert_x @ self.up_proj_w[expert_idx]
            expert_out = (F.silu(gate) * up) @ self.down_proj_w[expert_idx]
            expert_out = expert_out * flat_route_weights[assignment_mask].unsqueeze(-1)
            routed_out.index_add_(0, assigned_tokens, expert_out)

        counts = torch.bincount(flat_expert_ids, minlength=self.num_experts)
        if self.training:
            current_counts = counts.detach().to(self.last_expert_counts.dtype)
            if self.router_balance_mode == "loss_free_bias":
                self.last_expert_counts.add_(current_counts)
            else:
                self.last_expert_counts.copy_(current_counts)
        self.expert_counts.add_(counts.detach().to(self.expert_counts.dtype))
        self.last_topk_expert_ids = topk_expert_ids.detach()

        if self.training and self.router_balance_mode == "aux_loss":
            assignment_density = F.one_hot(
                topk_expert_ids, num_classes=self.num_experts
            ).float().mean(dim=(0, 1))
            probability_density = gate_scores.mean(dim=0)
            self._last_router_aux_loss = self.num_experts * (
                assignment_density * probability_density
            ).sum()
        else:
            self._last_router_aux_loss = None

        if self.shared_expert:
            shared_out = self.shared_down(
                F.silu(self.shared_gate(flat_x)) * self.shared_up(flat_x)
            ).to(flat_x.dtype)
        else:
            shared_out = torch.zeros_like(flat_x)

        output = self.shared_output_gate * shared_out + self.routed_output_gate * routed_out
        return output.reshape(batch_size, seq_len, self.hidden_size)

    def router_auxiliary_loss(self) -> torch.Tensor:
        if self._last_router_aux_loss is None:
            return self.router.weight.sum() * 0.0
        return self.router_aux_loss_coef * self._last_router_aux_loss

    @torch.no_grad()
    def update_loss_free_bias(self, expert_counts: torch.Tensor) -> None:
        if self.router_balance_mode != "loss_free_bias":
            return
        counts = expert_counts.to(self.router_selection_bias)
        correction = torch.sign(counts.mean() - counts)
        self.router_selection_bias.add_(self.router_bias_update_rate * correction)
        self.router_selection_bias.sub_(self.router_selection_bias.mean())
        self.last_expert_counts.zero_()

    @torch.no_grad()
    def reset_expert_counts(self) -> None:
        self.expert_counts.zero_()