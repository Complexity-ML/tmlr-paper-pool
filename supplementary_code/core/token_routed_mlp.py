"""
Token-Routed MLP — Deterministic Mixture-of-Experts for the Complexity architecture.

Innovation: each token is routed to a small set of experts based on its
token ID. Routing is deterministic (no learned router, no load-balancing
loss) and fully parallel across experts.

Key design choices:
1. Zipf-balanced greedy bin-packing: tokens sorted by corpus frequency are
   assigned one-by-one to the expert with the lowest accumulated load, so
   each expert sees ~1/E of the total token mass (not just 1/E of the
   vocabulary).
2. Per-layer routing permutation: each layer applies a deterministic
   permutation of expert indices, which preserves Zipf load balance
   (permutations are measure-preserving) while giving each layer a
   different token→expert assignment for richer specialization.
3. Shared Lexical Expert: a dense SwiGLU MLP that ALL tokens pass through,
   capturing common syntactic / lexical patterns. The output is
       out(x) = g_s * SharedMLP(x) + g_r * RoutedMixture_e(x)
   where (g_s, g_r) are either learned scalar gates or 1 / 1.
4. Top-K deterministic Zipf: each token can activate K experts via cyclic
   shift of the Zipf primary, with the primary keeping weight w in [0, 1]
   and the remaining (1 - w) split equally across the (K - 1) secondaries.
   K = 1 recovers the classical single-expert Zipf routing.
5. Sparse dispatch: only the tokens assigned to expert e are sent through
   that expert's weights (padded grouped bmm, no masked dense computation).

Reference: Section 3.2 of the paper.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenRoutedMLP(nn.Module):
    """Token-Routed MLP with optional Shared Lexical Expert and learned gates.

    Args:
        hidden_size:
            Model hidden dimension.
        intermediate_size:
            Total routed FFN width. Each routed expert receives
            ``intermediate_size // num_experts``.
        num_experts:
            Number of routed experts (E).
        vocab_size:
            Tokenizer vocabulary size; used to size the token→expert table.
        token_frequencies:
            Optional 1-D tensor of length ``vocab_size`` with corpus token
            frequencies. When provided, Zipf bin-packing is used; otherwise
            the routing falls back to ``token_id % num_experts``.
        shared_expert:
            If True, add a dense SwiGLU shared expert that all tokens pass
            through.
        shared_intermediate_size:
            Width of the shared expert. Defaults to ``intermediate_size``
            (a full dense-width shared expert, as used in the 300M run).
        top_k:
            Number of experts activated per token. ``top_k=1`` is the
            classical single-expert Zipf routing.
        top_k_primary_weight:
            Blend weight in [0, 1] for the primary expert when ``top_k>1``.
            The remaining ``(1 - w)`` is split equally across the
            ``(top_k - 1)`` cyclic-shifted secondary experts.
        use_shared_routed_gates:
            If True, learn two scalar gates ``g_s`` and ``g_r`` multiplying
            the shared and routed mixtures respectively. Without them, the
            two contributions are simply summed.
        shared_gate_init, routed_gate_init:
            Initial values of the learned gates ``g_s`` and ``g_r``.
        layer_idx:
            Layer index, used to seed the per-layer expert permutation so
            that different layers route tokens to different experts.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        vocab_size: int,
        token_frequencies: Optional[torch.Tensor] = None,
        shared_expert: bool = True,
        shared_intermediate_size: Optional[int] = None,
        top_k: int = 1,
        top_k_primary_weight: float = 0.95,
        use_shared_routed_gates: bool = False,
        shared_gate_init: float = 1.0,
        routed_gate_init: float = 1.0,
        layer_idx: int = 0,
    ):
        super().__init__()

        if intermediate_size % num_experts != 0:
            raise ValueError(
                "intermediate_size must be divisible by num_experts; got "
                f"{intermediate_size} / {num_experts}"
            )

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.vocab_size = vocab_size
        self.expert_intermediate_size = intermediate_size // num_experts

        # Top-K deterministic Zipf parameters.
        self.top_k = max(1, int(top_k))
        self._primary_weight = (
            min(1.0, max(0.0, float(top_k_primary_weight))) if self.top_k > 1 else 1.0
        )

        # Routed expert weights — three stacked [E, *, *] tensors.
        self.gate_proj_w = nn.Parameter(
            torch.randn(num_experts, hidden_size, self.expert_intermediate_size) * 0.02
        )
        self.up_proj_w = nn.Parameter(
            torch.randn(num_experts, hidden_size, self.expert_intermediate_size) * 0.02
        )
        self.down_proj_w = nn.Parameter(
            torch.randn(num_experts, self.expert_intermediate_size, hidden_size) * 0.02
        )

        # Shared lexical expert (dense SwiGLU).
        self.use_shared_expert = bool(shared_expert)
        self.use_shared_routed_gates = bool(use_shared_routed_gates)
        if self.use_shared_expert:
            shared_size = shared_intermediate_size or intermediate_size
            self.shared_gate = nn.Linear(hidden_size, shared_size, bias=False)
            self.shared_up = nn.Linear(hidden_size, shared_size, bias=False)
            self.shared_down = nn.Linear(shared_size, hidden_size, bias=False)
            if self.use_shared_routed_gates:
                self.shared_output_gate = nn.Parameter(torch.tensor(float(shared_gate_init)))
                self.routed_output_gate = nn.Parameter(torch.tensor(float(routed_gate_init)))

        # Token→expert mapping (Zipf-balanced bin-packing or modulo fallback),
        # then a per-layer permutation that preserves load balance.
        self.register_buffer(
            "token_to_expert",
            self._build_routing_table(vocab_size, num_experts, token_frequencies, layer_idx),
        )

    @staticmethod
    def _build_routing_table(
        vocab_size: int,
        num_experts: int,
        token_frequencies: Optional[torch.Tensor],
        layer_idx: int,
    ) -> torch.Tensor:
        """Build the deterministic token→expert table.

        Step 1: Zipf greedy bin-packing if ``token_frequencies`` is given,
        else modulo fallback.
        Step 2: apply a layer-specific permutation of expert indices.
        Both steps preserve load balance (a permutation is measure-preserving
        over the load distribution).
        """
        if token_frequencies is not None:
            freqs = token_frequencies
            sorted_indices = freqs.argsort(descending=True)
            mapping = torch.empty(vocab_size, dtype=torch.long)
            expert_loads = [0.0] * num_experts
            for rank_pos in range(vocab_size):
                token_id = int(sorted_indices[rank_pos].item())
                e = min(range(num_experts), key=lambda i: expert_loads[i])
                mapping[token_id] = e
                expert_loads[e] += float(freqs[token_id].item())
        else:
            mapping = torch.arange(vocab_size, dtype=torch.long) % num_experts

        g = torch.Generator().manual_seed(0xC0DE + int(layer_idx))
        permutation = torch.randperm(num_experts, generator=g)
        return permutation[mapping]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with deterministic sparse dispatch.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            token_ids: [batch, seq_len]

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        B, S, H = hidden_states.shape

        if token_ids is None:
            return self._forward_all_experts(hidden_states)

        token_ids_clamped = token_ids.clamp(0, self.vocab_size - 1)
        expert_ids = self.token_to_expert[token_ids_clamped]   # [B, S]

        flat_x = hidden_states.reshape(-1, H)
        flat_expert_ids = expert_ids.reshape(-1)

        # Shared expert (dense, all tokens).
        if self.use_shared_expert:
            shared_out = self.shared_down(
                F.silu(self.shared_gate(flat_x)) * self.shared_up(flat_x)
            ).to(flat_x.dtype)
        else:
            shared_out = flat_x.new_zeros(flat_x.shape)

        # Top-K dispatch: primary expert + (K-1) cyclic-shifted secondaries.
        if self.top_k == 1:
            routed_out = self._dispatch_once(flat_x, flat_expert_ids, H)
        else:
            secondary_w = (1.0 - self._primary_weight) / (self.top_k - 1)
            routed_out = flat_x.new_zeros(flat_x.shape)
            for k in range(self.top_k):
                w = self._primary_weight if k == 0 else secondary_w
                ids_k = (
                    flat_expert_ids
                    if k == 0
                    else (flat_expert_ids + k) % self.num_experts
                )
                routed_out = routed_out + w * self._dispatch_once(flat_x, ids_k, H)

        # Combine via learned scalar gates if enabled, else plain sum.
        if self.use_shared_expert and self.use_shared_routed_gates:
            out = self.shared_output_gate * shared_out + self.routed_output_gate * routed_out
        else:
            out = shared_out + routed_out
        return out.view(B, S, H)

    def _dispatch_once(
        self,
        flat_x: torch.Tensor,
        expert_ids: torch.Tensor,
        H: int,
    ) -> torch.Tensor:
        """Single sparse dispatch pass for the given expert assignment.

        Sort tokens by expert ID, pad each bucket to ``max(counts)``, run
        three batched matmuls (gate / up / down with SwiGLU in between),
        scatter back to the original token order.
        """
        sorted_expert_ids, sorted_idx = torch.sort(expert_ids, stable=True)
        sorted_x = flat_x[sorted_idx]
        counts = torch.bincount(expert_ids, minlength=self.num_experts).tolist()
        offsets, off = [], 0
        for c in counts:
            offsets.append(off)
            off += c

        capacity = max(counts) if counts else 0
        if capacity == 0:
            return flat_x.new_zeros(flat_x.shape)

        padded = sorted_x.new_zeros(self.num_experts, capacity, H)
        for e in range(self.num_experts):
            n = counts[e]
            if n == 0:
                continue
            s = offsets[e]
            padded[e, :n] = sorted_x[s:s + n]

        gate = torch.bmm(padded, self.gate_proj_w)
        up = torch.bmm(padded, self.up_proj_w)
        intermediate = F.silu(gate) * up
        out_padded = torch.bmm(intermediate, self.down_proj_w)

        out = flat_x.new_zeros(flat_x.shape)
        for e in range(self.num_experts):
            n = counts[e]
            if n == 0:
                continue
            s = offsets[e]
            out[sorted_idx[s:s + n]] = out_padded[e, :n].to(out.dtype)
        return out

    def _forward_all_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Fallback: average across experts (inference without token_ids)."""
        flat = hidden_states.reshape(-1, self.hidden_size)
        out = flat.new_zeros(flat.shape)
        for e in range(self.num_experts):
            gate_e = flat @ self.gate_proj_w[e]
            up_e = flat @ self.up_proj_w[e]
            out = out + (F.silu(gate_e) * up_e) @ self.down_proj_w[e]
        out = out / self.num_experts
        if self.use_shared_expert:
            shared = self.shared_down(
                F.silu(self.shared_gate(flat)) * self.shared_up(flat)
            )
            if self.use_shared_routed_gates:
                out = self.shared_output_gate * shared + self.routed_output_gate * out
            else:
                out = out + shared
        return out.view_as(hidden_states)
