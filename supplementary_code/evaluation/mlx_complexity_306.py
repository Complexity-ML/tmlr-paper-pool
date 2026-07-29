"""Standalone MLX implementation for the public Dense-306/TR-MOE-306 pair.

This module is intentionally bundled with the evaluation artifact so the
held-out measurement does not depend on the project-specific vllm-i64 runtime
or on an unversioned local mlx-lm model plugin.
"""

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    rope_scaling: Optional[dict] = None
    norm_eps: float = 1e-6
    num_experts: int = 4
    shared_expert: bool = True
    shared_intermediate_size: Optional[int] = None
    use_shared_routed_gates: bool = False
    use_mu_guidance: bool = False
    use_qk_norm: bool = True
    tie_word_embeddings: bool = True
    top_k: int = 1
    top_k_primary_weight: Optional[float] = None
    mlp_type: str = "token_routed"
    use_token_routed_mlp: bool = True
    # Semantic LSH routing: layers >= lsh_from_layer route on a fixed
    # random-hyperplane hash of h (sign, zero-threshold) instead of token_id.
    routing_strategy: str = "modulo_cyclic"
    lsh_routing: bool = False
    lsh_bits: int = 0                # 0 -> ceil(log2(num_experts))
    lsh_from_layer: int = 0


class Attention(nn.Module):
    """GQA + Mu-KQV bias + optional QK-norm."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = dim // self.n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.use_mu_guidance = args.use_mu_guidance
        if self.use_mu_guidance:
            self.mu_to_q = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
            self.mu_to_k = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
            self.mu_to_v = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)

        self.use_qk_norm = args.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=args.norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=args.norm_eps)

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mu_prev: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if mu_prev is not None and self.use_mu_guidance:
            q = q + self.mu_to_q(mu_prev)
            k = k + self.mu_to_k(mu_prev)
            v = v + self.mu_to_v(mu_prev)

        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class TokenRoutedMLP(nn.Module):
    """Deterministic token-routed MoE with optional shared expert.

    Routing: token_id → expert_id via a precomputed buffer (Zipf bin-packing,
    or modulo fallback at init — overwritten when weights are loaded).
    """

    def __init__(self, args: ModelArgs, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.intermediate_size = args.intermediate_size
        self.num_experts = args.num_experts
        self.vocab_size = args.vocab_size
        self.top_k = max(1, int(args.top_k))
        primary_weight = args.top_k_primary_weight
        if primary_weight is None:
            primary_weight = 0.95
        self.primary_weight = (
            min(1.0, max(0.0, float(primary_weight))) if self.top_k > 1 else 1.0
        )

        expert_inter = self.intermediate_size // self.num_experts

        # Routed experts as raw [E, H, I_e] / [E, I_e, H] tensors (matches PyTorch ckpt).
        self.gate_proj_w = mx.zeros((self.num_experts, self.hidden_size, expert_inter))
        self.up_proj_w = mx.zeros((self.num_experts, self.hidden_size, expert_inter))
        self.down_proj_w = mx.zeros((self.num_experts, expert_inter, self.hidden_size))

        self.use_shared_expert = args.shared_expert
        if self.use_shared_expert:
            shared_inter = args.shared_intermediate_size or self.intermediate_size
            self.shared_inter = shared_inter
            self.shared_gate = nn.Linear(self.hidden_size, shared_inter, bias=False)
            self.shared_up = nn.Linear(self.hidden_size, shared_inter, bias=False)
            self.shared_down = nn.Linear(shared_inter, self.hidden_size, bias=False)

        self.use_output_gates = args.use_shared_routed_gates
        if self.use_output_gates:
            self.shared_output_gate = mx.zeros(())
            self.routed_output_gate = mx.zeros(())

        # The checkpoint stores only the primary lexical route. Additional
        # top-k routes are derived cyclically from that primary assignment.
        base = (mx.arange(self.vocab_size) % self.num_experts).astype(mx.int32)
        self.token_to_expert = base

        # Semantic LSH routing for deep layers (>= lsh_from_layer).
        self.lsh = bool(args.lsh_routing) and layer_idx >= int(args.lsh_from_layer)
        if self.lsh:
            self.lsh_bits = int(args.lsh_bits) or max(1, (self.num_experts - 1).bit_length())
            self.lsh_planes = mx.zeros((self.lsh_bits, self.hidden_size))
            self.lsh_bucket_to_expert = (
                mx.arange(1 << self.lsh_bits) % self.num_experts
            ).astype(mx.int32)

    def _shared(self, flat_x: mx.array) -> mx.array:
        return self.shared_down(nn.silu(self.shared_gate(flat_x)) * self.shared_up(flat_x))

    def _lsh_route(self, flat_x: mx.array):
        """LSH zero-threshold routing -> (primary, secondary) expert ids, each [N]."""
        proj = flat_x @ self.lsh_planes.T                       # [N, bits]
        bit_vals = (2 ** mx.arange(self.lsh_bits)).astype(mx.int32)
        bits_b = (proj > 0).astype(mx.int32)
        bucket = (bits_b * bit_vals).sum(-1)                    # [N]
        primary = self.lsh_bucket_to_expert[bucket]
        if self.top_k < 2:
            return [primary]
        # neighbour route: flip the lowest-|margin| bit
        flip = mx.argmin(mx.abs(proj), axis=-1)                 # [N]
        ind = (mx.arange(self.lsh_bits) == flip[:, None]).astype(mx.int32)
        nbucket = (mx.abs(bits_b - ind) * bit_vals).sum(-1)
        secondary = self.lsh_bucket_to_expert[nbucket]
        routes = [primary, secondary]
        for k in range(2, self.top_k):
            routes.append((primary + k) % self.num_experts)
        return routes

    def __call__(self, hidden_states: mx.array, token_ids: Optional[mx.array]) -> mx.array:
        B, S, H = hidden_states.shape
        flat_x = hidden_states.reshape(-1, H)

        shared_out = self._shared(flat_x) if self.use_shared_expert else mx.zeros_like(flat_x)

        if token_ids is None:
            # Average all experts (no routing info available).
            routed = mx.zeros_like(flat_x)
            for e in range(self.num_experts):
                gate_e = flat_x @ self.gate_proj_w[e]
                up_e = flat_x @ self.up_proj_w[e]
                routed = routed + (nn.silu(gate_e) * up_e) @ self.down_proj_w[e]
            routed = routed / self.num_experts
        else:
            # Per-route expert assignments: LSH (deep layers) or lexical table.
            if self.lsh:
                route_ids = self._lsh_route(flat_x)
            else:
                ids = mx.clip(token_ids, 0, self.vocab_size - 1).reshape(-1)
                primary = self.token_to_expert[ids]
                route_ids = [primary]
                route_ids.extend(
                    (primary + k) % self.num_experts
                    for k in range(1, self.top_k)
                )

            routed = mx.zeros_like(flat_x)
            for k in range(self.top_k):
                if self.top_k == 1:
                    weight = 1.0
                else:
                    weight = (
                        self.primary_weight
                        if k == 0
                        else (1.0 - self.primary_weight) / (self.top_k - 1)
                    )
                expert_ids_k = route_ids[k]
                # Group-by-expert with mask multiplication (avoids host syncs).
                routed_k = mx.zeros_like(flat_x)
                for e in range(self.num_experts):
                    mask = (expert_ids_k == e).astype(flat_x.dtype)[:, None]
                    gate_e = flat_x @ self.gate_proj_w[e]
                    up_e = flat_x @ self.up_proj_w[e]
                    out_e = (nn.silu(gate_e) * up_e) @ self.down_proj_w[e]
                    routed_k = routed_k + mask * out_e
                routed = routed + weight * routed_k

        if self.use_output_gates:
            out = self.shared_output_gate * shared_out + self.routed_output_gate * routed
        else:
            out = shared_out + routed
        return out.reshape(B, S, H)


class DenseSwiGLUMLP(nn.Module):
    """Dense baseline MLP with checkpoint-compatible parameter names."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(
            args.hidden_size,
            args.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            args.hidden_size,
            args.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            args.intermediate_size,
            args.hidden_size,
            bias=False,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        token_ids: Optional[mx.array] = None,
    ) -> mx.array:
        del token_ids
        return self.down_proj(
            nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class MuGuidance(nn.Module):
    """mu_contextual = mu + mu_proj(hidden_states)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.mu = mx.zeros((hidden_size,))
        self.mu_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.mu + self.mu_proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int = 0):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.self_attn = Attention(args)
        routed = (
            bool(args.use_token_routed_mlp)
            and args.mlp_type == "token_routed"
            and args.num_experts > 1
        )
        self.mlp = (
            TokenRoutedMLP(args, layer_idx)
            if routed
            else DenseSwiGLUMLP(args)
        )
        self.use_mu_guidance = args.use_mu_guidance
        if self.use_mu_guidance:
            self.mu_guidance = MuGuidance(args.hidden_size)

    def __call__(
        self,
        x: mx.array,
        token_ids: Optional[mx.array] = None,
        mu_prev: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        x = x + self.self_attn(self.input_layernorm(x), mu_prev=mu_prev, mask=mask, cache=cache)
        x = x + self.mlp(self.post_attention_layernorm(x), token_ids=token_ids)
        mu_contextual = self.mu_guidance(x) if self.use_mu_guidance else None
        return x, mu_contextual


class ComplexityInner(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.use_mu_guidance = args.use_mu_guidance
        if self.use_mu_guidance:
            self.mu_init = mx.zeros((args.hidden_size,))

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
    ) -> mx.array:
        x = self.embed_tokens(inputs)

        if mask is None:
            mask = create_attention_mask(x, cache[0] if cache is not None else None)

        mu_prev = None
        if self.use_mu_guidance:
            B, L = inputs.shape
            mu_prev = mx.broadcast_to(self.mu_init, (B, L, x.shape[-1]))

        if cache is None:
            cache = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            x, mu_contextual = layer(
                x, token_ids=inputs, mu_prev=mu_prev, mask=mask, cache=cache[i]
            )
            if mu_contextual is not None:
                mu_prev = mu_contextual

        return self.norm(x)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = ComplexityInner(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
    ) -> mx.array:
        out = self.model(inputs, mask=mask, cache=cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Map the public HF exports into this MLX module tree.

        The public files store the transformer at the root, whereas mlx-lm
        nests it under ``model``. RoPE frequencies are recomputed by MLX.
        Tensor orientation is already compatible for attention, dense MLP,
        shared MLP, and raw expert matrices.
        """

        sanitized: dict[str, mx.array] = {}
        for key, value in weights.items():
            if key.endswith("self_attn.rotary_emb.inv_freq"):
                continue
            mapped = key if key.startswith("model.") else f"model.{key}"
            sanitized[mapped] = value
        return sanitized

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self) -> int:
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self) -> int:
        return self.args.num_key_value_heads
