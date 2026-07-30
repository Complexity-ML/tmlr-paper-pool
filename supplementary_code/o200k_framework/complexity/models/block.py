"""
Transformer Block - the basic building unit.

A block consists of:
1. Attention (with pre-norm)
2. MLP/FFN (with pre-norm)
3. Residual connections
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from ..config import ModelConfig
from ..core.attention import AttentionConfig
from ..core.mlp import MLPConfig
from ..core.registry import ATTENTION_REGISTRY, MLP_REGISTRY, NORMALIZATION_REGISTRY



class TransformerBlock(nn.Module):
    """
    Single Transformer block with configurable components.

    Architecture (Pre-Norm):
        x = x + attention(norm1(x))
        x = x + mlp(norm2(x))
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Pre-attention normalization
        self.input_layernorm = NORMALIZATION_REGISTRY.build(
            config.norm_type,
            config.hidden_size,
            eps=config.norm_eps,
        )

        # Attention
        attn_config = AttentionConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            attention_dropout=config.attention_dropout,
            use_qk_norm=config.use_qk_norm,
            sliding_window=config.sliding_window,
            use_sdpa=config.use_sdpa,
            rope_type=config.rope_type,
            use_mup_attn_scale=getattr(config, "use_mup_attn_scale", False),
        )
        self.self_attn = ATTENTION_REGISTRY.build(config.attention_type, attn_config)

        # Post-attention normalization
        self.post_attention_layernorm = NORMALIZATION_REGISTRY.build(
            config.norm_type,
            config.hidden_size,
            eps=config.norm_eps,
        )

        # MLP
        mlp_config = MLPConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            num_experts=config.num_experts,
            expert_initialization=getattr(
                config, "expert_initialization", "gpt_normal"
            ),
            initializer_range=getattr(config, "initializer_range", 0.02),
            vocab_size=config.vocab_size,
            routing_strategy=getattr(config, 'routing_strategy', 'modulo_cyclic'),
            token_frequencies=config.token_frequencies,
            lsh_routing=getattr(config, 'lsh_routing', False),
            lsh_bits=getattr(config, 'lsh_bits', 0),
            lsh_from_layer=getattr(config, 'lsh_from_layer', 0),
            lsh_threshold_mode=getattr(config, 'lsh_threshold_mode', 'zero'),
            shared_expert=getattr(config, 'shared_expert', False),
            shared_intermediate_size=getattr(config, 'shared_intermediate_size', None),
            shared_expert_chunk_tokens=getattr(config, 'shared_expert_chunk_tokens', 0),
            use_shared_routed_gates=getattr(config, 'use_shared_routed_gates', False),
            shared_gate_init=getattr(config, 'shared_gate_init', 1.0),
            routed_gate_init=getattr(config, 'routed_gate_init', 1.0),
            top_k=getattr(config, 'top_k', 1),
            top_k_primary_weight=getattr(config, 'top_k_primary_weight', None),
            router_balance_mode=getattr(config, 'router_balance_mode', 'aux_loss'),
            router_aux_loss_coef=getattr(config, 'router_aux_loss_coef', 0.01),
            router_bias_update_rate=getattr(config, 'router_bias_update_rate', 0.001),
            layer_idx=layer_idx,
            static_expert_capacity=getattr(config, 'static_expert_capacity', False),
            collect_moe_telemetry=getattr(config, 'collect_moe_telemetry', False),
            use_custom_kernels=getattr(config, 'use_custom_kernels', 'auto'),
            use_cggr=getattr(config, 'use_cggr', False),
        )
        self.mlp = MLP_REGISTRY.build(config.mlp_type, mlp_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        token_ids: Optional[torch.Tensor] = None,
        velocity_state: Optional[torch.Tensor] = None,
        sort_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through the transformer block.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            past_key_value: Optional KV cache
            use_cache: Whether to return updated KV cache
            token_ids: Optional token IDs for MoE routing
            velocity_state: Unused (kept for backward compat)
            sort_idx: Unused (sort_idx computed internally by token_routed)

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
            past_key_value: Optional updated KV cache
            velocity_state: None (kept for backward compat)
        """
        residual = hidden_states

        # Self Attention
        hidden_states = self.input_layernorm(hidden_states)

        attn_kwargs = dict(
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states, new_kv = self.self_attn(hidden_states, **attn_kwargs)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, token_ids=token_ids)
        hidden_states = residual + hidden_states

        return hidden_states, new_kv, None, None
