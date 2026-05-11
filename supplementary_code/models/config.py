"""
Model configuration for the Complexity architecture.

All hyperparameters are collected in a single dataclass.  Preset
configurations provide reproducible model sizes from the paper.
"""

import inspect
from dataclasses import dataclass
from typing import Optional


@dataclass
class ComplexityConfig:
    """
    Configuration for Complexity models.

    The defaults match the 166M-parameter model used in the paper's
    iso-parameter ablations.
    """

    # --- Vocabulary and embeddings ---
    vocab_size: int = 100_000
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4       # GQA groups
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02

    # --- Special tokens ---
    pad_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 0

    # --- Token-Routed MLP ---
    use_token_routed_mlp: bool = True
    num_experts: int = 4
    shared_expert: bool = True                       # Shared Lexical Expert
    shared_intermediate_size: Optional[int] = None   # default = intermediate_size
    top_k: int = 1                                   # routed-expert top-K (>=1)
    top_k_primary_weight: float = 0.95               # primary blend weight when top_k>1
    use_shared_routed_gates: bool = False            # learn scalar gates g_s, g_r
    shared_gate_init: float = 1.0
    routed_gate_init: float = 1.0

    # --- Mu-Guidance ---
    use_mu_guidance: bool = True       # Cross-layer contextual mu

    # --- Attention ---
    use_qk_norm: bool = True           # QK normalization

    # ================================================================
    # Presets
    # ================================================================

    @classmethod
    def complexity_tiny(cls) -> "ComplexityConfig":
        """~15M params (debugging)."""
        return cls(hidden_size=256, intermediate_size=704,
                   num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2)

    @classmethod
    def complexity_20m(cls) -> "ComplexityConfig":
        """~20M params (quick experiments)."""
        return cls(hidden_size=320, intermediate_size=896,
                   num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4)

    @classmethod
    def complexity_small(cls) -> "ComplexityConfig":
        """~50M params."""
        return cls(hidden_size=512, intermediate_size=1408,
                   num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4)

    @classmethod
    def complexity_150m(cls) -> "ComplexityConfig":
        """~166M params (paper's main ablation size)."""
        return cls(hidden_size=768, intermediate_size=2048,
                   num_hidden_layers=12, num_attention_heads=12, num_key_value_heads=4)

    @classmethod
    def complexity_350m(cls) -> "ComplexityConfig":
        """~350M params."""
        return cls(hidden_size=1280, intermediate_size=3456,
                   num_hidden_layers=20, num_attention_heads=16, num_key_value_heads=4)

    @classmethod
    def complexity_300m_tr(cls) -> "ComplexityConfig":
        """Corrected ~300M Token-Routed scaling model (iso-batch run).

        Routed experts are kept small (intermediate_size=256 -> 64 per expert)
        while the Shared Lexical Expert carries the bulk of the FFN capacity
        (shared_intermediate_size=3840). Top-K=2 uses a 50/50 blend between
        primary and cyclic-shifted secondary expert. Learned scalar gates start
        with a strong dense residual path (g_s=1.0) and a small routed residual
        branch (g_r=0.1). Mu-guidance is disabled in the matched 300M scaling
        run.
        """
        return cls(vocab_size=32_000, hidden_size=1024, intermediate_size=256,
                   num_hidden_layers=18, num_attention_heads=16, num_key_value_heads=4,
                   max_position_embeddings=2048, use_token_routed_mlp=True,
                   num_experts=4, shared_expert=True,
                   shared_intermediate_size=3840,
                   top_k=2, top_k_primary_weight=0.5,
                   use_shared_routed_gates=True,
                   shared_gate_init=1.0, routed_gate_init=0.1,
                   use_mu_guidance=False)

    @classmethod
    def complexity_300m_dense(cls) -> "ComplexityConfig":
        """Corrected ~300M dense SwiGLU scaling baseline."""
        return cls(vocab_size=32_000, hidden_size=1024, intermediate_size=4096,
                   num_hidden_layers=18, num_attention_heads=16, num_key_value_heads=4,
                   max_position_embeddings=2048, use_token_routed_mlp=False,
                   shared_expert=False, use_mu_guidance=False)

    @classmethod
    def complexity_1b(cls) -> "ComplexityConfig":
        """~1B params."""
        return cls(hidden_size=2048, intermediate_size=5632,
                   num_hidden_layers=24, num_attention_heads=16, num_key_value_heads=8,
                   num_experts=8)

    # ================================================================
    # Serialization
    # ================================================================

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ComplexityConfig":
        """Create from dictionary, ignoring unknown keys."""
        valid = set(inspect.signature(cls).parameters.keys())
        return cls(**{k: v for k, v in d.items() if k in valid})
