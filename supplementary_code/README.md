# COMPLEXITY-DEEP: Supplementary Code

This repository contains the corrected source code for the COMPLEXITY-DEEP
architecture described in the paper. The supplementary code now reflects the
residual Token-Routed implementation used in the 300M scaling run, where the
Token-Routed model starts behind dense, crosses over after early specialization,
and finishes ahead at matched tokens seen.

## Structure

```
.
├── core/                     # Core model components
│   ├── attention.py          # Mu-Guided Attention implementation
│   ├── mlp.py                # SwiGLU MLP components
│   ├── layer.py              # Decoder layer with all components
│   ├── token_routed_mlp.py   # Token-Routed MLP (deterministic routing)
│   ├── normalization.py      # RMSNorm implementation
│   ├── rotary.py             # Rotary Position Embeddings (RoPE)
│   └── safety.py             # Safety clamping mechanisms
├── models/
│   ├── config.py             # Model configuration
│   ├── modeling.py           # Full model implementation
│   └── utils.py              # Utility functions
├── cuda/                     # CUDA/Triton optimizations
│   ├── triton_token_routed.py  # Triton-accelerated Token-Routed MLP
│   ├── triton_mu_qkv.py        # Triton Mu-guided attention
│   ├── fused_attention.py      # Fused attention kernels
│   ├── fused_mlp.py            # Fused MLP kernels
│   └── persistent_cggr.py      # Persistent CGGR optimization
├── training/
│   ├── train_complexity.py   # Training script
│   └── online_self_rl.py     # Importable inference-time self-RL module
├── evaluation/
│   └── run_benchmarks.py     # Benchmark evaluation script
└── configs/
    └── model_config.json     # Model configuration
```

## Key Components

### Token-Routed MLP
Deterministic routing via `expert_idx = token_id % num_experts`. No load balancing loss required.

### Mu-Guided Attention
Latent state μ from previous layer guides K, Q, V projections, creating bidirectional information flow.

### Corrected Scaling Presets
The code includes iso-parameter 300M presets:
- `300m_dense`: 306.5M dense SwiGLU baseline.
- `300m_tr`: 306.5M residual Token-Routed with a 3840-wide shared expert,
  4 small routed experts, top-k=2, shared/routed gate init 1.0/0.1, and
  Mu-Guidance disabled for the matched scaling run.

The old dynamic-controller path is not part of the corrected architecture.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers
- datasets
- tqdm
- triton (optional, for CUDA optimizations)

## Usage

### Training
```bash
python training/train_complexity.py --size 150m --dataset your_dataset
```

### Inference-time full-model self-RL
The framework includes an importable online module for serving-time
self-reinforcement.  It is meant to be called by an inference server when the
model hesitates, has high entropy / low confidence, or receives corrective user
feedback.

```python
from supplementary_code.training.online_self_rl import (
    OnlineSelfRLEngine,
    OnlineSelfRLConfig,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)
engine = OnlineSelfRLEngine(
    model=model,
    tokenizer=tokenizer,
    optimizer=optimizer,
    config=OnlineSelfRLConfig(entropy_trigger=5.0, confidence_trigger=0.20),
)

response, stats, episode = engine.infer_and_maybe_update(
    "Explain why this code is failing.",
    user_feedback="wrong",   # optional; also triggers on uncertainty
)
```

The online engine updates the full model on the current inference episode when
the model is uncertain or receives corrective feedback.

Corrected 300M scaling commands used for the paper comparison from the
`complexity-framework` training repo are mirrored in
`training/reproduce_300m_scaling.sh`:

```bash
# Dense baseline: 8 GPUs, 1,048,576 tokens/step
torchrun --nproc_per_node=8 scripts/train_300m_dense_local.py \
  --dataset fineweb --tokenizer ./tokenizer --steps 7629 \
  --batch-size 64 --seq-len 2048 --bf16 --grad-ckpt \
  --eval-steps 250 --eval-batches 32 --log-steps 20 \
  --save-steps 500 --save-dir checkpoints/8b-300m-dense \
  --save-total-limit 3 --run-name 8b-300m-dense

# Token-Routed: 8 GPUs, 1,048,576 tokens/step (iso-batch with dense)
torchrun --nproc_per_node=8 scripts/train_300m_tr_local.py \
  --dataset fineweb --tokenizer ./tokenizer --steps 7629 \
  --batch-size 64 --seq-len 2048 --bf16 --grad-ckpt \
  --eval-steps 250 --eval-batches 32 --log-steps 20 \
  --intermediate-size 256 --shared-intermediate-size 3840 \
  --top-k 2 --top-k-primary-weight 0.5 \
  --shared-gate-init 1.0 --routed-gate-init 0.1 \
  --save-steps 500 --save-dir checkpoints/8b-300m-tr \
  --save-total-limit 3 --run-name 8b-300m-tr
```

Both runs use the same global batch (1,048,576 tokens/step) so matched raw
step number equals matched tokens seen. The Token-Routed model trails the dense
baseline during the early specialization phase (peak gap +0.31 around step 40)
and first crosses over at logged train step 740 and validation step 750. At
step 1000 (≈1.05B tokens), Token-Routed train loss is 3.5324 versus 3.5500 for
dense (gap −0.018 in favor of Token-Routed).
Expert utilization remains balanced throughout (≈0.248/0.264/0.248/0.240, zero
dead experts).

### Evaluation
```bash
python evaluation/run_benchmarks.py --checkpoint path/to/checkpoint.pt --max-samples 500
```

## License

CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0)
