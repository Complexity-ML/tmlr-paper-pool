# Independent held-out language-model evaluation

This directory records paired negative-log-likelihood evaluations of the final
Dense-306 and TR-MOE-306 checkpoints. No model weights are updated.

## Corpora

- The Pile: `EleutherAI/pile_val_test`, test split, revision
  `05b327037e6301f256d8df32193756edc4c8e3bd`, first 191 documents.
- C4: `allenai/c4`, English configuration, validation split, revision
  `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`, first 529 documents.
- Each corpus contributes 262,144 scored targets in contiguous blocks of 2,048
  tokens, packed with the exact checkpoint tokenizer and separated by EOS.

Both selections are independent of the declared FineWeb-Edu training stream.
No document-level decontamination against FineWeb-Edu was performed, so the
result is reported as an independent-corpus evaluation rather than proof that
no source text could overlap.

## Primary reproduction with Apple MLX

The manuscript-facing result should be generated with Apple MLX/Metal, loading
the public safetensors directly and bypassing the project-specific vllm-i64
runtime:

```bash
PYTHONPATH=/path/to/mlx-lm python3 \
  supplementary_code/evaluation/run_heldout_nll_mlx.py \
  --dense-model /path/to/Dense-306 \
  --routed-model /path/to/TR-MOE-306 \
  --tokenizer /path/to/Dense-306/tokenizer.json \
  --output supplementary_code/results/300m_heldout/pile_test_results_mlx.json
```

The bundled `evaluation/mlx_complexity_306.py` implementation supports both the
dense SwiGLU baseline and the routed residual model, uses the exported primary
route, and derives the second route cyclically as specified by the evaluation
runtime. Its SHA-256 is written into the result JSON.

The recorded MLX results are:

| Corpus | Dense NLL | Routed NLL | Routed minus dense (95% CI) |
| --- | ---: | ---: | ---: |
| C4 validation | 3.4161 | **3.4066** | **-0.0095** [-0.0127, -0.0063] |
| The Pile test | **2.8690** | 2.8769 | +0.0079 [+0.0022, +0.0138] |

The ordering changes by corpus; no uniform held-out superiority is claimed.

## Cross-engine reference

An optional PyTorch CPU/CUDA/MPS reference can be produced with `vllm-i64`
installed or on `PYTHONPATH`:

```bash
PYTHONPATH=/path/to/vllm-i64 python3 \
  supplementary_code/evaluation/run_heldout_nll.py \
  --dense-model /path/to/Dense-306 \
  --routed-model /path/to/TR-MOE-306 \
  --tokenizer /path/to/Dense-306/tokenizer.json \
  --device cpu \
  --output supplementary_code/results/300m_heldout/pile_test_results_vllm_i64.json
```

Both JSON outputs contain aggregate NLL and perplexity, per-block paired values,
model and tokenizer hashes, runtime information, and a deterministic
10,000-sample paired-bootstrap 95% confidence interval. Agreement between the
MLX result and the optional reference is an implementation cross-check; the MLX
result is the primary held-out measurement.
