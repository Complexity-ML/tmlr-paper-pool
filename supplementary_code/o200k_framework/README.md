# o200k framework snapshot used for reported runs

This directory is an audit-corrected snapshot of the `complexity-framework` training harness used for the reported o200k Token-Routed experiments and ablations.

Source commit:

```text
anonymous-snapshot
feat: add H200 1B ablation launcher
```

## What is included

- `complexity/`: model, Token-Routed MLP, o200k pretraining CLI, dataset streaming, CUDA/Triton fallbacks.
- `configs/run_configs/ablations_100m/`: seven 100M ablation configs.
- `scripts/ablations_100m/`: local diagnostics and H200 1B launcher.
- `scripts/train_100m_o200k_tr_local.py`, `scripts/train_300m_tr_local.py`, `scripts/train_300m_dense_local.py`.

The sole reported 300M Token-Routed entrypoint is `scripts/train_300m_tr_local.py`. Its defaults are fixed to the verified run configuration: the included `tokenizer-32k/` asset, vocabulary 32,000, hidden size 1,024, 18 layers, GQA 16/4, shared width 3,840, four routed experts of width 64, `modulo_balanced_secondary`, deterministic top-2 with fixed 0.5/0.5 expert weights, and learned shared/routed branch gates initialized to 1.0/0.1. Startup rejects any tokenizer whose vocabulary is not exactly 32,000. The unrelated legacy o200k 300M profile is intentionally not included as a run config.

For the full reported command rather than smoke-test defaults, run `scripts/run_verified_300m_8xb300.sh tr` and `scripts/run_verified_300m_8xb300.sh dense` on an 8-GPU node. The launcher records the 7,620-step, batch-64/GPU, sequence-2,048, BF16, seed-42 configuration. The evaluation loader uses a fixed stream from the FineWeb-Edu `train` split and is not an independent held-out validation set.

## Verified routing behavior

`complexity/core/mlp/token_routed.py` uses fixed token-ID lookup tables; there is no learned expert-selection router. The routing controls are:

- `modulo_balanced_secondary` uses a permuted modulo primary route and a deterministic load-balanced secondary route. This explicitly names the lookup realised by the reported runs.
- Random uses deterministic random auxiliary routes distinct from the primary route.
- Modulo and round-robin use adjacent shifted auxiliary routes.
- Explicit `zipf` requires a token-frequency table and fails loudly when it is absent.

Top-2 expert weights are fixed at 0.5/0.5 in the reported configurations. The learned shared/routed gates are branch-level scalars, not routing logits. This corrected snapshot prevents both silent strategy fallback and accidental use of frequency-balanced auxiliary routes in non-frequency controls.

## 100M ablations

The seven ablations are:

```text
100m_modulo_balanced_secondary_shared
100m_modulo_balanced_secondary_no_shared
100m_modulo_shared
100m_random_shared
100m_round_robin_shared
100m_shared_only
100m_dense_residual
```

The 1B-token launcher is:

```bash
cd supplementary_code/o200k_framework
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
python -m pip install -e '.[cuda,dev]'
PYTHON=python scripts/ablations_100m/run_h200_1b_all.sh
```

Budget per ablation:

```text
1908 steps × batch 256 × seq 2048 = 1.000341504B tokens
```

## Verification

After installation:

```bash
PYTHONPATH=. pytest tests/test_100m_ablation_configs.py -q
```

Expected result:

```text
8 passed
```
