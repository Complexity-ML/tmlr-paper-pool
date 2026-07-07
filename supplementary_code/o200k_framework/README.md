# o200k framework snapshot used for reported runs

This directory is a source snapshot of the `complexity-framework` training harness used for the current o200k Token-Routed experiments and ablations.

Source commit:

```text
anonymous-snapshot
feat: add H200 1B ablation launcher
```

## What is included

- `complexity/`: model, Token-Routed MLP, o200k pretraining CLI, dataset streaming, CUDA/Triton fallbacks.
- `configs/run_configs/ablations_100m/`: seven 100M ablation configs.
- `scripts/ablations_100m/`: local diagnostics and H200 1B launcher.
- `configs/run_configs/300m_o200k_tr_rocm_scale.yaml`: 300M o200k TR scale config.
- `scripts/train_100m_o200k_tr_local.py`, `scripts/train_300m_tr_local.py`, `scripts/train_300m_dense_local.py`.

## Critical routing fix included

`complexity/core/mlp/token_routed.py` preserves the requested control strategy for top-k auxiliary routes:

- Zipf uses Zipf-balanced auxiliary routes.
- Random uses deterministic random auxiliary routes distinct from the primary route.
- Modulo and round-robin use shifted auxiliary routes.

This prevents the previous invalid control where `random/modulo/round_robin` primary routes could receive Zipf-balanced auxiliary routes.

## 100M ablations

The seven ablations are:

```text
100m_zipf_shared
100m_zipf_no_shared
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

Expected result in the source repository at the snapshot commit:

```text
5 passed
```
