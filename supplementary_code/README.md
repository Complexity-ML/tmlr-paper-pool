# COMPLEXITY-DEEP supplementary code

This archive contains two code paths:

1. `o200k_framework/` — the code path used for the current reported o200k Token-Routed runs and ablations.
2. The legacy files directly under `supplementary_code/` (`core/`, `models/`, `training/`, etc.) — a compact 32k-tokenizer reference implementation kept for architecture readability/backward compatibility.

For reproducing the reported o200k experiments, use `o200k_framework/`.

## Reported-run code path

```text
supplementary_code/o200k_framework/
```

This is a source snapshot of `complexity-framework` commit:

```text
anonymous-snapshot
feat: add H200 1B ablation launcher
```

It includes:

- `complexity/` model and training package.
- `scripts/train_100m_o200k_tr_local.py`.
- `scripts/ablations_100m/` local/H200 ablation launchers.
- `configs/run_configs/ablations_100m/` seven 100M ablation configs.
- `configs/run_configs/300m_o200k_tr_rocm_scale.yaml`.
- `tests/test_100m_ablation_configs.py`.

The included Token-Routed implementation contains the top-k auxiliary routing fix used by the reported ablations: random/modulo/round-robin controls do not receive Zipf-balanced auxiliary routes.

## Quick install for o200k framework snapshot

```bash
cd supplementary_code/o200k_framework
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
python -m pip install -e '.[cuda,dev]'
```

For CPU-only inspection, install a CPU PyTorch wheel instead of the CUDA wheel.

## Verify routing-control behavior

```bash
cd supplementary_code/o200k_framework
. .venv/bin/activate
PYTHONPATH=. pytest tests/test_100m_ablation_configs.py -q
```

Expected:

```text
5 passed
```

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

On one H200, the provided launcher runs 1B tokens per ablation:

```bash
cd supplementary_code/o200k_framework
. .venv/bin/activate
PYTHON=python scripts/ablations_100m/run_h200_1b_all.sh
```

Budget per run:

```text
1908 steps × batch 256 × seq 2048 = 1.000341504B tokens
```

## Legacy 32k reference code

The older compact implementation under `supplementary_code/core`, `supplementary_code/models`, and `supplementary_code/training` is not the source of the current o200k ablation results. It is retained as a smaller readable reference implementation.

Important differences from the reported-run path:

- 32k tokenizer instead of o200k.
- No random/modulo/round-robin ablation harness.
- Simpler top-k scheme.
- Different model-size accounting because of vocabulary size.

Do not use the legacy 32k path to reproduce the o200k ablation tables.

## License

CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0)
