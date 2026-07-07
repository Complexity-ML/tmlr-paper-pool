# Supplementary code

This archive contains the anonymized o200k training harness used for the reported Token-Routed lexical residual experiments.

```text
supplementary_code/o200k_framework/
```

The previous compact 32k reference implementation has been removed from this review artifact to avoid presenting stale code paths that do not correspond to the reported runs.

## Included code path

`o200k_framework/` includes:

- `complexity/`: model and training package.
- `scripts/train_100m_o200k_tr_local.py`.
- `scripts/train_300m_tr_local.py` and `scripts/train_300m_dense_local.py`.
- `configs/run_configs/ablations_100m/`: seven 100M ablation configs.
- `configs/run_configs/300m_o200k_tr_rocm_scale.yaml`.
- `scripts/ablations_100m/`: local diagnostics and 1B-token ablation launchers.
- `tests/test_100m_ablation_configs.py`.
- `tokenizer-o200k/tiktoken_config.json` for `o200k_base`.

The implementation in this artifact is the no-guidance lexical-routing code path used by the reported runs.

## Verify routing-control behavior

```bash
cd supplementary_code/o200k_framework
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
python -m pip install -e '.[cuda,dev]'
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
100m_dense_residual
100m_zipf_no_shared
100m_modulo_shared
100m_random_shared
100m_round_robin_shared
100m_shared_only
```

Each 1B-token run uses:

```text
1908 steps × batch 256 × seq 2048 = 1.000341504B tokens
```

## License

CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0)
