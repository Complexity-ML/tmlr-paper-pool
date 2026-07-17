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
- `scripts/run_verified_300m_8xb300.sh`: explicit commands for the reported 7,620-step matched-token pair.
- `configs/run_configs/ablations_100m/`: seven 100M ablation configs.
- `scripts/ablations_100m/`: local diagnostics and 1B-token ablation launchers.
- `tests/test_100m_ablation_configs.py`.
- `tokenizer-o200k/tiktoken_config.json` for `o200k_base`.
- `tokenizer-32k/`: exact 32,000-entry tokenizer required by the verified 300M checkpoint.
- `results/corrected_300m_scaling.csv`: corrected 300M matched-token summary. Its `eval_loss` fields come from a fixed stream drawn from the FineWeb-Edu training split, not an independent held-out validation set.
- `results/100m_raw/*.csv`: seven complete exploratory B200 logs, named after the realized routing mechanisms. Historical frequency-aware run labels are not retained because those runs fell back to modulo-primary/balanced-secondary lookup.

The implementation is an audit-corrected reproduction of the no-guidance lexical-routing path used by the reported runs. It names the realised routing explicitly as `modulo_balanced_secondary`; requesting frequency-aware routing without a token-frequency table now raises an error instead of silently changing strategies. Routed experts are selected by a fixed token-ID lookup, not by a learned router. For top-2 runs, expert outputs use fixed 0.5/0.5 weights; learned shared/routed gates are branch scalars rather than expert-selection logits.

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
8 passed
```

## 100M ablations

The seven ablations are:

```text
100m_modulo_balanced_secondary_shared
100m_dense_residual
100m_modulo_balanced_secondary_no_shared
100m_modulo_shared
100m_random_shared
100m_round_robin_shared
100m_shared_only
```

Each 1B-token run uses:

```text
954 steps × 2 GPUs × batch/GPU 256 × seq 2048 = 1.000341504B tokens
```

## License

CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0)
