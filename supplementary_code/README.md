# Supplementary code

This archive contains the anonymized training harness and audit artifacts used for the reported fixed token-identity residual-routing experiments. The primary 300M pair uses the included 32,000-entry tokenizer; the smaller control suites use o200k tokenization.

```text
supplementary_code/o200k_framework/
```

The previous compact 32k reference implementation has been removed from this review artifact to avoid presenting stale code paths that do not correspond to the reported runs.

## Included code path

`o200k_framework/` includes:

- `complexity/`: model and training package.
- `scripts/train_100m_o200k_tr_local.py`.
- `scripts/train_300m_tr_local.py` and `scripts/train_300m_dense_local.py`.
- `scripts/run_verified_300m_8xb300.sh`: explicit commands for the reported 7,629-step matched-token pair. Metrics are logged every ten steps, so the final reported training-NLL row is step 7,620 whereas the evaluated final checkpoints are step 7,629.
- `configs/run_configs/ablations_100m/`: seven 100M ablation configs.
- `scripts/ablations_100m/`: local diagnostics and 1B-token ablation launchers.
- `tests/test_100m_ablation_configs.py`.
- Integer-exact `int64` token-frequency counters used only by the explicit
  frequency-aware ablations. They avoid silent saturation above `2^24`
  repeated occurrences. The primary 300M route and historical Panel A do not
  read token frequencies; Panel B's fixed control does.
- `tokenizer-o200k/tiktoken_config.json` for `o200k_base`.
- `tokenizer-32k/`: exact 32,000-entry tokenizer required by the verified 300M checkpoint.
- `results/corrected_300m_scaling.csv`: corrected 300M matched-token summary. Its `eval_loss` fields come from a fixed stream drawn from the FineWeb-Edu training split, not an independent held-out validation set.
- `scripts/render_verified_architecture.py`: regenerates the two-panel vector architecture figure, showing the controlled causal-GQA backbone and the distinction between fixed parameter selection and contextual computation.
- `scripts/render_300m_scaling_figure.py`: regenerates the paper's vector and raster matched-token training and fixed diagnostic-stream figure from that summary.
- `results/100m_raw/*.csv`: seven complete exploratory B200 logs, named after the realized routing mechanisms. Historical frequency-aware run labels are not retained because those runs fell back to modulo-primary/balanced-secondary lookup. Their launchers intentionally retained no checkpoints, so this panel cannot be rescored post hoc.
- `results/100m_router_short/`: four-condition, 99.6M-token learned-router promotion diagnostic reported as Panel B, with raw metrics, realized configurations, and reconstruction notes.
- `results/1b_router_dense/`: matched 1.0003B-token auxiliary-router and dense-control logs and run configurations reported in the full-budget control.

The primary 300M reproduction uses the explicit `modulo_cyclic` strategy:
the checkpoint-verified permuted-modulo primary route and its cyclic successor,
with no corpus-frequency input. It uses fixed 0.5/0.5 expert weights, routed
width 256 split into four experts of width 64, and learned branch gates
initialized to 0.5/0.5. The historical Panel A controls retain their
separately audited cardinality-balanced secondary strategies, while the
Panel B launcher explicitly selects its corpus-balanced control. Routed experts are selected by a fixed
token-ID lookup, not by a learned router; branch gates are scalars rather than
expert-selection logits.

Independent-corpus evaluation is provided in `evaluation/` and `results/300m_heldout/`. The manuscript-facing measurements use Apple MLX/Metal and contain per-block values for pinned C4-validation and Pile-test subsets.

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
10 passed
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
