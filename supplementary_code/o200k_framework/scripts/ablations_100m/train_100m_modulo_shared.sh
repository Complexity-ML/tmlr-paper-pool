#!/usr/bin/env bash
set -euo pipefail

# 100M ablation: 100m_modulo_shared
# Realized study: ~100M parameters, 954 steps x effective global batch 512 x seq 2048 = 1.0003B tokens.
# Override with extra CLI args after the script, e.g. --steps 10 --dataset random.

python3 scripts/train_100m_o200k_tr_local.py \
  --config configs/run_configs/ablations_100m/100m_modulo_shared.yaml \
  "$@"
