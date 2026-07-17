#!/usr/bin/env bash
set -euo pipefail

# Four matched 100M controls on 2x RTX PRO 6000.
# Full budget per run: 100 steps x 32 seq/GPU x 2 GPUs x 8 accumulation x 2048 tokens
#                      = 104,857,600 tokens.
# Default mode is a two-step random-data smoke test. Start paid training with:
#   MODE=full SEED=42 scripts/ablations_100m/run_2x_rtx_pro_6000_router_baselines.sh

MODE="${MODE:-smoke}"
SEED="${SEED:-42}"
TOKENS_PATH="${TOKENS_PATH:-/root/data/fineweb_edu_o200k_1p05b}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

RUNS=(
  100m_modulo_balanced_secondary_shared
  100m_dense_residual
  100m_learned_aux_shared
  100m_learned_loss_free_shared
)

case "${MODE}" in
  smoke)
    COMMON_ARGS=(
      --dataset random
      --tokenizer o200k_base
      --vocab-size 200019
      --steps 2
      --batch-size 2
      --seq-len 128
      --eval-steps 0
      --log-steps 1
      --save-steps 0
      --loss-chunk-tokens 128
    )
    PREFIX="rtxpro2-smoke-s${SEED}"
    ;;
  full)
    if [[ ! -f "${TOKENS_PATH}/tokens.idx.json" || ! -f "${TOKENS_PATH}/tokens.bin" ]]; then
      echo "Verified token shard missing at ${TOKENS_PATH}" >&2
      echo "Run scripts/prepare_fineweb_o200k_shard.py first." >&2
      exit 3
    fi
    COMMON_ARGS=(
      --dataset tokens
      --tokens-path "${TOKENS_PATH}"
      --tokenizer o200k_base
      --vocab-size 200019
      --steps 100
      --batch-size 32
      --gradient-accumulation-steps 8
      --seq-len 2048
      --loss-backend liger
      --eval-steps 100
      --eval-batches 16
      --log-steps 10
      --save-steps 100
      --loss-chunk-tokens 1024
    )
    PREFIX="rtxpro2-100m-s${SEED}"
    ;;
  *)
    echo "MODE must be smoke or full" >&2
    exit 2
    ;;
esac

python - <<'PY'
import torch
count = torch.cuda.device_count()
if count != 2:
    raise SystemExit(f"Expected exactly 2 CUDA GPUs, found {count}")
for idx in range(count):
    print(idx, torch.cuda.get_device_name(idx))
PY

mkdir -p "runs/${PREFIX}"
for name in "${RUNS[@]}"; do
  run_name="${PREFIX}-${name}"
  echo "=== ${run_name} ==="
  torchrun --standalone --nproc_per_node=2 scripts/train_100m_o200k_tr_local.py \
    --config "configs/run_configs/ablations_100m/${name}.yaml" \
    --seed "${SEED}" \
    --run-name "${run_name}" \
    --save-dir "checkpoints/${run_name}" \
    "${COMMON_ARGS[@]}" \
    "$@" \
    2>&1 | tee "runs/${PREFIX}/${name}.log"
done
