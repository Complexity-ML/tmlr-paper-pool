#!/usr/bin/env bash
set -euo pipefail

# Reproduction launcher for the reported matched-token 300M configuration.
# Run from the o200k_framework directory on one 8xB300 node.
# DATASET="fineweb" reads Hugging Face FineWeb-Edu sample-10BT, split=train.

MODEL="${1:-tr}"
COMMON=(
  --dataset fineweb
  --tokenizer ./tokenizer-32k
  --steps 7629
  --batch-size 64
  --seq-len 2048
  --lr 3e-4
  --seed 42
  --bf16
  --log-steps 10
  --eval-steps 250
  --eval-batches 8
  --save-steps 1000
  --save-total-limit 3
)

case "$MODEL" in
  tr)
    torchrun --standalone --nproc_per_node=8 scripts/train_300m_tr_local.py \
      "${COMMON[@]}" \
      --intermediate-size 256 \
      --shared-intermediate-size 3840 \
      --shared-gate-init 0.5 \
      --routed-gate-init 0.5 \
      --top-k 2 \
      --top-k-primary-weight 0.5 \
      --run-name 300m-tr-verified-b300 \
      --save-dir checkpoints/300m-tr-verified-b300
    ;;
  dense)
    torchrun --standalone --nproc_per_node=8 scripts/train_300m_dense_local.py \
      "${COMMON[@]}" \
      --run-name 300m-dense-verified-b300 \
      --save-dir checkpoints/300m-dense-verified-b300
    ;;
  *)
    echo "usage: $0 {tr|dense}" >&2
    exit 2
    ;;
esac
