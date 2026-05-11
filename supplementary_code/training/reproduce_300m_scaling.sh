#!/usr/bin/env bash
set -euo pipefail

# Run from the corrected complexity-framework repository.
# Dense and Token-Routed use the same tokenizer, sequence length, optimizer
# recipe, FineWeb-Edu stream, global batch (1,048,576 tokens/step), and 8B-token
# budget. Matched raw step number therefore equals matched tokens seen.

case "${1:-}" in
  dense)
    torchrun --nproc_per_node="${NPROC_PER_NODE:-8}" scripts/train_300m_dense_local.py \
      --dataset fineweb \
      --tokenizer ./tokenizer \
      --steps 7629 \
      --batch-size 64 \
      --seq-len 2048 \
      --bf16 \
      --grad-ckpt \
      --eval-steps 250 \
      --eval-batches 32 \
      --log-steps 20 \
      --save-steps 500 \
      --save-dir checkpoints/8b-300m-dense \
      --save-total-limit 3 \
      --run-name 8b-300m-dense
    ;;
  tr)
    torchrun --nproc_per_node="${NPROC_PER_NODE:-8}" scripts/train_300m_tr_local.py \
      --dataset fineweb \
      --tokenizer ./tokenizer \
      --steps 7629 \
      --batch-size 64 \
      --seq-len 2048 \
      --bf16 \
      --grad-ckpt \
      --eval-steps 250 \
      --eval-batches 32 \
      --log-steps 20 \
      --intermediate-size 256 \
      --shared-intermediate-size 3840 \
      --top-k 2 \
      --top-k-primary-weight 0.5 \
      --shared-gate-init 1.0 \
      --routed-gate-init 0.1 \
      --save-steps 500 \
      --save-dir checkpoints/8b-300m-tr \
      --save-total-limit 3 \
      --run-name 8b-300m-tr
    ;;
  *)
    echo "usage: $0 {dense|tr}" >&2
    exit 2
    ;;
esac
