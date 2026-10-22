#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

INPUT="${INPUT:-data/example}" # Or data/example.mp4.
OUTPUT="${OUTPUT:-outputs/example}"
BACKEND="${BACKEND:-vggt_omega}"
CHECKPOINT="${CHECKPOINT:-checkpoints/vggt_omega_1b_512.pt}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
SALAD_CHECKPOINT="${SALAD_CHECKPOINT:-checkpoints/dino_salad.ckpt}"
DINOV2_CHECKPOINT="${DINOV2_CHECKPOINT:-checkpoints/dinov2_vitb14_pretrain.pth}"

python demo.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --backend "$BACKEND" \
  --checkpoint "$CHECKPOINT" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --salad-checkpoint "$SALAD_CHECKPOINT" \
  --dinov2-checkpoint "$DINOV2_CHECKPOINT" \
  "$@"
