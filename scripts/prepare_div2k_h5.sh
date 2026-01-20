#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$(command -v python3)"

DIV2K_DIR="${DIV2K_DIR:-$ROOT/data/div2k}"
OUT_DIR="${OUT_DIR:-$ROOT/artifacts/h5}"
SCALE="${SCALE:-2}"
PATCH_SIZE="${PATCH_SIZE:-33}"
STRIDE="${STRIDE:-64}"

mkdir -p "$OUT_DIR"

"$PY" "$ROOT/third_party/srcnn_pytorch/prepare.py" \
  --images-dir "$DIV2K_DIR/HR/train" \
  --output-path "$OUT_DIR/div2k_train_x${SCALE}.h5" \
  --scale "$SCALE" \
  --patch-size "$PATCH_SIZE" \
  --stride "$STRIDE"

"$PY" "$ROOT/third_party/srcnn_pytorch/prepare.py" \
  --images-dir "$DIV2K_DIR/HR/val" \
  --output-path "$OUT_DIR/div2k_val_x${SCALE}.h5" \
  --scale "$SCALE" \
  --eval
