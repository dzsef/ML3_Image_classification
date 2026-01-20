#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZIP_DIR="${ZIP_DIR:-$ROOT/data/_zips/div2k}"
OUT_DIR="${DIV2K_DIR:-$ROOT/data/div2k}"
SCALE="${SCALE:-2}"
DOWNLOAD_LR="${DOWNLOAD_LR:-0}"
BASE="https://data.vision.ee.ethz.ch/cvl/DIV2K"

mkdir -p "$ZIP_DIR" \
         "$OUT_DIR/HR/train" "$OUT_DIR/HR/val" \
         "$OUT_DIR/LR_bicubic/X${SCALE}/train" "$OUT_DIR/LR_bicubic/X${SCALE}/val"

fetch() {
  local url="$1" dst="$2"
  [[ -f "$dst" ]] && return 0
  command -v wget >/dev/null 2>&1 && wget -c "$url" -O "$dst" || curl -L "$url" -o "$dst"
}

unpack_into() {
  local zip="$1" target="$2" tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN
  unzip -q "$zip" -d "$tmp"
  rsync -a --delete "$(find "$tmp" -mindepth 1 -maxdepth 1 -type d | head -n 1)/" "$target/"
}

fetch "$BASE/DIV2K_train_HR.zip"  "$ZIP_DIR/DIV2K_train_HR.zip"
fetch "$BASE/DIV2K_valid_HR.zip"  "$ZIP_DIR/DIV2K_valid_HR.zip"
unpack_into "$ZIP_DIR/DIV2K_train_HR.zip" "$OUT_DIR/HR/train"
unpack_into "$ZIP_DIR/DIV2K_valid_HR.zip" "$OUT_DIR/HR/val"

if [[ "$DOWNLOAD_LR" == "1" ]]; then
  fetch "$BASE/DIV2K_train_LR_bicubic_X${SCALE}.zip" "$ZIP_DIR/DIV2K_train_LR_bicubic_X${SCALE}.zip"
  fetch "$BASE/DIV2K_valid_LR_bicubic_X${SCALE}.zip" "$ZIP_DIR/DIV2K_valid_LR_bicubic_X${SCALE}.zip"
  unpack_into "$ZIP_DIR/DIV2K_train_LR_bicubic_X${SCALE}.zip" "$OUT_DIR/LR_bicubic/X${SCALE}/train"
  unpack_into "$ZIP_DIR/DIV2K_valid_LR_bicubic_X${SCALE}.zip" "$OUT_DIR/LR_bicubic/X${SCALE}/val"
fi
