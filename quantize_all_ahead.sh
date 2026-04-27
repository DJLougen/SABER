#!/usr/bin/env bash
# Parallel quantize-ahead producer. Writes all K-quants to disk so the
# upload loop never has to wait for llama-quantize.
set -euo pipefail

OUT=/home/djl/ornstein3.6-saber-quants
BASE=Ornstein3.6-35B-A3B-SABER
Q=/home/djl/llama.cpp/build/bin/llama-quantize
Q8=$OUT/${BASE}-Q8_0.gguf

quant_ahead() {
  local type="$1" suffix="$2"
  local out="$OUT/${BASE}-${suffix}.gguf"
  if [[ -f "$out" ]]; then
    echo "[skip] $suffix already on disk"
    return
  fi
  echo ">>> [ahead] quantize $suffix at $(date)"
  "$Q" --allow-requantize "$Q8" "$out" "$type"
  ls -lh "$out"
}

# Q6_K already exists from the prior run; skip it.
quant_ahead Q5_K_M Q5_K_M
quant_ahead Q5_K_S Q5_K_S
quant_ahead Q4_K_M Q4_K_M
quant_ahead Q4_K_S Q4_K_S
quant_ahead Q3_K_M Q3_K_M
quant_ahead Q3_K_S Q3_K_S
quant_ahead Q2_K   Q2_K

echo "=== [ahead] all quants produced at $(date) ==="
ls -lh "$OUT"/*.gguf
