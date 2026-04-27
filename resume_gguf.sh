#!/usr/bin/env bash
# Resume GGUF pipeline: Q6_K upload was killed (stalled). Retry upload then
# continue through the rest of the suite.
set -euo pipefail

OUT=/home/djl/ornstein3.6-saber-quants
BASE=Ornstein3.6-35B-A3B-SABER
REPO=DJLougen/Ornstein3.6-35B-A3B-SABER-GGUF
LLAMA=/home/djl/llama.cpp
Q=$LLAMA/build/bin/llama-quantize
Q8=$OUT/${BASE}-Q8_0.gguf

cd "$OUT"

upload_delete() {
  local suffix="$1"
  local out="$OUT/${BASE}-${suffix}.gguf"
  [[ -f "$out" ]] || { echo "!! missing $out"; return 0; }
  echo ">>> upload $suffix at $(date)"
  # Retry up to 3 times on transient failures
  local tries=0
  until conda run -n torch hf upload "$REPO" "$out" "${BASE}-${suffix}.gguf" \
        --repo-type model --commit-message "Add ${suffix}"; do
    tries=$((tries+1))
    [[ $tries -ge 3 ]] && { echo "!! upload of $suffix failed 3 times"; exit 1; }
    echo "!! upload attempt $tries failed, retrying in 30s..."
    sleep 30
  done
  rm -f "$out"
  echo ">>> deleted $out"
}

quant_upload_delete() {
  local type="$1" suffix="$2"
  local out="$OUT/${BASE}-${suffix}.gguf"
  if [[ ! -f "$out" ]]; then
    echo ">>> quantize $suffix at $(date)"
    "$Q" --allow-requantize "$Q8" "$out" "$type"
  fi
  ls -lh "$out"
  upload_delete "$suffix"
}

# First: finish Q6_K (already on disk from prior run)
upload_delete Q6_K

# Continue with the rest
quant_upload_delete Q5_K_M Q5_K_M
quant_upload_delete Q5_K_S Q5_K_S
quant_upload_delete Q4_K_M Q4_K_M
quant_upload_delete Q4_K_S Q4_K_S
quant_upload_delete Q3_K_M Q3_K_M
quant_upload_delete Q3_K_S Q3_K_S
quant_upload_delete Q2_K   Q2_K

echo "=== Done at $(date). Kept: $Q8 ==="
ls -lh "$OUT"/*.gguf
