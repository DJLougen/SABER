#!/usr/bin/env bash
# Convert SABER bf16 to GGUF, quantize 8-bit and under, upload each incrementally.
set -euo pipefail

SRC=/home/djl/ornstein3.6-saber/src
OUT=/home/djl/ornstein3.6-saber-quants
BASE=Ornstein3.6-35B-A3B-SABER
REPO=DJLougen/Ornstein3.6-35B-A3B-SABER-GGUF
LLAMA=/home/djl/llama.cpp
CONVERT=$LLAMA/convert_hf_to_gguf.py
Q=$LLAMA/build/bin/llama-quantize

BF16=$OUT/${BASE}-BF16.gguf
Q8=$OUT/${BASE}-Q8_0.gguf

mkdir -p "$OUT"
cd "$OUT"

echo "=== Waiting for SABER safetensors ==="
while [[ ! -s /home/djl/ornstein3.6-saber/saber_result.json ]]; do sleep 30; done
while ! ls "$SRC"/model-*.safetensors >/dev/null 2>&1; do sleep 30; done
echo "SABER safetensors ready at $(date)"
du -sh "$SRC"

echo "=== [1] Create GGUF repo (if needed) ==="
conda run -n torch hf repo create "$REPO" --repo-type model --private false || true

echo "=== [2] Convert bf16 safetensors -> BF16 GGUF ==="
if [[ ! -f "$BF16" ]]; then
  conda run -n torch python3 "$CONVERT" "$SRC" --outfile "$BF16" --outtype bf16
fi
ls -lh "$BF16"

echo "=== [3] Produce Q8_0 from BF16 ==="
if [[ ! -f "$Q8" ]]; then
  "$Q" "$BF16" "$Q8" Q8_0
fi
ls -lh "$Q8"

echo "=== [4] Upload Q8_0 ==="
conda run -n torch hf upload "$REPO" "$Q8" "${BASE}-Q8_0.gguf" \
  --repo-type model --commit-message "Add Q8_0"

echo "=== [5] Drop BF16 to save disk ==="
rm -f "$BF16"

echo "=== [6] Quantize (from Q8), upload, delete — in descending precision ==="
quant_upload_delete() {
  local type="$1" suffix="$2"
  local out="$OUT/${BASE}-${suffix}.gguf"
  echo ">>> quantize $suffix at $(date)"
  if [[ ! -f "$out" ]]; then
    "$Q" --allow-requantize "$Q8" "$out" "$type"
  fi
  ls -lh "$out"
  echo ">>> upload $suffix at $(date)"
  conda run -n torch hf upload "$REPO" "$out" "${BASE}-${suffix}.gguf" \
    --repo-type model --commit-message "Add ${suffix}"
  rm -f "$out"
  echo ">>> deleted $out"
}

quant_upload_delete Q6_K   Q6_K
quant_upload_delete Q5_K_M Q5_K_M
quant_upload_delete Q5_K_S Q5_K_S
quant_upload_delete Q4_K_M Q4_K_M
quant_upload_delete Q4_K_S Q4_K_S
quant_upload_delete Q3_K_M Q3_K_M
quant_upload_delete Q3_K_S Q3_K_S
quant_upload_delete Q2_K   Q2_K

echo "=== Done at $(date). Kept: $Q8 ==="
ls -lh "$OUT"/*.gguf
