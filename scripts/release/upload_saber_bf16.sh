#!/usr/bin/env bash
# Upload bf16 SABER safetensors + model card to HuggingFace.
set -euo pipefail

SRC=/home/djl/ornstein3.6-saber/src
REPO=DJLougen/Ornstein3.6-35B-A3B-SABER
README_SRC=/tmp/saber/ornstein36_saber_readme.md

echo "=== Waiting for SABER to finish (saber_result.json) ==="
while [[ ! -s /home/djl/ornstein3.6-saber/saber_result.json ]]; do sleep 30; done
echo "SABER finished at $(date)"
cat /home/djl/ornstein3.6-saber/saber_result.json | head -40

echo "=== Staging README + result manifest into src ==="
cp "$README_SRC" "$SRC/README.md"
cp /home/djl/ornstein3.6-saber/saber_result.json "$SRC/saber_result.json"

echo "=== Creating repo (if needed) ==="
conda run -n torch hf repo create "$REPO" --repo-type model --private false || true

echo "=== Uploading full folder to $REPO ==="
conda run -n torch hf upload "$REPO" "$SRC" . \
  --repo-type model \
  --commit-message "Initial SABER release: bf16 safetensors + card"

echo "=== Upload done at $(date) ==="
