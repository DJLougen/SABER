#!/usr/bin/env bash
set -euo pipefail

# Small SABER probe. Use this to verify setup before running a larger sweep.
python run_saber.py \
  --model google/gemma-4-E2B-it \
  --output-dir runs/gemma4_e2b_probe \
  --extraction-method svd \
  --n-harmful 30 \
  --n-harmless 30 \
  --n-capability 30 \
  --alpha-base 0.8 \
  --layer-top-k 8
