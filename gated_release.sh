#!/usr/bin/env bash
set -euo pipefail

while pgrep -f /tmp/saber/continue_saber_35b.py >/dev/null; do
  sleep 30
done

cd /tmp/saber
conda run -n torch python /tmp/saber/validate_ornstein36_zero_refusal.py \
  > /home/djl/ornstein3.6-saber/validation.log 2>&1

cp /home/djl/ornstein3.6-saber/ornstein36_saber_readme.md /home/djl/ornstein3.6-saber/src/README.md
cp /home/djl/ornstein3.6-saber/saber_result.json /home/djl/ornstein3.6-saber/src/saber_result.json

conda run -n torch hf upload DJLougen/Ornstein3.6-35B-A3B-SABER /home/djl/ornstein3.6-saber/src . \
  --repo-type model \
  --commit-message "Refresh SABER after zero-refusal validation gate"
