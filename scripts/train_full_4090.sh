#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
data_root="${DATA_ROOT:-/root/autodl-tmp/wh5319/data/mosei_full}"
encoder_root="${ENCODER_ROOT:-$project_root/models/bert-mini}"
run_root="${RUN_ROOT:-$project_root/runs/mosei_full_s2026}"

cd "$project_root"
python src/train.py \
  --data-root "$data_root" \
  --encoder "$encoder_root" \
  --output-dir "$run_root" \
  --device cuda
