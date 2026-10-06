#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
csd_root="${CSD_ROOT:-/root/autodl-tmp/wh5319/mosei_csd/data}"
sdk_root="${SDK_ROOT:-/root/autodl-tmp/wh5319/CMU-MultimodalSDK}"
tokenizer_root="${TOKENIZER_ROOT:-$project_root/models/bert-mini}"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp/wh5319/data/mosei_full}"

cd "$project_root"
python src/build_full_mosei_from_csd.py \
  --csd-dir "$csd_root" \
  --sdk-root "$sdk_root" \
  --tokenizer "$tokenizer_root" \
  --output-dir "$output_root" \
  --sequence-length 50 \
  --unassigned-policy train \
  --expected-label-rows 23259 \
  --local-files-only
