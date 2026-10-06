#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/wh5319/data/mosei_mmsa_standard}"
ENCODER_ROOT="${ENCODER_ROOT:-models/bert-base-uncased}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_RUN="${BASE_RUN:-runs/mosei_paper_s1111}"
PARETO_RUN="${PARETO_RUN:-runs/mosei_pareto_s1111}"
RESULT_JSON="${RESULT_JSON:-results/mosei_pareto_ensemble_test.json}"

PYTHONPATH=src "${PYTHON_BIN}" src/train.py \
  --data-root "${DATA_ROOT}" \
  --encoder "${ENCODER_ROOT}" \
  --config configs/adatrustmsa_mosei_paper.json \
  --output-dir "${BASE_RUN}" \
  --device cuda

PYTHONPATH=src "${PYTHON_BIN}" src/train.py \
  --data-root "${DATA_ROOT}" \
  --encoder "${ENCODER_ROOT}" \
  --config configs/adatrustmsa_mosei_pareto_finetune.json \
  --output-dir "${PARETO_RUN}" \
  --initialize-from "${BASE_RUN}/best.pt" \
  --device cuda

PYTHONPATH=src "${PYTHON_BIN}" src/evaluate_pareto_ensemble.py \
  --data-root "${DATA_ROOT}" \
  --regression-checkpoint "${BASE_RUN}/best.pt" \
  --classification-checkpoint "${PARETO_RUN}/best.pt" \
  --output "${RESULT_JSON}" \
  --device cuda
