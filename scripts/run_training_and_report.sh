#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-transformer_200ep_$(date +%Y%m%d-%H%M%S)}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SAVE_EVERY="${SAVE_EVERY:-10}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
LR="${LR:-0.0001}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv-mac/bin/python}"
RESULTS_DIR="results/${RUN_NAME}"
CHECKPOINT_DIR="${RESULTS_DIR}/checkpoints"
REPORT_CSV="results/proposed_metrics_report.csv"

TRAIN_COMMAND="${PYTHON_BIN} train_colab.py --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --lr ${LR} --save-every ${SAVE_EVERY} --num-samples ${NUM_SAMPLES} --checkpoint-dir ${CHECKPOINT_DIR} --results-dir ${RESULTS_DIR}"
TEST_COMMAND="${PYTHON_BIN} scripts/export_proposed_metrics_report.py --run-name ${RUN_NAME} --training-csv ${RESULTS_DIR}/training_metrics.csv --training-summary ${RESULTS_DIR}/summary.json --train-samples ${NUM_SAMPLES} --out ${REPORT_CSV}"

echo "== Entrenando ${RUN_NAME} =="
echo "$TRAIN_COMMAND"
eval "$TRAIN_COMMAND"

echo "== Actualizando reporte unificado train/test =="
echo "$TEST_COMMAND"
eval "$TEST_COMMAND"

echo "== Listo =="
echo "Entrenamiento: ${RESULTS_DIR}"
echo "CSV unificado: ${REPORT_CSV}"
