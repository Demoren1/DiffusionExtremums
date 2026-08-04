#!/usr/bin/env bash
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-ras}"
cd /home/udeneev-av/DiffusionExtremums

LR="${LR:-1e-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_STEPS="${MAX_STEPS:-2000}"
K_ENC="${K_ENC:-32}"
N_LOSS="${N_LOSS:-256}"
D_MODEL="${D_MODEL:-128}"
D_EMB="${D_EMB:-128}"
N_LAYERS="${N_LAYERS:-1}"
N_HEADS="${N_HEADS:-4}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
VAL_CONFIGS="${VAL_CONFIGS:-20}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-results/dataset_hypernet}"
CORPUS_DIR="${CORPUS_DIR:-data/processed/targets_relu_h16}"
MLP_HIDDEN="${MLP_HIDDEN:-16}"

echo "============================================================"
echo " DatasetHypernet (examples → weights)"
echo "------------------------------------------------------------"
echo "  model      = K_enc=${K_ENC} N_loss=${N_LOSS}"
echo "               d_model=${D_MODEL} d_emb=${D_EMB}"
echo "               layers=${N_LAYERS} heads=${N_HEADS}"
echo "  training   = lr=${LR} bs=${BATCH_SIZE} steps=${MAX_STEPS}"
echo "  data       = ${CORPUS_DIR} mlp_hidden=${MLP_HIDDEN}"
echo "============================================================"

exec python -m src.scripts.train_dataset_hypernet \
    --lr "${LR}" --batch-size "${BATCH_SIZE}" --max-steps "${MAX_STEPS}" \
    --k-enc "${K_ENC}" --n-loss "${N_LOSS}" \
    --d-model "${D_MODEL}" --d-emb "${D_EMB}" \
    --n-layers "${N_LAYERS}" --n-heads "${N_HEADS}" \
    --device "${DEVICE}" --seed "${SEED}" --val-configs "${VAL_CONFIGS}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --corpus-dir "${CORPUS_DIR}" --mlp-hidden "${MLP_HIDDEN}"
