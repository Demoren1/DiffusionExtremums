#!/usr/bin/env bash
# Обучение диффузионной модели на 2D two-moons (sanity-check DDPM).
#
# Использование:
#   bash sh_scripts/train_toy2d.sh
#   bash sh_scripts/train_toy2d.sh 0 500 512 1e-3   # GPU epochs batch lr
set -euo pipefail
cd /home/udeneev-av/DiffusionExtremums

# --- GPU ---
export CUDA_VISIBLE_DEVICES="${1:-7}"

# --- гиперпараметры ---
EPOCHS="${2:-500}"
BATCH="${3:-512}"
LR="${4:-1e-3}"
NAME="toy2d"

echo "[train_toy2d] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} epochs=${EPOCHS} batch=${BATCH} lr=${LR}"

python -m src.train \
    --data_source toy2d \
    --name "${NAME}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH}" \
    --lr "${LR}"
