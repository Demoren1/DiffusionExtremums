#!/usr/bin/env bash
# Обучение диффузионной модели над ВЕСАМИ TargetMLP.
#
# Первый запуск сам построит датасет весов (обучит популяцию MLP) — это долго.
# Для отладки используйте маленькие --n_networks и --hidden_dim.
#
# Использование:
#   bash sh_scripts/train_weights.sh
#   bash sh_scripts/train_weights.sh 0 1000 256 1e-3 32 2000
#   bash sh_scripts/train_weights.sh 0 50  16  1e-3 16 50     # отладка
set -euo pipefail
cd /home/udeneev-av/DiffusionExtremums


# --- GPU ---
export CUDA_VISIBLE_DEVICES="${1:-0}"

# --- гиперпараметры ---
EPOCHS="${2:-1000}"
BATCH="${3:-256}"
LR="${4:-1e-3}"
HIDDEN="${5:-32}"          # ширина скрытого слоя TargetMLP
N_NETS="${6:-2000}"        # размер популяции MLP (датасет весов)
NAME="weights"

echo "[train_weights] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} epochs=${EPOCHS} batch=${BATCH} lr=${LR} hidden=${HIDDEN} n_networks=${N_NETS}"

python -m src.train \
    --data_source weights \
    --name "${NAME}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH}" \
    --lr "${LR}" \
    --hidden_dim "${HIDDEN}" \
    --n_networks "${N_NETS}"
