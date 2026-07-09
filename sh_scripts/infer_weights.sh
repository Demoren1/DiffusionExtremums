#!/usr/bin/env bash
# Инференс диффузионной модели над весами: генерация весов -> TargetMLP ->
# оценка воспроизведения функции и совпадения экстремумов.
#
# Использование:
#   bash sh_scripts/infer_weights.sh
#   bash sh_scripts/infer_weights.sh 0 64 checkpoints/latest.pt
set -euo pipefail
cd /home/udeneev-av/DiffusionExtremums

# --- GPU ---
export CUDA_VISIBLE_DEVICES="${1:-7}"

# --- параметры ---
N_SAMPLES="${2:-64}"
CKPT="${3:-checkpoints/latest.pt}"

echo "[infer_weights] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} n_samples=${N_SAMPLES} ckpt=${CKPT}"

python -m src.inference \
    --ckpt "${CKPT}" \
    --data_source weights \
    --n_samples "${N_SAMPLES}"
