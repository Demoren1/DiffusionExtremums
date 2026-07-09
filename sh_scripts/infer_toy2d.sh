#!/usr/bin/env bash
# Инференс диффузионной модели на 2D two-moons: сэмплинг + сравнение с ground truth.
#
# Использование:
#   bash sh_scripts/infer_toy2d.sh
#   bash sh_scripts/infer_toy2d.sh 0 5000 checkpoints/latest.pt
set -euo pipefail
cd /home/udeneev-av/DiffusionExtremums


# --- GPU ---
export CUDA_VISIBLE_DEVICES="${1:-0}"

# --- параметры ---
N_SAMPLES="${2:-5000}"
CKPT="${3:-checkpoints/latest.pt}"

echo "[infer_toy2d] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} n_samples=${N_SAMPLES} ckpt=${CKPT}"

python -m src.inference \
    --ckpt "${CKPT}" \
    --data_source toy2d \
    --n_samples "${N_SAMPLES}"
