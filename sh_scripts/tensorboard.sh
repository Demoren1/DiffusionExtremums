#!/usr/bin/env bash
# Запуск TensorBoard для просмотра логов обучения.
#
# Использование:
#   bash sh_scripts/tensorboard.sh
#   bash sh_scripts/tensorboard.sh 6006
set -euo pipefail
cd /home/udeneev-av/DiffusionExtremums

PORT="${1:-44444}"
echo "[tensorboard] serving runs/ on http://localhost:${PORT}"
tensorboard --logdir runs --port "${PORT}"
