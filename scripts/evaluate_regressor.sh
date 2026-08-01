#!/usr/bin/env bash
# =============================================================================
# evaluate_regressor.sh
# -----------------------------------------------------------------------------
# Запуск оценки детерминированного oracle-регрессора effective map (базлайн).
#
# Оценка обученного MLP-регрессора 14 -> 1056 (config features -> effective
# linear map) на полном наборе метрик (normalized/raw MSE, Toeplitz-ness,
# kernel recovery, functional test MSE для oracle conv / target MLP /
# oracle regressor). Метрики сохраняются в OUTPUT_DIR, фигуры — в FIGURES_DIR.
#
# Скрипт:
#   1. Активирует conda-окружение `myenv`.
#   2. Формирует и печатает шапку со значениями всех параметров.
#   3. Вызывает CLI `python -m src.scripts.evaluate_oracle_regressor`.
#
# Любой параметр можно переопределить через переменную окружения, например:
#     N_EVAL_TRAIN=40 N_EVAL_VAL=40 bash scripts/evaluate_regressor.sh
#     GPU=3 bash scripts/evaluate_regressor.sh
#
# Запуск:
#     bash scripts/evaluate_regressor.sh
# =============================================================================
set -euo pipefail

# -----------------------------------------------------------------------------
# Активация conda-окружения
# -----------------------------------------------------------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv
cd /home/udeneev-av/DiffusionExtremums

# -----------------------------------------------------------------------------
# Параметры
# -----------------------------------------------------------------------------
# Путь к чекпоинту обученного oracle-регрессора
CHECKPOINT="${CHECKPOINT:-results/regressor_oracle/regressor_best.pt}"
# Каталог с таргетами (effective maps + configs.json)
TARGETS_DIR="${TARGETS_DIR:-data/processed/targets_eff}"
# Число train-конфигов для оценки (виденные при обучении)
N_EVAL_TRAIN="${N_EVAL_TRAIN:-20}"
# Число val-конфигов для оценки (held-out, не виденные при обучении)
N_EVAL_VAL="${N_EVAL_VAL:-20}"
# Устройство torch
DEVICE="${DEVICE:-cuda}"
# Каталог для результатов оценки (JSON/CSV)
OUTPUT_DIR="${OUTPUT_DIR:-results/regressor_oracle/evaluation}"
# Каталог для фигур (PNG)
FIGURES_DIR="${FIGURES_DIR:-figures/regressor_oracle}"
# Путь к train_log.csv для графика обучения (пусто = <checkpoint_dir>/train_log.csv)
TRAIN_LOG="${TRAIN_LOG:-}"
# Seed для воспроизводимости
SEED="${SEED:-0}"
# GPU для запуска (пусто или "auto" = карта по умолчанию; иначе — id, напр. "3")
GPU="${GPU:-6}"

# -----------------------------------------------------------------------------
# Выбор GPU
# -----------------------------------------------------------------------------
if [[ -n "${GPU}" && "${GPU}" != "auto" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU}"
    DEVICE_FOR_CLI="cuda"
else
    DEVICE_FOR_CLI="${DEVICE}"
fi

# -----------------------------------------------------------------------------
# Шапка: печать значений параметров перед запуском
# -----------------------------------------------------------------------------
echo "============================================================"
echo " Оценка oracle-регрессора (детерминированный базлайн)"
echo "------------------------------------------------------------"
echo "  Данные:"
echo "    checkpoint          = ${CHECKPOINT}"
echo "    targets_dir         = ${TARGETS_DIR}"
echo "  Оценка:"
echo "    n_eval_train        = ${N_EVAL_TRAIN}  (виденные конфиги)"
echo "    n_eval_val          = ${N_EVAL_VAL}  (held-out конфиги)"
echo "  Вывод:"
echo "    output_dir          = ${OUTPUT_DIR}"
echo "    figures_dir         = ${FIGURES_DIR}"
echo "    train_log           = '${TRAIN_LOG}' (пусто = <checkpoint_dir>/train_log.csv)"
echo "  Прочее:"
echo "    device              = ${DEVICE}"
echo "    gpu                 = '${GPU}' (пусто/auto = по умолчанию; иначе id карты)"
echo "    seed                = ${SEED}"
echo "============================================================"

# -----------------------------------------------------------------------------
# Сборка команды вызова CLI
# -----------------------------------------------------------------------------
CMD=(python -m src.scripts.evaluate_oracle_regressor
    --checkpoint "${CHECKPOINT}"
    --targets-dir "${TARGETS_DIR}"
    --n-eval-train "${N_EVAL_TRAIN}"
    --n-eval-val "${N_EVAL_VAL}"
    --device "${DEVICE_FOR_CLI}"
    --output-dir "${OUTPUT_DIR}"
    --figures-dir "${FIGURES_DIR}"
    --seed "${SEED}"
)

# Опциональные флаги
if [[ -n "${TRAIN_LOG}" ]]; then
    CMD+=(--train-log "${TRAIN_LOG}")
fi

# -----------------------------------------------------------------------------
# Запуск
# -----------------------------------------------------------------------------
echo "Запуск: ${CMD[*]}"
"${CMD[@]}"
EVAL_RC=$?

exit ${EVAL_RC}
