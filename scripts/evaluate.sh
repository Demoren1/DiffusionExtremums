#!/usr/bin/env bash
# =============================================================================
# evaluate.sh
# -----------------------------------------------------------------------------
# Запуск 3-методной оценки: from-scratch MLP, learned conv, oracle conv.
#
# Скрипт:
#   1. Активирует conda-окружение `myenv`.
#   2. Формирует и печатает шапку со значениями всех параметров.
#   3. Вызывает CLI `python -m src.scripts.evaluate` с заданными флагами.
#
# Оценка: 3 метода (from-scratch MLP, learned conv, oracle conv) на
# n_eval_train train-конфигах + n_eval_val val-конфигах (held-out). Метрики:
# test MSE, generalization gap, relative-to-conv MSE, Toeplitz-ness score,
# kernel recovery (cosine sim, L2). Результаты сохраняются в OUTPUT_DIR,
# фигуры — в FIGURES_DIR.
#
# Любой параметр можно переопределить через переменную окружения, например:
#     N_EVAL_TRAIN=40 N_EVAL_VAL=40 bash scripts/evaluate.sh
#     GPU=3 bash scripts/evaluate.sh
#
# Запуск:
#     bash scripts/evaluate.sh
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
# Каталог с таргетами (effective maps + configs.json)
TARGETS_DIR="${TARGETS_DIR:-data/processed/targets_eff}"
# Число train-конфигов для оценки (виденные при обучении)
N_EVAL_TRAIN="${N_EVAL_TRAIN:-20}"
# Число val-конфигов для оценки (held-out, не виденные при обучении)
N_EVAL_VAL="${N_EVAL_VAL:-20}"
# Число hold-out конфигов для детерминированного train/val сплита (из SEED)
VAL_CONFIGS="${VAL_CONFIGS:-50}"
# n_train для baseline "from-scratch MLP"
N_TRAIN_MLP="${N_TRAIN_MLP:-1024}"
# Устройство torch
DEVICE="${DEVICE:-cuda}"
# Каталог для результатов оценки (JSON/CSV)
OUTPUT_DIR="${OUTPUT_DIR:-results/evaluation}"
# Каталог для фигур (PNG)
FIGURES_DIR="${FIGURES_DIR:-figures}"
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
echo " 3-методная оценка (from-scratch MLP, learned conv, oracle conv)"
echo "------------------------------------------------------------"
echo "  Данные:"
echo "    targets_dir  = ${TARGETS_DIR}"
echo "  Оценка:"
echo "    n_eval_train = ${N_EVAL_TRAIN}  (виденные конфиги)"
echo "    n_eval_val   = ${N_EVAL_VAL}  (held-out конфиги)"
echo "    val_configs  = ${VAL_CONFIGS}  (детерминированный сплит из SEED)"
echo "    n_train_mlp  = ${N_TRAIN_MLP}  (для from-scratch MLP)"
echo "  Вывод:"
echo "    output_dir   = ${OUTPUT_DIR}"
echo "    figures_dir  = ${FIGURES_DIR}"
echo "  Прочее:"
echo "    device       = ${DEVICE}"
echo "    gpu          = '${GPU}' (пусто/auto = по умолчанию; иначе id карты)"
echo "    seed         = ${SEED}"
echo "============================================================"

# -----------------------------------------------------------------------------
# Сборка команды вызова CLI
# -----------------------------------------------------------------------------
CMD=(python -m src.scripts.evaluate
    --targets-dir "${TARGETS_DIR}"
    --n-eval-train "${N_EVAL_TRAIN}"
    --n-eval-val "${N_EVAL_VAL}"
    --val-configs "${VAL_CONFIGS}"
    --n-train-mlp "${N_TRAIN_MLP}"
    --device "${DEVICE_FOR_CLI}"
    --output-dir "${OUTPUT_DIR}"
    --figures-dir "${FIGURES_DIR}"
    --seed "${SEED}"
)

# -----------------------------------------------------------------------------
# Запуск
# -----------------------------------------------------------------------------
echo "Запуск: ${CMD[*]}"
"${CMD[@]}"
EVAL_RC=$?

exit ${EVAL_RC}