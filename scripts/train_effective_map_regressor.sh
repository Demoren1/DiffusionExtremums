#!/usr/bin/env bash
# =============================================================================
# train_effective_map_regressor.sh
# -----------------------------------------------------------------------------
# Запуск обучения детерминированного регрессора effective map.
#
# По умолчанию запускается ОРАКУЛЬНЫЙ базлайн (oracle): регрессор учится на
# точных kernel-derived Toeplitz-таргетах (target_mode=oracle). Это валидированный
# эталонный режим. Режим learned (усреднённые MLP-derived effective maps) оставлен
# как sanity-check: чтобы запустить его, передайте TARGET_MODE=learned и
# CHECKPOINT_DIR=results/regressor_sanity.
#
# Простой MLP-регрессор 14 -> 1056 (config features -> effective linear map).
# Сплит train/val — по конфигам, детерминированный из SEED / VAL_CONFIGS.
# Нормализатор весов фитится только на train-таргетах. Чекпоинты +
# train_log.csv + split.json сохраняются в CHECKPOINT_DIR.
#
# Скрипт:
#   1. Активирует conda-окружение `myenv`.
#   2. Формирует и печатает шапку со значениями всех параметров.
#   3. Вызывает CLI `python -m src.scripts.train_effective_map_regressor`.
#
# Любой параметр можно переопределить через переменную окружения, например:
#     STEPS=50000 bash scripts/train_effective_map_regressor.sh
#     GPU=3 BATCH_SIZE=128 bash scripts/train_effective_map_regressor.sh
#
# Запуск (по умолчанию — oracle-базлайн):
#     bash scripts/train_effective_map_regressor.sh
#
# Запуск sanity-check режима learned:
#     TARGET_MODE=learned CHECKPOINT_DIR=results/regressor_sanity \
#         bash scripts/train_effective_map_regressor.sh
# =============================================================================
set -euo pipefail

# -----------------------------------------------------------------------------
# Активация conda-окружения
# -----------------------------------------------------------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv
cd /home/udeneev-av/DiffusionExtremums

# -----------------------------------------------------------------------------
# Параметры данных
# -----------------------------------------------------------------------------
# Каталог с таргетами (eff_maps.pt + configs.json). По умолчанию — Strategy B.
TARGETS_DIR="${TARGETS_DIR:-data/processed/targets_eff}"
# Число hold-out конфигов для валидации (детерминированный сплит из SEED)
VAL_CONFIGS="${VAL_CONFIGS:-50}"
# Режим построения таргета регрессии:
#   oracle  — точный kernel-derived Toeplitz-таргет (базлайн по умолчанию);
#   learned — усреднённые MLP-derived effective maps (sanity-check).
TARGET_MODE="${TARGET_MODE:-oracle}"

# -----------------------------------------------------------------------------
# Гиперпараметры обучения
# -----------------------------------------------------------------------------
# Пиковая скорость обучения AdamW (cosine decay до LR_MIN)
LR="${LR:-1e-3}"
# Финальная скорость обучения для cosine schedule
LR_MIN="${LR_MIN:-1e-6}"
# Weight decay для AdamW
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
# Размер батча (число конфигов на шаг)
BATCH_SIZE="${BATCH_SIZE:-64}"
# Число эпох (полных проходов по train-конфигам). Используется, если STEPS пуст.
EPOCHS="${EPOCHS:-20000}"
# Альтернатива: общее число шагов оптимизатора (приоритет над EPOCHS). Пусто = по эпохам.
STEPS="${STEPS:-}"
# Максимальная норма градиента для clipping (0 = выключено)
GRAD_CLIP="${GRAD_CLIP:-1.0}"
# Early-stopping patience (число eval-ов без улучшения; 0 = выключено)
PATIENCE="${PATIENCE:-50}"

# -----------------------------------------------------------------------------
# Логирование и чекпоинты
# -----------------------------------------------------------------------------
# Каталог для чекпоинтов и логов (train_log.csv, split.json, tensorboard/)
CHECKPOINT_DIR="${CHECKPOINT_DIR:-results/regressor_oracle}"
# Каталог для TensorBoard-логов (пусто = <CHECKPOINT_DIR>/tensorboard)
LOG_DIR="${LOG_DIR:-}"
# Логировать train loss каждые N шагов
LOG_EVERY="${LOG_EVERY:-100}"
# Считать val loss каждые N шагов
EVAL_EVERY="${EVAL_EVERY:-500}"
# Сохранять периодический чекпоинт каждые N шагов
SAVE_EVERY="${SAVE_EVERY:-2000}"
# Необязательный путь к чекпоинту для возобновления обучения
RESUME="${RESUME:-}"

# -----------------------------------------------------------------------------
# Устройство и воспроизводимость
# -----------------------------------------------------------------------------
# Устройство torch: 'auto', 'cuda' или 'cpu'
DEVICE="${DEVICE:-cuda}"
# GPU для обучения. Пусто или "auto" — карта по умолчанию (cuda:0); иначе — id.
GPU="${GPU:-6}"
# Seed для воспроизводимости
SEED="${SEED:-0}"

# -----------------------------------------------------------------------------
# Архитектура модели (по умолчанию 14 -> 256 -> 512 -> 512 -> 1056)
# -----------------------------------------------------------------------------
HIDDEN_DIMS="${HIDDEN_DIMS:-256,512,512}"
ACTIVATION="${ACTIVATION:-gelu}"
# Пусто = использовать residual; "0" = выключить
NO_RESIDUAL="${NO_RESIDUAL:-}"
NO_LAYER_NORM="${NO_LAYER_NORM:-}"
DROPOUT="${DROPOUT:-0.0}"

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
echo " Обучение регрессора effective map"
echo "------------------------------------------------------------"
echo "  Данные:"
echo "    targets_dir     = ${TARGETS_DIR}"
echo "    val_configs      = ${VAL_CONFIGS}  (детерминированный сплит из SEED)"
echo "    target_mode      = ${TARGET_MODE}  (oracle = базлайн; learned = sanity-check)"
echo "  Обучение:"
echo "    lr               = ${LR}"
echo "    lr_min           = ${LR_MIN}"
echo "    weight_decay     = ${WEIGHT_DECAY}"
echo "    batch_size       = ${BATCH_SIZE}"
echo "    epochs           = ${EPOCHS}  (используется, если STEPS пусто)"
echo "    steps            = '${STEPS}' (приоритет над EPOCHS; пусто = по эпохам)"
echo "    grad_clip        = ${GRAD_CLIP}"
echo "    patience         = ${PATIENCE}"
echo "  Логи/чекпоинты:"
echo "    checkpoint_dir   = ${CHECKPOINT_DIR}"
echo "    log_dir          = '${LOG_DIR}' (пусто = <checkpoint_dir>/tensorboard)"
echo "    log_every        = ${LOG_EVERY}"
echo "    eval_every       = ${EVAL_EVERY}"
echo "    save_every       = ${SAVE_EVERY}"
echo "    resume           = '${RESUME}' (пусто = с нуля)"
echo "  Прочее:"
echo "    device           = ${DEVICE}"
echo "    gpu              = '${GPU}' (пусто/auto = по умолчанию; иначе id карты)"
echo "    seed             = ${SEED}"
echo "  Модель:"
echo "    hidden_dims      = ${HIDDEN_DIMS}"
echo "    activation       = ${ACTIVATION}"
echo "    no_residual      = '${NO_RESIDUAL}'"
echo "    no_layer_norm    = '${NO_LAYER_NORM}'"
echo "    dropout          = ${DROPOUT}"
echo "============================================================"

# -----------------------------------------------------------------------------
# Сборка команды вызова CLI
# -----------------------------------------------------------------------------
CMD=(python -m src.scripts.train_effective_map_regressor
    --targets-dir "${TARGETS_DIR}"
    --target-mode "${TARGET_MODE}"
    --val-configs "${VAL_CONFIGS}"
    --lr "${LR}"
    --lr-min "${LR_MIN}"
    --weight-decay "${WEIGHT_DECAY}"
    --batch-size "${BATCH_SIZE}"
    --grad-clip "${GRAD_CLIP}"
    --patience "${PATIENCE}"
    --checkpoint-dir "${CHECKPOINT_DIR}"
    --log-every "${LOG_EVERY}"
    --eval-every "${EVAL_EVERY}"
    --save-every "${SAVE_EVERY}"
    --device "${DEVICE_FOR_CLI}"
    --seed "${SEED}"
    --hidden-dims "${HIDDEN_DIMS}"
    --activation "${ACTIVATION}"
    --dropout "${DROPOUT}"
)

# Опциональные флаги
if [[ -n "${LOG_DIR}" ]]; then
    CMD+=(--log-dir "${LOG_DIR}")
fi
if [[ -n "${STEPS}" ]]; then
    CMD+=(--steps "${STEPS}")
else
    CMD+=(--epochs "${EPOCHS}")
fi
if [[ -n "${RESUME}" ]]; then
    CMD+=(--resume "${RESUME}")
fi
if [[ "${NO_RESIDUAL}" == "1" ]]; then
    CMD+=(--no-residual)
fi
if [[ "${NO_LAYER_NORM}" == "1" ]]; then
    CMD+=(--no-layer-norm)
fi

# -----------------------------------------------------------------------------
# Запуск обучения
# -----------------------------------------------------------------------------
echo "Запуск: ${CMD[*]}"
"${CMD[@]}"
TRAIN_RC=$?

exit ${TRAIN_RC}
