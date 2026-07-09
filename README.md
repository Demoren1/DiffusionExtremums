# DiffusionExtremums

Проект: обучение **диффузионной модели**, которая генерирует **веса** маленькой
MLP, воспроизводящей периодическую целевую функцию (и её экстремумы).

Долгосрочная цель — по весам модели (и области определения) определять её
экстремумы. На текущем шаге реализован полный **пайплайн диффузии**: обучение с
логированием в TensorBoard и инференс.

## Идея

Целевая функция (см. [`toy_exps/sin_exp.ipynb`](toy_exps/sin_exp.ipynb:41)):

```
y = sin(k0 * x + b0) + sin(k1 * x + b1),   k = (0.7, 1.5), b = (1, -1)
```

1. Обучаем **популяцию** маленьких MLP (`1 → hidden(tanh) → 1`) из разных сидов
   приближать эту функцию.
2. Собираем flatten-векторы весов каждой сети — это **данные для диффузии**.
3. Обучаем DDPM генерировать векторы весов.
4. На инференсе: сэмплируем веса → собираем из них MLP → сравниваем
   восстановленную функцию с целевой и считаем совпадение экстремумов.

Пайплайн **data-agnostic**: денойзер работает с произвольным вектором
`x ∈ R^d` (+ опциональное условие), а источник данных переключается флагом.
Для sanity-check реализации DDPM используется 2D-датасет two-moons, где виден
ground truth.

## Структура проекта

```
src/
  config.py              — dataclass-конфиги (TargetConfig, TargetMLPConfig,
                          WeightsDatasetConfig, Toy2DConfig, DDPMConfig,
                          DenoiserConfig, TrainConfig, InferenceConfig,
                          ExperimentConfig)
  utils.py               — get_device (cuda/cpu), set_seed, flatten/unflatten весов
  target.py              — целевая функция + TargetMLP (hidden — параметр конфига)
  datasets/
    weights_dataset.py   — генерация популяции MLP + Dataset векторов весов
    toy2d.py             — two-moons Dataset (sanity-check)
  diffusion/
    ddpm.py              — beta/alpha-расписания, q(x_t|x_0), sampling-loop
    denoiser.py          — MLP-денойзер + sinusoidal time-embedding + conditioning
    ema.py               — EMA весов денойзера (улучшает качество сэмплов)
  train.py               — тренировочный цикл, TensorBoard, чекпоинты, EMA, val-split
  inference.py           — сэмплинг, инстанцирование MLP, оценка функции и экстремумов
datasets/                — сохранённые векторы весов + статистика нормализации
checkpoints/             — чекпоинты модели
runs/                    — логи TensorBoard
outputs/                 — картинки инференса
sh_scripts/              — bash-скрипты запуска (train/infer/tensorboard) с CUDA_VISIBLE_DEVICES
toy_exps/                — ноутбук с целевой функцией
requirements.txt
PLAN_step1_diffusion_pipeline.md  — план шага 1
```

## Установка

```bash
pip install -r requirements.txt
```

Требуется `torch>=2.0`. Код рассчитан на CUDA, но работает и на CPU
(устройство выбирается автоматически через `device="auto"`).

## Быстрый старт

Все команды запускаются из корня проекта. GPU выбирается через
`CUDA_VISIBLE_DEVICES` (первый аргумент bash-скриптов, по умолчанию `0`).

### 1. Sanity-check на two-moons

Проверка корректности реализации DDPM на интерпретируемом 2D-распределении:

```bash
# через bash-скрипты (рекомендуется)
bash sh_scripts/train_toy2d.sh            # GPU=0, epochs=500, batch=512, lr=1e-3
bash sh_scripts/infer_toy2d.sh            # GPU=0, n_samples=5000

# или напрямую
python -m src.train --data_source toy2d --name toy2d --epochs 500 --batch_size 512 --lr 1e-3
python -m src.inference --ckpt checkpoints/latest.pt --data_source toy2d --n_samples 5000
```

Картинка со сравнением сгенерированных точек и ground truth:
`outputs/toy2d_inference.png`.

### 2. Диффузия над весами MLP

```bash
# через bash-скрипты (рекомендуется)
bash sh_scripts/train_weights.sh 0 1000 256 1e-3 32 2000   # GPU epochs batch lr hidden n_networks
bash sh_scripts/infer_weights.sh 0 64                       # GPU n_samples

# отладочный запуск (быстро)
bash sh_scripts/train_weights.sh 0 50 16 1e-3 16 50

# или напрямую
python -m src.train --data_source weights --name weights \
    --epochs 1000 --batch_size 256 --lr 1e-3 \
    --hidden_dim 32 --n_networks 2000
python -m src.inference --ckpt checkpoints/latest.pt --data_source weights --n_samples 64

# отключить EMA / канонизацию (для экспериментов)
python -m src.train --data_source weights --no_ema --no_canonicalize ...
python -m src.inference --ckpt checkpoints/latest.pt --no_ema
```

Результат: `outputs/weights_inference.png` + метрики в stdout
(MSE восстановленной функции, число совпавших минимумов/максимумов).

> Внимание: построение датасета весов (`--n_networks 2000`) обучает 2000
> маленьких MLP — это занимает время. Для отладки используйте
> `--n_networks 50 --hidden_dim 16`.

### Bash-скрипты (`sh_scripts/`)

| Скрипт | Назначение | Аргументы |
|---|---|---|
| [`train_toy2d.sh`](sh_scripts/train_toy2d.sh:1) | обучение на two-moons | `[GPU] [epochs] [batch] [lr]` |
| [`infer_toy2d.sh`](sh_scripts/infer_toy2d.sh:1) | инференс two-moons | `[GPU] [n_samples] [ckpt]` |
| [`train_weights.sh`](sh_scripts/train_weights.sh:1) | обучение на весах MLP | `[GPU] [epochs] [batch] [lr] [hidden] [n_networks]` |
| [`infer_weights.sh`](sh_scripts/infer_weights.sh:1) | инференс весов | `[GPU] [n_samples] [ckpt]` |
| [`tensorboard.sh`](sh_scripts/tensorboard.sh:1) | запуск TensorBoard | `[port]` |

Все скрипты устанавливают `export CUDA_VISIBLE_DEVICES` (первый аргумент, по
умолчанию `0`) и пробрасывают остальные параметры в Python-модули.

## Логирование (TensorBoard)

```bash
tensorboard --logdir runs
```

В логах:
- `train/loss`, `train/lr` — скаляры;
- `val/loss` — loss на отложенной валидационной выборке (если `val_fraction > 0`);
- `weights/<имя параметра>` — гистограммы весов денойзера;
- `samples/toy2d` — scatter сгенерированных точек;
- `samples/toy2d_ema` — то же, но с EMA-весами (если `use_ema_for_sampling`);
- `samples/weights` — графики функций сгенерированных MLP vs целевая;
- `samples/weights_ema` — то же с EMA-весами.

## Ключевые параметры (в [`src/config.py`](src/config.py:1))

| Параметр | Где | Описание |
|---|---|---|
| `target_mlp.hidden_dim` | `TargetMLPConfig` | ширина скрытого слоя MLP (>10) |
| `weights_dataset.n_networks` | `WeightsDatasetConfig` | размер популяции MLP |
| `weights_dataset.normalize` | `WeightsDatasetConfig` | стандартизация весов |
| `weights_dataset.canonicalize` | `WeightsDatasetConfig` | канонизация нейронов (убирает симметрии) |
| `ddpm.num_timesteps` | `DDPMConfig` | число шагов диффузии T |
| `ddpm.schedule` | `DDPMConfig` | `"linear"` / `"cosine"` |
| `ddpm.objective` | `DDPMConfig` | `"epsilon"` / `"x0"` |
| `denoiser.cond_dim` | `DenoiserConfig` | размер условия (0 = безусловно) |
| `train.device` | `TrainConfig` | `"auto"` / `"cuda"` / `"cpu"` |
| `train.val_fraction` | `TrainConfig` | доля val-выборки (0 = без split) |
| `train.ema_decay` | `TrainConfig` | decay EMA денойзера (0 = выключено) |
| `train.use_ema_for_sampling` | `TrainConfig` | использовать EMA-веса для сэмплов |

## Метрики качества (инференс весов)

- **MSE** восстановленной функции к целевой (mean / median по сгенерированным сетям).
- **Доля рабочих сетей** — сколько сгенерированных весов дают конечную функцию.
- **Совпадение экстремумов** — сколько локальных минимумов/максимумов
  сгенерированной функции попадают в допуск `π / max(k)` от истинных.

## Улучшения (по результатам ревизии Step 1)

- **Канонизация нейронов** ([`canonicalize_mlp`](src/target.py:72)) — убирает
  перестановочные и знаковые симметрии скрытых нейронов до flatten, так что
  эквивалентные сети отображаются в один вектор весов (снижает мультимодальность
  данных). Включается флагом `weights_dataset.canonicalize`.
- **EMA денойзера** ([`src/diffusion/ema.py`](src/diffusion/ema.py:1)) —
  экспоненциальное сглаживание весов; EMA-копия сохраняется в чекпоинте и
  используется при сэмплинге (`train.ema_decay`, `--no_ema` для отключки).
- **Train/val split** — отложенная выборка и периодическое логирование `val/loss`
  (`train.val_fraction`).
- **Кеширование статистик** — логирование сэмплов весов больше не перечитывает
  датасет с диска на каждом вызове.

## Что дальше (шаг 2)

- Conditioning: параметры целевой функции (`amp`, `shift`) как условие для
  диффузии над весами — объединение вариантов 1 и 2.
- Подбор ширины `hidden` и размера популяции эмпирически.
- Полноценный бейзлайн после длительного обучения (фиксация MSE и совпадения
  экстремумов).
