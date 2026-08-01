"""Target collection pipeline: train N MLPs per dataset, save converged weights.

Phase 2 (Approach B): for each dataset ID in a corpus, train ``n_mlp`` MLPs
(different random initializations) to convergence and save their flattened
converged weight vectors. These weight vectors are the **regression targets**
for the effective-map regressor pipeline.

Output format (documented in the saved ``metadata.json``):
- ``weights.pt``: tensor of shape ``(n_datasets, n_mlp, D)`` float32, where
  ``D = 8352``. ``weights[i, j]`` is the converged weight vector of the j-th MLP
  (j-th random init) on the i-th dataset, in the canonical flatten order
  (fc1.weight, fc1.bias, fc2.weight, fc2.bias, C-order).
- ``losses.pt``: tensor of shape ``(n_datasets, n_mlp, 3)`` float32 with columns
  ``(train_mse, val_mse, test_mse)`` for each converged MLP.
- ``configs.json``: list of dataset configs (one per dataset), each a dict with
  family, kernel, radius, noise_std, n_train, n_test, seed, L, and dataset_id.
  These are the conditioning inputs for the model.
- ``dataset_ids.json``: list of dataset_id strings (parallel to axis 0).
- ``metadata.json``: full description of the format, the weight vectorization
  order, the generation/training parameters, and the codec dimensions.

The pipeline is **resumable**: it writes a per-dataset checkpoint after each
dataset completes, so an interrupted run can be restarted and will skip already
completed datasets. Datasets are processed independently (embarrassingly
parallel); a simple sequential loop is used here, but the per-dataset function
``collect_targets_for_dataset`` can be called from a multiprocessing pool.

**Multi-GPU parallelization**: ``CollectConfig.gpus`` selects a list of GPU ids.
With >1 GPU the corpus is sharded across GPUs (one ``torch.multiprocessing.spawn``
process per GPU, pinned via ``CUDA_VISIBLE_DEVICES``); each worker trains its
shard and writes a shard file, then the shards are merged into the standard
output format (byte-compatible with the sequential path). With 0/1 GPU the
original sequential single-device path is used. Resumability is per-shard
(``_progress_shard_{r}.pt``) in the parallel path, so there are no cross-GPU races.
"""
import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing
from tqdm.auto import tqdm

from src.configs.base import DatasetConfig
from src.data.dataset import DatasetInstance, DatasetFamily
from src.data.registry import dataset_id_from_config
from src.models.weight_codec import WeightCodec
from src.training.train_mlp import TrainConfig, TrainResult, train_mlp_to_convergence


@dataclass(frozen=True)
class CollectConfig:
    """Configuration for the target collection pipeline.

    Attributes:
        n_datasets: Number of datasets to generate and process.
        n_mlp: Number of MLPs (random initializations) to train per dataset.
        n_train: Training samples per dataset (Phase 2 uses 1024: an
            over-determined system 1024*32 >> 8352 params -> good generalization).
        n_test: Test samples per dataset (for the test_mse metadata).
        noise_std: Noise std for all datasets. None samples from the family grid.
        families: Restrict to a subset of families; None means all 5.
        corpus_seed: Base RNG seed for sampling dataset configs.
        mlp_seed_base: MLP init seeds are ``mlp_seed_base + j`` for j in
            [0, n_mlp). Different j -> different init -> different converged
            weights. The base is offset by the dataset index so different
            datasets get independent init streams.
        L: Input length (fixed 32).
        H: Hidden width (fixed 128 -> 8352 params).
        out_dir: Directory to save outputs (weights.pt, configs.json, ...).
        gpus: Optional list of GPU ids for multi-GPU parallel collection. If
            provided with >1 entry, the corpus is sharded across these GPUs
            (one process per GPU via ``torch.multiprocessing.spawn``) and the
            shards are merged into the standard output format. If None or a
            single GPU, the original sequential single-device path is used.
        train: ``TrainConfig`` for each MLP (lr, steps, patience, ...). Its
            ``L``/``H``/``seed`` are overridden per-MLP by the pipeline.
    """

    n_datasets: int = 2000
    n_mlp: int = 8
    n_train: int = 1024
    n_test: int = 512
    noise_std: Optional[float] = 0.1
    families: Optional[Tuple[str, ...]] = None
    corpus_seed: int = 0
    mlp_seed_base: int = 1000
    L: int = 32
    H: int = 128
    out_dir: str = "data/processed/targets"
    gpus: Optional[List[int]] = None
    train: TrainConfig = field(default_factory=TrainConfig)


def collect_targets_for_dataset(
    inst: DatasetInstance,
    n_mlp: int,
    mlp_seed_base: int,
    train_cfg: TrainConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Train ``n_mlp`` MLPs on one dataset and return their converged weights.

    Args:
        inst: The materialized dataset (x_train, y_train, x_test, y_test, ...).
        n_mlp: Number of MLPs (random inits) to train.
        mlp_seed_base: Base seed; MLP j uses seed ``mlp_seed_base + j``.
        train_cfg: Training hyperparameters (``L``/``H``/``seed`` overridden).

    Returns:
        ``(weights, losses)`` where
        - ``weights``: ``[n_mlp, D]`` float32 (CPU), canonical flatten order.
        - ``losses``: ``[n_mlp, 3]`` float32 (CPU) = (train_mse, val_mse, test_mse).
    """
    codec = WeightCodec(L=train_cfg.L, H=train_cfg.H)
    D = codec.D
    weights = torch.zeros(n_mlp, D, dtype=torch.float32)
    losses = torch.zeros(n_mlp, 3, dtype=torch.float32)

    for j in range(n_mlp):
        # Per-MLP config: override L/H/seed so each init is independent.
        cfg = TrainConfig(
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
            steps=train_cfg.steps,
            min_steps=train_cfg.min_steps,
            patience=train_cfg.patience,
            tol=train_cfg.tol,
            val_frac=train_cfg.val_frac,
            lr_min_ratio=train_cfg.lr_min_ratio,
            grad_clip=train_cfg.grad_clip,
            L=train_cfg.L,
            H=train_cfg.H,
            seed=mlp_seed_base + j,
            device=train_cfg.device,
            eval_every=train_cfg.eval_every,
        )
        res = train_mlp_to_convergence(
            inst.x_train, inst.y_train, inst.x_test, inst.y_test, config=cfg)
        weights[j] = res.theta
        losses[j] = torch.tensor(
            [res.train_mse, res.val_mse, res.test_mse], dtype=torch.float32)
    return weights, losses


def _config_to_record(cfg: DatasetConfig) -> dict:
    """Serialize a ``DatasetConfig`` to a JSON-friendly dict (with dataset_id)."""
    d = {
        "family": cfg.family,
        "kernel": [float(v) for v in cfg.kernel],
        "radius": int(cfg.radius),
        "noise_std": float(cfg.noise_std),
        "n_train": int(cfg.n_train),
        "n_test": int(cfg.n_test),
        "seed": int(cfg.seed),
        "L": int(cfg.L),
        "dataset_id": dataset_id_from_config(cfg),
    }
    return d


def _write_metadata(
    out_dir: str, collect_cfg: CollectConfig, codec: WeightCodec,
    n_datasets: int, n_mlp: int,
) -> None:
    """Write ``metadata.json`` describing the output format and parameters."""
    meta = {
        "format_version": 1,
        "description": (
            "Converged MLP weights as regression targets (Phase 2, Approach B). "
            "For each dataset, n_mlp MLPs (different random inits) were trained "
            "to convergence; their flattened weights are the targets."),
        "files": {
            "weights.pt": "float32 tensor [n_datasets, n_mlp, D] of converged "
                          "weight vectors (canonical flatten order).",
            "losses.pt": "float32 tensor [n_datasets, n_mlp, 3] = "
                         "(train_mse, val_mse, test_mse) per converged MLP.",
            "configs.json": "list of dataset config dicts (conditioning inputs), "
                            "parallel to axis 0 of weights.pt.",
            "dataset_ids.json": "list of dataset_id strings, parallel to axis 0.",
            "metadata.json": "this file.",
        },
        "weight_vectorization": {
            "order": ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"],
            "layout": "C-order / row-major (PyTorch nn.Linear storage)",
            "shapes": {k: list(v) for k, v in codec.shapes.items()},
            "offsets": codec.offsets,
            "sizes": codec.sizes,
            "D": codec.D,
        },
        "mlp_architecture": {
            "class": "src.models.mlp.MLPModel",
            "L": codec.L,
            "H": codec.H,
            "n_params": codec.D,
            "activation": "none (linear MLP)",
        },
        "collection": {
            "n_datasets": n_datasets,
            "n_mlp": n_mlp,
            "n_train": collect_cfg.n_train,
            "n_test": collect_cfg.n_test,
            "noise_std": collect_cfg.noise_std,
            "families": list(collect_cfg.families) if collect_cfg.families else "all",
            "corpus_seed": collect_cfg.corpus_seed,
            "mlp_seed_base": collect_cfg.mlp_seed_base,
        },
        "training": asdict(collect_cfg.train),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Multi-GPU parallelization (Phase 2+): one process per GPU, shard the corpus
# ---------------------------------------------------------------------------

def resolve_gpus(gpu_arg: str) -> List[int]:
    """Parse a GPU-selection string into a list of GPU ids.

    Args:
        gpu_arg: ``"auto"`` -> all visible CUDA GPUs (empty list if CUDA is
            unavailable); otherwise a comma-separated list of non-negative
            integers, e.g. ``"0,1,2,3"``.

    Returns:
        Sorted list of unique GPU ids. For ``"auto"`` with no CUDA, returns
        ``[]`` (the caller then falls back to the sequential CPU path).
    """
    gpu_arg = gpu_arg.strip()
    if gpu_arg.lower() == "auto":
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))
    ids: List[int] = []
    for tok in gpu_arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        ids.append(int(tok))
    if not ids:
        return []
    ids = sorted(set(ids))
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        bad = [i for i in ids if i < 0 or i >= n]
        if bad:
            raise ValueError(f"GPU ids {bad} out of range [0, {n})")
    return ids


def _shard_indices(n_datasets: int, n_shards: int) -> List[List[int]]:
    """Split ``range(n_datasets)`` into ``n_shards`` contiguous, near-equal shards.

    Returns a list of lists of global dataset indices, one per shard. The
    remainder is distributed to the first shards, so earlier shards are never
    smaller than later ones. The union is exactly ``range(n_datasets)`` with no
    overlap. Empty shards (when ``n_datasets < n_shards``) are returned as ``[]``.
    """
    n_shards = max(1, n_shards)
    base, rem = divmod(n_datasets, n_shards)
    shards: List[List[int]] = []
    start = 0
    for r in range(n_shards):
        size = base + (1 if r < rem else 0)
        shards.append(list(range(start, start + size)))
        start += size
    return shards


def _generate_corpus_configs(config: CollectConfig) -> List[DatasetConfig]:
    """Deterministically generate the full list of dataset configs from the seed.

    This is cheap (no training / data materialization) and is called both by the
    main process (to write ``configs.json``) and by each worker (to index its
    shard). Because it depends only on ``corpus_seed`` and the family grid, every
    process produces the identical config list, so global dataset indices stay
    aligned across shards. This is the same generation logic as the sequential
    path, so parallel and sequential runs produce identical configs (and, since
    ``mlp_seed_base`` depends only on the global index, identical weights).
    """
    rng = np.random.default_rng(config.corpus_seed)
    family_sampler = DatasetFamily(
        families=config.families, n_test=config.n_test, L=config.L)
    configs: List[DatasetConfig] = []
    for _ in range(config.n_datasets):
        configs.append(family_sampler.sample_random_config(
            rng, n_train=config.n_train, noise_std=config.noise_std))
    return configs


def _write_configs_and_ids(out_dir: str, configs: List[DatasetConfig]) -> None:
    """Write ``configs.json`` and ``dataset_ids.json`` (parallel to weights axis 0)."""
    records = [_config_to_record(c) for c in configs]
    with open(os.path.join(out_dir, "configs.json"), "w") as f:
        json.dump(records, f, indent=2)
    with open(os.path.join(out_dir, "dataset_ids.json"), "w") as f:
        json.dump([r["dataset_id"] for r in records], f, indent=2)


def _collect_shard_worker(
    rank: int,
    gpu_ids: List[int],
    config: CollectConfig,
    shards: List[List[int]],
    shard_dir: str,
    show_progress: bool,
) -> None:
    """Worker process: train the MLPs for one corpus shard on one GPU.

    Pinned to ``gpu_ids[rank]`` via ``CUDA_VISIBLE_DEVICES`` (set before any CUDA
    call, so ``torch.device("cuda")`` inside the worker refers to that single
    physical GPU). Writes a shard file ``_shard_{rank}.pt`` containing the
    converged weights/losses for ``shards[rank]`` (a list of global dataset
    indices). Resumable via a per-shard checkpoint ``_progress_shard_{rank}.pt``.

    ``torch.multiprocessing.spawn`` passes the same ``args`` tuple to every
    process, so the full ``shards`` list is passed and each worker selects its
    own shard via ``shards[rank]``.

    Args:
        rank: Worker index in ``[0, len(gpu_ids))``.
        gpu_ids: List of physical GPU ids; this worker uses ``gpu_ids[rank]``.
        config: Pipeline config (``config.train.device`` is overridden to
            ``"cuda"`` so the pinned GPU is used).
        shards: Full list of corpus shards (one per worker); this worker uses
            ``shards[rank]``.
        shard_dir: Directory for shard files and per-shard checkpoints.
        show_progress: Whether to show a tqdm bar for this shard.
    """
    # Select this worker's shard (spawn passes the same args to all processes).
    shard_indices = shards[rank]
    # Pin to this GPU BEFORE any CUDA operation. In a freshly-spawned process no
    # CUDA context exists yet, so this takes effect and "cuda" -> this GPU.
    # NOTE: this MUST be the first CUDA-related action in the child process.
    # Importing modules that call ``torch.cuda.is_available()`` at import time
    # (e.g. a module-level ``DEFAULT_DEVICE = torch.device("cuda" if ...)``)
    # would initialize the CUDA runtime here *before* this line, binding the
    # context to physical GPU 0 and making the env var below a no-op. The
    # ``train_mlp`` module avoids this by computing its default device lazily.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[rank])

    codec = WeightCodec(L=config.L, H=config.H)
    n_mlp = config.n_mlp
    D = codec.D

    # Override the training device to "cuda" (== the pinned GPU). Rebuild the
    # frozen TrainConfig so the rest of the hyperparameters are preserved.
    train_cfg = TrainConfig(
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        steps=config.train.steps,
        min_steps=config.train.min_steps,
        patience=config.train.patience,
        tol=config.train.tol,
        val_frac=config.train.val_frac,
        lr_min_ratio=config.train.lr_min_ratio,
        grad_clip=config.train.grad_clip,
        L=config.train.L,
        H=config.train.H,
        seed=config.train.seed,
        device="cuda",
        eval_every=config.train.eval_every,
    )

    # Regenerate the full config list (deterministic) and a sampler for data.
    all_configs = _generate_corpus_configs(config)
    family_sampler = DatasetFamily(
        families=config.families, n_test=config.n_test, L=config.L)

    # Resumability: load this shard's checkpoint (set of done global indices).
    ckpt_path = os.path.join(shard_dir, f"_progress_shard_{rank}.pt")
    done_set: set = set()
    done_weights: List[torch.Tensor] = []
    done_losses: List[torch.Tensor] = []
    done_indices: List[int] = []
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        done_set = set(int(i) for i in ckpt["indices"])
        ckpt_by_idx = {int(i): (w, l) for i, (w, l) in zip(
            ckpt["indices"], zip(ckpt["weights"], ckpt["losses"]))}
        for gi in shard_indices:
            if gi in done_set:
                done_indices.append(gi)
                done_weights.append(ckpt_by_idx[gi][0])
                done_losses.append(ckpt_by_idx[gi][1])
        print(f"[shard {rank}] resuming: {len(done_set)}/{len(shard_indices)} "
              f"datasets already done")

    # Train the remaining datasets in this shard (in ascending index order).
    todo = [gi for gi in shard_indices if gi not in done_set]
    it = tqdm(todo, desc=f"shard {rank} (gpu {gpu_ids[rank]})",
              disable=not show_progress, position=rank, leave=True)
    for gi in it:
        inst = family_sampler.sample_dataset(all_configs[gi])
        mlp_seed_base = config.mlp_seed_base + gi * 10000
        w, l = collect_targets_for_dataset(inst, n_mlp, mlp_seed_base, train_cfg)
        done_weights.append(w)
        done_losses.append(l)
        done_indices.append(gi)

        # Periodic per-shard checkpoint (every 8 datasets, or at shard end).
        if (len(done_indices) % 8 == 0) or (len(done_indices) == len(shard_indices)):
            torch.save(
                {"indices": done_indices,
                 "weights": torch.stack(done_weights).cpu(),
                 "losses": torch.stack(done_losses).cpu()},
                ckpt_path,
            )

    # Stack results (handle empty shards gracefully).
    if done_weights:
        w_stack = torch.stack(done_weights).cpu().float()
        l_stack = torch.stack(done_losses).cpu().float()
    else:
        w_stack = torch.zeros(0, n_mlp, D, dtype=torch.float32)
        l_stack = torch.zeros(0, n_mlp, 3, dtype=torch.float32)

    # Final per-shard checkpoint.
    torch.save(
        {"indices": done_indices, "weights": w_stack, "losses": l_stack},
        ckpt_path,
    )

    # Write the final shard file (the merger reads this).
    shard_path = os.path.join(shard_dir, f"_shard_{rank}.pt")
    torch.save(
        {"shard_id": rank,
         "indices": torch.tensor(done_indices, dtype=torch.long),
         "weights": w_stack,
         "losses": l_stack},
        shard_path,
    )
    print(f"[shard {rank}] wrote {len(done_indices)} datasets -> {shard_path}")


def _merge_shards(
    shard_dir: str, n_shards: int, n_datasets: int, n_mlp: int, D: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge per-GPU shard files into the full ``[n_datasets, n_mlp, D]`` tensors.

    Each shard file stores ``{"indices": long tensor, "weights": [k, n_mlp, D],
    "losses": [k, n_mlp, 3]}``. This scatters each shard's rows into the correct
    global indices, validates full coverage with no overlaps, and returns the
    assembled ``weights`` and ``losses`` tensors (float32, CPU).

    Raises:
        RuntimeError: if shards are missing, overlap, or do not cover all
            ``n_datasets`` indices.
    """
    weights = torch.zeros(n_datasets, n_mlp, D, dtype=torch.float32)
    losses = torch.zeros(n_datasets, n_mlp, 3, dtype=torch.float32)
    covered = torch.zeros(n_datasets, dtype=torch.bool)

    for r in range(n_shards):
        shard_path = os.path.join(shard_dir, f"_shard_{r}.pt")
        if not os.path.exists(shard_path):
            raise RuntimeError(f"missing shard file: {shard_path}")
        shard = torch.load(shard_path, map_location="cpu")
        idx = shard["indices"].long()
        w = shard["weights"].float()
        l = shard["losses"].float()
        if w.shape[0] != idx.shape[0]:
            raise RuntimeError(
                f"shard {r}: weights rows {w.shape[0]} != indices {idx.shape[0]}")
        if w.numel() > 0 and w.shape[1:] != (n_mlp, D):
            raise RuntimeError(
                f"shard {r}: weights shape {tuple(w.shape)} != (*, {n_mlp}, {D})")
        if idx.numel() > 0 and covered[idx].any():
            dup = idx[covered[idx]].tolist()
            raise RuntimeError(f"shard {r} overlaps covered indices: {dup}")
        if idx.numel() > 0:
            weights[idx] = w
            losses[idx] = l
            covered[idx] = True

    missing = int((~covered).sum().item())
    if missing:
        miss_idx = torch.nonzero(~covered).flatten().tolist()
        raise RuntimeError(
            f"merge incomplete: {missing} datasets not covered: {miss_idx}")
    return weights, losses


def collect_targets_parallel(
    config: CollectConfig, gpu_ids: List[int], show_progress: bool = True,
) -> str:
    """Run the collection pipeline across multiple GPUs in parallel.

    Spawns one process per GPU (via ``torch.multiprocessing.spawn``); each
    process trains a contiguous shard of the corpus on its pinned GPU and writes
    a shard file. After all workers finish, the shards are merged into the
    standard Phase 2 output format (``weights.pt``, ``losses.pt``,
    ``configs.json``, ``dataset_ids.json``, ``metadata.json``) -- byte-compatible
    with the sequential pipeline, just assembled from shards.

    The corpus is generated deterministically from ``corpus_seed``, so every
    worker produces the same config list and global dataset indices stay aligned.
    Resumability is per-shard (``_progress_shard_{r}.pt``), so an interrupted run
    can be restarted and each worker skips its already-completed datasets.

    Args:
        config: Pipeline config. ``config.train.device`` is overridden to
            ``"cuda"`` in each worker (pinned to its GPU).
        gpu_ids: List of physical GPU ids to use (len >= 2 for parallelism).
        show_progress: Whether workers show tqdm bars.

    Returns:
        The output directory path.
    """
    out_dir = config.out_dir
    os.makedirs(out_dir, exist_ok=True)
    shard_dir = out_dir  # shard + checkpoint files live in out_dir

    codec = WeightCodec(L=config.L, H=config.H)
    n_datasets = config.n_datasets
    n_mlp = config.n_mlp
    n_shards = len(gpu_ids)

    # Write configs + ids up front (deterministic; exist even if interrupted).
    configs = _generate_corpus_configs(config)
    _write_configs_and_ids(out_dir, configs)

    shards = _shard_indices(n_datasets, n_shards)
    print(f"[collect_targets] parallel: {n_shards} GPUs {gpu_ids}, "
          f"{n_datasets} datasets -> shard sizes {[len(s) for s in shards]}")

    # Spawn one worker per GPU. spawn blocks until all finish (or one raises).
    torch.multiprocessing.spawn(
        _collect_shard_worker,
        args=(gpu_ids, config, shards, shard_dir, show_progress),
        nprocs=n_shards,
        join=True,
    )

    # Merge shards into the final tensors.
    weights, losses = _merge_shards(shard_dir, n_shards, n_datasets, n_mlp, codec.D)

    # Final save: full tensors + metadata (same format as the sequential path).
    torch.save(weights, os.path.join(out_dir, "weights.pt"))
    torch.save(losses, os.path.join(out_dir, "losses.pt"))
    _write_metadata(out_dir, config, codec, n_datasets, n_mlp)

    # Clean up shard files and per-shard checkpoints after a successful merge.
    for r in range(n_shards):
        for fn in (f"_shard_{r}.pt", f"_progress_shard_{r}.pt"):
            p = os.path.join(shard_dir, fn)
            if os.path.exists(p):
                os.remove(p)

    print(f"[collect_targets] saved {n_datasets} datasets x {n_mlp} MLPs "
          f"-> {out_dir} (weights {tuple(weights.shape)}, D={codec.D}) "
          f"[merged from {n_shards} shards]")
    return out_dir


def collect_targets(config: CollectConfig, show_progress: bool = True) -> str:
    """Run the full target collection pipeline and save outputs to disk.

    Generates a corpus of ``n_datasets`` datasets, trains ``n_mlp`` MLPs per
    dataset to convergence, and saves weights + losses + configs + metadata to
    ``config.out_dir``. Resumable: skips datasets already present in the
    per-dataset checkpoint file.

    Args:
        config: Pipeline configuration.
        show_progress: Whether to show a tqdm progress bar.

    Returns:
        The output directory path.
    """
    # Multi-GPU path: if more than one GPU is configured, shard the corpus
    # across GPUs (one process per GPU) and merge the shards. Otherwise fall
    # through to the original sequential single-device path (backward compatible).
    if config.gpus and len(config.gpus) > 1:
        return collect_targets_parallel(config, config.gpus, show_progress)

    out_dir = config.out_dir
    os.makedirs(out_dir, exist_ok=True)

    codec = WeightCodec(L=config.L, H=config.H)
    n_datasets = config.n_datasets
    n_mlp = config.n_mlp

    # Pre-allocate output tensors.
    weights = torch.zeros(n_datasets, n_mlp, codec.D, dtype=torch.float32)
    losses = torch.zeros(n_datasets, n_mlp, 3, dtype=torch.float32)

    # Resumability: load any previously saved per-dataset checkpoint.
    ckpt_path = os.path.join(out_dir, "_progress.pt")
    start_idx = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        weights[:ckpt["n_done"]] = ckpt["weights"][:ckpt["n_done"]]
        losses[:ckpt["n_done"]] = ckpt["losses"][:ckpt["n_done"]]
        start_idx = ckpt["n_done"]
        print(f"[collect_targets] resuming from dataset {start_idx}/{n_datasets}")

    # Generate the corpus deterministically from corpus_seed.
    rng = np.random.default_rng(config.corpus_seed)
    family_sampler = DatasetFamily(
        families=config.families, n_test=config.n_test, L=config.L)

    # Build configs list (reproducible). We regenerate the same configs up to
    # start_idx to keep indices aligned, but only train the remaining ones.
    configs: List[DatasetConfig] = []
    for _ in range(n_datasets):
        cfg = family_sampler.sample_random_config(
            rng, n_train=config.n_train, noise_std=config.noise_std)
        configs.append(cfg)

    # Save configs + ids (written once, before training, so they exist even if
    # the run is interrupted).
    records = [_config_to_record(c) for c in configs]
    with open(os.path.join(out_dir, "configs.json"), "w") as f:
        json.dump(records, f, indent=2)
    with open(os.path.join(out_dir, "dataset_ids.json"), "w") as f:
        json.dump([r["dataset_id"] for r in records], f, indent=2)

    # Train remaining datasets.
    indices = range(start_idx, n_datasets)
    it = tqdm(indices, desc="collect targets", disable=not show_progress)
    for i in it:
        inst = family_sampler.sample_dataset(configs[i])
        # Independent init stream per dataset: offset base by dataset index.
        mlp_seed_base = config.mlp_seed_base + i * 10000
        w, l = collect_targets_for_dataset(
            inst, n_mlp, mlp_seed_base, config.train)
        weights[i] = w
        losses[i] = l

        # Checkpoint after each dataset (cheap: only copy the done slice).
        torch.save(
            {"weights": weights[: i + 1], "losses": losses[: i + 1],
             "n_done": i + 1},
            ckpt_path,
        )

    # Final save: full tensors + metadata.
    torch.save(weights, os.path.join(out_dir, "weights.pt"))
    torch.save(losses, os.path.join(out_dir, "losses.pt"))
    _write_metadata(out_dir, config, codec, n_datasets, n_mlp)

    # Clean up the progress checkpoint after a successful full run.
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print(f"[collect_targets] saved {n_datasets} datasets x {n_mlp} MLPs "
          f"-> {out_dir} (weights {tuple(weights.shape)}, D={codec.D})")
    return out_dir
