"""Target collection pipeline: train N MLPs per dataset, save converged weights."""
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
    """Configuration for the target collection pipeline."""

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
    """Train n_mlp MLPs on one dataset and return their converged weights."""
    codec = WeightCodec(L=train_cfg.L, H=train_cfg.H)
    D = codec.D
    weights = torch.zeros(n_mlp, D, dtype=torch.float32)
    losses = torch.zeros(n_mlp, 3, dtype=torch.float32)

    for j in range(n_mlp):
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
            "activation": "relu (fc2(relu(fc1(x))))",
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


def resolve_gpus(gpu_arg: str) -> List[int]:
    """Parse a GPU-selection string into a list of GPU ids."""
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
    """Split range(n_datasets) into n_shards contiguous, near-equal shards."""
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
    """Deterministically generate the full list of dataset configs from the seed."""
    rng = np.random.default_rng(config.corpus_seed)
    family_sampler = DatasetFamily(
        families=config.families, n_test=config.n_test, L=config.L)
    configs: List[DatasetConfig] = []
    for _ in range(config.n_datasets):
        configs.append(family_sampler.sample_random_config(
            rng, n_train=config.n_train, noise_std=config.noise_std))
    return configs


def _write_configs_and_ids(out_dir: str, configs: List[DatasetConfig]) -> None:
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
    # Pin to this GPU before any CUDA call, or the runtime binds to physical GPU 0.
    shard_indices = shards[rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[rank])

    codec = WeightCodec(L=config.L, H=config.H)
    n_mlp = config.n_mlp
    D = codec.D

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

    all_configs = _generate_corpus_configs(config)
    family_sampler = DatasetFamily(
        families=config.families, n_test=config.n_test, L=config.L)

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

        if (len(done_indices) % 8 == 0) or (len(done_indices) == len(shard_indices)):
            torch.save(
                {"indices": done_indices,
                 "weights": torch.stack(done_weights).cpu(),
                 "losses": torch.stack(done_losses).cpu()},
                ckpt_path,
            )

    if done_weights:
        w_stack = torch.stack(done_weights).cpu().float()
        l_stack = torch.stack(done_losses).cpu().float()
    else:
        w_stack = torch.zeros(0, n_mlp, D, dtype=torch.float32)
        l_stack = torch.zeros(0, n_mlp, 3, dtype=torch.float32)

    torch.save(
        {"indices": done_indices, "weights": w_stack, "losses": l_stack},
        ckpt_path,
    )

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
    """Merge per-GPU shard files into the full [n_datasets, n_mlp, D] tensors."""
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
    """Run the collection pipeline across multiple GPUs in parallel."""
    out_dir = config.out_dir
    os.makedirs(out_dir, exist_ok=True)
    shard_dir = out_dir

    codec = WeightCodec(L=config.L, H=config.H)
    n_datasets = config.n_datasets
    n_mlp = config.n_mlp
    n_shards = len(gpu_ids)

    configs = _generate_corpus_configs(config)
    _write_configs_and_ids(out_dir, configs)

    shards = _shard_indices(n_datasets, n_shards)
    print(f"[collect_targets] parallel: {n_shards} GPUs {gpu_ids}, "
          f"{n_datasets} datasets -> shard sizes {[len(s) for s in shards]}")

    torch.multiprocessing.spawn(
        _collect_shard_worker,
        args=(gpu_ids, config, shards, shard_dir, show_progress),
        nprocs=n_shards,
        join=True,
    )

    weights, losses = _merge_shards(shard_dir, n_shards, n_datasets, n_mlp, codec.D)

    torch.save(weights, os.path.join(out_dir, "weights.pt"))
    torch.save(losses, os.path.join(out_dir, "losses.pt"))
    _write_metadata(out_dir, config, codec, n_datasets, n_mlp)

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
    """Run the full target collection pipeline and save outputs to disk."""
    if config.gpus and len(config.gpus) > 1:
        return collect_targets_parallel(config, config.gpus, show_progress)

    out_dir = config.out_dir
    os.makedirs(out_dir, exist_ok=True)

    codec = WeightCodec(L=config.L, H=config.H)
    n_datasets = config.n_datasets
    n_mlp = config.n_mlp

    weights = torch.zeros(n_datasets, n_mlp, codec.D, dtype=torch.float32)
    losses = torch.zeros(n_datasets, n_mlp, 3, dtype=torch.float32)

    ckpt_path = os.path.join(out_dir, "_progress.pt")
    start_idx = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        weights[:ckpt["n_done"]] = ckpt["weights"][:ckpt["n_done"]]
        losses[:ckpt["n_done"]] = ckpt["losses"][:ckpt["n_done"]]
        start_idx = ckpt["n_done"]
        print(f"[collect_targets] resuming from dataset {start_idx}/{n_datasets}")

    rng = np.random.default_rng(config.corpus_seed)
    family_sampler = DatasetFamily(
        families=config.families, n_test=config.n_test, L=config.L)

    configs: List[DatasetConfig] = []
    for _ in range(n_datasets):
        cfg = family_sampler.sample_random_config(
            rng, n_train=config.n_train, noise_std=config.noise_std)
        configs.append(cfg)

    records = [_config_to_record(c) for c in configs]
    with open(os.path.join(out_dir, "configs.json"), "w") as f:
        json.dump(records, f, indent=2)
    with open(os.path.join(out_dir, "dataset_ids.json"), "w") as f:
        json.dump([r["dataset_id"] for r in records], f, indent=2)

    indices = range(start_idx, n_datasets)
    it = tqdm(indices, desc="collect targets", disable=not show_progress)
    for i in it:
        inst = family_sampler.sample_dataset(configs[i])
        mlp_seed_base = config.mlp_seed_base + i * 10000
        w, l = collect_targets_for_dataset(
            inst, n_mlp, mlp_seed_base, config.train)
        weights[i] = w
        losses[i] = l

        torch.save(
            {"weights": weights[: i + 1], "losses": losses[: i + 1],
             "n_done": i + 1},
            ckpt_path,
        )

    torch.save(weights, os.path.join(out_dir, "weights.pt"))
    torch.save(losses, os.path.join(out_dir, "losses.pt"))
    _write_metadata(out_dir, config, codec, n_datasets, n_mlp)

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print(f"[collect_targets] saved {n_datasets} datasets x {n_mlp} MLPs "
          f"-> {out_dir} (weights {tuple(weights.shape)}, D={codec.D})")
    return out_dir
