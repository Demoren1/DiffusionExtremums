"""Training loop for DatasetHypernet: examples → embedding → weights → loss."""
import json, os, time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import DatasetFamily
from src.data.corpus_loader import CorpusBundle, config_from_record, load_relu_corpus
from src.models.dataset_hypernet import DatasetEncoder, WeightDecoder, DatasetHypernet
from src.models.weight_codec import WeightCodec
from src.utils.seeding import set_seed


@dataclass(frozen=True)
class DatasetHypernetConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_steps: int = 2000
    grad_clip: float = 1.0
    log_every: int = 50
    eval_every: int = 200
    save_every: int = 500
    patience: int = 10
    device: str = "auto"
    seed: int = 0
    checkpoint_dir: str = "results/dataset_hypernet"
    resume: Optional[str] = None
    K_enc: int = 32
    N_loss: int = 256
    d_model: int = 128
    d_emb: int = 128
    n_layers: int = 1
    n_heads: int = 4
    corpus_dir: str = "data/processed/targets_relu_h16"
    mlp_hidden: int = 16


def _resolve_device(d):
    return torch.device("cuda" if d=="auto" and torch.cuda.is_available() else d)


def _precompute_datasets(bundle, device):
    cache = {}
    family = DatasetFamily(n_test=512, L=32)
    for i in range(bundle.n_configs):
        cfg = config_from_record(bundle.configs[i])
        inst = family.sample_dataset(cfg)
        x = inst.x_train.float().to(device)
        y = inst.y_train.float().to(device)
        cache[i] = (x, y)
    return cache


def train_dataset_hypernet(config: DatasetHypernetConfig) -> str:
    set_seed(config.seed)
    device = _resolve_device(config.device)
    print(f"[dataset_hypernet] device={device}")
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    bundle = load_relu_corpus(corpus_dir=config.corpus_dir)
    D = WeightCodec(L=32, H=config.mlp_hidden).D
    print(f"[dataset_hypernet] corpus: {bundle.n_configs} datasets x {bundle.n_mlp} MLPs, D={D}")

    encoder = DatasetEncoder(
        L=32, K_enc=config.K_enc, d_model=config.d_model,
        d_emb=config.d_emb, n_layers=config.n_layers, n_heads=config.n_heads)
    decoder = WeightDecoder(d_emb=config.d_emb, D=D)
    model = DatasetHypernet(encoder, decoder, mlp_hidden=config.mlp_hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[dataset_hypernet] model params: {n_params:,}")

    print("[dataset_hypernet] precomputing datasets...")
    dataset_cache = _precompute_datasets(bundle, device)
    print(f"[dataset_hypernet] cached {len(dataset_cache)} datasets")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    rng_gen = torch.Generator().manual_seed(config.seed)

    val_pts = {}
    for i in range(bundle.n_configs):
        x_all, y_all = dataset_cache[i]
        n = min(512, x_all.shape[0])
        val_pts[i] = (x_all[:n], y_all[:n])

    step = 0
    best_val = float("inf")
    best_step = -1
    patience_counter = 0
    model.train()
    start_time = time.time()

    while step < config.max_steps:
        ds_np = np.random.default_rng().integers(0, bundle.n_configs, size=config.batch_size)

        x_enc, y_enc, x_loss, y_loss = [], [], [], []
        for i in range(config.batch_size):
            did = int(ds_np[i])
            x_all, y_all = dataset_cache[did]
            total = config.K_enc + config.N_loss
            idx = torch.randint(0, x_all.shape[0], (total,), generator=rng_gen)
            x_enc.append(x_all[idx[:config.K_enc]])
            y_enc.append(y_all[idx[:config.K_enc]])
            x_loss.append(x_all[idx[config.K_enc:]])
            y_loss.append(y_all[idx[config.K_enc:]])
        x_enc = torch.stack(x_enc).to(device)
        y_enc = torch.stack(y_enc).to(device)
        x_loss = torch.stack(x_loss).to(device)
        y_loss = torch.stack(y_loss).to(device)

        loss = model.compute_loss(x_enc, y_enc, x_loss, y_loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        step += 1

        if step % config.log_every == 0 or step == 1:
            el = time.time() - start_time
            print(f"[dataset_hypernet] step {step}/{config.max_steps} loss {loss.item():.6g} ({el:.1f}s)")

        if step % config.eval_every == 0:
            model.eval()
            total_val = 0.0
            with torch.no_grad():
                for did in range(bundle.n_configs):
                    x_all, y_all = val_pts[did]
                    x_e = x_all[:config.K_enc].unsqueeze(0)
                    y_e = y_all[:config.K_enc].unsqueeze(0)
                    x_l = x_all[config.K_enc:].unsqueeze(0)
                    y_l = y_all[config.K_enc:].unsqueeze(0)
                    total_val += model.compute_loss(x_e, y_e, x_l, y_l).item()
            val_loss = total_val / bundle.n_configs
            model.train()
            print(f"[dataset_hypernet] step {step} val loss {val_loss:.6g}")

            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_step = step
                patience_counter = 0
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": asdict(config),
                    "step": step, "best_val": best_val,
                }, os.path.join(config.checkpoint_dir, "best.pt"))
            else:
                patience_counter += 1
                if config.patience > 0 and patience_counter >= config.patience:
                    print(f"[dataset_hypernet] early stop at step {step} (best {best_val:.6g} @ {best_step})")
                    break

        if step % config.save_every == 0:
            torch.save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": asdict(config), "step": step, "best_val": best_val,
            }, os.path.join(config.checkpoint_dir, "checkpoint.pt"))

    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "config": asdict(config), "step": step, "best_val": best_val,
    }, os.path.join(config.checkpoint_dir, "checkpoint.pt"))
    print(f"[dataset_hypernet] done: best val {best_val:.6g} @ step {best_step} -> {config.checkpoint_dir}")
    return config.checkpoint_dir
