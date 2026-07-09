"""Training loop for the DDPM with TensorBoard logging and checkpoints.

Usage:
    python -m src.train --data_source toy2d --epochs 500
    python -m src.train --data_source weights --epochs 1000

The script is data-source agnostic: it builds either the 2D toy dataset or the
weights dataset, constructs the denoiser + DDPM, and trains.  Logging includes
scalar loss/lr, weight histograms, and periodic sample figures.

Improvements over the initial version:
  * **EMA** of the denoiser weights (used for sampling / checkpointing).
  * **Train / validation split** with periodic val-loss logging.
  * **Cached dataset statistics** — sample-figure logging no longer re-reads the
    weights dataset from disk on every call.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import ExperimentConfig
from .utils import get_device, set_seed
from .diffusion.denoiser import build_denoiser
from .diffusion.ddpm import DDPM
from .diffusion.ema import EMA
from .datasets.toy2d import Toy2DDataset
from .datasets.weights_dataset import load_or_build, WeightsDataset


# --------------------------------------------------------------------------- #
# Dataset / data_dim construction
# --------------------------------------------------------------------------- #
def build_dataset(cfg: ExperimentConfig, device: torch.device):
    """Return (dataset, data_dim).  dataset[i] is a 1-D tensor."""
    if cfg.data_source == "toy2d":
        ds = Toy2DDataset(cfg.toy2d)
        return ds, ds.data_dim
    elif cfg.data_source == "weights":
        ds = load_or_build(cfg.target, cfg.target_mlp, cfg.weights_dataset,
                           device=device, verbose=True)
        return ds, ds.data_dim
    else:
        raise ValueError(f"unknown data_source: {cfg.data_source}")


# --------------------------------------------------------------------------- #
# Sample-figure logging
# --------------------------------------------------------------------------- #
def _log_toy2d_samples(ddpm: DDPM, writer: SummaryWriter, step: int,
                       device: torch.device, n: int = 2000,
                       use_ema: bool = False, ema: Optional[EMA] = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ctx = ema.swap(ddpm.denoiser) if (use_ema and ema is not None) else _nullcontext(ddpm)
    with ctx:
        samples = ddpm.sample((n, 2), device, n_steps=1000).cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.4, c="#2563eb")
    tag = "samples/toy2d_ema" if (use_ema and ema is not None) else "samples/toy2d"
    ax.set_title(f"DDPM samples @ step {step}")
    ax.set_aspect("equal")
    writer.add_figure(tag, fig, step)
    plt.close(fig)


def _log_weights_samples(ddpm: DDPM, writer: SummaryWriter, step: int,
                         device: torch.device, cfg: ExperimentConfig,
                         dataset: WeightsDataset, n: int = 8,
                         use_ema: bool = False, ema: Optional[EMA] = None):
    """Log generated-MLP function plots.

    ``dataset`` is passed in so we reuse its cached mean/std instead of re-reading
    the weights file from disk on every call.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = cfg.target_mlp.n_params
    ctx = ema.swap(ddpm.denoiser) if (use_ema and ema is not None) else _nullcontext(ddpm)
    with ctx:
        samples = ddpm.sample((n, d), device, n_steps=1000).cpu()

    # De-normalize using the *cached* dataset statistics (no disk read).
    mean, std = dataset.mean, dataset.std
    if mean is not None and std is not None:
        samples = samples * std + mean

    from .target import build_target_mlp, sample_target_grid
    x, y_true = sample_target_grid(cfg.target)
    x = x.to(device)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x.cpu(), y_true.cpu(), color="black", lw=1, label="target")
    for i in range(n):
        mlp = build_target_mlp(cfg.target_mlp, device=device)
        mlp.load_flat(samples[i].to(device))
        with torch.no_grad():
            y_pred = mlp(x).cpu()
        ax.plot(x.cpu(), y_pred, lw=0.7, alpha=0.6)
    tag = "samples/weights_ema" if (use_ema and ema is not None) else "samples/weights"
    ax.set_title(f"Generated MLPs @ step {step}")
    ax.legend()
    writer.add_figure(tag, fig, step)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Tiny null-context helper (avoids importing contextlib just for this)
# --------------------------------------------------------------------------- #
class _nullcontext:
    def __init__(self, *args):
        pass

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(ddpm: DDPM, cfg: ExperimentConfig, step: int, out_dir: str,
                    ema: Optional[EMA] = None):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ckpt_{step}.pt")
    payload = {
        "step": step,
        "ddpm_state": ddpm.state_dict(),
        "config": cfg.to_dict(),
    }
    if ema is not None:
        payload["ema_state"] = ema.state_dict()
    torch.save(payload, path)
    # Also keep a "latest" pointer.
    latest = os.path.join(out_dir, "latest.pt")
    torch.save({"step": step, "path": path}, latest)


# --------------------------------------------------------------------------- #
# Validation loss
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _eval_val_loss(ddpm: DDPM, val_loader: DataLoader,
                   device: torch.device) -> float:
    """Mean DDPM loss over the validation set."""
    ddpm.eval()
    losses = []
    for batch in val_loader:
        x0 = batch.to(device).float()
        losses.append(ddpm.training_loss(x0).item())
    ddpm.train()
    return float(sum(losses) / max(len(losses), 1))


# --------------------------------------------------------------------------- #
# Main training entry point
# --------------------------------------------------------------------------- #
def train(cfg: ExperimentConfig):
    device = get_device(cfg.train.device)
    set_seed(cfg.train.seed)
    print(f"[train] device = {device}")

    # --- data ---
    dataset, data_dim = build_dataset(cfg, device)
    print(f"[train] data_source={cfg.data_source}  N={len(dataset)}  d={data_dim}")

    # --- train / val split ---
    val_fraction = cfg.train.val_fraction
    if val_fraction > 0 and len(dataset) > 10:
        n_val = max(1, int(len(dataset) * val_fraction))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(cfg.train.seed),
        )
        print(f"[train] split: train={n_train}  val={n_val}")
    else:
        train_ds, val_ds = dataset, None
        print("[train] no validation split (val_fraction=0 or dataset too small)")

    loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, drop_last=True, pin_memory=(device.type == "cuda"),
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=cfg.train.batch_size, shuffle=False,
            num_workers=cfg.train.num_workers, drop_last=False,
        )

    # --- model ---
    cfg.denoiser.data_dim = data_dim
    denoiser = build_denoiser(cfg.denoiser, device=device)
    ddpm = DDPM(denoiser, cfg.ddpm).to(device)

    opt = torch.optim.AdamW(denoiser.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)

    # --- EMA ---
    use_ema = cfg.train.ema_decay > 0
    ema = EMA(denoiser, decay=cfg.train.ema_decay) if use_ema else None
    if use_ema:
        print(f"[train] EMA enabled (decay={cfg.train.ema_decay}, "
              f"update_after={cfg.train.ema_update_after})")

    writer = SummaryWriter(os.path.join(cfg.train.log_dir, cfg.name))
    writer.add_text("config", str(cfg.to_dict()), 0)

    # --- loop ---
    step = 0
    for epoch in range(cfg.train.num_epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch}", leave=False)
        for batch in pbar:
            x0 = batch.to(device).float()
            loss = ddpm.training_loss(x0)
            opt.zero_grad()
            loss.backward()
            opt.step()

            # EMA update (after the optimizer step).
            if ema is not None:
                ema.update(denoiser, step=step,
                           update_after=cfg.train.ema_update_after)

            if step % cfg.train.log_every == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/lr", opt.param_groups[0]["lr"], step)

            # Validation loss.
            if val_loader is not None and step > 0 and step % cfg.train.val_every == 0:
                val_loss = _eval_val_loss(ddpm, val_loader, device)
                writer.add_scalar("val/loss", val_loss, step)

            if step % cfg.train.sample_every == 0:
                # Log a few denoiser weight histograms.
                for name, p in denoiser.named_parameters():
                    writer.add_histogram(f"weights/{name}", p.detach().cpu(), step)
                # Sample figures (use EMA weights if enabled & requested).
                use_ema_samp = use_ema and cfg.train.use_ema_for_sampling
                if cfg.data_source == "toy2d":
                    _log_toy2d_samples(ddpm, writer, step, device, use_ema=use_ema_samp, ema=ema)
                else:
                    _log_weights_samples(ddpm, writer, step, device, cfg,
                                         dataset, use_ema=use_ema_samp, ema=ema)

            if step > 0 and step % cfg.train.ckpt_every == 0:
                save_checkpoint(ddpm, cfg, step, cfg.train.ckpt_dir, ema=ema)

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            step += 1

    # Final checkpoint + samples.
    save_checkpoint(ddpm, cfg, step, cfg.train.ckpt_dir, ema=ema)
    use_ema_samp = use_ema and cfg.train.use_ema_for_sampling
    if cfg.data_source == "toy2d":
        _log_toy2d_samples(ddpm, writer, step, device, use_ema=use_ema_samp, ema=ema)
    else:
        _log_weights_samples(ddpm, writer, step, device, cfg,
                             dataset, use_ema=use_ema_samp, ema=ema)
    writer.close()
    print(f"[train] done. step={step}  checkpoints in {cfg.train.ckpt_dir}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", choices=["toy2d", "weights"], default="toy2d")
    parser.add_argument("--name", default="default")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None,
                        help="TargetMLP hidden width (weights data_source)")
    parser.add_argument("--n_networks", type=int, default=None,
                        help="population size (weights data_source)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no_ema", action="store_true", help="disable EMA")
    parser.add_argument("--no_canonicalize", action="store_true",
                        help="disable weight canonicalization (weights data_source)")
    args = parser.parse_args(argv)

    cfg = ExperimentConfig(name=args.name, data_source=args.data_source)
    if args.epochs is not None:
        cfg.train.num_epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.device is not None:
        cfg.train.device = args.device
    if args.hidden_dim is not None:
        cfg.target_mlp.hidden_dim = args.hidden_dim
    if args.n_networks is not None:
        cfg.weights_dataset.n_networks = args.n_networks
    if args.no_ema:
        cfg.train.ema_decay = 0.0
    if args.no_canonicalize:
        cfg.weights_dataset.canonicalize = False

    train(cfg)


if __name__ == "__main__":
    main()
