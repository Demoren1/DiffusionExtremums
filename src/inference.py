"""Inference / sampling script.

Loads a trained DDPM checkpoint, generates samples, and evaluates them:
  - toy2d:  scatter of generated points vs the two-moons ground truth.
  - weights: instantiate TargetMLP from generated weights, compare the
    reconstructed function to the target, and report extrema agreement.

Usage:
    python -m src.inference --ckpt checkpoints/latest.pt --data_source weights
    python -m src.inference --ckpt checkpoints/latest.pt --data_source toy2d
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import torch

from .config import ExperimentConfig
from .utils import get_device, set_seed
from .diffusion.denoiser import build_denoiser, MLPDenoiser
from .diffusion.ddpm import DDPM
from .diffusion.ema import EMA
from .target import build_target_mlp, sample_target_grid, target_function


# --------------------------------------------------------------------------- #
# Load model from checkpoint
# --------------------------------------------------------------------------- #
def load_ddpm(ckpt_path: str, device: torch.device,
              use_ema: bool = True) -> DDPM:
    """Load a DDPM from a checkpoint.

    If ``use_ema`` is True and the checkpoint contains EMA shadow weights, they
    are copied into the denoiser (overriding the raw trained weights) — this
    typically yields better samples.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    blob = torch.load(ckpt_path, map_location="cpu")
    # Resolve "latest" pointer.
    if "path" in blob and "ddpm_state" not in blob:
        blob = torch.load(blob["path"], map_location="cpu")

    cfg = ExperimentConfig.from_dict(blob["config"])
    cfg.denoiser.data_dim = cfg.target_mlp.n_params if cfg.data_source == "weights" else 2
    denoiser = build_denoiser(cfg.denoiser, device=device)
    ddpm = DDPM(denoiser, cfg.ddpm).to(device)
    ddpm.load_state_dict(blob["ddpm_state"])

    # Load EMA weights into the denoiser if available and requested.
    ema_sd = blob.get("ema_state")
    if use_ema and ema_sd is not None:
        ema = EMA(denoiser, decay=0.0)
        ema.load_state_dict(ema_sd)
        ema.copy_to(denoiser)
        print("[infer] using EMA weights for sampling")
    elif use_ema and ema_sd is None:
        print("[infer] EMA requested but checkpoint has none; using raw weights")

    ddpm.eval()
    return ddpm, cfg


# --------------------------------------------------------------------------- #
# Extrema utilities
# --------------------------------------------------------------------------- #
def find_local_extrema(x: torch.Tensor, y: torch.Tensor, kind: str = "min"):
    """Return x-coordinates of local extrema of y sampled on grid x.

    A point is a local min (max) if it is lower (higher) than both neighbours.
    """
    y = y.squeeze()
    x = x.squeeze()
    if kind == "min":
        mask = (y[1:-1] < y[:-2]) & (y[1:-1] < y[2:])
    else:
        mask = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
    return x[1:-1][mask].cpu().numpy()


def match_extrema(pred: np.ndarray, true: np.ndarray, tol: float) -> int:
    """Count how many predicted extrema are within ``tol`` of a true one."""
    count = 0
    used = np.zeros(len(true), dtype=bool)
    for p in pred:
        dists = np.abs(true - p)
        j = int(np.argmin(dists))
        if dists[j] < tol and not used[j]:
            count += 1
            used[j] = True
    return count


# --------------------------------------------------------------------------- #
# Inference: weights
# --------------------------------------------------------------------------- #
def infer_weights(ddpm: DDPM, cfg: ExperimentConfig, device: torch.device):
    set_seed(cfg.inference.seed)
    n = cfg.inference.n_samples
    d = cfg.target_mlp.n_params
    n_steps = cfg.inference.sampling_timesteps or cfg.ddpm.num_timesteps

    samples = ddpm.sample((n, d), device, n_steps=n_steps).cpu()

    # De-normalize.
    path = cfg.weights_dataset.out_path
    if os.path.exists(path):
        blob = torch.load(path, map_location="cpu")
        mean, std = blob.get("mean"), blob.get("std")
        if mean is not None and std is not None:
            samples = samples * std + mean

    # Evaluation grid.
    x = torch.linspace(cfg.inference.eval_x_min, cfg.inference.eval_x_max,
                       cfg.inference.eval_n).unsqueeze(1).to(device)
    y_true = target_function(x, cfg.target).cpu().squeeze().numpy()
    true_minima = find_local_extrema(x.cpu(), torch.from_numpy(y_true), "min")
    true_maxima = find_local_extrema(x.cpu(), torch.from_numpy(y_true), "max")

    mses, min_match, max_match, n_valid = [], [], [], 0
    for i in range(n):
        mlp = build_target_mlp(cfg.target_mlp, device=device)
        mlp.load_flat(samples[i].to(device))
        with torch.no_grad():
            y_pred = mlp(x).cpu().squeeze().numpy()
        if not np.isfinite(y_pred).all():
            continue
        n_valid += 1
        mses.append(float(np.mean((y_pred - y_true) ** 2)))
        pred_min = find_local_extrema(x.cpu(), torch.from_numpy(y_pred), "min")
        pred_max = find_local_extrema(x.cpu(), torch.from_numpy(y_pred), "max")
        # Tolerance ~ half the local period of the slower component.
        tol = np.pi / max(cfg.target.k)
        min_match.append(match_extrema(pred_min, true_minima, tol))
        max_match.append(match_extrema(pred_max, true_maxima, tol))

    print(f"[infer] valid networks: {n_valid}/{n}")
    if mses:
        print(f"[infer] MSE  mean={np.mean(mses):.4f}  median={np.median(mses):.4f}")
        print(f"[infer] minima matched: mean={np.mean(min_match):.1f} / {len(true_minima)}")
        print(f"[infer] maxima matched: mean={np.mean(max_match):.1f} / {len(true_maxima)}")

    # Save a figure.
    os.makedirs(cfg.inference.out_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x.cpu(), y_true, color="black", lw=1.2, label="target")
    for i in range(min(n, 8)):
        mlp = build_target_mlp(cfg.target_mlp, device=device)
        mlp.load_flat(samples[i].to(device))
        with torch.no_grad():
            y_pred = mlp(x).cpu().squeeze().numpy()
        ax.plot(x.cpu(), y_pred, lw=0.7, alpha=0.6)
    ax.set_title("Generated MLPs vs target")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(cfg.inference.out_dir, "weights_inference.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[infer] figure saved -> {out}")


# --------------------------------------------------------------------------- #
# Inference: toy2d
# --------------------------------------------------------------------------- #
def infer_toy2d(ddpm: DDPM, cfg: ExperimentConfig, device: torch.device):
    set_seed(cfg.inference.seed)
    n = cfg.inference.n_samples
    n_steps = cfg.inference.sampling_timesteps or cfg.ddpm.num_timesteps
    samples = ddpm.sample((n, 2), device, n_steps=n_steps).cpu().numpy()

    from .datasets.toy2d import make_two_moons
    gt = make_two_moons(cfg.toy2d.n_samples, cfg.toy2d.noise, cfg.toy2d.seed)

    os.makedirs(cfg.inference.out_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].scatter(gt[:, 0], gt[:, 1], s=1, alpha=0.3, c="#2563eb"); axes[0].set_title("ground truth")
    axes[1].scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.3, c="#e11d48"); axes[1].set_title("DDPM samples")
    for ax in axes:
        ax.set_aspect("equal")
    fig.tight_layout()
    out = os.path.join(cfg.inference.out_dir, "toy2d_inference.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[infer] figure saved -> {out}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/latest.pt")
    parser.add_argument("--data_source", choices=["toy2d", "weights"], default=None,
                        help="if omitted, read from checkpoint config")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no_ema", action="store_true",
                        help="use raw trained weights instead of EMA weights")
    args = parser.parse_args(argv)

    device = get_device(args.device or "auto")
    ddpm, cfg = load_ddpm(args.ckpt, device, use_ema=not args.no_ema)
    if args.data_source is not None:
        cfg.data_source = args.data_source
    if args.n_samples is not None:
        cfg.inference.n_samples = args.n_samples

    print(f"[infer] data_source={cfg.data_source}  device={device}")
    if cfg.data_source == "weights":
        infer_weights(ddpm, cfg, device)
    else:
        infer_toy2d(ddpm, cfg, device)


if __name__ == "__main__":
    main()
