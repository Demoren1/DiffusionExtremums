"""CLI: convert 8352-dim MLP weight targets to 1056-dim effective maps (Strategy B).

Loads the Phase 2 collected targets (``weights.pt`` [n_datasets, n_mlp, 8352])
and converts each weight vector to the effective linear map
``(M = W2 @ W1, b_eff = W2 @ b1 + b2)`` of shape [1056], removing the gauge
freedom of the linear-MLP factorization. Saves the converted targets to a new
directory alongside copies of the config / id / loss files.

Verification: all 50 MLPs per dataset compute the same function, so their
effective maps should be (approximately) identical. The script computes and
reports the per-dataset std of the effective maps across the 50 MLPs; it should
be small (confirming the gauge freedom is removed).

Output layout (``data/processed/targets_eff/``)::

    eff_maps.pt      : float32 tensor [n_datasets, n_mlp, 1056]
    configs.json     : copy of the original config list
    dataset_ids.json : copy of the original dataset_id list
    losses.pt        : copy of the original losses tensor
    metadata.json    : documents the 1056-dim format + SVD factorization

Examples::

    # Default: convert data/processed/targets -> data/processed/targets_eff
    python -m src.scripts.convert_targets

    # Custom paths
    python -m src.scripts.convert_targets \
        --in-dir data/processed/targets \
        --out-dir data/processed/targets_eff

Run with the conda env activated:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate myenv
"""
import argparse
import json
import os
import shutil
import sys
from typing import Dict

import torch

from src.models.effective_map import (
    DEFAULT_EFF_D,
    DEFAULT_H,
    DEFAULT_L,
    EffectiveMapCodec,
    weights_to_effective_map,
)


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Convert 8352-dim MLP weight targets to 1056-dim effective "
                    "maps (Strategy B).")
    p.add_argument("--in-dir", type=str, default="data/processed/targets",
                   help="Directory with the original 8352-dim targets "
                        "(weights.pt, configs.json, ...).")
    p.add_argument("--out-dir", type=str, default="data/processed/targets_eff",
                   help="Output directory for the 1056-dim effective-map "
                        "targets.")
    p.add_argument("--L", type=int, default=DEFAULT_L,
                   help="MLP input/output dimension (default 32).")
    p.add_argument("--H", type=int, default=DEFAULT_H,
                   help="MLP hidden width (default 128).")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device for the conversion (default cuda; the SVD is "
                        "batched over 25000 vectors).")
    p.add_argument("--batch-size", type=int, default=5000,
                   help="Batch size for the batched conversion (default 5000).")
    p.add_argument("--no-copy-losses", action="store_true",
                   help="Do not copy losses.pt to the output dir.")
    return p.parse_args(argv)


def _load_json(path: str):
    """Load a JSON file and return the parsed object."""
    with open(path, "r") as f:
        return json.load(f)


def _save_json(obj, path: str) -> None:
    """Write ``obj`` as indented JSON to ``path``."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def convert_targets(in_dir: str, out_dir: str, L: int = DEFAULT_L,
                    H: int = DEFAULT_H, device: str = "cuda",
                    batch_size: int = 5000,
                    copy_losses: bool = True) -> Dict:
    """Convert the 8352-dim targets to 1056-dim effective maps.

    Args:
        in_dir: Directory with the original targets (weights.pt, ...).
        out_dir: Output directory for the effective-map targets.
        L, H: MLP dimensions.
        device: torch device for the conversion.
        batch_size: Batch size for the batched SVD-free forward conversion.
        copy_losses: If True, copy losses.pt to the output dir.

    Returns:
        A dict of diagnostics: shapes, per-dataset std statistics.
    """
    if not os.path.isdir(in_dir):
        raise FileNotFoundError(f"in-dir not found: {in_dir}")

    weights_path = os.path.join(in_dir, "weights.pt")
    configs_path = os.path.join(in_dir, "configs.json")
    ids_path = os.path.join(in_dir, "dataset_ids.json")
    losses_path = os.path.join(in_dir, "losses.pt")
    meta_path = os.path.join(in_dir, "metadata.json")
    for p in (weights_path, configs_path, ids_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing required file: {p}")

    print(f"[convert_targets] loading weights from {weights_path}")
    weights = torch.load(weights_path, map_location="cpu").float()
    if weights.dim() != 3:
        raise ValueError(
            f"weights.pt must be 3-D [n_configs, n_mlp, D], got "
            f"{tuple(weights.shape)}")
    n_configs, n_mlp, D = weights.shape[0], weights.shape[1], weights.shape[2]
    print(f"[convert_targets] weights: {tuple(weights.shape)} "
          f"(n_configs={n_configs}, n_mlp={n_mlp}, D={D})")

    configs = _load_json(configs_path)
    dataset_ids = _load_json(ids_path)
    if len(configs) != n_configs:
        raise ValueError(
            f"configs.json has {len(configs)} entries != weights axis 0 "
            f"{n_configs}")
    if len(dataset_ids) != n_configs:
        raise ValueError(
            f"dataset_ids.json has {len(dataset_ids)} entries != {n_configs}")

    # Original metadata (for reference in the new metadata).
    orig_meta = _load_json(meta_path) if os.path.exists(meta_path) else {}

    # ------------------------------------------------------------------
    # Convert: [n_configs, n_mlp, D] -> [n_configs, n_mlp, D_eff]
    # ------------------------------------------------------------------
    codec = EffectiveMapCodec(L=L, H=H)
    if D != codec.D:
        raise ValueError(
            f"weights D={D} != codec D={codec.D} for L={L}, H={H}")
    D_eff = codec.D_eff
    print(f"[convert_targets] converting {n_configs * n_mlp} weight vectors "
          f"({D}-dim) -> effective maps ({D_eff}-dim) on {device}")

    dev = torch.device(device if torch.cuda.is_available() and device == "cuda"
                       else device)
    flat = weights.reshape(n_configs * n_mlp, D)  # [N, D]
    eff_flat = torch.empty(n_configs * n_mlp, D_eff, dtype=torch.float32)
    n_done = 0
    with torch.no_grad():
        for i in range(0, flat.shape[0], batch_size):
            chunk = flat[i:i + batch_size].to(dev)
            eff_chunk = weights_to_effective_map(chunk, L=L, H=H).cpu()
            eff_flat[i:i + batch_size] = eff_chunk
            n_done = min(i + batch_size, flat.shape[0])
            if n_done % (5 * batch_size) == 0 or n_done == flat.shape[0]:
                print(f"  converted {n_done}/{flat.shape[0]}")
    eff_maps = eff_flat.view(n_configs, n_mlp, D_eff).contiguous()
    print(f"[convert_targets] eff_maps: {tuple(eff_maps.shape)}")

    # ------------------------------------------------------------------
    # Verification: per-dataset std of eff maps across the 50 MLPs.
    # All 50 MLPs compute the same function -> same (M, b_eff) -> low std.
    # ------------------------------------------------------------------
    # std over the n_mlp axis: [n_configs, D_eff]
    per_dataset_std = eff_maps.std(dim=1, unbiased=False)  # [n_configs, D_eff]
    # Summary: mean / median / max std across dims, averaged over configs.
    mean_std = per_dataset_std.mean().item()
    median_std = per_dataset_std.median().item()
    max_std = per_dataset_std.max().item()
    # Also the per-dataset mean std (averaged over dims), then stats over configs.
    per_dataset_mean_std = per_dataset_std.mean(dim=1)  # [n_configs]
    print(f"[convert_targets] per-dataset eff-map std (across {n_mlp} MLPs):")
    print(f"  mean std (all dims, all configs) = {mean_std:.6e}")
    print(f"  median std                       = {median_std:.6e}")
    print(f"  max std                          = {max_std:.6e}")
    print(f"  per-dataset mean std: "
          f"min={per_dataset_mean_std.min().item():.6e}, "
          f"max={per_dataset_mean_std.max().item():.6e}, "
          f"mean={per_dataset_mean_std.mean().item():.6e}")

    # Compare to the original weights' per-dataset std (gauge freedom is large).
    orig_per_dataset_std = weights.std(dim=1, unbiased=False)  # [n_configs, D]
    print(f"[convert_targets] original weights per-dataset std (for comparison):")
    print(f"  mean std = {orig_per_dataset_std.mean().item():.6e}, "
          f"max std = {orig_per_dataset_std.max().item():.6e}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    eff_path = os.path.join(out_dir, "eff_maps.pt")
    torch.save(eff_maps, eff_path)
    print(f"[convert_targets] saved eff_maps -> {eff_path}")

    # Copy configs, dataset_ids, losses.
    shutil.copyfile(configs_path, os.path.join(out_dir, "configs.json"))
    shutil.copyfile(ids_path, os.path.join(out_dir, "dataset_ids.json"))
    if copy_losses and os.path.exists(losses_path):
        shutil.copyfile(losses_path, os.path.join(out_dir, "losses.pt"))
        print(f"[convert_targets] copied losses.pt")

    # Write metadata.
    metadata = {
        "format_version": 2,
        "strategy": "B (effective linear map, no gauge freedom)",
        "description": (
            "Effective linear map of the converged linear MLP "
            "(Linear(32,128)->Linear(128,32)). Removes the gauge freedom of "
            "the (W1,W2) factorization: all 50 MLPs per dataset compute the "
            "same function -> the same (M, b_eff)."),
        "files": {
            "eff_maps.pt": f"float32 tensor [{n_configs}, {n_mlp}, {D_eff}] "
                           "of effective-map vectors.",
            "configs.json": "copy of the original config list (conditioning).",
            "dataset_ids.json": "copy of the original dataset_id list.",
            "losses.pt": "copy of the original losses tensor "
                         "[n_configs, n_mlp, 3].",
            "metadata.json": "this file.",
        },
        "effective_map_layout": {
            "order": ["M_flat", "b_eff"],
            "M_flat": {
                "shape": [L, L],
                "size": L * L,
                "offset": 0,
                "layout": "row-major (C-order); M[i,j] at index i*L+j",
                "definition": "M = fc2.weight @ fc1.weight  (effective map)",
            },
            "b_eff": {
                "shape": [L],
                "size": L,
                "offset": L * L,
                "definition": "b_eff = fc2.weight @ fc1.bias + fc2.bias",
            },
            "D_eff": D_eff,
        },
        "svd_factorization": {
            "description": (
                "To instantiate an MLP from (M, b_eff): SVD M = U @ diag(S) "
                "@ V^T; W1[H,L] first L rows = diag(sqrt(S)) @ V^T, rest 0; "
                "W2[L,H] first L cols = U @ diag(sqrt(S)), rest 0; b1[H]=0, "
                "b2[L]=b_eff. Verify: W2 @ W1 = U @ diag(S) @ V^T = M."),
            "svd_floor": 1e-10,
        },
        "mlp_architecture": {
            "class": "src.models.mlp.MLPModel",
            "L": L,
            "H": H,
            "n_params_full": D,
            "n_params_effective": D_eff,
            "activation": "none (linear MLP)",
        },
        "collection": orig_meta.get("collection", {}),
        "verification": {
            "per_dataset_std_mean": mean_std,
            "per_dataset_std_median": median_std,
            "per_dataset_std_max": max_std,
            "original_weights_std_mean": orig_per_dataset_std.mean().item(),
            "note": ("The per-dataset std of the effective maps across the 50 "
                     "MLPs should be small (they compute the same function), "
                     "confirming the gauge freedom is removed."),
        },
    }
    _save_json(metadata, os.path.join(out_dir, "metadata.json"))
    print(f"[convert_targets] saved metadata -> {os.path.join(out_dir, 'metadata.json')}")

    return {
        "n_configs": n_configs,
        "n_mlp": n_mlp,
        "D": D,
        "D_eff": D_eff,
        "eff_maps_shape": tuple(eff_maps.shape),
        "per_dataset_std_mean": mean_std,
        "per_dataset_std_median": median_std,
        "per_dataset_std_max": max_std,
        "original_weights_std_mean": orig_per_dataset_std.mean().item(),
    }


def main(argv=None) -> int:
    """Run the weight->effective-map conversion."""
    args = parse_args(argv)
    diag = convert_targets(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        L=args.L,
        H=args.H,
        device=args.device,
        batch_size=args.batch_size,
        copy_losses=not args.no_copy_losses,
    )
    print("\n[convert_targets] DONE. Summary:")
    for k, v in diag.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
