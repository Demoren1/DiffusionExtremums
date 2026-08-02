"""Evaluation: 3-method comparison + Toeplitz-ness + kernel recovery."""
import csv
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from src.configs.base import DatasetConfig
from src.data.dataset import conv1d_same, generate_dataset
from src.evaluation._shared import (
    config_dict_to_datasetconfig,
    mse,
    sample_eval_config_indices,
    stats_of,
)
from src.smoke.models import Conv1dModel

DEFAULT_L: int = 32
DEFAULT_H: int = 128


def kernel_to_effective_map(
    kernel: torch.Tensor,
    L: int = DEFAULT_L,
) -> torch.Tensor:
    """Build the oracle Toeplitz effective map from a convolution kernel."""
    k = kernel.reshape(-1).to(torch.float32)
    K = k.shape[0]
    if K % 2 == 0:
        raise ValueError(f"kernel size K must be odd, got {K}")
    r = K // 2
    M = torch.zeros(L, L, dtype=torch.float32)
    for i in range(L):
        for j in range(L):
            d = i - j
            idx = d + r
            if 0 <= idx < K:
                M[i, j] = k[idx]
    b_eff = torch.zeros(L, dtype=torch.float32)
    return torch.cat([M.reshape(-1), b_eff], dim=0)


def effective_map_to_matrix(
    eff_map: torch.Tensor,
    L: int = DEFAULT_L,
) -> torch.Tensor:
    """Extract the LxL effective matrix M from a 1056-dim effective map."""
    e = eff_map.reshape(*eff_map.shape[:-1], L * L + L)
    M_flat = e[..., :L * L]
    return M_flat.reshape(*e.shape[:-1], L, L)
from src.training.train_mlp import TrainConfig, train_mlp_to_convergence


@dataclass
class MethodResult:
    method: str
    test_mse: float
    train_mse: float
    n_samples: int = 1
    test_mse_std: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None


@dataclass
class DatasetEval:
    config_idx: int
    split: str
    family: str
    radius: int
    noise_std: float
    methods: Dict[str, MethodResult] = field(default_factory=dict)
    toeplitzness: Dict[str, float] = field(default_factory=dict)
    kernel_recovery: Dict[str, float] = field(default_factory=dict)


def eval_oracle_conv(
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    kernel: torch.Tensor,
) -> MethodResult:
    k_np = kernel.reshape(-1).cpu().numpy().astype(np.float32)
    x_test_np = x_test.cpu().numpy().astype(np.float32)
    x_train_np = x_train.cpu().numpy().astype(np.float32)
    pred_test = torch.from_numpy(conv1d_same(x_test_np, k_np))
    pred_train = torch.from_numpy(conv1d_same(x_train_np, k_np))
    return MethodResult(
        method="oracle_conv",
        test_mse=mse(pred_test, y_test.cpu()),
        train_mse=mse(pred_train, y_train.cpu()),
    )


def train_learned_conv(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    kernel_size: int,
    device: Union[str, torch.device] = "cuda",
    steps: int = 3000,
    lr: float = 3e-3,
    seed: int = 0,
) -> MethodResult:
    device = torch.device(device)
    torch.manual_seed(seed)
    model = Conv1dModel(L=x_train.shape[1], kernel_size=kernel_size, bias=False)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)

    x_tr = x_train.to(device)
    y_tr = y_train.to(device)
    x_te = x_test.to(device)
    y_te = y_test.to(device)

    for step in range(1, steps + 1):
        model.train()
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x_tr), y_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        test_mse = mse(model(x_te), y_te)
        train_mse = mse(model(x_tr), y_tr)
    return MethodResult(method="learned_conv", test_mse=test_mse, train_mse=train_mse)


def eval_from_scratch_mlp(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    n_train: int = 1024,
    device: Union[str, torch.device] = "cuda",
    seed: int = 0,
) -> MethodResult:
    if x_train.shape[0] > n_train:
        rng = np.random.default_rng(seed)
        idx = torch.from_numpy(rng.choice(x_train.shape[0], n_train,
                                          replace=False)).long()
        x_tr = x_train[idx]
        y_tr = y_train[idx]
    else:
        x_tr = x_train
        y_tr = y_train

    cfg = TrainConfig(
        L=x_train.shape[1],
        H=128,
        seed=seed,
        device=str(device),
        steps=3000,
        min_steps=200,
        patience=200,
    )
    result = train_mlp_to_convergence(x_tr, y_tr, x_test, y_test, config=cfg)
    return MethodResult(
        method="from_scratch_mlp",
        test_mse=result.test_mse,
        train_mse=result.train_mse,
    )


def toeplitzness(M: torch.Tensor) -> float:
    """Toeplitz-ness score: mean diagonal std (lower = more Toeplitz)."""
    M = M.detach().cpu().float()
    L = M.shape[0]
    stds: List[float] = []
    for d in range(-(L - 1), L):
        diag = torch.diagonal(M, offset=d, dim1=0, dim2=1)
        if diag.numel() > 1:
            stds.append(float(diag.std(unbiased=False).item()))
    if not stds:
        return 0.0
    return float(np.mean(stds))


def recover_kernel_from_matrix(
    M: torch.Tensor,
    max_radius: int = 3,
) -> torch.Tensor:
    """Extract the implied kernel by averaging each diagonal of M."""
    M = M.detach().cpu().float()
    L = M.shape[0]
    K = 2 * max_radius + 1
    kernel = torch.zeros(K)
    for j, d in enumerate(range(-max_radius, max_radius + 1)):
        diag = torch.diagonal(M, offset=d, dim1=0, dim2=1)
        if diag.numel() > 0:
            kernel[j] = diag.mean()
    return kernel


def kernel_recovery(
    generated_M: torch.Tensor,
    gt_kernel: torch.Tensor,
    max_radius: int = 3,
) -> Dict[str, float]:
    recovered = recover_kernel_from_matrix(generated_M, max_radius=max_radius)
    K_rec = recovered.shape[0]
    K_gt = gt_kernel.shape[0]
    r_gt = K_gt // 2
    gt_padded = torch.zeros(K_rec)
    offset = max_radius - r_gt
    for j, v in enumerate(gt_kernel.tolist()):
        if 0 <= offset + j < K_rec:
            gt_padded[offset + j] = v

    a = recovered
    b = gt_padded
    na = a.norm()
    nb = b.norm()
    if na < 1e-12 or nb < 1e-12:
        cos_sim = 0.0
    else:
        cos_sim = float((torch.dot(a, b) / (na * nb)).item())
    l2 = float((a - b).norm().item())
    return {"cosine_sim": cos_sim, "l2_dist": l2}


def evaluate_one_dataset(
    config: Union[DatasetConfig, Dict],
    config_idx: int,
    split: str,
    n_train_mlp: int = 1024,
    device: Union[str, torch.device] = "cuda",
    target_eff_map: Optional[torch.Tensor] = None,
    L: int = DEFAULT_L,
    H: int = DEFAULT_H,
    seed: int = 0,
) -> DatasetEval:
    device = torch.device(device)
    if isinstance(config, dict):
        ds_config = config_dict_to_datasetconfig(config)
        family = str(config["family"])
        radius = int(config["radius"])
        noise_std = float(config["noise_std"])
    else:
        ds_config = config
        family = config.family
        radius = config.radius
        noise_std = config.noise_std

    x_train, y_train, x_test, y_test, kernel, _ = generate_dataset(ds_config)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    kernel = kernel.to(device)

    result = DatasetEval(
        config_idx=config_idx,
        split=split,
        family=family,
        radius=radius,
        noise_std=noise_std,
    )

    result.methods["from_scratch_mlp"] = eval_from_scratch_mlp(
        x_train, y_train, x_test, y_test,
        n_train=n_train_mlp, device=device, seed=seed)

    K = 2 * radius + 1
    result.methods["learned_conv"] = train_learned_conv(
        x_train, y_train, x_test, y_test,
        kernel_size=K, device=device, seed=seed)

    result.methods["oracle_conv"] = eval_oracle_conv(
        x_test, y_test, x_train, y_train, kernel)

    oracle_M = effective_map_to_matrix(kernel_to_effective_map(kernel, L=L))
    result.toeplitzness["oracle"] = toeplitzness(oracle_M)
    if target_eff_map is not None:
        target_M = effective_map_to_matrix(target_eff_map, L=L)
        result.toeplitzness["target"] = toeplitzness(target_M)

    return result


def evaluate_datasets(
    configs: List[Dict],
    train_config_indices: List[int],
    val_config_indices: List[int],
    n_eval_train: int = 20,
    n_eval_val: int = 20,
    n_train_mlp: int = 1024,
    device: Union[str, torch.device] = "cuda",
    target_eff_maps: Optional[torch.Tensor] = None,
    L: int = DEFAULT_L,
    H: int = DEFAULT_H,
    seed: int = 0,
    verbose: bool = True,
) -> List[DatasetEval]:
    device = torch.device(device)
    all_indices = sample_eval_config_indices(
        train_config_indices, val_config_indices,
        n_eval_train, n_eval_val, seed)

    results: List[DatasetEval] = []

    for k, (idx, split) in enumerate(all_indices):
        if verbose:
            print(f"  [{k+1}/{len(all_indices)}] config {idx} ({split}, "
                  f"family={configs[idx]['family']}, "
                  f"radius={configs[idx]['radius']})")
        target_eff = None
        if target_eff_maps is not None:
            target_eff = target_eff_maps[idx, 0]
        res = evaluate_one_dataset(
            configs[idx], idx, split,
            n_train_mlp=n_train_mlp, device=device,
            target_eff_map=target_eff, L=L, H=H, seed=seed + k)
        results.append(res)

    return results


def aggregate_results(results: List[DatasetEval]) -> Dict[str, Any]:
    methods = ["from_scratch_mlp", "learned_conv", "oracle_conv"]
    splits = ["train", "val"]

    by_method: Dict[str, Dict[str, Dict]] = {m: {s: [] for s in splits} for m in methods}
    toeplitz: Dict[str, Dict[str, list]] = {src: {s: [] for s in splits}
                                            for src in ["oracle", "target"]}
    kernel_rec: Dict[str, Dict[str, list]] = {s: {"cosine_sim": [], "l2_dist": []}
                                              for s in splits}
    rel_to_conv: Dict[str, Dict[str, list]] = {m: {s: [] for s in splits} for m in methods}

    for r in results:
        s = r.split
        conv_mse = r.methods["learned_conv"].test_mse if "learned_conv" in r.methods else None
        for m in methods:
            if m in r.methods:
                by_method[m][s].append(r.methods[m].test_mse)
                if conv_mse and conv_mse > 1e-12:
                    rel_to_conv[m][s].append(r.methods[m].test_mse / conv_mse)
        for src, val in r.toeplitzness.items():
            toeplitz[src][s].append(val)
        if r.kernel_recovery:
            kernel_rec[s]["cosine_sim"].append(r.kernel_recovery.get("cosine_sim", 0.0))
            kernel_rec[s]["l2_dist"].append(r.kernel_recovery.get("l2_dist", 0.0))

    summary = {
        "by_method": {m: {s: stats_of(by_method[m][s]) for s in splits} for m in methods},
        "toeplitzness": {src: {s: stats_of(toeplitz[src][s]) for s in splits}
                         for src in toeplitz},
        "kernel_recovery": {s: {k: stats_of(v) for k, v in kernel_rec[s].items()}
                            for s in splits},
        "relative_to_conv": {m: {s: stats_of(rel_to_conv[m][s]) for s in splits}
                             for m in methods},
    }
    return summary


def print_comparison_table(results: List[DatasetEval], summary: Dict) -> None:
    """Print a human-readable comparison table to stdout."""
    methods = ["from_scratch_mlp", "learned_conv", "oracle_conv"]
    splits = ["train", "val"]

    print("\n" + "=" * 80)
    print(" EVALUATION RESULTS: Test MSE (mean ± std across datasets)")
    print("=" * 80)
    header = f"{'Method':<22} {'train configs':>20} {'val configs':>20}"
    print(header)
    print("-" * 80)
    for m in methods:
        row = f"{m:<22}"
        for s in splits:
            st = summary["by_method"][m][s]
            if st["mean"] is not None:
                row += f"  {st['mean']:.6f}±{st['std']:.6f}".rjust(18) + "  "
            else:
                row += "  N/A".rjust(18) + "  "
        print(row)

    print("\n" + "-" * 80)
    print(" Relative-to-learned-conv MSE (method MSE / learned conv MSE)")
    print("-" * 80)
    print(f"{'Method':<22} {'train':>20} {'val':>20}")
    for m in methods:
        row = f"{m:<22}"
        for s in splits:
            st = summary["relative_to_conv"][m][s]
            if st["mean"] is not None:
                row += f"  {st['mean']:.3f}±{st['std']:.3f}".rjust(18) + "  "
            else:
                row += "  N/A".rjust(18) + "  "
        print(row)

    print("\n" + "-" * 80)
    print(" Toeplitz-ness score (mean diagonal std; lower = more Toeplitz)")
    print("-" * 80)
    print(f"{'Source':<22} {'train':>20} {'val':>20}")
    for src in ["oracle", "target"]:
        row = f"{src:<22}"
        for s in splits:
            st = summary["toeplitzness"][src][s]
            if st["mean"] is not None:
                row += f"  {st['mean']:.6f}±{st['std']:.6f}".rjust(18) + "  "
            else:
                row += "  N/A".rjust(18) + "  "
        print(row)

    print("\n" + "-" * 80)
    print(" Kernel recovery (generated M -> recovered kernel vs ground truth)")
    print("-" * 80)
    print(f"{'Metric':<22} {'train':>20} {'val':>20}")
    for k in ["cosine_sim", "l2_dist"]:
        row = f"{k:<22}"
        for s in splits:
            st = summary["kernel_recovery"][s][k]
            if st["mean"] is not None:
                row += f"  {st['mean']:.6f}±{st['std']:.6f}".rjust(18) + "  "
            else:
                row += "  N/A".rjust(18) + "  "
        print(row)
    print("=" * 80)


def save_results(
    results: List[DatasetEval],
    summary: Dict,
    output_dir: str,
) -> None:
    """Save evaluation results (metrics) as JSON and CSV to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    def _result_to_dict(r: DatasetEval) -> Dict:
        d = {
            "config_idx": r.config_idx,
            "split": r.split,
            "family": r.family,
            "radius": r.radius,
            "noise_std": r.noise_std,
            "methods": {m: asdict(v) for m, v in r.methods.items()},
            "toeplitzness": r.toeplitzness,
            "kernel_recovery": r.kernel_recovery,
        }
        return d

    full = [_result_to_dict(r) for r in results]
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(full, f, indent=2)

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "name", "split", "mean", "std", "n"])
        for m, sd in summary["by_method"].items():
            for s, st in sd.items():
                w.writerow(["test_mse", m, s, st["mean"], st["std"], st["n"]])
        for src, sd in summary["toeplitzness"].items():
            for s, st in sd.items():
                w.writerow(["toeplitzness", src, s, st["mean"], st["std"], st["n"]])
        for s, sd in summary["kernel_recovery"].items():
            for k, st in sd.items():
                w.writerow(["kernel_recovery", k, s, st["mean"], st["std"], st["n"]])
        for m, sd in summary["relative_to_conv"].items():
            for s, st in sd.items():
                w.writerow(["relative_to_conv", m, s, st["mean"], st["std"], st["n"]])

    print(f"\nResults saved to {output_dir}/ (results.json, summary.json, summary.csv)")
