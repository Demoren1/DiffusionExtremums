"""Smoke test for the multi-GPU parallel target collection (Phase 2+).

Verifies that the multi-GPU parallelization (``torch.multiprocessing.spawn``,
one process per GPU, corpus sharding + merge) produces output that is:

(a) **Format-correct**: the merged files (``weights.pt``, ``losses.pt``,
    ``configs.json``, ``dataset_ids.json``, ``metadata.json``) have exactly the
    Phase 2 shapes and contents (``weights [n_datasets, n_mlp, D]`` with
    ``D = 8352``, ``losses [n_datasets, n_mlp, 3]``, etc.).
(b) **Equivalent to the sequential path**: running the same config sequentially
    on one GPU and in parallel across multiple GPUs gives (near-)identical
    weights and losses. This holds because the corpus is generated
    deterministically from ``corpus_seed`` and ``mlp_seed_base`` depends only on
    the global dataset index, so sharding does not change any per-dataset result.
(c) **Resumable**: an interrupted parallel run (simulated by a partial shard
    checkpoint) resumes and completes correctly.
(d) **Clean**: no leftover shard / checkpoint files after a successful merge.

If fewer than 2 CUDA GPUs are available, the multi-GPU comparison is skipped
(the single-GPU sequential path is still exercised as a regression check).

Run with::

    python -m src.smoke.test_collect_parallel
"""
import json
import os
import shutil
import sys
import tempfile

import torch

from src.models.weight_codec import WeightCodec
from src.training.collect_targets import (
    CollectConfig,
    _merge_shards,
    _shard_indices,
    collect_targets,
    collect_targets_parallel,
    resolve_gpus,
)
from src.training.train_mlp import TrainConfig


def _make_config(out_dir: str, n_datasets: int = 4, n_mlp: int = 2,
                 gpus=None) -> CollectConfig:
    """Build a small, fast CollectConfig for testing."""
    return CollectConfig(
        n_datasets=n_datasets,
        n_mlp=n_mlp,
        n_train=256,
        n_test=128,
        noise_std=0.1,
        corpus_seed=7,
        L=32,
        H=128,
        out_dir=out_dir,
        gpus=gpus,
        train=TrainConfig(lr=3e-3, steps=300, min_steps=100, patience=100,
                          L=32, H=128, device="cuda"),
    )


def _check_output_format(out_dir: str, n_datasets: int, n_mlp: int) -> None:
    """Assert the output directory has the correct Phase 2 format."""
    codec = WeightCodec(L=32, H=128)
    for fn in ("weights.pt", "losses.pt", "configs.json",
               "dataset_ids.json", "metadata.json"):
        path = os.path.join(out_dir, fn)
        assert os.path.exists(path), f"missing output file: {fn}"

    weights = torch.load(os.path.join(out_dir, "weights.pt"))
    losses = torch.load(os.path.join(out_dir, "losses.pt"))
    with open(os.path.join(out_dir, "configs.json")) as f:
        configs = json.load(f)
    with open(os.path.join(out_dir, "dataset_ids.json")) as f:
        ids = json.load(f)
    with open(os.path.join(out_dir, "metadata.json")) as f:
        meta = json.load(f)

    assert weights.shape == (n_datasets, n_mlp, codec.D), \
        f"weights shape {weights.shape} != ({n_datasets}, {n_mlp}, {codec.D})"
    assert weights.dtype == torch.float32, f"weights dtype {weights.dtype}"
    assert losses.shape == (n_datasets, n_mlp, 3), \
        f"losses shape {losses.shape} != ({n_datasets}, {n_mlp}, 3)"
    assert losses.dtype == torch.float32, f"losses dtype {losses.dtype}"
    assert len(configs) == n_datasets
    assert len(ids) == n_datasets
    assert meta["weight_vectorization"]["D"] == codec.D
    assert meta["mlp_architecture"]["n_params"] == codec.D
    assert meta["collection"]["n_datasets"] == n_datasets
    assert meta["collection"]["n_mlp"] == n_mlp

    # Weights must be finite (no NaN/Inf from a corrupt merge).
    assert torch.isfinite(weights).all(), "weights contain non-finite entries"
    assert torch.isfinite(losses).all(), "losses contain non-finite entries"

    # Round-trip a weight vector through the codec.
    theta = weights[0, 0]
    model = codec.instantiate(theta)
    x = torch.randn(8, 32)
    with torch.no_grad():
        y = model(x)
    assert torch.isfinite(y).all(), "instantiated MLP produced non-finite output"
    assert y.shape == (8, 32)

    print(f"  format OK: weights {tuple(weights.shape)}, losses {tuple(losses.shape)}, "
          f"configs {len(configs)}, D={codec.D}")


def _check_no_leftovers(out_dir: str, n_shards: int) -> None:
    """Assert no shard / checkpoint files remain after a successful merge."""
    for r in range(n_shards):
        for fn in (f"_shard_{r}.pt", f"_progress_shard_{r}.pt"):
            p = os.path.join(out_dir, fn)
            assert not os.path.exists(p), f"leftover file not cleaned up: {fn}"
    assert not os.path.exists(os.path.join(out_dir, "_progress.pt")), \
        "leftover sequential checkpoint"


def test_shard_indices() -> None:
    """Unit test: ``_shard_indices`` partitions range(n) with no gaps/overlaps."""
    print("\n=== Unit test: _shard_indices ===")
    for n, k in [(0, 4), (1, 4), (4, 4), (7, 3), (10, 4), (100, 8)]:
        shards = _shard_indices(n, k)
        assert len(shards) == k, f"expected {k} shards, got {len(shards)}"
        union = sorted(i for s in shards for i in s)
        assert union == list(range(n)), \
            f"shards for n={n},k={k} do not cover range({n}): {union}"
        # No overlaps.
        assert len(union) == len(set(union)), f"overlaps for n={n}, k={k}"
        # Contiguous.
        for s in shards:
            assert s == list(range(s[0], s[0] + len(s))) if s else True, \
                f"shard not contiguous: {s}"
        sizes = [len(s) for s in shards]
        print(f"  n={n}, k={k}: sizes={sizes}  OK")


def test_resolve_gpus() -> None:
    """Unit test: ``resolve_gpus`` parses 'auto' and comma lists correctly."""
    print("\n=== Unit test: resolve_gpus ===")
    auto = resolve_gpus("auto")
    if torch.cuda.is_available():
        assert auto == list(range(torch.cuda.device_count())), \
            f"auto should return all {torch.cuda.device_count()} GPUs"
        print(f"  auto -> {auto}  OK")
    else:
        assert auto == [], "auto with no CUDA should return []"
        print(f"  auto -> [] (no CUDA)  OK")

    explicit = resolve_gpus("0,1,2,3")
    assert explicit == [0, 1, 2, 3], f"explicit parse failed: {explicit}"
    print(f"  '0,1,2,3' -> {explicit}  OK")

    dedup = resolve_gpus("1,1,0,0")
    assert dedup == [0, 1], f"dedup/sort failed: {dedup}"
    print(f"  '1,1,0,0' -> {dedup} (dedup+sort)  OK")

    empty = resolve_gpus("")
    assert empty == [], f"empty string should give []: {empty}"
    print(f"  '' -> []  OK")


def test_sequential_regression() -> str:
    """Run the sequential single-GPU path and check the output format."""
    print("\n=== Sequential single-GPU regression test ===")
    out_dir = tempfile.mkdtemp(prefix="targets_seq_")
    cfg = _make_config(out_dir, n_datasets=4, n_mlp=2, gpus=None)
    # Force CPU if no CUDA, else cuda:0.
    if not torch.cuda.is_available():
        cfg = CollectConfig(
            n_datasets=4, n_mlp=2, n_train=256, n_test=128, noise_std=0.1,
            corpus_seed=7, L=32, H=128, out_dir=out_dir, gpus=None,
            train=TrainConfig(lr=3e-3, steps=300, min_steps=100, patience=100,
                              L=32, H=128, device="cpu"),
        )
    collect_targets(cfg, show_progress=False)
    _check_output_format(out_dir, 4, 2)
    print(f"  out_dir: {out_dir}")
    return out_dir


def test_parallel_multi_gpu(gpu_ids: list) -> str:
    """Run the multi-GPU parallel path and check the merged output."""
    n_datasets = 4
    n_mlp = 2
    n_shards = len(gpu_ids)
    print(f"\n=== Multi-GPU parallel test ({n_shards} GPUs {gpu_ids}) ===")
    out_dir = tempfile.mkdtemp(prefix="targets_par_")
    cfg = _make_config(out_dir, n_datasets=n_datasets, n_mlp=n_mlp,
                       gpus=gpu_ids)
    collect_targets(cfg, show_progress=False)
    _check_output_format(out_dir, n_datasets, n_mlp)
    _check_no_leftovers(out_dir, n_shards)
    print(f"  out_dir: {out_dir}")
    return out_dir


def test_parallel_vs_sequential(gpu_ids: list) -> None:
    """Compare parallel and sequential outputs for the same config.

    Because the corpus is deterministic and ``mlp_seed_base`` depends only on
    the global dataset index, the parallel and sequential paths should produce
    (near-)identical weights and losses. We use ``allclose`` with a loose
    tolerance to tolerate minor GPU floating-point differences.
    """
    print(f"\n=== Parallel vs sequential equivalence ({len(gpu_ids)} GPUs) ===")
    n_datasets = 4
    n_mlp = 2

    # Sequential run on gpu_ids[0].
    seq_dir = tempfile.mkdtemp(prefix="targets_seq_eq_")
    seq_cfg = _make_config(seq_dir, n_datasets=n_datasets, n_mlp=n_mlp, gpus=None)
    # Pin the sequential run to the same first GPU for a fair comparison.
    seq_cfg = CollectConfig(
        n_datasets=n_datasets, n_mlp=n_mlp, n_train=256, n_test=128,
        noise_std=0.1, corpus_seed=7, L=32, H=128, out_dir=seq_dir, gpus=None,
        train=TrainConfig(lr=3e-3, steps=300, min_steps=100, patience=100,
                           L=32, H=128, device=f"cuda:{gpu_ids[0]}"),
    )
    collect_targets(seq_cfg, show_progress=False)

    # Parallel run across gpu_ids.
    par_dir = tempfile.mkdtemp(prefix="targets_par_eq_")
    par_cfg = _make_config(par_dir, n_datasets=n_datasets, n_mlp=n_mlp,
                           gpus=gpu_ids)
    collect_targets(par_cfg, show_progress=False)

    seq_w = torch.load(os.path.join(seq_dir, "weights.pt"))
    par_w = torch.load(os.path.join(par_dir, "weights.pt"))
    seq_l = torch.load(os.path.join(seq_dir, "losses.pt"))
    par_l = torch.load(os.path.join(par_dir, "losses.pt"))

    assert seq_w.shape == par_w.shape, \
        f"shape mismatch: seq {seq_w.shape} vs par {par_w.shape}"
    assert seq_l.shape == par_l.shape, \
        f"shape mismatch: seq {seq_l.shape} vs par {par_l.shape}"

    # Weights should match closely (same seeds -> same inits -> same convergence).
    w_close = torch.allclose(seq_w, par_w, atol=1e-4, rtol=1e-3)
    l_close = torch.allclose(seq_l, par_l, atol=1e-4, rtol=1e-3)
    max_w_diff = (seq_w - par_w).abs().max().item()
    max_l_diff = (seq_l - par_l).abs().max().item()
    print(f"  weights: max abs diff = {max_w_diff:.2e}  "
          f"-> {'MATCH' if w_close else 'DIFFER'}")
    print(f"  losses : max abs diff = {max_l_diff:.2e}  "
          f"-> {'MATCH' if l_close else 'DIFFER'}")
    assert w_close, f"weights differ: max abs diff {max_w_diff:.2e}"
    assert l_close, f"losses differ: max abs diff {max_l_diff:.2e}"

    # configs.json and dataset_ids.json must be identical (deterministic corpus).
    with open(os.path.join(seq_dir, "configs.json")) as f:
        seq_configs = json.load(f)
    with open(os.path.join(par_dir, "configs.json")) as f:
        par_configs = json.load(f)
    assert seq_configs == par_configs, "configs.json differs between seq and par"
    with open(os.path.join(seq_dir, "dataset_ids.json")) as f:
        seq_ids = json.load(f)
    with open(os.path.join(par_dir, "dataset_ids.json")) as f:
        par_ids = json.load(f)
    assert seq_ids == par_ids, "dataset_ids.json differs between seq and par"
    print("  configs.json + dataset_ids.json identical  OK")

    shutil.rmtree(seq_dir, ignore_errors=True)
    shutil.rmtree(par_dir, ignore_errors=True)


def test_parallel_resumability(gpu_ids: list) -> None:
    """Simulate an interrupted parallel run and verify it resumes + completes."""
    n_datasets = 4
    n_mlp = 2
    n_shards = len(gpu_ids)
    print(f"\n=== Parallel resumability test ({n_shards} GPUs) ===")
    out_dir = tempfile.mkdtemp(prefix="targets_resume_")

    # Write a fake partial shard checkpoint for shard 0: only dataset index 0
    # done. Shard 0 owns the first ceil(n_datasets/n_shards) indices.
    shards = _shard_indices(n_datasets, n_shards)
    shard0_indices = shards[0]
    codec = WeightCodec(L=32, H=128)
    fake_w = torch.randn(1, n_mlp, codec.D, dtype=torch.float32)
    fake_l = torch.randn(1, n_mlp, 3, dtype=torch.float32)
    torch.save(
        {"indices": [shard0_indices[0]], "weights": fake_w, "losses": fake_l},
        os.path.join(out_dir, f"_progress_shard_0.pt"),
    )
    print(f"  pre-wrote partial checkpoint for shard 0 "
          f"(index {shard0_indices[0]} done)")

    cfg = _make_config(out_dir, n_datasets=n_datasets, n_mlp=n_mlp,
                       gpus=gpu_ids)
    collect_targets(cfg, show_progress=False)
    _check_output_format(out_dir, n_datasets, n_mlp)
    _check_no_leftovers(out_dir, n_shards)

    # The resumed dataset (index shard0_indices[0]) should match the fake
    # checkpoint's weights (it was skipped, not retrained).
    weights = torch.load(os.path.join(out_dir, "weights.pt"))
    assert torch.allclose(weights[shard0_indices[0]], fake_w[0]), \
        "resumed dataset weights do not match the checkpoint (should be skipped)"
    print(f"  resumed index {shard0_indices[0]} preserved checkpoint weights  OK")
    shutil.rmtree(out_dir, ignore_errors=True)


def test_merge_shards_validation() -> None:
    """Unit test: ``_merge_shards`` detects missing/overlapping/incomplete shards."""
    print("\n=== Unit test: _merge_shards validation ===")
    tmp = tempfile.mkdtemp(prefix="merge_test_")
    n_datasets, n_mlp, D = 4, 2, 8352
    try:
        # Missing shard file.
        try:
            _merge_shards(tmp, 2, n_datasets, n_mlp, D)
            assert False, "should have raised (missing shard)"
        except RuntimeError as e:
            assert "missing shard" in str(e), str(e)
            print("  missing shard detected  OK")

        # Overlapping shards.
        torch.save({"indices": torch.tensor([0, 1], dtype=torch.long),
                    "weights": torch.zeros(2, n_mlp, D),
                    "losses": torch.zeros(2, n_mlp, 3)},
                   os.path.join(tmp, "_shard_0.pt"))
        torch.save({"indices": torch.tensor([1, 2], dtype=torch.long),
                    "weights": torch.zeros(2, n_mlp, D),
                    "losses": torch.zeros(2, n_mlp, 3)},
                   os.path.join(tmp, "_shard_1.pt"))
        try:
            _merge_shards(tmp, 2, n_datasets, n_mlp, D)
            assert False, "should have raised (overlap)"
        except RuntimeError as e:
            assert "overlap" in str(e), str(e)
            print("  overlapping shards detected  OK")

        # Complete coverage: shards 0=[0,1] and 1=[2,3] cover all 4.
        os.remove(os.path.join(tmp, "_shard_1.pt"))
        torch.save({"indices": torch.tensor([2, 3], dtype=torch.long),
                    "weights": torch.zeros(2, n_mlp, D),
                    "losses": torch.zeros(2, n_mlp, 3)},
                   os.path.join(tmp, "_shard_1.pt"))
        w, l = _merge_shards(tmp, 2, n_datasets, n_mlp, D)
        assert w.shape == (n_datasets, n_mlp, D)
        assert l.shape == (n_datasets, n_mlp, 3)
        print("  complete coverage merges  OK")

        # Incomplete coverage: both shard files present, but index 3 is left
        # uncovered (shard 1 covers only [2]).
        os.remove(os.path.join(tmp, "_shard_1.pt"))
        torch.save({"indices": torch.tensor([2], dtype=torch.long),
                    "weights": torch.zeros(1, n_mlp, D),
                    "losses": torch.zeros(1, n_mlp, 3)},
                   os.path.join(tmp, "_shard_1.pt"))
        try:
            _merge_shards(tmp, 2, n_datasets, n_mlp, D)
            assert False, "should have raised (incomplete)"
        except RuntimeError as e:
            assert "not covered" in str(e), str(e)
            print("  incomplete coverage detected  OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    """Run all parallel-collection smoke tests and assert the validation gates."""
    print("=" * 90)
    print("Multi-GPU parallel collection smoke test (Phase 2+)")
    print("=" * 90)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    gpu {i}: {torch.cuda.get_device_name(i)}")

    all_ok = True

    # 1. Unit tests (always run).
    try:
        test_shard_indices()
        test_resolve_gpus()
        test_merge_shards_validation()
        print("  unit tests PASS")
    except AssertionError as e:
        print(f"  unit tests FAIL: {e}")
        all_ok = False

    # 2. Sequential regression (always run; CPU fallback if no CUDA).
    try:
        seq_dir = test_sequential_regression()
        print("  sequential regression PASS")
        shutil.rmtree(seq_dir, ignore_errors=True)
    except AssertionError as e:
        print(f"  sequential regression FAIL: {e}")
        all_ok = False

    # 3. Multi-GPU tests (require >= 2 CUDA GPUs).
    gpu_ids = resolve_gpus("auto")
    if torch.cuda.is_available() and len(gpu_ids) >= 2:
        # Use the first 2 GPUs for the test (keeps it fast and focused).
        test_gpus = gpu_ids[:2]
        try:
            par_dir = test_parallel_multi_gpu(test_gpus)
            print("  multi-GPU parallel PASS")
            shutil.rmtree(par_dir, ignore_errors=True)
        except AssertionError as e:
            print(f"  multi-GPU parallel FAIL: {e}")
            all_ok = False

        try:
            test_parallel_vs_sequential(test_gpus)
            print("  parallel vs sequential equivalence PASS")
        except AssertionError as e:
            print(f"  parallel vs sequential equivalence FAIL: {e}")
            all_ok = False

        try:
            test_parallel_resumability(test_gpus)
            print("  parallel resumability PASS")
        except AssertionError as e:
            print(f"  parallel resumability FAIL: {e}")
            all_ok = False
    else:
        print("\n[SKIP] multi-GPU tests require >= 2 CUDA GPUs "
              f"(found {len(gpu_ids)}); single-GPU regression already passed.")

    print("\n" + "=" * 90)
    if all_ok:
        print("PARALLEL COLLECTION SMOKE TEST PASSED")
        return 0
    print("PARALLEL COLLECTION SMOKE TEST FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
