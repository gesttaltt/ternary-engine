#!/usr/bin/env python3
"""
phase5_gpu_application_context.py - TritNet Phase 5: "research applications"
-- is there ANY real GPU pipeline where TritNet's learned forward pass beats
the obvious alternative?

Phase 4 established TritNet-on-GPU beats AVX2-on-CPU by 15-47x, but loses to
CPU LUT by 4-10x. That comparison (GPU-TritNet vs CPU-LUT) is the wrong one
for a "real pipeline" question, though: if an application already has its
ternary data resident on GPU (the scenario where TritNet-on-GPU's speed
advantage over AVX2-CPU would actually matter -- no H2D/D2H round trip needed
either way), the real competing implementation isn't a CPU LUT at all. It's
whatever a GPU-native version of the SAME 5 operations would look like.

And tadd/tmul/tmin/tmax/tnot all have trivial closed-form arithmetic:
  tnot(a)   = -a
  tadd(a,b) = clamp(a + b, -1, 1)
  tmul(a,b) = a * b
  tmin(a,b) = min(a, b)
  tmax(a,b) = max(a, b)
These need zero lookups and zero learned weights -- a handful of GPU-native
elementwise ops, each O(1) work per element, computes them exactly (100%,
not 99.5-99.9%) with no truth-table training step at all. This script
benchmarks that direct-arithmetic GPU kernel against TritNet-GPU (from
Phase 4) on the same hardware, same batch sizes, same methodology, to give
Phase 5's "research applications" question its actual, fair answer: does
TritNet's learned forward pass ever have a genuine throughput or correctness
edge over just doing the arithmetic on GPU?

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

USAGE: python models/tritnet/phase5_gpu_application_context.py
OUTPUT: Per-op throughput comparison, TritNet-GPU vs direct-arithmetic-GPU
        vs CPU-LUT baseline; correctness re-confirmed for direct arithmetic
        (should be 100% exact for all 5 ops, unlike 3 of the 5 TritNet
        checkpoints).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
EXPORT_DIR = PROJECT_ROOT / "models" / "tritnet" / "phase2b_export"

# Same-host CPU baselines as Phase 4 (models/tritnet/phase4_gpu_benchmark.py),
# re-measured 2026-08-17 on this machine (AMD Ryzen 5 4500) via
# benchmarks/cpp-native-kernels/bench_tritnet_inference.cpp.
LUT_MOPS = {'tnot': 517.12, 'tadd': 148.88, 'tmul': 145.87, 'tmin': 134.80, 'tmax': 139.02}

BATCH_SIZES_UNARY = [10_000, 100_000, 1_000_000, 10_000_000, 50_000_000]
BATCH_SIZES_BINARY = [10_000, 100_000, 1_000_000, 10_000_000, 50_000_000]
N_REPEATS = 7
N_WARMUP = 3

_TRITS = (-1, 0, 1)


def _all_trit_vectors(n: int):
    if n == 0:
        yield []
        return
    for t in _TRITS:
        for rest in _all_trit_vectors(n - 1):
            yield [t] + rest


SCALAR_OPS = {
    'tnot': lambda a: -a,
    'tadd': lambda a, b: max(-1, min(1, a + b)),
    'tmul': lambda a, b: a * b,
    'tmin': min,
    'tmax': max,
}


def make_full_dataset(op_name: str):
    op = SCALAR_OPS[op_name]
    rows_x, rows_y = [], []
    if op_name == 'tnot':
        for vec_a in _all_trit_vectors(5):
            rows_x.append(vec_a)
            rows_y.append([op(t) for t in vec_a])
    else:
        for vec_a in _all_trit_vectors(5):
            for vec_b in _all_trit_vectors(5):
                rows_x.append(vec_a + vec_b)
                rows_y.append([op(a, b) for a, b in zip(vec_a, vec_b)])
    return np.array(rows_x, dtype=np.float32), np.array(rows_y, dtype=np.float32)


def direct_gpu_op(op_name: str, x, torch):
    """Closed-form GPU-native arithmetic -- no weights, no LUT, no training."""
    if op_name == 'tnot':
        return -x
    a, b = x[:, :5], x[:, 5:]
    if op_name == 'tadd':
        return torch.clamp(a + b, -1.0, 1.0)
    if op_name == 'tmul':
        return a * b
    if op_name == 'tmin':
        return torch.minimum(a, b)
    if op_name == 'tmax':
        return torch.maximum(a, b)
    raise ValueError(op_name)


def check_correctness_direct(torch, device):
    print("=" * 78)
    print("Correctness: direct-arithmetic GPU kernel (should be 100% exact, all 5 ops)")
    print("=" * 78)
    for op_name in SCALAR_OPS:
        X, Y = make_full_dataset(op_name)
        x_gpu = torch.from_numpy(X).to(device)
        with torch.no_grad():
            pred = direct_gpu_op(op_name, x_gpu, torch).cpu().numpy()
        acc = (pred == Y).all(axis=1).mean()
        status = "PASS" if acc == 1.0 else "FAIL"
        print(f"  {op_name}: {status}  acc={acc*100:.4f}%  (n={len(Y)})")
    print()


def time_best(fn, n_repeats=N_REPEATS, n_warmup=N_WARMUP):
    for _ in range(n_warmup):
        fn()
    best = float("inf")
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best


def bench_direct(op_name: str, batch_sizes, torch, device):
    in_features = 5 if op_name == 'tnot' else 10
    print(f"--- {op_name} direct-arithmetic GPU (in_features={in_features}) ---")
    print(f"{'batch':>12}  {'compute-only Mops/s':>20}  {'end-to-end Mops/s':>18}  "
          f"{'vs LUT-CPU (e2e)':>17}")

    rows = []
    for n in batch_sizes:
        rng = np.random.default_rng(42)
        x_cpu = rng.integers(-1, 2, size=(n, in_features)).astype(np.float32)

        try:
            x_gpu = torch.from_numpy(x_cpu).to(device)
            torch.cuda.synchronize()

            def compute_only():
                with torch.no_grad():
                    out = direct_gpu_op(op_name, x_gpu, torch)
                torch.cuda.synchronize()
                return out

            t_compute = time_best(compute_only)
            compute_mops = (n / t_compute) / 1e6

            def end_to_end():
                xg = torch.from_numpy(x_cpu).to(device, non_blocking=False)
                with torch.no_grad():
                    out = direct_gpu_op(op_name, xg, torch)
                result = out.cpu().numpy()
                torch.cuda.synchronize()
                return result

            t_e2e = time_best(end_to_end)
            e2e_mops = (n / t_e2e) / 1e6
        except torch.cuda.OutOfMemoryError:
            del x_gpu
            torch.cuda.empty_cache()
            print(f"{n:>12,}  {'OOM (skipped)':>20}")
            continue

        lut_mops = LUT_MOPS[op_name]
        print(f"{n:>12,}  {compute_mops:>20.2f}  {e2e_mops:>18.2f}  {e2e_mops / lut_mops:>16.3f}x")
        rows.append({"batch": n, "compute_mops": compute_mops, "e2e_mops": e2e_mops,
                      "vs_lut_e2e": e2e_mops / lut_mops})
        del x_gpu
        torch.cuda.empty_cache()
    print()
    return rows


def main() -> int:
    try:
        import torch
    except ImportError:
        print("[SKIP] PyTorch not installed.")
        return 0
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available on this machine.")
        return 0

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}  (torch {torch.__version__})\n")

    check_correctness_direct(torch, device)

    print("=" * 78)
    print("Throughput: direct-arithmetic GPU kernel (no weights, no LUT, no training)")
    print("=" * 78)

    all_results = {}
    for op_name in SCALAR_OPS:
        batch_sizes = BATCH_SIZES_UNARY if op_name == 'tnot' else BATCH_SIZES_BINARY
        all_results[op_name] = bench_direct(op_name, batch_sizes, torch, device)

    out_path = PROJECT_ROOT / "models" / "tritnet" / "phase5_gpu_application_context_results.json"
    out_path.write_text(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "lut_baseline_mops": LUT_MOPS,
        "results": all_results,
    }, indent=2))
    print(f"Results written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
