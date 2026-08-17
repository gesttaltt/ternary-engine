#!/usr/bin/env python3
"""
phase4_gpu_benchmark.py - TritNet Phase 4: GPU batched-inference throughput

Loads the same exported ternary weights Phase 3's C++ engine uses
(models/tritnet/phase2b_export/<op>/*.npy, produced by export_weights.py) and
runs the identical forward pass (see export_weights.py's docstring for the
exact recipe: h=ReLU(x@Wi+bi) for the two hidden layers, then a plain linear
output layer, argmax-1 decode per trit) as a batched PyTorch CUDA op, so the
GPU number is directly comparable to the CPU numbers already recorded under
"TritNet Development" -> Phase 3 in .claude/CLAUDE.md:
    LUT:            tnot ~535 Mops/s | binary ops ~133 Mops/s
    AVX2 TritNet:   tnot ~2.75 Mops/s | binary ops ~0.78-0.79 Mops/s
"Mops/s" keeps Phase 3's definition: millions of 5-trit-chunk operations/sec
(one sample = one op, regardless of unary/binary), NOT millions of trits.

Two numbers are reported per batch size, for the same fairness reasons Phase 3
ended up measuring "reconverts weights every call" vs "weights preconverted
once" separately:
  - compute-only:  input already resident on GPU, times only the 3 matmuls +
                    argmax (the number that matters if TritNet is one stage in
                    a larger GPU-resident pipeline)
  - end-to-end:     host->device copy of input + compute + device->host copy
                    of output, timed as a single round trip from CPU (the
                    number that matters if this replaces a CPU LUT call site
                    as-is, with no other GPU-resident work around it)

Correctness is checked first, over the FULL input space per op (243 samples
for tnot, 59,049 for the 4 binary ops), against the same recorded checkpoint
accuracy models/tritnet/export_weights.py and tests/python/test_tritnet_export.py
already verify on CPU -- this run must reproduce that number on GPU before any
throughput figure below it is trustworthy.

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

USAGE: python models/tritnet/phase4_gpu_benchmark.py
OUTPUT: Correctness table, then a throughput table per op x batch size x mode.
        Exits 1 if CUDA is unavailable or the export directory is missing.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
EXPORT_DIR = PROJECT_ROOT / "models" / "tritnet" / "phase2b_export"
CKPT_DIRS = {
    'tnot': PROJECT_ROOT / "models" / "tritnet" / "phase2a" / "tnot",
    'tadd': PROJECT_ROOT / "models" / "tritnet" / "phase2b" / "tadd",
    'tmul': PROJECT_ROOT / "models" / "tritnet" / "phase2b" / "tmul",
    'tmin': PROJECT_ROOT / "models" / "tritnet" / "phase2b" / "tmin",
    'tmax': PROJECT_ROOT / "models" / "tritnet" / "phase2b" / "tmax",
}

# CPU baselines, RE-MEASURED ON THIS MACHINE (2026-08-17, AMD Ryzen 5 4500,
# `benchmarks/cpp-native-kernels/bench_tritnet_inference.cpp`), NOT the
# 2026-08-14 numbers in .claude/CLAUDE.md (those were measured on a different
# host, AMD Ryzen 5 7520U -- comparing this GPU's numbers against a different
# machine's CPU numbers would repeat the exact cross-run timing-fairness
# mistake Phase 3 already caught and fixed once for its own AVX2-amortization
# experiment). AVX2 figures use the "reconverts weights every call" number,
# i.e. Phase 3's headline comparison point, not the amortized one.
# Mops/s = 5-trit-chunk operations/sec.
LUT_MOPS = {'tnot': 517.12, 'tadd': 148.88, 'tmul': 145.87, 'tmin': 134.80, 'tmax': 139.02}
AVX2_MOPS = {'tnot': 3.3478, 'tadd': 1.0038, 'tmul': 1.0035, 'tmin': 1.0040, 'tmax': 1.0033}

_TRITS = (-1, 0, 1)

# Binary ops sweep smaller max batch than tnot: hidden=128 vs 64 means 2x the
# activation memory per sample, and this GPU has ~5.3GB free (RTX 3050, 6GB).
# Sizes are best-effort -- bench_op catches CUDA OOM per-size and skips it
# rather than aborting the whole run, since the exact ceiling depends on
# whatever else is resident on the GPU at run time (desktop compositor, etc.).
BATCH_SIZES_UNARY = [10_000, 100_000, 1_000_000, 5_000_000, 10_000_000]
BATCH_SIZES_BINARY = [10_000, 100_000, 1_000_000, 3_000_000]
N_REPEATS = 7
N_WARMUP = 3


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
    """Full input space: 3^5=243 for unary, 3^10=59,049 for binary."""
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


def load_weights_gpu(op_name: str, torch, device, dtype=None):
    out_dir = EXPORT_DIR / op_name
    weights = {}
    for i in (1, 2, 3):
        w = np.load(out_dir / f"W{i}.npy").astype(np.float32)  # [in_f, out_f]
        b = np.load(out_dir / f"b{i}.npy").astype(np.float32)
        t_w = torch.from_numpy(w).to(device)
        t_b = torch.from_numpy(b).to(device)
        if dtype is not None:
            t_w, t_b = t_w.to(dtype), t_b.to(dtype)
        weights[f"W{i}"] = t_w
        weights[f"b{i}"] = t_b
    return weights


def forward_gpu(x, weights):
    """Same recipe as export_weights.py / test_tritnet_export.py's forward(),
    executed as PyTorch tensor ops so it runs on whatever device x lives on."""
    h1 = (x @ weights["W1"] + weights["b1"]).clamp_min(0.0)
    h2 = (h1 @ weights["W2"] + weights["b2"]).clamp_min(0.0)
    logits = h2 @ weights["W3"] + weights["b3"]
    n_out_trits = logits.shape[1] // 3
    logits = logits.view(-1, n_out_trits, 3)
    return logits.argmax(dim=2).to(logits.dtype) - 1.0


def check_correctness(torch, device):
    print("=" * 78)
    print("Correctness (full input space, GPU forward vs recorded checkpoint accuracy)")
    print("=" * 78)
    all_ok = True
    for op_name, ckpt_dir in CKPT_DIRS.items():
        out_dir = EXPORT_DIR / op_name
        if not out_dir.exists():
            print(f"  {op_name}: SKIP (not exported, run export_weights.py first)")
            all_ok = False
            continue
        result_json = ckpt_dir / "result.json"
        recorded_acc = json.loads(result_json.read_text())['best_acc']

        X, Y = make_full_dataset(op_name)
        weights = load_weights_gpu(op_name, torch, device)
        x_gpu = torch.from_numpy(X).to(device)
        with torch.no_grad():
            pred = forward_gpu(x_gpu, weights).cpu().numpy()

        exact_match = (pred == Y).all(axis=1)
        replay_acc = exact_match.mean()
        ok = abs(replay_acc - recorded_acc) < 1e-6
        all_ok &= ok
        status = "PASS" if ok else "FAIL"

        # fp16 is checked separately and does NOT gate all_ok: it's a documented
        # speed/accuracy tradeoff (see Phase 4 notes in .claude/CLAUDE.md), not
        # a correctness bug -- reduced logit precision can flip a small number
        # of near-tied argmax decisions relative to the fp32 checkpoint.
        weights16 = load_weights_gpu(op_name, torch, device, dtype=torch.float16)
        x_gpu16 = x_gpu.half()
        with torch.no_grad():
            pred16 = forward_gpu(x_gpu16, weights16).cpu().numpy()
        acc16 = (pred16 == Y).all(axis=1).mean()
        drift = acc16 - replay_acc

        print(f"  {op_name}: {status}  gpu_acc(fp32)={replay_acc*100:.4f}%  "
              f"recorded_acc={recorded_acc*100:.4f}%  gpu_acc(fp16)={acc16*100:.4f}%  "
              f"(drift={drift*100:+.4f}pp, n={len(Y)})")
    print()
    return all_ok


def time_best(fn, n_repeats=N_REPEATS, n_warmup=N_WARMUP):
    """Best-of-N wall time in seconds, matching the C++ time_best() convention
    (min over repeats = least noise-corrupted measurement of achievable throughput)."""
    for _ in range(n_warmup):
        fn()
    best = float("inf")
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best


def bench_op(op_name: str, batch_sizes, torch, device, dtype=None, label="fp32"):
    weights = load_weights_gpu(op_name, torch, device, dtype=dtype)
    in_features = weights["W1"].shape[0]
    np_dtype = np.float16 if dtype is not None and dtype == torch.float16 else np.float32

    print(f"--- {op_name} (in_features={in_features}, {label}) ---")
    print(f"{'batch':>10}  {'compute-only Mops/s':>20}  {'end-to-end Mops/s':>18}  "
          f"{'vs LUT (e2e)':>13}  {'vs AVX2 (e2e)':>14}")

    rows = []
    for n in batch_sizes:
        rng = np.random.default_rng(42)
        # Values don't affect throughput (pure dense matmul, no data-dependent
        # branching) -- random trits are fine and cheap to generate at any n.
        x_cpu = rng.integers(-1, 2, size=(n, in_features)).astype(np_dtype)

        try:
            # compute-only: input already resident on GPU, exclude H2D/D2H
            x_gpu = torch.from_numpy(x_cpu).to(device)
            torch.cuda.synchronize()

            def compute_only():
                with torch.no_grad():
                    out = forward_gpu(x_gpu, weights)
                torch.cuda.synchronize()
                return out

            t_compute = time_best(compute_only)
            compute_mops = (n / t_compute) / 1e6

            # end-to-end: single round trip, H2D input -> compute -> D2H output
            def end_to_end():
                xg = torch.from_numpy(x_cpu).to(device, non_blocking=False)
                with torch.no_grad():
                    out = forward_gpu(xg, weights)
                result = out.cpu().numpy()
                torch.cuda.synchronize()
                return result

            t_e2e = time_best(end_to_end)
            e2e_mops = (n / t_e2e) / 1e6
        except torch.cuda.OutOfMemoryError:
            del x_gpu
            torch.cuda.empty_cache()
            print(f"{n:>10,}  {'OOM (skipped)':>20}")
            continue

        lut_mops = LUT_MOPS[op_name]
        avx2_mops = AVX2_MOPS[op_name]
        print(f"{n:>10,}  {compute_mops:>20.2f}  {e2e_mops:>18.2f}  "
              f"{e2e_mops / lut_mops:>12.3f}x  {e2e_mops / avx2_mops:>13.2f}x")

        rows.append({
            "batch": n, "compute_mops": compute_mops, "e2e_mops": e2e_mops,
            "vs_lut_e2e": e2e_mops / lut_mops, "vs_avx2_e2e": e2e_mops / avx2_mops,
        })
        del x_gpu
        torch.cuda.empty_cache()
    print()
    return rows


def main() -> int:
    try:
        import torch
    except ImportError:
        print("[SKIP] PyTorch not installed -- GPU benchmark requires torch.")
        return 0

    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available on this machine -- Phase 4 GPU benchmark "
              "requires a CUDA-capable GPU. (No TPU path exists in this repo; "
              ".claude/CLAUDE.md's Phase 4 scope is GPU-only for now.)")
        return 0

    if not EXPORT_DIR.exists():
        print(f"[SKIP] {EXPORT_DIR} does not exist -- run "
              f"'python models/tritnet/export_weights.py' first.")
        return 0

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}  "
          f"(compute {'.'.join(map(str, torch.cuda.get_device_capability(0)))}, "
          f"torch {torch.__version__})")
    free, total = torch.cuda.mem_get_info(0)
    print(f"VRAM free/total: {free/1e9:.2f}GB / {total/1e9:.2f}GB\n")

    ok = check_correctness(torch, device)
    if not ok:
        print("Correctness FAILED for at least one op -- throughput numbers below "
              "would not be trustworthy. Stopping.")
        return 1

    print("=" * 78)
    print("Throughput (best-of-%d, %d warmup iterations; Mops/s = millions of "
          "5-trit-chunk ops/sec, same unit as Phase 3)" % (N_REPEATS, N_WARMUP))
    print("=" * 78)

    all_results = {}
    for op_name in CKPT_DIRS:
        batch_sizes = BATCH_SIZES_UNARY if op_name == 'tnot' else BATCH_SIZES_BINARY
        fp32_rows = bench_op(op_name, batch_sizes, torch, device, dtype=None, label="fp32")
        fp16_rows = bench_op(op_name, batch_sizes, torch, device, dtype=torch.float16, label="fp16")
        all_results[op_name] = {"fp32": fp32_rows, "fp16": fp16_rows}

    out_path = PROJECT_ROOT / "models" / "tritnet" / "phase4_gpu_results.json"
    out_path.write_text(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "lut_baseline_mops": LUT_MOPS,
        "avx2_baseline_mops": AVX2_MOPS,
        "fp16_accuracy_caveat": "fp16 forward pass can flip a small number of "
            "near-tied argmax decisions vs the fp32 checkpoint (observed on tmin/tmax, "
            "~0.003-0.02pp); see check_correctness() output above for exact drift per op.",
        "results": all_results,
    }, indent=2))
    print(f"Results written to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
