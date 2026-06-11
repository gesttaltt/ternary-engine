"""
bench_zero_skip_gemm.py - Zero-Skip Ternary GEMM Benchmark

Validates and quantifies the zero-skip optimization from research/FINDINGS.md.

Core finding: ~33% of ternary weights are exactly zero. In the inner product
sum_k(A[i,k] * B[k,j]), any term where B[k,j] = 0 contributes nothing and
can be skipped. The 3-adic valuation distribution predicts this fraction
precisely: P(value = 0) = 1/3 for uniform random ternary values.

Three implementations are compared:
  1. dense_blas      - A @ B_float (NumPy BLAS, processes all K terms)
  2. sign_split      - A @ B_pos - A @ B_neg (two binary matmuls, correct
                       decomposition but BLAS still touches all elements)
  3. zero_skip_cols  - Per-column: A[:, nz] @ B[nz, j] (true zero skip,
                       Python-level loop — shows the cost model, not speed)

Key outputs:
  - Actual zero fraction vs theoretical 1/3 (validates FINDINGS.md)
  - Effective ops per output element for each approach
  - Timing comparison across matrix sizes
  - Break-even analysis: at what K does zero-skip cover its overhead?

USAGE: python3 benchmarks/bench_zero_skip_gemm.py
       python3 benchmarks/bench_zero_skip_gemm.py --sizes 256 512 1024
       python3 benchmarks/bench_zero_skip_gemm.py --sparsity 0.1 0.33 0.5 0.9

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0
"""

import sys
import time
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import ternary_zero_skip_gemm as _zs
    HAS_ZS = True
    HAS_AVX2 = _zs.has_avx2
except ImportError:
    HAS_ZS = False
    HAS_AVX2 = False

# ---------------------------------------------------------------------------
# GEMM implementations
# ---------------------------------------------------------------------------

def matmul_dense_blas(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Baseline: convert B to float, call BLAS. Processes all K terms."""
    return A @ B.astype(np.float32)


def matmul_sign_split(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Decompose B into B_pos (where B=+1) and B_neg (where B=-1).
    C = A @ B_pos - A @ B_neg

    Two BLAS calls on binary {0,1} matrices. Same total element count as
    dense, but each sub-matrix is ~33% dense. BLAS still processes all
    elements — the speedup here is theoretical for custom sparse/hardware
    kernels, not NumPy. Included as the canonical correct decomposition.
    """
    B_pos = (B > 0).astype(np.float32)   # 1 where B=+1, else 0
    B_neg = (B < 0).astype(np.float32)   # 1 where B=-1, else 0
    return A @ B_pos - A @ B_neg


def matmul_zero_skip_cols(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    True zero-skip: for each output column j, only accumulate over rows k
    where B[k, j] != 0.

    C[:, j] = A[:, nz] @ B[nz, j]   where nz = nonzero(B[:, j])

    This eliminates ~33% of multiply-accumulates. The Python loop over N
    columns makes this slower than BLAS for large matrices — it illustrates
    the cost model and correctness of the skip, not a Python-optimized path.
    A C++ kernel would amortize the loop overhead.
    """
    M = A.shape[0]
    N = B.shape[1]
    C = np.zeros((M, N), dtype=np.float32)
    for j in range(N):
        nz = np.flatnonzero(B[:, j])
        if nz.size > 0:
            C[:, j] = A[:, nz] @ B[nz, j].astype(np.float32)
    return C


def matmul_zero_skip_cpp_scalar(A: np.ndarray, weights) -> np.ndarray:
    """C++ zero-skip kernel, scalar path (precomputed CSC, no AVX2)."""
    return weights.gemm(A, use_avx2=False)


def matmul_zero_skip_cpp_avx2(A: np.ndarray, weights) -> np.ndarray:
    """C++ zero-skip kernel, AVX2 path (precomputed CSC + gather + FMA)."""
    return weights.gemm(A, use_avx2=True)


# ---------------------------------------------------------------------------
# Sparsity analysis
# ---------------------------------------------------------------------------

def measure_sparsity(B: np.ndarray) -> dict:
    """Measure actual zero fraction and valuation distribution."""
    total = B.size
    n_zero = int(np.sum(B == 0))
    n_pos  = int(np.sum(B >  0))
    n_neg  = int(np.sum(B <  0))

    # Effective ops per output element: only non-zero B entries contribute
    mean_nonzero_per_col = float(np.mean(np.count_nonzero(B, axis=0)))
    K = B.shape[0]

    return {
        "zero_fraction":           n_zero / total,
        "pos_fraction":            n_pos  / total,
        "neg_fraction":            n_neg  / total,
        "theoretical_zero":        1 / 3,
        "mean_nonzero_per_col":    mean_nonzero_per_col,
        "K":                       K,
        "ops_per_output_dense":    K,
        "ops_per_output_zeroskip": mean_nonzero_per_col,
        "ops_reduction_pct":       (1.0 - mean_nonzero_per_col / K) * 100,
    }


def generate_ternary_matrix(rows: int, cols: int, zero_fraction: float,
                             rng: np.random.Generator) -> np.ndarray:
    """
    Generate a ternary matrix with a given zero fraction.
    Remaining values are split evenly between -1 and +1.
    """
    vals = rng.choice(
        np.array([-1, 0, 1], dtype=np.int8),
        size=(rows, cols),
        p=[
            (1 - zero_fraction) / 2,
            zero_fraction,
            (1 - zero_fraction) / 2,
        ],
    )
    return vals


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------

def time_fn(fn, *args, warmup: int = 5, repeats: int = 20) -> dict:
    for _ in range(warmup):
        fn(*args)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    arr = np.array(times) * 1e3  # ms
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms":   float(np.mean(arr)),
        "std_ms":    float(np.std(arr)),
        "min_ms":    float(np.min(arr)),
    }


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def verify_correctness(M: int, K: int, N: int, rng: np.random.Generator) -> bool:
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = generate_ternary_matrix(K, N, zero_fraction=1/3, rng=rng)

    ref    = matmul_dense_blas(A, B)
    sign   = matmul_sign_split(A, B)
    py_skip= matmul_zero_skip_cols(A, B)

    ok = (bool(np.allclose(ref, sign,    atol=1e-4)) and
          bool(np.allclose(ref, py_skip, atol=1e-4)))

    if HAS_ZS:
        import ternary_zero_skip_gemm as zs
        W = zs.ZeroSkipWeights(B)
        cpp_scalar = W.gemm(A, use_avx2=False)
        cpp_avx2   = W.gemm(A, use_avx2=True)
        ok = (ok and
              bool(np.allclose(ref, cpp_scalar, atol=1e-4)) and
              bool(np.allclose(ref, cpp_avx2,   atol=1e-4)))

    return ok


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_size(M: int, K: int, N: int, zero_fraction: float,
             rng: np.random.Generator, skip_col_threshold: int = 256) -> dict:
    """Run all implementations for one (M, K, N) configuration."""
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = generate_ternary_matrix(K, N, zero_fraction=zero_fraction, rng=rng)

    sp = measure_sparsity(B)

    t_dense = time_fn(matmul_dense_blas, A, B)
    t_sign  = time_fn(matmul_sign_split, A, B)

    # Python zero-skip: slow for large N due to loop overhead
    if N <= skip_col_threshold:
        t_py_skip = time_fn(matmul_zero_skip_cols, A, B, warmup=2, repeats=5)
    else:
        t_py_skip = None

    # C++ zero-skip kernels
    t_cpp_scalar = t_cpp_avx2 = t_precompute = None
    if HAS_ZS:
        import ternary_zero_skip_gemm as zs

        # Time the precompute separately
        t_pre = time_fn(zs.ZeroSkipWeights, B, warmup=3, repeats=10)
        t_precompute = t_pre

        # Build weights once, then time the GEMM
        weights = zs.ZeroSkipWeights(B)
        t_cpp_scalar = time_fn(matmul_zero_skip_cpp_scalar, A, weights)
        t_cpp_avx2   = time_fn(matmul_zero_skip_cpp_avx2,   A, weights)

    ops = M * N * K
    def gops(t_ms):
        return ops / (t_ms * 1e6)

    return {
        "M": M, "K": K, "N": N,
        "zero_fraction":           sp["zero_fraction"],
        "theoretical_zero":        sp["theoretical_zero"],
        "ops_per_output_dense":    sp["ops_per_output_dense"],
        "ops_per_output_zeroskip": sp["ops_per_output_zeroskip"],
        "ops_reduction_pct":       sp["ops_reduction_pct"],
        "dense":        {**t_dense,       "gops_s": gops(t_dense["median_ms"])},
        "sign_split":   {**t_sign,        "gops_s": gops(t_sign["median_ms"])},
        "py_zero_skip": (
            {**t_py_skip, "gops_s": gops(t_py_skip["median_ms"])} if t_py_skip else None
        ),
        "cpp_precompute": t_precompute,
        "cpp_scalar": (
            {**t_cpp_scalar, "gops_s": gops(t_cpp_scalar["median_ms"])} if t_cpp_scalar else None
        ),
        "cpp_avx2": (
            {**t_cpp_avx2, "gops_s": gops(t_cpp_avx2["median_ms"])} if t_cpp_avx2 else None
        ),
    }


def print_size_result(r: dict) -> None:
    M, K, N = r["M"], r["K"], r["N"]
    print(f"\n  {M}×{K}×{N}  (M×K×N,  {M*K*N:,} ops)")
    print(f"    Sparsity:  actual={r['zero_fraction']:.3f}  theoretical={r['theoretical_zero']:.3f}")
    print(f"    Zero-skip eliminates {r['ops_reduction_pct']:.1f}% of multiply-accumulates")
    if r["cpp_precompute"]:
        print(f"    CSC precompute: {r['cpp_precompute']['median_ms']:.3f} ms  (one-time cost per weight matrix)")
    print()

    d      = r["dense"]
    s      = r["sign_split"]
    py_sk  = r["py_zero_skip"]
    cs     = r["cpp_scalar"]
    ca     = r["cpp_avx2"]

    def row(label, t, ref_ms):
        if t is None:
            return f"    {label:<28}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}"
        ratio = ref_ms / t["median_ms"]
        return (f"    {label:<28}  {t['median_ms']:>10.3f}  {t['gops_s']:>10.2f}  "
                f"  {ratio:>8.2f}×")

    print(f"    {'Method':<28}  {'Median ms':>10}  {'Gops/s':>10}  {'vs dense':>10}")
    print(f"    {'-'*28}  {'-'*10}  {'-'*10}  {'-'*10}")
    print(f"    {'dense BLAS (NumPy)':<28}  {d['median_ms']:>10.3f}  {d['gops_s']:>10.2f}  {'(baseline)':>10}")
    print(row("sign split (Py)", s, d["median_ms"]))
    if py_sk:
        print(row("zero-skip cols (Py loop)", py_sk, d["median_ms"]))
    print(row("zero-skip C++ scalar", cs, d["median_ms"]))
    print(row("zero-skip C++ AVX2", ca, d["median_ms"]))


def run_sparsity_sweep(K: int, N: int, zero_fractions: list[float],
                       rng: np.random.Generator) -> None:
    """Show how ops-reduction scales with zero fraction, for a fixed K×N."""
    M = 128
    print(f"\n  Sparsity sweep  (M={M}, K={K}, N={N})")
    print(f"  {'Zero %':>8}  {'Actual %':>10}  {'Ops saved':>12}  "
          f"{'Dense ms':>10}  {'SignSplit ms':>13}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*13}")

    for zf in zero_fractions:
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = generate_ternary_matrix(K, N, zero_fraction=zf, rng=rng)
        sp = measure_sparsity(B)
        td = time_fn(matmul_dense_blas, A, B, warmup=5, repeats=10)
        ts = time_fn(matmul_sign_split, A, B, warmup=5, repeats=10)
        print(f"  {zf*100:>7.0f}%  {sp['zero_fraction']*100:>9.1f}%  "
              f"  {sp['ops_reduction_pct']:>10.1f}%  "
              f"  {td['median_ms']:>9.3f}  {ts['median_ms']:>12.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Zero-skip ternary GEMM benchmark (validates research/FINDINGS.md)"
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[128, 256, 512, 1024],
        help="K dimensions to benchmark (square: M=N=K/4, or pass K directly)",
    )
    parser.add_argument(
        "--sparsity", nargs="+", type=float,
        default=[0.1, 0.2, 0.33, 0.5, 0.7, 0.9],
        help="Zero fractions for sparsity sweep",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results JSON to this path",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("=" * 70)
    print("  Zero-Skip Ternary GEMM Benchmark")
    print("  Validates finding: research/FINDINGS.md § p-adic / 3-adic Structure")
    print("=" * 70)

    # ---- Correctness check --------------------------------------------------
    print("\nCorrectness check (M=32, K=60, N=40) ...", end=" ")
    ok = verify_correctness(32, 60, 40, np.random.default_rng(args.seed + 1))
    if not ok:
        print("FAIL — aborting.")
        return 1
    print("PASS")

    # ---- Per-size benchmarks -------------------------------------------------
    print("\n" + "=" * 70)
    print("  Matrix benchmarks  (zero_fraction = 1/3,  uniform random ternary)")
    print("=" * 70)

    all_results = []
    for K in args.sizes:
        M = max(16, K // 4)
        N = max(16, K // 4)
        r = run_size(M, K, N, zero_fraction=1/3, rng=rng)
        print_size_result(r)
        all_results.append(r)

    # ---- Sparsity sweep ------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Sparsity sweep  (how ops-savings scale with zero fraction)")
    print("=" * 70)
    run_sparsity_sweep(K=512, N=128, zero_fractions=args.sparsity, rng=rng)

    # ---- Summary -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)

    avg_zero   = float(np.mean([r["zero_fraction"] for r in all_results]))
    avg_saving = float(np.mean([r["ops_reduction_pct"] for r in all_results]))

    sign_ratios = [r["dense"]["median_ms"] / r["sign_split"]["median_ms"]
                   for r in all_results]
    avg_sign_ratio = float(np.mean(sign_ratios))

    print(f"\n  Average measured zero fraction:  {avg_zero:.3f}  (theoretical 0.333)")
    print(f"  Average ops saved (zero-skip):   {avg_saving:.1f}%")
    print(f"  Sign-split vs dense (NumPy):     {avg_sign_ratio:.2f}×")

    if HAS_ZS:
        scalar_ratios = [r["dense"]["median_ms"] / r["cpp_scalar"]["median_ms"]
                         for r in all_results if r["cpp_scalar"]]
        avx2_ratios   = [r["dense"]["median_ms"] / r["cpp_avx2"]["median_ms"]
                         for r in all_results if r["cpp_avx2"]]
        if scalar_ratios:
            print(f"  C++ scalar vs dense (avg):       {float(np.mean(scalar_ratios)):.2f}×")
        if avx2_ratios:
            print(f"  C++ AVX2   vs dense (avg):       {float(np.mean(avx2_ratios)):.2f}×")

    print()
    print("  Key takeaway:")
    print("    Zero-skip eliminates ~33% of multiply-accumulates (confirmed).")
    if HAS_ZS and avx2_ratios:
        avg_avx2 = float(np.mean(avx2_ratios))
        if avg_avx2 >= 1.0:
            print(f"    C++ AVX2 zero-skip beats NumPy BLAS by {avg_avx2:.2f}× on average.")
        else:
            print(f"    C++ AVX2 zero-skip reaches {avg_avx2:.2f}× of NumPy BLAS.")
            print("    BLAS is memory-bandwidth-bound here; savings show at larger K.")
    print("    Sign-split (Python/NumPy) can't exploit sparsity — BLAS touches all zeros.")
    print("    The C++ CSC kernel skips them at the instruction level.")

    # ---- Save results --------------------------------------------------------
    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = PROJECT_ROOT / "benchmarks" / "results" / f"zero_skip_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp":    datetime.now().isoformat(),
        "seed":         args.seed,
        "sizes":        args.sizes,
        "sparsity":     args.sparsity,
        "results":      all_results,
        "summary": {
            "avg_zero_fraction":   avg_zero,
            "avg_ops_saving_pct":  avg_saving,
            "avg_sign_split_ratio": avg_sign_ratio,
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
