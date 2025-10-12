"""
Phase 0 Benchmark - Minimal validation of optimization claims.

**Design Notes**:
- Measures Python+C++ system speedup (not pure C++ LUT speedup)
- Uses seeded random data for reproducibility
- Includes correctness spot-checks
- See benchmarks/docs/DESIGN_REVIEW.md for full analysis

Usage:
    python benchmarks/bench_phase0.py

Expected runtime: ~30 seconds
"""

import sys
import os
import time
import numpy as np
import json
from datetime import datetime

# Add parent directory to path to find the compiled module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import ternary_core_simd_full as tc
except ImportError:
    print("ERROR: Module not compiled.")
    print("Run: python setup.py build_ext --inplace")
    exit(1)

# Import reference implementations
from reference import (
    ref_tadd, ref_tmul, ref_tmin, ref_tmax, ref_tnot,
    ref_op_array
)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def verify_correctness(tc_module):
    """
    Spot-check correctness of optimized operations.

    Critical: Ensure we're measuring fast correct answers, not fast wrong answers.
    """
    print("\n" + "="*60)
    print("Correctness Verification (Spot Check)")
    print("="*60)

    test_cases = [
        # (a, b, expected_tadd, expected_tmul)
        (0, 0, 0, 2),  # -1 + -1 = -1 (sat), -1 * -1 = +1
        (1, 1, 1, 1),  # 0 + 0 = 0, 0 * 0 = 0
        (2, 2, 2, 2),  # +1 + +1 = +1 (sat), +1 * +1 = +1
        (0, 2, 1, 0),  # -1 + 1 = 0, -1 * 1 = -1
    ]

    errors = []
    for a, b, exp_add, exp_mul in test_cases:
        A = np.array([a], dtype=np.uint8)
        B = np.array([b], dtype=np.uint8)

        result_add = tc_module.tadd(A, B)[0]
        result_mul = tc_module.tmul(A, B)[0]

        if result_add != exp_add:
            errors.append(f"tadd({a}, {b}) = {result_add}, expected {exp_add}")
        if result_mul != exp_mul:
            errors.append(f"tmul({a}, {b}) = {result_mul}, expected {exp_mul}")

    if errors:
        print("  ❌ CORRECTNESS ERRORS DETECTED:")
        for err in errors:
            print(f"     {err}")
        print("\n  ABORTING: Fix correctness issues before benchmarking!")
        exit(1)
    else:
        print("  ✅ All spot checks passed")


def benchmark_op(func, args, iterations=1000, warmup=100):
    """
    Simple benchmark with warmup.

    Args:
        func: Function to benchmark
        args: List of arguments
        iterations: Number of timed iterations
        warmup: Number of warmup iterations

    Returns:
        Median time in nanoseconds
    """
    # Warmup
    for _ in range(warmup):
        _ = func(*args)

    # Measure
    times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        _ = func(*args)
        end = time.perf_counter_ns()
        times.append(end - start)

    return np.median(times)


def run_microbenchmarks():
    """
    Benchmark 1: Scalar speedup validation

    Tests small arrays (8-31 elements) to force scalar code path.
    Measures Python+C++ system speedup, not pure C++ optimization.
    """
    print("\n" + "="*60)
    print("Benchmark 1: Microbenchmarks (Scalar Speedup)")
    print("="*60)
    print("Note: Measures system speedup (Python+C++), not pure C++ LUT speedup")
    print()

    sizes = [8, 16, 31]
    ops = [
        ('tadd', tc.tadd, lambda A, B: ref_op_array(ref_tadd, A, B)),
        ('tmul', tc.tmul, lambda A, B: ref_op_array(ref_tmul, A, B)),
        ('tmin', tc.tmin, lambda A, B: ref_op_array(ref_tmin, A, B)),
        ('tmax', tc.tmax, lambda A, B: ref_op_array(ref_tmax, A, B)),
        ('tnot', tc.tnot, lambda A: ref_op_array(ref_tnot, A)),
    ]

    results = []
    speedups = []

    for op_name, opt_func, ref_func in ops:
        for size in sizes:
            # Generate reproducible random data
            A = np.random.choice([0, 1, 2], size).astype(np.uint8)
            B = np.random.choice([0, 1, 2], size).astype(np.uint8) if op_name != 'tnot' else None

            args_opt = [A] if B is None else [A, B]
            args_ref = [A] if B is None else [A, B]

            # Benchmark with reduced iterations for speed
            time_opt = benchmark_op(opt_func, args_opt, iterations=1000, warmup=50)
            time_ref = benchmark_op(ref_func, args_ref, iterations=100, warmup=10)

            speedup = time_ref / time_opt
            speedups.append(speedup)

            print(f"  {op_name:4s} (n={size:2d}): {speedup:5.1f}x speedup")

            results.append({
                'operation': op_name,
                'size': size,
                'time_opt_ns': float(time_opt),
                'time_ref_ns': float(time_ref),
                'speedup': float(speedup)
            })

    median_speedup = np.median(speedups)
    print(f"\n  Median speedup: {median_speedup:.1f}x")
    print(f"  Target: 3-10x → {'✅ PASS' if 3 <= median_speedup <= 10 else '❌ FAIL'}")

    return results, median_speedup


def run_size_sweep():
    """
    Benchmark 2: Array size sweep

    Tests realistic array sizes (8 to 1M elements) to measure:
    - SIMD efficiency
    - Overall system speedup
    - Peak throughput
    """
    print("\n" + "="*60)
    print("Benchmark 2: Array Size Sweep (SIMD Efficiency)")
    print("="*60)

    sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144, 1048576]

    results = []
    speedups = []

    for size in sizes:
        # Generate reproducible random data
        A = np.random.choice([0, 1, 2], size).astype(np.uint8)
        B = np.random.choice([0, 1, 2], size).astype(np.uint8)

        # Adaptive iterations: more for small arrays, fewer for large
        iterations = max(50, min(1000, 10000 // size))  # Minimum 50 (increased from 10)

        time_opt = benchmark_op(tc.tadd, [A, B], iterations=iterations, warmup=min(iterations//10, 50))
        time_ref = benchmark_op(lambda A, B: ref_op_array(ref_tadd, A, B), [A, B], iterations=min(iterations, 50), warmup=5)

        throughput = size / (time_opt / 1e9)  # trits/sec
        speedup = time_ref / time_opt
        simd_pct = ((size // 32) * 32 / size) * 100

        speedups.append(speedup)

        print(f"  n={size:7d}: {throughput/1e6:7.1f} Mtrits/s, {speedup:5.2f}x, {simd_pct:5.1f}% SIMD")

        results.append({
            'size': size,
            'throughput': float(throughput),
            'speedup': float(speedup),
            'simd_pct': float(simd_pct)
        })

    # Geometric mean of speedups
    geomean_speedup = np.exp(np.mean(np.log(speedups)))
    max_throughput = max(r['throughput'] for r in results)

    print(f"\n  Geometric mean speedup: {geomean_speedup:.2f}x")
    print(f"  Peak throughput: {max_throughput/1e6:.1f} Mtrits/s")
    print(f"  Target overall: 1.3-1.5x → {'✅ PASS' if 1.3 <= geomean_speedup <= 2.0 else '❌ FAIL'}")
    print(f"  Target throughput: >30M → {'✅ PASS' if max_throughput > 30e6 else '❌ FAIL'}")

    return results, geomean_speedup, max_throughput


def main():
    print("="*60)
    print("  Phase 0 Benchmark Suite - Minimal Validation")
    print("="*60)
    print()
    print("Goal: Validate optimization claims")
    print("  - Scalar speedup: 3-10x")
    print("  - Overall improvement: 30-50% (1.3-1.5x)")
    print("  - Peak throughput: >30 Mtrits/s")
    print()
    print(f"Random seed: {RANDOM_SEED} (for reproducibility)")
    print("Estimated runtime: ~30 seconds")

    # Verify correctness before benchmarking
    verify_correctness(tc)

    start_time = time.time()

    # Run benchmarks
    micro_results, scalar_speedup = run_microbenchmarks()
    sweep_results, overall_speedup, peak_throughput = run_size_sweep()

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "="*60)
    print("  Phase 0 Validation Summary")
    print("="*60)

    scalar_pass = bool(3 <= scalar_speedup <= 10)
    overall_pass = bool(1.3 <= overall_speedup <= 2.0)
    throughput_pass = bool(peak_throughput > 30e6)

    print(f"  Scalar speedup:   {scalar_speedup:5.1f}x  {'✅' if scalar_pass else '❌'}")
    print(f"  Overall speedup:  {overall_speedup:5.2f}x  {'✅' if overall_pass else '❌'}")
    print(f"  Peak throughput:  {peak_throughput/1e6:5.1f} Mtrits/s  {'✅' if throughput_pass else '❌'}")

    all_pass = scalar_pass and overall_pass and throughput_pass

    print()
    if all_pass:
        print("  ✅ PHASE 0 VALIDATED - All targets met!")
    else:
        print("  ⚠️  PHASE 0 INCOMPLETE - Some targets not met")

    print(f"\n  Total runtime: {elapsed:.1f}s")
    print("="*60)

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'random_seed': RANDOM_SEED,
        'elapsed_sec': elapsed,
        'microbenchmarks': micro_results,
        'size_sweep': sweep_results,
        'summary': {
            'scalar_speedup': float(scalar_speedup),
            'overall_speedup': float(overall_speedup),
            'peak_throughput': float(peak_throughput),
            'validation': {
                'scalar_pass': scalar_pass,
                'overall_pass': overall_pass,
                'throughput_pass': throughput_pass,
                'all_pass': all_pass
            }
        }
    }

    # Ensure results directory exists (relative to script location)
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"phase0_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved: {output_file}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
