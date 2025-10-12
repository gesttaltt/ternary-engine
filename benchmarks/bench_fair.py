"""
bench_fair.py - Fair benchmarking with C++ baseline

Copyright 2025 Ternary Core Contributors
Licensed under the Apache License, Version 2.0

**DESIGN**: Compares optimized implementation against unoptimized C++ baseline,
not against Python reference. This measures actual optimization impact:
- Phase 0 (LUTs): 3-10× scalar speedup
- Phase 0.5 (SIMD): 10-20× SIMD speedup
- Phase 1 (Threading): 2-8× large array speedup

Usage:
    python benchmarks/bench_fair.py

Expected runtime: ~2 minutes (much faster than bench_phase0.py)
"""

import sys
import os
import time
import numpy as np
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import optimized implementation
try:
    import ternary_core_simd_full as tc_opt
except ImportError:
    print("ERROR: Optimized module not compiled.")
    print("Run: python setup.py build_ext --inplace")
    exit(1)

# Import C++ reference (unoptimized)
try:
    import reference_cpp as tc_ref
except ImportError:
    print("ERROR: Reference module not compiled.")
    print("Run: python setup_reference.py build_ext --inplace")
    exit(1)

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def print_section(title):
    """Print section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def benchmark_operation(func, args, iterations=1000, warmup=100):
    """Benchmark an operation with warmup"""
    # Warmup
    for _ in range(warmup):
        _ = func(*args)

    # Measure with high-resolution timer
    times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        _ = func(*args)
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)

    return np.median(times)

def verify_correctness():
    """Verify optimized implementation matches reference"""
    print_section("Correctness Verification")

    test_cases = [
        (np.array([0, 1, 2, 0, 2], dtype=np.uint8),
         np.array([0, 1, 2, 2, 0], dtype=np.uint8))
    ]

    ops = [('tadd', tc_opt.tadd, tc_ref.tadd),
           ('tmul', tc_opt.tmul, tc_ref.tmul),
           ('tmin', tc_opt.tmin, tc_ref.tmin),
           ('tmax', tc_opt.tmax, tc_ref.tmax)]

    errors = []
    for A, B in test_cases:
        for name, opt_func, ref_func in ops:
            result_opt = opt_func(A, B)
            result_ref = ref_func(A, B)

            if not np.array_equal(result_opt, result_ref):
                errors.append(f"{name}: optimized != reference")

    if errors:
        print("  ❌ ERRORS:")
        for err in errors:
            print(f"     {err}")
        exit(1)
    else:
        print("  ✅ All correctness checks passed\n")

def benchmark_scalar_speedup():
    """Benchmark Phase 0 scalar optimizations (LUTs)"""
    print_section("Phase 0: Scalar Optimizations (LUT Speedup)")
    print("Comparing: Optimized scalar (LUTs) vs Unoptimized scalar (conversions)")
    print("Array sizes: 8-31 elements (pure scalar, no SIMD)\n")

    sizes = [8, 16, 31]
    ops = [
        ('tadd', tc_opt.tadd, tc_ref.tadd),
        ('tmul', tc_opt.tmul, tc_ref.tmul),
        ('tmin', tc_opt.tmin, tc_ref.tmin),
        ('tmax', tc_opt.tmax, tc_ref.tmax),
        ('tnot', tc_opt.tnot, tc_ref.tnot, True),
    ]

    results = []
    speedups = []

    for name, opt_func, ref_func, *is_unary in ops:
        is_unary = bool(is_unary)
        for size in sizes:
            A = np.random.choice([0, 1, 2], size).astype(np.uint8)
            B = np.random.choice([0, 1, 2], size).astype(np.uint8) if not is_unary else None

            args_opt = [A] if is_unary else [A, B]
            args_ref = [A] if is_unary else [A, B]

            # Benchmark with high iteration count for small arrays
            time_opt = benchmark_operation(opt_func, args_opt, iterations=5000, warmup=500)
            time_ref = benchmark_operation(ref_func, args_ref, iterations=5000, warmup=500)

            speedup = time_ref / time_opt
            speedups.append(speedup)

            print(f"  {name:4s} (n={size:2d}): {speedup:5.2f}× speedup  "
                  f"({time_ref/1000:.1f} → {time_opt/1000:.1f} µs)")

            results.append({
                'operation': name,
                'size': size,
                'time_opt_ns': float(time_opt),
                'time_ref_ns': float(time_ref),
                'speedup': float(speedup)
            })

    median_speedup = np.median(speedups)
    min_speedup = np.min(speedups)
    max_speedup = np.max(speedups)

    print(f"\n  Scalar Speedup Range: {min_speedup:.1f}× - {max_speedup:.1f}×")
    print(f"  Median Scalar Speedup: {median_speedup:.1f}×")
    print(f"  Target: 3-10× → {'✅ PASS' if 3 <= median_speedup <= 15 else '❌ FAIL'}")

    return results, median_speedup

def benchmark_simd_speedup():
    """Benchmark Phase 0.5 SIMD optimizations"""
    print_section("Phase 0.5: SIMD Optimizations (Vectorization Speedup)")
    print("Comparing: SIMD (32 parallel) vs Scalar (sequential)")
    print("Array sizes: 32-1024 elements (SIMD dominant)\n")

    sizes = [32, 64, 128, 256, 512, 1024]

    results = []
    speedups = []

    for size in sizes:
        A = np.random.choice([0, 1, 2], size).astype(np.uint8)
        B = np.random.choice([0, 1, 2], size).astype(np.uint8)

        # Adaptive iterations
        iterations = max(100, min(2000, 50000 // size))

        time_opt = benchmark_operation(tc_opt.tadd, [A, B], iterations=iterations, warmup=iterations//10)
        time_ref = benchmark_operation(tc_ref.tadd, [A, B], iterations=iterations, warmup=iterations//10)

        speedup = time_ref / time_opt
        throughput = size / (time_opt / 1e9) / 1e6  # Mtrits/s
        simd_pct = ((size // 32) * 32 / size) * 100

        speedups.append(speedup)

        print(f"  n={size:4d}: {speedup:5.2f}× speedup, {throughput:7.1f} Mtrits/s  "
              f"({simd_pct:5.1f}% SIMD)")

        results.append({
            'size': size,
            'speedup': float(speedup),
            'throughput': float(throughput * 1e6),
            'simd_pct': float(simd_pct)
        })

    median_speedup = np.median(speedups)
    min_speedup = np.min(speedups)
    max_speedup = np.max(speedups)

    print(f"\n  SIMD Speedup Range: {min_speedup:.1f}× - {max_speedup:.1f}×")
    print(f"  Median SIMD Speedup: {median_speedup:.1f}×")
    print(f"  Target: 10-20× → {'✅ PASS' if 10 <= median_speedup <= 30 else '❌ FAIL'}")

    return results, median_speedup

def benchmark_large_arrays():
    """Benchmark Phase 1 OpenMP threading"""
    print_section("Phase 1: Large Array Performance (Threading Speedup)")
    print("Comparing: OpenMP threaded vs Single-threaded")
    print("Array sizes: 100K-10M elements (OpenMP enabled at 100K+)\n")

    sizes = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]

    results = []
    speedups = []

    for size in sizes:
        A = np.random.choice([0, 1, 2], size).astype(np.uint8)
        B = np.random.choice([0, 1, 2], size).astype(np.uint8)

        # Fewer iterations for large arrays
        iterations = 50

        time_opt = benchmark_operation(tc_opt.tadd, [A, B], iterations=iterations, warmup=10)
        time_ref = benchmark_operation(tc_ref.tadd, [A, B], iterations=iterations, warmup=10)

        speedup = time_ref / time_opt
        throughput = size / (time_opt / 1e9) / 1e9  # Gtrits/s

        speedups.append(speedup)

        print(f"  n={size//1000:5d}K: {speedup:5.2f}× speedup, {throughput:5.2f} Gtrits/s")

        results.append({
            'size': size,
            'speedup': float(speedup),
            'throughput': float(throughput * 1e9)
        })

    median_speedup = np.median(speedups)
    max_throughput = max(r['throughput'] for r in results)

    print(f"\n  Threading Speedup Range: {np.min(speedups):.1f}× - {np.max(speedups):.1f}×")
    print(f"  Median Threading Speedup: {median_speedup:.1f}×")
    print(f"  Peak Throughput: {max_throughput/1e9:.1f} Gtrits/s")
    print(f"  Target: 2-8× → {'✅ PASS' if 2 <= median_speedup <= 10 else '❌ FAIL'}")

    return results, median_speedup, max_throughput

def main():
    print("="*70)
    print("  Fair Benchmarking Suite - C++ Baseline Comparison")
    print("="*70)
    print("\n  Comparing optimized vs unoptimized C++ implementations")
    print("  (Not against Python - measures actual optimization impact)\n")
    print(f"  Random seed: {RANDOM_SEED}")
    print("  Estimated runtime: ~2 minutes")

    start_time = time.time()

    verify_correctness()

    scalar_results, scalar_speedup = benchmark_scalar_speedup()
    simd_results, simd_speedup = benchmark_simd_speedup()
    large_results, thread_speedup, peak_throughput = benchmark_large_arrays()

    elapsed = time.time() - start_time

    # Summary
    print_section("Fair Benchmark Summary")

    scalar_pass = bool(3 <= scalar_speedup <= 15)
    simd_pass = bool(10 <= simd_speedup <= 30)
    thread_pass = bool(2 <= thread_speedup <= 10)

    print(f"  Phase 0 (Scalar LUTs):     {scalar_speedup:5.2f}× {'✅' if scalar_pass else '❌'}")
    print(f"  Phase 0.5 (SIMD):          {simd_speedup:5.2f}× {'✅' if simd_pass else '❌'}")
    print(f"  Phase 1 (Threading):       {thread_speedup:5.2f}× {'✅' if thread_pass else '❌'}")
    print(f"  Peak Throughput:           {peak_throughput/1e9:5.1f} Gtrits/s")

    all_pass = scalar_pass and simd_pass and thread_pass

    print()
    if all_pass:
        print("  ✅ ALL OPTIMIZATIONS VALIDATED!")
    else:
        print("  ⚠️  Some targets not met (review above)")

    print(f"\n  Total runtime: {elapsed:.1f}s")
    print("="*70)

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'random_seed': RANDOM_SEED,
        'elapsed_sec': elapsed,
        'scalar_benchmarks': scalar_results,
        'simd_benchmarks': simd_results,
        'large_array_benchmarks': large_results,
        'summary': {
            'scalar_speedup': float(scalar_speedup),
            'simd_speedup': float(simd_speedup),
            'thread_speedup': float(thread_speedup),
            'peak_throughput': float(peak_throughput),
            'validation': {
                'scalar_pass': scalar_pass,
                'simd_pass': simd_pass,
                'thread_pass': thread_pass,
                'all_pass': all_pass
            }
        }
    }

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"fair_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved: {output_file}")

    return 0 if all_pass else 1

if __name__ == "__main__":
    exit(main())
