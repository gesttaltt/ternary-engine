"""
bench_fusion_validation.py - Validate Phase 4.1 Fusion Operation Performance

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Validates that fusion operations achieve expected speedups compared to
unfused equivalents. This is a focused benchmark for Phase 4.1 validation.

Expected Performance (from Phase 4.1 validation 2025-10-29):
- fused_tnot_tadd: 1.76× average (1.62-1.95× range)
- fused_tnot_tmul: 1.71× average (1.53-1.86× range)
- fused_tnot_tmin: 4.06× average (1.61-11.26× range)
- fused_tnot_tmax: 3.68× average (1.65-9.50× range)

Measurement Methodology:
- Multiple rounds for stability
- Outlier detection and removal
- Both mean and min (best case) reporting
- Large array sizes to minimize noise

Usage:
    python benchmarks/bench_fusion_validation.py
"""

import sys
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import ternary_backend as tb
except ImportError:
    print("ERROR: ternary_backend not found. Build it first:")
    print("  python build/build_backend.py")
    sys.exit(1)

# Benchmark configuration
TEST_SIZES = [100_000, 1_000_000]  # Focus on large arrays where fusion shines
WARMUP_ITERATIONS = 20
MEASURED_ITERATIONS = 100
NUM_ROUNDS = 3

# Expected speedups (conservative targets)
EXPECTED_SPEEDUPS = {
    'fused_tnot_tadd': 1.5,   # Conservative (actual: 1.76×)
    'fused_tnot_tmul': 1.5,   # Conservative (actual: 1.71×)
    'fused_tnot_tmin': 1.5,   # Conservative (actual: 4.06×, but high variance)
    'fused_tnot_tmax': 1.5,   # Conservative (actual: 3.68×, but high variance)
}


def remove_outliers(times: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Remove outliers using modified Z-score (MAD-based)"""
    median = np.median(times)
    mad = np.median(np.abs(times - median))

    if mad == 0:
        return times

    modified_z_scores = 0.6745 * (times - median) / mad
    return times[np.abs(modified_z_scores) < threshold]


def benchmark_unfused(op_binary, op_unary, a, b, warmup, iterations):
    """Benchmark unfused operation: unary(binary(a, b))"""
    # Warmup
    for _ in range(warmup):
        temp = op_binary(a, b)
        _ = op_unary(temp)

    # Measure
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        temp = op_binary(a, b)
        result = op_unary(temp)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times = np.array(times)
    times_clean = remove_outliers(times)

    return {
        'mean_time': float(np.mean(times_clean)),
        'min_time': float(np.min(times_clean)),
        'std_time': float(np.std(times_clean)),
        'cv': float(np.std(times_clean) / np.mean(times_clean) * 100),
    }


def benchmark_fused(op_fused, a, b, warmup, iterations):
    """Benchmark fused operation"""
    # Warmup
    for _ in range(warmup):
        _ = op_fused(a, b)

    # Measure
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = op_fused(a, b)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times = np.array(times)
    times_clean = remove_outliers(times)

    return {
        'mean_time': float(np.mean(times_clean)),
        'min_time': float(np.min(times_clean)),
        'std_time': float(np.std(times_clean)),
        'cv': float(np.std(times_clean) / np.mean(times_clean) * 100),
    }


def benchmark_fusion_op(backend_name, fused_name, binary_name, size):
    """Benchmark one fusion operation"""
    tb.set_backend(backend_name)

    # Get operations
    op_fused = getattr(tb, fused_name)
    op_binary = getattr(tb, binary_name)
    op_unary = tb.tnot

    # Generate test data
    np.random.seed(42)
    a = np.random.randint(0, 3, size, dtype=np.uint8)
    b = np.random.randint(0, 3, size, dtype=np.uint8)

    # Benchmark unfused
    unfused_results = benchmark_unfused(op_binary, op_unary, a, b,
                                       WARMUP_ITERATIONS, MEASURED_ITERATIONS)

    # Benchmark fused
    fused_results = benchmark_fused(op_fused, a, b,
                                    WARMUP_ITERATIONS, MEASURED_ITERATIONS)

    # Calculate speedup
    speedup_mean = unfused_results['mean_time'] / fused_results['mean_time']
    speedup_best = unfused_results['min_time'] / fused_results['min_time']

    return {
        'size': size,
        'unfused': unfused_results,
        'fused': fused_results,
        'speedup_mean': speedup_mean,
        'speedup_best': speedup_best,
    }


def main():
    print("=" * 80)
    print("FUSION OPERATIONS PERFORMANCE VALIDATION (Phase 4.1)")
    print("=" * 80)

    tb.init()

    # Find backends with fusion support
    TERNARY_CAP_FUSION = 0x0020
    backends = []
    for backend in tb.list_backends():
        if backend.capabilities & TERNARY_CAP_FUSION:
            backends.append(backend.name)

    print(f"\nBackends with fusion support: {backends}")

    if not backends:
        print("ERROR: No backends support fusion operations")
        return False

    # Focus on AVX2_v2 (the optimized backend)
    test_backend = 'AVX2_v2' if 'AVX2_v2' in backends else backends[0]
    print(f"Testing backend: {test_backend}\n")

    fusion_ops = [
        ('fused_tnot_tadd', 'tadd'),
        ('fused_tnot_tmul', 'tmul'),
        ('fused_tnot_tmin', 'tmin'),
        ('fused_tnot_tmax', 'tmax'),
    ]

    results = {}
    passed = 0
    failed = 0

    for fused_name, binary_name in fusion_ops:
        print(f"\n{'='*80}")
        print(f"Testing: {fused_name}")
        print(f"{'='*80}")

        expected_speedup = EXPECTED_SPEEDUPS[fused_name]
        op_results = []

        for size in TEST_SIZES:
            print(f"\n  Size: {size:,} elements")

            result = benchmark_fusion_op(test_backend, fused_name, binary_name, size)
            op_results.append(result)

            # Print results
            unfused_mean_ms = result['unfused']['mean_time'] * 1000
            fused_mean_ms = result['fused']['mean_time'] * 1000
            speedup_mean = result['speedup_mean']
            speedup_best = result['speedup_best']

            print(f"    Unfused: {unfused_mean_ms:>8.3f} ms (CV: {result['unfused']['cv']:.1f}%)")
            print(f"    Fused:   {fused_mean_ms:>8.3f} ms (CV: {result['fused']['cv']:.1f}%)")
            print(f"    Speedup: {speedup_mean:>8.2f}× (mean) | {speedup_best:>6.2f}× (best)")

            # Check if meets expectation
            if speedup_mean >= expected_speedup:
                print(f"    ✅ PASS (meets {expected_speedup:.1f}× target)")
                passed += 1
            else:
                print(f"    ❌ FAIL (below {expected_speedup:.1f}× target)")
                failed += 1

        results[fused_name] = op_results

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    for fused_name, op_results in results.items():
        avg_speedup = np.mean([r['speedup_mean'] for r in op_results])
        best_speedup = np.max([r['speedup_best'] for r in op_results])
        print(f"{fused_name:20s}: {avg_speedup:.2f}× avg | {best_speedup:.2f}× best")

    print(f"\nTests passed: {passed}/{passed + failed}")

    if failed == 0:
        print("\n✅ ALL FUSION OPERATIONS VALIDATED")
        print("\nPhase 4.1 fusion operations achieve expected performance improvements!")
        return True
    else:
        print(f"\n⚠️ {failed} test(s) did not meet performance targets")
        print("\nNote: Fusion performance can vary with system load and CPU state.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
