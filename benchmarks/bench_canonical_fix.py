"""
bench_canonical_fix.py - Benchmark after canonical indexing fix

This benchmark validates the performance of the ternary engine after
fixing the canonical indexing bug (LUT mismatch between SIMD and scalar paths).

Compares:
1. Ternary Engine (canonical indexing, now fixed)
2. NumPy INT8 baseline (saturated arithmetic)
3. Overhead breakdown (conversion vs kernel)

Usage:
    python benchmarks/bench_canonical_fix.py

Output:
    - Correctness verification
    - Performance comparison at various array sizes
    - Crossover point analysis
    - Overhead breakdown
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import ternary_simd_engine as te


def benchmark(func, *args, repetitions=100, warmup=10):
    """Run benchmark with statistics."""
    # Warmup
    for _ in range(warmup):
        _ = func(*args)

    # Measure
    times = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        result = func(*args)
        # Force materialization
        _ = result[0] if hasattr(result, '__getitem__') else result
        times.append(time.perf_counter_ns() - start)

    times = np.array(times)
    return {
        'mean_ns': float(np.mean(times)),
        'std_ns': float(np.std(times)),
        'min_ns': float(np.min(times)),
        'max_ns': float(np.max(times)),
        'p50_ns': float(np.percentile(times, 50)),
        'p99_ns': float(np.percentile(times, 99)),
    }


def ternary_tadd_full(a_int8, b_int8):
    """Full pipeline: int8 -> ternary -> operation -> int8."""
    a_u = (a_int8 + 1).astype(np.uint8)
    b_u = (b_int8 + 1).astype(np.uint8)
    r = te.tadd(a_u, b_u)
    return r.astype(np.int8) - 1


def numpy_tadd_saturated(a, b):
    """NumPy saturated addition."""
    return np.clip(a.astype(np.int16) + b.astype(np.int16), -1, 1).astype(np.int8)


def main():
    print("=" * 70)
    print("BENCHMARK: Ternary Engine vs NumPy (Post Canonical Fix)")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # ==========================================================================
    # CORRECTNESS VERIFICATION
    # ==========================================================================
    print("-" * 70)
    print("PHASE 1: Correctness Verification")
    print("-" * 70)

    np.random.seed(42)
    test_sizes = [32, 64, 100, 1000, 10000, 100000]
    all_correct = True

    for size in test_sizes:
        a = np.random.randint(-1, 2, size, dtype=np.int8)
        b = np.random.randint(-1, 2, size, dtype=np.int8)

        r_ternary = ternary_tadd_full(a, b)
        r_numpy = numpy_tadd_saturated(a, b)

        match = np.array_equal(r_ternary, r_numpy)
        status = "OK" if match else "FAIL"
        print(f"  Size {size:>7}: {status}")

        if not match:
            all_correct = False

    print()
    if all_correct:
        print("  [PASS] All correctness tests passed!")
    else:
        print("  [FAIL] Correctness tests failed - aborting benchmark")
        return 1

    # ==========================================================================
    # PERFORMANCE COMPARISON
    # ==========================================================================
    print()
    print("-" * 70)
    print("PHASE 2: Performance Comparison (Full Pipeline)")
    print("-" * 70)
    print()
    print("  Full pipeline includes: int8->uint8 conversion + kernel + uint8->int8")
    print()

    sizes = [64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]
    results = []

    print(f"  {'Size':>10} | {'Ternary':>12} | {'NumPy':>12} | {'Speedup':>8} | {'Winner':>8}")
    print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}")

    crossover_point = None

    for size in sizes:
        np.random.seed(42)
        a = np.random.randint(-1, 2, size, dtype=np.int8)
        b = np.random.randint(-1, 2, size, dtype=np.int8)

        # Benchmark ternary (full pipeline)
        t_ternary = benchmark(ternary_tadd_full, a, b)

        # Benchmark NumPy
        t_numpy = benchmark(numpy_tadd_saturated, a, b)

        speedup = t_numpy['mean_ns'] / t_ternary['mean_ns']
        winner = "TERNARY" if speedup > 1.0 else "NUMPY"

        # Track crossover
        if speedup > 1.0 and crossover_point is None:
            crossover_point = size

        ternary_mops = size / (t_ternary['mean_ns'] / 1e9) / 1e6
        numpy_mops = size / (t_numpy['mean_ns'] / 1e9) / 1e6

        print(f"  {size:>10} | {ternary_mops:>9.1f} M/s | {numpy_mops:>9.1f} M/s | {speedup:>7.2f}x | {winner:>8}")

        results.append({
            'size': size,
            'ternary_ns': t_ternary['mean_ns'],
            'numpy_ns': t_numpy['mean_ns'],
            'speedup': speedup,
            'winner': winner,
            'ternary_mops': ternary_mops,
            'numpy_mops': numpy_mops,
        })

    print()
    if crossover_point:
        print(f"  Crossover point: {crossover_point} elements (ternary wins above this)")
    else:
        print("  Crossover point: NOT FOUND (ternary never wins in tested range)")

    # ==========================================================================
    # OVERHEAD BREAKDOWN
    # ==========================================================================
    print()
    print("-" * 70)
    print("PHASE 3: Overhead Breakdown (size=100,000)")
    print("-" * 70)
    print()

    size = 100000
    np.random.seed(42)
    a = np.random.randint(-1, 2, size, dtype=np.int8)
    b = np.random.randint(-1, 2, size, dtype=np.int8)

    # Pre-convert for kernel-only benchmark
    a_u = (a + 1).astype(np.uint8)
    b_u = (b + 1).astype(np.uint8)

    # Full pipeline
    t_full = benchmark(ternary_tadd_full, a, b)

    # Kernel only (no conversion)
    t_kernel = benchmark(te.tadd, a_u, b_u)

    # Conversion only
    def conversion_only(a, b):
        a_u = (a + 1).astype(np.uint8)
        b_u = (b + 1).astype(np.uint8)
        return a_u, b_u
    t_conv = benchmark(conversion_only, a, b)

    # Back conversion
    r = te.tadd(a_u, b_u)
    def back_conversion(r):
        return r.astype(np.int8) - 1
    t_back = benchmark(back_conversion, r)

    full_ms = t_full['mean_ns'] / 1e6
    kernel_ms = t_kernel['mean_ns'] / 1e6
    conv_ms = t_conv['mean_ns'] / 1e6
    back_ms = t_back['mean_ns'] / 1e6

    print(f"  Full pipeline:      {full_ms:>8.3f} ms (100.0%)")
    print(f"  Input conversion:   {conv_ms:>8.3f} ms ({conv_ms/full_ms*100:>5.1f}%)")
    print(f"  SIMD kernel:        {kernel_ms:>8.3f} ms ({kernel_ms/full_ms*100:>5.1f}%)")
    print(f"  Output conversion:  {back_ms:>8.3f} ms ({back_ms/full_ms*100:>5.1f}%)")

    # Kernel-only comparison
    print()
    print("  Kernel-only vs NumPy (no conversion overhead):")
    t_numpy = benchmark(numpy_tadd_saturated, a, b)
    numpy_ms = t_numpy['mean_ns'] / 1e6
    kernel_speedup = numpy_ms / kernel_ms
    print(f"  SIMD kernel:   {kernel_ms:>8.3f} ms")
    print(f"  NumPy:         {numpy_ms:>8.3f} ms")
    print(f"  Speedup:       {kernel_speedup:>8.2f}x")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("  1. CORRECTNESS: All tests passed after canonical LUT fix")
    print()
    print("  2. FULL PIPELINE PERFORMANCE:")
    if crossover_point:
        print(f"     - Ternary wins above {crossover_point} elements")
        best_speedup = max(r['speedup'] for r in results)
        print(f"     - Best speedup: {best_speedup:.2f}x at {max(sizes)} elements")
    else:
        print("     - NumPy wins at all tested sizes")
    print()
    print("  3. KERNEL-ONLY PERFORMANCE:")
    print(f"     - SIMD kernel is {kernel_speedup:.1f}x faster than NumPy")
    print(f"     - Conversion overhead: {(conv_ms + back_ms) / full_ms * 100:.1f}% of total time")
    print()
    print("  4. RECOMMENDATIONS:")
    if kernel_speedup > 2.0:
        print("     - For data already in ternary format: use ternary engine")
        print("     - For int8 data requiring conversion: depends on array size")
    else:
        print("     - NumPy is competitive; ternary advantage is in memory compression")
    print()
    print("=" * 70)

    # Save results
    output_dir = ROOT / "benchmarks" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"canonical_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'crossover_point': crossover_point,
            'kernel_speedup': kernel_speedup,
            'overhead_breakdown': {
                'full_ms': full_ms,
                'kernel_ms': kernel_ms,
                'conv_in_ms': conv_ms,
                'conv_out_ms': back_ms,
            }
        }, f, indent=2)

    print(f"Results saved to: {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
