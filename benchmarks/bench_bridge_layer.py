"""
bench_bridge_layer.py - Benchmark the Bridge Layer (Int8 Fused Operations)

This benchmark validates the algebraic bridge layer that fuses format conversion
with kernel operations, eliminating NumPy overhead.

Compares:
1. NumPy baseline (saturated arithmetic)
2. Naive pipeline (int8 -> uint8 conversion in Python, then kernel)
3. Bridge layer (fused int8 operations in C++)

Expected results:
- Bridge layer should be ~30x faster than naive pipeline
- Bridge layer should match or beat NumPy

Usage:
    python benchmarks/bench_bridge_layer.py
"""

import sys
import time
from pathlib import Path
from datetime import datetime
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def benchmark(func, *args, repetitions=100, warmup=10):
    """Run benchmark with statistics."""
    for _ in range(warmup):
        _ = func(*args)

    times = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        result = func(*args)
        _ = result[0] if hasattr(result, '__getitem__') else result
        times.append(time.perf_counter_ns() - start)

    times = np.array(times)
    return {
        'mean_ns': float(np.mean(times)),
        'std_ns': float(np.std(times)),
        'min_ns': float(np.min(times)),
        'p50_ns': float(np.percentile(times, 50)),
    }


def main():
    print("=" * 80)
    print("BRIDGE LAYER BENCHMARK: Fused Int8 Operations")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Import after path setup
    try:
        import ternary_simd_engine as te
    except ImportError as e:
        print(f"ERROR: Could not import ternary_simd_engine: {e}")
        print("Please build the module first: python build/build.py")
        return 1

    # Check if bridge functions are available
    if not hasattr(te, 'tadd_int8'):
        print("ERROR: Bridge layer functions not found in module.")
        print("The module may need to be rebuilt with the latest code.")
        return 1

    print("Module loaded successfully. Bridge layer functions available.")
    print()

    # ==========================================================================
    # CORRECTNESS VERIFICATION
    # ==========================================================================
    print("-" * 80)
    print("PHASE 1: Correctness Verification")
    print("-" * 80)

    np.random.seed(42)
    test_sizes = [32, 64, 100, 1000, 10000]
    all_correct = True

    for size in test_sizes:
        a = np.random.randint(-1, 2, size, dtype=np.int8)
        b = np.random.randint(-1, 2, size, dtype=np.int8)

        # Bridge layer
        r_bridge = te.tadd_int8(a, b)

        # NumPy reference
        r_numpy = np.clip(a.astype(np.int16) + b.astype(np.int16), -1, 1).astype(np.int8)

        match = np.array_equal(r_bridge, r_numpy)
        status = "OK" if match else "FAIL"
        print(f"  Size {size:>7}: {status}")

        if not match:
            all_correct = False
            # Show first mismatch
            for i in range(min(10, size)):
                if r_bridge[i] != r_numpy[i]:
                    print(f"    Mismatch at {i}: a={a[i]}, b={b[i]}, bridge={r_bridge[i]}, numpy={r_numpy[i]}")

    print()
    if all_correct:
        print("  [PASS] All correctness tests passed!")
    else:
        print("  [FAIL] Correctness tests failed - results may be invalid")
        return 1

    # ==========================================================================
    # PERFORMANCE COMPARISON
    # ==========================================================================
    print()
    print("-" * 80)
    print("PHASE 2: Performance Comparison")
    print("-" * 80)
    print()

    # Define the three approaches
    def numpy_saturated_add(a, b):
        """NumPy baseline."""
        return np.clip(a.astype(np.int16) + b.astype(np.int16), -1, 1).astype(np.int8)

    def naive_pipeline(a, b):
        """Naive pipeline with Python conversion."""
        a_u = (a + 1).astype(np.uint8)
        b_u = (b + 1).astype(np.uint8)
        r = te.tadd(a_u, b_u)
        return r.astype(np.int8) - 1

    def bridge_layer(a, b):
        """Fused bridge layer."""
        return te.tadd_int8(a, b)

    sizes = [64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]

    print(f"  {'Size':>10} | {'NumPy':>12} | {'Naive':>12} | {'Bridge':>12} | {'Bridge/Naive':>12} | {'Bridge/NumPy':>12}")
    print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    results = []

    for size in sizes:
        np.random.seed(42)
        a = np.random.randint(-1, 2, size, dtype=np.int8)
        b = np.random.randint(-1, 2, size, dtype=np.int8)

        t_numpy = benchmark(numpy_saturated_add, a, b)
        t_naive = benchmark(naive_pipeline, a, b)
        t_bridge = benchmark(bridge_layer, a, b)

        numpy_mops = size / (t_numpy['mean_ns'] / 1e9) / 1e6
        naive_mops = size / (t_naive['mean_ns'] / 1e9) / 1e6
        bridge_mops = size / (t_bridge['mean_ns'] / 1e9) / 1e6

        speedup_naive = t_naive['mean_ns'] / t_bridge['mean_ns']
        speedup_numpy = t_numpy['mean_ns'] / t_bridge['mean_ns']

        print(f"  {size:>10} | {numpy_mops:>9.1f} M/s | {naive_mops:>9.1f} M/s | {bridge_mops:>9.1f} M/s | {speedup_naive:>11.2f}x | {speedup_numpy:>11.2f}x")

        results.append({
            'size': size,
            'numpy_mops': numpy_mops,
            'naive_mops': naive_mops,
            'bridge_mops': bridge_mops,
            'speedup_vs_naive': speedup_naive,
            'speedup_vs_numpy': speedup_numpy,
        })

    # ==========================================================================
    # ALL OPERATIONS COMPARISON
    # ==========================================================================
    print()
    print("-" * 80)
    print("PHASE 3: All Bridge Operations (size=100,000)")
    print("-" * 80)
    print()

    size = 100000
    np.random.seed(42)
    a = np.random.randint(-1, 2, size, dtype=np.int8)
    b = np.random.randint(-1, 2, size, dtype=np.int8)

    operations = [
        ('tadd_int8', te.tadd_int8, lambda a, b: np.clip(a.astype(np.int16) + b.astype(np.int16), -1, 1).astype(np.int8)),
        ('tmul_int8', te.tmul_int8, lambda a, b: (a * b).astype(np.int8)),
        ('tmin_int8', te.tmin_int8, lambda a, b: np.minimum(a, b)),
        ('tmax_int8', te.tmax_int8, lambda a, b: np.maximum(a, b)),
    ]

    print(f"  {'Operation':<15} | {'Bridge M/s':>12} | {'NumPy M/s':>12} | {'Speedup':>10} | {'Correct':>8}")
    print(f"  {'-'*15}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*8}")

    for name, bridge_fn, numpy_fn in operations:
        t_bridge = benchmark(bridge_fn, a, b)
        t_numpy = benchmark(numpy_fn, a, b)

        bridge_mops = size / (t_bridge['mean_ns'] / 1e9) / 1e6
        numpy_mops = size / (t_numpy['mean_ns'] / 1e9) / 1e6
        speedup = t_numpy['mean_ns'] / t_bridge['mean_ns']

        # Correctness check
        r_bridge = bridge_fn(a, b)
        r_numpy = numpy_fn(a, b)
        correct = "OK" if np.array_equal(r_bridge, r_numpy) else "FAIL"

        print(f"  {name:<15} | {bridge_mops:>9.1f} M/s | {numpy_mops:>9.1f} M/s | {speedup:>9.2f}x | {correct:>8}")

    # tnot (unary)
    t_bridge = benchmark(te.tnot_int8, a)
    t_numpy = benchmark(lambda x: -x, a)
    bridge_mops = size / (t_bridge['mean_ns'] / 1e9) / 1e6
    numpy_mops = size / (t_numpy['mean_ns'] / 1e9) / 1e6
    speedup = t_numpy['mean_ns'] / t_bridge['mean_ns']
    r_bridge = te.tnot_int8(a)
    r_numpy = -a
    correct = "OK" if np.array_equal(r_bridge, r_numpy) else "FAIL"
    print(f"  {'tnot_int8':<15} | {bridge_mops:>9.1f} M/s | {numpy_mops:>9.1f} M/s | {speedup:>9.2f}x | {correct:>8}")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    avg_speedup_naive = np.mean([r['speedup_vs_naive'] for r in results])
    avg_speedup_numpy = np.mean([r['speedup_vs_numpy'] for r in results])
    max_speedup_naive = max(r['speedup_vs_naive'] for r in results)
    max_speedup_numpy = max(r['speedup_vs_numpy'] for r in results)

    print(f"  Bridge Layer vs Naive Pipeline:")
    print(f"    Average speedup: {avg_speedup_naive:.2f}x")
    print(f"    Maximum speedup: {max_speedup_naive:.2f}x")
    print()
    print(f"  Bridge Layer vs NumPy:")
    print(f"    Average speedup: {avg_speedup_numpy:.2f}x")
    print(f"    Maximum speedup: {max_speedup_numpy:.2f}x")
    print()

    if avg_speedup_numpy >= 1.0:
        print("  [SUCCESS] Bridge layer is faster than NumPy on average!")
    else:
        print("  [NOTE] Bridge layer is competitive with NumPy")

    if avg_speedup_naive >= 10.0:
        print("  [SUCCESS] Bridge layer eliminates conversion overhead as expected!")
    else:
        print(f"  [NOTE] Speedup vs naive ({avg_speedup_naive:.1f}x) lower than expected (30x)")

    print()
    print("=" * 80)
    print("ALGEBRAIC INTERPRETATION")
    print("=" * 80)
    print("""
  The Bridge Layer implements the isomorphism φ: Int8 → Uint8 defined by:

    φ(x) = x + 1       (maps -1 → 0, 0 → 1, +1 → 2)
    φ⁻¹(y) = y - 1     (inverse mapping)

  By fusing φ and φ⁻¹ with the kernel operation in SIMD registers:

    result = φ⁻¹(kernel(φ(a), φ(b)))

  We eliminate:
    - 4 NumPy array allocations (2 input, 2 output conversions)
    - 4 memory round-trips (write temp arrays, read them back)
    - Python interpreter overhead for each NumPy operation

  The result is a direct int8 → int8 operation with zero conversion overhead.
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
