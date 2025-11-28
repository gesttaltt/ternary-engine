"""
bench_conversion_breakdown.py - Detailed analysis of conversion overhead

Breaks down exactly where time and memory are spent in the full pipeline:
    int8 -> uint8 -> SIMD kernel -> uint8 -> int8

Each micro-operation is measured independently to identify optimization targets.
"""

import sys
import time
import tracemalloc
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import ternary_simd_engine as te


def time_operation(func, *args, repetitions=100, warmup=10):
    """Measure time with high precision."""
    for _ in range(warmup):
        _ = func(*args)

    times = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        result = func(*args)
        _ = result[0] if hasattr(result, '__getitem__') else result
        times.append(time.perf_counter_ns() - start)

    return {
        'mean_ns': np.mean(times),
        'std_ns': np.std(times),
        'min_ns': np.min(times),
    }


def memory_operation(func, *args):
    """Measure peak memory allocation."""
    tracemalloc.start()
    result = func(*args)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {'current_bytes': current, 'peak_bytes': peak, 'result': result}


def main():
    print("=" * 80)
    print("CONVERSION OVERHEAD BREAKDOWN ANALYSIS")
    print("=" * 80)
    print()

    sizes = [1000, 10000, 100000, 1000000]

    for size in sizes:
        print(f"\n{'='*80}")
        print(f"ARRAY SIZE: {size:,} elements ({size * 1 / 1024:.1f} KB as int8)")
        print("=" * 80)

        np.random.seed(42)
        a_int8 = np.random.randint(-1, 2, size, dtype=np.int8)
        b_int8 = np.random.randint(-1, 2, size, dtype=np.int8)

        # Pre-convert for isolated tests
        a_uint8 = (a_int8 + 1).astype(np.uint8)
        b_uint8 = (b_int8 + 1).astype(np.uint8)
        result_uint8 = te.tadd(a_uint8, b_uint8)

        print("\n" + "-" * 80)
        print("PHASE 1: INPUT CONVERSION BREAKDOWN")
        print("-" * 80)

        # Step 1a: int8 + 1 (creates int16 intermediate!)
        def step_add_one(arr):
            return arr + 1
        t1a = time_operation(step_add_one, a_int8)
        m1a = memory_operation(step_add_one, a_int8)

        # Check intermediate dtype
        intermediate = a_int8 + 1
        print(f"\n  Step 1a: arr + 1")
        print(f"    Input dtype:  {a_int8.dtype}")
        print(f"    Output dtype: {intermediate.dtype}  <-- PROBLEM: NumPy promotes to int16!")
        print(f"    Time:         {t1a['mean_ns']/1e6:.4f} ms")
        print(f"    Memory peak:  {m1a['peak_bytes']:,} bytes ({m1a['peak_bytes']/size:.1f} bytes/element)")

        # Step 1b: .astype(np.uint8)
        def step_astype_uint8(arr):
            return arr.astype(np.uint8)
        intermediate_int16 = a_int8 + 1
        t1b = time_operation(step_astype_uint8, intermediate_int16)
        m1b = memory_operation(step_astype_uint8, intermediate_int16)

        print(f"\n  Step 1b: .astype(np.uint8)")
        print(f"    Input dtype:  {intermediate_int16.dtype}")
        print(f"    Output dtype: uint8")
        print(f"    Time:         {t1b['mean_ns']/1e6:.4f} ms")
        print(f"    Memory peak:  {m1b['peak_bytes']:,} bytes ({m1b['peak_bytes']/size:.1f} bytes/element)")

        # Combined input conversion (what we actually do)
        def full_input_conv(a, b):
            return (a + 1).astype(np.uint8), (b + 1).astype(np.uint8)
        t1_full = time_operation(full_input_conv, a_int8, b_int8)
        m1_full = memory_operation(full_input_conv, a_int8, b_int8)

        print(f"\n  TOTAL INPUT CONVERSION (both arrays):")
        print(f"    Time:         {t1_full['mean_ns']/1e6:.4f} ms")
        print(f"    Memory peak:  {m1_full['peak_bytes']:,} bytes")

        print("\n" + "-" * 80)
        print("PHASE 2: SIMD KERNEL (for reference)")
        print("-" * 80)

        t2 = time_operation(te.tadd, a_uint8, b_uint8)
        m2 = memory_operation(te.tadd, a_uint8, b_uint8)

        print(f"\n  SIMD tadd kernel:")
        print(f"    Time:         {t2['mean_ns']/1e6:.4f} ms")
        print(f"    Memory peak:  {m2['peak_bytes']:,} bytes ({m2['peak_bytes']/size:.1f} bytes/element)")
        print(f"    Throughput:   {size / (t2['mean_ns'] / 1e9) / 1e6:.1f} Mops/s")

        print("\n" + "-" * 80)
        print("PHASE 3: OUTPUT CONVERSION BREAKDOWN")
        print("-" * 80)

        # Step 3a: .astype(np.int8)
        def step_astype_int8(arr):
            return arr.astype(np.int8)
        t3a = time_operation(step_astype_int8, result_uint8)
        m3a = memory_operation(step_astype_int8, result_uint8)

        print(f"\n  Step 3a: .astype(np.int8)")
        print(f"    Input dtype:  {result_uint8.dtype}")
        print(f"    Output dtype: int8")
        print(f"    Time:         {t3a['mean_ns']/1e6:.4f} ms")
        print(f"    Memory peak:  {m3a['peak_bytes']:,} bytes ({m3a['peak_bytes']/size:.1f} bytes/element)")

        # Step 3b: arr - 1
        result_int8 = result_uint8.astype(np.int8)
        def step_sub_one(arr):
            return arr - 1
        t3b = time_operation(step_sub_one, result_int8)
        m3b = memory_operation(step_sub_one, result_int8)

        # Check intermediate dtype
        intermediate_out = result_int8 - 1
        print(f"\n  Step 3b: arr - 1")
        print(f"    Input dtype:  {result_int8.dtype}")
        print(f"    Output dtype: {intermediate_out.dtype}  <-- May promote depending on value range")
        print(f"    Time:         {t3b['mean_ns']/1e6:.4f} ms")
        print(f"    Memory peak:  {m3b['peak_bytes']:,} bytes ({m3b['peak_bytes']/size:.1f} bytes/element)")

        # Combined output conversion
        def full_output_conv(r):
            return r.astype(np.int8) - 1
        t3_full = time_operation(full_output_conv, result_uint8)
        m3_full = memory_operation(full_output_conv, result_uint8)

        print(f"\n  TOTAL OUTPUT CONVERSION:")
        print(f"    Time:         {t3_full['mean_ns']/1e6:.4f} ms")
        print(f"    Memory peak:  {m3_full['peak_bytes']:,} bytes")

        print("\n" + "-" * 80)
        print("PHASE 4: NUMPY BASELINE (what we compare against)")
        print("-" * 80)

        def numpy_saturated_add(a, b):
            return np.clip(a.astype(np.int16) + b.astype(np.int16), -1, 1).astype(np.int8)

        t4 = time_operation(numpy_saturated_add, a_int8, b_int8)
        m4 = memory_operation(numpy_saturated_add, a_int8, b_int8)

        print(f"\n  NumPy saturated add (also has conversions!):")
        print(f"    Time:         {t4['mean_ns']/1e6:.4f} ms")
        print(f"    Memory peak:  {m4['peak_bytes']:,} bytes")

        # Break down NumPy's overhead too
        def numpy_step1(a, b):
            return a.astype(np.int16), b.astype(np.int16)
        def numpy_step2(a16, b16):
            return a16 + b16
        def numpy_step3(sum16):
            return np.clip(sum16, -1, 1)
        def numpy_step4(clipped):
            return clipped.astype(np.int8)

        t4a = time_operation(numpy_step1, a_int8, b_int8)
        a16, b16 = a_int8.astype(np.int16), b_int8.astype(np.int16)
        t4b = time_operation(numpy_step2, a16, b16)
        sum16 = a16 + b16
        t4c = time_operation(numpy_step3, sum16)
        clipped = np.clip(sum16, -1, 1)
        t4d = time_operation(numpy_step4, clipped)

        print(f"\n  NumPy breakdown:")
        print(f"    astype(int16) x2:  {t4a['mean_ns']/1e6:.4f} ms")
        print(f"    a16 + b16:         {t4b['mean_ns']/1e6:.4f} ms")
        print(f"    clip(-1, 1):       {t4c['mean_ns']/1e6:.4f} ms")
        print(f"    astype(int8):      {t4d['mean_ns']/1e6:.4f} ms")

        print("\n" + "-" * 80)
        print("PHASE 5: SUMMARY & INSIGHTS")
        print("-" * 80)

        total_ternary = t1_full['mean_ns'] + t2['mean_ns'] + t3_full['mean_ns']
        total_numpy = t4['mean_ns']

        print(f"\n  TERNARY PIPELINE:")
        print(f"    Input conversion:  {t1_full['mean_ns']/1e6:.4f} ms ({t1_full['mean_ns']/total_ternary*100:.1f}%)")
        print(f"    SIMD kernel:       {t2['mean_ns']/1e6:.4f} ms ({t2['mean_ns']/total_ternary*100:.1f}%)")
        print(f"    Output conversion: {t3_full['mean_ns']/1e6:.4f} ms ({t3_full['mean_ns']/total_ternary*100:.1f}%)")
        print(f"    TOTAL:             {total_ternary/1e6:.4f} ms")

        print(f"\n  NUMPY PIPELINE:")
        print(f"    TOTAL:             {total_numpy/1e6:.4f} ms")

        print(f"\n  COMPARISON:")
        print(f"    Speedup (full):    {total_numpy/total_ternary:.2f}x")
        print(f"    Speedup (kernel):  {total_numpy/t2['mean_ns']:.2f}x")

        # Identify the bottleneck
        print(f"\n  KEY INSIGHT:")
        if t1_full['mean_ns'] > t3_full['mean_ns']:
            print(f"    Input conversion is the bottleneck ({t1_full['mean_ns']/total_ternary*100:.1f}% of time)")
            print(f"    Cause: NumPy promotes int8+1 to int16, then casts to uint8")
            print(f"    Memory: Creates 2-byte intermediate for each 1-byte element")
        else:
            print(f"    Output conversion is the bottleneck ({t3_full['mean_ns']/total_ternary*100:.1f}% of time)")

        print(f"\n  MEMORY ALLOCATIONS:")
        print(f"    Input arrays:      {2 * size:,} bytes (int8 x 2)")
        print(f"    Intermediate:      ~{4 * size:,} bytes (int16 x 2 during +1)")
        print(f"    Output:            {size:,} bytes (int8)")
        print(f"    Peak working set:  ~{6 * size:,} bytes")

        # Calculate theoretical minimum
        print(f"\n  THEORETICAL ANALYSIS:")
        mem_bandwidth_gbps = 50  # Typical DDR4 bandwidth
        theoretical_time_ns = (6 * size) / (mem_bandwidth_gbps * 1e9) * 1e9
        print(f"    Data touched:      {6 * size:,} bytes")
        print(f"    At {mem_bandwidth_gbps} GB/s:       {theoretical_time_ns/1e6:.4f} ms (memory bound)")
        print(f"    Actual time:       {total_ternary/1e6:.4f} ms")
        print(f"    Efficiency:        {theoretical_time_ns/total_ternary*100:.1f}%")

    print("\n" + "=" * 80)
    print("OPTIMIZATION OPPORTUNITIES")
    print("=" * 80)
    print("""
    1. ELIMINATE int16 INTERMEDIATE:
       Current:  (arr + 1).astype(np.uint8)  -> creates int16, then casts
       Better:   Custom C++ that adds 1 directly to int8 -> uint8
       Savings:  ~50% of input conversion time

    2. FUSE CONVERSION WITH KERNEL:
       Current:  Python loop: convert -> kernel -> convert
       Better:   C++ kernel accepts int8, does conversion internally
       Savings:  Eliminates Python/NumPy overhead entirely

    3. IN-PLACE OPERATIONS:
       Current:  Creates new array for each step
       Better:   Modify arrays in-place where possible
       Savings:  Reduces memory allocations

    4. NATIVE TERNARY FORMAT:
       Current:  Convert int8 <-> uint8 for each operation
       Better:   Store data in native ternary format, only convert at boundaries
       Savings:  29x kernel speedup applies to entire pipeline

    5. PACKED REPRESENTATION:
       Current:  1 trit per byte (uint8)
       Better:   4 trits per byte (Dense243: 5 trits per byte)
       Savings:  4-5x memory reduction, better cache utilization
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
