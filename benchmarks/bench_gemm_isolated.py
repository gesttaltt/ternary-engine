"""
bench_gemm_isolated.py - Isolated Component Benchmarking for GEMM Performance Analysis

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Isolates each component of GEMM to identify performance bottlenecks:
1. Baseline: Single element operations (LUT-only)
2. Dense243 unpacking overhead
3. Memory access patterns (aligned vs unaligned)
4. GEMM kernel components (multiply-accumulate, skipping zeros)
5. Full GEMM vs NumPy comparison

Output: Statistical analysis for root cause identification
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import ternary_simd_engine as tc
    HAS_SIMD = True
except ImportError:
    HAS_SIMD = False
    print("ERROR: ternary_simd_engine not found")
    sys.exit(1)

try:
    import ternary_dense243_module as td
    HAS_DENSE243 = True
except ImportError:
    HAS_DENSE243 = False

try:
    import ternary_tritnet_gemm as gemm
    HAS_GEMM = True
except ImportError:
    HAS_GEMM = False

# Test configuration
WARMUP = 100
ITERATIONS = 1000
TEST_SIZES = [1000, 10000, 100000, 1000000]  # For element-wise ops
GEMM_SIZES = [(64, 64, 65), (128, 128, 130), (256, 256, 255)]  # For GEMM


def benchmark_function(func, *args, warmup=WARMUP, iterations=ITERATIONS):
    """Benchmark a function and return statistics"""
    # Warmup
    for _ in range(warmup):
        func(*args)

    # Measure
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times_ns = np.array(times) * 1e9
    return {
        'median_ns': float(np.median(times_ns)),
        'mean_ns': float(np.mean(times_ns)),
        'std_ns': float(np.std(times_ns)),
        'min_ns': float(np.min(times_ns)),
        'max_ns': float(np.max(times_ns)),
        'p95_ns': float(np.percentile(times_ns, 95)),
        'cv': float(np.std(times_ns) / np.mean(times_ns)),  # Coefficient of variation
    }


def benchmark_1_baseline_operations():
    """Benchmark 1: Baseline single-element operations (LUT access)"""
    print("\n" + "="*80)
    print("BENCHMARK 1: Baseline Operations (LUT Access)")
    print("="*80)

    results = {}

    for size in TEST_SIZES:
        # Create test arrays
        a = np.random.randint(0, 3, size, dtype=np.uint8)
        a = np.where(a == 0, 0x00, np.where(a == 2, 0x10, 0x01))

        b = np.random.randint(0, 3, size, dtype=np.uint8)
        b = np.where(b == 0, 0x00, np.where(b == 2, 0x10, 0x01))

        # Benchmark operations
        ops = ['tadd', 'tmul', 'tmin', 'tmax', 'tnot']
        results[size] = {}

        for op in ops:
            if op == 'tnot':
                stats = benchmark_function(getattr(tc, op), a)
            else:
                stats = benchmark_function(getattr(tc, op), a, b)

            ns_per_elem = stats['median_ns'] / size
            gops_s = size / stats['median_ns']

            results[size][op] = {
                **stats,
                'ns_per_elem': ns_per_elem,
                'gops_s': gops_s,
            }

            print(f"  {op:6s} | {size:8d} elem | {ns_per_elem:8.3f} ns/elem | {gops_s:8.3f} Gops/s")

    return results


def benchmark_2_dense243_packing():
    """Benchmark 2: Dense243 pack/unpack overhead"""
    print("\n" + "="*80)
    print("BENCHMARK 2: Dense243 Pack/Unpack Overhead")
    print("="*80)

    if not HAS_DENSE243:
        print("  SKIP: Dense243 module not available")
        return None

    results = {}

    for size in TEST_SIZES:
        # Create test data
        trits_2bit = np.random.randint(0, 3, size, dtype=np.uint8)
        trits_2bit = np.where(trits_2bit == 0, 0x00, np.where(trits_2bit == 2, 0x10, 0x01))

        # Benchmark pack
        pack_stats = benchmark_function(td.pack, trits_2bit)

        # Pack once for unpack test
        packed = td.pack(trits_2bit)

        # Benchmark unpack
        unpack_stats = benchmark_function(td.unpack, packed, size)

        ns_per_trit_pack = pack_stats['median_ns'] / size
        ns_per_trit_unpack = unpack_stats['median_ns'] / size

        results[size] = {
            'pack': {**pack_stats, 'ns_per_trit': ns_per_trit_pack},
            'unpack': {**unpack_stats, 'ns_per_trit': ns_per_trit_unpack},
        }

        print(f"  Size {size:8d} | Pack: {ns_per_trit_pack:6.3f} ns/trit | Unpack: {ns_per_trit_unpack:6.3f} ns/trit")

    return results


def benchmark_3_memory_access_patterns():
    """Benchmark 3: Memory access pattern effects"""
    print("\n" + "="*80)
    print("BENCHMARK 3: Memory Access Patterns")
    print("="*80)

    results = {}

    for size in [10000, 100000, 1000000]:
        # Sequential access
        a = np.random.randint(0, 3, size, dtype=np.uint8)
        a = np.where(a == 0, 0x00, np.where(a == 2, 0x10, 0x01))
        b = np.random.randint(0, 3, size, dtype=np.uint8)
        b = np.where(b == 0, 0x00, np.where(b == 2, 0x10, 0x01))

        sequential_stats = benchmark_function(tc.tadd, a, b, warmup=50, iterations=500)

        # Strided access (every 8th element)
        a_strided = a[::8]
        b_strided = b[::8]
        strided_stats = benchmark_function(tc.tadd, a_strided, b_strided, warmup=50, iterations=500)

        # Random access (shuffle indices)
        indices = np.arange(size)
        np.random.shuffle(indices)
        a_random = a[indices]
        b_random = b[indices]
        random_stats = benchmark_function(tc.tadd, a_random, b_random, warmup=50, iterations=500)

        results[size] = {
            'sequential': sequential_stats,
            'strided': strided_stats,
            'random': random_stats,
            'stride_penalty': strided_stats['median_ns'] / sequential_stats['median_ns'],
            'random_penalty': random_stats['median_ns'] / sequential_stats['median_ns'],
        }

        print(f"  Size {size:8d} | Sequential: {sequential_stats['median_ns']:.0f} ns | "
              f"Strided: {strided_stats['median_ns']:.0f} ns ({results[size]['stride_penalty']:.2f}×) | "
              f"Random: {random_stats['median_ns']:.0f} ns ({results[size]['random_penalty']:.2f}×)")

    return results


def benchmark_4_gemm_components():
    """Benchmark 4: GEMM component breakdown"""
    print("\n" + "="*80)
    print("BENCHMARK 4: GEMM Component Breakdown")
    print("="*80)

    if not HAS_GEMM or not HAS_DENSE243:
        print("  SKIP: GEMM or Dense243 module not available")
        return None

    results = {}

    for M, N, K in GEMM_SIZES:
        print(f"\n  Matrix size: {M}×{N}×{K}")

        # Generate test data
        A = np.random.randn(M, K).astype(np.float32)
        B_trits = np.random.randint(0, 3, (K, N), dtype=np.uint8)
        B_trits_2bit = np.where(B_trits == 0, 0x00, np.where(B_trits == 2, 0x10, 0x01))

        # Component 1: Pack weights to Dense243
        def pack_weights():
            K_packed = (K + 4) // 5
            packed = np.zeros((K_packed, N), dtype=np.uint8)
            for n in range(N):
                col = B_trits_2bit[:, n]
                packed[:, n] = td.pack(col)[:K_packed]
            return packed

        pack_stats = benchmark_function(pack_weights, warmup=10, iterations=50)
        B_packed = pack_weights()

        # Component 2: GEMM execution
        gemm_stats = benchmark_function(gemm.gemm, A, B_packed, M, N, K, warmup=10, iterations=50)

        # Component 3: NumPy baseline (FP32 matmul)
        B_full = (B_trits.astype(np.float32) - 1)
        numpy_stats = benchmark_function(np.matmul, A, B_full, warmup=10, iterations=50)

        # Theoretical operations
        ops = M * N * K

        results[f"{M}x{N}x{K}"] = {
            'M': M, 'N': N, 'K': K,
            'operations': ops,
            'pack_time_ns': pack_stats['median_ns'],
            'gemm_time_ns': gemm_stats['median_ns'],
            'numpy_time_ns': numpy_stats['median_ns'],
            'gemm_gops_s': ops / gemm_stats['median_ns'],
            'numpy_gops_s': ops / numpy_stats['median_ns'],
            'slowdown_vs_numpy': gemm_stats['median_ns'] / numpy_stats['median_ns'],
            'pack_overhead_pct': (pack_stats['median_ns'] / gemm_stats['median_ns']) * 100,
        }

        print(f"    Pack weights:    {pack_stats['median_ns']/1e6:.2f} ms")
        print(f"    GEMM execute:    {gemm_stats['median_ns']/1e6:.2f} ms ({results[f'{M}x{N}x{K}']['gemm_gops_s']:.2f} Gops/s)")
        print(f"    NumPy baseline:  {numpy_stats['median_ns']/1e6:.2f} ms ({results[f'{M}x{N}x{K}']['numpy_gops_s']:.2f} Gops/s)")
        print(f"    Slowdown:        {results[f'{M}x{N}x{K}']['slowdown_vs_numpy']:.1f}×")
        print(f"    Pack overhead:   {results[f'{M}x{N}x{K}']['pack_overhead_pct']:.1f}%")

    return results


def benchmark_5_theoretical_limits():
    """Benchmark 5: Theoretical performance limits"""
    print("\n" + "="*80)
    print("BENCHMARK 5: Theoretical Performance Limits")
    print("="*80)

    # CPU specs (estimated for AMD Ryzen)
    cpu_ghz = 3.5  # Base clock
    cores = 12
    avx2_width = 8  # 8 float32 per vector
    fma_per_cycle = 2  # FMA units

    # Theoretical peak
    peak_flops = cpu_ghz * 1e9 * cores * avx2_width * fma_per_cycle
    peak_gflops = peak_flops / 1e9

    # Memory bandwidth (DDR4-3200 dual channel estimate)
    mem_bandwidth_gb_s = 51.2

    results = {
        'cpu_ghz': cpu_ghz,
        'cores': cores,
        'avx2_width': avx2_width,
        'fma_units': fma_per_cycle,
        'peak_gflops': peak_gflops,
        'mem_bandwidth_gb_s': mem_bandwidth_gb_s,
    }

    print(f"  CPU: {cores} cores @ {cpu_ghz} GHz")
    print(f"  AVX2 width: {avx2_width} floats")
    print(f"  FMA units: {fma_per_cycle} per core")
    print(f"  Peak FLOPS: {peak_gflops:.1f} Gflops/s (theoretical max)")
    print(f"  Memory bandwidth: {mem_bandwidth_gb_s} GB/s")

    # Calculate memory bandwidth limits for GEMM
    print(f"\n  Memory Bandwidth Limits (per matrix size):")
    for M, N, K in GEMM_SIZES:
        # Data volume: read A (M*K), read B_packed (K/5*N), write C (M*N)
        data_volume_bytes = (M * K * 4) + ((K+4)//5 * N) + (M * N * 4)
        data_volume_mb = data_volume_bytes / 1e6

        # Time limited by memory bandwidth
        time_mem_limited_s = data_volume_bytes / (mem_bandwidth_gb_s * 1e9)

        # Operations
        ops = M * N * K

        # Throughput if memory-bound
        gops_mem_limited = ops / time_mem_limited_s / 1e9

        results[f"mem_limit_{M}x{N}x{K}"] = {
            'data_volume_mb': data_volume_mb,
            'time_mem_limited_ms': time_mem_limited_s * 1000,
            'gops_mem_limited': gops_mem_limited,
        }

        print(f"    {M}×{N}×{K}: {data_volume_mb:.2f} MB → "
              f"{time_mem_limited_s * 1000:.2f} ms @ {gops_mem_limited:.2f} Gops/s (mem-bound)")

    return results


def run_all_benchmarks():
    """Run all isolated benchmarks"""
    print("="*80)
    print("  ISOLATED GEMM COMPONENT BENCHMARKS")
    print("="*80)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Warmup iterations: {WARMUP}")
    print(f"Measured iterations: {ITERATIONS}")

    all_results = {}

    # Benchmark 1: Baseline operations
    all_results['benchmark_1_baseline'] = benchmark_1_baseline_operations()

    # Benchmark 2: Dense243 packing
    all_results['benchmark_2_dense243'] = benchmark_2_dense243_packing()

    # Benchmark 3: Memory access patterns
    all_results['benchmark_3_memory'] = benchmark_3_memory_access_patterns()

    # Benchmark 4: GEMM components
    all_results['benchmark_4_gemm_components'] = benchmark_4_gemm_components()

    # Benchmark 5: Theoretical limits
    all_results['benchmark_5_theoretical'] = benchmark_5_theoretical_limits()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"bench_gemm_isolated_{timestamp}.json"

    full_results = {
        'timestamp': timestamp,
        'warmup_iterations': WARMUP,
        'measured_iterations': ITERATIONS,
        'test_sizes_elementwise': TEST_SIZES,
        'test_sizes_gemm': [(M, N, K) for M, N, K in GEMM_SIZES],
        'benchmarks': all_results,
    }

    with open(output_file, 'w') as f:
        json.dump(full_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print("="*80)

    return full_results


if __name__ == '__main__':
    results = run_all_benchmarks()
    sys.exit(0)
