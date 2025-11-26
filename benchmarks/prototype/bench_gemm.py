"""
bench_gemm.py - TritNet GEMM Performance Benchmark

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Benchmarks ternary matrix multiplication (GEMM) performance:
- Naive reference implementation
- AVX2-optimized SIMD kernel
- Comparison with NumPy BLAS baseline
- Scaling behavior across matrix sizes

Usage:
    python benchmarks/bench_gemm.py                    # Full suite
    python benchmarks/bench_gemm.py --quick            # Quick test
    python benchmarks/bench_gemm.py --output=results/  # Custom output
"""

import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import ternary_tritnet_gemm as gemm
    HAS_GEMM = True
except ImportError:
    HAS_GEMM = False
    print("ERROR: ternary_tritnet_gemm not found. Build the module first:")
    print("  python build/build_tritnet_gemm.py")
    sys.exit(1)

try:
    import ternary_dense243_module as td
    HAS_DENSE243 = True
except ImportError:
    HAS_DENSE243 = False
    print("WARNING: ternary_dense243_module not found. Packing will be slower.")
    print("  Build with: python build/build_dense243.py")

# Benchmark configuration
# Matrix sizes: (M, N, K) for C[M×N] = A[M×K] @ B[K×N]
# NOTE: K must be multiple of 5 for Dense243 packing
TEST_SIZES_FULL = [
    (8, 16, 20),       # Tiny (TritNet)
    (64, 64, 65),      # Small
    (128, 128, 130),   # Medium
    (256, 256, 255),   # Medium-large
    (512, 512, 510),   # Large
    (1024, 1024, 1025), # Very large
    (2048, 2048, 2050), # Huge
]

TEST_SIZES_QUICK = [
    (8, 16, 20),       # Tiny
    (128, 128, 130),   # Medium
    (512, 512, 510),   # Large
    (2048, 2048, 2050), # Huge
]

WARMUP_ITERATIONS = 10
MEASURED_ITERATIONS = 50

# Dense243 packing helpers (fallback if module not available)
def pack_dense243_fallback(trits_2bit):
    """Pack 2-bit trits to Dense243 format (5 trits/byte) - Python fallback"""
    n = len(trits_2bit)
    n_packed = (n + 4) // 5
    packed = np.zeros(n_packed, dtype=np.uint8)

    for i in range(0, n, 5):
        group = trits_2bit[i:i+5]
        # Pad to 5 if needed
        if len(group) < 5:
            group = np.pad(group, (0, 5 - len(group)), constant_values=0b01)

        # Convert 2-bit to 0-2
        trits_012 = np.where(group == 0b00, 0, np.where(group == 0b10, 2, 1))

        # Base-3 encoding
        packed[i // 5] = (trits_012[0] + trits_012[1]*3 + trits_012[2]*9 +
                          trits_012[3]*27 + trits_012[4]*81)

    return packed


def pack_weights_to_dense243(weights_trits, use_module=True):
    """Pack ternary weights from 2-bit to Dense243 format

    Input: weights_trits [K × N] in 2-bit encoding
    Output: packed [K_packed × N] where K_packed = ceil(K / 5)

    Each column is packed independently into Dense243 format.
    """
    K, N = weights_trits.shape
    K_packed = (K + 4) // 5
    packed = np.zeros((K_packed, N), dtype=np.uint8)

    # Pack each column separately
    for n in range(N):
        column = weights_trits[:, n]

        if use_module and HAS_DENSE243:
            # Fast path: Use compiled module for this column
            packed_column = td.pack(column)
            # May have extra padding, truncate to K_packed
            packed[:, n] = packed_column[:K_packed]
        else:
            # Slow path: Python fallback
            packed[:, n] = pack_dense243_fallback(column)

    return packed


def generate_test_matrices(M, N, K):
    """Generate test matrices A[M×K] and B[K×N] in ternary format"""
    # Activations: FP32
    A = np.random.randn(M, K).astype(np.float32)

    # Weights: Ternary {-1, 0, +1} in 2-bit encoding
    B_trits_012 = np.random.randint(0, 3, (K, N), dtype=np.int8)  # {0, 1, 2}
    B_trits_2bit = np.where(B_trits_012 == 0, 0b00,
                            np.where(B_trits_012 == 2, 0b10, 0b01))

    # Convert to Dense243 format
    B_packed = pack_weights_to_dense243(B_trits_2bit.astype(np.uint8))

    # For NumPy reference: Convert to {-1, 0, +1}
    B_full = B_trits_012.astype(np.float32) - 1  # {0,1,2} → {-1,0,+1}

    return A, B_packed, B_full


def benchmark_gemm_implementation(A, B_packed, M, N, K, name, warmup, iterations):
    """Benchmark a single GEMM implementation"""
    # Warmup
    for _ in range(warmup):
        C = gemm.gemm(A, B_packed, M, N, K)

    # Measure
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        C = gemm.gemm(A, B_packed, M, N, K)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times = np.array(times)
    median_time = np.median(times)
    mean_time = np.mean(times)
    std_time = np.std(times)

    # Calculate throughput
    ops = M * N * K  # Operations
    gops_s = ops / median_time / 1e9

    return {
        'name': name,
        'median_time_s': median_time,
        'mean_time_s': mean_time,
        'std_time_s': std_time,
        'throughput_gops_s': gops_s,
        'ops': ops,
    }


def benchmark_numpy_reference(A, B_full, M, N, K, warmup, iterations):
    """Benchmark NumPy BLAS matmul for reference"""
    # Warmup
    for _ in range(warmup):
        C = A @ B_full

    # Measure
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        C = A @ B_full
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times = np.array(times)
    median_time = np.median(times)
    mean_time = np.mean(times)
    std_time = np.std(times)

    # Calculate throughput
    ops = M * N * K
    gops_s = ops / median_time / 1e9

    return {
        'name': 'NumPy BLAS',
        'median_time_s': median_time,
        'mean_time_s': mean_time,
        'std_time_s': std_time,
        'throughput_gops_s': gops_s,
        'ops': ops,
    }


def validate_correctness(M, N, K):
    """Validate GEMM produces correct results vs NumPy"""
    print(f"Validating correctness on {M}×{N}×{K}...", end=' ')

    A, B_packed, B_full = generate_test_matrices(M, N, K)

    # Ternary GEMM
    C_ternary = gemm.gemm(A, B_packed, M, N, K)

    # NumPy reference
    C_numpy = A @ B_full

    # Compare
    max_error = np.abs(C_ternary - C_numpy).max()
    mean_error = np.abs(C_ternary - C_numpy).mean()

    if max_error < 1e-4:
        print(f"✅ PASS (max_error={max_error:.2e})")
        return True
    else:
        print(f"❌ FAIL (max_error={max_error:.2e})")
        print(f"   Mean error: {mean_error:.2e}")
        return False


def run_benchmark_suite(sizes, output_dir, quick):
    """Run full benchmark suite"""
    print("="*80)
    print("  TritNet GEMM Performance Benchmark")
    print("="*80)
    print()

    # System info
    print(f"Module: ternary_tritnet_gemm")
    print(f"Dense243 module: {'Available' if HAS_DENSE243 else 'Not available (slower packing)'}")
    print(f"NumPy BLAS: {np.__config__.show()}")
    print()

    # Correctness validation
    print("="*80)
    print("  Correctness Validation")
    print("="*80)

    validation_passed = True
    for M, N, K in [(8, 16, 20), (64, 64, 65)]:
        if not validate_correctness(M, N, K):
            validation_passed = False

    if not validation_passed:
        print("\n❌ Correctness validation FAILED. Aborting benchmarks.")
        return None

    print()

    # Performance benchmarks
    print("="*80)
    print("  Performance Benchmarks")
    print("="*80)
    print()

    results = []

    for M, N, K in sizes:
        print(f"Matrix size: {M}×{N}×{K} ({M*N*K:,} operations)")

        # Generate test data
        A, B_packed, B_full = generate_test_matrices(M, N, K)

        # Benchmark GEMM
        gemm_result = benchmark_gemm_implementation(
            A, B_packed, M, N, K, "TritNet GEMM",
            WARMUP_ITERATIONS, MEASURED_ITERATIONS
        )

        # Benchmark NumPy
        numpy_result = benchmark_numpy_reference(
            A, B_full, M, N, K,
            WARMUP_ITERATIONS, MEASURED_ITERATIONS
        )

        # Calculate speedup
        speedup = numpy_result['throughput_gops_s'] / gemm_result['throughput_gops_s']

        print(f"  TritNet GEMM: {gemm_result['throughput_gops_s']:6.2f} Gops/s "
              f"({gemm_result['median_time_s']*1000:.2f} ms)")
        print(f"  NumPy BLAS:   {numpy_result['throughput_gops_s']:6.2f} Gops/s "
              f"({numpy_result['median_time_s']*1000:.2f} ms)")
        print(f"  Ratio:        {speedup:.2f}× (NumPy/Ternary)")
        print()

        results.append({
            'matrix_size': {'M': M, 'N': N, 'K': K},
            'operations': M * N * K,
            'gemm': gemm_result,
            'numpy': numpy_result,
            'speedup_numpy_vs_ternary': speedup,
        })

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"bench_gemm_results_{timestamp}.json"

    full_results = {
        'timestamp': timestamp,
        'test_type': 'quick' if quick else 'full',
        'module': 'ternary_tritnet_gemm',
        'has_dense243': HAS_DENSE243,
        'warmup_iterations': WARMUP_ITERATIONS,
        'measured_iterations': MEASURED_ITERATIONS,
        'results': results,
    }

    with open(output_file, 'w') as f:
        json.dump(full_results, f, indent=2)

    print("="*80)
    print(f"Results saved to: {output_file}")
    print("="*80)

    return full_results


def print_summary(results):
    """Print summary of benchmark results"""
    print("\n" + "="*80)
    print("  Summary")
    print("="*80)

    if not results:
        return

    # Best performance
    best_gemm = max(results['results'], key=lambda x: x['gemm']['throughput_gops_s'])
    best_size = best_gemm['matrix_size']

    print(f"\nBest GEMM Performance:")
    print(f"  {best_gemm['gemm']['throughput_gops_s']:.2f} Gops/s "
          f"at {best_size['M']}×{best_size['N']}×{best_size['K']}")

    # Average speedup
    avg_speedup = np.mean([r['speedup_numpy_vs_ternary'] for r in results['results']])
    print(f"\nAverage Ratio (NumPy/Ternary): {avg_speedup:.2f}×")

    # Performance targets
    print(f"\nPerformance Targets:")
    print(f"  Target: 20-30 Gops/s on large matrices")
    print(f"  Achieved: {best_gemm['gemm']['throughput_gops_s']:.2f} Gops/s")

    if best_gemm['gemm']['throughput_gops_s'] >= 20:
        print(f"  Status: ✅ TARGET MET")
    elif best_gemm['gemm']['throughput_gops_s'] >= 15:
        print(f"  Status: ⚠️  Close to target")
    else:
        print(f"  Status: ❌ Below target (optimization needed)")


def main():
    parser = argparse.ArgumentParser(
        description='TritNet GEMM performance benchmark'
    )
    parser.add_argument('--quick', action='store_true',
                       help='Quick benchmark (fewer sizes)')
    parser.add_argument('--output', type=str, default='benchmarks/results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select test sizes
    sizes = TEST_SIZES_QUICK if args.quick else TEST_SIZES_FULL

    # Run benchmarks
    results = run_benchmark_suite(sizes, output_dir, args.quick)

    # Print summary
    if results:
        print_summary(results)
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
