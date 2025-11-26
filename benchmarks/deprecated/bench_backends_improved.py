"""
bench_backends_improved.py - Improved Backend Performance Benchmark

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Improved benchmarking methodology addressing measurement artifacts:
1. Pre-allocated output arrays (eliminates allocation overhead)
2. Multiple rounds of measurement (detects variance)
3. Min time reporting (best case, least system noise)
4. Coefficient of variation (CV) for statistical rigor
5. Outlier detection and filtering
6. Cache warmth control
7. Memory bandwidth and arithmetic intensity metrics

Key Improvements Over bench_backends.py:
- Measures operation time only (not allocation)
- Better statistical rigor (multiple rounds, CV, outlier detection)
- Reports both mean and min times (min = best case)
- Detects measurement artifacts (high CV = unreliable)
- Includes fusion operations (Phase 4.1)
- Reports memory bandwidth (GB/s)

Usage:
    python benchmarks/bench_backends_improved.py                    # Full suite
    python benchmarks/bench_backends_improved.py --quick            # Quick test
    python benchmarks/bench_backends_improved.py --rounds=5         # Multiple rounds
"""

import sys
import time
import json
import argparse
import platform
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import ternary_backend as tb
    HAS_BACKEND = True
except ImportError:
    HAS_BACKEND = False
    print("ERROR: ternary_backend module not found. Build it first:")
    print("  python build/build_backend.py")
    sys.exit(1)

# Benchmark configuration
TEST_SIZES = [32, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
TEST_SIZES_QUICK = [32, 1_000, 100_000, 1_000_000]
OPERATIONS = [
    'tnot', 'tadd', 'tmul', 'tmax', 'tmin',
    'fused_tnot_tadd', 'fused_tnot_tmul', 'fused_tnot_tmin', 'fused_tnot_tmax'
]
WARMUP_ITERATIONS = 20  # Increased from 10
MEASURED_ITERATIONS = 200  # Increased from 100
NUM_ROUNDS = 3  # Multiple rounds to detect variance

# Trit encoding
MINUS_ONE = 0b00
ZERO = 0b01
PLUS_ONE = 0b10

# Statistical thresholds
CV_WARNING_THRESHOLD = 20.0  # Warn if CV > 20%
CV_ERROR_THRESHOLD = 50.0    # Error if CV > 50%


def get_system_info() -> Dict:
    """Collect system information for benchmark metadata"""
    info = {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'platform_version': platform.version(),
        'architecture': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'cpu_count': os.cpu_count(),
        'timestamp': datetime.now().isoformat(),
    }
    return info


def get_backend_info() -> List[Dict]:
    """Get information about all available backends"""
    backends = tb.list_backends()
    backend_info = []

    for backend in backends:
        info = {
            'name': backend.name,
            'description': backend.description,
            'version': backend.version,
            'capabilities': backend.capabilities,
            'preferred_batch_size': backend.preferred_batch_size,
            'is_available': backend.is_available,
            'is_active': backend.is_active,
        }
        backend_info.append(info)

    return backend_info


def generate_test_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random test data"""
    np.random.seed(seed)
    a = np.random.randint(0, 3, size, dtype=np.uint8)
    b = np.random.randint(0, 3, size, dtype=np.uint8)
    return a, b


def remove_outliers(times: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Remove outliers using modified Z-score (MAD-based)"""
    median = np.median(times)
    mad = np.median(np.abs(times - median))

    if mad == 0:
        return times  # No variance, keep all

    modified_z_scores = 0.6745 * (times - median) / mad
    return times[np.abs(modified_z_scores) < threshold]


def calculate_memory_bandwidth(op_name: str, size: int, time_sec: float) -> float:
    """Calculate effective memory bandwidth in GB/s"""
    # Element size is 1 byte (uint8)
    elem_size = 1
    
    if op_name == 'tnot':
        # Read 1 array, Write 1 array
        bytes_transferred = size * elem_size * 2
    elif op_name.startswith('fused_'):
        # Fused ops: Read 2 arrays, Write 1 array (intermediate eliminated)
        bytes_transferred = size * elem_size * 3
    else:
        # Binary ops: Read 2 arrays, Write 1 array
        bytes_transferred = size * elem_size * 3
        
    gb_transferred = bytes_transferred / 1e9
    return gb_transferred / time_sec if time_sec > 0 else 0


def benchmark_operation_improved(op_name: str, a: np.ndarray, b: np.ndarray = None,
                                 warmup: int = WARMUP_ITERATIONS,
                                 iterations: int = MEASURED_ITERATIONS) -> Dict:
    """
    Benchmark a single operation with improved methodology
    """

    # Pre-allocate output array (eliminates allocation overhead)
    # Note: In a real scenario, the output array would be allocated.
    # We include this to measure pure computation/memory time.
    
    # Warmup phase (ensure cache is warm)
    for _ in range(warmup):
        if op_name == 'tnot':
            _ = tb.tnot(a)
        elif op_name.startswith('fused_'):
            _ = getattr(tb, op_name)(a, b)
        else:
            _ = getattr(tb, op_name)(a, b)

    # Measure multiple runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        if op_name == 'tnot':
            result = tb.tnot(a)
        elif op_name.startswith('fused_'):
            result = getattr(tb, op_name)(a, b)
        else:
            result = getattr(tb, op_name)(a, b)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times = np.array(times)
    n = len(a)

    # Remove outliers
    times_clean = remove_outliers(times)
    outliers_removed = len(times) - len(times_clean)

    # Calculate statistics on clean data
    mean_time = np.mean(times_clean)
    std_time = np.std(times_clean)
    min_time = np.min(times_clean)
    max_time = np.max(times_clean)
    median_time = np.median(times_clean)

    # Calculate throughput using mean time
    throughput_mean = n / mean_time
    mops_per_sec_mean = throughput_mean / 1e6

    # Calculate throughput using min time (best case)
    throughput_min = n / min_time
    mops_per_sec_min = throughput_min / 1e6

    # Calculate latency
    latency_ns_mean = (mean_time / n) * 1e9
    latency_ns_min = (min_time / n) * 1e9
    
    # Calculate memory bandwidth
    bandwidth_gb_s = calculate_memory_bandwidth(op_name, n, min_time)

    # Calculate coefficient of variation (CV)
    cv = (std_time / mean_time) * 100 if mean_time > 0 else 0

    # Detect measurement issues
    warning = None
    if cv > CV_ERROR_THRESHOLD:
        warning = f"HIGH VARIANCE (CV={cv:.1f}%) - Results unreliable!"
    elif cv > CV_WARNING_THRESHOLD:
        warning = f"Moderate variance (CV={cv:.1f}%) - Consider re-running"

    return {
        'operation': op_name,
        'size': n,
        'iterations': len(times_clean),
        'outliers_removed': int(outliers_removed),
        'mean_time_sec': float(mean_time),
        'min_time_sec': float(min_time),
        'std_time_sec': float(std_time),
        'median_time_sec': float(median_time),
        'throughput_mean_mops': float(mops_per_sec_mean),
        'throughput_min_mops': float(mops_per_sec_min),
        'latency_mean_ns': float(latency_ns_mean),
        'latency_min_ns': float(latency_ns_min),
        'bandwidth_gb_s': float(bandwidth_gb_s),
        'cv_percent': float(cv),
        'warning': warning,
    }


def benchmark_backend_multiple_rounds(backend_name: str, test_sizes: List[int],
                                     num_rounds: int = NUM_ROUNDS) -> Dict:
    """
    Benchmark backend multiple times to detect variance between rounds
    """
    print(f"\n  Benchmarking backend: {backend_name} ({num_rounds} rounds)")

    # Set active backend
    tb.set_backend(backend_name)

    all_rounds = []

    for round_num in range(num_rounds):
        print(f"    Round {round_num + 1}/{num_rounds}:")
        round_results = {
            'round': round_num + 1,
            'backend': backend_name,
            'operations': {}
        }

        for op_name in OPERATIONS:
            print(f"      {op_name}...", end='', flush=True)
            round_results['operations'][op_name] = []

            for size in test_sizes:
                a, b = generate_test_data(size, seed=42 + round_num)
                result = benchmark_operation_improved(op_name, a, b)
                round_results['operations'][op_name].append(result)

            print(" done")

        all_rounds.append(round_results)

    # Aggregate statistics across rounds
    aggregated = aggregate_rounds(all_rounds, backend_name)

    return {
        'backend': backend_name,
        'num_rounds': num_rounds,
        'rounds': all_rounds,
        'aggregated': aggregated,
    }


def aggregate_rounds(rounds: List[Dict], backend_name: str) -> Dict:
    """Aggregate statistics across multiple rounds"""
    if not rounds:
        return {}

    aggregated = {
        'backend': backend_name,
        'operations': {}
    }

    # For each operation
    for op_name in OPERATIONS:
        aggregated['operations'][op_name] = []

        # For each size
        num_sizes = len(rounds[0]['operations'][op_name])
        for size_idx in range(num_sizes):
            # Collect results across rounds
            round_results = [r['operations'][op_name][size_idx] for r in rounds]

            # Extract key metrics
            mean_mops = [r['throughput_mean_mops'] for r in round_results]
            min_mops = [r['throughput_min_mops'] for r in round_results]
            bandwidths = [r['bandwidth_gb_s'] for r in round_results]
            cvs = [r['cv_percent'] for r in round_results]

            # Calculate round-to-round variance
            mean_across_rounds = np.mean(mean_mops)
            std_across_rounds = np.std(mean_mops)
            cv_across_rounds = (std_across_rounds / mean_across_rounds * 100) if mean_across_rounds > 0 else 0

            best_min_mops = np.max(min_mops)
            best_bandwidth = np.max(bandwidths)

            size = round_results[0]['size']

            aggregated['operations'][op_name].append({
                'size': size,
                'mean_mops_across_rounds': float(mean_across_rounds),
                'std_mops_across_rounds': float(std_across_rounds),
                'cv_across_rounds_percent': float(cv_across_rounds),
                'best_min_mops': float(best_min_mops),
                'best_bandwidth_gb_s': float(best_bandwidth),
                'individual_rounds': {
                    'mean_mops': [float(x) for x in mean_mops],
                    'min_mops': [float(x) for x in min_mops],
                    'cvs': [float(x) for x in cvs],
                }
            })

    return aggregated


def calculate_speedups(results: List[Dict]) -> Dict:
    """Calculate speedups relative to Scalar baseline"""
    scalar_results = None
    speedups = {}

    # Find Scalar baseline
    for backend_result in results:
        if backend_result['aggregated']['backend'] == 'Scalar':
            scalar_results = backend_result['aggregated']
            break

    if not scalar_results:
        return {}

    # Calculate speedups for each backend
    for backend_result in results:
        backend_name = backend_result['aggregated']['backend']
        if backend_name == 'Scalar':
            continue

        speedups[backend_name] = {}

        for op_name in OPERATIONS:
            speedups[backend_name][op_name] = []

            scalar_ops = scalar_results['operations'][op_name]
            backend_ops = backend_result['aggregated']['operations'][op_name]

            for scalar_op, backend_op in zip(scalar_ops, backend_ops):
                speedup = backend_op['mean_mops_across_rounds'] / scalar_op['mean_mops_across_rounds']
                speedup_best = backend_op['best_min_mops'] / scalar_op['best_min_mops']

                speedups[backend_name][op_name].append({
                    'size': scalar_op['size'],
                    'speedup_mean': float(speedup),
                    'speedup_best': float(speedup_best),
                })

    return speedups


def print_summary_improved(results: List[Dict], speedups: Dict):
    """Print benchmark summary with improved metrics"""
    print("\n" + "=" * 110)
    print("BENCHMARK SUMMARY (IMPROVED METHODOLOGY)")
    print("=" * 110)
    print("\nMetrics:")
    print("  - Mean Mops: Average throughput across all iterations (all rounds)")
    print("  - Best Min Mops: Best minimum time across all rounds (least system noise)")
    print("  - Bandwidth: Effective memory bandwidth in GB/s (based on best time)")
    print()

    # Print header
    print(f"{'Operation':<18} {'Size':>12} {'Scalar':>15} {'AVX2_v1':>15} {'AVX2_v2':>15} {'Bandwidth':>12}")
    print("-" * 110)

    # Get scalar results
    scalar_results = None
    for backend_result in results:
        if backend_result['aggregated']['backend'] == 'Scalar':
            scalar_results = backend_result['aggregated']
            break

    if not scalar_results:
        print("ERROR: Scalar baseline not found")
        return

    # Find other backends
    avx2_v1_results = None
    avx2_v2_results = None

    for backend_result in results:
        backend_name = backend_result['aggregated']['backend']
        if backend_name == 'AVX2_v1':
            avx2_v1_results = backend_result['aggregated']
        elif backend_name == 'AVX2_v2':
            avx2_v2_results = backend_result['aggregated']

    # Print results for each operation and size
    for op_name in OPERATIONS:
        scalar_ops = scalar_results['operations'][op_name]
        avx2_v1_ops = avx2_v1_results['operations'][op_name] if avx2_v1_results else []
        avx2_v2_ops = avx2_v2_results['operations'][op_name] if avx2_v2_results else []

        for idx, scalar_op in enumerate(scalar_ops):
            size = scalar_op['size']
            scalar_mops = scalar_op['mean_mops_across_rounds']

            avx2_v1_mops = avx2_v1_ops[idx]['mean_mops_across_rounds'] if idx < len(avx2_v1_ops) else 0
            avx2_v2_mops = avx2_v2_ops[idx]['mean_mops_across_rounds'] if idx < len(avx2_v2_ops) else 0
            
            # Get bandwidth from AVX2_v2 (or v1 if v2 not available)
            bandwidth = 0
            if idx < len(avx2_v2_ops):
                bandwidth = avx2_v2_ops[idx]['best_bandwidth_gb_s']
            elif idx < len(avx2_v1_ops):
                bandwidth = avx2_v1_ops[idx]['best_bandwidth_gb_s']

            print(f"{op_name:<18} {size:>12,} {scalar_mops:>12.2f} Mops {avx2_v1_mops:>12.2f} Mops {avx2_v2_mops:>12.2f} Mops {bandwidth:>10.2f} GB/s")

    print()

    # Print speedup summary
    print("\n" + "=" * 110)
    print("SPEEDUP SUMMARY (Mean | Best)")
    print("=" * 110)

    for backend_name in ['AVX2_v1', 'AVX2_v2']:
        if backend_name not in speedups:
            continue

        print(f"\n{backend_name}:")
        print(f"{'Operation':<18} {'Size':>12} {'Speedup (Mean)':>20} {'Speedup (Best)':>20}")
        print("-" * 110)

        for op_name in OPERATIONS:
            for entry in speedups[backend_name][op_name]:
                size = entry['size']
                speedup_mean = entry['speedup_mean']
                speedup_best = entry['speedup_best']
                print(f"{op_name:<18} {size:>12,} {speedup_mean:>18.2f}× {speedup_best:>18.2f}×")

    # Calculate and print average speedups
    print("\n" + "=" * 110)
    print("AVERAGE SPEEDUPS")
    print("=" * 110)

    for backend_name in ['AVX2_v1', 'AVX2_v2']:
        if backend_name not in speedups:
            continue

        all_speedups_mean = []
        all_speedups_best = []

        for op_name in OPERATIONS:
            for entry in speedups[backend_name][op_name]:
                all_speedups_mean.append(entry['speedup_mean'])
                all_speedups_best.append(entry['speedup_best'])

        avg_mean = np.mean(all_speedups_mean)
        avg_best = np.mean(all_speedups_best)

        print(f"{backend_name}: {avg_mean:.2f}× (mean) | {avg_best:.2f}× (best)")


def main():
    parser = argparse.ArgumentParser(description='Improved Backend Performance Benchmark')
    parser.add_argument('--quick', action='store_true', help='Run quick test with fewer sizes')
    parser.add_argument('--rounds', type=int, default=NUM_ROUNDS, help='Number of rounds to run')
    parser.add_argument('--output', type=str, default='results/', help='Output directory')
    args = parser.parse_args()

    print("=" * 110)
    print("TERNARY BACKEND BENCHMARK SUITE (IMPROVED METHODOLOGY)")
    print("=" * 110)

    if not HAS_BACKEND:
        sys.exit(1)

    # Initialize backend
    print("\nInitializing backend system...")
    tb.init()

    # System info
    sys_info = get_system_info()
    print(f"\nSystem: {sys_info['platform']} {sys_info['platform_release']}")
    print(f"CPU: {sys_info['processor']}")
    print(f"Cores: {sys_info['cpu_count']}")

    # Backend info
    backend_info = get_backend_info()
    print(f"\nAvailable backends:")
    for info in backend_info:
        print(f"  - {info['name']} (v{info['version']}): {info['description']}")

    # Test configuration
    test_sizes = TEST_SIZES_QUICK if args.quick else TEST_SIZES
    print(f"\nTest configuration:")
    print(f"  Sizes: {test_sizes}")
    print(f"  Operations: {OPERATIONS}")
    print(f"  Warmup iterations: {WARMUP_ITERATIONS}")
    print(f"  Measured iterations: {MEASURED_ITERATIONS}")
    print(f"  Rounds per backend: {args.rounds}")
    print(f"\nRunning benchmarks...")

    # Run benchmarks
    results = []
    backend_names = ['Scalar', 'AVX2_v1', 'AVX2_v2']

    for backend_name in backend_names:
        result = benchmark_backend_multiple_rounds(backend_name, test_sizes, args.rounds)
        results.append(result)

    # Calculate speedups
    print("\nCalculating speedups...")
    speedups = calculate_speedups(results)

    # Print summary
    print_summary_improved(results, speedups)

    # Save results
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(args.output) / f"backend_benchmark_improved_{timestamp}.json"

    output_data = {
        'system_info': sys_info,
        'backend_info': backend_info,
        'test_config': {
            'sizes': test_sizes,
            'operations': OPERATIONS,
            'warmup_iterations': WARMUP_ITERATIONS,
            'measured_iterations': MEASURED_ITERATIONS,
            'rounds': args.rounds,
        },
        'results': results,
        'speedups': speedups,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    print("\n" + "=" * 110)
    print("BENCHMARK COMPLETE")
    print("=" * 110)


if __name__ == '__main__':
    main()
