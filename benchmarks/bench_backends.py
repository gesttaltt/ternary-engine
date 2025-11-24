"""
bench_backends.py - Backend Performance Comparison Benchmark

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Comprehensive backend benchmarking suite for v1.2.0 backend system.
Compares performance across all available backends:
- Scalar: Portable reference implementation
- AVX2_v1: Baseline AVX2 (v1.1.0)
- AVX2_v2: Optimized AVX2 with v1.2.0 improvements

Metrics:
- Throughput (Mops/s)
- Latency (ns/element)
- Speedup vs Scalar baseline
- Statistical variance and confidence intervals

Usage:
    python benchmarks/bench_backends.py                    # Full suite
    python benchmarks/bench_backends.py --quick            # Quick test
    python benchmarks/bench_backends.py --output=results/  # Custom output
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
OPERATIONS = ['tnot', 'tadd', 'tmul', 'tmax', 'tmin']
WARMUP_ITERATIONS = 10
MEASURED_ITERATIONS = 100

# Trit encoding
MINUS_ONE = 0b00
ZERO = 0b01
PLUS_ONE = 0b10


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


def benchmark_operation(op_name: str, a: np.ndarray, b: np.ndarray = None,
                       warmup: int = WARMUP_ITERATIONS,
                       iterations: int = MEASURED_ITERATIONS) -> Dict:
    """Benchmark a single operation with statistical analysis"""

    # Warmup
    for _ in range(warmup):
        if op_name == 'tnot':
            _ = tb.tnot(a)
        else:
            _ = getattr(tb, op_name)(a, b)

    # Measure multiple runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        if op_name == 'tnot':
            _ = tb.tnot(a)
        else:
            _ = getattr(tb, op_name)(a, b)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times = np.array(times)
    n = len(a)

    # Calculate statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    median_time = np.median(times)

    # Calculate throughput (operations per second)
    throughput = n / mean_time
    mops_per_sec = throughput / 1e6

    # Calculate latency (nanoseconds per element)
    latency_ns = (mean_time / n) * 1e9

    # Calculate coefficient of variation (CV)
    cv = (std_time / mean_time) * 100 if mean_time > 0 else 0

    return {
        'operation': op_name,
        'size': n,
        'iterations': iterations,
        'mean_time_sec': float(mean_time),
        'std_time_sec': float(std_time),
        'min_time_sec': float(min_time),
        'max_time_sec': float(max_time),
        'median_time_sec': float(median_time),
        'throughput_ops_per_sec': float(throughput),
        'throughput_mops_per_sec': float(mops_per_sec),
        'latency_ns_per_element': float(latency_ns),
        'coefficient_of_variation_percent': float(cv),
    }


def benchmark_backend(backend_name: str, test_sizes: List[int]) -> Dict:
    """Benchmark all operations for a specific backend"""
    print(f"\n  Benchmarking backend: {backend_name}")

    # Set active backend
    tb.set_backend(backend_name)

    results = {
        'backend': backend_name,
        'operations': {}
    }

    for op_name in OPERATIONS:
        print(f"    {op_name}...", end='', flush=True)
        results['operations'][op_name] = []

        for size in test_sizes:
            a, b = generate_test_data(size)
            result = benchmark_operation(op_name, a, b)
            results['operations'][op_name].append(result)

        print(" done")

    return results


def calculate_speedups(results: List[Dict]) -> Dict:
    """Calculate speedups relative to Scalar baseline"""
    scalar_results = None
    speedups = {}

    # Find Scalar baseline
    for backend_result in results:
        if backend_result['backend'] == 'Scalar':
            scalar_results = backend_result
            break

    if not scalar_results:
        return {}

    # Calculate speedups for each backend
    for backend_result in results:
        backend_name = backend_result['backend']
        speedups[backend_name] = {}

        for op_name in OPERATIONS:
            speedups[backend_name][op_name] = []

            scalar_ops = scalar_results['operations'][op_name]
            backend_ops = backend_result['operations'][op_name]

            for scalar_op, backend_op in zip(scalar_ops, backend_ops):
                speedup = scalar_op['throughput_mops_per_sec'] / backend_op['throughput_mops_per_sec']
                speedup = 1.0 / speedup  # Invert to get speedup (higher is better)
                speedups[backend_name][op_name].append({
                    'size': backend_op['size'],
                    'speedup': float(speedup),
                    'scalar_mops': float(scalar_op['throughput_mops_per_sec']),
                    'backend_mops': float(backend_op['throughput_mops_per_sec']),
                })

    return speedups


def print_summary(results: List[Dict], speedups: Dict):
    """Print summary table"""
    print("\n" + "="*90)
    print("BENCHMARK SUMMARY")
    print("="*90)

    # Print header
    print(f"\n{'Operation':<10} {'Size':>12} ", end='')
    for backend_result in results:
        print(f"{backend_result['backend']:>15}", end='')
    print()
    print("-"*90)

    # Print results for each operation and size
    for op_name in OPERATIONS:
        first_size = True
        num_sizes = len(results[0]['operations'][op_name])
        for size_idx in range(num_sizes):
            size = results[0]['operations'][op_name][size_idx]['size']

            if first_size:
                print(f"{op_name:<10}", end='')
                first_size = False
            else:
                print(f"{'':10}", end='')

            print(f" {size:>12,}", end='')

            for backend_result in results:
                op_result = backend_result['operations'][op_name][size_idx]
                mops = op_result['throughput_mops_per_sec']
                print(f" {mops:>11,.2f} Mops", end='')

            print()
        print()

    # Print speedup summary
    print("\n" + "="*90)
    print("SPEEDUP SUMMARY (relative to Scalar baseline)")
    print("="*90)

    for backend_name in speedups:
        if backend_name == 'Scalar':
            continue

        print(f"\n{backend_name}:")
        print(f"{'Operation':<10} {'Size':>12} {'Speedup':>12}")
        print("-"*40)

        for op_name in OPERATIONS:
            for speedup_result in speedups[backend_name][op_name]:
                size = speedup_result['size']
                speedup = speedup_result['speedup']
                print(f"{op_name:<10} {size:>12,} {speedup:>10.2f}×")


def save_results(results: List[Dict], speedups: Dict, system_info: Dict,
                backend_info: List[Dict], output_dir: Path):
    """Save benchmark results to JSON file"""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"backend_benchmark_{timestamp}.json"

    data = {
        'system_info': system_info,
        'backend_info': backend_info,
        'results': results,
        'speedups': speedups,
        'test_configuration': {
            'test_sizes': TEST_SIZES,
            'operations': OPERATIONS,
            'warmup_iterations': WARMUP_ITERATIONS,
            'measured_iterations': MEASURED_ITERATIONS,
        }
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Backend performance benchmark")
    parser.add_argument('--quick', action='store_true', help='Run quick test with fewer sizes')
    parser.add_argument('--output', type=str, default='results', help='Output directory for results')
    args = parser.parse_args()

    test_sizes = TEST_SIZES_QUICK if args.quick else TEST_SIZES
    output_dir = PROJECT_ROOT / args.output

    print("="*90)
    print("TERNARY BACKEND BENCHMARK SUITE (v1.2.0)")
    print("="*90)

    # Initialize backend system
    print("\nInitializing backend system...")
    if not tb.init():
        print("ERROR: Backend initialization failed")
        return 1

    # Get system and backend info
    system_info = get_system_info()
    backend_info = get_backend_info()

    print(f"\nSystem: {system_info['platform']} {system_info['platform_release']}")
    print(f"CPU: {system_info['processor']}")
    print(f"Cores: {system_info['cpu_count']}")
    print(f"\nAvailable backends:")
    for backend in backend_info:
        print(f"  - {backend['name']} (v{backend['version']}): {backend['description']}")

    print(f"\nTest configuration:")
    print(f"  Sizes: {test_sizes}")
    print(f"  Operations: {OPERATIONS}")
    print(f"  Warmup iterations: {WARMUP_ITERATIONS}")
    print(f"  Measured iterations: {MEASURED_ITERATIONS}")

    # Run benchmarks for each backend
    print("\nRunning benchmarks...")
    all_results = []
    for backend in backend_info:
        backend_results = benchmark_backend(backend['name'], test_sizes)
        all_results.append(backend_results)

    # Calculate speedups
    print("\nCalculating speedups...")
    speedups = calculate_speedups(all_results)

    # Print summary
    print_summary(all_results, speedups)

    # Save results
    output_file = save_results(all_results, speedups, system_info, backend_info, output_dir)

    print("\n" + "="*90)
    print("BENCHMARK COMPLETE")
    print("="*90)
    print(f"\nResults saved to: {output_file}")
    print("\nKey findings:")

    # Print key findings
    for backend_name in speedups:
        if backend_name == 'Scalar':
            continue

        # Calculate average speedup across all operations and sizes
        total_speedup = 0
        count = 0
        for op_name in OPERATIONS:
            for speedup_result in speedups[backend_name][op_name]:
                total_speedup += speedup_result['speedup']
                count += 1

        avg_speedup = total_speedup / count if count > 0 else 0
        print(f"  {backend_name}: {avg_speedup:.2f}× average speedup over Scalar")

    return 0


if __name__ == "__main__":
    sys.exit(main())
