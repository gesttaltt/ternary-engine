"""
bench_phase0.py - Production-grade Python benchmark suite for Ternary Engine

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Comprehensive benchmarking suite that measures:
- Throughput (operations/second)
- Latency (nanoseconds per element)
- Speedup vs Python baseline
- Scaling behavior across array sizes

Usage:
    python benchmarks/bench_phase0.py                    # Run full suite
    python benchmarks/bench_phase0.py --quick            # Quick test
    python benchmarks/bench_phase0.py --output=results/  # Custom output dir
"""

import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import ternary_simd_engine as tc
    HAS_TERNARY_ENGINE = True
except ImportError:
    HAS_TERNARY_ENGINE = False
    print("WARNING: ternary_simd_engine not found. Build the module first:")
    print("  python build.py")
    sys.exit(1)

# Benchmark configuration
TEST_SIZES = [32, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
TEST_SIZES_QUICK = [32, 1_000, 100_000, 1_000_000]
OPERATIONS = ['tadd', 'tmul', 'tmin', 'tmax', 'tnot']
WARMUP_ITERATIONS = 100
MEASURED_ITERATIONS = 1000

# Trit encoding
MINUS_ONE = 0b00
ZERO = 0b01
PLUS_ONE = 0b10


class BenchmarkResult:
    """Container for a single benchmark result"""
    def __init__(self, operation: str, size: int, time_ns: float, iterations: int):
        self.operation = operation
        self.size = size
        self.time_ns = time_ns
        self.iterations = iterations
        self.time_per_op = time_ns / iterations
        self.time_per_elem = self.time_per_op / size
        self.throughput_mops = (size * iterations) / (time_ns / 1e9) / 1e6

    def to_dict(self) -> Dict:
        return {
            'operation': self.operation,
            'size': self.size,
            'time_ns_total': self.time_ns,
            'iterations': self.iterations,
            'time_ns_per_op': self.time_per_op,
            'time_ns_per_elem': self.time_per_elem,
            'throughput_mops': self.throughput_mops,
        }


class PythonBaseline:
    """Pure Python reference implementation for baseline comparison"""

    @staticmethod
    def trit_to_int(trit):
        """Convert trit encoding to integer"""
        if trit == MINUS_ONE:
            return -1
        elif trit == PLUS_ONE:
            return 1
        else:
            return 0

    @staticmethod
    def int_to_trit(val):
        """Convert integer to trit encoding"""
        if val < 0:
            return MINUS_ONE
        elif val > 0:
            return PLUS_ONE
        else:
            return ZERO

    @staticmethod
    def tadd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Saturated ternary addition"""
        result = np.zeros_like(a)
        for i in range(len(a)):
            val_a = PythonBaseline.trit_to_int(a[i])
            val_b = PythonBaseline.trit_to_int(b[i])
            sum_val = val_a + val_b
            # Saturate to [-1, +1]
            if sum_val > 1:
                sum_val = 1
            elif sum_val < -1:
                sum_val = -1
            result[i] = PythonBaseline.int_to_trit(sum_val)
        return result

    @staticmethod
    def tmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Ternary multiplication"""
        result = np.zeros_like(a)
        for i in range(len(a)):
            val_a = PythonBaseline.trit_to_int(a[i])
            val_b = PythonBaseline.trit_to_int(b[i])
            result[i] = PythonBaseline.int_to_trit(val_a * val_b)
        return result

    @staticmethod
    def tmin(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Ternary minimum"""
        result = np.zeros_like(a)
        for i in range(len(a)):
            val_a = PythonBaseline.trit_to_int(a[i])
            val_b = PythonBaseline.trit_to_int(b[i])
            result[i] = PythonBaseline.int_to_trit(min(val_a, val_b))
        return result

    @staticmethod
    def tmax(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Ternary maximum"""
        result = np.zeros_like(a)
        for i in range(len(a)):
            val_a = PythonBaseline.trit_to_int(a[i])
            val_b = PythonBaseline.trit_to_int(b[i])
            result[i] = PythonBaseline.int_to_trit(max(val_a, val_b))
        return result

    @staticmethod
    def tnot(a: np.ndarray) -> np.ndarray:
        """Ternary negation"""
        result = np.zeros_like(a)
        for i in range(len(a)):
            val_a = PythonBaseline.trit_to_int(a[i])
            result[i] = PythonBaseline.int_to_trit(-val_a)
        return result


def generate_test_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate reproducible test data"""
    np.random.seed(seed)
    # Generate random trits (0b00, 0b01, 0b10)
    a = np.random.choice([MINUS_ONE, ZERO, PLUS_ONE], size=size).astype(np.uint8)
    b = np.random.choice([MINUS_ONE, ZERO, PLUS_ONE], size=size).astype(np.uint8)
    return a, b


def benchmark_operation(func, a: np.ndarray, b: np.ndarray = None,
                        warmup: int = WARMUP_ITERATIONS,
                        iterations: int = MEASURED_ITERATIONS) -> float:
    """Benchmark a single operation with warmup"""
    # Warmup
    for _ in range(warmup):
        if b is not None:
            _ = func(a, b)
        else:
            _ = func(a)

    # Measured run
    start = time.perf_counter_ns()
    for _ in range(iterations):
        if b is not None:
            _ = func(a, b)
        else:
            _ = func(a)
    end = time.perf_counter_ns()

    return end - start


def run_benchmark_suite(sizes: List[int], verbose: bool = True) -> Dict:
    """Run complete benchmark suite"""
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'module': 'ternary_simd_engine',
            'numpy_version': np.__version__,
            'test_sizes': sizes,
            'warmup_iterations': WARMUP_ITERATIONS,
            'measured_iterations': MEASURED_ITERATIONS,
        },
        'results_optimized': [],
        'results_baseline': [],
    }

    if verbose:
        print("=" * 80)
        print("  TERNARY ENGINE BENCHMARK SUITE")
        print("=" * 80)
        print(f"\nTest sizes: {sizes}")
        print(f"Warmup iterations: {WARMUP_ITERATIONS}")
        print(f"Measured iterations: {MEASURED_ITERATIONS}")
        print("\n" + "-" * 80)

    for size in sizes:
        if verbose:
            print(f"\nArray size: {size:,} elements")
            print("-" * 80)

        a, b = generate_test_data(size)

        for op_name in OPERATIONS:
            # Get operation functions
            tc_func = getattr(tc, op_name)
            py_func = getattr(PythonBaseline, op_name)

            # Benchmark optimized version
            if op_name == 'tnot':
                time_ns = benchmark_operation(tc_func, a)
            else:
                time_ns = benchmark_operation(tc_func, a, b)

            result_opt = BenchmarkResult(op_name, size, time_ns, MEASURED_ITERATIONS)
            results['results_optimized'].append(result_opt.to_dict())

            # Benchmark Python baseline (only for smaller sizes to avoid timeout)
            if size <= 10_000:
                if op_name == 'tnot':
                    time_ns_py = benchmark_operation(py_func, a, warmup=10, iterations=100)
                else:
                    time_ns_py = benchmark_operation(py_func, a, b, warmup=10, iterations=100)

                result_py = BenchmarkResult(op_name, size, time_ns_py, 100)
                results['results_baseline'].append(result_py.to_dict())

                speedup = result_py.time_per_elem / result_opt.time_per_elem
            else:
                speedup = None

            if verbose:
                speedup_str = f"{speedup:.1f}x" if speedup else "N/A"
                print(f"  {op_name:8s} | {result_opt.throughput_mops:8.2f} Mops/s | "
                      f"{result_opt.time_per_elem:8.3f} ns/elem | Speedup: {speedup_str}")

    if verbose:
        print("\n" + "=" * 80)
        print("  BENCHMARK COMPLETE")
        print("=" * 80)

    return results


def save_results(results: Dict, output_dir: Path):
    """Save results to JSON and CSV"""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = output_dir / f"bench_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Save CSV (optimized results)
    csv_path = output_dir / f"bench_results_{timestamp}.csv"
    with open(csv_path, 'w') as f:
        f.write("operation,size,time_ns_total,time_ns_per_elem,throughput_mops\n")
        for r in results['results_optimized']:
            f.write(f"{r['operation']},{r['size']},{r['time_ns_total']:.2f},"
                   f"{r['time_ns_per_elem']:.4f},{r['throughput_mops']:.2f}\n")
    print(f"CSV saved to: {csv_path}")

    return json_path, csv_path


def print_summary(results: Dict):
    """Print summary statistics"""
    print("\n" + "=" * 80)
    print("  PERFORMANCE SUMMARY")
    print("=" * 80)

    # Calculate peak throughput per operation
    print("\nPeak Throughput:")
    for op_name in OPERATIONS:
        op_results = [r for r in results['results_optimized'] if r['operation'] == op_name]
        if op_results:
            peak = max(op_results, key=lambda x: x['throughput_mops'])
            print(f"  {op_name:8s}: {peak['throughput_mops']:8.2f} Mops/s "
                  f"(at {peak['size']:,} elements)")

    # Calculate average speedup (for sizes where baseline exists)
    if results['results_baseline']:
        print("\nAverage Speedup vs Python:")
        for op_name in OPERATIONS:
            opt_results = {r['size']: r for r in results['results_optimized']
                          if r['operation'] == op_name}
            base_results = {r['size']: r for r in results['results_baseline']
                           if r['operation'] == op_name}

            speedups = []
            for size in base_results:
                if size in opt_results:
                    speedup = (base_results[size]['time_ns_per_elem'] /
                              opt_results[size]['time_ns_per_elem'])
                    speedups.append(speedup)

            if speedups:
                avg_speedup = sum(speedups) / len(speedups)
                print(f"  {op_name:8s}: {avg_speedup:6.1f}x")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Ternary Engine Benchmark Suite')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmark with fewer sizes')
    parser.add_argument('--output', type=str, default='benchmarks/results',
                       help='Output directory for results (default: benchmarks/results)')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')

    args = parser.parse_args()

    sizes = TEST_SIZES_QUICK if args.quick else TEST_SIZES
    verbose = not args.quiet

    # Run benchmark suite
    results = run_benchmark_suite(sizes, verbose=verbose)

    # Save results
    output_dir = PROJECT_ROOT / args.output
    save_results(results, output_dir)

    # Print summary
    if verbose:
        print_summary(results)


if __name__ == '__main__':
    main()
