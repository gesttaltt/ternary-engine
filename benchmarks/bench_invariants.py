"""
bench_invariants.py - Comprehensive invariant measurement suite

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

This benchmark measures BOTH geometric and hardware invariants to understand
WHY canonical indexing provides massive speedups in certain performance regions.

MEASUREMENTS:
1. Geometric invariants (entropy, correlation, fractals)
2. Hardware behavior (timing, cache effects, scaling)
3. Performance profiles (throughput across different input characteristics)

GOAL:
Identify ≥3 distinct performance regions with statistical significance (p < 0.05)
where different optimizations excel.

USAGE:
    python benchmarks/bench_invariants.py
    python benchmarks/bench_invariants.py --quick  # Fewer iterations

OUTPUT:
    benchmarks/results/invariant_measurements_YYYYMMDD_HHMMSS.json
    reports/invariant_analysis_YYYYMMDD_HHMMSS.md (generated later by cluster_analysis.py)
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Import measurement modules
from benchmarks.utils.geometric_metrics import GeometricMetrics
from benchmarks.utils.hardware_metrics import HardwareMetrics

# Import production module
try:
    import ternary_simd_engine as tse
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
    print("WARNING: ternary_simd_engine not available. Build it first with: python build/build.py")


class InvariantBenchmark:
    """Comprehensive invariant measurement suite."""

    def __init__(self, quick_mode: bool = False):
        """
        Initialize benchmark.

        Args:
            quick_mode: If True, use fewer iterations for faster results
        """
        self.quick_mode = quick_mode
        self.iterations = 100 if quick_mode else 1000
        self.hw_metrics = HardwareMetrics()
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'quick_mode': quick_mode,
                'iterations': self.iterations,
                'system_info': self.hw_metrics.get_system_info(),
            },
            'measurements': [],
        }

    def measure_dataset(self, data_a: np.ndarray, data_b: np.ndarray, dataset_name: str) -> dict:
        """
        Measure all invariants for a given dataset.

        Args:
            data_a: First operand array
            data_b: Second operand array
            dataset_name: Name for this dataset (e.g., "low_entropy")

        Returns:
            Dictionary with all measurements
        """
        print(f"\n{'='*80}")
        print(f"  MEASURING: {dataset_name}")
        print(f"{'='*80}")

        measurement = {
            'dataset_name': dataset_name,
            'size': len(data_a),
            'geometric_metrics_a': {},
            'geometric_metrics_b': {},
            'performance': {},
        }

        # 1. Geometric metrics
        print("1. Computing geometric metrics...")
        geo_a = GeometricMetrics(data_a)
        geo_b = GeometricMetrics(data_b)

        measurement['geometric_metrics_a'] = geo_a.compute_all_metrics()
        measurement['geometric_metrics_b'] = geo_b.compute_all_metrics()

        # Average metrics
        measurement['geometric_metrics_avg'] = {
            key: (measurement['geometric_metrics_a'][key] + measurement['geometric_metrics_b'][key]) / 2
            for key in measurement['geometric_metrics_a'].keys()
        }

        print(f"   Entropy (avg): {measurement['geometric_metrics_avg']['entropy']:.4f}")
        print(f"   Correlation lag1 (avg): {measurement['geometric_metrics_avg']['autocorrelation_lag1']:.4f}")
        print(f"   Repetitiveness (avg): {measurement['geometric_metrics_avg']['repetitiveness']:.4f}")

        # 2. Performance measurements
        if ENGINE_AVAILABLE:
            print("2. Measuring performance on all operations...")
            measurement['performance'] = self._measure_performance(data_a, data_b)
        else:
            print("2. SKIPPED (engine not available)")
            measurement['performance'] = {}

        return measurement

    def _measure_performance(self, a: np.ndarray, b: np.ndarray) -> dict:
        """
        Measure performance of all operations on given arrays.

        Args:
            a: First operand
            b: Second operand

        Returns:
            Dictionary with performance metrics for each operation
        """
        operations = ['tadd', 'tmul', 'tmin', 'tmax', 'tnot']
        perf = {}

        for op_name in operations:
            op_func = getattr(tse, op_name)

            # Measure with hardware metrics
            if op_name == 'tnot':
                # Unary operation
                hw_result = self.hw_metrics.measure_operation(
                    op_func, a, iterations=self.iterations
                )
            else:
                # Binary operation
                hw_result = self.hw_metrics.measure_operation(
                    op_func, a, b, iterations=self.iterations
                )

            # Calculate throughput
            time_per_op_ns = hw_result['time_ns_per_iter']
            throughput_mops = (len(a) / (time_per_op_ns / 1e9)) / 1e6

            perf[op_name] = {
                'time_ns_per_iter': time_per_op_ns,
                'throughput_mops': throughput_mops,
                'estimated_cache_behavior': hw_result['estimated_cache_behavior'],
            }

            print(f"   {op_name}: {throughput_mops:.2f} Mops/s")

        return perf

    def run_measurements(self):
        """Run measurements on all datasets."""
        print("\n" + "="*80)
        print("  INVARIANT MEASUREMENT SUITE")
        print("="*80)
        print(f"Quick mode: {self.quick_mode}")
        print(f"Iterations: {self.iterations}")
        print(f"Engine available: {ENGINE_AVAILABLE}")

        # Datasets to measure
        datasets = [
            ('low_entropy', 'low'),
            ('medium_entropy', 'medium'),
            ('high_entropy', 'high'),
        ]

        # Additional: random datasets with different characteristics
        print("\nGenerating additional test datasets...")
        from benchmarks.utils.geometric_metrics import generate_synthetic_dataset

        for name, entropy_level in datasets:
            # Generate dataset
            data_a, _ = generate_synthetic_dataset(entropy_level, size=1000000)
            data_b, _ = generate_synthetic_dataset(entropy_level, size=1000000)

            # Measure
            measurement = self.measure_dataset(data_a, data_b, name)
            self.results['measurements'].append(measurement)

        # Save results
        self._save_results()

    def _save_results(self):
        """Save measurement results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = PROJECT_ROOT / "benchmarks" / "results" / f"invariant_measurements_{timestamp}.json"

        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print("\n" + "="*80)
        print(f"✅ RESULTS SAVED: {results_file}")
        print("="*80)

        # Print summary
        self._print_summary()

    def _print_summary(self):
        """Print summary of measurements."""
        print("\nSUMMARY:")
        print("-" * 80)

        for measurement in self.results['measurements']:
            name = measurement['dataset_name']
            geo = measurement['geometric_metrics_avg']

            print(f"\n{name.upper()}:")
            print(f"  Entropy:        {geo['entropy']:.4f}")
            print(f"  Correlation:    {geo['autocorrelation_lag1']:+.4f}")
            print(f"  Repetitiveness: {geo['repetitiveness']:.4f}")

            if measurement['performance']:
                perf = measurement['performance']
                tadd_throughput = perf['tadd']['throughput_mops']
                print(f"  tadd throughput: {tadd_throughput:.2f} Mops/s")


def main():
    parser = argparse.ArgumentParser(description='Measure invariants across different input characteristics')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (fewer iterations)')

    args = parser.parse_args()

    benchmark = InvariantBenchmark(quick_mode=args.quick)
    benchmark.run_measurements()


if __name__ == '__main__':
    main()
