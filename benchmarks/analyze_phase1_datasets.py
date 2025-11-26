"""
analyze_phase1_datasets.py - Analyze the exact patterns in Phase 1 datasets

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

PURPOSE:
Phase 1 found 40× slowdown with low_entropy dataset, but investigation shows only 1.4× slowdown.
This script analyzes the EXACT patterns in Phase 1 datasets to understand the discrepancy.

USAGE:
    python benchmarks/analyze_phase1_datasets.py
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.utils.geometric_metrics import GeometricMetrics
from benchmarks.utils.hardware_metrics import HardwareMetrics

try:
    import ternary_simd_engine as tse
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
    print("ERROR: ternary_simd_engine not available")
    sys.exit(1)


def analyze_dataset(dataset_path: Path, name: str):
    """Analyze a Phase 1 dataset."""
    print("\n" + "="*80)
    print(f"  ANALYZING: {name}")
    print("="*80)

    # Load dataset
    data = np.load(dataset_path)
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Size: {len(data):,} elements")

    # Show first 100 elements
    print(f"\nFirst 100 elements:")
    print(data[:100])

    # Find repeating pattern
    print(f"\nPattern analysis:")

    # Check if it's a simple repeating pattern
    for pattern_len in [1, 2, 4, 8, 16, 32, 64, 128, 256, 400]:
        if pattern_len > len(data):
            break

        pattern = data[:pattern_len]

        # Check if this pattern repeats throughout
        num_reps = len(data) // pattern_len
        reconstructed = np.tile(pattern, num_reps)

        if len(reconstructed) < len(data):
            reconstructed = np.concatenate([reconstructed, pattern[:len(data) - len(reconstructed)]])
        else:
            reconstructed = reconstructed[:len(data)]

        matches = np.sum(reconstructed == data)
        match_rate = matches / len(data)

        if match_rate > 0.99:
            print(f"  Pattern length {pattern_len}: {match_rate*100:.2f}% match")
            print(f"    Pattern: {pattern}")
            break

    # Geometric metrics
    print(f"\nGeometric metrics:")
    geo = GeometricMetrics(data)
    metrics = geo.compute_all_metrics()

    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")

    # Performance test
    if ENGINE_AVAILABLE:
        print(f"\nPerformance test (tadd):")
        hw_metrics = HardwareMetrics()

        # Test with same dataset for both operands
        result = hw_metrics.measure_operation(tse.tadd, data, data, iterations=100)

        time_per_op_ns = result['time_ns_per_iter']
        throughput_mops = (len(data) / (time_per_op_ns / 1e9)) / 1e6

        print(f"  Throughput: {throughput_mops:.2f} Mops/s")
        print(f"  Time per op: {time_per_op_ns:.2f} ns")
        print(f"  Cache behavior: {result['estimated_cache_behavior']}")

    return data, metrics


def main():
    datasets_dir = PROJECT_ROOT / "benchmarks" / "datasets" / "synthetic"

    print("="*80)
    print("  PHASE 1 DATASET ANALYSIS")
    print("="*80)
    print("\nPurpose: Understand why Phase 1 found 40× slowdown")
    print("Expected: low_entropy should show extreme performance degradation")

    # Analyze each dataset
    datasets = {
        'low_entropy': datasets_dir / "low_entropy_1M.npy",
        'medium_entropy': datasets_dir / "medium_entropy_1M.npy",
        'high_entropy': datasets_dir / "high_entropy_1M.npy",
    }

    results = {}
    for name, path in datasets.items():
        if path.exists():
            data, metrics = analyze_dataset(path, name)
            results[name] = {'data': data, 'metrics': metrics}
        else:
            print(f"\n⚠️  WARNING: {path} not found")

    # Compare performance
    if ENGINE_AVAILABLE and results:
        print("\n" + "="*80)
        print("  PERFORMANCE COMPARISON")
        print("="*80)

        hw_metrics = HardwareMetrics()

        for name in ['low_entropy', 'medium_entropy', 'high_entropy']:
            if name not in results:
                continue

            data = results[name]['data']
            result = hw_metrics.measure_operation(tse.tadd, data, data, iterations=1000)

            time_per_op_ns = result['time_ns_per_iter']
            throughput_mops = (len(data) / (time_per_op_ns / 1e9)) / 1e6

            print(f"\n{name:>15}: {throughput_mops:>10.2f} Mops/s")

        # Calculate ratios
        if 'low_entropy' in results and 'high_entropy' in results:
            low_data = results['low_entropy']['data']
            high_data = results['high_entropy']['data']

            low_result = hw_metrics.measure_operation(tse.tadd, low_data, low_data, iterations=1000)
            high_result = hw_metrics.measure_operation(tse.tadd, high_data, high_data, iterations=1000)

            low_throughput = (len(low_data) / (low_result['time_ns_per_iter'] / 1e9)) / 1e6
            high_throughput = (len(high_data) / (high_result['time_ns_per_iter'] / 1e9)) / 1e6

            ratio = high_throughput / low_throughput

            print("\n" + "-"*80)
            print(f"Performance ratio (high/low): {ratio:.2f}×")

            if ratio > 10:
                print(f"⚠️  CONFIRMED: {ratio:.0f}× slowdown reproduced!")
            else:
                print(f"❓ DISCREPANCY: Only {ratio:.2f}× slowdown (expected 40×)")


if __name__ == '__main__':
    main()
