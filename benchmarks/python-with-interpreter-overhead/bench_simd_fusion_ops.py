#!/usr/bin/env python3
"""
bench_fusion.py - Benchmark fusion engine performance vs separate operations

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Measures the performance improvement of fused operations over separate operations.
Validates the claimed 1.5-11× speedup range from ternary_fusion.h documentation.
"""

import sys
import time
import numpy as np
from pathlib import Path
import argparse
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import ternary_simd_engine as base

# Fusion operations are integrated into main engine (ternary_simd_engine)
# Aliasing for compatibility with benchmark structure
fusion = base

# Test array sizes
DEFAULT_SIZES = [32, 100, 1_000, 10_000, 100_000, 1_000_000]
QUICK_SIZES = [32, 1_000, 100_000, 1_000_000]

WARMUP_ITERS = 50
MEASURE_ITERS = 200

def benchmark_operation(op_func, a, b, warmup=WARMUP_ITERS, measure=MEASURE_ITERS):
    """Benchmark a single operation"""
    # Warmup
    for _ in range(warmup):
        _ = op_func(a, b)

    # Measure
    times = []
    for _ in range(measure):
        start = time.perf_counter()
        result = op_func(a, b)
        end = time.perf_counter()
        times.append(end - start)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times),
    }

def benchmark_fused_vs_separate(operation, size, warmup, measure):
    """Compare fused vs separate operations for a given operation"""
    # Create test arrays
    np.random.seed(42)
    a = np.random.randint(0, 3, size, dtype=np.uint8)
    b = np.random.randint(0, 3, size, dtype=np.uint8)

    # Map operation names to functions
    operations = {
        'tnot_tadd': {
            'fused': fusion.fused_tnot_tadd,
            'separate': lambda a, b: base.tnot(base.tadd(a, b)),
            'description': 'tnot(tadd(a, b))'
        },
        'tnot_tmul': {
            'fused': fusion.fused_tnot_tmul,
            'separate': lambda a, b: base.tnot(base.tmul(a, b)),
            'description': 'tnot(tmul(a, b))'
        },
        'tnot_tmin': {
            'fused': fusion.fused_tnot_tmin,
            'separate': lambda a, b: base.tnot(base.tmin(a, b)),
            'description': 'tnot(tmin(a, b))'
        },
        'tnot_tmax': {
            'fused': fusion.fused_tnot_tmax,
            'separate': lambda a, b: base.tnot(base.tmax(a, b)),
            'description': 'tnot(tmax(a, b))'
        },
    }

    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}")

    op_funcs = operations[operation]

    # Benchmark separate operations
    separate_stats = benchmark_operation(op_funcs['separate'], a, b, warmup, measure)

    # Benchmark fused operation
    fused_stats = benchmark_operation(op_funcs['fused'], a, b, warmup, measure)

    # Calculate speedup
    speedup = separate_stats['mean'] / fused_stats['mean']

    return {
        'operation': operation,
        'description': op_funcs['description'],
        'size': size,
        'separate': separate_stats,
        'fused': fused_stats,
        'speedup': speedup,
    }

def format_time(seconds):
    """Format time in appropriate units"""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f} µs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.2f} s"

def print_benchmark_results(results):
    """Print benchmark results in readable format"""
    print("\n" + "="*80)
    print(f"  Array size: {results['size']:,} elements")
    print("="*80)
    print(f"\nOperation: {results['description']}")
    print(f"\nSeparate operations:")
    print(f"  Time: {format_time(results['separate']['mean'])} ± {format_time(results['separate']['std'])}")

    print(f"\nFused operation:")
    print(f"  Time: {format_time(results['fused']['mean'])} ± {format_time(results['fused']['std'])}")

    print(f"\nSpeedup: {results['speedup']:.2f}×")

    # Calculate coefficient of variation
    separate_cv = (results['separate']['std'] / results['separate']['mean']) * 100
    fused_cv = (results['fused']['std'] / results['fused']['mean']) * 100

    print(f"\nStability:")
    print(f"  Separate CV: {separate_cv:.1f}%")
    print(f"  Fused CV: {fused_cv:.1f}%")

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark fusion engine performance'
    )
    parser.add_argument('--quick', action='store_true',
                       help='Quick benchmark (fewer array sizes)')
    parser.add_argument('--operation', type=str,
                       choices=['tnot_tadd', 'tnot_tmul', 'tnot_tmin', 'tnot_tmax', 'all'],
                       default='all',
                       help='Operation to benchmark')
    parser.add_argument('--size', type=int,
                       help='Single array size to test')
    parser.add_argument('--output', type=str,
                       help='Output JSON file for results')

    args = parser.parse_args()

    print("="*80)
    print("  FUSION ENGINE PERFORMANCE BENCHMARK")
    print("="*80)
    print(f"\nWarmup iterations: {WARMUP_ITERS}")
    print(f"Measurement iterations: {MEASURE_ITERS}")

    # Determine test sizes
    if args.size:
        test_sizes = [args.size]
    elif args.quick:
        test_sizes = QUICK_SIZES
    else:
        test_sizes = DEFAULT_SIZES

    # Determine operations to test
    if args.operation == 'all':
        operations = ['tnot_tadd', 'tnot_tmul', 'tnot_tmin', 'tnot_tmax']
    else:
        operations = [args.operation]

    print(f"Test sizes: {test_sizes}")
    print(f"Operations: {operations}")

    # Run benchmarks
    all_results = []

    for operation in operations:
        for size in test_sizes:
            print(f"\nBenchmarking {operation} with size {size:,}...")
            results = benchmark_fused_vs_separate(operation, size, WARMUP_ITERS, MEASURE_ITERS)
            print_benchmark_results(results)
            all_results.append(results)

    # Summary
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80)

    for operation in operations:
        op_results = [r for r in all_results if r['operation'] == operation]
        speedups = [r['speedup'] for r in op_results]

        print(f"\n{operation}:")
        print(f"  Speedup range: {min(speedups):.2f}× - {max(speedups):.2f}×")
        print(f"  Average speedup: {np.mean(speedups):.2f}×")
        print(f"  Median speedup: {np.median(speedups):.2f}×")

    # Overall summary
    all_speedups = [r['speedup'] for r in all_results]
    print(f"\nOverall:")
    print(f"  Speedup range: {min(all_speedups):.2f}× - {max(all_speedups):.2f}×")
    print(f"  Average speedup: {np.mean(all_speedups):.2f}×")
    print(f"  Median speedup: {np.median(all_speedups):.2f}×")

    # Validation against claims
    print("\n" + "="*80)
    print("  VALIDATION AGAINST DOCUMENTED CLAIMS")
    print("="*80)

    claims = {
        'tnot_tadd': {'min': 1.62, 'max': 15.52, 'avg': 1.94},
        'tnot_tmul': {'min': 1.53, 'max': 1.86, 'avg': 1.71},
        'tnot_tmin': {'min': 1.61, 'max': 11.26, 'avg': 4.06},
        'tnot_tmax': {'min': 1.65, 'max': 9.50, 'avg': 3.68},
    }

    for operation in operations:
        op_results = [r for r in all_results if r['operation'] == operation]
        speedups = [r['speedup'] for r in op_results]
        measured_min = min(speedups)
        measured_max = max(speedups)
        measured_avg = np.mean(speedups)

        if operation in claims:
            claim = claims[operation]
            print(f"\n{operation}:")
            print(f"  Documented range: {claim['min']:.2f}× - {claim['max']:.2f}× (avg {claim['avg']:.2f}×)")
            print(f"  Measured range:   {measured_min:.2f}× - {measured_max:.2f}× (avg {measured_avg:.2f}×)")

            # Check if measured is within reasonable range
            if measured_min >= claim['min'] * 0.8:  # Allow 20% variance
                print(f"  ✓ Minimum speedup validated")
            else:
                print(f"  ✗ Minimum speedup below documented claim")

    print("\n" + "="*80)

    # Save results if output file specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = []
            for r in all_results:
                json_r = {
                    'operation': r['operation'],
                    'description': r['description'],
                    'size': int(r['size']),
                    'speedup': float(r['speedup']),
                    'separate_mean': float(r['separate']['mean']),
                    'fused_mean': float(r['fused']['mean']),
                }
                json_results.append(json_r)

            json.dump(json_results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    return 0

if __name__ == '__main__':
    sys.exit(main())
