"""
bench_with_load_context.py - Benchmarks with System Load Monitoring

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Runs the complete benchmark suite while monitoring system load to provide
context for reproducibility. Records external application activity
(browsers, Docker, etc.) that may affect results.

USAGE:
    python benchmarks/bench_with_load_context.py              # Full suite
    python benchmarks/bench_with_load_context.py --quick      # Quick mode
    python benchmarks/bench_with_load_context.py --report     # System report only

OUTPUT:
    - JSON results with load context in benchmarks/results/load_aware/
    - Human-readable summary with reproducibility notes
"""

import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Callable
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.utils.system_load_monitor import (
    SystemLoadMonitor,
    LoadAwareBenchmark,
    print_system_report
)

# Import benchmark targets
try:
    import ternary_simd_engine as tc
    HAS_CORE_ENGINE = True
except ImportError:
    HAS_CORE_ENGINE = False
    print("WARNING: ternary_simd_engine not found")

try:
    import ternary_backend as tb
    tb.init()
    HAS_BACKEND = True
except ImportError:
    HAS_BACKEND = False
    print("WARNING: ternary_backend not found")


# Test configurations
QUICK_SIZES = [1_000, 100_000, 1_000_000]
FULL_SIZES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
OPERATIONS = ['tadd', 'tmul', 'tmin', 'tmax', 'tnot']
WARMUP_ITERATIONS = 50
MEASURED_ITERATIONS = 500

# Trit encoding
MINUS_ONE, ZERO, PLUS_ONE = 0b00, 0b01, 0b10


def generate_test_data(size: int, seed: int = 42):
    """Generate reproducible test data"""
    np.random.seed(seed)
    a = np.random.choice([MINUS_ONE, ZERO, PLUS_ONE], size=size).astype(np.uint8)
    b = np.random.choice([MINUS_ONE, ZERO, PLUS_ONE], size=size).astype(np.uint8)
    return a, b


def benchmark_operation(func: Callable, a: np.ndarray, b: np.ndarray = None,
                        warmup: int = WARMUP_ITERATIONS,
                        iterations: int = MEASURED_ITERATIONS) -> Dict:
    """Benchmark a single operation with statistics"""
    # Warmup
    for _ in range(warmup):
        if b is not None:
            _ = func(a, b)
        else:
            _ = func(a)

    # Measured runs - collect individual timings
    timings = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        if b is not None:
            _ = func(a, b)
        else:
            _ = func(a)
        end = time.perf_counter_ns()
        timings.append(end - start)

    # Calculate statistics
    timings_arr = np.array(timings)
    total_ns = timings_arr.sum()
    mean_ns = timings_arr.mean()
    std_ns = timings_arr.std()
    min_ns = timings_arr.min()
    max_ns = timings_arr.max()
    median_ns = np.median(timings_arr)

    # Throughput based on median (more robust)
    elements_per_iter = len(a)
    throughput_mops = (elements_per_iter / (median_ns / 1e9)) / 1e6

    return {
        'total_ns': float(total_ns),
        'mean_ns': float(mean_ns),
        'std_ns': float(std_ns),
        'min_ns': float(min_ns),
        'max_ns': float(max_ns),
        'median_ns': float(median_ns),
        'cv_percent': float((std_ns / mean_ns) * 100) if mean_ns > 0 else 0,
        'iterations': iterations,
        'throughput_mops': float(throughput_mops),
        'ns_per_element': float(median_ns / elements_per_iter)
    }


def run_core_benchmarks(sizes: List[int], monitor: SystemLoadMonitor) -> Dict:
    """Run core engine benchmarks with load context"""
    if not HAS_CORE_ENGINE:
        return {'error': 'ternary_simd_engine not available'}

    results = {
        'module': 'ternary_simd_engine',
        'sizes': sizes,
        'operations': {},
        'load_samples': []
    }

    total_tests = len(sizes) * len(OPERATIONS)
    current_test = 0

    for size in sizes:
        a, b = generate_test_data(size)

        for op_name in OPERATIONS:
            current_test += 1
            print(f"  [{current_test}/{total_tests}] {op_name} @ {size:,} elements...", end=" ")

            # Sample load before operation
            load_before = monitor.get_snapshot()

            # Get function and run benchmark
            func = getattr(tc, op_name)
            if op_name == 'tnot':
                result = benchmark_operation(func, a)
            else:
                result = benchmark_operation(func, a, b)

            # Sample load after operation
            load_after = monitor.get_snapshot()

            # Store result with load context
            key = f"{op_name}_{size}"
            results['operations'][key] = {
                **result,
                'operation': op_name,
                'size': size,
                'load_before': {
                    'cpu_percent': load_before['cpu_percent_total'],
                    'load_score': load_before['load_score'],
                    'classification': load_before['load_classification']
                },
                'load_after': {
                    'cpu_percent': load_after['cpu_percent_total'],
                    'load_score': load_after['load_score']
                }
            }

            print(f"{result['throughput_mops']:.2f} Mops/s (CV: {result['cv_percent']:.1f}%)")

    return results


def run_backend_benchmarks(sizes: List[int], monitor: SystemLoadMonitor) -> Dict:
    """Run backend benchmarks with load context"""
    if not HAS_BACKEND:
        return {'error': 'ternary_backend not available'}

    # Use AVX2_v2 backend
    tb.set_backend('AVX2_v2')

    results = {
        'module': 'ternary_backend',
        'backend': 'AVX2_v2',
        'sizes': sizes,
        'operations': {},
        'fusion_operations': {}
    }

    # Regular operations
    for size in sizes:
        a, b = generate_test_data(size)

        for op_name in OPERATIONS:
            print(f"  Backend {op_name} @ {size:,}...", end=" ")

            func = getattr(tb, op_name)
            if op_name == 'tnot':
                result = benchmark_operation(func, a)
            else:
                result = benchmark_operation(func, a, b)

            key = f"{op_name}_{size}"
            results['operations'][key] = {
                **result,
                'operation': op_name,
                'size': size
            }

            print(f"{result['throughput_mops']:.2f} Mops/s")

    # Fusion operations
    fusion_ops = ['fused_tnot_tadd', 'fused_tnot_tmul', 'fused_tnot_tmin', 'fused_tnot_tmax']

    for size in sizes:
        a, b = generate_test_data(size)

        for op_name in fusion_ops:
            print(f"  Backend {op_name} @ {size:,}...", end=" ")

            func = getattr(tb, op_name)
            result = benchmark_operation(func, a, b)

            key = f"{op_name}_{size}"
            results['fusion_operations'][key] = {
                **result,
                'operation': op_name,
                'size': size
            }

            print(f"{result['throughput_mops']:.2f} Mops/s")

    return results


def calculate_summary_stats(results: Dict) -> Dict:
    """Calculate summary statistics from benchmark results"""
    summary = {
        'peak_throughput_by_op': {},
        'avg_throughput_by_op': {},
        'cv_summary': {}
    }

    if 'operations' in results:
        for op_name in OPERATIONS:
            op_results = [
                v for k, v in results['operations'].items()
                if k.startswith(op_name)
            ]
            if op_results:
                throughputs = [r['throughput_mops'] for r in op_results]
                cvs = [r['cv_percent'] for r in op_results]

                summary['peak_throughput_by_op'][op_name] = max(throughputs)
                summary['avg_throughput_by_op'][op_name] = sum(throughputs) / len(throughputs)
                summary['cv_summary'][op_name] = {
                    'min': min(cvs),
                    'max': max(cvs),
                    'avg': sum(cvs) / len(cvs)
                }

    return summary


def run_full_suite(quick: bool = False) -> Dict:
    """Run complete benchmark suite with load monitoring"""
    monitor = SystemLoadMonitor()
    sizes = QUICK_SIZES if quick else FULL_SIZES

    print("="*70)
    print("  LOAD-AWARE BENCHMARK SUITE")
    print("="*70)

    # Initial system report
    print("\n--- Initial System State ---")
    initial_snapshot = monitor.get_snapshot()
    print(monitor.get_load_summary())

    if initial_snapshot['recommendations']:
        print("\nRecommendations for cleaner results:")
        for rec in initial_snapshot['recommendations']:
            print(f"  - {rec}")

    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'quick_mode': quick,
            'sizes': sizes,
            'warmup_iterations': WARMUP_ITERATIONS,
            'measured_iterations': MEASURED_ITERATIONS
        },
        'system_context': {
            'initial': {
                'load_classification': initial_snapshot['load_classification'],
                'load_score': initial_snapshot['load_score'],
                'cpu_percent': initial_snapshot['cpu_percent_total'],
                'memory_percent': initial_snapshot['memory_percent'],
                'high_load_apps': [p['name'] for p in initial_snapshot['high_load_processes']],
                'load_by_category': initial_snapshot['load_by_category']
            },
            'platform': initial_snapshot['platform_info']
        },
        'benchmarks': {}
    }

    # Run core engine benchmarks
    print("\n" + "="*70)
    print("  CORE ENGINE BENCHMARKS (ternary_simd_engine)")
    print("="*70 + "\n")

    core_results = run_core_benchmarks(sizes, monitor)
    results['benchmarks']['core_engine'] = core_results

    if 'operations' in core_results:
        summary = calculate_summary_stats(core_results)
        results['benchmarks']['core_engine']['summary'] = summary

    # Run backend benchmarks
    print("\n" + "="*70)
    print("  BACKEND BENCHMARKS (ternary_backend AVX2_v2)")
    print("="*70 + "\n")

    backend_results = run_backend_benchmarks(sizes, monitor)
    results['benchmarks']['backend'] = backend_results

    if 'operations' in backend_results:
        summary = calculate_summary_stats(backend_results)
        results['benchmarks']['backend']['summary'] = summary

    # Final system snapshot
    print("\n--- Final System State ---")
    final_snapshot = monitor.get_snapshot()
    print(monitor.get_load_summary())

    results['system_context']['final'] = {
        'load_classification': final_snapshot['load_classification'],
        'load_score': final_snapshot['load_score'],
        'cpu_percent': final_snapshot['cpu_percent_total'],
        'memory_percent': final_snapshot['memory_percent']
    }

    # Calculate load stability
    load_change = final_snapshot['load_score'] - initial_snapshot['load_score']
    results['system_context']['load_stability'] = {
        'score_change': round(load_change, 2),
        'stable': abs(load_change) < 15,
        'note': 'Stable' if abs(load_change) < 15 else 'Load changed during benchmark'
    }

    # Reproducibility assessment
    results['reproducibility'] = assess_reproducibility(results)

    return results


def assess_reproducibility(results: Dict) -> Dict:
    """Assess overall reproducibility of benchmark results"""
    context = results['system_context']
    initial_load = context['initial']['load_classification']

    assessment = {
        'rating': 'unknown',
        'confidence': 0,
        'notes': []
    }

    # Base confidence on load level
    if initial_load == 'low':
        assessment['rating'] = 'excellent'
        assessment['confidence'] = 95
        assessment['notes'].append("Low system load - results highly reproducible")
    elif initial_load == 'medium':
        assessment['rating'] = 'good'
        assessment['confidence'] = 80
        assessment['notes'].append("Medium system load - results should be reproducible within 10%")
    elif initial_load == 'high':
        assessment['rating'] = 'fair'
        assessment['confidence'] = 60
        assessment['notes'].append("High system load - results may vary by 20-30%")
    else:
        assessment['rating'] = 'poor'
        assessment['confidence'] = 40
        assessment['notes'].append("Very high system load - results may vary significantly")

    # Adjust for stability
    if not context.get('load_stability', {}).get('stable', True):
        assessment['confidence'] -= 10
        assessment['notes'].append("System load changed during benchmark")

    # Note high-load apps
    high_load_apps = context['initial'].get('high_load_apps', [])
    if high_load_apps:
        assessment['notes'].append(f"Active high-load apps: {', '.join(high_load_apps[:5])}")

    # Specific category impacts
    load_by_cat = context['initial'].get('load_by_category', {})
    if load_by_cat.get('browsers', 0) > 10:
        assessment['notes'].append(f"Browser CPU usage ({load_by_cat['browsers']:.1f}%) may affect results")
    if load_by_cat.get('docker', 0) > 5:
        assessment['notes'].append("Docker activity detected")

    return assessment


def print_summary(results: Dict):
    """Print human-readable summary"""
    print("\n" + "="*70)
    print("  BENCHMARK SUMMARY")
    print("="*70)

    # System context
    ctx = results['system_context']
    print(f"\nSystem Load Context:")
    print(f"  Initial: {ctx['initial']['load_classification'].upper()} "
          f"(score: {ctx['initial']['load_score']:.0f}/100)")
    print(f"  CPU: {ctx['initial']['cpu_percent']:.1f}% | "
          f"RAM: {ctx['initial']['memory_percent']:.1f}%")

    if ctx['initial']['high_load_apps']:
        print(f"  High-load apps: {', '.join(ctx['initial']['high_load_apps'][:5])}")

    # Core engine results
    if 'core_engine' in results['benchmarks']:
        core = results['benchmarks']['core_engine']
        if 'summary' in core:
            print(f"\nCore Engine Peak Throughput:")
            for op, mops in core['summary']['peak_throughput_by_op'].items():
                print(f"  {op:8s}: {mops:10.2f} Mops/s")

    # Backend results
    if 'backend' in results['benchmarks']:
        backend = results['benchmarks']['backend']
        if 'summary' in backend:
            print(f"\nBackend (AVX2_v2) Peak Throughput:")
            for op, mops in backend['summary']['peak_throughput_by_op'].items():
                print(f"  {op:8s}: {mops:10.2f} Mops/s")

    # Reproducibility
    repro = results.get('reproducibility', {})
    print(f"\nReproducibility Assessment:")
    print(f"  Rating: {repro.get('rating', 'unknown').upper()}")
    print(f"  Confidence: {repro.get('confidence', 0)}%")
    for note in repro.get('notes', []):
        print(f"  - {note}")

    print("\n" + "="*70)


def save_results(results: Dict, output_dir: Path = None) -> Path:
    """Save results to JSON"""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "benchmarks" / "results" / "load_aware"

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"bench_load_aware_{timestamp}.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark suite with system load monitoring'
    )
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmark with fewer sizes')
    parser.add_argument('--report', action='store_true',
                       help='Print system report only, no benchmarks')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results')

    args = parser.parse_args()

    if args.report:
        print_system_report()
        return

    # Run full suite
    results = run_full_suite(quick=args.quick)

    # Print summary
    print_summary(results)

    # Save results
    output_dir = Path(args.output) if args.output else None
    save_results(results, output_dir)


if __name__ == '__main__':
    main()
