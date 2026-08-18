"""
bench_simd_core_ops.py - Production-grade Python benchmark suite for Ternary Engine

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
import platform
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent  # fixed 2026-08-12: was 1 .parent short of repo root
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

try:
    import ternary_simd_engine as tc
    HAS_TERNARY_ENGINE = True
except ImportError:
    HAS_TERNARY_ENGINE = False
    print("WARNING: ternary_simd_engine not found. Build the module first:")
    print("  python build.py")
    sys.exit(1)

from benchmark_framework import compute_timing_statistics  # noqa: E402

# Benchmark configuration
TEST_SIZES = [32, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
TEST_SIZES_QUICK = [32, 1_000, 100_000, 1_000_000]
OPERATIONS = ['tadd', 'tmul', 'tmin', 'tmax', 'tnot']
WARMUP_ITERATIONS = 100

# Statistical-rigor upgrade (2026-08-18, CLAUDE.md gap #8 follow-up): this
# script used to bracket ONE block of 1,000 back-to-back calls with a single
# perf_counter_ns() pair and divide -- a single sample, so no variance/CV/CI
# was ever computable, unlike bench_fair_baseline.py and (as of the same
# session) bench_simd_fusion_ops.py, which both take repeated independent
# timing samples via BenchmarkRunner's statistics engine. Replaced with the
# same idea: BATCH_ITERATIONS calls form one timed block (amortizing
# per-call/dispatch overhead, same purpose the old MEASURED_ITERATIONS=1000
# single block served), and MEASUREMENT_RUNS independent blocks are each
# timed separately so a median/stdev/CV/95%-CI can be computed across them --
# this is what actually protects against the thermal/clock-drift bias this
# project's own "interleaved_timing" convention warns about, which a single
# block can never reveal regardless of how large it is.
#
# Deliberately NOT reported as a re-validation of the "35,042 Mops/s peak
# throughput" headline figure this script is the source of: that number was
# measured on Windows x64 (the only platform this project formally validates
# production claims on, per CLAUDE.md), while this session's numbers are
# Linux-local (see run_benchmark_suite()'s printed platform info and the
# JSON metadata). This is a methodology upgrade to the script itself,
# runnable and meaningful on any platform, not a new headline claim.
BATCH_ITERATIONS = 200
MEASUREMENT_RUNS = 10
# Kept for informational continuity with prior runs' metadata (same order of
# magnitude as the old single-block MEASURED_ITERATIONS=1000: BATCH_ITERATIONS
# * MEASUREMENT_RUNS = 2,000 total calls per cell, vs. 1,000 before).
MEASURED_ITERATIONS = BATCH_ITERATIONS * MEASUREMENT_RUNS

# Trit encoding
MINUS_ONE = 0b00
ZERO = 0b01
PLUS_ONE = 0b10


def get_cpu_info() -> Dict:
    """Collect CPU and system information for benchmark metadata"""
    info = {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'platform_version': platform.version(),
        'architecture': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
    }

    # Get CPU count
    try:
        info['cpu_count_logical'] = os.cpu_count()
    except:
        info['cpu_count_logical'] = 'unknown'

    # Get OMP_NUM_THREADS if set, otherwise set to cpu_count for consistency
    omp_threads = os.environ.get('OMP_NUM_THREADS')
    if omp_threads is None:
        # Set to logical CPU count for consistent results
        cpu_count = os.cpu_count()
        if cpu_count:
            os.environ['OMP_NUM_THREADS'] = str(cpu_count)
            info['omp_num_threads'] = cpu_count
            info['omp_threads_auto_set'] = True
        else:
            info['omp_num_threads'] = 'default'
            info['omp_threads_auto_set'] = False
    else:
        info['omp_num_threads'] = omp_threads
        info['omp_threads_auto_set'] = False

    # Platform-specific CPU detection
    if platform.system() == 'Linux':
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                # Extract CPU model
                for line in cpuinfo.split('\n'):
                    if 'model name' in line:
                        info['cpu_model'] = line.split(':')[1].strip()
                        break
                # Check for AVX2 support
                info['has_avx2'] = 'avx2' in cpuinfo.lower()
        except:
            info['cpu_model'] = 'unknown'
            info['has_avx2'] = 'unknown'
    elif platform.system() == 'Windows':
        try:
            import subprocess
            result = subprocess.run(['wmic', 'cpu', 'get', 'name'],
                                  capture_output=True, text=True, timeout=2)
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                info['cpu_model'] = lines[1].strip()
        except:
            info['cpu_model'] = 'unknown'
        info['has_avx2'] = 'unknown'  # Would need CPUID check
    elif platform.system() == 'Darwin':
        try:
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                  capture_output=True, text=True, timeout=2)
            info['cpu_model'] = result.stdout.strip()

            # Check for AVX2
            result = subprocess.run(['sysctl', 'machdep.cpu.features'],
                                  capture_output=True, text=True, timeout=2)
            info['has_avx2'] = 'AVX2' in result.stdout
        except:
            info['cpu_model'] = 'unknown'
            info['has_avx2'] = 'unknown'

    return info


class OpBenchmarkResult:
    """Container for a single benchmark result.

    Named distinctly from benchmark_framework.BenchmarkResult (a different,
    baseline-vs-optimized-pair shape) to avoid confusion now that this file
    imports from that module too.

    time_ns/iterations represent the MEDIAN block (one of MEASUREMENT_RUNS
    independent BATCH_ITERATIONS-call blocks) -- median-of-blocks, not
    mean-of-all-calls, per this project's own median-based statistical
    convention (see bench_fair_baseline.py). stats, if provided, carries the
    full compute_timing_statistics() output across all blocks' per-call
    times, for the CV/stdev/CI fields to_dict() adds.
    """
    def __init__(self, operation: str, size: int, time_ns: float, iterations: int,
                 stats: Dict[str, float] = None, measurement_runs: int = 1):
        self.operation = operation
        self.size = size
        self.time_ns = time_ns
        self.iterations = iterations
        self.time_per_op = time_ns / iterations
        self.time_per_elem = self.time_per_op / size
        self.throughput_mops = (size * iterations) / (time_ns / 1e9) / 1e6
        self.stats = stats
        self.measurement_runs = measurement_runs

    def to_dict(self) -> Dict:
        d = {
            'operation': self.operation,
            'size': self.size,
            'time_ns_total': self.time_ns,
            'iterations': self.iterations,
            'time_ns_per_op': self.time_per_op,
            'time_ns_per_elem': self.time_per_elem,
            'throughput_mops': self.throughput_mops,
        }
        if self.stats is not None:
            # Additive fields only -- the 7 keys above are unchanged so
            # benchmark_validator.py, bench_regression_detect.py, and
            # run_all_benchmarks.py (all of which read this schema) keep
            # working without modification.
            d['measurement_runs'] = self.measurement_runs
            d['cv_percent'] = self.stats['cv']
            d['stdev_ns_per_op'] = self.stats['stdev']
            # 95% CI on time -> Mops/s is inversely proportional to time, so
            # the CI bounds flip: the time upper bound gives the Mops/s
            # lower bound and vice versa.
            ci_lo_ns, ci_hi_ns = self.stats['ci_lower'], self.stats['ci_upper']
            if ci_hi_ns > 0:
                d['throughput_mops_ci_lower'] = (self.size / (ci_hi_ns / 1e9)) / 1e6
            if ci_lo_ns > 0:
                d['throughput_mops_ci_upper'] = (self.size / (ci_lo_ns / 1e9)) / 1e6
        return d


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
                        batch: int = BATCH_ITERATIONS,
                        runs: int = MEASUREMENT_RUNS) -> Dict[str, float]:
    """Benchmark a single operation across `runs` independent timed blocks
    of `batch` back-to-back calls each, returning per-call-time statistics
    (compute_timing_statistics()'s dict, units: ns per call).

    Replaces the old single-block design (module docstring above has the
    full rationale): a lone block, however large, can never distinguish
    "consistently fast" from "fast on average but drifted mid-run" --
    independent blocks can, since a later block affected by thermal
    throttling or scheduler noise shows up as a high-CV outlier in the
    returned stats instead of silently blending into one undifferentiated
    total.
    """
    # Warmup (once, before the first timed block)
    for _ in range(warmup):
        if b is not None:
            _ = func(a, b)
        else:
            _ = func(a)

    per_call_times_ns = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        for _ in range(batch):
            if b is not None:
                _ = func(a, b)
            else:
                _ = func(a)
        end = time.perf_counter_ns()
        per_call_times_ns.append((end - start) / batch)

    return compute_timing_statistics(per_call_times_ns)


def run_benchmark_suite(sizes: List[int], verbose: bool = True) -> Dict:
    """Run complete benchmark suite"""
    # Collect hardware info
    hw_info = get_cpu_info()

    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'module': 'ternary_simd_engine',
            'numpy_version': np.__version__,
            'test_sizes': sizes,
            'warmup_iterations': WARMUP_ITERATIONS,
            'measured_iterations': MEASURED_ITERATIONS,  # BATCH_ITERATIONS * MEASUREMENT_RUNS, kept for continuity
            'batch_iterations': BATCH_ITERATIONS,
            'measurement_runs': MEASUREMENT_RUNS,
            'hardware': hw_info,
        },
        'results_optimized': [],
        'results_baseline': [],
    }

    if verbose:
        print("=" * 80)
        print("  TERNARY ENGINE BENCHMARK SUITE")
        print("=" * 80)
        print(f"\nHardware Info:")
        print(f"  CPU: {hw_info.get('cpu_model', 'unknown')}")
        print(f"  Architecture: {hw_info.get('architecture', 'unknown')}")
        print(f"  Logical CPUs: {hw_info.get('cpu_count_logical', 'unknown')}")
        print(f"  AVX2 Support: {hw_info.get('has_avx2', 'unknown')}")
        omp_note = " (auto-set)" if hw_info.get('omp_threads_auto_set', False) else ""
        print(f"  OMP Threads: {hw_info.get('omp_num_threads', 'default')}{omp_note}")

        # Warn if AVX2 not detected
        if hw_info.get('has_avx2') == False:
            print("\n  WARNING: AVX2 not detected! Performance will be severely degraded.")
            print("  This module requires AVX2 support (Intel Haswell 2013+ or AMD Excavator 2015+)")

        # Performance consistency warnings
        print(f"\nBenchmark Configuration:")
        print(f"  Test sizes: {sizes}")
        print(f"  Warmup iterations: {WARMUP_ITERATIONS}")
        print(f"  Measurement: {MEASUREMENT_RUNS} independent blocks x {BATCH_ITERATIONS} calls/block "
              f"({MEASURED_ITERATIONS} total) -- reports median + CV across blocks")

        print(f"\nPerformance Notes:")
        print(f"  - Results may vary with CPU frequency scaling and power states")
        print(f"  - For most consistent results, disable CPU frequency scaling")
        print(f"  - Close other applications to minimize background interference")

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
                stats = benchmark_operation(tc_func, a)
            else:
                stats = benchmark_operation(tc_func, a, b)

            result_opt = OpBenchmarkResult(
                op_name, size,
                time_ns=stats['median'] * BATCH_ITERATIONS,
                iterations=BATCH_ITERATIONS,
                stats=stats,
                measurement_runs=MEASUREMENT_RUNS,
            )
            results['results_optimized'].append(result_opt.to_dict())

            # Benchmark Python baseline (only for smaller sizes to avoid timeout).
            # batch=20/runs=5 keeps the same 100-call total budget the old
            # single-block warmup=10/iterations=100 used, just split into
            # independent blocks so a CV/stats dict is computable here too.
            if size <= 10_000:
                py_batch, py_runs = 20, 5
                if op_name == 'tnot':
                    stats_py = benchmark_operation(py_func, a, warmup=10, batch=py_batch, runs=py_runs)
                else:
                    stats_py = benchmark_operation(py_func, a, b, warmup=10, batch=py_batch, runs=py_runs)

                result_py = OpBenchmarkResult(
                    op_name, size,
                    time_ns=stats_py['median'] * py_batch,
                    iterations=py_batch,
                    stats=stats_py,
                    measurement_runs=py_runs,
                )
                results['results_baseline'].append(result_py.to_dict())

                speedup = result_py.time_per_elem / result_opt.time_per_elem
            else:
                speedup = None

            if verbose:
                speedup_str = f"{speedup:.1f}x" if speedup else "N/A"
                cv_note = f" (cv={result_opt.stats['cv']:.1f}%)" if result_opt.stats else ""
                print(f"  {op_name:8s} | {result_opt.throughput_mops:8.2f} Mops/s{cv_note} | "
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

    # High-CV warning, matching bench_fair_baseline.py's convention: a cell
    # with high block-to-block variance means "rerun on an idle machine"
    # before citing that specific number, not "the run failed."
    high_cv = [r for r in results['results_optimized'] if r.get('cv_percent', 0) > 15.0]
    if high_cv:
        print(f"\n[WARN] {len(high_cv)} cell(s) have CV > 15% -- rerun on an idle "
              f"machine before publishing these numbers:")
        for r in high_cv:
            print(f"  {r['operation']:8s} @ {r['size']:>10,}: cv={r['cv_percent']:.1f}%")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Ternary Engine Benchmark Suite')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmark with fewer sizes')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results (default: benchmarks/results)')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    parser.add_argument('--no-unicode', action='store_true',
                       help='Avoid Unicode characters in output (for CI compatibility)')

    args = parser.parse_args()

    # Set environment variable for Unicode handling
    if args.no_unicode:
        os.environ['BENCHMARK_NO_UNICODE'] = '1'

    sizes = TEST_SIZES_QUICK if args.quick else TEST_SIZES
    verbose = not args.quiet

    # Run benchmark suite
    results = run_benchmark_suite(sizes, verbose=verbose)

    # Save results (use default path if not specified)
    if args.output:
        output_dir = PROJECT_ROOT / args.output
    else:
        output_dir = PROJECT_ROOT / "benchmarks" / "results"
    save_results(results, output_dir)

    # Print summary
    if verbose:
        print_summary(results)


if __name__ == '__main__':
    main()
