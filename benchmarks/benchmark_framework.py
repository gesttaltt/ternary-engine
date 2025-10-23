"""
benchmark_framework.py - Rigorous Statistical Benchmark Framework

Truth-first benchmarking with mandatory variance reporting.

Usage:
    from benchmark_framework import BenchmarkRunner, BenchmarkConfig

    config = BenchmarkConfig(
        sizes=[1_000, 10_000, 100_000],
        warmup_runs=20,
        measurement_runs=100
    )

    runner = BenchmarkRunner(config)
    result = runner.benchmark("Operation Name", baseline_fn, optimized_fn, array_size)
    result.print_summary()
"""

import time
import statistics
import math
from dataclasses import dataclass
from typing import Callable, List, Dict, Any
import numpy as np


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""
    sizes: List[int]
    warmup_runs: int = 20
    measurement_runs: int = 100
    max_cv_percent: float = 20.0  # Maximum coefficient of variation for "stable" result
    min_speedup: float = 1.2  # Minimum speedup to claim success


@dataclass
class BenchmarkResult:
    """Statistical results from a single benchmark"""
    name: str
    size: int
    baseline_median_ns: float
    baseline_mean_ns: float
    baseline_stdev_ns: float
    baseline_cv_percent: float
    optimized_median_ns: float
    optimized_mean_ns: float
    optimized_stdev_ns: float
    optimized_cv_percent: float
    speedup: float
    ci_95_lower: float
    ci_95_upper: float
    is_stable: bool
    meets_target: bool

    def print_summary(self):
        """Print detailed benchmark summary"""
        print(f"\n{'='*80}")
        print(f"  {self.name} ({self.size:,} elements)")
        print(f"{'='*80}")

        print(f"\nBaseline:")
        print(f"  Median:  {self.baseline_median_ns/1000:12.2f} μs")
        print(f"  Mean:    {self.baseline_mean_ns/1000:12.2f} μs")
        print(f"  Stdev:   {self.baseline_stdev_ns/1000:12.2f} μs")
        print(f"  CV:      {self.baseline_cv_percent:12.1f}%")

        print(f"\nOptimized:")
        print(f"  Median:  {self.optimized_median_ns/1000:12.2f} μs")
        print(f"  Mean:    {self.optimized_mean_ns/1000:12.2f} μs")
        print(f"  Stdev:   {self.optimized_stdev_ns/1000:12.2f} μs")
        print(f"  CV:      {self.optimized_cv_percent:12.1f}%")

        print(f"\nSpeedup: {self.speedup:.2f}×")
        print(f"95% CI:  [{self.ci_95_lower:.2f}×, {self.ci_95_upper:.2f}×]")

        stability = "✓ Stable" if self.is_stable else "⚠ Unstable"
        target = "✓ Meets target" if self.meets_target else "✗ Below target"

        print(f"\nQuality:")
        print(f"  Stability: {stability}")
        print(f"  Target:    {target}")

    def to_dict(self) -> Dict[str, Any]:
        """Export results as dictionary"""
        return {
            'name': self.name,
            'size': self.size,
            'baseline': {
                'median_us': self.baseline_median_ns / 1000,
                'mean_us': self.baseline_mean_ns / 1000,
                'stdev_us': self.baseline_stdev_ns / 1000,
                'cv_percent': self.baseline_cv_percent
            },
            'optimized': {
                'median_us': self.optimized_median_ns / 1000,
                'mean_us': self.optimized_mean_ns / 1000,
                'stdev_us': self.optimized_stdev_ns / 1000,
                'cv_percent': self.optimized_cv_percent
            },
            'speedup': self.speedup,
            'ci_95': [self.ci_95_lower, self.ci_95_upper],
            'is_stable': self.is_stable,
            'meets_target': self.meets_target
        }


class BenchmarkRunner:
    """Rigorous benchmark runner with statistical validation"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def _compute_statistics(self, times_ns: List[float]) -> Dict[str, float]:
        """Compute comprehensive statistics from timing samples"""
        median = statistics.median(times_ns)
        mean = statistics.mean(times_ns)
        stdev = statistics.stdev(times_ns) if len(times_ns) > 1 else 0.0
        cv = (stdev / mean * 100) if mean > 0 else 0.0

        # 95% confidence interval for the mean
        n = len(times_ns)
        ci_margin = 1.96 * (stdev / math.sqrt(n)) if n > 1 else 0.0
        ci_lower = mean - ci_margin
        ci_upper = mean + ci_margin

        return {
            'median': median,
            'mean': mean,
            'stdev': stdev,
            'cv': cv,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }

    def benchmark(
        self,
        name: str,
        baseline_fn: Callable[[], Any],
        optimized_fn: Callable[[], Any],
        size: int,
        target_speedup: float = None
    ) -> BenchmarkResult:
        """
        Run rigorous benchmark comparing baseline vs optimized.

        Args:
            name: Descriptive name for the operation
            baseline_fn: Function to benchmark (baseline)
            optimized_fn: Function to benchmark (optimized)
            size: Array size being tested
            target_speedup: Minimum expected speedup (default from config)

        Returns:
            BenchmarkResult with comprehensive statistics
        """
        if target_speedup is None:
            target_speedup = self.config.min_speedup

        # Warmup
        for _ in range(self.config.warmup_runs):
            _ = baseline_fn()
            _ = optimized_fn()

        # Measure baseline
        times_baseline = []
        for _ in range(self.config.measurement_runs):
            start = time.perf_counter_ns()
            _ = baseline_fn()
            times_baseline.append(time.perf_counter_ns() - start)

        # Measure optimized
        times_optimized = []
        for _ in range(self.config.measurement_runs):
            start = time.perf_counter_ns()
            _ = optimized_fn()
            times_optimized.append(time.perf_counter_ns() - start)

        # Compute statistics
        stats_baseline = self._compute_statistics(times_baseline)
        stats_optimized = self._compute_statistics(times_optimized)

        # Compute speedup
        speedup = stats_baseline['median'] / stats_optimized['median']

        # Compute 95% CI for speedup
        # Using ratio of means with delta method approximation
        speedup_lower = stats_baseline['ci_lower'] / stats_optimized['ci_upper']
        speedup_upper = stats_baseline['ci_upper'] / stats_optimized['ci_lower']

        # Quality checks
        is_stable = (
            stats_baseline['cv'] < self.config.max_cv_percent and
            stats_optimized['cv'] < self.config.max_cv_percent
        )

        meets_target = speedup >= target_speedup

        return BenchmarkResult(
            name=name,
            size=size,
            baseline_median_ns=stats_baseline['median'],
            baseline_mean_ns=stats_baseline['mean'],
            baseline_stdev_ns=stats_baseline['stdev'],
            baseline_cv_percent=stats_baseline['cv'],
            optimized_median_ns=stats_optimized['median'],
            optimized_mean_ns=stats_optimized['mean'],
            optimized_stdev_ns=stats_optimized['stdev'],
            optimized_cv_percent=stats_optimized['cv'],
            speedup=speedup,
            ci_95_lower=speedup_lower,
            ci_95_upper=speedup_upper,
            is_stable=is_stable,
            meets_target=meets_target
        )

    def benchmark_suite(
        self,
        name: str,
        baseline_fn: Callable[[int], Callable],
        optimized_fn: Callable[[int], Callable],
        target_speedups: Dict[int, float] = None
    ) -> List[BenchmarkResult]:
        """
        Run benchmarks across all configured sizes.

        Args:
            name: Suite name
            baseline_fn: Function that takes size and returns benchmark function
            optimized_fn: Function that takes size and returns benchmark function
            target_speedups: Dict mapping size -> minimum speedup (optional)

        Returns:
            List of BenchmarkResult for each size
        """
        results = []

        print(f"\n{'='*80}")
        print(f"  BENCHMARK SUITE: {name}")
        print(f"{'='*80}")
        print(f"Configuration:")
        print(f"  Warmup runs: {self.config.warmup_runs}")
        print(f"  Measurement runs: {self.config.measurement_runs}")
        print(f"  Stability threshold: CV < {self.config.max_cv_percent}%")
        print(f"  Default target: {self.config.min_speedup}× speedup")

        for size in self.config.sizes:
            target = target_speedups.get(size, self.config.min_speedup) if target_speedups else self.config.min_speedup

            baseline_callable = baseline_fn(size)
            optimized_callable = optimized_fn(size)

            result = self.benchmark(
                name=f"{name} @ {size:,}",
                baseline_fn=baseline_callable,
                optimized_fn=optimized_callable,
                size=size,
                target_speedup=target
            )

            results.append(result)

        return results

    @staticmethod
    def print_summary_table(results: List[BenchmarkResult]):
        """Print summary table of all results"""
        print(f"\n{'='*80}")
        print("  SUMMARY")
        print(f"{'='*80}")
        print(f"{'Size':>10} | {'Speedup':>10} | {'95% CI':>20} | {'CV (B/O)':>12} | {'Status':>15}")
        print("-" * 80)

        all_stable = True
        all_meet_target = True

        for r in results:
            stability = "✓" if r.is_stable else "⚠"
            target = "✓" if r.meets_target else "✗"

            if not r.is_stable:
                all_stable = False
            if not r.meets_target:
                all_meet_target = False

            status = f"{stability} {target}"

            print(
                f"{r.size:10,} | "
                f"{r.speedup:10.2f}× | "
                f"[{r.ci_95_lower:5.2f}, {r.ci_95_upper:5.2f}] | "
                f"{r.baseline_cv_percent:5.1f}/{r.optimized_cv_percent:5.1f}% | "
                f"{status:>15}"
            )

        print("-" * 80)

        print(f"\n{'='*80}")
        print("  VERDICT")
        print(f"{'='*80}")

        if all_stable and all_meet_target:
            print("✓ SUCCESS: All benchmarks stable and meet targets")
        elif all_meet_target:
            print("⚠ PARTIAL: Targets met but high variance detected")
        elif all_stable:
            print("✗ FAIL: Stable measurements but targets not met")
        else:
            print("✗ FAIL: High variance and/or targets not met")

        print(f"\nMeasurement Quality:")
        print(f"  All stable (CV < 20%): {'✓ Yes' if all_stable else '✗ No'}")
        print(f"  All meet targets:      {'✓ Yes' if all_meet_target else '✗ No'}")

        avg_speedup = sum(r.speedup for r in results) / len(results)
        print(f"\nAverage speedup: {avg_speedup:.2f}×")
        print(f"Range: {min(r.speedup for r in results):.2f}× - {max(r.speedup for r in results):.2f}×")

        print(f"{'='*80}\n")


# Conservative success criteria based on Phase 4.0 lessons
CONSERVATIVE_TARGETS = {
    1_000: 1.2,      # Small: 20% minimum (lowered from original 5%)
    10_000: 1.2,     # Small: 20% minimum
    100_000: 1.3,    # Medium: 30% minimum (lowered from original 20%)
    1_000_000: 1.4,  # Large: 40% minimum (lowered from original 30%)
    10_000_000: 1.4  # Large: 40% minimum (was 1.5× originally)
}


if __name__ == "__main__":
    # Example usage
    print("Benchmark Framework - Example Usage")
    print("=" * 80)

    config = BenchmarkConfig(
        sizes=[1000, 10000, 100000],
        warmup_runs=10,
        measurement_runs=50
    )

    runner = BenchmarkRunner(config)

    # Example: Benchmark numpy operations
    def make_baseline(size):
        a = np.random.rand(size)
        b = np.random.rand(size)
        return lambda: a + b

    def make_optimized(size):
        a = np.random.rand(size)
        b = np.random.rand(size)
        return lambda: np.add(a, b)

    results = runner.benchmark_suite(
        "NumPy Add Example",
        make_baseline,
        make_optimized
    )

    runner.print_summary_table(results)

    print("\nIndividual result details:")
    results[0].print_summary()
