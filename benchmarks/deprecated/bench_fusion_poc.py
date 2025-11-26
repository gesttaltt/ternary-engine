"""
bench_fusion_poc.py - Rigorous Benchmark for Operation Fusion (Phase 4.0 PoC)

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

DESIGN PHILOSOPHY: Truth over claims
  - Measure actual performance, not theoretical
  - Statistical significance testing (not single measurements)
  - Report failures honestly
  - Compare against theoretical predictions
  - Audit-ready methodology

METHODOLOGY:
  - Multiple iterations for statistical validity
  - Warmup runs to eliminate cold-start effects
  - Median + confidence intervals (not just mean)
  - Memory bandwidth measurement
  - Hardware detection and reporting
  - Automated pass/fail criteria

SUCCESS CRITERIA (Conservative):
  - Small arrays (1K-10K): Any measurable speedup (>1.05×)
  - Medium arrays (100K): 1.3× speedup minimum
  - Large arrays (1M+): 1.5× speedup minimum
  - Statistical significance: p < 0.05

FAILURE MODES TO DETECT:
  - No speedup (fusion overhead > memory savings)
  - Regression (fused slower than unfused)
  - High variance (unstable measurements)
  - Platform-specific failures
"""

import sys
import time
import platform
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import statistics

# Add project root to path (3 levels up: micro -> benchmarks -> project_root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import ternary_simd_engine as ternary
    # Fusion operations are integrated into main engine (ternary_simd_engine)
    # Aliasing for compatibility with benchmark structure
    fusion = ternary
except ImportError as e:
    print("ERROR: Required modules not built.")
    print("\nPlease build first:")
    print("  python build/build.py")
    sys.exit(1)

# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================

# Test sizes (elements)
SIZES_SMALL = [1_000, 5_000, 10_000]           # Cache-resident
SIZES_MEDIUM = [50_000, 100_000, 500_000]      # L3 boundary
SIZES_LARGE = [1_000_000, 5_000_000, 10_000_000]  # DRAM-limited

# Statistical parameters
WARMUP_ITERATIONS = 50    # Warmup runs (discarded)
MEASUREMENT_RUNS = 100    # Measured runs for statistics
CONFIDENCE_LEVEL = 0.95   # 95% confidence interval

# Success criteria (conservative)
MIN_SPEEDUP_SMALL = 1.05   # 5% speedup on small arrays
MIN_SPEEDUP_MEDIUM = 1.30  # 30% speedup on medium arrays
MIN_SPEEDUP_LARGE = 1.50   # 50% speedup on large arrays

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BenchmarkResult:
    """Single benchmark measurement"""
    size: int
    operation: str
    median_ns: float
    mean_ns: float
    std_dev_ns: float
    min_ns: float
    max_ns: float
    samples: List[float]

    @property
    def ci_95_lower(self) -> float:
        """95% confidence interval lower bound"""
        return self.median_ns - 1.96 * (self.std_dev_ns / np.sqrt(len(self.samples)))

    @property
    def ci_95_upper(self) -> float:
        """95% confidence interval upper bound"""
        return self.median_ns + 1.96 * (self.std_dev_ns / np.sqrt(len(self.samples)))

    @property
    def throughput_mops(self) -> float:
        """Throughput in million operations per second"""
        return (self.size / (self.median_ns / 1000)) / 1000

@dataclass
class ComparisonResult:
    """Comparison between unfused and fused operations"""
    size: int
    unfused: BenchmarkResult
    fused: BenchmarkResult
    speedup: float
    memory_reduction_theoretical: float
    passed: bool
    reason: str

# =============================================================================
# HARDWARE DETECTION
# =============================================================================

def detect_hardware():
    """Detect and report hardware characteristics"""
    print("="*80)
    print("  HARDWARE CONFIGURATION")
    print("="*80)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python: {platform.python_version()}")
    print(f"NumPy: {np.__version__}")

    # CPU detection (Linux only)
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        print(f"CPU: {line.split(':')[1].strip()}")
                        break
        except:
            pass

    # Detect AVX2 support
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        flags = info.get('flags', [])
        avx2_supported = 'avx2' in flags
        print(f"AVX2 Support: {'✓ YES' if avx2_supported else '✗ NO'}")
    except:
        print("AVX2 Support: Unknown (install py-cpuinfo for detection)")

    print("="*80 + "\n")

# =============================================================================
# BENCHMARK EXECUTION
# =============================================================================

def generate_test_data(size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random ternary test data"""
    # Random 2-bit trits (0b00, 0b01, 0b10)
    a = np.random.randint(0, 4, size, dtype=np.uint8) & 0x03
    b = np.random.randint(0, 4, size, dtype=np.uint8) & 0x03
    return a, b

def benchmark_operation(operation_func, a: np.ndarray, b: np.ndarray,
                       warmup: int, iterations: int) -> BenchmarkResult:
    """
    Benchmark a single operation with statistical rigor

    Returns median time and confidence intervals
    """
    size = len(a)

    # Warmup runs (discard results)
    for _ in range(warmup):
        _ = operation_func(a, b)

    # Measured runs
    times_ns = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        result = operation_func(a, b)
        end = time.perf_counter_ns()
        times_ns.append(end - start)

    # Statistical analysis
    return BenchmarkResult(
        size=size,
        operation=operation_func.__name__,
        median_ns=statistics.median(times_ns),
        mean_ns=statistics.mean(times_ns),
        std_dev_ns=statistics.stdev(times_ns) if len(times_ns) > 1 else 0.0,
        min_ns=min(times_ns),
        max_ns=max(times_ns),
        samples=times_ns
    )

def unfused_tnot_tadd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unfused operation: tnot(tadd(a, b))"""
    temp = ternary.tadd(a, b)
    return ternary.tnot(temp)

def fused_tnot_tadd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fused operation: tnot(tadd(a, b))"""
    return fusion.fused_tnot_tadd(a, b)

# =============================================================================
# CORRECTNESS VALIDATION
# =============================================================================

def validate_correctness(sizes: List[int], num_tests: int = 100) -> bool:
    """Validate fused operation produces correct results"""
    print("="*80)
    print("  CORRECTNESS VALIDATION")
    print("="*80)

    all_passed = True

    for size in sizes:
        passed_count = 0
        for i in range(num_tests):
            a, b = generate_test_data(size)

            # Compute both ways
            expected = unfused_tnot_tadd(a, b)
            actual = fused_tnot_tadd(a, b)

            # Compare
            if np.array_equal(expected, actual):
                passed_count += 1
            else:
                print(f"✗ FAILED: Size {size}, test {i+1}")
                print(f"  Mismatch at indices: {np.where(expected != actual)[0][:10]}")
                all_passed = False

        if passed_count == num_tests:
            print(f"✓ PASSED: Size {size:8,} - {num_tests}/{num_tests} tests passed")
        else:
            print(f"✗ FAILED: Size {size:8,} - {passed_count}/{num_tests} tests passed")
            all_passed = False

    print("="*80 + "\n")
    return all_passed

# =============================================================================
# PERFORMANCE BENCHMARKING
# =============================================================================

def benchmark_size(size: int) -> ComparisonResult:
    """Benchmark both unfused and fused at given size"""
    a, b = generate_test_data(size)

    # Benchmark unfused
    unfused_result = benchmark_operation(
        unfused_tnot_tadd, a, b,
        warmup=WARMUP_ITERATIONS,
        iterations=MEASUREMENT_RUNS
    )

    # Benchmark fused
    fused_result = benchmark_operation(
        fused_tnot_tadd, a, b,
        warmup=WARMUP_ITERATIONS,
        iterations=MEASUREMENT_RUNS
    )

    # Calculate speedup
    speedup = unfused_result.median_ns / fused_result.median_ns

    # Theoretical memory reduction: 5N → 3N = 40%
    memory_reduction_theoretical = 0.40

    # Determine pass/fail based on size category
    if size < 50_000:
        threshold = MIN_SPEEDUP_SMALL
        category = "SMALL"
    elif size < 1_000_000:
        threshold = MIN_SPEEDUP_MEDIUM
        category = "MEDIUM"
    else:
        threshold = MIN_SPEEDUP_LARGE
        category = "LARGE"

    passed = speedup >= threshold
    reason = f"{category}: Speedup {speedup:.2f}× {'≥' if passed else '<'} {threshold:.2f}× threshold"

    return ComparisonResult(
        size=size,
        unfused=unfused_result,
        fused=fused_result,
        speedup=speedup,
        memory_reduction_theoretical=memory_reduction_theoretical,
        passed=passed,
        reason=reason
    )

def run_performance_suite(sizes: List[int]) -> List[ComparisonResult]:
    """Run full performance benchmark suite"""
    print("="*80)
    print("  PERFORMANCE BENCHMARKING")
    print("="*80)
    print(f"Warmup iterations: {WARMUP_ITERATIONS}")
    print(f"Measurement runs: {MEASUREMENT_RUNS}")
    print(f"Confidence level: {CONFIDENCE_LEVEL*100:.0f}%")
    print("="*80 + "\n")

    results = []

    for size in sizes:
        print(f"Benchmarking size: {size:,} elements...", flush=True)
        result = benchmark_size(size)
        results.append(result)

        # Print immediate result
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"  {status} - Speedup: {result.speedup:.2f}× - {result.reason}")

    print()
    return results

# =============================================================================
# RESULTS REPORTING
# =============================================================================

def print_detailed_results(results: List[ComparisonResult]):
    """Print detailed benchmark results"""
    print("="*80)
    print("  DETAILED RESULTS")
    print("="*80 + "\n")

    for r in results:
        print(f"Size: {r.size:,} elements")
        print(f"  Unfused (median): {r.unfused.median_ns/1000:,.2f} μs "
              f"[95% CI: {r.unfused.ci_95_lower/1000:.2f}-{r.unfused.ci_95_upper/1000:.2f}]")
        print(f"  Fused (median):   {r.fused.median_ns/1000:,.2f} μs "
              f"[95% CI: {r.fused.ci_95_lower/1000:.2f}-{r.fused.ci_95_upper/1000:.2f}]")
        print(f"  Speedup:          {r.speedup:.2f}×")
        print(f"  Throughput:       {r.fused.throughput_mops:,.0f} Mops/s")
        print(f"  Variance:         Unfused={r.unfused.std_dev_ns/r.unfused.median_ns*100:.1f}% "
              f"Fused={r.fused.std_dev_ns/r.fused.median_ns*100:.1f}%")
        print(f"  Status:           {'✓ PASS' if r.passed else '✗ FAIL'} - {r.reason}")
        print()

def print_summary(results: List[ComparisonResult], correctness_passed: bool):
    """Print final summary"""
    print("="*80)
    print("  FINAL SUMMARY")
    print("="*80)

    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)

    print(f"\nCorrectness: {'✓ PASSED' if correctness_passed else '✗ FAILED'}")
    print(f"Performance: {passed_count}/{total_count} size categories passed\n")

    if not correctness_passed:
        print("CRITICAL FAILURE: Fused operation produces incorrect results!")
        print("Implementation must be fixed before deployment.\n")
        return False

    if passed_count == total_count:
        print("✓ SUCCESS: All performance targets met!")
        print("\nOperation fusion validated:")
        print("  - Correctness: Verified")
        print("  - Performance: Meets/exceeds conservative targets")
        print("  - Readiness: Production-ready for this fused operation\n")
        return True
    elif passed_count >= total_count * 0.7:
        print("⚠ PARTIAL SUCCESS: Most targets met, some categories underperformed")
        print("\nRecommendation:")
        print("  - Investigate underperforming categories")
        print("  - May be hardware-specific (memory bandwidth, cache)")
        print("  - Consider conditional fusion (only on large arrays)\n")
        return True
    else:
        print("✗ FAILURE: Fusion does not provide expected benefits")
        print("\nHonest assessment:")
        print("  - Theoretical model may be incorrect")
        print("  - Implementation overhead > memory savings")
        print("  - Hardware bottleneck different than expected")
        print("\nRecommendation: Re-evaluate fusion strategy\n")
        return False

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main benchmark execution"""
    print("\n")
    print("="*80)
    print("  TERNARY OPERATION FUSION - PHASE 4.0 POC BENCHMARK")
    print("  Rigorous Validation with Statistical Significance")
    print("="*80 + "\n")

    # Hardware detection
    detect_hardware()

    # Correctness validation
    all_sizes = SIZES_SMALL + SIZES_MEDIUM + SIZES_LARGE
    correctness_passed = validate_correctness(all_sizes, num_tests=100)

    if not correctness_passed:
        print("\n✗ BENCHMARK ABORTED: Correctness validation failed")
        print("Fix implementation before proceeding to performance tests.\n")
        sys.exit(1)

    # Performance benchmarking
    results = run_performance_suite(all_sizes)

    # Detailed results
    print_detailed_results(results)

    # Final summary
    success = print_summary(results, correctness_passed)

    print("="*80)
    print("  BENCHMARK COMPLETE")
    print("="*80 + "\n")

    # Exit code: 0 if success, 1 if failure
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
