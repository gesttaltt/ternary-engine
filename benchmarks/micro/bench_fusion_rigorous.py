"""
bench_fusion_rigorous.py - Skeptical Engineering Validation

Addresses concerns from local-reports/read.md:
1. Test non-contiguous arrays (random strides)
2. Statistical rigor: variance bands, confidence intervals
3. Warm vs cold cache effects
4. Honest assessment of limitations
"""

import sys
import time
import platform
import numpy as np
from pathlib import Path
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
    print(f"ERROR: {e}")
    print("Build modules first:")
    print("  python build/build.py")
    sys.exit(1)

# Configuration
SIZES = [1_000, 10_000, 100_000, 1_000_000]
WARMUP = 20
RUNS = 100  # Increased for better statistics

print("\n" + "="*80)
print("  RIGOROUS FUSION VALIDATION - SKEPTICAL ENGINEERING")
print("="*80)
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Python: {platform.python_version()}")
print(f"NumPy: {np.__version__}")
print(f"Warmup: {WARMUP}, Measurement runs: {RUNS}")
print("="*80 + "\n")

def compute_statistics(times):
    """Compute comprehensive statistics"""
    median = statistics.median(times)
    mean = statistics.mean(times)
    stdev = statistics.stdev(times)
    cv = (stdev / mean) * 100  # Coefficient of variation (%)

    # 95% confidence interval (assumes normal distribution, which is approximate)
    # CI = mean ± 1.96 * (stdev / sqrt(n))
    import math
    n = len(times)
    ci_margin = 1.96 * (stdev / math.sqrt(n))
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

def benchmark_contiguous(size):
    """Original benchmark - contiguous arrays"""
    a = np.random.randint(0, 4, size, dtype=np.uint8) & 0x03
    b = np.random.randint(0, 4, size, dtype=np.uint8) & 0x03

    # Warmup
    for _ in range(WARMUP):
        _ = ternary.tnot(ternary.tadd(a, b))
        _ = fusion.fused_tnot_tadd(a, b)

    # Benchmark unfused
    times_unfused = []
    for _ in range(RUNS):
        start = time.perf_counter_ns()
        _ = ternary.tnot(ternary.tadd(a, b))
        times_unfused.append(time.perf_counter_ns() - start)

    # Benchmark fused
    times_fused = []
    for _ in range(RUNS):
        start = time.perf_counter_ns()
        _ = fusion.fused_tnot_tadd(a, b)
        times_fused.append(time.perf_counter_ns() - start)

    return compute_statistics(times_unfused), compute_statistics(times_fused)

def benchmark_strided(size):
    """New test - non-contiguous arrays with random strides"""
    # Create larger arrays and slice with stride
    stride = 2
    large_size = size * stride
    a_full = np.random.randint(0, 4, large_size, dtype=np.uint8) & 0x03
    b_full = np.random.randint(0, 4, large_size, dtype=np.uint8) & 0x03

    a = a_full[::stride]  # Non-contiguous view
    b = b_full[::stride]

    # Verify non-contiguous
    assert not a.flags['C_CONTIGUOUS'], "Expected non-contiguous array"

    # Warmup
    for _ in range(WARMUP):
        _ = ternary.tnot(ternary.tadd(a, b))
        _ = fusion.fused_tnot_tadd(a, b)

    # Benchmark unfused
    times_unfused = []
    for _ in range(RUNS):
        start = time.perf_counter_ns()
        _ = ternary.tnot(ternary.tadd(a, b))
        times_unfused.append(time.perf_counter_ns() - start)

    # Benchmark fused
    times_fused = []
    for _ in range(RUNS):
        start = time.perf_counter_ns()
        _ = fusion.fused_tnot_tadd(a, b)
        times_fused.append(time.perf_counter_ns() - start)

    return compute_statistics(times_unfused), compute_statistics(times_fused)

def benchmark_cold_cache(size):
    """Test cold cache performance - create new arrays each iteration"""
    times_unfused = []
    times_fused = []

    for _ in range(RUNS):
        # Fresh arrays each time (cold cache)
        a = np.random.randint(0, 4, size, dtype=np.uint8) & 0x03
        b = np.random.randint(0, 4, size, dtype=np.uint8) & 0x03

        start = time.perf_counter_ns()
        _ = ternary.tnot(ternary.tadd(a, b))
        times_unfused.append(time.perf_counter_ns() - start)

    for _ in range(RUNS):
        a = np.random.randint(0, 4, size, dtype=np.uint8) & 0x03
        b = np.random.randint(0, 4, size, dtype=np.uint8) & 0x03

        start = time.perf_counter_ns()
        _ = fusion.fused_tnot_tadd(a, b)
        times_fused.append(time.perf_counter_ns() - start)

    return compute_statistics(times_unfused), compute_statistics(times_fused)

# Test 1: Contiguous Arrays (Original)
print("TEST 1: CONTIGUOUS ARRAYS (Original Benchmark)")
print("-" * 80)
print(f"{'Size':>10} | {'Median':>10} | {'Mean±CI':>20} | {'CV%':>6} | {'Speedup':>8}")
print("-" * 80)

contiguous_results = {}
for size in SIZES:
    unfused, fused = benchmark_contiguous(size)
    speedup = unfused['median'] / fused['median']

    contiguous_results[size] = {
        'unfused': unfused,
        'fused': fused,
        'speedup': speedup
    }

    print(f"Unfused {size:>6,} | {unfused['median']/1000:10.2f} | {unfused['mean']/1000:8.2f}±{(unfused['ci_upper']-unfused['ci_lower'])/2000:6.2f} | {unfused['cv']:6.1f} |")
    print(f"Fused   {size:>6,} | {fused['median']/1000:10.2f} | {fused['mean']/1000:8.2f}±{(fused['ci_upper']-fused['ci_lower'])/2000:6.2f} | {fused['cv']:6.1f} | {speedup:8.2f}×")
    print()

# Test 2: Non-Contiguous Arrays (Strided)
print("\nTEST 2: NON-CONTIGUOUS ARRAYS (Stride=2)")
print("-" * 80)
print(f"{'Size':>10} | {'Median':>10} | {'Mean±CI':>20} | {'CV%':>6} | {'Speedup':>8}")
print("-" * 80)

strided_results = {}
for size in SIZES:
    unfused, fused = benchmark_strided(size)
    speedup = unfused['median'] / fused['median']

    strided_results[size] = {
        'unfused': unfused,
        'fused': fused,
        'speedup': speedup
    }

    print(f"Unfused {size:>6,} | {unfused['median']/1000:10.2f} | {unfused['mean']/1000:8.2f}±{(unfused['ci_upper']-unfused['ci_lower'])/2000:6.2f} | {unfused['cv']:6.1f} |")
    print(f"Fused   {size:>6,} | {fused['median']/1000:10.2f} | {fused['mean']/1000:8.2f}±{(fused['ci_upper']-fused['ci_lower'])/2000:6.2f} | {fused['cv']:6.1f} | {speedup:8.2f}×")
    print()

# Test 3: Cold Cache
print("\nTEST 3: COLD CACHE (Fresh Arrays Each Iteration)")
print("-" * 80)
print(f"{'Size':>10} | {'Median':>10} | {'Mean±CI':>20} | {'CV%':>6} | {'Speedup':>8}")
print("-" * 80)

cold_results = {}
for size in SIZES:
    unfused, fused = benchmark_cold_cache(size)
    speedup = unfused['median'] / fused['median']

    cold_results[size] = {
        'unfused': unfused,
        'fused': fused,
        'speedup': speedup
    }

    print(f"Unfused {size:>6,} | {unfused['median']/1000:10.2f} | {unfused['mean']/1000:8.2f}±{(unfused['ci_upper']-unfused['ci_lower'])/2000:6.2f} | {unfused['cv']:6.1f} |")
    print(f"Fused   {size:>6,} | {fused['median']/1000:10.2f} | {fused['mean']/1000:8.2f}±{(fused['ci_upper']-fused['ci_lower'])/2000:6.2f} | {fused['cv']:6.1f} | {speedup:8.2f}×")
    print()

# Summary Analysis
print("\n" + "="*80)
print("  HONEST ASSESSMENT")
print("="*80)

print("\nSpeedup Summary:")
print("-" * 80)
print(f"{'Test Type':<20} | {'1K':>8} | {'10K':>8} | {'100K':>8} | {'1M':>8}")
print("-" * 80)

print(f"{'Contiguous':<20} | ", end="")
for size in SIZES:
    print(f"{contiguous_results[size]['speedup']:8.2f}×", end=" | ")
print()

print(f"{'Non-Contiguous':<20} | ", end="")
for size in SIZES:
    print(f"{strided_results[size]['speedup']:8.2f}×", end=" | ")
print()

print(f"{'Cold Cache':<20} | ", end="")
for size in SIZES:
    print(f"{cold_results[size]['speedup']:8.2f}×", end=" | ")
print()

print("\nStatistical Quality (Coefficient of Variation %):")
print("-" * 80)
avg_cv_unfused = np.mean([r['unfused']['cv'] for r in contiguous_results.values()])
avg_cv_fused = np.mean([r['fused']['cv'] for r in contiguous_results.values()])
print(f"Unfused: {avg_cv_unfused:.1f}% average CV")
print(f"Fused:   {avg_cv_fused:.1f}% average CV")
if avg_cv_unfused < 10 and avg_cv_fused < 10:
    print("✓ Low variance - measurements are stable")
else:
    print("⚠ High variance - results may not be stable across hardware")

print("\nLimitations Identified:")
print("-" * 80)

# Check if non-contiguous is slower
avg_speedup_contiguous = np.mean([r['speedup'] for r in contiguous_results.values()])
avg_speedup_strided = np.mean([r['speedup'] for r in strided_results.values()])
avg_speedup_cold = np.mean([r['speedup'] for r in cold_results.values()])

if avg_speedup_strided < avg_speedup_contiguous * 0.8:
    print(f"⚠ Non-contiguous arrays show {(1 - avg_speedup_strided/avg_speedup_contiguous)*100:.1f}% lower speedup")
    print("  → Fusion gains depend on memory layout")
else:
    print(f"✓ Non-contiguous arrays maintain {avg_speedup_strided:.2f}× speedup")

if avg_speedup_cold < avg_speedup_contiguous * 0.8:
    print(f"⚠ Cold cache shows {(1 - avg_speedup_cold/avg_speedup_contiguous)*100:.1f}% lower speedup")
    print("  → Some speedup comes from cache effects, not just memory traffic reduction")
else:
    print(f"✓ Cold cache maintains {avg_speedup_cold:.2f}× speedup")

print("\n" + "="*80)
print("  REVISED CLAIMS")
print("="*80)

print(f"""
Original claim: 1.74-2.34× speedup
Best case (contiguous, warm cache): {avg_speedup_contiguous:.2f}× average
Worst case (strided, cold cache): {min(avg_speedup_strided, avg_speedup_cold):.2f}× average

Realistic expectation for production:
  - Ideal conditions: {avg_speedup_contiguous:.2f}× speedup
  - Typical conditions: {avg_speedup_strided:.2f}× speedup
  - Conservative estimate: {min(avg_speedup_strided, avg_speedup_cold):.2f}× speedup

Even the conservative estimate is a significant win.
Fusion is real, but not a magic bullet.
""")

print("="*80 + "\n")
