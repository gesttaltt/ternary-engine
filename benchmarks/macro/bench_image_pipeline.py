"""
bench_image_pipeline.py - Macro Benchmark: Ternary Image Processing Pipeline

Realistic workload: Multi-stage image processing with ternary operations
- Convolution-like operations (tmul)
- Bias correction (tadd)
- Activation (tnot) ← FUSION OPPORTUNITY
- Additional filters

Goal: Measure actual end-to-end speedup, not isolated micro-kernel performance.
"""

import sys
import time
import statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import platform

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

print("\n" + "="*80)
print("  MACRO BENCHMARK: TERNARY IMAGE PROCESSING PIPELINE")
print("="*80)
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Python: {platform.python_version()}")
print(f"NumPy: {np.__version__}")
print("="*80 + "\n")

# =============================================================================
# REALISTIC IMAGE PROCESSING PIPELINE
# =============================================================================

def ternary_image_pipeline_unfused(image, kernel, bias, threshold):
    """
    Realistic ternary image processing pipeline - UNFUSED version

    Steps:
    1. Convolution-like operation (simulated with tmul)
    2. Bias correction (tadd)
    3. Activation function (tnot) ← Could be fused with step 2
    4. Thresholding (tmin)
    5. Final adjustment (tmul)
    """
    # Step 1: Convolution (element-wise multiply as proxy)
    conv_result = ternary.tmul(image, kernel)

    # Step 2: Bias correction
    biased = ternary.tadd(conv_result, bias)

    # Step 3: Activation (UNFUSED - separate operation)
    activated = ternary.tnot(biased)

    # Step 4: Thresholding
    thresholded = ternary.tmin(activated, threshold)

    # Step 5: Final adjustment
    result = ternary.tmul(thresholded, kernel)

    return result


def ternary_image_pipeline_fused(image, kernel, bias, threshold):
    """
    Realistic ternary image processing pipeline - FUSED version

    Same pipeline, but step 2+3 are fused (tadd+tnot → fused_tnot_tadd)
    """
    # Step 1: Convolution
    conv_result = ternary.tmul(image, kernel)

    # Step 2+3: Bias + Activation (FUSED)
    activated = fusion.fused_tnot_tadd(conv_result, bias)

    # Step 4: Thresholding
    thresholded = ternary.tmin(activated, threshold)

    # Step 5: Final adjustment
    result = ternary.tmul(thresholded, kernel)

    return result


# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================

# Image sizes (realistic: 256x256 to 4096x4096)
IMAGE_SIZES = [
    (256, 256),      # Small: 65K pixels
    (512, 512),      # Medium: 262K pixels
    (1024, 1024),    # Large: 1M pixels
    (2048, 2048),    # Very large: 4M pixels
]

WARMUP_RUNS = 5
MEASUREMENT_RUNS = 30

print("WORKLOAD ANALYSIS")
print("-" * 80)
print("Pipeline stages:")
print("  1. Convolution (tmul)")
print("  2. Bias correction (tadd)")
print("  3. Activation (tnot)        ← FUSION with step 2")
print("  4. Thresholding (tmin)")
print("  5. Final adjustment (tmul)")
print()
print("Total operations: 5")
print("Fusible operations: 1 (tadd+tnot)")
print("Fusion coverage: 20% of operations")
print()
print("Theoretical prediction (from micro-vs-macro.md):")
print("  Micro speedup: 1.77×")
print("  Fusion coverage: 20%")
print("  Expected macro speedup: ~1.13× (13% improvement)")
print("="*80 + "\n")

# =============================================================================
# CORRECTNESS VALIDATION
# =============================================================================

print("CORRECTNESS VALIDATION")
print("-" * 80)

size = 1000
image = np.random.randint(0, 4, size, dtype=np.uint8) & 0x03
kernel = np.random.randint(0, 4, size, dtype=np.uint8) & 0x03
bias = np.random.randint(0, 4, size, dtype=np.uint8) & 0x03
threshold = np.random.randint(0, 4, size, dtype=np.uint8) & 0x03

result_unfused = ternary_image_pipeline_unfused(image, kernel, bias, threshold)
result_fused = ternary_image_pipeline_fused(image, kernel, bias, threshold)

if np.array_equal(result_unfused, result_fused):
    print("✓ Correctness: Fused pipeline produces identical results")
else:
    print("✗ FAIL: Fused pipeline produces different results!")
    print("Aborting benchmark.")
    sys.exit(1)

print()

# =============================================================================
# PERFORMANCE BENCHMARKING
# =============================================================================

print("MACRO PERFORMANCE BENCHMARKING")
print("-" * 80)
print(f"{'Image Size':>12} | {'Unfused (ms)':>15} | {'Fused (ms)':>15} | {'Speedup':>10} | {'Status':>10}")
print("-" * 80)

all_speedups = []

for height, width in IMAGE_SIZES:
    n_pixels = height * width

    # Generate test data
    image = np.random.randint(0, 4, n_pixels, dtype=np.uint8) & 0x03
    kernel = np.random.randint(0, 4, n_pixels, dtype=np.uint8) & 0x03
    bias = np.random.randint(0, 4, n_pixels, dtype=np.uint8) & 0x03
    threshold = np.random.randint(0, 4, n_pixels, dtype=np.uint8) & 0x03

    # Warmup
    for _ in range(WARMUP_RUNS):
        _ = ternary_image_pipeline_unfused(image, kernel, bias, threshold)
        _ = ternary_image_pipeline_fused(image, kernel, bias, threshold)

    # Benchmark unfused
    times_unfused = []
    for _ in range(MEASUREMENT_RUNS):
        start = time.perf_counter()
        _ = ternary_image_pipeline_unfused(image, kernel, bias, threshold)
        times_unfused.append((time.perf_counter() - start) * 1000)  # Convert to ms

    # Benchmark fused
    times_fused = []
    for _ in range(MEASUREMENT_RUNS):
        start = time.perf_counter()
        _ = ternary_image_pipeline_fused(image, kernel, bias, threshold)
        times_fused.append((time.perf_counter() - start) * 1000)

    # Statistics
    median_unfused = statistics.median(times_unfused)
    median_fused = statistics.median(times_fused)
    speedup = median_unfused / median_fused

    all_speedups.append(speedup)

    status = "✓" if speedup >= 1.05 else "⚠"

    print(f"{height}x{width:>5} | {median_unfused:15.2f} | {median_fused:15.2f} | {speedup:10.2f}× | {status:>10}")

print("-" * 80)

# =============================================================================
# ANALYSIS
# =============================================================================

avg_speedup = statistics.mean(all_speedups)
min_speedup = min(all_speedups)
max_speedup = max(all_speedups)

print("\n" + "="*80)
print("  MACRO SPEEDUP ANALYSIS")
print("="*80)

print(f"\nEnd-to-End Pipeline Speedup:")
print(f"  Average: {avg_speedup:.3f}× ({(avg_speedup-1)*100:.1f}% improvement)")
print(f"  Range:   {min_speedup:.3f}× - {max_speedup:.3f}×")

print(f"\nComparison to Predictions:")
print(f"  Micro speedup (isolated tadd+tnot):  1.77×")
print(f"  Predicted macro (20% coverage):      ~1.13× (13%)")
print(f"  Measured macro (actual pipeline):    {avg_speedup:.3f}× ({(avg_speedup-1)*100:.1f}%)")

if avg_speedup >= 1.13:
    delta = avg_speedup - 1.13
    print(f"  ✓ Exceeds prediction by {delta:.3f}× ({delta*100:.1f} percentage points)")
elif avg_speedup >= 1.10:
    delta = 1.13 - avg_speedup
    print(f"  ⚠ Slightly below prediction by {delta:.3f}× ({delta*100:.1f} percentage points)")
else:
    delta = 1.13 - avg_speedup
    print(f"  ✗ Significantly below prediction by {delta:.3f}× ({delta*100:.1f} percentage points)")

print(f"\nHonest Assessment:")
if avg_speedup >= 1.10:
    print(f"  ✓ Fusion provides meaningful {(avg_speedup-1)*100:.1f}% speedup in realistic pipeline")
    print(f"  → Worth deploying Phase 4.1")
    print(f"  → Consider investigating Phase 4.2/4.3 (more fusion patterns)")
elif avg_speedup >= 1.05:
    print(f"  ⚠ Modest {(avg_speedup-1)*100:.1f}% speedup - useful but not spectacular")
    print(f"  → Worth deploying Phase 4.1")
    print(f"  → Phase 4.2/4.3 likely not worth engineering cost")
else:
    print(f"  ✗ Only {(avg_speedup-1)*100:.1f}% speedup - too small to justify")
    print(f"  → Re-evaluate fusion strategy")
    print(f"  → May indicate Python overhead dominates")

print("\n" + "="*80)
print("  IMAGE PROCESSING PIPELINE BENCHMARK COMPLETE")
print("="*80)

# Exit code based on success
if avg_speedup >= 1.05:
    print(f"\n✓ Macro speedup ({avg_speedup:.3f}×) meets 5% minimum threshold")
    sys.exit(0)
else:
    print(f"\n✗ Macro speedup ({avg_speedup:.3f}×) below 5% threshold")
    sys.exit(1)
