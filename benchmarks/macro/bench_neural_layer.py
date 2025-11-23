"""
bench_neural_layer.py - Macro Benchmark: Ternary Neural Network Layer

Realistic workload: Forward pass through a ternary neural network layer
- Matrix multiply (simulated with many tmul operations)
- Bias add (tadd)
- Activation (tnot) ← FUSION OPPORTUNITY

Goal: Measure end-to-end layer performance (closer to real ML workload).
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
    print("Build modules first")
    sys.exit(1)

print("\n" + "="*80)
print("  MACRO BENCHMARK: TERNARY NEURAL NETWORK LAYER")
print("="*80)
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Python: {platform.python_version()}")
print(f"NumPy: {np.__version__}")
print("="*80 + "\n")

# =============================================================================
# TERNARY NEURAL LAYER (Simplified)
# =============================================================================

def ternary_matmul_simple(X, W):
    """
    Simplified ternary matrix multiply (element-wise proxy).
    Real matmul would be more complex, but this simulates the computational load.
    """
    # Simulate matrix multiply with element-wise operations
    # In real implementation, this would be proper dot products
    result = ternary.tmul(X, W)
    return result


def ternary_layer_forward_unfused(X, W, b):
    """
    Ternary neural network layer - UNFUSED

    Forward pass: Z = matmul(X, W) + b, A = activation(Z)
    """
    # Matrix multiply (simulated)
    Z = ternary_matmul_simple(X, W)

    # Add bias (UNFUSED from activation)
    Z_biased = ternary.tadd(Z, b)

    # Activation
    A = ternary.tnot(Z_biased)

    return A


def ternary_layer_forward_fused(X, W, b):
    """
    Ternary neural network layer - FUSED

    Same forward pass, but bias+activation fused
    """
    # Matrix multiply (simulated)
    Z = ternary_matmul_simple(X, W)

    # Bias + Activation (FUSED)
    A = fusion.fused_tnot_tadd(Z, b)

    return A


# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================

# Layer sizes (typical neural network dimensions)
LAYER_CONFIGS = [
    ("Small", 1024),          # 1K neurons
    ("Medium", 10_000),       # 10K neurons
    ("Large", 100_000),       # 100K neurons
    ("Very Large", 1_000_000) # 1M neurons (transformer-scale)
]

WARMUP_RUNS = 5
MEASUREMENT_RUNS = 30

print("WORKLOAD ANALYSIS")
print("-" * 80)
print("Neural layer operations:")
print("  1. Matrix multiply (tmul) - dominates computation")
print("  2. Bias add (tadd)")
print("  3. Activation (tnot)       ← FUSION with step 2")
print()
print("Total operations: 3")
print("Fusible operations: 1 (tadd+tnot)")
print("Fusion coverage: ~33% of operations")
print("  BUT: Matrix multiply dominates wall-clock time")
print("  Real fusion coverage: ~15-25% of runtime")
print()
print("Theoretical prediction:")
print("  Micro speedup: 1.77×")
print("  Fusion coverage: ~20% (runtime)")
print("  Expected macro speedup: ~1.13× (13% improvement)")
print("="*80 + "\n")

# =============================================================================
# CORRECTNESS VALIDATION
# =============================================================================

print("CORRECTNESS VALIDATION")
print("-" * 80)

size = 1000
X = np.random.randint(0, 4, size, dtype=np.uint8) & 0x03
W = np.random.randint(0, 4, size, dtype=np.uint8) & 0x03
b = np.random.randint(0, 4, size, dtype=np.uint8) & 0x03

result_unfused = ternary_layer_forward_unfused(X, W, b)
result_fused = ternary_layer_forward_fused(X, W, b)

if np.array_equal(result_unfused, result_fused):
    print("✓ Correctness: Fused layer produces identical results")
else:
    print("✗ FAIL: Fused layer produces different results!")
    sys.exit(1)

print()

# =============================================================================
# PERFORMANCE BENCHMARKING
# =============================================================================

print("MACRO PERFORMANCE BENCHMARKING")
print("-" * 80)
print(f"{'Layer Size':>15} | {'Unfused (ms)':>15} | {'Fused (ms)':>15} | {'Speedup':>10} | {'Status':>10}")
print("-" * 80)

all_speedups = []

for config_name, n_neurons in LAYER_CONFIGS:
    # Generate layer parameters
    X = np.random.randint(0, 4, n_neurons, dtype=np.uint8) & 0x03
    W = np.random.randint(0, 4, n_neurons, dtype=np.uint8) & 0x03
    b = np.random.randint(0, 4, n_neurons, dtype=np.uint8) & 0x03

    # Warmup
    for _ in range(WARMUP_RUNS):
        _ = ternary_layer_forward_unfused(X, W, b)
        _ = ternary_layer_forward_fused(X, W, b)

    # Benchmark unfused
    times_unfused = []
    for _ in range(MEASUREMENT_RUNS):
        start = time.perf_counter()
        _ = ternary_layer_forward_unfused(X, W, b)
        times_unfused.append((time.perf_counter() - start) * 1000)

    # Benchmark fused
    times_fused = []
    for _ in range(MEASUREMENT_RUNS):
        start = time.perf_counter()
        _ = ternary_layer_forward_fused(X, W, b)
        times_fused.append((time.perf_counter() - start) * 1000)

    # Statistics
    median_unfused = statistics.median(times_unfused)
    median_fused = statistics.median(times_fused)
    speedup = median_unfused / median_fused

    all_speedups.append(speedup)

    status = "✓" if speedup >= 1.05 else "⚠"

    print(f"{config_name:>15} | {median_unfused:15.2f} | {median_fused:15.2f} | {speedup:10.2f}× | {status:>10}")

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

print(f"\nEnd-to-End Layer Speedup:")
print(f"  Average: {avg_speedup:.3f}× ({(avg_speedup-1)*100:.1f}% improvement)")
print(f"  Range:   {min_speedup:.3f}× - {max_speedup:.3f}×")

print(f"\nComparison to Predictions:")
print(f"  Micro speedup (isolated tadd+tnot):  1.77×")
print(f"  Predicted macro (~20% coverage):     ~1.13× (13%)")
print(f"  Measured macro (actual layer):       {avg_speedup:.3f}× ({(avg_speedup-1)*100:.1f}%)")

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
    print(f"  ✓ Fusion provides {(avg_speedup-1)*100:.1f}% speedup in neural layer forward pass")
    print(f"  → Useful for ternary neural networks")
    print(f"  → Worth deploying for ML workloads")
elif avg_speedup >= 1.05:
    print(f"  ⚠ Modest {(avg_speedup-1)*100:.1f}% speedup")
    print(f"  → Some benefit but not game-changing")
else:
    print(f"  ✗ Only {(avg_speedup-1)*100:.1f}% speedup - marginal")
    print(f"  → Matrix multiply dominates, fusion impact limited")

print("\n" + "="*80)
print("  NEURAL LAYER BENCHMARK COMPLETE")
print("="*80)

if avg_speedup >= 1.05:
    print(f"\n✓ Macro speedup ({avg_speedup:.3f}×) meets 5% threshold")
    sys.exit(0)
else:
    print(f"\n✗ Macro speedup ({avg_speedup:.3f}×) below 5% threshold")
    sys.exit(1)
