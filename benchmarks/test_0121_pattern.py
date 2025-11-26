"""
test_0121_pattern.py - Deep dive into [0,1,2,1] pattern pathology

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

CRITICAL FINDING:
The specific pattern [0, 1, 2, 1] causes 37× slowdown compared to random!

This pattern has unique property:
- Autocorrelation lag1: 0.0 (no adjacent correlation)
- Autocorrelation lag2: -1.0 (perfect NEGATIVE correlation at lag 2!)

This script tests why this specific pattern is pathological.

USAGE:
    python benchmarks/test_0121_pattern.py
"""

import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import ternary_simd_engine as tse
except ImportError:
    print("ERROR: ternary_simd_engine not available")
    sys.exit(1)

from benchmarks.utils.hardware_metrics import HardwareMetrics


def test_pattern_variants():
    """Test variations of the [0,1,2,1] pattern to isolate the issue."""

    print("="*80)
    print("  TESTING [0,1,2,1] PATTERN VARIANTS")
    print("="*80)

    size = 1_000_000
    hw_metrics = HardwareMetrics()

    patterns = {
        '[0,1,2,1] - ORIGINAL PATHOLOGICAL': np.tile(np.array([0, 1, 2, 1], dtype=np.uint8), size // 4),
        '[0,1,2,0] - Length 4, symmetric': np.tile(np.array([0, 1, 2, 0], dtype=np.uint8), size // 4),
        '[0,1,2] - Length 3, simple cycle': np.tile(np.array([0, 1, 2], dtype=np.uint8), size // 3 + 1)[:size],
        '[0,2,1,0] - Length 4, different order': np.tile(np.array([0, 2, 1, 0], dtype=np.uint8), size // 4),
        '[1,1,1,1] - Constant (all 1s)': np.ones(size, dtype=np.uint8),
        '[0,0,2,2] - Pairs': np.tile(np.array([0, 0, 2, 2], dtype=np.uint8), size // 4),
        'Random - Baseline': np.random.randint(0, 3, size=size, dtype=np.uint8),
    }

    results = {}

    for name, data in patterns.items():
        # Ensure correct size
        if len(data) > size:
            data = data[:size]

        result = hw_metrics.measure_operation(tse.tadd, data, data, iterations=1000)
        time_per_op_ns = result['time_ns_per_iter']
        throughput_mops = (size / (time_per_op_ns / 1e9)) / 1e6

        results[name] = throughput_mops

        print(f"{name:40}: {throughput_mops:>10.2f} Mops/s")

    # Calculate ratios relative to random
    random_throughput = results['Random - Baseline']

    print("\n" + "="*80)
    print("  SLOWDOWN RELATIVE TO RANDOM")
    print("="*80)

    for name, throughput in results.items():
        if name == 'Random - Baseline':
            continue
        ratio = random_throughput / throughput
        status = "⚠️ PATHOLOGICAL" if ratio > 10 else ("⚠️ SLOW" if ratio > 2 else "✅ OK")
        print(f"{name:40}: {ratio:>6.2f}× slower {status}")


def test_dtype_impact():
    """Test if int32 vs uint8 matters (Phase 1 low_entropy was int32)."""

    print("\n" + "="*80)
    print("  TESTING DTYPE IMPACT (int32 vs uint8)")
    print("="*80)

    size = 1_000_000
    hw_metrics = HardwareMetrics()

    pattern = [0, 1, 2, 1]

    # Test with uint8 (2-bit encoding standard)
    data_uint8 = np.tile(np.array(pattern, dtype=np.uint8), size // len(pattern))
    result_uint8 = hw_metrics.measure_operation(tse.tadd, data_uint8, data_uint8, iterations=1000)
    throughput_uint8 = (size / (result_uint8['time_ns_per_iter'] / 1e9)) / 1e6

    # Test with int32 (like Phase 1 dataset)
    data_int32 = np.tile(np.array(pattern, dtype=np.int32), size // len(pattern))
    result_int32 = hw_metrics.measure_operation(tse.tadd, data_int32, data_int32, iterations=1000)
    throughput_int32 = (size / (result_int32['time_ns_per_iter'] / 1e9)) / 1e6

    print(f"uint8 (standard):     {throughput_uint8:>10.2f} Mops/s")
    print(f"int32 (Phase 1):      {throughput_int32:>10.2f} Mops/s")
    print(f"Ratio (uint8/int32):  {throughput_uint8 / throughput_int32:>10.2f}×")

    if throughput_int32 < throughput_uint8 * 0.5:
        print("\n⚠️  CRITICAL: int32 dtype causes significant slowdown!")
        print("   This suggests the kernel may not handle int32 efficiently")
    else:
        print("\n✅ Dtype does not significantly impact performance")


def test_simd_alignment():
    """Test if the pattern causes SIMD alignment issues."""

    print("\n" + "="*80)
    print("  TESTING SIMD ALIGNMENT (32 trits/vector)")
    print("="*80)

    size = 1_000_000
    hw_metrics = HardwareMetrics()

    pattern = [0, 1, 2, 1]

    # Test with sizes that are/aren't multiples of 32 (SIMD width)
    test_sizes = [
        (999968, "32-aligned (999968 = 31249 * 32)"),
        (1000000, "Not 32-aligned"),
        (1048576, "Power of 2 (2^20)"),
    ]

    for test_size, desc in test_sizes:
        data = np.tile(np.array(pattern, dtype=np.uint8), test_size // len(pattern) + 1)[:test_size]

        result = hw_metrics.measure_operation(tse.tadd, data, data, iterations=1000)
        throughput = (test_size / (result['time_ns_per_iter'] / 1e9)) / 1e6

        print(f"{desc:40}: {throughput:>10.2f} Mops/s")


def main():
    test_pattern_variants()
    test_dtype_impact()
    test_simd_alignment()

    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80)
    print("\nThe [0,1,2,1] pattern is pathological because:")
    print("1. It has perfect negative autocorrelation at lag 2")
    print("2. This creates a specific memory access pattern")
    print("3. Likely causes cache line conflicts or false sharing")
    print("\nNext steps:")
    print("- Profile with VTune/perf to measure actual cache misses")
    print("- Check assembly for vectorization quality")
    print("- Consider special handling for repetitive patterns in Phase 2")


if __name__ == '__main__':
    main()
