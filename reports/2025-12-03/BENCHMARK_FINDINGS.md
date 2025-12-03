# Ternary Engine Benchmark Findings

**Doc-Type:** Benchmark Report · Date: 2025-12-03 · Platform: Windows 11 x64 AMD Ryzen

---

## Executive Summary

Comprehensive benchmark validation of the Ternary Engine on Windows 11 with AMD Ryzen processor (12 logical cores, AMD64 Family 23 Model 96).

| Metric | Value | Context |
|:-------|:------|:--------|
| Peak Throughput | **19.57 Gops/s** | tnot @ 100K elements |
| Peak Binary Op | **15.53 Gops/s** | tmin @ 100K elements |
| Bridge Layer Max | **12.90 Gops/s** | @ 262K elements |
| Fused Ops Max | **5.56 Gops/s** | fused_tnot_tadd @ 1M elements |
| Avg Speedup vs Python | **1,900x** | Pure Python baseline |
| Bridge vs NumPy | **44.58x** max | @ 262K elements |
| Fused vs NumPy | **29.12x** max | @ 1M elements |

**All 4/4 performance tests PASSED.**

---

## System Configuration

| Component | Value |
|:----------|:------|
| Platform | Windows-11-10.0.26100-SP0 |
| Processor | AMD64 Family 23 Model 96 Stepping 1 (AMD Ryzen) |
| Logical CPUs | 12 |
| OMP Threads | 12 (auto-set) |
| Module | ternary_simd_engine.cp312-win_amd64.pyd |

---

## 1. Core SIMD Operations (Gops/s)

### Peak Throughput by Operation

| Operation | Peak Gops/s | Optimal Size | Speedup vs Python |
|:----------|:------------|:-------------|:------------------|
| tnot | **19.57** | 100,000 | 2,634x |
| tmin | **15.53** | 100,000 | 3,837x |
| tmax | **15.48** | 100,000 | 3,951x |
| tadd | **15.34** | 100,000 | 4,022x |
| tmul | **15.10** | 100,000 | 3,993x |

### Throughput Scaling (Gops/s)

| Size | tadd | tmul | tmin | tmax | tnot |
|:-----|:-----|:-----|:-----|:-----|:-----|
| 32 | 0.023 | 0.023 | 0.023 | 0.018 | 0.029 |
| 1,000 | 0.675 | 0.679 | 0.653 | 0.673 | 0.887 |
| 100,000 | **15.34** | **15.10** | **15.53** | **15.48** | **19.57** |
| 1,000,000 | 3.36 | 4.78 | 6.26 | 8.03 | 8.22 |

**Observation:** Peak throughput at 100K elements (L2 cache sweet spot). Larger arrays become memory-bandwidth limited.

---

## 2. Bridge Layer Performance

The Bridge Layer eliminates Python/NumPy conversion overhead by fusing format conversion with SIMD operations.

### Performance vs Array Size

| Size | Bridge Gops/s | NumPy Gops/s | Speedup |
|:-----|:--------------|:-------------|:--------|
| 64 | 0.033 | 0.008 | 3.96x |
| 256 | 0.134 | 0.038 | 3.57x |
| 1,024 | 0.508 | 0.117 | 4.34x |
| 4,096 | 1.85 | 0.257 | 7.19x |
| 16,384 | 5.09 | 0.406 | 12.54x |
| 65,536 | 10.21 | 0.460 | 22.22x |
| 262,144 | **12.90** | 0.289 | **44.58x** |
| 1,048,576 | 2.47 | 0.252 | 9.80x |

### Bridge Layer Summary

| Metric | Value |
|:-------|:------|
| Crossover Point | 64 elements (wins even for tiny arrays) |
| Avg Speedup vs NumPy | 13.52x |
| Max Speedup vs NumPy | **44.58x** |
| Avg Speedup vs Naive | 16.88x |
| Max Speedup vs Naive | **48.69x** |

---

## 3. Fused Operations (Gops/s)

Fused operations combine multiple ternary operations in a single pass, eliminating intermediate arrays.

| Size | Ternary Fused | NumPy Baseline | Speedup |
|:-----|:--------------|:---------------|:--------|
| 100,000 | 4.48 | 0.34 | 13.15x |
| 1,000,000 | **5.56** | 0.19 | **29.12x** |
| 10,000,000 | 4.87 | 0.22 | 22.22x |

**Key insight:** Fused operations maintain high throughput even at large scales because they reduce memory traffic.

---

## 4. All Bridge Operations @ 100K Elements

| Operation | Bridge Gops/s | NumPy Gops/s | Speedup | Notes |
|:----------|:--------------|:-------------|:--------|:------|
| tadd_int8 | 7.91 | 0.31 | **25.30x** | Saturated add - our strength |
| tmul_int8 | 7.43 | 1.08 | **6.88x** | Multiplication - our strength |
| tmin_int8 | 8.15 | 18.71 | 0.44x | NumPy intrinsic faster |
| tmax_int8 | 8.19 | 17.78 | 0.46x | NumPy intrinsic faster |
| tnot_int8 | 9.00 | 25.33 | 0.36x | NumPy intrinsic faster |

**Analysis:**
- **Ternary wins** for complex operations (saturated add, multiply)
- **NumPy wins** for simple operations (min, max, negate) where it uses single intrinsics
- Bridge Layer's value is in **fused operations** and **memory efficiency**

---

## 5. Performance Characteristics

### Optimal Operating Points

| Regime | Array Size | Characteristics |
|:-------|:-----------|:----------------|
| Overhead-dominated | < 1,000 | Call overhead dominates, ~20-30 Mops/s |
| L1 Cache | 1K - 32K | Good scaling, ~500-1,500 Mops/s |
| **L2 Cache (Sweet Spot)** | 32K - 256K | **Peak throughput: 12-19 Gops/s** |
| L3 Cache | 256K - 8M | Declining: 3-8 Gops/s |
| Memory-bound | > 8M | Memory bandwidth limited: 3-5 Gops/s |

### Memory Bandwidth Analysis

At peak (100K elements, 15 Gops/s):
- Read: 2 arrays × 100K bytes = 200 KB
- Write: 1 array × 100K bytes = 100 KB
- Total: 300 KB per operation
- Bandwidth: 300 KB × 15G ops/s = **4.5 TB/s** (theoretical, cache-local)

At 1M elements (memory-bound):
- Throughput: ~5 Gops/s
- Bandwidth: 3 MB × 5G = **15 GB/s** (approaching DDR4 limits)

---

## 6. Key Insights

### Ternary Value Proposition

1. **Memory Efficiency**: 2-bit encoding vs 8-bit = **4x compression**
2. **Complex Operations**: Saturated arithmetic faster than NumPy
3. **Fused Operations**: Up to **29x speedup** by eliminating intermediates
4. **Edge/Mobile**: Reduced memory footprint critical for inference

### Where Binary Wins

1. **Simple operations**: min, max, negate - single AVX2 instruction
2. **Raw throughput**: Binary INT8 add is 1 instruction vs 3 shuffles
3. **No conversion**: Native format, no encoding overhead

### Recommended Use Cases

| Use Case | Recommendation |
|:---------|:---------------|
| AI Model Compression | **Ternary** - 4x memory savings |
| Neural Network Inference | **Ternary** - bandwidth-limited scenarios |
| Simple Arithmetic | Binary - raw throughput |
| Edge Deployment | **Ternary** - memory-constrained |
| High-throughput Compute | Binary - peak Gops/s |

---

## 7. Test Results Summary

| Test | Expected | Actual | Status |
|:-----|:---------|:-------|:-------|
| crossover_tadd_int8 | < 20,000 elements | 64 elements | **PASS** |
| speedup_tadd_int8_large | > 10.0x | 15.09x | **PASS** |
| speedup_fused_op_large | > 20.0x | 29.12x | **PASS** |
| correctness_bridge_layer | 100% match | PASS | **PASS** |

**Total: 4/4 tests passed**

---

## 8. Files Generated

```
benchmarks/results/2025-12-03_01-07-50/bridge_layer_performance_results.json
benchmarks/benchmarks/results/bench_results_20251203_010632.json
benchmarks/benchmarks/results/bench_results_20251203_010632.csv
```

---

## 9. Next Steps: Memory Efficiency Benchmarking

To fully validate ternary's value proposition, we need C++ benchmarks that measure:

1. **Memory Footprint**: Bytes per weight for ternary vs INT8/FP16/FP32
2. **Bandwidth Utilization**: Actual GB/s achieved vs theoretical peak
3. **Cache Efficiency**: L1/L2/L3 hit rates
4. **Power Efficiency**: Operations per watt (requires hardware counters)

See: `benchmarks/cpp-native-kernels/` for the C++ benchmark architecture.

---

**Validated:** 2025-12-03 01:07 UTC · **Platform:** Windows 11 x64 AMD Ryzen · **Status:** All Tests Passed
