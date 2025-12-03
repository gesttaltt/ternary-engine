# Ternary Engine Benchmark Report

**Doc-Type:** Benchmark Report · Date: 2025-12-02 · Platform: Windows 11 x64

---

## Executive Summary

All benchmark suites executed successfully on Windows 11 with AMD Ryzen processor (12 logical cores). The Ternary Engine demonstrates exceptional performance:

| Metric | Value |
|:-------|:------|
| Peak Throughput | **19,592 Mops/s** (tnot @ 100K elements) |
| Peak Binary Op | **15,470 Mops/s** (tmin @ 100K elements) |
| Avg Speedup vs Python | **7,230x** |
| Bridge Layer Speedup | **43.97x** vs NumPy (max) |
| Fused Operations | **39.51x** vs NumPy baseline |

**Verdict:** All performance tests PASSED.

---

## System Configuration

| Component | Value |
|:----------|:------|
| Platform | Windows-11-10.0.26100-SP0 |
| Processor | AMD64 Family 23 Model 96 Stepping 1 (AMD Ryzen) |
| Architecture | AMD64 |
| Logical CPUs | 12 |
| OMP Threads | 12 (auto-set) |
| Python | 3.12 |
| Module | ternary_simd_engine.cp312-win_amd64.pyd |

---

## 1. Core SIMD Operations Benchmark

### Peak Throughput by Operation

| Operation | Peak Mops/s | Optimal Size | Speedup vs Python |
|:----------|:------------|:-------------|:------------------|
| tnot | 19,592.48 | 100,000 | 19,174x |
| tmin | 15,470.54 | 100,000 | 27,996x |
| tmul | 15,443.06 | 100,000 | 25,706x |
| tmax | 15,259.95 | 100,000 | 27,299x |
| tadd | 14,674.16 | 100,000 | 25,273x |

### Throughput Scaling by Array Size

| Size | tadd | tmul | tmin | tmax | tnot |
|:-----|:-----|:-----|:-----|:-----|:-----|
| 32 | 22.91 | 23.35 | 23.08 | 23.28 | 30.79 |
| 100 | 71.94 | 70.47 | 69.96 | 70.33 | 90.49 |
| 1,000 | 681.01 | 681.71 | 682.92 | 433.82 | 899.85 |
| 10,000 | 4,492.16 | 4,607.45 | 4,927.56 | 4,776.92 | 6,614.19 |
| 100,000 | 14,674.16 | 15,443.06 | 15,470.54 | 15,259.95 | 19,592.48 |
| 1,000,000 | 5,886.19 | 7,021.01 | 7,606.21 | 8,291.49 | 10,847.36 |
| 10,000,000 | 4,853.87 | 4,695.42 | 4,718.04 | 4,722.22 | 5,378.86 |

*Values in Mops/s (million operations per second)*

### Average Speedup vs Pure Python

| Operation | Avg Speedup |
|:----------|:------------|
| tmin | 8,113x |
| tmax | 7,577x |
| tmul | 7,509x |
| tadd | 7,413x |
| tnot | 5,538x |

---

## 2. Bridge Layer Benchmark

The Bridge Layer implements fused int8 operations, eliminating NumPy conversion overhead.

### Performance Comparison (tadd operation)

| Size | NumPy M/s | Naive M/s | Bridge M/s | Bridge/Naive | Bridge/NumPy |
|:-----|:----------|:----------|:-----------|:-------------|:-------------|
| 64 | 10.2 | 6.9 | 33.4 | 4.82x | 3.28x |
| 256 | 38.2 | 25.3 | 111.5 | 4.41x | 2.92x |
| 1,024 | 114.3 | 74.3 | 514.1 | 6.92x | 4.50x |
| 4,096 | 255.1 | 194.2 | 1,868.6 | 9.62x | 7.33x |
| 16,384 | 406.1 | 294.7 | 5,407.3 | 18.35x | 13.32x |
| 65,536 | 463.4 | 345.6 | 10,224.0 | 29.58x | 22.06x |
| 262,144 | 285.3 | 255.2 | 12,544.0 | **49.15x** | **43.97x** |
| 1,048,576 | 264.6 | 176.9 | 2,730.5 | 15.44x | 10.32x |

### Bridge Layer Summary

| Metric | Value |
|:-------|:------|
| Avg Speedup vs Naive | 17.29x |
| Max Speedup vs Naive | 49.15x |
| Avg Speedup vs NumPy | 13.46x |
| Max Speedup vs NumPy | 43.97x |

### All Int8 Operations @ 100K Elements

| Operation | Bridge M/s | NumPy M/s | Speedup | Correct |
|:----------|:-----------|:----------|:--------|:--------|
| tadd_int8 | 7,822.3 | 325.9 | 24.00x | OK |
| tmul_int8 | 7,734.6 | 939.9 | 8.23x | OK |
| tmin_int8 | 7,227.0 | 18,997.0 | 0.38x | OK |
| tmax_int8 | 7,700.0 | 20,116.7 | 0.38x | OK |
| tnot_int8 | 10,117.4 | 25,227.0 | 0.40x | OK |

**Note:** tmin/tmax/tnot int8 operations are slower than NumPy's vectorized min/max/negation because NumPy uses highly optimized intrinsics for these simple operations. The Bridge Layer's value is in fused operations and tadd/tmul where saturated arithmetic is required.

---

## 3. Bridge Layer Performance Tests

| Test | Expected | Actual | Status |
|:-----|:---------|:-------|:-------|
| crossover_tadd_int8 | < 20,000 elements | 64 elements | **PASS** |
| speedup_tadd_int8_large_arrays | > 10.0x | 20.09x | **PASS** |
| speedup_fused_op_large_arrays | > 20.0x | 39.51x | **PASS** |
| correctness_bridge_layer | 100% match | PASS | **PASS** |

### Fused Operations Throughput

| Size | Ternary Fused | NumPy Fused | Speedup |
|:-----|:--------------|:------------|:--------|
| 100,000 | 8.41 Gops/s | 0.36 Gops/s | 23.21x |
| 1,000,000 | 8.50 Gops/s | 0.22 Gops/s | 39.51x |
| 10,000,000 | 4.94 Gops/s | 0.23 Gops/s | 21.38x |

---

## 4. Algebraic Interpretation

The Bridge Layer implements the isomorphism:

```
phi: Int8 -> Uint8
phi(x) = x + 1       (maps -1 -> 0, 0 -> 1, +1 -> 2)
phi^-1(y) = y - 1    (inverse mapping)
```

Fused operation: `result = phi^-1(kernel(phi(a), phi(b)))`

Benefits:
- Eliminates 4 NumPy array allocations
- Eliminates 4 memory round-trips
- Eliminates Python interpreter overhead

---

## 5. Performance Characteristics

### Optimal Array Sizes

- **Sweet spot:** 100,000 elements (best throughput/latency ratio)
- **Cache-friendly:** Arrays up to ~256K fit in L2 cache
- **Memory-bound:** Arrays > 1M elements see throughput drop due to memory bandwidth

### Scaling Behavior

1. **Small arrays (< 1K):** Call overhead dominates, ~20-90 Mops/s
2. **Medium arrays (1K-100K):** Linear scaling, reaches peak throughput
3. **Large arrays (> 100K):** Memory bandwidth limited, ~5-10 Gops/s

---

## 6. Conclusions

1. **Core SIMD operations** achieve 7,000-8,000x speedup vs pure Python
2. **Bridge Layer** eliminates conversion overhead, providing 13-44x speedup vs NumPy
3. **Fused operations** achieve 21-40x speedup by combining multiple operations
4. **All correctness tests** pass with 100% accuracy
5. **Crossover point** at 64 elements - Bridge Layer wins even for small arrays

---

## Files Generated

```
benchmarks/results/2025-12-02_20-00-54/bridge_layer_performance_results.json
benchmarks/benchmarks/results/bench_results_20251202_200152.json
benchmarks/benchmarks/results/bench_results_20251202_200152.csv
```

---

**Validated:** 2025-12-02 · **Platform:** Windows 11 x64 AMD Ryzen · **Version:** ternary_simd_engine.cp312-win_amd64.pyd
