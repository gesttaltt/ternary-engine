# Gops/s Comparative Benchmark Suite

**Doc-Type:** Benchmark Documentation · Version 1.0 · Updated 2025-12-02

---

## Overview

This benchmark suite measures **true kernel performance in Gops/s** (billions of operations per second) by comparing:

1. **Ternary SIMD Operations** - Our LUT-based AVX2 implementation
2. **Binary INT8 Operations** - Standard AVX2 integer arithmetic

The goal is honest measurement without Python interpreter overhead.

---

## Files

| File | Purpose |
|:-----|:--------|
| `bench_gops_comparative.cpp` | Main benchmark implementation |
| `include/bench_throughput.h` | C-compatible benchmark framework API |
| `build_gops_bench.py` | Python build script |

---

## Compilation

### Windows (Developer Command Prompt)

Open "Developer Command Prompt for VS 2022" and run:

```cmd
cd C:\path\to\ternary-engine\benchmarks\cpp-native-kernels
mkdir bin

cl /O2 /arch:AVX2 /std:c++17 /EHsc ^
   /I..\..\src ^
   /I.\include ^
   /Fe:bin\bench_gops.exe ^
   bench_gops_comparative.cpp
```

### Linux/macOS (GCC)

```bash
cd benchmarks/cpp-native-kernels
mkdir -p bin

g++ -O3 -march=native -mavx2 -std=c++17 \
    -I../../src \
    -I./include \
    bench_gops_comparative.cpp \
    -o bin/bench_gops
```

### Linux/macOS (Clang)

```bash
clang++ -O3 -march=native -mavx2 -std=c++17 \
    -I../../src \
    -I./include \
    bench_gops_comparative.cpp \
    -o bin/bench_gops
```

---

## Usage

```bash
./bin/bench_gops                # Full benchmark suite
./bin/bench_gops --quick        # Quick test (3 sizes)
./bin/bench_gops --csv          # CSV output
./bin/bench_gops --size=1000000 # Specific size only
```

---

## Methodology

### Fair Comparison

Both ternary and binary operations use:
- AVX2 256-bit vectors (32 bytes)
- 32 elements per vector operation
- Same memory access patterns

### Operations Compared

| Ternary | Binary Baseline | Notes |
|:--------|:----------------|:------|
| tadd (saturated add) | _mm256_add_epi8 | Wrapping vs saturated |
| tadd (saturated add) | _mm256_adds_epi8 | Both saturated |
| tmul | int8 mul (unpack/pack) | Ternary: LUT, Binary: complex |
| tmin | _mm256_min_epi8 | Direct comparison |
| tmax | _mm256_max_epi8 | Direct comparison |

### Metrics

| Metric | Formula | Unit |
|:-------|:--------|:-----|
| Throughput | elements / mean_time_ns | Gops/s |
| Bandwidth | 3 × elements / mean_time_ns | GB/s |
| CV | stddev / mean × 100 | % |

---

## Expected Results

### Ternary Strengths

- **tadd/tmul**: LUT-based operations can be faster than complex arithmetic
- **Memory efficiency**: 2-bit encoding vs 8-bit provides 4× compression
- **AI inference**: Memory bandwidth is often the bottleneck

### Binary Strengths

- **min/max**: Direct single-instruction operations
- **Simple add**: Single instruction vs 3 shuffles for ternary

### Key Insight

**Ternary's value is NOT raw throughput** - it's memory efficiency for AI models:
- 4× memory compression (2-bit vs 8-bit)
- Reduced memory bandwidth requirements
- Suitable for edge/mobile inference

---

## Architecture

### Ternary SIMD (3 shuffles per binary op)

```
Input: trits_a, trits_b (32 elements each)

1. Load canonical LUTs into registers
2. Dual-shuffle: contrib_a = shuffle(CANON_A, trits_a)
                 contrib_b = shuffle(CANON_B, trits_b)
3. Combine indices: indices = add(contrib_a, contrib_b)
4. Final shuffle: result = shuffle(OP_LUT, indices)

Output: 32 ternary results
```

### Binary INT8 (1 instruction per op)

```
Input: int8_a, int8_b (32 elements each)

1. result = _mm256_add_epi8(int8_a, int8_b)

Output: 32 INT8 results
```

---

## Statistical Rigor

Each benchmark reports:
- **Mean**: Average execution time
- **Stddev**: Standard deviation
- **Min/Max**: Best/worst case
- **P50/P95/P99**: Percentiles
- **CV**: Coefficient of variation (lower = more stable)

Warmup iterations (100) are excluded from measurements.

---

## Integration with Existing Benchmarks

This suite complements:
- `bench_kernels.cpp` - SIMD vs scalar comparison (ME/s)
- `bench_fusion.cpp` - Fused operation validation
- Python benchmarks - Higher-level integration tests

---

## Version History

| Date | Version | Description |
|:-----|:--------|:------------|
| 2025-12-02 | 1.0 | Initial Gops/s comparative benchmark suite |
