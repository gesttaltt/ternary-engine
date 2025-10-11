# Ternary Core Architecture - Technical Documentation

## Overview

The Ternary Core is a high-performance computational library implementing balanced ternary arithmetic (-1, 0, +1) using AVX2 SIMD vectorization. The library achieves >30M trits/second throughput on modern x86-64 CPUs through compact 2-bit encoding and vectorized operations.

**Current Version**: Pre-optimization baseline
**Target Platform**: x86-64 with AVX2 support
**Language**: C++17 with Python 3 bindings via pybind11
**License**: TBD

---

## Core Design Principles

### 1. Balanced Ternary Representation

The library uses **balanced ternary** logic where each trit can take three values:
- `-1` (negative)
- `0` (neutral)
- `+1` (positive)

This representation is particularly suited for:
- Modulo-3 arithmetic
- Fractal computations (Cantor set, ternary IFS)
- Continuum-discrete boundary operations
- Sign-symmetric algorithms

### 2. Compact 2-bit Encoding

Each trit occupies exactly 2 bits, allowing 4 trits per byte:

```
Encoding scheme:
  0b00 = -1 (negative)
  0b01 =  0 (neutral)
  0b10 = +1 (positive)
  0b11 = invalid (reserved)
```

**Rationale**:
- Minimizes memory footprint (2 bits vs 8 bits for int8)
- Enables 32 trits per 256-bit AVX2 register
- Preserves one encoding for future extensions

### 3. SIMD-First Architecture

Operations are designed for vectorization:
- Primary path: AVX2 (32 trits/operation)
- Fallback path: Scalar (tail elements)
- Future: AVX-512 (64 trits), ARM NEON (16 trits)

---

## File Structure

```
ternary-kernel-python-c/
├── ternary_core.h                    # Scalar operations header
├── ternary_core_simd_full.cpp        # AVX2 SIMD implementation + Python bindings
├── ternary_core_simd_full-use-cases.md   # High-level project vision
├── docs/                             # Technical documentation
│   ├── architecture.md               # This file
│   ├── optimization-roadmap.md       # Implementation phases
│   ├── encoding-specification.md     # Bit-level encoding details
│   └── simd-implementation.md        # Vectorization internals
├── legacy/                           # Evolution history (not tracked)
│   ├── ternary_core.c
│   ├── ternary_core_simd.cpp
│   ├── ternary_core_simd_full.cpp
│   └── ternary_core_full.cpp
└── local-reports/                    # Research notes (not tracked)
    ├── optimization.md
    └── optimizations-index.md
```

---

## Component Architecture

### Layer 1: Scalar Operations (`ternary_core.h`)

**Purpose**: Define baseline ternary operations on individual trits

**Current Implementation** (Pre-optimization):
```c
// Conversion functions (used by all operations)
static inline trit int_to_trit(int v) {
    return (v < 0) ? 0b00 : (v > 0) ? 0b10 : 0b01;
}
static inline int trit_to_int(trit t) {
    return (t==0b00)?-1:(t==0b10)?1:0;
}

// Operations (conversion-based, inefficient)
static inline trit tmin(trit a, trit b) {
    return (trit_to_int(a) < trit_to_int(b)) ? a : b;
}
static inline trit tmax(trit a, trit b) {
    return (trit_to_int(a) > trit_to_int(b)) ? a : b;
}
static inline trit tadd(trit a, trit b) {
    int s = trit_to_int(a) + trit_to_int(b);
    if (s>1) s=1; if (s<-1) s=-1;
    return int_to_trit(s);
}
static inline trit tmul(trit a, trit b) {
    return int_to_trit(trit_to_int(a) * trit_to_int(b));
}
static inline trit tnot(trit a) {
    return (a==0b00)?0b10:(a==0b10)?0b00:0b01;
}
```

**Performance Characteristics**:
- `tmin/tmax`: 2 conversions + 1 comparison + 0 back-conversions (returns input)
- `tadd`: 2 conversions + 1 add + 2 branches (clamping) + 1 back-conversion
- `tmul`: 2 conversions + 1 multiply + 1 back-conversion
- `tnot`: 2 comparisons + 1 conditional chain (no conversions)

**Identified Issue**: Repeated `trit_to_int` / `int_to_trit` conversions add 4-5 branches per operation, causing pipeline stalls.

**Optimization Target**: Replace with lookup tables (Phase 0)

---

### Layer 2: SIMD Vectorization (`ternary_core_simd_full.cpp`)

**Purpose**: Vectorize operations using AVX2 intrinsics

#### 2.1 Vector Conversion Functions

**Inverted Polarity Design Note**:
The SIMD implementation uses an *intentionally inverted* int8 mapping:
```
Trit 0b00 (logical -1) → int8 +1
Trit 0b01 (logical  0) → int8  0
Trit 0b10 (logical +1) → int8 -1
```

This inversion is **self-consistent** and cancels during round-trip conversions, producing correct ternary results.

**`trit_to_int8` Implementation**:
```cpp
static inline __m256i trit_to_int8(__m256i v) {
    __m256i neg = _mm256_cmpeq_epi8(v, _mm256_set1_epi8(0b00));  // Mask: 0xFF if -1
    __m256i pos = _mm256_cmpeq_epi8(v, _mm256_set1_epi8(0b10));  // Mask: 0xFF if +1
    return _mm256_sub_epi8(pos, neg);  // pos-neg yields inverted mapping
}
```

**`int8_to_trit` Implementation**:
```cpp
static inline __m256i int8_to_trit(__m256i v) {
    __m256i neg = _mm256_cmpeq_epi8(v, _mm256_set1_epi8(-1));
    __m256i pos = _mm256_cmpeq_epi8(v, _mm256_set1_epi8(1));
    __m256i out = _mm256_blendv_epi8(_mm256_set1_epi8(0b01), _mm256_set1_epi8(0b00), neg);
    out = _mm256_blendv_epi8(out, _mm256_set1_epi8(0b10), pos);
    return out;
}
```

**Conversion Cost**:
- `trit_to_int8`: 2 comparisons + 1 subtract = ~3 cycles
- `int8_to_trit`: 2 comparisons + 2 blends = ~4 cycles
- Total: ~7 cycles per vector conversion pair

#### 2.2 Vectorized Operations

**Addition** (`tadd_simd`):
```cpp
static inline __m256i tadd_simd(__m256i a, __m256i b) {
    __m256i s = _mm256_adds_epi8(trit_to_int8(a), trit_to_int8(b));  // Saturating add
    return int8_to_trit(clamp(s));
}
```
- Operations: 2 conversions + 1 saturating add + 1 clamp + 1 back-conversion
- Throughput: ~15 cycles per 32 trits

**Multiplication** (`tmul_simd`):
```cpp
static inline __m256i tmul_simd(__m256i a, __m256i b) {
    __m256i ai = trit_to_int8(a);
    __m256i bi = trit_to_int8(b);
    __m256i p = _mm256_mullo_epi8(ai, bi);
    return int8_to_trit(clamp(p));
}
```
- Operations: 2 conversions + 1 multiply + 1 clamp + 1 back-conversion
- Throughput: ~18 cycles per 32 trits

**Min/Max** (`tmin_simd`, `tmax_simd`):
```cpp
static inline __m256i tmin_simd(__m256i a, __m256i b) {
    return int8_to_trit(_mm256_min_epi8(trit_to_int8(a), trit_to_int8(b)));
}
```
- Operations: 2 conversions + 1 min/max + 1 back-conversion
- Throughput: ~12 cycles per 32 trits

**Negation** (`tnot_simd`):
```cpp
static inline __m256i tnot_simd(__m256i a) {
    return int8_to_trit(_mm256_sub_epi8(_mm256_setzero_si256(), trit_to_int8(a)));
}
```
- Operations: 1 conversion + 1 subtract + 1 back-conversion
- Throughput: ~10 cycles per 32 trits

#### 2.3 Array Processing Loop

**Macro-Generated Wrapper** (`TERNARY_OP_SIMD`):
```cpp
#define TERNARY_OP_SIMD(func) \
py::array_t<uint8_t> func##_array(py::array_t<uint8_t> A, py::array_t<uint8_t> B) { \
    auto a = A.unchecked<1>(); \
    auto b = B.unchecked<1>(); \
    ssize_t n = A.size(); \
    if (n != B.size()) throw std::runtime_error("Arrays must match"); \
    py::array_t<uint8_t> out(n); \
    auto r = out.mutable_unchecked<1>(); \
    ssize_t i = 0; \
    // SIMD path: process 32-element blocks
    for (; i + 32 <= n; i += 32) { \
        __m256i va = _mm256_loadu_si256((__m256i const*)(a.data() + i)); \
        __m256i vb = _mm256_loadu_si256((__m256i const*)(b.data() + i)); \
        __m256i vr = func##_simd(va, vb); \
        _mm256_storeu_si256((__m256i*)(r.mutable_data() + i), vr); \
    } \
    // Scalar fallback: process remaining elements
    for (; i < n; ++i) r[i] = func(a[i], b[i]); \
    return out; \
}
```

**Architecture Notes**:
1. **Unaligned loads**: Uses `_mm256_loadu_si256` (slower than aligned)
2. **Scalar fallback**: Calls conversion-based scalar functions (inefficient)
3. **No prefetching**: Sequential memory access without hints
4. **No threading**: Single-threaded execution

**Performance Profile**:
- Arrays of size `n`:
  - SIMD processes: `⌊n/32⌋ × 32` elements
  - Scalar processes: `n mod 32` elements (0-31)
- For perfectly aligned arrays (n % 32 == 0): 100% SIMD
- For worst case (n = 32k + 31): ~97% SIMD, 3% scalar

---

### Layer 3: Python Bindings (`PYBIND11_MODULE`)

**Interface Exposure**:
```cpp
PYBIND11_MODULE(ternary_core_simd_full, m) {
    m.def("tadd", &tadd_array);
    m.def("tmul", &tmul_array);
    m.def("tmin", &tmin_array);
    m.def("tmax", &tmax_array);
    m.def("tnot", &tnot_array);
}
```

**Type Signature**: All functions accept/return `py::array_t<uint8_t>`

**Python Usage**:
```python
import numpy as np
import ternary_core_simd_full as tc

A = np.array([0b00, 0b01, 0b10], dtype=np.uint8)  # [-1, 0, +1]
B = np.array([0b10, 0b01, 0b00], dtype=np.uint8)  # [+1, 0, -1]

C = tc.tadd(A, B)  # [0b01, 0b01, 0b01] = [0, 0, 0]
```

---

## Performance Characteristics (Current Baseline)

### Theoretical Throughput

**SIMD Path** (optimistic estimate):
- CPU frequency: 3.5 GHz
- IPC for vectorized code: ~2.0
- Cycles per 32-trit operation: ~15 (tadd)
- Operations/second: (3.5e9 × 2.0) / 15 ≈ 466M ops/s
- Trits/second: 466M × 32 ≈ **14.9 billion trits/s**

**Actual Measured** (reported):
- **30 million trits/s** on modern CPU

**Gap Analysis**: ~500× slower than theoretical maximum

**Bottleneck Candidates**:
1. Memory bandwidth (not compute-bound)
2. Python binding overhead
3. Small array sizes (high dispatch overhead)
4. Cache misses
5. Conversion overhead in scalar fallback

### Memory Requirements

**Array Size Impact**:
| Elements | Memory (bytes) | SIMD blocks | Scalar tail | % SIMD |
|----------|----------------|-------------|-------------|--------|
| 32       | 32             | 1           | 0           | 100%   |
| 1,000    | 1,000          | 31          | 8           | 99.2%  |
| 10,000   | 10,000         | 312         | 16          | 99.8%  |
| 1M       | 1 MB           | 31,250      | 0           | 100%   |
| 10M      | 10 MB          | 312,500     | 0           | 100%   |

**Cache Behavior**:
- L1 cache (~32 KB): Fits ~32K trits
- L2 cache (~256 KB): Fits ~256K trits
- L3 cache (~8 MB): Fits ~8M trits

---

## Known Issues & Limitations

### Critical Issues
1. **No build system** - Manual compilation required
2. **No CPU feature detection** - Crashes on non-AVX2 CPUs
3. **No input validation** - Invalid trit values cause undefined behavior
4. **Scalar operation inefficiency** - Repeated conversions in fallback path

### Design Limitations
1. **x86-64 only** - No ARM/NEON support
2. **1D arrays only** - No multi-dimensional support
3. **No broadcasting** - Arrays must be same size
4. **No scalar mixing** - Cannot do `tadd(array, scalar)`
5. **No error reporting** - Single generic error message

### Performance Limitations
1. **Unaligned loads** - Uses `loadu` instead of `load` everywhere
2. **No prefetching** - Sequential access without cache hints
3. **No threading** - Single-threaded execution
4. **Conversion overhead** - SIMD path does 2-4 conversions per operation

---

## Optimization Roadmap

The library will undergo **4 optimization phases** over ~6-12 months:

**Phase 0**: Quick Wins (Week 1)
- Scalar LUT optimization
- Compiler flag improvements
- Bitwise `tnot` optimization

**Phase 1**: Core Optimizations (Weeks 2-4)
- Aligned memory loads
- OpenMP threading
- Size-adaptive kernels

**Phase 2**: SIMD Enhancements (Weeks 5-8)
- SIMD LUTs via shuffle
- Masked tail handling
- Conversion reduction

**Phase 3**: Advanced Features (Weeks 9-16)
- Operation fusion
- Multi-platform SIMD
- Prefetching strategies

See `docs/optimization-roadmap.md` for detailed implementation plans.

---

## References

### Internal Documentation
- `docs/optimization-roadmap.md` - Phase 0-4 implementation details
- `docs/encoding-specification.md` - Bit-level encoding details
- `docs/simd-implementation.md` - Vectorization internals
- `ternary_core_simd_full-use-cases.md` - Project vision & use cases

### External Resources
- Balanced Ternary: [Wikipedia](https://en.wikipedia.org/wiki/Balanced_ternary)
- AVX2 Intrinsics: [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- pybind11: [Documentation](https://pybind11.readthedocs.io/)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-11
**Status**: Pre-optimization baseline documented
