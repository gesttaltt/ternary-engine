# Source Code Overview

## Introduction

This document provides a high-level overview of the pure source code files in the ternary-engine library and guides you through understanding the implementation.

---

## Core Source Files

The library consists of **five primary components** that implement the complete balanced ternary logic system with fusion operations:

### 1. `ternary_lut_gen.h` - Compile-Time LUT Generation Framework

**Purpose**: Constexpr LUT generation infrastructure (OPT-AUTO-LUT)

**Key Components**:
- `make_binary_lut()` - Template for 16-entry binary operation LUTs
- `make_unary_lut()` - Template for 4-entry unary operation LUTs
- `make_unary_lut_padded()` - Template for 16-entry padded unary LUTs
- Constexpr conversion helpers

**Size**: ~80 lines
**Dependencies**: `<array>`, `<cstdint>`
**Benefit**: Compile-time generation, reduced maintenance overhead, zero runtime cost

### 2. `ternary_algebra.h` - Foundation Layer

**Purpose**: Core definitions and constexpr-generated scalar operations

**Key Components**:
- Trit encoding system (2-bit representation)
- Constexpr-generated lookup tables (LUTs) for all operations
- Force-inlined scalar operation implementations
- Conversion and packing utilities

**Documentation**: [`docs/ternary-engine-header.md`](./ternary-engine-header.md)

**Size**: 108 lines
**Dependencies**: `stdint.h`, `ternary_lut_gen.h`
**Performance**: 3-10x faster than conversion-based approach

### 3. `ternary_simd_engine.cpp` - Acceleration Layer

**Purpose**: AVX2-vectorized array operations with Python bindings

**Key Components**:
- SIMD implementations using `_mm256_shuffle_epi8`
- Template-based unified processing with optional masking (OPT-HASWELL-02)
- OpenMP parallelization for large arrays (OPT-001)
- Pybind11 Python integration
- Centralized error handling via `ternary_errors.h`

**Documentation**: [`docs/ternary-engine-simd.md`](./ternary-engine-simd.md)

**Size**: 331 lines
**Dependencies**: `immintrin.h`, `pybind11`, `omp.h`, `ternary_algebra.h`, `ternary_errors.h`
**Performance**: 100x faster than pure Python

### 4. `ternary_errors.h` - Error Handling

**Purpose**: Domain-specific exception types for ternary operations

**Location**: `src/core/common/ternary_errors.h`

**Key Components**:
- `TernaryError` - Base exception class
- `ArraySizeMismatchError` - Binary operation size validation
- `InvalidTritError` - Trit value validation (reserved)
- `AllocationError` - Memory allocation failures (reserved)

**Documentation**: [`docs/error-handling.md`](./error-handling.md)

**Size**: 120 lines
**Dependencies**: `<stdexcept>`, `<string>`
**Design Principle**: YAGNI - minimal exception set, expand only when needed

### 5. Fusion Operations (Integrated in Main Module)

**Purpose**: Fused operation chains for improved performance

**Location**: `src/core/simd/ternary_fusion.h`

**Key Operations**:
- `fused_tnot_tadd(a, b)` - Single-pass tnot(tadd(a, b))
- `fused_tnot_tmul(a, b)` - Single-pass tnot(tmul(a, b))
- `fused_tnot_tmin(a, b)` - Single-pass tnot(tmin(a, b))
- `fused_tnot_tmax(a, b)` - Single-pass tnot(tmax(a, b))

**Performance**: 1.53-11.26× speedup vs separate operations (eliminates intermediate memory traffic)

**Note**: Previously in separate `ternary_fusion_engine` module, now integrated into main `ternary_simd_engine` module (2025-11-22 consolidation)

---

## Architecture Layers

```
┌─────────────────────────────────────────────────┐
│           Python Application Layer              │
│         (NumPy arrays, high-level API)          │
└────────────────────┬────────────────────────────┘
                     │ pybind11
┌────────────────────▼────────────────────────────┐
│      ternary_simd_engine.cpp (331 lines)        │
│  ┌──────────────────────────────────────────┐  │
│  │  Python Bindings (pybind11)              │  │
│  │  • PYBIND11_MODULE definitions           │  │
│  └──────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────┐  │
│  │  Error Handling (ternary_errors.h)      │  │
│  │  • ArraySizeMismatchError validation     │  │
│  └──────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────┐  │
│  │  Template Processing Layer               │  │
│  │  • process_binary_array<Sanitize>        │  │
│  │  • process_unary_array<Sanitize>         │  │
│  │  (OPT-HASWELL-02: optional masking)      │  │
│  └──────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────┐  │
│  │  Execution Path Selection                │  │
│  │  PATH 1: OpenMP (n >= 100K)              │  │
│  │  PATH 2: Serial SIMD (n < 100K)          │  │
│  │  PATH 3: Scalar tail (remaining)         │  │
│  └──────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────┐  │
│  │  SIMD Operations (AVX2)                  │  │
│  │  • tadd_simd<>(), tmul_simd<>(), ...     │  │
│  │  • _mm256_shuffle_epi8 (32 parallel)     │  │
│  │  • maybe_mask<Sanitize>() helper         │  │
│  └────────────┬─────────────────────────────┘  │
└───────────────┼─────────────────────────────────┘
                │ #include
┌───────────────▼─────────────────────────────────┐
│       ternary_algebra.h (108 lines)             │
│  ┌──────────────────────────────────────────┐  │
│  │  Constexpr LUT Generation Framework      │  │
│  │  (uses ternary_lut_gen.h)                │  │
│  └──────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────┐  │
│  │  Lookup Tables (Constexpr-Generated)     │  │
│  │  • TADD_LUT = make_binary_lut(λ)         │  │
│  │  • TMUL_LUT, TMIN_LUT, TMAX_LUT          │  │
│  │  • TNOT_LUT = make_unary_lut(λ)          │  │
│  │  (OPT-AUTO-LUT: algorithm-as-docs)       │  │
│  └──────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────┐  │
│  │  Scalar Operations                       │  │
│  │  • tadd(), tmul(), tmin(), tmax()        │  │
│  │  • tnot()                                │  │
│  │  (force-inlined, LUT-based)              │  │
│  └──────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────┐  │
│  │  Type Definitions & Utilities            │  │
│  │  • typedef uint8_t trit                  │  │
│  │  • int_to_trit(), trit_to_int()          │  │
│  │  • pack_trits(), unpack_trit()           │  │
│  └──────────────────────────────────────────┘  │
└────────────────┬────────────────────────────────┘
                 │ #include
┌────────────────▼────────────────────────────────┐
│       ternary_lut_gen.h (~80 lines)             │
│  ┌──────────────────────────────────────────┐  │
│  │  Constexpr LUT Generation Templates      │  │
│  │  • make_binary_lut<Func>(λ) → [16]      │  │
│  │  • make_unary_lut<Func>(λ) → [4]        │  │
│  │  • make_unary_lut_padded<Func>(λ) → [16]│  │
│  └──────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────┐  │
│  │  Constexpr Conversion Helpers            │  │
│  │  • trit_to_int_constexpr()               │  │
│  │  • int_to_trit_constexpr()               │  │
│  │  • clamp_ternary()                       │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

---

## Operation Flow Example

Let's trace a simple operation: `result = tc.tadd(a, b)` where `a` and `b` are 100,000-element arrays.

### Step 1: Python Call
```python
import ternary_simd_engine as tc
import numpy as np

a = np.array([0, 1, 2] * 33334, dtype=np.uint8)  # 100,002 elements
b = np.array([2, 1, 0] * 33334, dtype=np.uint8)

result = tc.tadd(a, b)  # Calls into C++
```

### Step 2: Python Binding (ternary_simd_engine.cpp:303-305)
```cpp
py::array_t<uint8_t> tadd_array(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return process_binary_array<true>(A, B, tadd_simd<true>, tadd);
}
```

### Step 3: Template Instantiation (ternary_simd_engine.cpp:196-249)
```cpp
template <bool Sanitize = true, typename SimdOp, typename ScalarOp>
py::array_t<uint8_t> process_binary_array(
    py::array_t<uint8_t> A,
    py::array_t<uint8_t> B,
    SimdOp simd_op,     // = tadd_simd<true>
    ScalarOp scalar_op  // = tadd (from ternary_algebra.h)
)
```

### Step 4: Path Selection
```
n = 100,002 elements
n >= 100,000? YES → PATH 1: OpenMP Parallel
```

### Step 5: Parallel SIMD Processing (ternary_simd_engine.cpp:223-229)
```cpp
// Process 100,000 elements (3,125 blocks of 32) in parallel
#pragma omp parallel for schedule(static)
for (ssize_t idx = 0; idx < 100000; idx += 32) {
    // Each iteration:
    __m256i va = _mm256_loadu_si256(...);       // Load 32 trits from a
    __m256i vb = _mm256_loadu_si256(...);       // Load 32 trits from b
    __m256i vr = simd_op(va, vb);               // 32 parallel additions
    _mm256_storeu_si256(..., vr);               // Store 32 results
}
```

### Step 6: SIMD Operation (ternary_simd_engine.cpp:101-115)
```cpp
template <bool Sanitize = true>
static inline __m256i tadd_simd(__m256i a, __m256i b) {
    // Optional masking (OPT-HASWELL-02)
    __m256i a_masked = maybe_mask<Sanitize>(a);
    __m256i b_masked = maybe_mask<Sanitize>(b);

    // Build indices: (a << 2) | b
    __m256i a_shifted = _mm256_add_epi8(_mm256_add_epi8(a_masked, a_masked),
                                         _mm256_add_epi8(a_masked, a_masked));
    __m256i indices = _mm256_or_si256(a_shifted, b_masked);

    // Load constexpr-generated TADD_LUT from ternary_algebra.h
    __m256i lut = broadcast_lut_16(TADD_LUT.data());

    // 32 parallel lookups
    return _mm256_shuffle_epi8(lut, indices);
}
```

### Step 7: Scalar Tail (ternary_simd_engine.cpp:244-246)
```cpp
// Process remaining 2 elements
for (; i < 100002; ++i) {
    r[i] = scalar_op(a[i], b[i]);  // Scalar operation from ternary_algebra.h
}
```

### Step 8: Scalar LUT Lookup (ternary_algebra.h)
```cpp
static FORCE_INLINE trit tadd(trit a, trit b) {
    return TADD_LUT[(a << 2) | b];  // Direct constexpr-generated LUT access
}

// TADD_LUT is constexpr-generated at compile time:
constexpr auto TADD_LUT = make_binary_lut([](uint8_t a, uint8_t b) -> uint8_t {
    int sa = trit_to_int_constexpr(a);
    int sb = trit_to_int_constexpr(b);
    int sum = sa + sb;
    return int_to_trit_constexpr(clamp_ternary(sum));
});
```

### Result
- **100,000 elements**: Processed in parallel using AVX2 SIMD
- **2 remaining elements**: Processed using scalar LUT operations
- **Total time**: ~1-2 milliseconds (100x faster than Python)

---

## Key Design Principles

### 1. Lookup Table Optimization (Phase 0)

**Before** (conversion-based):
```c
int sum = trit_to_int(a) + trit_to_int(b);  // 2 conversions
if (sum > 1) sum = 1;                        // Branch
if (sum < -1) sum = -1;                      // Branch
return int_to_trit(sum);                     // 1 conversion
```

**After** (LUT-based):
```c
return TADD_LUT[(a << 2) | b];  // Single constexpr-generated LUT access
```

**Speedup**: 3-10x (eliminates conversions and branches)

### 1.5. Constexpr LUT Generation (OPT-AUTO-LUT)

**Before** (manual LUTs):
```c
static const uint8_t TADD_LUT[16] = {
    0b00, 0b00, 0b01, 0b00,
    0b00, 0b01, 0b10, 0b00,
    // ... manual maintenance burden
};
```

**After** (constexpr generation):
```cpp
constexpr auto TADD_LUT = make_binary_lut([](uint8_t a, uint8_t b) -> uint8_t {
    int sa = trit_to_int_constexpr(a);
    int sb = trit_to_int_constexpr(b);
    int sum = sa + sb;
    return int_to_trit_constexpr(clamp_ternary(sum));
});
```

**Benefits**:
- Algorithm is the documentation (centralized definition)
- Zero runtime cost (evaluated at compile time)
- Auditability (algebraic rules visible in code)
- Long-term maintainability

### 2. SIMD Vectorization (Phase 0.5)

**Technique**: Use `_mm256_shuffle_epi8` for 32 parallel LUT lookups

**Why not arithmetic SIMD?**
- Ternary operations don't map cleanly to integer arithmetic
- LUT approach unifies semantic domain (no conversions)
- Shuffle is as fast as arithmetic on modern CPUs

**Speedup**: 5-10x over scalar LUT

### 3. Template-Based Unification (Phase 2)

**Problem** (Phase 1): 6 execution paths per operation
- Aligned vs unaligned loads (2 variants)
- Manual unrolling (2x, 4x variants)
- OpenMP vs serial

**Solution** (Phase 2): Single template handles all operations
```cpp
template <bool Sanitize = true, typename SimdOp, typename ScalarOp>
py::array_t<uint8_t> process_binary_array(...) {
    // Universal processing logic with optional masking (OPT-HASWELL-02)
}
```

**Result**: 73% code reduction, <5% performance loss

### 3.5. Optional Input Sanitization (OPT-HASWELL-02)

**Technique**: Template-based compile-time masking control

```cpp
template <bool Sanitize = true>
static inline __m256i maybe_mask(__m256i v) {
    if constexpr (Sanitize)
        return _mm256_and_si256(v, _mm256_set1_epi8(0x03));
    else
        return v;
}
```

**Benefits**:
- Production use: `Sanitize=true` (default, safe)
- Advanced use: `Sanitize=false` (3-5% gain for validated pipelines)
- Zero runtime overhead (resolved at compile time)

### 4. Phase Coherence Philosophy

**Goal**: Reduce complexity while maintaining performance

**Eliminated Optimizations**:
- ❌ Aligned vs unaligned branching (~1% gain, 50% code increase)
- ❌ Manual loop unrolling (~2% gain, compiler does better)
- ✅ OpenMP parallelization (10x gain for large arrays, justified)

**Outcome**: 3 clean execution paths instead of 6 complex ones

---

## Performance Summary

### Throughput Comparison (10M elements)

| Implementation                    | Time    | Throughput      | Speedup |
|-----------------------------------|---------|-----------------|---------|
| Python (reference.py)             | 100 ms  | 100 ME/s        | 1x      |
| C++ naive (reference_cpp.cpp)     | 30 ms   | 333 ME/s        | 3x      |
| C++ LUT (ternary_algebra.h)       | 5 ms    | 2000 ME/s       | 20x     |
| C++ SIMD (ternary_simd_engine)    | 1 ms    | 10,000 ME/s     | 100x    |

*(ME/s = Million Elements per second)*

### Operation Breakdown (per element)

| Layer                      | Time       | Cycles | Operations                       |
|----------------------------|------------|--------|----------------------------------|
| Python reference           | 10 ns      | ~30    | Python loop + conversions        |
| C++ conversion-based       | 3 ns       | ~10    | Conversions + branches           |
| C++ LUT scalar             | 0.5 ns     | ~2     | Single array access (L1 cache)   |
| C++ SIMD (amortized)       | 0.1 ns     | ~0.3   | 32 elements / 10 cycles          |

---

## Reading Guide

### For Understanding the Implementation

1. **Start here**: Current document (overview)
2. **Core concepts**: [`docs/ternary-engine-header.md`](./ternary-engine-header.md)
   - Trit encoding
   - LUT design
   - Scalar operations
3. **Acceleration**: [`docs/ternary-engine-simd.md`](./ternary-engine-simd.md)
   - SIMD techniques
   - Template design
   - Execution paths
4. **Context**: [`docs/optimization-complexity-rationale.md`](./optimization-complexity-rationale.md)
   - Why certain optimizations were removed
   - Code simplification philosophy

### For Modifying the Code

1. **Adding new operations**:
   - Define constexpr LUT lambda in `ternary_algebra.h` using `make_binary_lut()` or `make_unary_lut()`
   - Add scalar function to `ternary_algebra.h` (force-inlined, LUT-based)
   - Add SIMD template function to `ternary_simd_engine.cpp`
   - Add wrapper function to `ternary_simd_engine.cpp`
   - Add Python binding to `PYBIND11_MODULE` in `ternary_simd_engine.cpp`

2. **Optimizing performance**:
   - See [`docs/ternary-engine-simd.md`](./ternary-engine-simd.md) § "Future Optimizations"
   - Profile first: `python benchmarks/bench_phase0.py`
   - Consider PGO: [`docs/PGO_README.md`](./PGO_README.md)

3. **Porting to new architectures**:
   - ARM/NEON: Replace AVX2 intrinsics in `ternary_simd_engine.cpp`
   - Keep `ternary_algebra.h` and `ternary_lut_gen.h` unchanged (portable)

### For Understanding the Evolution

1. **Design rationale**: [`docs/optimization-complexity-rationale.md`](./optimization-complexity-rationale.md)
2. **Architecture overview**: [`docs/architecture.md`](./architecture.md)
3. **Historical context**: [`docs/optimization-roadmap.md`](./optimization-roadmap.md)

---

## Testing the Code

### Correctness Tests

```bash
python tests/test_phase0.py
```

Validates SIMD operations against scalar reference.

### Performance Benchmarks

```bash
python benchmarks/bench_phase0.py
```

Measures throughput across different array sizes.

### OpenMP Scaling Test

```bash
python tests/test_omp.py
```

Verifies parallel scaling on multi-core systems.

---

## Building the Code

### Standard Build

```bash
python build/build.py
```

Produces `ternary_simd_engine.cp312-win_amd64.pyd` (or `.so` on Linux) with fusion operations included.

### With Profile-Guided Optimization

```bash
python build/build_pgo.py full
```

See [`docs/PGO_README.md`](../PGO_README.md) for details.

---

## File Dependencies

```
ternary_lut_gen.h
    ↓
    │ #include
    │
ternary_algebra.h
    ↓
    │ #include (+ ternary_errors.h)
    │
ternary_simd_engine.cpp
    ↓
    compiled via pybind11
    ↓
ternary_simd_engine.pyd/.so
    ↓
    imported by Python
    ↓
Python application
```

**No circular dependencies**: Clean, linear dependency structure with constexpr LUT generation at compile time.

---

## License

Both source files are licensed under Apache 2.0. See `LICENSE` and `NOTICE` files.

---

## Contributing

When modifying the source code:

1. **Maintain code simplification**: Only add complexity if it provides >10% performance gain
2. **Update documentation**: Keep this doc and component docs in sync
3. **Add tests**: Update `tests/test_phase0.py` for new operations
4. **Benchmark**: Run `benchmarks/bench_phase0.py` before/after changes
5. **Document optimization IDs**: Use OPT-XXX tags for traceability

---

## Summary

The ternary-engine library achieves 100x speedups through:

1. **Constexpr LUT generation** (`ternary_lut_gen.h`): Compile-time generation with zero runtime cost (OPT-AUTO-LUT)
2. **LUT-based operations** (`ternary_algebra.h`): Eliminates conversion overhead
3. **SIMD parallelization** (`ternary_simd_engine.cpp`): 32 operations per instruction
4. **Template-based design**: Code reuse without performance cost (Phase 2)
5. **Optional masking** (OPT-HASWELL-02): Compile-time sanitization control
6. **OpenMP threading** (OPT-001): Scales to multiple cores for large arrays (≥100K elements)
7. **Centralized error handling** (`ternary_errors.h`): Domain-specific exceptions with YAGNI principle
8. **Code simplification**: Maximum simplicity for stable performance

Four core files, ~640 lines of code, 100x performance improvement.
