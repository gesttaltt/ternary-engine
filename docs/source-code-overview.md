# Source Code Overview

## Introduction

This document provides a high-level overview of the pure source code files in the ternary-kernel-python-c library and guides you through understanding the implementation.

---

## Core Source Files

The library consists of **two primary source files** that implement the complete balanced ternary logic system:

### 1. `ternary_core.h` - Foundation Layer

**Purpose**: Core definitions and scalar operations

**Key Components**:
- Trit encoding system (2-bit representation)
- Lookup tables (LUTs) for all operations
- Scalar operation implementations
- Conversion and packing utilities

**Documentation**: [`docs/ternary-core-header.md`](./ternary-core-header.md)

**Size**: 125 lines
**Dependencies**: `stdint.h` only
**Performance**: 3-10x faster than conversion-based approach

### 2. `ternary_core_simd_full.cpp` - Acceleration Layer

**Purpose**: AVX2-vectorized array operations with Python bindings

**Key Components**:
- SIMD implementations using `_mm256_shuffle_epi8`
- Template-based unified processing
- OpenMP parallelization for large arrays
- Pybind11 Python integration

**Documentation**: [`docs/ternary-core-simd.md`](./ternary-core-simd.md)

**Size**: 297 lines
**Dependencies**: `immintrin.h`, `pybind11`, `omp.h`, `ternary_core.h`
**Performance**: 100x faster than pure Python

---

## Architecture Layers

```
┌─────────────────────────────────────────────────┐
│           Python Application Layer              │
│         (NumPy arrays, high-level API)          │
└────────────────────┬────────────────────────────┘
                     │ pybind11
┌────────────────────▼────────────────────────────┐
│     ternary_core_simd_full.cpp (297 lines)     │
│  ┌──────────────────────────────────────────┐  │
│  │  Python Bindings (pybind11)              │  │
│  └──────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────┐  │
│  │  Template Processing Layer               │  │
│  │  • process_binary_array<>                │  │
│  │  • process_unary_array<>                 │  │
│  └──────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────┐  │
│  │  Execution Path Selection                │  │
│  │  PATH 1: OpenMP (n >= 100K)              │  │
│  │  PATH 2: Serial SIMD (n < 100K)          │  │
│  │  PATH 3: Scalar tail (n < 32)            │  │
│  └──────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────┐  │
│  │  SIMD Operations (AVX2)                  │  │
│  │  • tadd_simd(), tmul_simd(), ...         │  │
│  │  • _mm256_shuffle_epi8 (32 parallel)     │  │
│  └────────────┬─────────────────────────────┘  │
└───────────────┼─────────────────────────────────┘
                │ #include
┌───────────────▼─────────────────────────────────┐
│       ternary_core.h (125 lines)                │
│  ┌──────────────────────────────────────────┐  │
│  │  Lookup Tables (LUTs)                    │  │
│  │  • TADD_LUT[16], TMUL_LUT[16], ...       │  │
│  │  • TNOT_LUT[4]                           │  │
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
└─────────────────────────────────────────────────┘
```

---

## Operation Flow Example

Let's trace a simple operation: `result = tc.tadd(a, b)` where `a` and `b` are 100,000-element arrays.

### Step 1: Python Call
```python
import ternary_core_simd_full as tc
import numpy as np

a = np.array([0, 1, 2] * 33334, dtype=np.uint8)  # 100,002 elements
b = np.array([2, 1, 0] * 33334, dtype=np.uint8)

result = tc.tadd(a, b)  # Calls into C++
```

### Step 2: Python Binding (ternary_core_simd_full.cpp:269-271)
```cpp
py::array_t<uint8_t> tadd_array(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return process_binary_array(A, B, tadd_simd, tadd);
}
```

### Step 3: Template Instantiation (ternary_core_simd_full.cpp:165-215)
```cpp
template <typename SimdOp, typename ScalarOp>
py::array_t<uint8_t> process_binary_array(
    py::array_t<uint8_t> A,
    py::array_t<uint8_t> B,
    SimdOp simd_op,     // = tadd_simd
    ScalarOp scalar_op  // = tadd (from ternary_core.h)
)
```

### Step 4: Path Selection
```
n = 100,002 elements
n >= 100,000? YES → PATH 1: OpenMP Parallel
```

### Step 5: Parallel SIMD Processing (ternary_core_simd_full.cpp:186-198)
```cpp
// Process 100,000 elements (3,125 blocks of 32) in parallel
#pragma omp parallel for schedule(static)
for (ssize_t idx = 0; idx < 100000; idx += 32) {
    // Each iteration:
    __m256i va = _mm256_loadu_si256(...);  // Load 32 trits from a
    __m256i vb = _mm256_loadu_si256(...);  // Load 32 trits from b
    __m256i vr = tadd_simd(va, vb);        // 32 parallel additions
    _mm256_storeu_si256(..., vr);          // Store 32 results
}
```

### Step 6: SIMD Operation (ternary_core_simd_full.cpp:80-94)
```cpp
static inline __m256i tadd_simd(__m256i a, __m256i b) {
    // Build indices: (a << 2) | b
    __m256i indices = ...;

    // Load TADD_LUT from ternary_core.h
    __m256i lut = broadcast_lut_16(TADD_LUT);

    // 32 parallel lookups
    return _mm256_shuffle_epi8(lut, indices);
}
```

### Step 7: Scalar Tail (ternary_core_simd_full.cpp:210-212)
```cpp
// Process remaining 2 elements
for (; i < 100002; ++i) {
    r[i] = tadd(a[i], b[i]);  // Scalar operation from ternary_core.h
}
```

### Step 8: Scalar LUT Lookup (ternary_core.h:108-110)
```cpp
static FORCE_INLINE trit tadd(trit a, trit b) {
    return TADD_LUT[(a << 2) | b];  // Direct array access
}
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
return TADD_LUT[(a << 2) | b];  // Single array access
```

**Speedup**: 3-10x (eliminates conversions and branches)

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
template <typename SimdOp, typename ScalarOp>
py::array_t<uint8_t> process_binary_array(...) {
    // Universal processing logic
}
```

**Result**: 73% code reduction, <5% performance loss

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
| C++ LUT (ternary_core.h)          | 5 ms    | 2000 ME/s       | 20x     |
| C++ SIMD (ternary_core_simd_full) | 1 ms    | 10,000 ME/s     | 100x    |

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
2. **Core concepts**: [`docs/ternary-core-header.md`](./ternary-core-header.md)
   - Trit encoding
   - LUT design
   - Scalar operations
3. **Acceleration**: [`docs/ternary-core-simd.md`](./ternary-core-simd.md)
   - SIMD techniques
   - Template design
   - Execution paths
4. **Context**: [`docs/optimization-complexity-rationale.md`](./optimization-complexity-rationale.md)
   - Why certain optimizations were removed
   - Phase coherence philosophy

### For Modifying the Code

1. **Adding new operations**:
   - Add LUT to `ternary_core.h`
   - Add scalar function to `ternary_core.h`
   - Add SIMD function to `ternary_core_simd_full.cpp`
   - Add wrapper to `ternary_core_simd_full.cpp`
   - Add Python binding to `PYBIND11_MODULE`

2. **Optimizing performance**:
   - See [`docs/ternary-core-simd.md`](./ternary-core-simd.md) § "Future Optimizations"
   - Profile first: `python benchmarks/bench_phase0.py`
   - Consider PGO: [`docs/PGO_README.md`](./PGO_README.md)

3. **Porting to new architectures**:
   - ARM/NEON: Replace AVX2 intrinsics in `ternary_core_simd_full.cpp`
   - Keep `ternary_core.h` unchanged (portable)

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
python build/scripts/setup.py build_ext --inplace
```

Produces `ternary_core_simd_full.cp312-win_amd64.pyd` (or `.so` on Linux).

### With Profile-Guided Optimization

```bash
python build/scripts/setup_pgo.py
```

See [`docs/PGO_README.md`](./PGO_README.md) for details.

---

## File Dependencies

```
ternary_core.h
    ↑
    │ #include
    │
ternary_core_simd_full.cpp
    ↓
    compiled via pybind11
    ↓
ternary_core_simd_full.pyd/.so
    ↓
    imported by Python
    ↓
Python application
```

**No circular dependencies**: Clean, linear dependency structure.

---

## License

Both source files are licensed under Apache 2.0. See `LICENSE` and `NOTICE` files.

---

## Contributing

When modifying the source code:

1. **Maintain phase coherence**: Only add complexity if it provides >10% performance gain
2. **Update documentation**: Keep this doc and component docs in sync
3. **Add tests**: Update `tests/test_phase0.py` for new operations
4. **Benchmark**: Run `benchmarks/bench_phase0.py` before/after changes
5. **Document optimization IDs**: Use OPT-XXX tags for traceability

---

## Summary

The ternary-kernel-python-c library achieves 100x speedups through:

1. **LUT-based operations** (`ternary_core.h`): Eliminates conversion overhead
2. **SIMD parallelization** (`ternary_core_simd_full.cpp`): 32 operations per instruction
3. **Template-based design**: Code reuse without performance cost
4. **OpenMP threading**: Scales to multiple cores for large arrays
5. **Phase coherence**: Maximum simplicity for stable performance

Two files, 422 lines of code, 100x performance improvement.
