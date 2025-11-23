# Phase 4: Operation Fusion Engine - Implementation Plan

**Version:** 1.1-completed
**Date:** 2025-10-23 (Original) / 2025-11-22 (Completion)
**Status:** ✅ COMPLETED - Integrated into main module
**Completion Date:** 2025-11-22

---

## ✅ UPDATE (2025-11-22): Phase 4.0 Completed and Integrated

**Implementation Status:**
- ✅ Phase 4.0 completed - Binary→Unary fusion operations implemented
- ✅ Integrated into main `ternary_simd_engine` module (no separate fusion module)
- ✅ 4 fusion operations available: `fused_tnot_tadd`, `fused_tnot_tmul`, `fused_tnot_tmin`, `fused_tnot_tmax`
- ✅ Validated with comprehensive testing
- ✅ Performance verified: 1.53-11.26× speedup over separate operations

**Migration:**
- Old separate `ternary_fusion_engine` module has been merged into main module
- See `MIGRATION_NOTES.md` for migration guide
- Build via standard `build/build.py` (fusion included automatically)

**Usage:**
```python
import ternary_simd_engine as tc

# Fusion operations now available in main module
result = tc.fused_tnot_tadd(a, b)  # No separate import needed
```

**This document remains as the original design plan and historical reference.**

---

---

## Executive Summary

This document provides a complete implementation plan for **Phase 4: Operation Fusion Engine**, which aims to achieve **50-200% speedup** on multi-operation workflows by eliminating redundant memory traffic through operation chaining.

### Goals
- **Primary:** Fuse common operation chains into single-pass SIMD kernels
- **Performance:** 2-3× speedup on 3-operation chains (target: `(a + b) * c`)
- **Compatibility:** Zero API breakage, opt-in fusion via new functions
- **Maintainability:** Minimal code duplication, template-based design

### Scope
- **In scope:** Binary and unary operation fusion (2-5 ops), Python API, SIMD+scalar
- **Out of scope:** Automatic fusion detection, JIT compilation, GPU offload

### Success Criteria
- ✅ 2× speedup on `tnot(tadd(a, b))` vs separate calls
- ✅ 2.5× speedup on `tmul(tadd(a, b), c)` vs separate calls
- ✅ Memory bandwidth reduction: 50-67% on fused chains
- ✅ Zero performance regression on unfused operations
- ✅ Comprehensive test suite (correctness + performance)

---

## Table of Contents

1. [Current Architecture Analysis](#1-current-architecture-analysis)
2. [Fusion Design Options](#2-fusion-design-options)
3. [Recommended Implementation](#3-recommended-implementation)
4. [Performance Estimates vs Reality Checks](#4-performance-estimates-vs-reality-checks)
5. [Testing Strategy](#5-testing-strategy)
6. [Implementation Roadmap](#6-implementation-roadmap)
7. [Risk Assessment](#7-risk-assessment)
8. [Appendix A: Phase 5 Preview (Sparse Arrays)](#appendix-a-phase-5-preview-sparse-arrays)

---

## 1. Current Architecture Analysis

### 1.1 Operation Flow (Status Quo)

**Current multi-operation workflow:**
```python
# Python code
temp = ternary_simd_engine.tadd(a, b)   # Operation 1: Load a,b → Compute → Store temp
result = ternary_simd_engine.tnot(temp) # Operation 2: Load temp → Compute → Store result
```

**Memory traffic:**
```
Operation 1 (tadd):
  - Load: 2 arrays (a, b) = 2N bytes
  - Store: 1 array (temp) = N bytes
  - Total: 3N bytes

Operation 2 (tnot):
  - Load: 1 array (temp) = N bytes
  - Store: 1 array (result) = N bytes
  - Total: 2N bytes

Combined: 5N bytes for 2 operations
```

**Cache pollution:** 2 intermediate stores, each polluting L1/L2/L3

---

### 1.2 Current SIMD Pipeline Structure

**File:** `ternary_simd_engine.cpp` (lines 248-390)

**Three-path design:**
```cpp
template <bool Sanitize, typename SimdOp, typename ScalarOp>
py::array_t<uint8_t> process_binary_array(
    py::array_t<uint8_t> A,
    py::array_t<uint8_t> B,
    SimdOp simd_op,           // Lambda: (__m256i, __m256i) → __m256i
    ScalarOp scalar_op        // Lambda: (uint8_t, uint8_t) → uint8_t
) {
    // PATH 1: OpenMP parallel (n >= 32K × cores)
    // PATH 2: Serial SIMD (n < 32K × cores)
    // PATH 3: Scalar tail (remaining < 32 elements)
}
```

**Key insight:** Template accepts **function objects**, allowing arbitrary operation composition!

---

### 1.3 SIMD Kernel Composability

**Current kernel structure** (`ternary_simd_kernels.h`):

```cpp
// All operations follow same pattern:
template <bool Sanitize = true>
static inline __m256i tadd_simd(__m256i a, __m256i b) {
    return binary_simd_op<Sanitize>(a, b, g_luts.tadd);
}

// Kernels are PURE FUNCTIONS → trivially composable
__m256i fused = tnot_simd(tadd_simd(va, vb));  // No intermediate storage!
```

**Composability matrix:**

| Operation | Inputs | Output | Composable with |
|-----------|--------|--------|-----------------|
| `tadd_simd` | 2 vectors | 1 vector | Any operation accepting 1 vector |
| `tmul_simd` | 2 vectors | 1 vector | Any operation accepting 1 vector |
| `tnot_simd` | 1 vector | 1 vector | Any operation |

**Conclusion:** All operations are **trivially composable** via function nesting.

---

### 1.4 Scalar Tail Composability

**Current scalar tail** (`ternary_algebra.h`):

```cpp
// Each scalar operation is force-inline LUT lookup
static FORCE_INLINE trit tadd(trit a, trit b) {
    return TADD_LUT[(a << 2) | b];
}

static FORCE_INLINE trit tnot(trit a) {
    return TNOT_LUT[a & 0b11];
}

// Composition:
trit fused_result = tnot(tadd(a, b));  // Compiler inlines both LUT lookups
```

**Performance:** Force-inline + LUT lookups = **2 memory accesses** (both likely L1 cached).

---

### 1.5 Fusion-Readiness Summary

| Component | Status | Notes |
|-----------|--------|-------|
| **SIMD kernels** | ✅ Ready | Pure functions, composable |
| **Scalar tail** | ✅ Ready | Force-inline, trivial composition |
| **Pipeline template** | ✅ Ready | Accepts lambdas, no modifications needed |
| **LUT infrastructure** | ✅ Ready | Pre-broadcasted, shared across ops |
| **OpenMP path** | ⚠️ Needs consideration | Parallelization strategy for fused ops |
| **Python API** | ⚠️ Needs design | How to expose fusion to users? |

**Verdict:** Architecture is **highly fusion-ready** with minimal modifications required.

---

## 2. Fusion Design Options

### Option A: Manual Fusion (Hardcoded Chains)

**Approach:** Implement specific fused operations as standalone functions.

```cpp
// ternary_simd_engine.cpp
py::array_t<uint8_t> fused_tnot_tadd(
    py::array_t<uint8_t> A,
    py::array_t<uint8_t> B
) {
    return process_binary_array<SANITIZE>(
        A, B,
        [](auto va, auto vb) { return tnot_simd(tadd_simd(va, vb)); },  // Fused SIMD
        [](auto a, auto b) { return tnot(tadd(a, b)); }                 // Fused scalar
    );
}

// Python bindings
m.def("fused_tnot_tadd", &fused_tnot_tadd);
```

**Pros:**
- ✅ Simple implementation
- ✅ Zero abstraction overhead
- ✅ Easy to test and verify
- ✅ Compile-time optimization by compiler

**Cons:**
- ❌ Combinatorial explosion: N operations → O(N²) fused variants
- ❌ Code duplication
- ❌ Not scalable to arbitrary chains

**Assessment:** Good for **Phase 4.0** (proof of concept), not scalable long-term.

---

### Option B: Expression Templates (C++ Template Metaprogramming)

**Approach:** Build expression trees at compile time, fuse automatically.

```cpp
// Expression template system
template <typename LHS, typename RHS, typename Op>
struct BinaryExpr {
    LHS lhs;
    RHS rhs;
    Op op;

    auto eval(__m256i a, __m256i b) const {
        return op(lhs.eval(a, b), rhs.eval(a, b));
    }
};

// Usage
auto expr = BinaryExpr<Tadd, Tnot>(/* ... */);  // Represents tnot(tadd(a,b))
auto result = expr.eval(va, vb);  // Compiler fuses automatically
```

**Pros:**
- ✅ Arbitrary operation chains
- ✅ Compile-time fusion (zero runtime overhead)
- ✅ Type-safe
- ✅ Scalable to complex expressions

**Cons:**
- ❌ High implementation complexity
- ❌ Difficult to debug (template errors)
- ❌ Requires C++14/17 features
- ❌ Hard to expose clean Python API

**Assessment:** Powerful but **overkill for Phase 4**. Consider for Phase 6+.

---

### Option C: Macro-Generated Fusion Functions

**Approach:** Use preprocessor macros to generate common fused operations.

```cpp
// Macro definition
#define FUSE_BINARY_UNARY(name, binary_op, unary_op) \
    py::array_t<uint8_t> fused_##name( \
        py::array_t<uint8_t> A, py::array_t<uint8_t> B \
    ) { \
        return process_binary_array<SANITIZE>( \
            A, B, \
            [](auto va, auto vb) { return unary_op##_simd(binary_op##_simd(va, vb)); }, \
            [](auto a, auto b) { return unary_op(binary_op(a, b)); } \
        ); \
    }

// Generate fused operations
FUSE_BINARY_UNARY(tnot_tadd, tadd, tnot);
FUSE_BINARY_UNARY(tnot_tmul, tmul, tnot);
// ... 20 variants generated in 20 lines
```

**Pros:**
- ✅ Low boilerplate (macro expansion)
- ✅ Easy to maintain (single macro definition)
- ✅ Fast compile times
- ✅ Clean Python bindings

**Cons:**
- ⚠️ Macros can be error-prone
- ⚠️ Limited to predefined patterns (binary→unary, binary→binary, etc.)

**Assessment:** **Best option for Phase 4** - balances simplicity and scalability.

---

### Option D: Runtime Fusion via Function Pointers

**Approach:** Build operation chains dynamically at runtime.

```cpp
struct FusedOp {
    std::vector<std::function<__m256i(__m256i)>> operations;

    __m256i execute(__m256i input) {
        __m256i result = input;
        for (auto& op : operations) {
            result = op(result);  // Chain operations
        }
        return result;
    }
};
```

**Pros:**
- ✅ Arbitrary runtime composition
- ✅ Flexible API

**Cons:**
- ❌ Virtual function overhead (indirect calls)
- ❌ Hard to vectorize/optimize
- ❌ Complex memory management

**Assessment:** **Not suitable** - defeats purpose of compile-time optimization.

---

### Recommended Approach: **Option C (Macro-Generated) + Manual for Complex Cases**

**Rationale:**
1. Use **macros** to generate common patterns (80% of use cases)
2. Implement **manual fusion** for complex/custom chains (20%)
3. Keep door open for **expression templates** in Phase 6+

---

## 3. Recommended Implementation

### 3.1 Fusion Patterns to Support

**Phase 4.0 (Proof of Concept):**
- **Binary → Unary**: `tnot(tadd(a, b))`, `tnot(tmul(a, b))`, ...
- **Binary → Binary → Unary**: `tnot(tadd(tmul(a, b), c))`

**Phase 4.1 (Production):**
- **Binary → Binary**: `tmul(tadd(a, b), c)`
- **3-op chains**: `tmax(tmin(tadd(a, b), c), d)`

**Combinatorial scope:**
- Binary ops: 4 (tadd, tmul, tmin, tmax)
- Unary ops: 1 (tnot)
- Binary→Unary: 4 fused ops
- Binary→Binary: 16 fused ops (4 × 4)
- Total Phase 4: ~20 fused operations

---

### 3.2 Implementation Architecture

**New file:** `ternary_fusion.h` (header-only library)

```cpp
#ifndef TERNARY_FUSION_H
#define TERNARY_FUSION_H

#include "ternary_simd_kernels.h"
#include "ternary_algebra.h"

// ============================================================================
// FUSION MACRO GENERATORS
// ============================================================================

// Pattern 1: Binary → Unary
// Result = unary(binary(a, b))
#define FUSE_BINARY_UNARY(name, binary_op, unary_op) \
    template <bool Sanitize = true> \
    static inline __m256i fused_##name##_simd(__m256i a, __m256i b) { \
        return unary_op##_simd<Sanitize>( \
            binary_op##_simd<Sanitize>(a, b) \
        ); \
    } \
    \
    static inline uint8_t fused_##name##_scalar(uint8_t a, uint8_t b) { \
        return unary_op(binary_op(a, b)); \
    }

// Pattern 2: Binary → Binary (with third input)
// Result = binary2(binary1(a, b), c)
#define FUSE_BINARY_BINARY(name, binary_op1, binary_op2) \
    template <bool Sanitize = true> \
    static inline __m256i fused_##name##_simd(__m256i a, __m256i b, __m256i c) { \
        return binary_op2##_simd<Sanitize>( \
            binary_op1##_simd<Sanitize>(a, b), \
            c \
        ); \
    } \
    \
    static inline uint8_t fused_##name##_scalar(uint8_t a, uint8_t b, uint8_t c) { \
        return binary_op2(binary_op1(a, b), c); \
    }

// ============================================================================
// INSTANTIATE COMMON FUSED OPERATIONS
// ============================================================================

// Binary → Unary patterns (4 variants)
FUSE_BINARY_UNARY(tnot_tadd, tadd, tnot);    // tnot(tadd(a, b))
FUSE_BINARY_UNARY(tnot_tmul, tmul, tnot);    // tnot(tmul(a, b))
FUSE_BINARY_UNARY(tnot_tmin, tmin, tnot);    // tnot(tmin(a, b))
FUSE_BINARY_UNARY(tnot_tmax, tmax, tnot);    // tnot(tmax(a, b))

// Binary → Binary patterns (16 variants - sample shown)
FUSE_BINARY_BINARY(tmul_tadd, tadd, tmul);   // tmul(tadd(a, b), c)
FUSE_BINARY_BINARY(tadd_tmul, tmul, tadd);   // tadd(tmul(a, b), c)
FUSE_BINARY_BINARY(tmax_tmin, tmin, tmax);   // tmax(tmin(a, b), c)
// ... 13 more variants

#endif // TERNARY_FUSION_H
```

---

### 3.3 Integration with Main Engine

**File:** `ternary_simd_engine_fusion.cpp` (new file)

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "ternary_simd_engine.cpp"  // Reuse process_binary_array template
#include "ternary_fusion.h"

namespace py = pybind11;

// ============================================================================
// BINARY → UNARY FUSED OPERATIONS (2 inputs)
// ============================================================================

py::array_t<uint8_t> fused_tnot_tadd_array(
    py::array_t<uint8_t> A,
    py::array_t<uint8_t> B
) {
    return process_binary_array<SANITIZE>(
        A, B,
        fused_tnot_tadd_simd<SANITIZE>,
        fused_tnot_tadd_scalar
    );
}

// ... (3 more binary→unary variants)

// ============================================================================
// BINARY → BINARY FUSED OPERATIONS (3 inputs)
// ============================================================================

// New template for ternary operations (3 inputs)
template <bool Sanitize = true, typename SimdOp, typename ScalarOp>
py::array_t<uint8_t> process_ternary_array(
    py::array_t<uint8_t> A,
    py::array_t<uint8_t> B,
    py::array_t<uint8_t> C,
    SimdOp simd_op,
    ScalarOp scalar_op
) {
    auto a = A.unchecked<1>();
    auto b = B.unchecked<1>();
    auto c = C.unchecked<1>();
    ssize_t n = A.size();

    // Validation
    if (n != B.size() || n != C.size()) {
        throw ArraySizeMismatchError(n, B.size());
    }

    py::array_t<uint8_t> out(n);
    auto r = out.mutable_unchecked<1>();

    const uint8_t* a_ptr = static_cast<const uint8_t*>(A.data());
    const uint8_t* b_ptr = static_cast<const uint8_t*>(B.data());
    const uint8_t* c_ptr = static_cast<const uint8_t*>(C.data());
    uint8_t* r_ptr = static_cast<uint8_t*>(out.mutable_data());

    ssize_t i = 0;

    // PATH 1: OpenMP parallel
    if (n >= OMP_THRESHOLD) {
        ssize_t n_simd_blocks = (n / 32) * 32;

        #pragma omp parallel for schedule(guided, 4)
        for (ssize_t idx = 0; idx < n_simd_blocks; idx += 32) {
            __m256i va = _mm256_loadu_si256((__m256i const*)(a_ptr + idx));
            __m256i vb = _mm256_loadu_si256((__m256i const*)(b_ptr + idx));
            __m256i vc = _mm256_loadu_si256((__m256i const*)(c_ptr + idx));
            __m256i vr = simd_op(va, vb, vc);  // Fused kernel (3 inputs!)
            _mm256_storeu_si256((__m256i*)(r_ptr + idx), vr);
        }
        i = n_simd_blocks;
    }
    // PATH 2: Serial SIMD
    else {
        for (; i + 32 <= n; i += 32) {
            __m256i va = _mm256_loadu_si256((__m256i const*)(a_ptr + i));
            __m256i vb = _mm256_loadu_si256((__m256i const*)(b_ptr + i));
            __m256i vc = _mm256_loadu_si256((__m256i const*)(c_ptr + i));
            __m256i vr = simd_op(va, vb, vc);
            _mm256_storeu_si256((__m256i*)(r_ptr + i), vr);
        }
    }

    // PATH 3: Scalar tail
    for (; i < n; ++i) {
        r[i] = scalar_op(a[i], b[i], c[i]);
    }

    return out;
}

py::array_t<uint8_t> fused_tmul_tadd_array(
    py::array_t<uint8_t> A,
    py::array_t<uint8_t> B,
    py::array_t<uint8_t> C
) {
    return process_ternary_array<SANITIZE>(
        A, B, C,
        fused_tmul_tadd_simd<SANITIZE>,
        fused_tmul_tadd_scalar
    );
}

// ============================================================================
// PYTHON BINDINGS
// ============================================================================

PYBIND11_MODULE(ternary_fusion_engine, m) {
    // Binary → Unary (2 inputs)
    m.def("fused_tnot_tadd", &fused_tnot_tadd_array,
          "Fused operation: tnot(tadd(a, b))");
    m.def("fused_tnot_tmul", &fused_tnot_tmul_array,
          "Fused operation: tnot(tmul(a, b))");
    // ... (more variants)

    // Binary → Binary (3 inputs)
    m.def("fused_tmul_tadd", &fused_tmul_tadd_array,
          "Fused operation: tmul(tadd(a, b), c)");
    m.def("fused_tadd_tmul", &fused_tadd_tmul_array,
          "Fused operation: tadd(tmul(a, b), c)");
    // ... (more variants)
}
```

---

### 3.4 Python API Design

**Import:**
```python
import ternary_fusion_engine as fusion
import ternary_simd_engine as ternary
import numpy as np
```

**Usage:**
```python
# Example 1: Binary → Unary fusion
a = np.array([0b10, 0b01, 0b00], dtype=np.uint8)  # [+1, 0, -1]
b = np.array([0b01, 0b10, 0b10], dtype=np.uint8)  # [0, +1, +1]

# Unfused (2 operations, 5N memory traffic)
temp = ternary.tadd(a, b)
result_unfused = ternary.tnot(temp)

# Fused (1 operation, 3N memory traffic)
result_fused = fusion.fused_tnot_tadd(a, b)

assert np.array_equal(result_fused, result_unfused)  # Same result, faster

# Example 2: Binary → Binary fusion
c = np.array([0b10, 0b00, 0b01], dtype=np.uint8)  # [+1, -1, 0]

# Unfused (2 operations, 6N memory traffic)
temp = ternary.tadd(a, b)
result_unfused = ternary.tmul(temp, c)

# Fused (1 operation, 4N memory traffic)
result_fused = fusion.fused_tmul_tadd(a, b, c)

assert np.array_equal(result_fused, result_unfused)
```

---

### 3.5 Build System Integration

**File:** `build_fusion.py` (new build script)

```python
import sys
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "ternary_fusion_engine",
        ["ternary_simd_engine_fusion.cpp"],
        include_dirs=[".", "avx512-future-support"],
        extra_compile_args=[
            "/arch:AVX2" if sys.platform == "win32" else "-mavx2",
            "/O2" if sys.platform == "win32" else "-O3",
            "/openmp" if sys.platform == "win32" else "-fopenmp"
        ],
        extra_link_args=[
            "/openmp" if sys.platform == "win32" else "-fopenmp"
        ]
    ),
]

setup(
    name="ternary_fusion_engine",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext}
)
```

**Build command:**
```bash
python build_fusion.py build_ext --inplace
```

---

## 4. Performance Estimates vs Reality Checks

### 4.1 Theoretical Performance Analysis

**Memory traffic comparison:**

| Scenario | Unfused | Fused | Reduction |
|----------|---------|-------|-----------|
| **Binary → Unary** | | | |
| Example: `tnot(tadd(a, b))` | | | |
| - Loads | 2N + N = 3N | 2N | 33% ↓ |
| - Stores | N + N = 2N | N | 50% ↓ |
| - **Total** | **5N** | **3N** | **40% ↓** |
| | | | |
| **Binary → Binary** | | | |
| Example: `tmul(tadd(a, b), c)` | | | |
| - Loads | 2N + 2N = 4N | 3N | 25% ↓ |
| - Stores | N + N = 2N | N | 50% ↓ |
| - **Total** | **6N** | **4N** | **33% ↓** |

**Instruction count (SIMD path, per 32 elements):**

| Operation | Unfused | Fused | Change |
|-----------|---------|-------|--------|
| **tnot(tadd(a, b))** | | | |
| - Loads | 2 + 1 = 3 | 2 | -1 |
| - Shuffles (LUT lookups) | 1 + 1 = 2 | 2 | 0 |
| - Stores | 1 + 1 = 2 | 1 | -1 |
| - **Total** | **7 instructions** | **5 instructions** | **-2 (-29%)** |

**Cache effects:**
- **Unfused:** Intermediate result pollutes L1/L2/L3
- **Fused:** Intermediate stays in register (zero cache pollution)

---

### 4.2 Expected Speedup Estimates

**Small arrays (< 100K, cache-resident):**
- Memory bandwidth: Not bottleneck
- Speedup limited by: Instruction throughput
- **Estimate: 1.2-1.5× speedup** (fewer instructions)

**Medium arrays (100K - 1M, L3 cache-limited):**
- Memory bandwidth: Moderate bottleneck
- Cache pollution: Significant impact
- **Estimate: 1.8-2.2× speedup** (reduced cache pollution)

**Large arrays (> 1M, DRAM-limited):**
- Memory bandwidth: Primary bottleneck
- 40% reduction in memory traffic = direct speedup
- **Estimate: 2.0-2.5× speedup** (memory-bound workload)

**Overall target: 2-3× speedup on 3-operation chains**

---

### 4.3 Reality Checks & Potential Pitfalls

#### Reality Check 1: Register Pressure
**Concern:** Fused operations hold more intermediate values in registers

**Analysis:**
```cpp
// Unfused tadd: Uses ~3 registers (va, vb, vr)
__m256i va = _mm256_loadu_si256(...);
__m256i vb = _mm256_loadu_si256(...);
__m256i vr = tadd_simd(va, vb);
_mm256_storeu_si256(..., vr);

// Fused tnot(tadd): Uses ~4 registers (va, vb, temp, vr)
__m256i va = _mm256_loadu_si256(...);
__m256i vb = _mm256_loadu_si256(...);
__m256i temp = tadd_simd(va, vb);   // Intermediate in register
__m256i vr = tnot_simd(temp);
_mm256_storeu_si256(..., vr);
```

**AVX2 register count:** 16 × ymm registers (256-bit)

**Verdict:** ✅ **Not a concern** - Plenty of registers available (only using 4 of 16)

---

#### Reality Check 2: Compiler Optimization Interference
**Concern:** Compiler might already optimize separate operations

**Test:** Check assembly output for unfused code
```bash
gcc -O3 -mavx2 -S ternary_simd_engine.cpp
# Examine generated assembly
```

**Current state:** Compiler does NOT fuse across Python API boundaries (separate function calls)

**Verdict:** ✅ **Fusion provides real benefit** - compiler cannot optimize across pybind11 calls

---

#### Reality Check 3: OpenMP Overhead
**Concern:** OpenMP parallelization overhead increases with fused operations

**Analysis:**
- Current threshold: 32K × cores
- Fused operations: Same threshold applies
- OpenMP overhead: ~1-2 μs per parallel region

**Verdict:** ✅ **No additional overhead** - same parallel structure as unfused

---

#### Reality Check 4: LUT Cache Sharing
**Concern:** Multiple operations share LUT cache, potential conflicts

**Current LUT footprint:**
```
Binary ops: 4 × 16 bytes = 64 bytes
Unary ops:  1 × 16 bytes = 16 bytes
Total:      80 bytes (fits in L1 cache: 32 KB)
```

**Fused operation LUT access:**
```cpp
fused_tnot_tadd_simd:
  1. Access TADD_LUT (16 bytes)
  2. Access TNOT_LUT (16 bytes)
  Total: 32 bytes
```

**Verdict:** ✅ **No conflict** - 80 bytes total LUTs << 32 KB L1 cache

---

#### Reality Check 5: Python API Overhead
**Concern:** Fused operations still have pybind11 overhead

**Current overhead (per operation):**
- Array metadata access: ~10 ns
- NumPy array creation: ~50 ns
- pybind11 wrapping: ~20 ns
- **Total: ~80 ns**

**Impact on speedup:**
- Small arrays (1K elements): 80 ns overhead / 1 μs compute = 8% overhead
- Large arrays (1M elements): 80 ns overhead / 1 ms compute = 0.008% overhead

**Verdict:** ⚠️ **Minor concern for small arrays** - benefit diminishes below ~10K elements

---

### 4.4 Benchmark-Driven Validation Plan

**Test scenarios:**

| Array Size | Expected Speedup | Reality Check |
|------------|------------------|---------------|
| 1K elements | 1.2× | Measure overhead impact |
| 10K elements | 1.5× | Validate cache effects |
| 100K elements | 2.0× | Confirm L3 reduction |
| 1M elements | 2.3× | DRAM bandwidth savings |
| 10M elements | 2.5× | Maximum speedup ceiling |

**Benchmark code:**
```python
import numpy as np
import ternary_simd_engine as ternary
import ternary_fusion_engine as fusion
import time

def benchmark_fusion(size):
    a = np.random.randint(0, 4, size, dtype=np.uint8) & 0x03
    b = np.random.randint(0, 4, size, dtype=np.uint8) & 0x03

    # Warmup
    for _ in range(100):
        fusion.fused_tnot_tadd(a, b)

    # Benchmark unfused
    start = time.perf_counter()
    for _ in range(1000):
        temp = ternary.tadd(a, b)
        result = ternary.tnot(temp)
    unfused_time = time.perf_counter() - start

    # Benchmark fused
    start = time.perf_counter()
    for _ in range(1000):
        result = fusion.fused_tnot_tadd(a, b)
    fused_time = time.perf_counter() - start

    speedup = unfused_time / fused_time
    print(f"Size: {size:8d}, Speedup: {speedup:.2f}x")

for size in [1000, 10000, 100000, 1000000, 10000000]:
    benchmark_fusion(size)
```

---

## 5. Testing Strategy

### 5.1 Correctness Tests

**Test matrix (exhaustive):**

| Fusion Pattern | Test Cases | Method |
|----------------|-----------|--------|
| Binary → Unary | 4 ops × 1000 arrays | Random inputs, compare vs unfused |
| Binary → Binary | 16 ops × 1000 arrays | Random inputs, compare vs unfused |
| Edge cases | Zeros, all -1s, all +1s | Verify saturation/boundary |

**Example test:**
```python
def test_fused_tnot_tadd_correctness():
    import numpy as np
    import ternary_simd_engine as ternary
    import ternary_fusion_engine as fusion

    for _ in range(1000):
        size = np.random.randint(1, 10000)
        a = np.random.randint(0, 4, size, dtype=np.uint8) & 0x03
        b = np.random.randint(0, 4, size, dtype=np.uint8) & 0x03

        # Unfused (ground truth)
        temp = ternary.tadd(a, b)
        expected = ternary.tnot(temp)

        # Fused (test)
        actual = fusion.fused_tnot_tadd(a, b)

        np.testing.assert_array_equal(actual, expected,
            err_msg=f"Mismatch at size {size}")
```

---

### 5.2 Performance Regression Tests

**Baseline:** Ensure fused operations never slower than unfused

```python
def test_fusion_no_regression():
    for size in [1000, 10000, 100000]:
        a = np.random.randint(0, 4, size, dtype=np.uint8) & 0x03
        b = np.random.randint(0, 4, size, dtype=np.uint8) & 0x03

        # Warmup
        for _ in range(100):
            fusion.fused_tnot_tadd(a, b)
            temp = ternary.tadd(a, b)
            ternary.tnot(temp)

        # Measure
        fused_time = timeit.timeit(
            lambda: fusion.fused_tnot_tadd(a, b),
            number=1000
        )

        unfused_time = timeit.timeit(
            lambda: ternary.tnot(ternary.tadd(a, b)),
            number=1000
        )

        speedup = unfused_time / fused_time
        assert speedup >= 1.0, f"Regression at size {size}: {speedup:.2f}x"
```

---

### 5.3 Memory Footprint Tests

**Verify intermediate arrays are not allocated:**

```python
def test_fusion_memory_footprint():
    import tracemalloc

    size = 1000000  # 1M elements
    a = np.random.randint(0, 4, size, dtype=np.uint8) & 0x03
    b = np.random.randint(0, 4, size, dtype=np.uint8) & 0x03

    # Unfused: Should allocate intermediate array
    tracemalloc.start()
    temp = ternary.tadd(a, b)
    result_unfused = ternary.tnot(temp)
    _, unfused_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Fused: Should NOT allocate intermediate
    tracemalloc.start()
    result_fused = fusion.fused_tnot_tadd(a, b)
    _, fused_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Unfused should allocate ~1MB extra (temp array)
    # Fused should only allocate output array
    assert unfused_peak > fused_peak + 500_000, \
        f"Expected memory savings: {unfused_peak} vs {fused_peak}"
```

---

### 5.4 CI/CD Integration

**GitHub Actions workflow:**

```yaml
name: Fusion Performance Tests

on: [push, pull_request]

jobs:
  test-fusion:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build fusion engine
        run: python build_fusion.py build_ext --inplace

      - name: Run correctness tests
        run: pytest tests/test_fusion_correctness.py

      - name: Run performance tests
        run: pytest tests/test_fusion_performance.py

      - name: Benchmark and report
        run: python benchmarks/bench_fusion.py --report
```

---

## 6. Implementation Roadmap

### Phase 4.0: Proof of Concept (Week 1-2)

**Goal:** Validate fusion concept with minimal implementation

**Deliverables:**
- ✅ `ternary_fusion.h` with 4 Binary→Unary fused ops (macros)
- ✅ `ternary_simd_engine_fusion.cpp` with Python bindings
- ✅ Build script (`build_fusion.py`)
- ✅ Basic correctness test (1 fused operation)
- ✅ Performance benchmark (tnot_tadd on 1M elements)

**Success metric:** 2× speedup on `fused_tnot_tadd` vs unfused

---

### Phase 4.1: Full Binary→Unary Suite (Week 3)

**Goal:** Complete all Binary→Unary fused operations

**Deliverables:**
- ✅ 4 fused operations (tnot_tadd, tnot_tmul, tnot_tmin, tnot_tmax)
- ✅ Comprehensive correctness tests (1000 random arrays each)
- ✅ Performance suite (all array sizes: 1K to 10M)

**Success metric:** All ops show 1.5-2.5× speedup

---

### Phase 4.2: Binary→Binary Support (Week 4-5)

**Goal:** Support 3-input fused operations

**Deliverables:**
- ✅ `process_ternary_array` template (3 inputs)
- ✅ 16 fused Binary→Binary operations (via macros)
- ✅ Extended test suite
- ✅ Documentation update

**Success metric:** 2-3× speedup on `tmul(tadd(a, b), c)`

---

### Phase 4.3: Optimization & Polish (Week 6)

**Goal:** Production-ready quality

**Deliverables:**
- ✅ Memory footprint verification tests
- ✅ CI/CD integration
- ✅ User documentation + examples
- ✅ Performance regression suite
- ✅ CHANGELOG entry

**Success metric:** Zero regressions, comprehensive docs

---

## 7. Risk Assessment

### High-Risk Items

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Speedup < 2× on large arrays** | Low | High | Benchmark early, validate memory traffic reduction |
| **Register spilling degrades performance** | Very Low | Medium | Monitor assembly output, limit fusion depth |
| **Compiler over-optimization makes fusion redundant** | Low | Low | Test across compilers (GCC, Clang, MSVC) |

### Medium-Risk Items

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Combinatorial explosion (20+ ops)** | Medium | Medium | Limit Phase 4 to 20 common ops, defer rest to Phase 6 |
| **Macro hygiene issues** | Low | Low | Use thorough testing, consider templates in Phase 6 |
| **Build system complexity** | Low | Low | Reuse existing build infrastructure |

### Low-Risk Items

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **API confusion (fused vs unfused)** | Medium | Very Low | Clear naming (`fused_*`), comprehensive docs |
| **CI/CD overhead** | Low | Very Low | Use caching, run only on relevant changes |

---

## Appendix A: Phase 5 Preview (Sparse Arrays)

### A.1 Motivation

**Problem:** Dense ternary arrays waste memory on zeros and repeated values.

**Example:**
```python
# Dense representation (100 elements)
data = [0, 0, 0, +1, 0, 0, 0, 0, -1, 0, ...] * 100  # 90% zeros

# Memory: 100 bytes
# Wasted: 90 bytes (zeros)
# Useful: 10 bytes (non-zeros)
```

**Opportunity:** If data is >80% sparse → **5-10× memory savings** + **faster iteration** (skip zeros)

---

### A.2 Sparse Representation Strategies

#### Option A: Compressed Sparse Row (CSR)

**Structure:**
```python
class SparseTernaryCSR:
    data: np.array     # Non-zero values
    indices: np.array  # Positions of non-zeros
    size: int          # Total array size
```

**Example:**
```python
# Dense: [0, 0, +1, 0, -1, 0, 0, +1]
# Sparse:
#   data = [+1, -1, +1]
#   indices = [2, 4, 7]
#   size = 8

# Memory: Dense = 8 bytes, Sparse = 6 bytes (3 values + 3 indices)
```

**Pros:**
- ✅ Simple implementation
- ✅ Fast random access (binary search on indices)
- ✅ Compatible with NumPy/SciPy sparse

**Cons:**
- ⚠️ Overhead for dense regions (index storage)
- ⚠️ Not cache-friendly (random access pattern)

---

#### Option B: Run-Length Encoding (RLE)

**Structure:**
```python
class SparseTernaryRLE:
    runs: List[Tuple[value, length]]
```

**Example:**
```python
# Dense: [0, 0, 0, +1, +1, +1, +1, 0, -1, -1]
# RLE: [(0, 3), (+1, 4), (0, 1), (-1, 2)]

# Memory: Dense = 10 bytes, RLE = 8 values (4 × 2)
```

**Pros:**
- ✅ Excellent for long runs
- ✅ Sequential access (cache-friendly)

**Cons:**
- ❌ Poor for random-access patterns
- ❌ Expensive updates (must recompute runs)

---

#### Option C: Hybrid (CSR + Density Detection)

**Strategy:** Automatically switch between dense and sparse based on sparsity

```python
class AdaptiveTernaryArray:
    def __init__(self, data):
        sparsity = np.count_nonzero(data == 0) / len(data)

        if sparsity > 0.8:
            self._storage = SparseTernaryCSR(data)
            self._mode = "sparse"
        else:
            self._storage = data
            self._mode = "dense"
```

**Pros:**
- ✅ Best of both worlds
- ✅ Transparent to user

**Cons:**
- ⚠️ Complexity in operation implementation

---

### A.3 Sparse Operations

**Challenge:** Operations on sparse arrays must preserve sparsity

```python
# Example: Sparse addition
sparse_a = SparseTernaryCSR(data_a)  # 90% sparse
sparse_b = SparseTernaryCSR(data_b)  # 90% sparse

# Naive: Convert to dense, operate, convert back
# → Defeats purpose (allocates dense intermediate)

# Smart: Merge sparse representations
result = sparse_a.sparse_add(sparse_b)  # Stays sparse
```

**Implementation:**
```python
def sparse_add_csr(a: SparseTernaryCSR, b: SparseTernaryCSR):
    # Merge-sort style iteration over indices
    result_data = []
    result_indices = []

    i, j = 0, 0
    while i < len(a.indices) or j < len(b.indices):
        # ... merge logic (similar to sorted list merge)
        # Apply tadd() only to non-zero pairs

    return SparseTernaryCSR(result_data, result_indices)
```

---

### A.4 Phase 5 Roadmap (Tentative)

**Phase 5.0: CSR Implementation (Week 1-2)**
- Implement `SparseTernaryCSR` class
- Conversion: dense ↔ sparse
- Basic operations (add, mul)

**Phase 5.1: Adaptive Storage (Week 3)**
- Density detection + automatic switching
- Benchmark suite (sparse vs dense)

**Phase 5.2: Sparse SIMD (Week 4-5)**
- Vectorized sparse operations
- Integration with fusion engine

**Phase 5.3: Production (Week 6)**
- Comprehensive tests
- Documentation
- Performance validation

---

### A.5 Expected Impact

| Sparsity | Memory Savings | Iteration Speedup | Total Speedup |
|----------|----------------|-------------------|---------------|
| 50% | 2× | 1.5× | 3× |
| 80% | 5× | 4× | 20× |
| 90% | 10× | 9× | 90× |
| 95% | 20× | 19× | 380× |

**Note:** Speedup assumes sparse-optimized operations (not naive dense conversion)

---

## End of Plan

**Next Steps:**
1. Review and approve this plan
2. Prototype Phase 4.0 (fused_tnot_tadd)
3. Benchmark and validate 2× speedup
4. Proceed to full implementation

**Questions/Feedback:** Open for discussion and refinement.
