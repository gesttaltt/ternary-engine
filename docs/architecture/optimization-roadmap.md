# Optimization Roadmap - Technical Implementation Plan

## Overview

This document outlines the **4-phase optimization roadmap** for the Ternary Engine library, with detailed technical specifications for each optimization. Total estimated effort: **6-12 months**.

**Expected Cumulative Performance Gains**: 10-50× depending on workload

---

## Phase 0: Quick Wins (Week 1)

**Goal**: Achieve 30-50% speedup with minimal risk and <1 week effort

**Timeline**: 5-7 days
**Effort**: 12-16 hours total
**Risk**: Very low
**Expected Gain**: 30-50% overall, 3-10× on scalar path

### OPT-086: Scalar Lookup Tables

**Current Problem** (`ternary_algebra.h:16-25`):
```c
static inline trit tadd(trit a, trit b) {
    int s = trit_to_int(a) + trit_to_int(b);  // 2 conversions
    if (s>1) s=1; if (s<-1) s=-1;             // 2 branches
    return int_to_trit(s);                     // 1 conversion
}
// Total: 3 conversions + 2 branches = ~10 cycles
```

**Optimized Solution**:
```c
// Precomputed 16-entry tables (4 bits input: a=2bits, b=2bits)
static const uint8_t TADD_LUT[16] = {
    // Index = (a << 2) | b
    // a=0b00 (-1): -1+-1=-1, -1+0=-1, -1+1=0, -1+invalid=undefined
    0b00, 0b00, 0b01, 0b00,
    // a=0b01 (0): 0+-1=-1, 0+0=0, 0+1=+1, 0+invalid=undefined
    0b00, 0b01, 0b10, 0b00,
    // a=0b10 (+1): 1+-1=0, 1+0=+1, 1+1=+1, 1+invalid=undefined
    0b01, 0b10, 0b10, 0b00,
    // a=0b11 (invalid): all undefined
    0b00, 0b00, 0b00, 0b00
};

static inline trit tadd(trit a, trit b) {
    return TADD_LUT[(a << 2) | b];  // Single load: ~2 cycles
}
```

**Implementation Steps**:
1. Generate LUTs for all 4 operations (tadd, tmul, tmin, tmax)
2. Verify correctness with exhaustive 16-case testing
3. Replace function bodies in `ternary_algebra.h`
4. Benchmark scalar performance

**Memory Cost**: 64 bytes (4 ops × 16 bytes), fits in L1 cache
**Performance**: 3-10× faster scalar operations
**Testing**: Exhaustive (16 cases × 4 operations = 64 tests)

**LUT Definitions Required**:
```c
static const uint8_t TADD_LUT[16] = { /* ... */ };
static const uint8_t TMUL_LUT[16] = { /* ... */ };
static const uint8_t TMIN_LUT[16] = { /* ... */ };
static const uint8_t TMAX_LUT[16] = { /* ... */ };
```

---

### OPT-091: Bitwise `tnot` Optimization

**Current Implementation**:
```c
static inline trit tnot(trit a) {
    return (a==0b00)?0b10:(a==0b10)?0b00:0b01;
}
// 2 comparisons + 2 branches = ~5 cycles
```

**Optimized Solution**:
```c
static inline trit tnot(trit a) {
    return a ^ 0b10;  // Single XOR: ~1 cycle
}
// Verification:
//   0b00 ^ 0b10 = 0b10 ✓  (-1 → +1)
//   0b01 ^ 0b10 = 0b11... wait, that's wrong!
```

**Correction** - XOR doesn't work! Need revised approach:
```c
static const uint8_t TNOT_LUT[4] = {
    0b10,  // tnot(0b00) = 0b10
    0b01,  // tnot(0b01) = 0b01
    0b00,  // tnot(0b10) = 0b00
    0b00   // tnot(0b11) = undefined
};
static inline trit tnot(trit a) {
    return TNOT_LUT[a];  // ~2 cycles
}
```

**Alternative Approach** - Arithmetic:
```c
static inline trit tnot(trit a) {
    // Exploit: -1→+1 (0b00→0b10), 0→0 (0b01→0b01), +1→-1 (0b10→0b00)
    // Pattern: swap bits if not zero
    return (a == 0b01) ? 0b01 : (a ^ 0b10);
}
```

**Performance**: 2-3× faster than branching version
**Effort**: 30 minutes

---

### OPT-111-114: Compiler Optimization Flags

**Current Compilation** (manual):
```bash
c++ -O3 -march=native -mavx2 -shared -std=c++17 -fPIC \
$(python3 -m pybind11 --includes) ternary_simd_engine.cpp \
-o ternary_simd_engine$(python3-config --extension-suffix)
```

**Enhanced Compilation**:
```bash
c++ -O3 -march=native -mavx2 \
    -flto \                              # Link-time optimization
    -ffast-math \                        # Aggressive math opts (safe for integer ops)
    -funroll-loops \                     # Loop unrolling
    -finline-functions \                 # Aggressive inlining
    -fomit-frame-pointer \               # Free up register
    -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    ternary_simd_engine.cpp \
    -o ternary_simd_engine$(python3-config --extension-suffix)
```

**For MSVC (Windows)**:
```bash
cl /O2 /GL /arch:AVX2 /std:c++17 /LD /EHsc \
   /I"pybind11_include_path" \
   ternary_simd_engine.cpp \
   /link /LTCG /OUT:ternary_simd_engine.pyd
```

**Expected Gain**: 5-20% depending on baseline
**Effort**: 30 minutes (update build instructions)

---

### OPT-051: Force Inline Critical Functions

**Modification to `ternary_algebra.h`**:
```c
// Add platform-specific inline hints
#ifdef _MSC_VER
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE __attribute__((always_inline)) inline
#endif

// Apply to all operations
static FORCE_INLINE trit tadd(trit a, trit b) { return TADD_LUT[(a << 2) | b]; }
static FORCE_INLINE trit tmul(trit a, trit b) { return TMUL_LUT[(a << 2) | b]; }
static FORCE_INLINE trit tmin(trit a, trit b) { return TMIN_LUT[(a << 2) | b]; }
static FORCE_INLINE trit tmax(trit a, trit b) { return TMAX_LUT[(a << 2) | b]; }
static FORCE_INLINE trit tnot(trit a)        { return TNOT_LUT[a]; }
```

**Rationale**: Eliminates function call overhead in scalar fallback loops
**Expected Gain**: 2-5% on small arrays
**Effort**: 15 minutes

---

### Phase 0 Deliverables

**Code Changes**:
1. `ternary_algebra.h` - LUT-based operations
2. Build instructions - Enhanced compiler flags
3. Test suite - LUT correctness validation

**Documentation**:
1. Performance comparison (before/after)
2. LUT generation methodology
3. Updated compilation guide

**Success Criteria**:
- All tests pass (zero correctness regressions)
- Scalar operations 3-10× faster (microbenchmark)
- Overall 30-50% speedup on mixed workloads
- No increase in binary size >10KB

---

## Phase 1: Core Optimizations (Weeks 2-4)

**Goal**: Achieve 2-3× overall speedup through foundational improvements

**Timeline**: 3 weeks
**Effort**: 60-80 hours
**Risk**: Low-medium
**Expected Gain**: 2-3× overall

### OPT-066: Aligned Memory Loads

**Current Problem**:
```cpp
__m256i va = _mm256_loadu_si256((__m256i const*)(a.data() + i));  // Unaligned load
```

**Optimization Strategy**:
1. Check array alignment at runtime
2. Use aligned loads when possible
3. Fall back to unaligned for misaligned data

**Implementation**:
```cpp
template<typename Func>
py::array_t<uint8_t> optimized_loop(py::array_t<uint8_t> A,
                                     py::array_t<uint8_t> B,
                                     Func simd_op) {
    auto a = A.unchecked<1>();
    auto b = B.unchecked<1>();
    ssize_t n = A.size();

    py::array_t<uint8_t> out(n);
    auto r = out.mutable_unchecked<1>();

    // Check 32-byte alignment
    bool a_aligned = (reinterpret_cast<uintptr_t>(a.data()) % 32) == 0;
    bool b_aligned = (reinterpret_cast<uintptr_t>(b.data()) % 32) == 0;
    bool r_aligned = (reinterpret_cast<uintptr_t>(r.mutable_data()) % 32) == 0;

    ssize_t i = 0;
    if (a_aligned && b_aligned && r_aligned) {
        // Fast path: aligned loads/stores
        for (; i + 32 <= n; i += 32) {
            __m256i va = _mm256_load_si256((__m256i const*)(a.data() + i));
            __m256i vb = _mm256_load_si256((__m256i const*)(b.data() + i));
            __m256i vr = simd_op(va, vb);
            _mm256_store_si256((__m256i*)(r.mutable_data() + i), vr);
        }
    } else {
        // Slow path: unaligned loads/stores
        for (; i + 32 <= n; i += 32) {
            __m256i va = _mm256_loadu_si256((__m256i const*)(a.data() + i));
            __m256i vb = _mm256_loadu_si256((__m256i const*)(b.data() + i));
            __m256i vr = simd_op(va, vb);
            _mm256_storeu_si256((__m256i*)(r.mutable_data() + i), vr);
        }
    }

    // Scalar tail
    for (; i < n; ++i) r[i] = scalar_op(a[i], b[i]);
    return out;
}
```

**Expected Gain**: 5-15% on aligned arrays
**Effort**: 2-3 days

---

### OPT-001: OpenMP Multi-threading

**Strategy**: Parallelize across array chunks for large arrays (>100K elements)

**Implementation**:
```cpp
#include <omp.h>

py::array_t<uint8_t> tadd_array_parallel(py::array_t<uint8_t> A,
                                          py::array_t<uint8_t> B) {
    auto a = A.unchecked<1>();
    auto b = B.unchecked<1>();
    ssize_t n = A.size();

    py::array_t<uint8_t> out(n);
    auto r = out.mutable_unchecked<1>();

    // Single-threaded for small arrays
    if (n < 100000) {
        return tadd_array(A, B);  // Use original implementation
    }

    // Multi-threaded for large arrays
    #pragma omp parallel for schedule(static)
    for (ssize_t chunk_start = 0; chunk_start < n; chunk_start += 1024) {
        ssize_t chunk_end = std::min(chunk_start + 1024, n);
        ssize_t i = chunk_start;

        // SIMD processing
        for (; i + 32 <= chunk_end; i += 32) {
            __m256i va = _mm256_loadu_si256((__m256i const*)(a.data() + i));
            __m256i vb = _mm256_loadu_si256((__m256i const*)(b.data() + i));
            __m256i vr = tadd_simd(va, vb);
            _mm256_storeu_si256((__m256i*)(r.mutable_data() + i), vr);
        }

        // Scalar tail
        for (; i < chunk_end; ++i) {
            r[i] = tadd(a[i], b[i]);
        }
    }

    return out;
}
```

**Compilation**:
```bash
c++ -O3 -march=native -mavx2 -fopenmp ...
```

**Expected Gain**: 2-8× on large arrays (scales with cores)
**Effort**: 1 week

---

### OPT-036-038: Size-Adaptive Kernels

**Strategy**: Use different code paths based on array size

**Implementation**:
```cpp
py::array_t<uint8_t> tadd_adaptive(py::array_t<uint8_t> A,
                                    py::array_t<uint8_t> B) {
    ssize_t n = A.size();

    if (n < 32) {
        return tadd_scalar_only(A, B);        // Pure scalar, no SIMD overhead
    } else if (n < 100000) {
        return tadd_array(A, B);              // Single-threaded SIMD
    } else {
        return tadd_array_parallel(A, B);     // Multi-threaded SIMD
    }
}
```

**Expected Gain**: 10-30% on mixed workloads
**Effort**: 2-3 days

---

## Phase 2: SIMD Enhancements (Weeks 5-8)

**Goal**: Optimize SIMD code path for 2-4× SIMD-specific speedup

**Timeline**: 4 weeks
**Effort**: 80-120 hours
**Risk**: Medium

### OPT-061: SIMD Shuffle-Based LUTs

**Concept**: Use `_mm256_shuffle_epi8` for vectorized table lookups

**Current SIMD Path**:
```cpp
// tadd: 7 vector operations (2 conversions + add + clamp + back-conversion)
__m256i tadd_simd(__m256i a, __m256i b) {
    __m256i ai = trit_to_int8(a);        // ~3 cycles
    __m256i bi = trit_to_int8(b);        // ~3 cycles
    __m256i s = _mm256_adds_epi8(ai, bi); // ~1 cycle
    s = clamp(s);                         // ~2 cycles
    return int8_to_trit(s);               // ~4 cycles
}
// Total: ~13 cycles
```

**Optimized SIMD LUT** (in development - Phase 2):
```cpp
// Idea: Use shuffle for 4-bit lookups (2 trits → 1 result trit)
// Challenge: AVX2 shuffle only handles 4-bit indices within 128-bit lanes
// Solution: Split into low/high nibbles, use two shuffles + blend
```

**Expected Gain**: 2× SIMD speedup (if feasible)
**Risk**: High - may not be practical for all operations
**Effort**: 2-3 weeks research + implementation

---

### OPT-081-084: Masked Tail Handling

**Goal**: Eliminate scalar fallback using SIMD masks

**Implementation**:
```cpp
// Process last <32 elements using masked SIMD
if (i < n) {
    // Create mask for remaining elements
    uint32_t remaining = n - i;
    uint32_t mask_bits = (1u << remaining) - 1;
    __m256i mask = _mm256_cvtepu8_epi8(_mm_loadu_si128(/* ... */));

    __m256i va = _mm256_maskload_epi8((int8_t*)(a.data() + i), mask);
    __m256i vb = _mm256_maskload_epi8((int8_t*)(b.data() + i), mask);
    __m256i vr = tadd_simd(va, vb);
    _mm256_maskstore_epi8((int8_t*)(r.mutable_data() + i), mask, vr);
}
```

**Expected Gain**: Eliminate up to 31 scalar operations per array
**Effort**: 1 week

---

## Phase 3: Advanced Features (Weeks 9-16)

**Goal**: Platform portability + operation fusion

**Timeline**: 8 weeks
**Effort**: 160-240 hours
**Risk**: Medium-high

### OPT-021-023: Operation Fusion

**Example**: Fused multiply-add `tadd(tmul(a, b), c)`

**Current** (3 separate operations):
```python
temp = tc.tmul(A, B)  # Allocate temp array, 32 trits/op
result = tc.tadd(temp, C)  # Another pass, cache miss likely
```

**Fused** (single operation):
```cpp
py::array_t<uint8_t> tfma_array(py::array_t<uint8_t> A,
                                 py::array_t<uint8_t> B,
                                 py::array_t<uint8_t> C) {
    // Process in single pass
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256(...);
        __m256i vb = _mm256_loadu_si256(...);
        __m256i vc = _mm256_loadu_si256(...);

        // Fused: no intermediate array
        __m256i temp = tmul_simd(va, vb);
        __m256i result = tadd_simd(temp, vc);

        _mm256_storeu_si256(..., result);
    }
}
```

**Expected Gain**: 20-50% on chained operations
**Effort**: 3-4 weeks for comprehensive fusion library

---

### OPT-076-078: Multi-Platform SIMD

**AVX-512 Implementation** (64 trits/op):
```cpp
#ifdef __AVX512F__
static inline __m512i tadd_simd_avx512(__m512i a, __m512i b) {
    // Similar logic, 512-bit vectors
}
#endif
```

**ARM NEON Implementation** (16 trits/op):
```cpp
#ifdef __ARM_NEON
static inline uint8x16_t tadd_simd_neon(uint8x16_t a, uint8x16_t b) {
    // ARM intrinsics
}
#endif
```

**Expected Gain**: Platform portability, 2× on AVX-512
**Effort**: 4-6 weeks

---

## Phase 3 (Revised): Production Refinements (Current Implementation)

**Goal**: Near-production quality with maintainability and extensibility focus

**Status**: IN PROGRESS (based on local-reports/optimization.md)
**Timeline**: 2-4 weeks
**Effort**: 60-100 hours
**Risk**: Low
**Expected Gain**: 2-15% performance, major maintainability/extensibility improvements

### Phase 3.1: Adaptive Threading (Suggestion #1)

**Current Problem**: Static `OMP_THRESHOLD = 100000` doesn't scale across CPU tiers

**Optimized Solution**:
```cpp
const ssize_t OMP_THRESHOLD = 32768 * std::thread::hardware_concurrency();
```

**Rationale**: Keeps load balancing efficient across CPU configurations
**Expected Gain**: 5-10% on systems with many cores
**Effort**: 30 minutes

---

### Phase 3.2: SIMD Width Abstraction Layer (Suggestion #2)

**Purpose**: Prepare for AVX-512BW / ARM SVE without rewriting kernels

**Implementation**:
```cpp
#ifdef __AVX512BW__
  #define VEC __m512i
  #define LOAD _mm512_loadu_si512
  #define STORE _mm512_storeu_si512
  #define SHUFFLE _mm512_shuffle_epi8
  #define VEC_SIZE 64
#else
  #define VEC __m256i
  #define LOAD _mm256_loadu_si256
  #define STORE _mm256_storeu_si256
  #define SHUFFLE _mm256_shuffle_epi8
  #define VEC_SIZE 32
#endif
```

**Rationale**: Compiler auto-selects ISA while keeping one codepath
**Expected Gain**: 2× throughput on AVX-512 systems
**Effort**: 1 week

---

### Phase 3.3: Prefetch Distance Tuning (Suggestion #3)

**Current Problem**: Static `_mm_prefetch(... + 256)` stride

**Optimized Solution**:
```cpp
constexpr int PREFETCH_DIST = 512;  // Tunable per CPU family

// In hot loop:
if (idx + PREFETCH_DIST < n_simd_blocks) {
    _mm_prefetch((const char*)(a_ptr + idx + PREFETCH_DIST), _MM_HINT_T0);
    _mm_prefetch((const char*)(b_ptr + idx + PREFETCH_DIST), _MM_HINT_T0);
}
```

**Rationale**: Empirically tune for Zen 2, Zen 4, Raptor Lake
**Expected Gain**: 2-5% throughput
**Effort**: 2 days (tuning + benchmarking)

---

### Phase 3.4: Optional Compile-Time Sanitization Switch (Suggestion #4)

**Purpose**: Enable CI sanitized vs. raw pipelines

**Implementation**:
```cpp
#ifdef TERNARY_NO_SANITIZE
constexpr bool SANITIZE = false;
#else
constexpr bool SANITIZE = true;
#endif

// Usage:
process_binary_array<SANITIZE>(A, B, simd_op, scalar_op);
```

**Rationale**: Easy `-DTERNARY_NO_SANITIZE` flags for performance testing
**Expected Gain**: 3-5% in validated data pipelines
**Effort**: 1 day

---

### Phase 3.5: Runtime Feature Detection (Suggestion #5)

**Purpose**: Expose CPU capability detection via cpuid

**Implementation** (new file: `ternary_cpu_detect.h`):
```cpp
#include <cpuid.h>

inline bool has_avx512bw() {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_max(0, nullptr) < 7) return false;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & bit_AVX512BW) != 0;
}

inline bool has_avx2() {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_max(0, nullptr) < 7) return false;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & bit_AVX2) != 0;
}

inline bool has_sve() {
    #ifdef __ARM_FEATURE_SVE
    return true;
    #else
    return false;
    #endif
}
```

**Rationale**: Enables dynamic fallback in future hybrid builds
**Expected Gain**: Portability, graceful degradation
**Effort**: 2 days

---

### Phase 3.6: Cross-Language Interop Layer (Suggestion #6)

**Purpose**: Direct integration in Rust, Zig, C# without Python

**Implementation** (new file: `ternary_c_api.h`):
```cpp
#ifndef TERNARY_C_API_H
#define TERNARY_C_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Pure C API for cross-language FFI
void tadd_simd_u8(const uint8_t* A, const uint8_t* B, uint8_t* R, size_t n);
void tmul_simd_u8(const uint8_t* A, const uint8_t* B, uint8_t* R, size_t n);
void tmin_simd_u8(const uint8_t* A, const uint8_t* B, uint8_t* R, size_t n);
void tmax_simd_u8(const uint8_t* A, const uint8_t* B, uint8_t* R, size_t n);
void tnot_simd_u8(const uint8_t* A, uint8_t* R, size_t n);

#ifdef __cplusplus
}
#endif

#endif // TERNARY_C_API_H
```

**Rationale**: Enables FFI without pybind11 dependency
**Expected Gain**: Ecosystem expansion
**Effort**: 2-3 days

---

### Phase 3.7: Advanced Microbenchmarking (Suggestion #7)

**Purpose**: Kernel-level benchmarks bypassing Python/NumPy overhead

**Implementation** (new file: `benchmarks/bench_kernels.cpp`):
```cpp
#include <chrono>
#include <iostream>
#include <random>
#include "ternary_simd_kernels.h"

void bench_tadd_simd(size_t N) {
    std::vector<uint8_t> A(N), B(N), R(N);

    // Initialize with random trits
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 2);
    for (size_t i = 0; i < N; ++i) {
        A[i] = dis(gen) == 0 ? 0b00 : (dis(gen) == 1 ? 0b01 : 0b10);
        B[i] = dis(gen) == 0 ? 0b00 : (dis(gen) == 1 ? 0b01 : 0b10);
    }

    // Benchmark
    auto start = std::chrono::steady_clock::now();
    for (size_t iter = 0; iter < 1000; ++iter) {
        for (size_t i = 0; i + 32 <= N; i += 32) {
            __m256i va = _mm256_loadu_si256((__m256i const*)(A.data() + i));
            __m256i vb = _mm256_loadu_si256((__m256i const*)(B.data() + i));
            __m256i vr = tadd_simd<true>(va, vb);
            _mm256_storeu_si256((__m256i*)(R.data() + i), vr);
        }
    }
    auto end = std::chrono::steady_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double throughput = (N * 1000.0) / (elapsed / 1e9);
    std::cout << "N=" << N << ", Throughput: " << throughput / 1e6 << " ME/s\n";
}
```

**Rationale**: Direct comparison with xsimd or Eigen
**Expected Gain**: Better profiling visibility
**Effort**: 3-4 days

---

### Phase 3.8: Constexpr LUT Generation (C++20) (Suggestion #9)

**Purpose**: Move LUT initialization to compile-time

**Current State**: Already implemented in `ternary_lut_gen.h` (C++17 constexpr)

**C++20 Enhancement**:
```cpp
// Current (runtime initialization):
namespace {
    struct BroadcastedLUTs {
        __m256i tadd;
        BroadcastedLUTs() : tadd(broadcast_lut_16(TADD_LUT.data())) {}
    };
    static const BroadcastedLUTs g_luts;
}

// C++20 potential (constexpr if __m256i becomes literal type):
namespace {
    constexpr auto g_luts = []() {
        BroadcastedLUTs luts;
        // ... compile-time initialization
        return luts;
    }();
}
```

**Status**: Partially complete (LUTs are constexpr, broadcast is runtime)
**Rationale**: Eliminates runtime broadcast overhead on startup
**Expected Gain**: Negligible (initialization is one-time)
**Effort**: 2 days (C++20 migration + testing)

---

### Phase 3.9: Profiler Annotations (Suggestion #10)

**Purpose**: Profiler integration infrastructure (roadmap)

**Implementation**:
```cpp
#ifdef TERNARY_ENABLE_PROFILING
#include <ittnotify.h>
__itt_domain* domain = __itt_domain_create("TernaryCore");
__itt_string_handle* handle_simd = __itt_string_handle_create("SIMD_Loop");
__itt_string_handle* handle_tail = __itt_string_handle_create("Scalar_Tail");
#else
#define __itt_task_begin(...)
#define __itt_task_end(...)
#endif

// In code:
#pragma omp parallel for schedule(guided)
for (ssize_t idx = 0; idx < n_simd_blocks; idx += 32) {
    __itt_task_begin(domain, __itt_null, __itt_null, handle_simd);
    // ... SIMD work
    __itt_task_end(domain);
}
```

**Rationale**: When integrated: visualize OpenMP block timing
**Expected Gain**: Better profiling insights
**Effort**: 2 days

---

## Phase 4: Specialization (Weeks 17+)

**Goal**: Domain-specific kernels for target applications

### OPT-011: Fractal Iteration Kernel

**Mandelbrot-style iteration** (repeated `tadd(tmul(z,z), c)`):
```cpp
py::array_t<uint8_t> mandelbrot_iterate(py::array_t<uint8_t> Z,
                                         py::array_t<uint8_t> C,
                                         int iterations) {
    // Fused iteration without materializing intermediates
    for (int iter = 0; iter < iterations; ++iter) {
        for (ssize_t i = 0; i + 32 <= n; i += 32) {
            __m256i z = _mm256_loadu_si256(...);
            __m256i c = _mm256_loadu_si256(...);

            // z = z*z + c (fused)
            __m256i z_squared = tmul_simd(z, z);
            z = tadd_simd(z_squared, c);

            _mm256_storeu_si256(..., z);
        }
    }
}
```

**Expected Gain**: 10-100× on targeted workloads
**Effort**: 2-4 weeks per kernel

---

## Implementation Timeline (Revised)

```
Week 1:     Phase 0 (Quick Wins) - COMPLETE
Weeks 2-4:  Phase 1 (Core Optimizations) - COMPLETE
Weeks 5-8:  Phase 2 (SIMD Enhancements) - COMPLETE
Weeks 9-10: Phase 3 (Production Refinements) - IN PROGRESS
Weeks 11+:  Phase 4 (Specialization) - PLANNED
```

**Milestones**:
- Week 1: 30-50% speedup (Phase 0 complete) ✓
- Week 4: 2-3× speedup (Phase 1 complete) ✓
- Week 8: 4-8× speedup (Phase 2 complete) ✓
- Week 10: Best-in-class architecture (Phase 3 in progress)
- Ongoing: Up to 50× on domain-specific workloads

---

## Success Metrics

### Performance Benchmarks
1. **Microbenchmarks**: Single-operation timing
2. **Array size sweep**: 32, 1K, 10K, 100K, 1M, 10M elements
3. **Operation chains**: Realistic multi-op sequences
4. **Domain kernels**: Fractal generation, modulo-3 arithmetic

### Correctness Validation
1. **Exhaustive testing**: All 16 input combinations per operation
2. **Random testing**: 1M random operations vs reference
3. **Regression suite**: Continuous validation

### Code Quality
1. **No correctness regressions**: All tests must pass
2. **Binary size**: No >50% increase
3. **Maintainability**: Code complexity metrics
4. **Documentation**: Updated for all changes

---

**Document Version**: 2.0
**Last Updated**: 2025-10-13
**Status**: Phase 3 in progress, implementing optimization.md suggestions
