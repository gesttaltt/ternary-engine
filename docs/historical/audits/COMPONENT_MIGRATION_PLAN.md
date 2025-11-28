# Component Migration Plan: Old Module → Backend System

**Date:** 2025-11-24
**Purpose:** Systematic migration of ALL proven optimizations from `ternary_simd_engine` to backend system
**Approach:** COPY working code, don't redesign

---

## Executive Summary

The old module (`bindings_core_ops.cpp`) has a **proven, optimized three-path architecture**:
- PATH 1: Large arrays (≥OMP_THRESHOLD) → OpenMP + Prefetch + Streaming
- PATH 2: Small arrays → Serial SIMD
- PATH 3: Scalar tail → Remainder elements

**We will port this EXACT architecture to backends, component by component.**

---

## Complete Architecture Analysis

### Old Module Structure (PROVEN - DO NOT REDESIGN)

```cpp
template <bool Sanitize, typename SimdOp, typename ScalarOp>
py::array_t<uint8_t> process_binary_array(...) {
    // Setup
    const uint8_t* a_ptr = A.data();
    const uint8_t* b_ptr = B.data();
    uint8_t* r_ptr = out.mutable_data();
    ssize_t i = 0;

    // PATH 1: Large arrays (n >= OMP_THRESHOLD)
    if (n >= OMP_THRESHOLD) {
        ssize_t n_simd_blocks = (n / 32) * 32;
        bool use_streaming = (n >= STREAM_THRESHOLD) && is_aligned_32(r_ptr);

        #pragma omp parallel for schedule(guided, 4)
        for (ssize_t idx = 0; idx < n_simd_blocks; idx += 32) {
            // Component 1: Prefetching
            if (idx + PREFETCH_DIST < n_simd_blocks) {
                _mm_prefetch((const char*)(a_ptr + idx + PREFETCH_DIST), _MM_HINT_T0);
                _mm_prefetch((const char*)(b_ptr + idx + PREFETCH_DIST), _MM_HINT_T0);
            }

            // Component 2: SIMD operation
            __m256i va = _mm256_loadu_si256((__m256i const*)(a_ptr + idx));
            __m256i vb = _mm256_loadu_si256((__m256i const*)(b_ptr + idx));
            __m256i vr = simd_op(va, vb);

            // Component 3: Streaming stores (conditional)
            if (use_streaming) {
                _mm256_stream_si256((__m256i*)(r_ptr + idx), vr);
            } else {
                _mm256_storeu_si256((__m256i*)(r_ptr + idx), vr);
            }
        }

        // Component 4: Memory fence after streaming
        if (use_streaming) {
            _mm_sfence();
        }

        i = n_simd_blocks;
    }
    // PATH 2: Small arrays (< OMP_THRESHOLD)
    else {
        for (; i + 32 <= n; i += 32) {
            __m256i va = _mm256_loadu_si256((__m256i const*)(a_ptr + i));
            __m256i vb = _mm256_loadu_si256((__m256i const*)(b_ptr + i));
            __m256i vr = simd_op(va, vb);
            _mm256_storeu_si256((__m256i*)(r_ptr + i), vr);
        }
    }

    // PATH 3: Scalar tail
    for (; i < n; ++i) {
        r[i] = scalar_op(a[i], b[i]);
    }
}
```

### Current Backend Structure (INCOMPLETE)

```cpp
static void avx2_v2_tadd(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    size_t i = 0;

    // ONLY ONE PATH - missing OpenMP!
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));  // No prefetch!
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i result = tadd_simd<false>(va, vb);
        _mm256_storeu_si256((__m256i*)(dst + i), result);  // No streaming!
    }

    // Scalar tail - OK
    for (; i < n; i++) {
        dst[i] = tadd(a[i], b[i]);
    }
}
```

**Missing:**
- PATH 1 (OpenMP path)
- Prefetching
- Streaming stores
- Memory fence
- Profiling (optional)

---

## Component-by-Component Migration Checklist

### Component 1: Three-Path Architecture ✅ PARTIALLY PRESENT

**Old Module:**
- PATH 1: `if (n >= OMP_THRESHOLD)` → OpenMP parallel
- PATH 2: `else` → Serial SIMD
- PATH 3: Scalar tail

**Backend:**
- ❌ PATH 1: MISSING
- ✅ PATH 2: Present (but it's the ONLY path)
- ✅ PATH 3: Present

**Migration:** Add PATH 1 conditional around existing SIMD loop

---

### Component 2: OpenMP Parallelization ❌ MISSING

**Source Code (bindings_core_ops.cpp:173):**
```cpp
#pragma omp parallel for schedule(guided, 4)
for (ssize_t idx = 0; idx < n_simd_blocks; idx += 32) {
    // SIMD operations
}
```

**Configuration:**
- Threshold: `OMP_THRESHOLD = 32768 × std::thread::hardware_concurrency()`
- From: `src/core/config/optimization_config.h`
- Schedule: `guided, 4` (NUMA-aware for multi-CCD CPUs)
- Block size: 32 elements (one __m256i)

**Migration Target:** All backend operation functions
- `ternary_backend_scalar.cpp` - N/A (scalar reference, no OpenMP needed)
- `ternary_backend_avx2_v1.cpp` - Add OpenMP to all operations
- `ternary_backend_avx2_v2.cpp` - Add OpenMP to all operations

**Migration Steps:**
1. Include `<omp.h>` in backend implementations
2. Include `optimization_config.h` for `OMP_THRESHOLD`
3. Add three-path structure to each operation
4. Wrap SIMD loop with `#pragma omp parallel for schedule(guided, 4)`

---

### Component 3: Prefetching ❌ MISSING

**Source Code (bindings_core_ops.cpp:176-179):**
```cpp
if (idx + PREFETCH_DIST < n_simd_blocks) {
    _mm_prefetch((const char*)(a_ptr + idx + PREFETCH_DIST), _MM_HINT_T0);
    _mm_prefetch((const char*)(b_ptr + idx + PREFETCH_DIST), _MM_HINT_T0);
}
```

**Configuration:**
- Distance: `PREFETCH_DIST = 256` (from `optimization_config.h`)
- Hint: `_MM_HINT_T0` (fetch to all cache levels)
- Applied to: All input arrays before load

**For Unary Operations:**
```cpp
if (idx + PREFETCH_DIST < n_simd_blocks) {
    _mm_prefetch((const char*)(a_ptr + idx + PREFETCH_DIST), _MM_HINT_T0);
}
```

**For Binary Operations:**
```cpp
if (idx + PREFETCH_DIST < n_simd_blocks) {
    _mm_prefetch((const char*)(a_ptr + idx + PREFETCH_DIST), _MM_HINT_T0);
    _mm_prefetch((const char*)(b_ptr + idx + PREFETCH_DIST), _MM_HINT_T0);
}
```

**Migration:** Add prefetching inside OpenMP loop, before SIMD loads

---

### Component 4: Streaming Stores ❌ MISSING

**Source Code (bindings_core_ops.cpp:168, 187-191):**
```cpp
// Setup (before OpenMP loop)
bool use_streaming = (n >= STREAM_THRESHOLD) && is_aligned_32(r_ptr);

// Inside OpenMP loop
if (use_streaming) {
    _mm256_stream_si256((__m256i*)(r_ptr + idx), vr);
} else {
    _mm256_storeu_si256((__m256i*)(r_ptr + idx), vr);
}
```

**Configuration:**
- Threshold: `STREAM_THRESHOLD = 262144` (from `optimization_config.h`)
- Alignment check: `is_aligned_32(r_ptr)` (from `optimization_config.h`)
- Applied to: Output array stores only

**Migration:**
1. Add alignment check helper if not present
2. Determine streaming flag before OpenMP loop
3. Conditional store inside loop

---

### Component 5: Memory Fence ❌ MISSING

**Source Code (bindings_core_ops.cpp:194-197):**
```cpp
if (use_streaming) {
    _mm_sfence();
}
```

**Purpose:** Ensure all streaming stores are globally visible before proceeding

**Migration:** Add `_mm_sfence()` after OpenMP loop if streaming was used

---

### Component 6: Profiling Markers ⚠️ OPTIONAL

**Source Code (bindings_core_ops.cpp:165, 200, etc.):**
```cpp
TERNARY_PROFILE_TASK_BEGIN(g_ternary_domain, g_task_omp);
// ... operations ...
TERNARY_PROFILE_TASK_END(g_ternary_domain);
```

**Purpose:** Performance profiling with Intel VTune / Tracy

**Migration:** Optional - can add later if profiling framework integrated

---

## Migration Strategy

### Phase 1: Add Required Headers and Configs

**Files to modify:**
- `src/core/simd/ternary_backend_avx2_v2.cpp`
- `src/core/simd/ternary_backend_avx2_v1.cpp`

**Add includes:**
```cpp
#include <omp.h>  // OpenMP support
#include <xmmintrin.h>  // For _mm_prefetch
#include "../config/optimization_config.h"  // OMP_THRESHOLD, STREAM_THRESHOLD, PREFETCH_DIST
```

**Add helper if not present:**
```cpp
static inline bool is_aligned_32(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 31) == 0;
}
```

---

### Phase 2: Migrate Binary Operations Template

**For each binary operation in backends (tadd, tmul, tmin, tmax):**

Replace current simple loop:
```cpp
// OLD (current backend - INCOMPLETE)
static void avx2_v2_tadd(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i result = tadd_simd<false>(va, vb);
        _mm256_storeu_si256((__m256i*)(dst + i), result);
    }
    for (; i < n; i++) {
        dst[i] = tadd(a[i], b[i]);
    }
}
```

With three-path architecture:
```cpp
// NEW (migrated from old module - COMPLETE)
static void avx2_v2_tadd(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    size_t i = 0;

    // PATH 1: Large arrays → OpenMP parallel with prefetch and streaming
    if (n >= OMP_THRESHOLD) {
        size_t n_simd_blocks = (n / 32) * 32;
        bool use_streaming = (n >= STREAM_THRESHOLD) && is_aligned_32(dst);

        #pragma omp parallel for schedule(guided, 4)
        for (size_t idx = 0; idx < n_simd_blocks; idx += 32) {
            // Prefetch next cache lines
            if (idx + PREFETCH_DIST < n_simd_blocks) {
                _mm_prefetch((const char*)(a + idx + PREFETCH_DIST), _MM_HINT_T0);
                _mm_prefetch((const char*)(b + idx + PREFETCH_DIST), _MM_HINT_T0);
            }

            // SIMD operation
            __m256i va = _mm256_loadu_si256((const __m256i*)(a + idx));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(b + idx));
            __m256i result = tadd_simd<false>(va, vb);

            // Streaming or regular store
            if (use_streaming) {
                _mm256_stream_si256((__m256i*)(dst + idx), result);
            } else {
                _mm256_storeu_si256((__m256i*)(dst + idx), result);
            }
        }

        // Memory fence after streaming stores
        if (use_streaming) {
            _mm_sfence();
        }

        i = n_simd_blocks;
    }
    // PATH 2: Small arrays → Serial SIMD
    else {
        for (; i + 32 <= n; i += 32) {
            __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
            __m256i result = tadd_simd<false>(va, vb);
            _mm256_storeu_si256((__m256i*)(dst + i), result);
        }
    }

    // PATH 3: Scalar tail
    for (; i < n; i++) {
        dst[i] = tadd(a[i], b[i]);
    }
}
```

**Apply to:**
- `avx2_v2_tadd`
- `avx2_v2_tmul`
- `avx2_v2_tmin`
- `avx2_v2_tmax`
- **AND fusion operations:** `avx2_v2_fused_tnot_tadd`, etc.

---

### Phase 3: Migrate Unary Operations Template

**For each unary operation (tnot):**

Same three-path structure, but only one prefetch:
```cpp
static void avx2_v2_tnot(uint8_t* dst, const uint8_t* src, size_t n) {
    size_t i = 0;

    // PATH 1: Large arrays
    if (n >= OMP_THRESHOLD) {
        size_t n_simd_blocks = (n / 32) * 32;
        bool use_streaming = (n >= STREAM_THRESHOLD) && is_aligned_32(dst);

        #pragma omp parallel for schedule(guided, 4)
        for (size_t idx = 0; idx < n_simd_blocks; idx += 32) {
            if (idx + PREFETCH_DIST < n_simd_blocks) {
                _mm_prefetch((const char*)(src + idx + PREFETCH_DIST), _MM_HINT_T0);
            }

            __m256i va = _mm256_loadu_si256((const __m256i*)(src + idx));
            __m256i result = tnot_simd<false>(va);

            if (use_streaming) {
                _mm256_stream_si256((__m256i*)(dst + idx), result);
            } else {
                _mm256_storeu_si256((__m256i*)(dst + idx), result);
            }
        }

        if (use_streaming) {
            _mm_sfence();
        }

        i = n_simd_blocks;
    }
    // PATH 2: Small arrays
    else {
        for (; i + 32 <= n; i += 32) {
            __m256i va = _mm256_loadu_si256((const __m256i*)(src + i));
            __m256i result = tnot_simd<false>(va);
            _mm256_storeu_si256((__m256i*)(dst + i), result);
        }
    }

    // PATH 3: Scalar tail
    for (; i < n; i++) {
        dst[i] = tnot(src[i]);
    }
}
```

---

### Phase 4: Build System Updates

**Ensure OpenMP is enabled in CMakeLists.txt or build scripts:**

```cmake
# OpenMP support (REQUIRED for performance)
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ternary_backend PUBLIC OpenMP::OpenMP_CXX)
endif()
```

**Or for MSVC direct flags:**
```cmake
target_compile_options(ternary_backend PRIVATE /openmp)
```

---

### Phase 5: Validation Checklist

After migration, verify:

- [ ] All backend operations compile without errors
- [ ] OpenMP pragmas present in all operations
- [ ] Prefetching added to all SIMD loops (PATH 1)
- [ ] Streaming stores conditional on threshold and alignment
- [ ] Memory fence after streaming stores
- [ ] Three-path architecture (OpenMP / Serial / Tail) present
- [ ] `test_fusion_correctness.py` passes (8/8 tests)
- [ ] `test_omp.py` passes (if exists)
- [ ] `bench_backend_fusion.py` shows speedup ≥1.0× (not regression)
- [ ] Backend performance ≥ old module performance

---

## Expected Performance After Migration

**Current (BROKEN):**
- Small arrays (1K-100K): 1.2-1.8× speedup
- Large arrays (1M): 0.67-0.75× **REGRESSION**

**After Migration (TARGET):**
- Small arrays (1K-100K): 1.2-1.8× speedup (no change)
- Large arrays (1M): **≥14× speedup** (matching old module)

**Success Criteria:**
- Backend fusion ≥ Unfused baseline (no regressions)
- Backend ≥ Old module performance (parity)
- All correctness tests passing

---

## Files to Modify

### Primary Migration Targets

1. **src/core/simd/ternary_backend_avx2_v2.cpp** (PRIORITY 1)
   - Add includes: `<omp.h>`, `<xmmintrin.h>`, `optimization_config.h`
   - Migrate all operations: tadd, tmul, tmin, tmax, tnot
   - Migrate fusion operations: fused_tnot_tadd, fused_tnot_tmul, fused_tnot_tmin, fused_tnot_tmax

2. **src/core/simd/ternary_backend_avx2_v1.cpp** (PRIORITY 2)
   - Same migration as v2 (ensure both backends have parity)

3. **Build system** (PRIORITY 1)
   - Ensure OpenMP is enabled in compilation

### Verification Files

4. **tests/python/test_fusion_correctness.py** - Correctness validation
5. **benchmarks/bench_backend_fusion.py** - Performance validation

---

## Migration Timeline

### Today (2025-11-24)

- [x] Complete component analysis
- [x] Create migration plan document
- [ ] Start migration: AVX2_v2 backend
  - [ ] Add headers and helpers
  - [ ] Migrate binary operations (tadd, tmul, tmin, tmax)
  - [ ] Migrate unary operation (tnot)
  - [ ] Migrate fusion operations

### Tomorrow (2025-11-25)

- [ ] Complete migration: AVX2_v1 backend
- [ ] Build system: Ensure OpenMP enabled
- [ ] Validation: Run all tests
- [ ] Benchmarking: Verify performance ≥ old module

### After Validation

- [ ] Commit component migration
- [ ] Update documentation
- [ ] Continue with Phase 3.2 (dual-shuffle XOR)
- [ ] Continue with Phase 3.3 (remaining fusion patterns)

---

## Conclusion

This is a **COPY operation, not a design operation**. The old module has proven optimizations that work. We systematically port each component to backends without changing the logic.

**Key Principle:** If it works in the old module, use it EXACTLY in the backends.

---

**Status:** Migration plan complete, ready to execute
**Next Step:** Begin Phase 1 (add headers) and Phase 2 (migrate binary operations)
