# SIMD Optimization Libraries

**Doc-Type:** Ecosystem Analysis · Version 1.0 · Generated 2025-12-09

This document analyzes SIMD libraries and optimization techniques relevant to Ternary Engine's performance goals.

---

## Table of Contents

1. [Overview](#overview)
2. [Simd Library](#1-simd-library)
3. [Google Highway](#2-google-highway)
4. [SimSIMD](#3-simsimd)
5. [simdjson](#4-simdjson)
6. [SIMD Techniques](#simd-techniques)
7. [Cross-Platform Strategies](#cross-platform-strategies)
8. [Lessons for Ternary Engine](#lessons-for-ternary-engine)

---

## Overview

Ternary Engine already has AVX2 SIMD support (45.3 Gops/s), but these libraries offer patterns for:
- Cross-platform abstraction (AVX2, AVX-512, NEON, SVE)
- Advanced optimization techniques
- Code maintainability

| Library | Focus | ISAs Supported | Language |
|---------|-------|----------------|----------|
| Simd | Image/ML | SSE-AVX512, NEON, SVE | C++ |
| Highway | Portable SIMD | All major | C++ |
| SimSIMD | Dot products | AVX2-512, NEON, SVE | C |
| simdjson | JSON parsing | AVX2, NEON, etc. | C++ |

---

## 1. Simd Library

### Repository Information

- **URL:** https://github.com/ermig1979/Simd
- **Stars:** 2,000+
- **Language:** C++
- **License:** MIT
- **Status:** Actively maintained

### What It Does

The Simd library provides SIMD-optimized algorithms for:
- Image processing
- Neural network operations
- Matrix computations

### Architecture

```cpp
// Simd library structure
namespace Simd {
    // Base namespace with platform detection
    namespace Base { /* Scalar implementations */ }

    #ifdef SIMD_SSE2_ENABLE
    namespace Sse2 { /* SSE2 implementations */ }
    #endif

    #ifdef SIMD_AVX2_ENABLE
    namespace Avx2 { /* AVX2 implementations */ }
    #endif

    #ifdef SIMD_AVX512_ENABLE
    namespace Avx512bw { /* AVX-512 implementations */ }
    #endif

    #ifdef SIMD_NEON_ENABLE
    namespace Neon { /* ARM NEON implementations */ }
    #endif
}
```

### Neural Network Operations

```cpp
// Simd neural network layer operations
namespace Simd {
namespace Avx2 {

// Convolution with SIMD
void SynetConvolution32fDirectNchw(
    const float * src, const ConvParam & p,
    const float * weight, const float * bias,
    float * dst
) {
    // AVX2-optimized convolution
    for (size_t dc = 0; dc < p.dstC; dc += F) {
        __m256 sum = _mm256_setzero_ps();

        for (size_t sc = 0; sc < p.srcC; ++sc) {
            for (size_t ky = 0; ky < p.kernelY; ++ky) {
                for (size_t kx = 0; kx < p.kernelX; ++kx) {
                    __m256 w = _mm256_loadu_ps(weight);
                    __m256 s = _mm256_set1_ps(*src);
                    sum = _mm256_fmadd_ps(w, s, sum);
                    weight += F;
                    src++;
                }
            }
        }

        _mm256_storeu_ps(dst, sum);
        dst += F;
    }
}

} // namespace Avx2
} // namespace Simd
```

### Why It Matters for Ternary Engine

**Learn:**
- Multi-ISA code organization
- Neural network primitive patterns
- Matrix operation optimizations

**Pattern to adopt:**
```cpp
// Ternary Engine could use similar namespace structure
namespace TernaryEngine {
    namespace Scalar { /* Fallback */ }
    namespace Avx2 { /* Current implementation */ }
    namespace Avx512 { /* Future */ }
    namespace Neon { /* Future */ }
}
```

---

## 2. Google Highway

### Repository Information

- **URL:** https://github.com/google/highway
- **Stars:** 4,000+
- **Language:** C++
- **License:** Apache 2.0
- **Status:** Very active (Google-maintained)

### Core Concept

Highway provides portable SIMD that compiles to optimal code for each target:

```cpp
#include "hwy/highway.h"

namespace hn = hwy::HWY_NAMESPACE;

// Write once, runs on all SIMD architectures
void MultiplyAdd(const float* a, const float* b, const float* c,
                 float* result, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    for (size_t i = 0; i < count; i += N) {
        auto va = hn::Load(d, a + i);
        auto vb = hn::Load(d, b + i);
        auto vc = hn::Load(d, c + i);
        auto vr = hn::MulAdd(va, vb, vc);  // a * b + c
        hn::Store(vr, d, result + i);
    }
}
```

### Key Features

1. **Target-agnostic code:**
   ```cpp
   // Same code compiles to:
   // - AVX-512 on Xeon
   // - AVX2 on mainstream x86
   // - NEON on ARM
   // - WASM SIMD in browsers
   // - RVV on RISC-V
   ```

2. **Scalable vectors:**
   ```cpp
   // Lanes() returns vector width at compile/runtime
   // Code automatically uses full vector width
   const hn::ScalableTag<float> d;
   size_t lanes = hn::Lanes(d);  // 4, 8, 16, etc.
   ```

3. **Type-safe operations:**
   ```cpp
   // Type system prevents mixing incompatible types
   auto vi32 = hn::Load(hn::ScalableTag<int32_t>(), ptr_i32);
   auto vf32 = hn::Load(hn::ScalableTag<float>(), ptr_f32);
   // vi32 + vf32 would fail to compile
   ```

### Highway for Ternary

```cpp
#include "hwy/highway.h"

namespace hn = hwy::HWY_NAMESPACE;

// Portable ternary operations using Highway
void TernaryAdd(const uint8_t* a, const uint8_t* b, uint8_t* result,
                size_t count) {
    const hn::ScalableTag<uint8_t> d;
    const size_t N = hn::Lanes(d);

    // Precomputed LUT for ternary addition
    // Values: 0=(-1), 1=(0), 2=(+1)
    alignas(64) static const uint8_t lut_data[16] = {
        // a + b for each combination
        0, 1, 2,  // a=0 (-1): -1+(-1)=-2→0, -1+0=-1→0, -1+1=0→1
        1, 1, 2,  // a=1 (0):   0+(-1)=-1→0,  0+0=0→1,   0+1=1→2
        2, 2, 2,  // a=2 (+1): +1+(-1)=0→1,  +1+0=+1→2, +1+1=+2→2
        0, 0, 0, 0, 0, 0, 0  // Padding
    };

    for (size_t i = 0; i < count; i += N) {
        auto va = hn::Load(d, a + i);
        auto vb = hn::Load(d, b + i);

        // Compute LUT index: a * 3 + b
        auto idx = hn::Add(hn::Mul(va, hn::Set(d, 3)), vb);

        // Table lookup (portable across ISAs)
        auto lut = hn::LoadDup128(d, lut_data);
        auto vr = hn::TableLookupBytes(lut, idx);

        hn::Store(vr, d, result + i);
    }
}
```

### Why It Matters for Ternary Engine

**Major opportunity:**
- Replace ISA-specific code with portable code
- Automatically support AVX-512, NEON, SVE, WASM
- Reduce maintenance burden

**Migration path:**
```cpp
// Current: AVX2-specific
__m256i tadd_avx2(__m256i a, __m256i b) {
    return _mm256_shuffle_epi8(lut, idx);
}

// Future: Highway portable
template <class D>
auto TAdd(D d, hn::Vec<D> a, hn::Vec<D> b) {
    return hn::TableLookupBytes(lut, hn::Add(hn::Mul(a, 3), b));
}
```

---

## 3. SimSIMD

### Repository Information

- **URL:** https://github.com/ashvardanian/SimSIMD
- **Stars:** 1,000+
- **Language:** C, Python
- **License:** Apache 2.0
- **Status:** Active

### What It Does

SimSIMD provides "up to 200× faster dot products and similarity metrics" with explicit SIMD implementations:

```c
// SimSIMD API
#include <simsimd/simsimd.h>

// Dot product with automatic dispatch
float result;
simsimd_dot_f32(a, b, n, &result);

// Cosine similarity
simsimd_cos_f32(a, b, n, &result);

// L2 distance
simsimd_l2sq_f32(a, b, n, &result);
```

### Implementation Pattern

```c
// SimSIMD's AVX2 dot product
void simsimd_dot_f32_avx2(
    const float* a,
    const float* b,
    size_t n,
    float* result
) {
    __m256 sum = _mm256_setzero_ps();

    for (size_t i = 0; i < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum
    __m128 low = _mm256_castps256_ps128(sum);
    __m128 high = _mm256_extractf128_ps(sum, 1);
    __m128 sum128 = _mm_add_ps(low, high);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);

    *result = _mm_cvtss_f32(sum128);
}
```

### Why It Matters for Ternary Engine

**Learn:**
- Efficient horizontal sum patterns
- Multi-ISA dispatch at runtime
- Python bindings for SIMD code

**Apply to ternary:**
```c
// Ternary dot product (future)
void ternary_dot_avx2(
    const uint8_t* a,  // Packed ternary
    const float* b,    // FP32 activations
    size_t n,
    float* result
) {
    __m256 sum = _mm256_setzero_ps();

    for (size_t i = 0; i < n; i += 32) {
        // Unpack 32 ternary values
        __m256i ta = unpack_ternary_32(a + i/4);

        // Convert to float (-1.0, 0.0, +1.0)
        __m256 fa = ternary_to_float(ta);

        // Load fp32 activations
        __m256 fb = _mm256_loadu_ps(b + i);

        // Multiply-add
        sum = _mm256_fmadd_ps(fa, fb, sum);
    }

    *result = hsum_float_8(sum);
}
```

---

## 4. simdjson

### Repository Information

- **URL:** https://github.com/simdjson/simdjson
- **Stars:** 20,000+
- **Language:** C++
- **License:** Apache 2.0
- **Status:** Very active

### Why It's Relevant

simdjson isn't directly related to ML, but demonstrates world-class SIMD optimization techniques:

1. **Branch-free algorithms**
2. **Data-parallel parsing**
3. **Cache-efficient design**
4. **Multi-ISA support**

### Key Techniques

```cpp
// simdjson's branch-free character classification
// Classify 64 bytes at once (AVX-512) or 32 (AVX2)

__m256i classify_structural_characters(__m256i input) {
    // Use shuffle for parallel lookup
    const __m256i structural_chars =
        _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0,  // 0-7
            0, 0, 1, 0, 0, 0, 0, 0,  // 8-15 (tab=1)
            // ... etc
        );

    // Parallel lookup - no branches!
    return _mm256_shuffle_epi8(structural_chars, input);
}
```

### Why It Matters for Ternary Engine

**Techniques to adopt:**

1. **Branch-free operations:**
   ```cpp
   // Instead of: if (a == 0) result = 0; else ...
   // Use: lookup tables or SIMD comparisons
   __m256i mask_zero = _mm256_cmpeq_epi8(a, _mm256_set1_epi8(1));
   result = _mm256_blendv_epi8(non_zero_result, zero_result, mask_zero);
   ```

2. **Parallel classification:**
   ```cpp
   // Classify all ternary values at once
   __m256i is_positive = _mm256_cmpeq_epi8(values, _mm256_set1_epi8(2));
   __m256i is_negative = _mm256_cmpeq_epi8(values, _mm256_set1_epi8(0));
   ```

---

## SIMD Techniques

### Technique 1: LUT via Shuffle

The most important technique for Ternary Engine:

```cpp
// _mm256_shuffle_epi8 as a 16-entry LUT (applied twice for 32 bytes)
__m256i lookup_lut(__m256i indices, __m256i lut) {
    // Each lane (16 bytes) gets its own lookup
    // Index must be 0-15, uses low 4 bits
    return _mm256_shuffle_epi8(lut, indices);
}

// Ternary operation via LUT
__m256i ternary_op(__m256i a, __m256i b, __m256i lut) {
    // Compute index: a * 3 + b (assuming a,b ∈ {0,1,2})
    __m256i three = _mm256_set1_epi8(3);
    __m256i idx = _mm256_add_epi8(
        _mm256_mullo_epi8(a, three),  // a * 3 (or use add+add)
        b
    );

    // Lookup in LUT
    return _mm256_shuffle_epi8(lut, idx);
}
```

### Technique 2: Horizontal Sum

Critical for dot products and reductions:

```cpp
// Sum all 8 floats in __m256
float hsum_float_8(__m256 v) {
    // Step 1: Add high 128 bits to low 128 bits
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(low, high);

    // Step 2: Horizontal add within 128 bits
    sum128 = _mm_hadd_ps(sum128, sum128);  // [a+b, c+d, a+b, c+d]
    sum128 = _mm_hadd_ps(sum128, sum128);  // [a+b+c+d, ...]

    return _mm_cvtss_f32(sum128);
}

// Sum all 32 bytes in __m256i (as int32)
int hsum_epi8_32(__m256i v) {
    // Use SAD (sum of absolute differences) with zero
    __m256i zero = _mm256_setzero_si256();
    __m256i sad = _mm256_sad_epu8(v, zero);  // 4 × 64-bit sums

    // Extract and sum the 4 values
    __m128i low = _mm256_castsi256_si128(sad);
    __m128i high = _mm256_extracti128_si256(sad, 1);
    __m128i sum = _mm_add_epi64(low, high);
    sum = _mm_add_epi64(sum, _mm_srli_si128(sum, 8));

    return _mm_cvtsi128_si32(sum);
}
```

### Technique 3: Packed Bit Operations

For efficient ternary storage and operations:

```cpp
// Pack 16 ternary values into 32 bits (2 bits each)
uint32_t pack_ternary_16(const uint8_t* values) {
    uint32_t packed = 0;
    for (int i = 0; i < 16; i++) {
        packed |= (values[i] & 0x3) << (i * 2);
    }
    return packed;
}

// Unpack with SIMD (AVX2)
__m256i unpack_ternary_64(const uint64_t* packed) {
    // Load 64 bits = 32 ternary values
    __m256i data = _mm256_set1_epi64x(*packed);

    // Shift and mask to extract 2-bit values
    const __m256i mask = _mm256_set1_epi8(0x03);
    const __m256i shift_amounts = _mm256_setr_epi32(
        0, 2, 4, 6, 8, 10, 12, 14
    );

    // This needs more complex unpacking...
    // Use shuffle and shifts to extract each pair of bits
}
```

### Technique 4: Conditional Selection

Branch-free conditional operations:

```cpp
// Select between two values based on mask
// Equivalent to: mask ? a : b
__m256i select(__m256i mask, __m256i a, __m256i b) {
    return _mm256_blendv_epi8(b, a, mask);
}

// Ternary sign multiplication without branches
// result = (t == 0) ? 0 : (t == 2) ? x : -x
__m256 ternary_mul_float(__m256i t, __m256 x) {
    // t: 0=(-1), 1=(0), 2=(+1)

    // Create masks
    __m256i is_zero = _mm256_cmpeq_epi8(t, _mm256_set1_epi8(1));
    __m256i is_pos = _mm256_cmpeq_epi8(t, _mm256_set1_epi8(2));

    // Create results
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 zero = _mm256_setzero_ps();

    // Select: is_zero ? 0 : (is_pos ? x : -x)
    __m256 result = _mm256_blendv_ps(neg_x, x, _mm256_castsi256_ps(is_pos));
    result = _mm256_blendv_ps(result, zero, _mm256_castsi256_ps(is_zero));

    return result;
}
```

---

## Cross-Platform Strategies

### Strategy 1: Preprocessor Dispatch (Current)

```cpp
// Ternary Engine's current approach
#if defined(__AVX2__)
void tadd(const uint8_t* a, const uint8_t* b, uint8_t* r, size_t n) {
    tadd_avx2(a, b, r, n);
}
#else
void tadd(const uint8_t* a, const uint8_t* b, uint8_t* r, size_t n) {
    tadd_scalar(a, b, r, n);
}
#endif
```

**Pros:** Simple, zero runtime overhead
**Cons:** Must compile separately for each target

### Strategy 2: Runtime Dispatch

```cpp
// Function pointer dispatch
typedef void (*tadd_fn)(const uint8_t*, const uint8_t*, uint8_t*, size_t);

tadd_fn get_tadd_impl() {
    if (cpu_has_avx512()) return tadd_avx512;
    if (cpu_has_avx2()) return tadd_avx2;
    if (cpu_has_sse41()) return tadd_sse41;
    return tadd_scalar;
}

// Called once at startup
static tadd_fn tadd_impl = get_tadd_impl();

void tadd(const uint8_t* a, const uint8_t* b, uint8_t* r, size_t n) {
    tadd_impl(a, b, r, n);
}
```

**Pros:** Single binary works everywhere
**Cons:** Function pointer overhead (usually negligible)

### Strategy 3: Highway Abstraction (Recommended Future)

```cpp
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace ternary {
namespace HWY_NAMESPACE {

template <class D>
HWY_INLINE void TAdd(D d, const uint8_t* a, const uint8_t* b,
                      uint8_t* r, size_t n) {
    using V = hn::Vec<D>;
    const size_t N = hn::Lanes(d);

    for (size_t i = 0; i < n; i += N) {
        V va = hn::Load(d, a + i);
        V vb = hn::Load(d, b + i);

        // Portable LUT lookup
        V lut = hn::LoadDup128(d, tadd_lut);
        V idx = hn::Add(hn::Mul(va, hn::Set(d, 3)), vb);
        V vr = hn::TableLookupBytes(lut, idx);

        hn::Store(vr, d, r + i);
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace ternary
HWY_AFTER_NAMESPACE();
```

**Pros:** Write once, optimal on all platforms
**Cons:** Learning curve, dependency

---

## Lessons for Ternary Engine

### Immediate Improvements

1. **Optimize horizontal sum:**
   ```cpp
   // Current: may use suboptimal reduction
   // Improved: use hsum_float_8() pattern
   ```

2. **Better LUT utilization:**
   ```cpp
   // Use dual-shuffle LUT (already identified as disabled)
   // Enable init_dual_shuffle_luts() for 1.5× speedup
   ```

3. **Cache-aware blocking:**
   ```cpp
   // Process data in L1-cache-sized blocks
   const size_t BLOCK_SIZE = 32 * 1024 / sizeof(uint8_t);
   for (size_t i = 0; i < n; i += BLOCK_SIZE) {
       process_block(a + i, b + i, r + i, min(BLOCK_SIZE, n - i));
   }
   ```

### Future Roadmap

1. **Phase 1: AVX-512 support**
   - 2× throughput vs AVX2
   - Better masking support
   - 64-byte LUTs

2. **Phase 2: ARM NEON support**
   - Mobile/edge deployment
   - Apple Silicon optimization

3. **Phase 3: Highway migration**
   - Portable codebase
   - Automatic optimization
   - WebAssembly support

### Code Organization Recommendation

```
src/core/simd/
├── ternary_ops.h          # Public API (dispatch)
├── ternary_luts.h         # LUT definitions
├── impl/
│   ├── scalar.cpp         # Fallback
│   ├── avx2.cpp           # Current production
│   ├── avx512.cpp         # Future
│   ├── neon.cpp           # Future
│   └── highway.cpp        # Future (replaces above)
└── dispatch.cpp           # Runtime CPU detection
```

---

## Action Items

### Immediate (This Week)

1. [ ] Enable dual-shuffle LUTs (disabled optimization)
2. [ ] Optimize horizontal sum in matmul
3. [ ] Add cache blocking to large operations

### Short-term (This Month)

1. [ ] Implement runtime dispatch system
2. [ ] Add AVX-512 kernels (if hardware available)
3. [ ] Benchmark against SimSIMD for dot products

### Medium-term (This Quarter)

1. [ ] Evaluate Highway for abstraction
2. [ ] Implement NEON kernels
3. [ ] Create unified benchmark suite

---

**Document Version:** 1.0
**Generated:** 2025-12-09
**Author:** Claude Code Analysis
