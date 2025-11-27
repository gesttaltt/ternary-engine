# SIMD Kernel Documentation

**Location:** `src/core/simd/`
**Status:** Production-ready (Windows x64 validated)
**Peak Throughput:** 35,042 Mops/s

---

## Overview

This directory contains the SIMD-accelerated kernels for ternary arithmetic operations. The implementation uses AVX2 (256-bit vectors) to process 32 trits per instruction, achieving ~35× speedup over scalar operations.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SIMD KERNEL ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │   CPU Detection  │───▶│  Backend Select  │───▶│ Active Backend   │  │
│  │  cpu_simd_       │    │  backend_        │    │  (AVX2/Scalar)   │  │
│  │  capability.h    │    │  registry_       │    │                  │  │
│  │                  │    │  dispatch.cpp    │    │                  │  │
│  └──────────────────┘    └──────────────────┘    └────────┬─────────┘  │
│                                                           │             │
│  ┌────────────────────────────────────────────────────────▼──────────┐ │
│  │                      BACKEND IMPLEMENTATIONS                       │ │
│  ├───────────────────┬───────────────────┬───────────────────────────┤ │
│  │ Scalar Reference  │    AVX2 v1        │       AVX2 v2             │ │
│  │ backend_scalar_   │ backend_avx2_     │ backend_avx2_             │ │
│  │   impl.cpp        │   v1_baseline.cpp │   v2_optimized.cpp        │ │
│  │ (Baseline/Verify) │ (Current Stable)  │ (v1.2.0 Optimizations)    │ │
│  └───────────────────┴───────────────────┴───────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                        KERNEL LAYER                                │ │
│  ├─────────────────┬─────────────────────┬────────────────────────────┤ │
│  │  SIMD Kernels   │  Fusion Operations  │  Optimization Headers     │ │
│  │ simd_avx2_      │ fused_binary_       │  • opt_canonical_index.h  │ │
│  │  32trit_ops.h   │   unary_ops.h       │  • opt_dual_shuffle_xor.h │ │
│  │  (32 trits/op)  │  (1.5-11× speedup)  │  • opt_lut_256byte_       │ │
│  │                 │                     │    expanded.h             │ │
│  └─────────────────┴─────────────────────┴────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                   ALGEBRA LAYER (src/core/algebra/)                │ │
│  │  ternary_algebra.h  •  ternary_lut_gen.h  •  Compile-time LUTs    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## File Index

### Core Kernels

| File | Purpose | Status |
|------|---------|--------|
| `simd_avx2_32trit_ops.h` | Main SIMD kernel functions (tadd, tmul, tmin, tmax, tnot) | Production |
| `fused_binary_unary_ops.h` | Fused operations (Phase 4.0/4.1) | Validated |
| `cpu_simd_capability.h` | Runtime CPU feature detection | Production |
| `scalar_golden_baseline.h` | Golden baseline for correctness verification | Production |

### Backend System

| File | Purpose | Status |
|------|---------|--------|
| `backend_plugin_api.h` | Abstract backend interface (C API) | Production |
| `backend_registry_dispatch.cpp` | Registration and dispatch implementation | Production |
| `backend_scalar_impl.cpp` | Scalar reference backend | Production |
| `backend_avx2_v1_baseline.cpp` | AVX2 backend (current stable) | Production |
| `backend_avx2_v2_optimized.cpp` | AVX2 backend (v1.2.0 optimizations) | Experimental |

### Optimization Headers

| File | Purpose | Status |
|------|---------|--------|
| `opt_canonical_index.h` | Canonical indexing (12-18% improvement) | Production |
| `opt_dual_shuffle_xor.h` | Dual-shuffle XOR optimization | Experimental |
| `opt_lut_256byte_expanded.h` | 256-byte expanded LUTs | Experimental |

---

## Quick Reference

### Operations Supported

| Operation | SIMD Function | Scalar Function | Throughput |
|-----------|---------------|-----------------|------------|
| Addition | `tadd_simd<true>(a, b)` | `tadd(a, b)` | ~3,500 ME/s |
| Multiply | `tmul_simd<true>(a, b)` | `tmul(a, b)` | ~3,500 ME/s |
| Minimum | `tmin_simd<true>(a, b)` | `tmin(a, b)` | ~3,500 ME/s |
| Maximum | `tmax_simd<true>(a, b)` | `tmax(a, b)` | ~3,500 ME/s |
| Negation | `tnot_simd<true>(a)` | `tnot(a)` | ~4,000 ME/s |

### Fused Operations (Phase 4.1)

| Operation | Function | Speedup |
|-----------|----------|---------|
| `tnot(tadd(a,b))` | `fused_tnot_tadd_simd<true>(a, b)` | 1.62-1.95× |
| `tnot(tmul(a,b))` | `fused_tnot_tmul_simd<true>(a, b)` | 1.53-1.86× |
| `tnot(tmin(a,b))` | `fused_tnot_tmin_simd<true>(a, b)` | 1.61-11.26× |
| `tnot(tmax(a,b))` | `fused_tnot_tmax_simd<true>(a, b)` | 1.65-9.50× |

---

## Usage Example

```cpp
#include "core/simd/simd_avx2_32trit_ops.h"
#include <immintrin.h>

void process_trits(const uint8_t* a, const uint8_t* b, uint8_t* result, size_t n) {
    size_t i = 0;

    // SIMD path: 32 trits per iteration
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i vr = tadd_simd<true>(va, vb);  // <true> = sanitize inputs
        _mm256_storeu_si256((__m256i*)(result + i), vr);
    }

    // Scalar tail
    for (; i < n; ++i) {
        result[i] = tadd(a[i], b[i]);
    }
}
```

---

## Documentation Index

- [SIMD Kernels](SIMD_KERNELS.md) - Core kernel implementation details
- [Fusion Operations](FUSION.md) - Phase 4.0/4.1 fused operations
- [Backend System](BACKEND_SYSTEM.md) - Multi-backend architecture
- [Optimizations](OPTIMIZATIONS.md) - Performance optimization techniques
- [CPU Detection](CPU_DETECTION.md) - Runtime ISA detection

---

## Related Documentation

- `src/core/algebra/` - Scalar LUT operations (dependency)
- `src/core/core_api.h` - Unified API entry point
- `benchmarks/cpp-native-kernels/` - Native C++ benchmarks

---

**Last Updated:** 2025-11-27
**Validated Platform:** Windows x64, MSVC 2022
