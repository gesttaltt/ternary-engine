# ARM NEON Support Roadmap

**Doc-Type:** Implementation Roadmap · Version 1.0 · Created 2026-03-19

Cross-platform SIMD abstraction for ARM NEON support, enabling ternary engine on mobile, embedded, and Apple Silicon platforms.

---

## Executive Summary

Ternary engine currently supports only x86-64 with AVX2. Adding ARM NEON support would enable:
- **Mobile devices** (ARM64 Android/iOS)
- **Apple Silicon** (M1/M2/M3 Macs)
- **Embedded systems** (ARM Cortex-A series)
- **Edge computing** (ARM-based IoT gateways)

**Reference**: RandomX's `intrin_portable.h` provides a battle-tested template for cross-platform SIMD abstraction.

---

## Technical Analysis

### Current State (x86-64 AVX2)

```cpp
// Core operation: 32 trits per AVX2 vector
using trit_vec_t = __m256i;

// LUT lookup via byte shuffle
__m256i tadd_simd(__m256i a, __m256i b) {
    __m256i indices = canonical_index_avx2(a, b);
    return _mm256_shuffle_epi8(lut, indices);
}
```

### ARM NEON Equivalents

| x86-64 AVX2 | ARM NEON | Notes |
|--------------|----------|-------|
| `__m256i` | `uint8x16_t` | 128-bit = 16 trits |
| `_mm256_loadu_si256` | `vld1q_u8` | Unaligned load |
| `_mm256_storeu_si256` | `vst1q_u8` | Unaligned store |
| `_mm256_shuffle_epi8` | `vqtbl1q_u8` | Lane-dependent table lookup |
| `_mm256_broadcastsi128_si256` | `vdupq_n_u8` + `vcombine` | Broadcast 16→32 bytes |
| `_mm_prefetch` | `prfm pldl1stm` | Software prefetch |

### Key Differences

1. **Vector width**: 128-bit (NEON) vs 256-bit (AVX2)
   - 16 trits/vector (NEON) vs 32 trits/vector (AVX2)
   - Requires more parallel work to saturate execution units

2. **LUT indexing**: NEON's `vqtbl1q_u8` has restrictions
   - Index bytes must be < 16 for full lookup
   - Canonical indexing needs adaptation

3. **Prefetch**: Different hint values
   - x86: `_MM_HINT_T0`, `_MM_HINT_NTA`
   - ARM: `PLDL1STREAM`, `PLDL1KEEP`, `PLIL1STREAM`

---

## Implementation Architecture

### Proposed File Structure

```
src/
├── common/
│   ├── ternary_intrinsics.h       # Cross-platform SIMD abstraction
│   ├── ternary_cpu_detect_arm.h    # ARM CPU feature detection
│   └── ternary_config.h            # Unified configuration
├── simd/
│   ├── simd_common_ops.h           # Portable SIMD kernels
│   ├── backend_avx2.cpp            # x86-64 AVX2 backend
│   └── backend_neon.cpp            # ARM NEON backend
└── engine/
    └── bindings_core_ops.cpp       # Updated with runtime dispatch
```

### Core Intrinsics Abstraction

```cpp
// src/common/ternary_intrinsics.h

#pragma once

#include <cstdint>

// ============================================================================
// Platform Detection
// ============================================================================

#if defined(__AVX2__)
    #define TERNARY_ARCH_AVX2 1
    #define TERNARY_TRITS_PER_VECTOR 32
#elif defined(__aarch64__) || defined(__ARM_NEON)
    #define TERNARY_ARCH_NEON 1
    #define TERNARY_TRITS_PER_VECTOR 16
#else
    #define TERNARY_ARCH_SCALAR 1
    #define TERNARY_TRITS_PER_VECTOR 1
#endif

// ============================================================================
// Type Definitions
// ============================================================================

#if defined(TERNARY_ARCH_AVX2)
    #include <immintrin.h>
    using trit_vec_t = __m256i;
    using trit_mask_t = __m256i;

#elif defined(TERNARY_ARCH_NEON)
    #include <arm_neon.h>
    using trit_vec_t = uint8x16_t;
    using trit_mask_t = uint8x16_t;

#else
    // Scalar fallback
    struct trit_vec_t { uint8_t v; };
    struct trit_mask_t { uint8_t v; };
#endif

// ============================================================================
// Memory Operations
// ============================================================================

FORCE_INLINE trit_vec_t trit_load(const uint8_t* ptr) {
    #if defined(TERNARY_ARCH_AVX2)
        return _mm256_loadu_si256((__m256i*)ptr);
    #elif defined(TERNARY_ARCH_NEON)
        return vld1q_u8(ptr);
    #else
        return *(trit_vec_t*)ptr;
    #endif
}

FORCE_INLINE void trit_store(uint8_t* ptr, trit_vec_t v) {
    #if defined(TERNARY_ARCH_AVX2)
        _mm256_storeu_si256((__m256i*)ptr, v);
    #elif defined(TERNARY_ARCH_NEON)
        vst1q_u8(ptr, v);
    #else
        *(trit_vec_t*)ptr = v;
    #endif
}

// ============================================================================
// LUT Lookup (Core Operation)
// ============================================================================

FORCE_INLINE trit_vec_t trit_lut_lookup(trit_vec_t lut, trit_vec_t indices) {
    #if defined(TERNARY_ARCH_AVX2)
        return _mm256_shuffle_epi8(lut, indices);
    #elif defined(TERNARY_ARCH_NEON)
        return vqtbl1q_u8(lut, indices);
    #else
        // Scalar fallback
        return scalar_lut_lookup(lut, indices);
    #endif
}

// ============================================================================
// Prefetch
// ============================================================================

FORCE_INLINE void trit_prefetch(const void* ptr, int hint) {
    #if defined(TERNARY_ARCH_AVX2)
        _mm_prefetch((const char*)ptr, hint);
    #elif defined(TERNARY_ARCH_NEON)
        // hint: 0=streaming, 1=T0, 2=T1, 3=T2
        switch(hint) {
            case 0: __asm__ volatile("prfm pldl1strm, [%0]" :: "r"(ptr)); break;
            case 1: __asm__ volatile("prfm pldl1keep, [%0]" :: "r"(ptr)); break;
            default: break;
        }
    #endif
}

// ============================================================================
// Broadcast (for LUT loading)
// ============================================================================

FORCE_INLINE trit_vec_t trit_broadcast_lut(const uint8_t* lut16) {
    #if defined(TERNARY_ARCH_AVX2)
        __m128i lut_128 = _mm_loadu_si128((__m128i*)lut16);
        return _mm256_broadcastsi128_si256(lut_128);
    #elif defined(TERNARY_ARCH_NEON)
        // NEON: load 16 bytes, duplicate to both lanes
        uint8x16_t lut = vld1q_u8(lut16);
        return vcombine_u8(lut, lut);  // Replicate for 32-byte LUT
    #else
        return scalar_broadcast_lut(lut16);
    #endif
}
```

---

## Implementation Phases

### Phase 1: Abstraction Layer (Week 1-2)

**Goal**: Create portable intrinsics header without modifying existing code

**Tasks**:
- [ ] Create `src/common/ternary_intrinsics.h` with platform detection
- [ ] Implement `trit_load`, `trit_store`, `trit_lut_lookup` for AVX2
- [ ] Implement stubs for NEON (compile-time disabled)
- [ ] Add compile-time assertions for vector sizes
- [ ] Write unit tests for abstraction layer

**Deliverable**: `ternary_intrinsics.h` - drop-in compatible with existing code

### Phase 2: ARM NEON Backend (Week 3-4)

**Goal**: Implement working NEON kernels

**Tasks**:
- [ ] Implement canonical indexing for NEON (adapt from AVX2)
- [ ] Create `backend_neon.cpp` with all operations
- [ ] Handle NEON `vqtbl1q_u8` index restrictions
- [ ] Implement ARM-specific prefetch hints
- [ ] Add runtime CPU detection for ARM

**Key Challenge**: Canonical indexing adaptation
```cpp
// AVX2: indices = shuffle_a + shuffle_b (256-bit)
// NEON: indices = vqtbl1q_u8(CANON_A, a) + vqtbl1q_u8(CANON_B, b)
// Note: NEON addition is per-lane, same semantics
```

### Phase 3: Runtime Dispatch (Week 5-6)

**Goal**: Seamlessly select optimal backend at runtime

**Tasks**:
- [ ] Extend `cpu_simd_capability.h` for ARM detection
- [ ] Create factory function: `get_ternary_backend()`
- [ ] Update bindings to use runtime dispatch
- [ ] Add benchmark comparison AVX2 vs NEON

**Architecture**:
```cpp
// src/common/ternary_backend.h
class ITernaryBackend {
public:
    virtual ~ITernaryBackend() = default;
    virtual void tadd(uint8_t* dst, const uint8_t* a, 
                      const uint8_t* b, size_t n) = 0;
    // ... other operations
};

ITernaryBackend* get_optimal_backend();  // Runtime selection
```

### Phase 4: Validation & Optimization (Week 7-8)

**Goal**: Production-ready ARM support

**Tasks**:
- [ ] Run full test suite on ARM hardware
- [ ] Validate mathematical correctness exhaustively
- [ ] Profile and optimize NEON hot paths
- [ ] Tune prefetch distance for ARM cache hierarchy
- [ ] Document ARM-specific performance notes

---

## Performance Expectations

| Platform | Vector Width | Expected Throughput | vs AVX2 Baseline |
|----------|--------------|---------------------|-------------------|
| x86-64 AVX2 | 32 trits | 45 Gops/s | 1.0x |
| ARM NEON (M1) | 16 trits | ~20-25 Gops/s | ~0.5x |
| ARM NEON (Cortex-A76) | 16 trits | ~15-20 Gops/s | ~0.4x |
| ARM NEON (Cortex-A55) | 16 trits | ~8-12 Gops/s | ~0.2x |

**Note**: Lower throughput per core is expected due to:
- Smaller vector width (16 vs 32 trits)
- Different execution unit design
- Compiler optimization differences

**Compensation**: ARM devices typically have more cores, enabling better parallel scaling.

---

## Verification Checklist

### Mathematical Correctness
- [ ] Exhaustive 3×3 truth table for all operations
- [ ] Algebraic invariants (commutativity, identity, involution)
- [ ] SIMD vs scalar comparison (must match exactly)
- [ ] Fuzz testing with random inputs

### Platform-Specific
- [ ] Apple Silicon M1/M2/M3 validation
- [ ] Android ARM64 (Qualcomm Snapdragon)
- [ ] ARM Cortex-A series (Raspberry Pi)
- [ ] ARM NEON on x86 (translations not needed)

### Performance
- [ ] Throughput > 10 Gops/s on modern ARM
- [ ] Scaling with array size matches AVX2 patterns
- [ ] Prefetch effectiveness validated

---

## Dependencies & Risks

### Dependencies
- ARM toolchain (gcc-aarch64-linux-gnu, arm-none-eabi)
- Physical ARM hardware for validation (CI: QEMU or cross-compile + remote)
- Apple Silicon Mac (for macOS validation)

### Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Compiler differences | Medium | Extensive testing on gcc/clang |
| NEON timing quirks | Medium | Profile-guided optimization |
| Index restriction (vqtbl) | High | Careful LUT layout design |
| No physical ARM hardware | Low | Cross-compile + remote testing |

---

## Reference Implementation

**RandomX `intrin_portable.h`** (800+ lines) provides the gold standard:
- Support for SSE2, AVX2, AVX-512, NEON, VSX
- Platform-specific optimizations
- Proven in production (Monero mining)

**Key lessons from RandomX**:
1. Keep abstraction minimal - only abstract what's needed
2. Use `FORCE_INLINE` for all intrinsics
3. Provide compile-time and runtime fallback
4. Test on actual hardware (emulation insufficient)

---

## Documentation Updates Required

After implementation:
- [ ] `README.md` - Add ARM support to production status
- [ ] `docs/environment/README.md` - Add Apple Silicon to platform matrix
- [ ] `docs/architecture/architecture.md` - Update SIMD section
- [ ] `docs/build-system/README.md` - Add ARM cross-compilation guide

---

**Status**: Planning
**Priority**: Medium
**Estimated Effort**: 6-8 weeks
**Required Resources**: ARM hardware for validation
