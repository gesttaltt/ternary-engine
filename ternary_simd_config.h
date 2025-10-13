// ternary_simd_config.h — SIMD width abstraction layer for multi-ISA support
//
// Copyright 2025 Ternary Core Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// =============================================================================
// CURRENT STATUS: FUTURE USE INFRASTRUCTURE
// =============================================================================
//
// ⚠️ NOTE: This abstraction layer is NOT yet used by ternary_simd_engine.cpp
//
// STATUS:
//   - This header provides multi-ISA abstraction macros (AVX-512, AVX2, NEON)
//   - Currently UNUSED by main engine (ternary_simd_engine.cpp still uses direct AVX2)
//   - USED by: ternary_c_api.h (C FFI layer), benchmarks/bench_kernels.cpp
//   - FUTURE: Refactor main engine to use these macros for AVX-512/ARM support
//
// WHY KEPT:
//   - Future-ready infrastructure for multi-platform expansion
//   - Documents portable SIMD patterns for AVX-512/ARM migration
//   - Enables gradual refactoring without breaking existing code
//
// TODO (Phase 4):
//   - Refactor ternary_simd_engine.cpp to use TERNARY_* macros
//   - Add runtime dispatch based on ternary_cpu_detect.h
//   - Benchmark AVX-512 vs AVX2 performance on capable hardware
//
// =============================================================================
// DESIGN RATIONALE
// =============================================================================
//
// OPT-PHASE3-02: SIMD width abstraction layer enables:
// - Single codebase for AVX2 (256-bit), AVX-512BW (512-bit), ARM NEON (128-bit)
// - Compiler auto-selects ISA based on -march flags
// - Zero-overhead abstractions (macros resolve at compile-time)
// - Future-proof for ARM SVE and RISC-V Vector extensions
//
// COMPILATION EXAMPLES:
//   g++ -march=native -mavx2       → Uses AVX2 (256-bit, 32 trits/op)
//   g++ -march=native -mavx512bw   → Uses AVX-512BW (512-bit, 64 trits/op)
//   aarch64-g++ -march=native      → Uses ARM NEON (128-bit, 16 trits/op)
//
// MACRO INTERFACE:
//   TERNARY_VEC          - Vector type (__m256i, __m512i, uint8x16_t, etc.)
//   TERNARY_LOAD(ptr)    - Load unaligned vector
//   TERNARY_STORE(ptr,v) - Store unaligned vector
//   TERNARY_SHUFFLE(v,i) - Shuffle bytes (LUT lookup)
//   TERNARY_AND(a,b)     - Bitwise AND
//   TERNARY_OR(a,b)      - Bitwise OR
//   TERNARY_ADD(a,b)     - Byte-wise addition (for index computation)
//   TERNARY_SET1(byte)   - Broadcast byte to all lanes
//   TERNARY_VEC_SIZE     - Elements per vector (32, 64, 16, etc.)
//
// =============================================================================

#ifndef TERNARY_SIMD_CONFIG_H
#define TERNARY_SIMD_CONFIG_H

#include <stdint.h>

// =============================================================================
// ISA Detection and Selection
// =============================================================================

// Priority: AVX-512BW > AVX2 > ARM NEON > Scalar
// Compiler defines these based on -march/-mcpu flags

#if defined(__AVX512BW__) && defined(__AVX512F__)
    #define TERNARY_USE_AVX512BW
    #include <immintrin.h>
#elif defined(__AVX2__)
    #define TERNARY_USE_AVX2
    #include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__aarch64__)
    #define TERNARY_USE_NEON
    #include <arm_neon.h>
#else
    #define TERNARY_USE_SCALAR
#endif

// =============================================================================
// AVX-512BW Configuration (512-bit vectors, 64 trits per operation)
// =============================================================================

#ifdef TERNARY_USE_AVX512BW

#define TERNARY_VEC __m512i
#define TERNARY_VEC_SIZE 64

#define TERNARY_LOAD(ptr) _mm512_loadu_si512((const __m512i*)(ptr))
#define TERNARY_STORE(ptr, v) _mm512_storeu_si512((__m512i*)(ptr), v)
#define TERNARY_SHUFFLE(v, indices) _mm512_shuffle_epi8(v, indices)
#define TERNARY_AND(a, b) _mm512_and_si512(a, b)
#define TERNARY_OR(a, b) _mm512_or_si512(a, b)
#define TERNARY_ADD(a, b) _mm512_add_epi8(a, b)
#define TERNARY_SET1(byte) _mm512_set1_epi8(byte)

// Stream stores for large arrays (non-temporal)
#define TERNARY_STREAM_STORE(ptr, v) _mm512_stream_si512((__m512i*)(ptr), v)

// LUT broadcast (64-byte table replicated across both 256-bit lanes)
// Note: AVX-512 shuffle operates on 128-bit lanes, so we need 64-byte LUTs
inline __m512i ternary_broadcast_lut_512(const uint8_t* lut_16) {
    // Load 16-byte LUT into lower 128 bits, then replicate across all 4 lanes
    __m128i lut_128 = _mm_loadu_si128((const __m128i*)lut_16);
    __m256i lut_256 = _mm256_broadcastsi128_si256(lut_128);
    return _mm512_broadcast_i64x4(lut_256);
}

#define TERNARY_BROADCAST_LUT(lut) ternary_broadcast_lut_512(lut)

// =============================================================================
// AVX2 Configuration (256-bit vectors, 32 trits per operation)
// =============================================================================

#elif defined(TERNARY_USE_AVX2)

#define TERNARY_VEC __m256i
#define TERNARY_VEC_SIZE 32

#define TERNARY_LOAD(ptr) _mm256_loadu_si256((const __m256i*)(ptr))
#define TERNARY_STORE(ptr, v) _mm256_storeu_si256((__m256i*)(ptr), v)
#define TERNARY_SHUFFLE(v, indices) _mm256_shuffle_epi8(v, indices)
#define TERNARY_AND(a, b) _mm256_and_si256(a, b)
#define TERNARY_OR(a, b) _mm256_or_si256(a, b)
#define TERNARY_ADD(a, b) _mm256_add_epi8(a, b)
#define TERNARY_SET1(byte) _mm256_set1_epi8(byte)

// Stream stores for large arrays
#define TERNARY_STREAM_STORE(ptr, v) _mm256_stream_si256((__m256i*)(ptr), v)

// LUT broadcast (16-byte table replicated across both 128-bit lanes)
inline __m256i ternary_broadcast_lut_256(const uint8_t* lut_16) {
    __m128i lut_128 = _mm_loadu_si128((const __m128i*)lut_16);
    return _mm256_broadcastsi128_si256(lut_128);
}

#define TERNARY_BROADCAST_LUT(lut) ternary_broadcast_lut_256(lut)

// =============================================================================
// ARM NEON Configuration (128-bit vectors, 16 trits per operation)
// =============================================================================

#elif defined(TERNARY_USE_NEON)

#define TERNARY_VEC uint8x16_t
#define TERNARY_VEC_SIZE 16

#define TERNARY_LOAD(ptr) vld1q_u8((const uint8_t*)(ptr))
#define TERNARY_STORE(ptr, v) vst1q_u8((uint8_t*)(ptr), v)
#define TERNARY_SHUFFLE(v, indices) vqtbl1q_u8(v, indices)  // Table lookup
#define TERNARY_AND(a, b) vandq_u8(a, b)
#define TERNARY_OR(a, b) vorrq_u8(a, b)
#define TERNARY_ADD(a, b) vaddq_u8(a, b)
#define TERNARY_SET1(byte) vdupq_n_u8(byte)

// Stream stores not available on NEON, use regular store
#define TERNARY_STREAM_STORE(ptr, v) TERNARY_STORE(ptr, v)

// LUT broadcast (16-byte table loaded directly)
inline uint8x16_t ternary_broadcast_lut_neon(const uint8_t* lut_16) {
    return vld1q_u8(lut_16);
}

#define TERNARY_BROADCAST_LUT(lut) ternary_broadcast_lut_neon(lut)

// =============================================================================
// Scalar Fallback (no SIMD)
// =============================================================================

#else // TERNARY_USE_SCALAR

// Scalar fallback: operations are element-wise
#define TERNARY_VEC uint8_t
#define TERNARY_VEC_SIZE 1

#define TERNARY_LOAD(ptr) (*(ptr))
#define TERNARY_STORE(ptr, v) (*(ptr) = (v))
#define TERNARY_SHUFFLE(v, indices) (v)  // No-op for scalar
#define TERNARY_AND(a, b) ((a) & (b))
#define TERNARY_OR(a, b) ((a) | (b))
#define TERNARY_ADD(a, b) ((a) + (b))
#define TERNARY_SET1(byte) (byte)
#define TERNARY_STREAM_STORE(ptr, v) TERNARY_STORE(ptr, v)
#define TERNARY_BROADCAST_LUT(lut) (lut)

#endif

// =============================================================================
// Portable Prefetch Hints
// =============================================================================

#if defined(__GNUC__) || defined(__clang__)
    #define TERNARY_PREFETCH(ptr) __builtin_prefetch((const char*)(ptr), 0, 3)
#elif defined(_MSC_VER)
    #include <intrin.h>
    #define TERNARY_PREFETCH(ptr) _mm_prefetch((const char*)(ptr), _MM_HINT_T0)
#else
    #define TERNARY_PREFETCH(ptr) ((void)0)  // No-op
#endif

// =============================================================================
// ISA Information Macros
// =============================================================================

#ifdef TERNARY_USE_AVX512BW
    #define TERNARY_ISA_NAME "AVX-512BW"
    #define TERNARY_ALIGNMENT 64
#elif defined(TERNARY_USE_AVX2)
    #define TERNARY_ISA_NAME "AVX2"
    #define TERNARY_ALIGNMENT 32
#elif defined(TERNARY_USE_NEON)
    #define TERNARY_ISA_NAME "ARM NEON"
    #define TERNARY_ALIGNMENT 16
#else
    #define TERNARY_ISA_NAME "Scalar"
    #define TERNARY_ALIGNMENT 1
#endif

// =============================================================================
// Usage Example (conceptual - not compiled here)
// =============================================================================

/*
// Portable binary operation using abstraction layer:
void portable_tadd(const uint8_t* A, const uint8_t* B, uint8_t* R, size_t n) {
    // Load LUT once
    TERNARY_VEC lut = TERNARY_BROADCAST_LUT(TADD_LUT.data());

    // Process in SIMD-width chunks
    for (size_t i = 0; i + TERNARY_VEC_SIZE <= n; i += TERNARY_VEC_SIZE) {
        TERNARY_VEC va = TERNARY_LOAD(A + i);
        TERNARY_VEC vb = TERNARY_LOAD(B + i);

        // Compute indices: (a << 2) | b
        TERNARY_VEC a_masked = TERNARY_AND(va, TERNARY_SET1(0x03));
        TERNARY_VEC b_masked = TERNARY_AND(vb, TERNARY_SET1(0x03));
        TERNARY_VEC a_shifted = TERNARY_ADD(TERNARY_ADD(a_masked, a_masked),
                                             TERNARY_ADD(a_masked, a_masked)); // a * 4
        TERNARY_VEC indices = TERNARY_OR(a_shifted, b_masked);

        // LUT lookup
        TERNARY_VEC vr = TERNARY_SHUFFLE(lut, indices);
        TERNARY_STORE(R + i, vr);
    }

    // Scalar tail
    for (size_t i = (n / TERNARY_VEC_SIZE) * TERNARY_VEC_SIZE; i < n; ++i) {
        R[i] = TADD_LUT[(A[i] << 2) | B[i]];
    }
}
*/

#endif // TERNARY_SIMD_CONFIG_H
