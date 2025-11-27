// ternary_c_api.h — Pure C API for cross-language FFI (Rust, Zig, C#, etc.)
//
// Copyright (c) 2025 Jonathan Verdun (Ternary Engine Project)
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
// DESIGN RATIONALE
// =============================================================================
//
// OPT-PHASE3-06: Cross-language interop layer enables:
// - Direct integration in Rust, Zig, C#, Go without Python dependency
// - Pure C ABI for maximum compatibility (no name mangling)
// - Header-only for easy distribution (no separate .so/.dll)
// - Memory management via caller (no allocations in C API)
//
// USAGE EXAMPLE (Rust):
//
//   use std::os::raw::{c_void, c_size_t};
//
//   extern "C" {
//       fn tadd_simd_u8(a: *const u8, b: *const u8, r: *mut u8, n: c_size_t);
//   }
//
//   let a = vec![0b00, 0b01, 0b10];  // [-1, 0, +1]
//   let b = vec![0b10, 0b01, 0b00];  // [+1, 0, -1]
//   let mut r = vec![0; 3];
//   unsafe { tadd_simd_u8(a.as_ptr(), b.as_ptr(), r.as_mut_ptr(), 3); }
//
// =============================================================================

#ifndef TERNARY_C_API_H
#define TERNARY_C_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Pure C FFI Declarations
// =============================================================================

// --- Binary Operations (element-wise) ---
// All operations process arrays of 2-bit trits packed in uint8_t
// Encoding: 0b00 = -1, 0b01 = 0, 0b10 = +1, 0b11 = invalid

/**
 * Ternary addition: R[i] = clamp(A[i] + B[i], -1, +1)
 *
 * @param A     Input array A (read-only)
 * @param B     Input array B (read-only)
 * @param R     Result array R (write-only, must be pre-allocated)
 * @param n     Number of elements (all arrays must have size n)
 */
void ternary_tadd_u8(const uint8_t* A, const uint8_t* B, uint8_t* R, size_t n);

/**
 * Ternary multiplication: R[i] = A[i] * B[i]
 *
 * @param A     Input array A (read-only)
 * @param B     Input array B (read-only)
 * @param R     Result array R (write-only, must be pre-allocated)
 * @param n     Number of elements (all arrays must have size n)
 */
void ternary_tmul_u8(const uint8_t* A, const uint8_t* B, uint8_t* R, size_t n);

/**
 * Ternary minimum: R[i] = min(A[i], B[i])
 *
 * @param A     Input array A (read-only)
 * @param B     Input array B (read-only)
 * @param R     Result array R (write-only, must be pre-allocated)
 * @param n     Number of elements (all arrays must have size n)
 */
void ternary_tmin_u8(const uint8_t* A, const uint8_t* B, uint8_t* R, size_t n);

/**
 * Ternary maximum: R[i] = max(A[i], B[i])
 *
 * @param A     Input array A (read-only)
 * @param B     Input array B (read-only)
 * @param R     Result array R (write-only, must be pre-allocated)
 * @param n     Number of elements (all arrays must have size n)
 */
void ternary_tmax_u8(const uint8_t* A, const uint8_t* B, uint8_t* R, size_t n);

// --- Unary Operations ---

/**
 * Ternary negation: R[i] = -A[i]
 * Flips sign: -1 → +1, +1 → -1, 0 → 0
 *
 * @param A     Input array A (read-only)
 * @param R     Result array R (write-only, must be pre-allocated)
 * @param n     Number of elements (all arrays must have size n)
 */
void ternary_tnot_u8(const uint8_t* A, uint8_t* R, size_t n);

// =============================================================================
// CPU Feature Detection (optional, for dynamic dispatch)
// =============================================================================

/**
 * Query available SIMD instruction set
 *
 * @return 0 = Scalar only
 *         1 = AVX2 (256-bit, 32 trits/op)
 *         2 = AVX-512BW (512-bit, 64 trits/op)
 *         3 = ARM NEON (128-bit, 16 trits/op)
 *         4 = ARM SVE (scalable)
 */
int ternary_detect_simd_level(void);

/**
 * Get human-readable SIMD level name
 *
 * @return String describing SIMD capability (e.g., "AVX2", "Scalar")
 */
const char* ternary_simd_level_name(void);

#ifdef __cplusplus
}
#endif

// =============================================================================
// Implementation (C++ with extern "C" linkage)
// =============================================================================

#ifdef __cplusplus

#include "../simd/simd_avx2_32trit_ops.h"
#include "../simd/cpu_simd_capability.h"
#include "../algebra/ternary_algebra.h"
#include <algorithm>

extern "C" {

// --- Binary Operations Implementation ---

void ternary_tadd_u8(const uint8_t* A, const uint8_t* B, uint8_t* R, size_t n) {
    size_t i = 0;

    // SIMD path (32 elements at a time)
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(A + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(B + i));
        __m256i vr = tadd_simd<true>(va, vb);
        _mm256_storeu_si256((__m256i*)(R + i), vr);
    }

    // Scalar tail (remaining elements)
    for (; i < n; ++i) {
        R[i] = tadd(A[i], B[i]);
    }
}

void ternary_tmul_u8(const uint8_t* A, const uint8_t* B, uint8_t* R, size_t n) {
    size_t i = 0;

    // SIMD path
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(A + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(B + i));
        __m256i vr = tmul_simd<true>(va, vb);
        _mm256_storeu_si256((__m256i*)(R + i), vr);
    }

    // Scalar tail
    for (; i < n; ++i) {
        R[i] = tmul(A[i], B[i]);
    }
}

void ternary_tmin_u8(const uint8_t* A, const uint8_t* B, uint8_t* R, size_t n) {
    size_t i = 0;

    // SIMD path
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(A + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(B + i));
        __m256i vr = tmin_simd<true>(va, vb);
        _mm256_storeu_si256((__m256i*)(R + i), vr);
    }

    // Scalar tail
    for (; i < n; ++i) {
        R[i] = tmin(A[i], B[i]);
    }
}

void ternary_tmax_u8(const uint8_t* A, const uint8_t* B, uint8_t* R, size_t n) {
    size_t i = 0;

    // SIMD path
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(A + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(B + i));
        __m256i vr = tmax_simd<true>(va, vb);
        _mm256_storeu_si256((__m256i*)(R + i), vr);
    }

    // Scalar tail
    for (; i < n; ++i) {
        R[i] = tmax(A[i], B[i]);
    }
}

// --- Unary Operations Implementation ---

void ternary_tnot_u8(const uint8_t* A, uint8_t* R, size_t n) {
    size_t i = 0;

    // SIMD path
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(A + i));
        __m256i vr = tnot_simd<true>(va);
        _mm256_storeu_si256((__m256i*)(R + i), vr);
    }

    // Scalar tail
    for (; i < n; ++i) {
        R[i] = tnot(A[i]);
    }
}

// --- CPU Feature Detection Implementation ---

int ternary_detect_simd_level(void) {
    SIMDLevel level = detect_best_simd();
    return static_cast<int>(level);
}

const char* ternary_simd_level_name(void) {
    SIMDLevel level = detect_best_simd();
    return simd_level_name(level);
}

} // extern "C"

#endif // __cplusplus

#endif // TERNARY_C_API_H
