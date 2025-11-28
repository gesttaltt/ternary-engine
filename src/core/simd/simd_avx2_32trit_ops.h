// simd_avx2_32trit_ops.h — AVX2 SIMD kernels processing 32 trits per operation
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
// PURPOSE:
// This header extracts the pure SIMD kernel functions from ternary_simd_engine.cpp
// for use in benchmarks and standalone C++ applications without pybind11 dependency.

#ifndef SIMD_AVX2_32TRIT_OPS_H
#define SIMD_AVX2_32TRIT_OPS_H

#include <immintrin.h>
#include <stdint.h>
#include "../algebra/ternary_algebra.h"
#include "opt_canonical_index.h"

// Helper: Load 16-entry LUT and broadcast to both 128-bit lanes of 256-bit vector
static inline __m256i broadcast_lut_16(const uint8_t* lut) {
    __m128i lut_128 = _mm_loadu_si128((const __m128i*)lut);
    return _mm256_broadcastsi128_si256(lut_128);
}

// --- Pre-broadcasted LUT Cache (OPT-LUT-BROADCAST) ---
// Uses CANONICAL LUTs for SIMD operations (dual-shuffle + ADD indexing)
namespace {
    struct BroadcastedLUTs {
        __m256i tadd;
        __m256i tmul;
        __m256i tmin;
        __m256i tmax;
        __m256i tnot;

        BroadcastedLUTs()
            : tadd(broadcast_lut_16(TADD_LUT_CANONICAL.data()))
            , tmul(broadcast_lut_16(TMUL_LUT_CANONICAL.data()))
            , tmin(broadcast_lut_16(TMIN_LUT_CANONICAL.data()))
            , tmax(broadcast_lut_16(TMAX_LUT_CANONICAL.data()))
            , tnot(broadcast_lut_16(TNOT_LUT_CANONICAL.data()))
        {}
    };

    static const BroadcastedLUTs g_luts;
}

// Helper: Optional masking for sanitization (OPT-HASWELL-02)
template <bool Sanitize = true>
static inline __m256i maybe_mask(__m256i v) {
    if constexpr (Sanitize)
        return _mm256_and_si256(v, _mm256_set1_epi8(0x03));
    else
        return v;
}

// Unified binary operation template with canonical indexing (Phase 3.2)
// Uses dual-shuffle + ADD instead of shift+OR for 12-18% improvement
template <bool Sanitize = true>
static inline __m256i binary_simd_op(__m256i a, __m256i b, __m256i lut) {
    __m256i a_masked = maybe_mask<Sanitize>(a);
    __m256i b_masked = maybe_mask<Sanitize>(b);

    // Canonical indexing: Dual-shuffle + ADD (Phase 3.2 optimization)
    // Load canonical LUTs (these are compile-time constants, will be optimized)
    __m256i canon_a = _mm256_load_si256((__m256i*)CANON_A_LUT_256);
    __m256i canon_b = _mm256_load_si256((__m256i*)CANON_B_LUT_256);

    // Dual-shuffle: Two parallel shuffles (no data dependency)
    __m256i contrib_a = _mm256_shuffle_epi8(canon_a, a_masked);
    __m256i contrib_b = _mm256_shuffle_epi8(canon_b, b_masked);

    // Combine with ADD (faster than OR due to port availability)
    __m256i indices = _mm256_add_epi8(contrib_a, contrib_b);

    // Final shuffle with combined index
    return _mm256_shuffle_epi8(lut, indices);
}

// --- SIMD Kernel Functions ---

template <bool Sanitize = true>
static inline __m256i tadd_simd(__m256i a, __m256i b) {
    return binary_simd_op<Sanitize>(a, b, g_luts.tadd);
}

template <bool Sanitize = true>
static inline __m256i tmul_simd(__m256i a, __m256i b) {
    return binary_simd_op<Sanitize>(a, b, g_luts.tmul);
}

template <bool Sanitize = true>
static inline __m256i tmin_simd(__m256i a, __m256i b) {
    return binary_simd_op<Sanitize>(a, b, g_luts.tmin);
}

template <bool Sanitize = true>
static inline __m256i tmax_simd(__m256i a, __m256i b) {
    return binary_simd_op<Sanitize>(a, b, g_luts.tmax);
}

template <bool Sanitize = true>
static inline __m256i tnot_simd(__m256i a) {
    __m256i indices = maybe_mask<Sanitize>(a);
    return _mm256_shuffle_epi8(g_luts.tnot, indices);
}

#endif // SIMD_AVX2_32TRIT_OPS_H
