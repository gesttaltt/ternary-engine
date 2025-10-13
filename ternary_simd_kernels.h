// ternary_simd_kernels.h — Standalone SIMD kernels (no pybind11 dependency)
//
// Copyright (c) 2025 Jonathan Verdun (Ternary Core Experimental Project)
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

#ifndef TERNARY_SIMD_KERNELS_H
#define TERNARY_SIMD_KERNELS_H

#include <immintrin.h>
#include <stdint.h>
#include "ternary_algebra.h"

// Helper: Load 16-entry LUT and broadcast to both 128-bit lanes of 256-bit vector
static inline __m256i broadcast_lut_16(const uint8_t* lut) {
    __m128i lut_128 = _mm_loadu_si128((const __m128i*)lut);
    return _mm256_broadcastsi128_si256(lut_128);
}

// --- Pre-broadcasted LUT Cache (OPT-LUT-BROADCAST) ---
namespace {
    struct BroadcastedLUTs {
        __m256i tadd;
        __m256i tmul;
        __m256i tmin;
        __m256i tmax;
        __m256i tnot;

        BroadcastedLUTs()
            : tadd(broadcast_lut_16(TADD_LUT.data()))
            , tmul(broadcast_lut_16(TMUL_LUT.data()))
            , tmin(broadcast_lut_16(TMIN_LUT.data()))
            , tmax(broadcast_lut_16(TMAX_LUT.data()))
            , tnot(broadcast_lut_16(TNOT_LUT_SIMD.data()))
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

// Unified binary operation template
template <bool Sanitize = true>
static inline __m256i binary_simd_op(__m256i a, __m256i b, __m256i lut) {
    __m256i a_masked = maybe_mask<Sanitize>(a);
    __m256i b_masked = maybe_mask<Sanitize>(b);
    __m256i a_shifted = _mm256_add_epi8(_mm256_add_epi8(a_masked, a_masked),
                                         _mm256_add_epi8(a_masked, a_masked)); // a * 4
    __m256i indices = _mm256_or_si256(a_shifted, b_masked);
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

#endif // TERNARY_SIMD_KERNELS_H
