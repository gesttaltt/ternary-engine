// tritnet_inference_avx2.h -- AVX2-vectorized TritNet inference (Phase 3)
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
// DESIGN
// =============================================================================
//
// Vectorizes tritnet_inference.h's forward3 (h = ReLU(x @ W + b)) across the
// OUTPUT dimension rather than the input/reduction dimension: W is stored
// [IN][HID] (row-major, x @ W layout), so for a fixed input index i, row
// W[i][0..HID-1] is contiguous in memory. The accumulation loop therefore
// broadcasts one input scalar x[i] at a time and does an 8-wide FMA against
// 8 contiguous output lanes, accumulating directly into the (memory-resident)
// output buffer -- an outer-product GEMV pattern, not a horizontal-reduction
// dot product (which int8 weights stored this way don't support cheaply).
//
// Requires HID and OUT_PADDED to both be multiples of 8 (no scalar tail) --
// generate_weights_header.py enforces this at codegen time (asserts on
// HIDDEN, pads W3/B3's output dim to OUT_PADDED). IN (5 or 10) is the
// scalar outer-loop trip count and does NOT need to be a multiple of 8.
//
// Ternary int8 weights are converted to float via
// _mm256_cvtepi8_epi32 (sign-extend 8x int8 -> 8x int32) then
// _mm256_cvtepi32_ps, one 8-wide load per inner-loop iteration.
//
// COMPILE: requires -mavx2 (or -march=native on an AVX2+ host) and
// <immintrin.h>. Guarded by __AVX2__ so including this header on a non-AVX2
// build target is a hard compile error rather than a silent scalar fallback
// -- callers select scalar vs AVX2 at compile time (see
// bench_tritnet_inference.cpp), matching this repo's existing convention of
// explicit build-time kernel selection for dev/benchmark code (production
// runtime dispatch, e.g. core/simd/cpu_simd_capability.h's has_avx2(), is a
// separate concern for when this graduates out of Phase 3).

#pragma once

#ifndef __AVX2__
#error "tritnet_inference_avx2.h requires AVX2 (-mavx2 or -march=native on an AVX2+ host)"
#endif

#include <cstdint>
#include <immintrin.h>

#include "tritnet_weights.h"
#include "../../../src/core/algebra/ternary_algebra.h"  // trit, trit_to_int, int_to_trit

namespace tritnet {
namespace avx2 {

namespace detail {

// out[0..HID-1] = ApplyReLU ? ReLU(x @ W + b) : (x @ W + b)
// x[IN], W[IN][HID] (row-major), b[HID], out[HID]; HID must be a multiple of 8.
template<int IN, int HID, bool ApplyReLU>
inline void layer_avx2(const float x[IN], const int8_t W[IN][HID],
                        const float B[HID], float out[HID]) {
    for (int j = 0; j < HID; j += 8) {
        _mm256_storeu_ps(&out[j], _mm256_loadu_ps(&B[j]));
    }

    for (int i = 0; i < IN; ++i) {
        const __m256 xi = _mm256_set1_ps(x[i]);
        const int8_t* row = W[i];
        for (int j = 0; j < HID; j += 8) {
            // Load 8 int8 weights (low 64 bits of a 128-bit reg), sign-extend
            // to 8x int32, convert to float, FMA into the running total.
            __m128i w8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row + j));
            __m256i w32 = _mm256_cvtepi8_epi32(w8);
            __m256 wf = _mm256_cvtepi32_ps(w32);

            __m256 acc = _mm256_loadu_ps(&out[j]);
            acc = _mm256_fmadd_ps(xi, wf, acc);
            _mm256_storeu_ps(&out[j], acc);
        }
    }

    if (ApplyReLU) {
        const __m256 zero = _mm256_setzero_ps();
        for (int j = 0; j < HID; j += 8) {
            __m256 v = _mm256_max_ps(_mm256_loadu_ps(&out[j]), zero);
            _mm256_storeu_ps(&out[j], v);
        }
    }
}

template<int IN, int HID, int NOUT, int OUT_PADDED>
inline void forward3_avx2(
    const float x[IN],
    const int8_t W1[IN][HID], const float B1[HID],
    const int8_t W2[HID][HID], const float B2[HID],
    const int8_t W3[HID][OUT_PADDED], const float B3[OUT_PADDED],
    int8_t out_trits[NOUT]
) {
    static_assert(HID % 8 == 0, "HID must be a multiple of 8 for AVX2 (no scalar tail)");
    static_assert(OUT_PADDED % 8 == 0, "OUT_PADDED must be a multiple of 8 for AVX2 (no scalar tail)");
    static_assert(OUT_PADDED >= NOUT * 3, "OUT_PADDED must cover all real logits");

    float h1[HID];
    layer_avx2<IN, HID, true>(x, W1, B1, h1);

    float h2[HID];
    layer_avx2<HID, HID, true>(h1, W2, B2, h2);

    float logits[OUT_PADDED];
    layer_avx2<HID, OUT_PADDED, false>(h2, W3, B3, logits);

    // Argmax decode -- identical to the scalar path, only reads the real
    // NOUT*3 lanes; padding lanes (always 0) are never touched.
    for (int k = 0; k < NOUT; ++k) {
        int best = 0;
        float best_val = logits[k * 3];
        for (int c = 1; c < 3; ++c) {
            if (logits[k * 3 + c] > best_val) {
                best_val = logits[k * 3 + c];
                best = c;
            }
        }
        out_trits[k] = static_cast<int8_t>(best - 1);
    }
}

inline void trits_to_floats(const trit* in, float* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = static_cast<float>(trit_to_int(in[i]));
}

inline void ints_to_trits(const int8_t* in, trit* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = int_to_trit(in[i]);
}

}  // namespace detail

/// AVX2 TritNet inference for tnot: 1 chunk of 5 trits in, 5 trits out.
inline void tritnet_tnot(const trit in[5], trit out[5]) {
    using namespace weights::tnot_weights;
    float x[5];
    detail::trits_to_floats(in, x, 5);
    int8_t y[5];
    detail::forward3_avx2<IN_FEATURES, HIDDEN, N_OUT_TRITS, OUT_PADDED>(x, W1, B1, W2, B2, W3, B3, y);
    detail::ints_to_trits(y, out, 5);
}

/// AVX2 TritNet inference for tadd: 2 chunks of 5 trits in, 5 trits out.
inline void tritnet_tadd(const trit a[5], const trit b[5], trit out[5]) {
    using namespace weights::tadd_weights;
    float x[10];
    detail::trits_to_floats(a, x, 5);
    detail::trits_to_floats(b, x + 5, 5);
    int8_t y[5];
    detail::forward3_avx2<IN_FEATURES, HIDDEN, N_OUT_TRITS, OUT_PADDED>(x, W1, B1, W2, B2, W3, B3, y);
    detail::ints_to_trits(y, out, 5);
}

/// AVX2 TritNet inference for tmul: 2 chunks of 5 trits in, 5 trits out.
inline void tritnet_tmul(const trit a[5], const trit b[5], trit out[5]) {
    using namespace weights::tmul_weights;
    float x[10];
    detail::trits_to_floats(a, x, 5);
    detail::trits_to_floats(b, x + 5, 5);
    int8_t y[5];
    detail::forward3_avx2<IN_FEATURES, HIDDEN, N_OUT_TRITS, OUT_PADDED>(x, W1, B1, W2, B2, W3, B3, y);
    detail::ints_to_trits(y, out, 5);
}

/// AVX2 TritNet inference for tmin: 2 chunks of 5 trits in, 5 trits out.
inline void tritnet_tmin(const trit a[5], const trit b[5], trit out[5]) {
    using namespace weights::tmin_weights;
    float x[10];
    detail::trits_to_floats(a, x, 5);
    detail::trits_to_floats(b, x + 5, 5);
    int8_t y[5];
    detail::forward3_avx2<IN_FEATURES, HIDDEN, N_OUT_TRITS, OUT_PADDED>(x, W1, B1, W2, B2, W3, B3, y);
    detail::ints_to_trits(y, out, 5);
}

/// AVX2 TritNet inference for tmax: 2 chunks of 5 trits in, 5 trits out.
inline void tritnet_tmax(const trit a[5], const trit b[5], trit out[5]) {
    using namespace weights::tmax_weights;
    float x[10];
    detail::trits_to_floats(a, x, 5);
    detail::trits_to_floats(b, x + 5, 5);
    int8_t y[5];
    detail::forward3_avx2<IN_FEATURES, HIDDEN, N_OUT_TRITS, OUT_PADDED>(x, W1, B1, W2, B2, W3, B3, y);
    detail::ints_to_trits(y, out, 5);
}

}  // namespace avx2
}  // namespace tritnet
