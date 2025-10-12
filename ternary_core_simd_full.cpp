// ternary_core_simd_full.cpp — AVX2-accelerated ternary logic operations
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
// DESIGN NOTE: Phase 0.5 - LUT-Based SIMD Implementation (OPT-061)
// This implementation uses SIMD shuffle instructions (_mm256_shuffle_epi8) for
// parallel lookup table operations, replacing the previous int8 arithmetic approach.
// Each operation performs 32 parallel LUT lookups across vector lanes, achieving
// the same LUT-based architecture used in scalar operations (Phase 0).
//
// Previous approach (Phase 0, pre-OPT-061):
// - Converted trits to inverted int8 representation
// - Used arithmetic SIMD intrinsics (_mm256_adds_epi8, _mm256_min_epi8, etc.)
// - Required conversions and clamping overhead
//
// Current approach (Phase 0.5, OPT-061):
// - Direct LUT lookups via _mm256_shuffle_epi8
// - No conversions or arithmetic operations
// - Completes Sidestep #2 (SIMD Vectorization with LUTs)

#include <immintrin.h>
#include <stdint.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "ternary_core.h"

// MSVC compatibility: ssize_t is not standard C++
#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

namespace py = pybind11;

// --- LUT-Based SIMD Operations (OPT-061) ---
// Each operation uses _mm256_shuffle_epi8 for 32 parallel LUT lookups.
// The LUTs are the same 16-entry tables used in scalar operations (ternary_core.h).

// Helper: Load 16-entry LUT and broadcast to both 128-bit lanes of 256-bit vector
static inline __m256i broadcast_lut_16(const uint8_t* lut) {
    // Load 16 bytes into lower 128 bits, then duplicate to upper 128 bits
    __m128i lut_128 = _mm_loadu_si128((const __m128i*)lut);
    return _mm256_broadcastsi128_si256(lut_128);
}

// Helper: Mask to ensure only lower 2 bits of each byte are used (sanitize input)
static inline __m256i mask_trit(__m256i v) {
    return _mm256_and_si256(v, _mm256_set1_epi8(0x03));
}

// --- Binary Operations (tadd, tmul, tmin, tmax) ---
// Index formula: (a << 2) | b, where a and b are 2-bit trits

static inline __m256i tadd_simd(__m256i a, __m256i b) {
    // Build 4-bit indices: (a << 2) | b
    // Since there's no byte-level shift in AVX2, use _mm256_add_epi8 with itself (x+x+x+x = x*4)
    __m256i a_masked = mask_trit(a);
    __m256i b_masked = mask_trit(b);
    __m256i a_shifted = _mm256_add_epi8(_mm256_add_epi8(a_masked, a_masked),
                                         _mm256_add_epi8(a_masked, a_masked)); // a * 4
    __m256i indices = _mm256_or_si256(a_shifted, b_masked);

    // Load and broadcast TADD_LUT
    __m256i lut = broadcast_lut_16(TADD_LUT);

    // Perform 32 parallel LUT lookups
    return _mm256_shuffle_epi8(lut, indices);
}

static inline __m256i tmul_simd(__m256i a, __m256i b) {
    __m256i a_masked = mask_trit(a);
    __m256i b_masked = mask_trit(b);
    __m256i a_shifted = _mm256_add_epi8(_mm256_add_epi8(a_masked, a_masked),
                                         _mm256_add_epi8(a_masked, a_masked));
    __m256i indices = _mm256_or_si256(a_shifted, b_masked);

    __m256i lut = broadcast_lut_16(TMUL_LUT);
    return _mm256_shuffle_epi8(lut, indices);
}

static inline __m256i tmin_simd(__m256i a, __m256i b) {
    __m256i a_masked = mask_trit(a);
    __m256i b_masked = mask_trit(b);
    __m256i a_shifted = _mm256_add_epi8(_mm256_add_epi8(a_masked, a_masked),
                                         _mm256_add_epi8(a_masked, a_masked));
    __m256i indices = _mm256_or_si256(a_shifted, b_masked);

    __m256i lut = broadcast_lut_16(TMIN_LUT);
    return _mm256_shuffle_epi8(lut, indices);
}

static inline __m256i tmax_simd(__m256i a, __m256i b) {
    __m256i a_masked = mask_trit(a);
    __m256i b_masked = mask_trit(b);
    __m256i a_shifted = _mm256_add_epi8(_mm256_add_epi8(a_masked, a_masked),
                                         _mm256_add_epi8(a_masked, a_masked));
    __m256i indices = _mm256_or_si256(a_shifted, b_masked);

    __m256i lut = broadcast_lut_16(TMAX_LUT);
    return _mm256_shuffle_epi8(lut, indices);
}

// --- Unary Operation (tnot) ---
// Index formula: a & 0x03 (2-bit trit value)
// Note: TNOT_LUT only has 4 entries, so we pad it to 16 for shuffle compatibility

static inline __m256i tnot_simd(__m256i a) {
    __m256i indices = mask_trit(a);

    // Create padded 16-entry TNOT LUT (replicate pattern)
    alignas(16) static const uint8_t TNOT_LUT_16[16] = {
        0b10, 0b01, 0b00, 0b00,  // Original TNOT_LUT[0-3]
        0b10, 0b01, 0b00, 0b00,  // Replicate for indices 4-7
        0b10, 0b01, 0b00, 0b00,  // Replicate for indices 8-11
        0b10, 0b01, 0b00, 0b00   // Replicate for indices 12-15
    };

    __m256i lut = broadcast_lut_16(TNOT_LUT_16);
    return _mm256_shuffle_epi8(lut, indices);
}

// --- Macro template for arrays ---
#define TERNARY_OP_SIMD(func) \
py::array_t<uint8_t> func##_array(py::array_t<uint8_t> A, py::array_t<uint8_t> B) { \
    auto a = A.unchecked<1>(); \
    auto b = B.unchecked<1>(); \
    ssize_t n = A.size(); \
    if (n != B.size()) throw std::runtime_error("Arrays must match"); \
    py::array_t<uint8_t> out(n); \
    auto r = out.mutable_unchecked<1>(); \
    const uint8_t* a_ptr = static_cast<const uint8_t*>(A.data()); \
    const uint8_t* b_ptr = static_cast<const uint8_t*>(B.data()); \
    uint8_t* r_ptr = static_cast<uint8_t*>(out.mutable_data()); \
    ssize_t i = 0; \
    for (; i + 32 <= n; i += 32) { \
        __m256i va = _mm256_loadu_si256((__m256i const*)(a_ptr + i)); \
        __m256i vb = _mm256_loadu_si256((__m256i const*)(b_ptr + i)); \
        __m256i vr = func##_simd(va, vb); \
        _mm256_storeu_si256((__m256i*)(r_ptr + i), vr); \
    } \
    for (; i < n; ++i) r[i] = func(a[i], b[i]); \
    return out; \
}

// --- Unary ---
py::array_t<uint8_t> tnot_array(py::array_t<uint8_t> A) {
    auto a = A.unchecked<1>();
    ssize_t n = A.size();
    py::array_t<uint8_t> out(n);
    auto r = out.mutable_unchecked<1>();
    const uint8_t* a_ptr = static_cast<const uint8_t*>(A.data());
    uint8_t* r_ptr = static_cast<uint8_t*>(out.mutable_data());
    ssize_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((__m256i const*)(a_ptr + i));
        __m256i vr = tnot_simd(va);
        _mm256_storeu_si256((__m256i*)(r_ptr + i), vr);
    }
    for (; i < n; ++i) r[i] = tnot(a[i]);
    return out;
}

// --- Instantiate wrappers ---
TERNARY_OP_SIMD(tadd)
TERNARY_OP_SIMD(tmul)
TERNARY_OP_SIMD(tmin)
TERNARY_OP_SIMD(tmax)

PYBIND11_MODULE(ternary_core_simd_full, m) {
    m.def("tadd", &tadd_array);
    m.def("tmul", &tmul_array);
    m.def("tmin", &tmin_array);
    m.def("tmax", &tmax_array);
    m.def("tnot", &tnot_array);
}
