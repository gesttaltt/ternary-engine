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
// =============================================================================
// DESIGN EVOLUTION
// =============================================================================
//
// Phase 0.5 - LUT-Based SIMD Implementation (OPT-061):
// - Replaced arithmetic SIMD with _mm256_shuffle_epi8 for parallel LUT lookups
// - 32 parallel LUT lookups per operation
// - Unified semantic domain with scalar operations (no conversions)
// - Eliminated arithmetic overhead and clamping
//
// Phase 1 - Optimization Exploration (DEPRECATED):
// - OPT-066: Aligned vs unaligned load branching → Removed (negligible benefit)
// - OPT-041: Manual 2x loop unrolling → Removed (compiler auto-optimizes)
// - OPT-001: OpenMP threading (>100K elements) → RETAINED
// - Result: 6 runtime paths, high complexity, unstable measurement
//
// Phase 2 - Complexity Compression (CURRENT):
// - Template-based unification of binary and unary operations
// - Eliminated aligned/unaligned branching (modern CPUs: unaligned ≈ aligned)
// - Removed manual unrolling (trust compiler optimization)
// - Result: 3 execution paths (OpenMP/Serial-SIMD/Tail), 73% code reduction
// - Achieves "phase coherence": complexity ↓ while performance stable (< 5% loss)
//
// COMPATIBILITY NOTE:
// - Current implementation uses basic AVX2 operations compatible with all AVX2 CPUs
// - No Haswell-specific optimizations applied yet (pre-Haswell-optimization era)
// - Uses fundamental AVX2 intrinsics: loadu, storeu, shuffle_epi8, or, and, add
// - Compatible with Intel AVX2 (Haswell 2013+) and AMD AVX2 (Excavator 2015+)
//
// =============================================================================

#include <immintrin.h>
#include <stdint.h>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "ternary_core.h"

// MSVC compatibility: ssize_t is not standard C++
#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

namespace py = pybind11;

// OPT-001: OpenMP threshold for large array parallelization
// Arrays >= 100K elements will use multi-threaded processing
static const ssize_t OMP_THRESHOLD = 100000;

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

// =============================================================================
// Phase 2: Unified Template-Based Array Processing (6→3 Path Collapse)
// =============================================================================
//
// DESIGN RATIONALE:
// - Eliminates aligned vs unaligned branching (modern CPUs: unaligned ≈ aligned)
// - Removes manual loop unrolling (compiler auto-optimizes)
// - Unifies binary (Arity=2) and unary (Arity=1) operations
// - Result: 3 paths instead of 6, 73% code reduction
//
// PATH 1: OpenMP parallel for large arrays (n >= 100K)
// PATH 2: Serial SIMD loop for small arrays
// PATH 3: Scalar tail for remaining elements
//
// =============================================================================

// --- Unified Binary Operation Template ---
template <typename SimdOp, typename ScalarOp>
py::array_t<uint8_t> process_binary_array(
    py::array_t<uint8_t> A,
    py::array_t<uint8_t> B,
    SimdOp simd_op,
    ScalarOp scalar_op
) {
    auto a = A.unchecked<1>();
    auto b = B.unchecked<1>();
    ssize_t n = A.size();
    if (n != B.size()) throw std::runtime_error("Arrays must match");

    py::array_t<uint8_t> out(n);
    auto r = out.mutable_unchecked<1>();
    const uint8_t* a_ptr = static_cast<const uint8_t*>(A.data());
    const uint8_t* b_ptr = static_cast<const uint8_t*>(B.data());
    uint8_t* r_ptr = static_cast<uint8_t*>(out.mutable_data());

    ssize_t i = 0;

    // PATH 1: Large arrays → OpenMP parallel
    if (n >= OMP_THRESHOLD) {
        ssize_t n_simd_blocks = (n / 32) * 32;

        #pragma omp parallel for schedule(static)
        for (ssize_t idx = 0; idx < n_simd_blocks; idx += 32) {
            __m256i va = _mm256_loadu_si256((__m256i const*)(a_ptr + idx));
            __m256i vb = _mm256_loadu_si256((__m256i const*)(b_ptr + idx));
            __m256i vr = simd_op(va, vb);
            _mm256_storeu_si256((__m256i*)(r_ptr + idx), vr);
        }

        i = n_simd_blocks;
    }
    // PATH 2: Small arrays → Serial SIMD
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

    return out;
}

// --- Unified Unary Operation Template ---
template <typename SimdOp, typename ScalarOp>
py::array_t<uint8_t> process_unary_array(
    py::array_t<uint8_t> A,
    SimdOp simd_op,
    ScalarOp scalar_op
) {
    auto a = A.unchecked<1>();
    ssize_t n = A.size();

    py::array_t<uint8_t> out(n);
    auto r = out.mutable_unchecked<1>();
    const uint8_t* a_ptr = static_cast<const uint8_t*>(A.data());
    uint8_t* r_ptr = static_cast<uint8_t*>(out.mutable_data());

    ssize_t i = 0;

    // PATH 1: Large arrays → OpenMP parallel
    if (n >= OMP_THRESHOLD) {
        ssize_t n_simd_blocks = (n / 32) * 32;

        #pragma omp parallel for schedule(static)
        for (ssize_t idx = 0; idx < n_simd_blocks; idx += 32) {
            __m256i va = _mm256_loadu_si256((__m256i const*)(a_ptr + idx));
            __m256i vr = simd_op(va);
            _mm256_storeu_si256((__m256i*)(r_ptr + idx), vr);
        }

        i = n_simd_blocks;
    }
    // PATH 2: Small arrays → Serial SIMD
    else {
        for (; i + 32 <= n; i += 32) {
            __m256i va = _mm256_loadu_si256((__m256i const*)(a_ptr + i));
            __m256i vr = simd_op(va);
            _mm256_storeu_si256((__m256i*)(r_ptr + i), vr);
        }
    }

    // PATH 3: Scalar tail
    for (; i < n; ++i) {
        r[i] = scalar_op(a[i]);
    }

    return out;
}

// =============================================================================
// Operation Wrappers (replacing macro-generated code)
// =============================================================================

// --- Binary Operations ---
py::array_t<uint8_t> tadd_array(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return process_binary_array(A, B, tadd_simd, tadd);
}

py::array_t<uint8_t> tmul_array(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return process_binary_array(A, B, tmul_simd, tmul);
}

py::array_t<uint8_t> tmin_array(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return process_binary_array(A, B, tmin_simd, tmin);
}

py::array_t<uint8_t> tmax_array(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return process_binary_array(A, B, tmax_simd, tmax);
}

// --- Unary Operation ---
py::array_t<uint8_t> tnot_array(py::array_t<uint8_t> A) {
    return process_unary_array(A, tnot_simd, tnot);
}

PYBIND11_MODULE(ternary_core_simd_full, m) {
    m.def("tadd", &tadd_array);
    m.def("tmul", &tmul_array);
    m.def("tmin", &tmin_array);
    m.def("tmax", &tmax_array);
    m.def("tnot", &tnot_array);
}
