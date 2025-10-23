// ternary_simd_engine_fusion.cpp — Operation Fusion Engine (Phase 4.0 PoC)
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

#include <immintrin.h>
#include <stdint.h>
#include <thread>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "ternary_algebra.h"
#include "ternary_errors.h"
#include "ternary_fusion.h"

// MSVC compatibility
#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

namespace py = pybind11;

// Reuse optimization parameters from main engine
static const ssize_t OMP_THRESHOLD = 32768 * std::thread::hardware_concurrency();
static const ssize_t STREAM_THRESHOLD = 1000000;
constexpr int PREFETCH_DIST = 512;

#ifdef TERNARY_NO_SANITIZE
constexpr bool SANITIZE = false;
#else
constexpr bool SANITIZE = true;
#endif

// =============================================================================
// UNIFIED BINARY ARRAY PROCESSING TEMPLATE (Reused from main engine)
// =============================================================================

template <bool Sanitize = true, typename SimdOp, typename ScalarOp>
py::array_t<uint8_t> process_binary_array_fusion(
    py::array_t<uint8_t> A,
    py::array_t<uint8_t> B,
    SimdOp simd_op,
    ScalarOp scalar_op
) {
    auto a = A.unchecked<1>();
    auto b = B.unchecked<1>();
    ssize_t n = A.size();

    // Validation
    if (n != B.size()) throw ArraySizeMismatchError(n, B.size());

    py::array_t<uint8_t> out(n);
    auto r = out.mutable_unchecked<1>();
    const uint8_t* a_ptr = static_cast<const uint8_t*>(A.data());
    const uint8_t* b_ptr = static_cast<const uint8_t*>(B.data());
    uint8_t* r_ptr = static_cast<uint8_t*>(out.mutable_data());

    ssize_t i = 0;

    // PATH 1: Large arrays → OpenMP parallel
    if (n >= OMP_THRESHOLD) {
        ssize_t n_simd_blocks = (n / 32) * 32;
        bool use_streaming = (n >= STREAM_THRESHOLD);

        #pragma omp parallel for schedule(guided, 4)
        for (ssize_t idx = 0; idx < n_simd_blocks; idx += 32) {
            // Prefetching
            if (idx + PREFETCH_DIST < n_simd_blocks) {
                _mm_prefetch((const char*)(a_ptr + idx + PREFETCH_DIST), _MM_HINT_T0);
                _mm_prefetch((const char*)(b_ptr + idx + PREFETCH_DIST), _MM_HINT_T0);
            }

            __m256i va = _mm256_loadu_si256((__m256i const*)(a_ptr + idx));
            __m256i vb = _mm256_loadu_si256((__m256i const*)(b_ptr + idx));
            __m256i vr = simd_op(va, vb);

            // Streaming stores for very large arrays
            if (use_streaming) {
                _mm256_stream_si256((__m256i*)(r_ptr + idx), vr);
            } else {
                _mm256_storeu_si256((__m256i*)(r_ptr + idx), vr);
            }
        }

        if (use_streaming) {
            _mm_sfence();
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

// =============================================================================
// PHASE 4.0 PoC: Single Fused Operation
// =============================================================================

py::array_t<uint8_t> fused_tnot_tadd_array(
    py::array_t<uint8_t> A,
    py::array_t<uint8_t> B
) {
    return process_binary_array_fusion<SANITIZE>(
        A, B,
        fused_tnot_tadd_simd<SANITIZE>,
        fused_tnot_tadd_scalar
    );
}

// =============================================================================
// PYTHON BINDINGS
// =============================================================================

PYBIND11_MODULE(ternary_fusion_engine, m) {
    m.doc() = "Ternary Operation Fusion Engine - Phase 4.0 Proof of Concept\n\n"
              "Implements fused ternary operations to reduce memory traffic\n"
              "and improve performance on multi-operation workflows.\n\n"
              "Phase 4.0 Status:\n"
              "  - Single fused operation: fused_tnot_tadd(a, b)\n"
              "  - Theoretical gain: 40% memory reduction, 2-3× speedup\n"
              "  - Validation: See benchmarks/bench_fusion_poc.py\n\n"
              "Usage:\n"
              "  import ternary_fusion_engine as fusion\n"
              "  result = fusion.fused_tnot_tadd(a, b)  # tnot(tadd(a, b))\n";

    m.def("fused_tnot_tadd", &fused_tnot_tadd_array,
          py::arg("a"), py::arg("b"),
          "Fused operation: tnot(tadd(a, b))\n\n"
          "Combines saturated ternary addition with negation in a single pass,\n"
          "eliminating intermediate array allocation and reducing memory traffic.\n\n"
          "Parameters:\n"
          "  a: NumPy uint8 array (2-bit trit encoding)\n"
          "  b: NumPy uint8 array (same size as a)\n\n"
          "Returns:\n"
          "  NumPy uint8 array: result = tnot(tadd(a, b))\n\n"
          "Performance:\n"
          "  - Small arrays (<100K): ~1.2-1.5× speedup\n"
          "  - Medium arrays (100K-1M): ~1.5-2.0× speedup\n"
          "  - Large arrays (>1M): ~2.0-2.5× speedup (theoretical)\n\n"
          "Note: Actual performance depends on system memory bandwidth.\n"
          "Run benchmarks/bench_fusion_poc.py for validation on your hardware.");
}
