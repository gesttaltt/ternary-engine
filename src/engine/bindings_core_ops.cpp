// ternary_simd_engine.cpp — AVX2-accelerated ternary logic operations
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
// CURRENT IMPLEMENTATION: DIRECT AVX2 INTRINSICS
// =============================================================================
//
// ⚠️ TODO (Phase 4): Refactor to use ternary_simd_config.h for multi-ISA support
//
// CURRENT STATE:
//   - This file uses DIRECT AVX2 intrinsics (__m256i, _mm256_*, etc.)
//   - Hardcoded to 256-bit vectors (32 trits per operation)
//   - Does NOT use abstraction layer from ternary_simd_config.h
//
// FUTURE REFACTORING:
//   - Replace __m256i → TERNARY_VEC
//   - Replace _mm256_loadu_si256 → TERNARY_LOAD
//   - Replace _mm256_storeu_si256 → TERNARY_STORE
//   - Replace _mm256_shuffle_epi8 → TERNARY_SHUFFLE
//   - Add runtime dispatch based on ternary_cpu_detect.h
//
// BENEFITS OF REFACTORING:
//   - Single codebase for AVX-512BW (512-bit, 64 trits/op)
//   - ARM NEON support (128-bit, 16 trits/op)
//   - Compiler auto-selects ISA based on -march flags
//   - DRY principle: eliminate duplicate SIMD patterns
//
// See ternary_simd_config.h for abstraction layer design.
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
// - OPT-HASWELL-02 applied: Template-based optional masking (3-5% gain)
// - OPT-HASWELL-01 (shift replacement) SKIPPED - Technical rationale:
//   * AVX2 lacks byte-level shift instructions (_mm256_slli_epi8 doesn't exist)
//   * Triple-add pattern is actually optimal for AVX2 (semantically equivalent to shift)
//   * Word-level shifts would require emulation/repacking (performance loss + complexity)
//   * Skipping this optimization ensures clean, reproducible benchmarks on AVX2 CPUs
//   * Avoids false performance deltas from instruction emulation artifacts
//   * Future: Can add byte-level shifts when targeting AVX-512BW or ARM SVE
// - Uses fundamental AVX2 intrinsics: loadu, storeu, shuffle_epi8, or, and, add
// - Compatible with Intel AVX2 (Haswell 2013+) and AMD AVX2 (Excavator 2015+)
//
// =============================================================================

#include <immintrin.h>
#include <stdint.h>
#include <thread>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "core/algebra/ternary_algebra.h"
#include "core/common/ternary_errors.h"
#include "core/simd/cpu_simd_capability.h"
#include "core/simd/simd_avx2_32trit_ops.h"
#include "core/simd/fused_binary_unary_ops.h"
#include "core/simd/fused_bridge_ops.h"
#include "core/config/optimization_config.h"
#include "core/profiling/ternary_profiler.h"
#include <vector>

namespace py = pybind11;

// =============================================================================
// Profiler Integration (zero overhead when disabled)
// =============================================================================
// Global profiler domain and task names for performance analysis
// Compile with -DTERNARY_ENABLE_VTUNE to enable VTune profiling
TERNARY_PROFILE_DOMAIN(g_ternary_domain, "TernaryCore");
TERNARY_PROFILE_TASK_NAME(g_task_omp, "OpenMP_Parallel");
TERNARY_PROFILE_TASK_NAME(g_task_simd, "Serial_SIMD");
TERNARY_PROFILE_TASK_NAME(g_task_tail, "Scalar_Tail");

// =============================================================================
// SIMD Kernels - Now imported from ternary_core/simd/ternary_simd_kernels.h
// =============================================================================
// All SIMD kernel implementations (tadd_simd, tmul_simd, tmin_simd, tmax_simd, tnot_simd)
// and fusion operations (fused_tnot_tadd_simd, etc.) are now defined in the core library.
//
// This eliminates code duplication and provides a single source of truth for SIMD operations.
//
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
// VALIDATION STRATEGY:
// - Input validation occurs at entry points (process_binary_array, process_unary_array)
// - Validation is inline for simplicity: single check per entry point
// - Uses centralized exception types from ternary_errors.h for consistency
// - If validation logic grows complex, consider extracting to validation helpers
// - Current validation: array size matching (binary ops), no validation needed (unary ops)
//
// =============================================================================

// --- Unified Binary Operation Template ---
//
// Templated on element type T (2026-08-18, CLAUDE.md gap #6 Cluster A): this
// function used to be duplicated near-byte-for-byte as process_binary_array
// (uint8_t) and process_binary_array_int8 (int8_t) -- same OMP/streaming-
// store/prefetch/sfence logic, differing only in element type. T is a pure
// mechanical parameterization; no logic inside this function depends on
// which of the two types it is (the intrinsics operate on raw 256-bit
// register contents regardless of signedness). See
// reports/2026-08-18/GAP6_DUPLICATION_SCOPE.md for the ~150 duplicated
// lines this replaces, and reports/2026-08-18/CLUSTER_A_TEMPLATE_UNIFICATION.md
// for the before/after benchmark this doc's "Modifying Hot Paths" rule
// requires before touching this file.
//
// Multi-dimensional arrays (2026-08-18): the SIMD/OMP paths below have
// always operated on a flat pointer + element count -- shape was never
// actually load-bearing to the computation, only to the old
// A.unchecked<1>() validation call, which rejected anything but exactly
// 1-D. Generalized to accept any shape, as long as A and B match exactly
// (no broadcasting -- see ArrayShapeMismatchError's own doc comment) and
// both are C-contiguous (required for flat indexing to be correct; same
// convention as py_array_validate.h's GEMM-binding checks, gap #6 Cluster
// C). The output array is constructed with the same shape as the input,
// not flattened.
//
// PERFORMANCE NOTE: the first version of this used py::buffer_info
// (A.request()/B.request()) for shape/contiguity access. buffer_info
// allocates two std::vectors (shape + strides) per call -- 4 heap
// allocations per binary op -- which this doc's own before/after benchmark
// (required before touching this file, see
// reports/2026-08-18/MULTIDIM_ARRAY_SUPPORT.md) caught as a real ~2x
// slowdown at small array sizes and ~6-7% even at 100K elements. Rewritten
// to use array_t's lightweight direct accessors (.flags(), .ndim(),
// .shape(d), .data()) instead, which dereference the numpy array's
// internal C struct directly with zero allocation -- the same fix already
// applied to py_array_validate.h for the same reason. std::vector is only
// constructed in the (rare, already-slow-path) shape-mismatch error case.
template <typename T, bool Sanitize = true, typename SimdOp, typename ScalarOp>
py::array_t<T> process_binary_array(
    py::array_t<T> A,
    py::array_t<T> B,
    SimdOp simd_op,
    ScalarOp scalar_op
) {
    if (!(A.flags() & py::array::c_style)) throw std::invalid_argument("A must be C-contiguous (row-major)");
    if (!(B.flags() & py::array::c_style)) throw std::invalid_argument("B must be C-contiguous (row-major)");

    ssize_t n = A.size();
    ssize_t ndim_a = A.ndim();
    ssize_t ndim_b = B.ndim();

    // ENTRY-POINT VALIDATION: Ensure arrays match in shape. 1-D is the
    // overwhelmingly common case (every pre-2026-08-18 caller), so it gets
    // its own branch using exactly the single-comparison check this file
    // always used -- not just for the (already fast) comparison itself,
    // but to avoid perturbing the compiler's view of the hot loop below
    // with the more general shape-array walk multi-dimensional inputs
    // need. For 1-D arrays "different shape" and "different total size"
    // are the same condition, so this also preserves
    // ArraySizeMismatchError's exact behavior/wording for every existing
    // 1-D caller unconditionally, not just as a side effect of the
    // general path below.
    if (ndim_a == 1 && ndim_b == 1) {
        if (n != B.size()) throw ArraySizeMismatchError(n, B.size());
    } else {
        bool same_shape = (ndim_a == ndim_b);
        const ssize_t* shape_a = A.shape();
        const ssize_t* shape_b = B.shape();
        for (ssize_t d = 0; same_shape && d < ndim_a; ++d) {
            same_shape = (shape_a[d] == shape_b[d]);
        }
        if (!same_shape) {
            if (n != B.size()) throw ArraySizeMismatchError(n, B.size());
            // Multi-dimensional arrays with the same element count but
            // incompatible shapes (e.g. (3,4) vs (4,3)) get the more
            // specific ArrayShapeMismatchError instead of a confusing
            // "sizes match" non-error.
            throw ArrayShapeMismatchError(
                std::vector<size_t>(shape_a, shape_a + ndim_a),
                std::vector<size_t>(shape_b, shape_b + ndim_b)
            );
        }
    }

    py::array_t<T> out = (ndim_a == 1)
        ? py::array_t<T>(n)
        : py::array_t<T>(std::vector<ssize_t>(A.shape(), A.shape() + ndim_a));
    const T* a_ptr = static_cast<const T*>(A.data());
    const T* b_ptr = static_cast<const T*>(B.data());
    T* r_ptr = static_cast<T*>(out.mutable_data());

    ssize_t i = 0;

    // PATH 1: Large arrays → OpenMP parallel (NUMA-aware scheduling)
    if (n >= OMP_THRESHOLD) {
        TERNARY_PROFILE_TASK_BEGIN(g_ternary_domain, g_task_omp);
        ssize_t n_simd_blocks = (n / 32) * 32;
        // FIX: Only use streaming stores if array is large AND output is 32-byte aligned
        bool use_streaming = (n >= STREAM_THRESHOLD) && is_aligned_32(r_ptr);

        // OPT-NUMA: guided scheduling for multi-CCD CPUs (Ryzen/EPYC)
        // Guided schedule adapts chunk sizes dynamically
        // Note: proc_bind(spread) removed for MSVC compatibility (OpenMP 4.0 feature)
        // FENCE FIX: split into parallel + for so each thread can sfence its own
        // NT stores exactly once, after its share of the loop but before the
        // region's implicit join barrier. A single sfence per thread guarantees
        // that thread's writes are globally visible; sfence-per-block (the prior
        // version) issued the same guarantee ~n_simd_blocks/32 times redundantly.
        #pragma omp parallel
        {
            #pragma omp for schedule(guided, 4)
            for (ssize_t idx = 0; idx < n_simd_blocks; idx += 32) {
                // OPT-PREFETCH: Prefetch next cache lines to hide memory latency
                if (idx + PREFETCH_DIST < n_simd_blocks) {
                    _mm_prefetch((const char*)(a_ptr + idx + PREFETCH_DIST), _MM_HINT_T0);
                    _mm_prefetch((const char*)(b_ptr + idx + PREFETCH_DIST), _MM_HINT_T0);
                }

                __m256i va = _mm256_loadu_si256((__m256i const*)(a_ptr + idx));
                __m256i vb = _mm256_loadu_si256((__m256i const*)(b_ptr + idx));
                __m256i vr = simd_op(va, vb);

                // OPT-STREAM: Use streaming stores for very large arrays to reduce cache pollution
                // FIX: Alignment check performed above, safe to use streaming stores here
                if (use_streaming) {
                    _mm256_stream_si256((__m256i*)(r_ptr + idx), vr);
                } else {
                    _mm256_storeu_si256((__m256i*)(r_ptr + idx), vr);
                }
            }

            // Fence once per thread, after this thread's share of streaming
            // stores, before the parallel region's implicit barrier.
            if (use_streaming) {
                _mm_sfence();
            }
        }

        i = n_simd_blocks;
        TERNARY_PROFILE_TASK_END(g_ternary_domain);
    }
    // PATH 2: Small arrays → Serial SIMD
    else {
        TERNARY_PROFILE_TASK_BEGIN(g_ternary_domain, g_task_simd);
        for (; i + 32 <= n; i += 32) {
            __m256i va = _mm256_loadu_si256((__m256i const*)(a_ptr + i));
            __m256i vb = _mm256_loadu_si256((__m256i const*)(b_ptr + i));
            __m256i vr = simd_op(va, vb);
            _mm256_storeu_si256((__m256i*)(r_ptr + i), vr);
        }
        TERNARY_PROFILE_TASK_END(g_ternary_domain);
    }

    // PATH 3: Scalar tail
    if (i < n) {
        TERNARY_PROFILE_TASK_BEGIN(g_ternary_domain, g_task_tail);
        for (; i < n; ++i) {
            r_ptr[i] = scalar_op(a_ptr[i], b_ptr[i]);
        }
        TERNARY_PROFILE_TASK_END(g_ternary_domain);
    }

    return out;
}

// --- Unified Unary Operation Template ---
// Templated on T for the same reason as process_binary_array above.
// Multi-dimensional arrays: same approach as process_binary_array (no
// shape-match needed here, just contiguity -- there's only one operand).
template <typename T, bool Sanitize = true, typename SimdOp, typename ScalarOp>
py::array_t<T> process_unary_array(
    py::array_t<T> A,
    SimdOp simd_op,
    ScalarOp scalar_op
) {
    if (!(A.flags() & py::array::c_style)) throw std::invalid_argument("A must be C-contiguous (row-major)");

    ssize_t n = A.size();
    ssize_t ndim_a = A.ndim();
    py::array_t<T> out = (ndim_a == 1)
        ? py::array_t<T>(n)
        : py::array_t<T>(std::vector<ssize_t>(A.shape(), A.shape() + ndim_a));
    const T* a_ptr = static_cast<const T*>(A.data());
    T* r_ptr = static_cast<T*>(out.mutable_data());

    ssize_t i = 0;

    // PATH 1: Large arrays → OpenMP parallel (NUMA-aware scheduling)
    if (n >= OMP_THRESHOLD) {
        TERNARY_PROFILE_TASK_BEGIN(g_ternary_domain, g_task_omp);
        ssize_t n_simd_blocks = (n / 32) * 32;
        // FIX: Only use streaming stores if array is large AND output is 32-byte aligned
        bool use_streaming = (n >= STREAM_THRESHOLD) && is_aligned_32(r_ptr);

        // OPT-NUMA: guided scheduling for multi-CCD CPUs (Ryzen/EPYC)
        // Guided schedule adapts chunk sizes dynamically
        // Note: proc_bind(spread) removed for MSVC compatibility (OpenMP 4.0 feature)
        // FENCE FIX: split into parallel + for so each thread can sfence its own
        // NT stores exactly once, after its share of the loop but before the
        // region's implicit join barrier. A single sfence issued by only the
        // master thread after the (combined) parallel-for join does not fence
        // other worker threads' non-temporal stores — see process_binary_array.
        #pragma omp parallel
        {
            #pragma omp for schedule(guided, 4)
            for (ssize_t idx = 0; idx < n_simd_blocks; idx += 32) {
                // OPT-PREFETCH: Prefetch next cache lines to hide memory latency
                if (idx + PREFETCH_DIST < n_simd_blocks) {
                    _mm_prefetch((const char*)(a_ptr + idx + PREFETCH_DIST), _MM_HINT_T0);
                }

                __m256i va = _mm256_loadu_si256((__m256i const*)(a_ptr + idx));
                __m256i vr = simd_op(va);

                // OPT-STREAM: Use streaming stores for very large arrays to reduce cache pollution
                // FIX: Alignment check performed above, safe to use streaming stores here
                if (use_streaming) {
                    _mm256_stream_si256((__m256i*)(r_ptr + idx), vr);
                } else {
                    _mm256_storeu_si256((__m256i*)(r_ptr + idx), vr);
                }
            }

            // Fence once per thread, after this thread's share of streaming
            // stores, before the parallel region's implicit barrier.
            if (use_streaming) {
                _mm_sfence();
            }
        }

        i = n_simd_blocks;
        TERNARY_PROFILE_TASK_END(g_ternary_domain);
    }
    // PATH 2: Small arrays → Serial SIMD
    else {
        TERNARY_PROFILE_TASK_BEGIN(g_ternary_domain, g_task_simd);
        for (; i + 32 <= n; i += 32) {
            __m256i va = _mm256_loadu_si256((__m256i const*)(a_ptr + i));
            __m256i vr = simd_op(va);
            _mm256_storeu_si256((__m256i*)(r_ptr + i), vr);
        }
        TERNARY_PROFILE_TASK_END(g_ternary_domain);
    }

    // PATH 3: Scalar tail
    if (i < n) {
        TERNARY_PROFILE_TASK_BEGIN(g_ternary_domain, g_task_tail);
        for (; i < n; ++i) {
            r_ptr[i] = scalar_op(a_ptr[i]);
        }
        TERNARY_PROFILE_TASK_END(g_ternary_domain);
    }

    return out;
}

// =============================================================================
// Operation Wrappers (replacing macro-generated code)
// =============================================================================

// --- Binary Operations ---
py::array_t<uint8_t> tadd_array(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return process_binary_array<uint8_t, SANITIZE>(A, B, tadd_simd<SANITIZE>, tadd);
}

py::array_t<uint8_t> tmul_array(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return process_binary_array<uint8_t, SANITIZE>(A, B, tmul_simd<SANITIZE>, tmul);
}

py::array_t<uint8_t> tmin_array(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return process_binary_array<uint8_t, SANITIZE>(A, B, tmin_simd<SANITIZE>, tmin);
}

py::array_t<uint8_t> tmax_array(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return process_binary_array<uint8_t, SANITIZE>(A, B, tmax_simd<SANITIZE>, tmax);
}

// --- Unary Operation ---
py::array_t<uint8_t> tnot_array(py::array_t<uint8_t> A) {
    return process_unary_array<uint8_t, SANITIZE>(A, tnot_simd<SANITIZE>, tnot);
}

// =============================================================================
// Fused Operations (Phase 4.1 - Validated)
// =============================================================================

py::array_t<uint8_t> fused_tnot_tadd_array(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return process_binary_array<uint8_t, SANITIZE>(A, B, fused_tnot_tadd_simd<SANITIZE>, fused_tnot_tadd_scalar);
}

py::array_t<uint8_t> fused_tnot_tmul_array(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return process_binary_array<uint8_t, SANITIZE>(A, B, fused_tnot_tmul_simd<SANITIZE>, fused_tnot_tmul_scalar);
}

py::array_t<uint8_t> fused_tnot_tmin_array(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return process_binary_array<uint8_t, SANITIZE>(A, B, fused_tnot_tmin_simd<SANITIZE>, fused_tnot_tmin_scalar);
}

py::array_t<uint8_t> fused_tnot_tmax_array(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return process_binary_array<uint8_t, SANITIZE>(A, B, fused_tnot_tmax_simd<SANITIZE>, fused_tnot_tmax_scalar);
}

// =============================================================================
// BRIDGE LAYER: Fused Int8 Operations (Zero Conversion Overhead)
// =============================================================================
//
// These functions accept int8 arrays directly (values -1, 0, +1) and return
// int8 arrays. The format conversion is fused with the kernel operation,
// eliminating ALL NumPy conversion overhead.
//
// PERFORMANCE IMPACT:
//   Before (naive): 97% time in Python/NumPy conversion, 3% in kernel
//   After (fused):  0% in conversion, 100% in kernel
//   Expected speedup: ~30x for typical workloads
//
// =============================================================================

// process_binary_array_int8 / process_unary_array_int8 removed 2026-08-18
// (CLAUDE.md gap #6, Cluster A): they were near-byte-identical to
// process_binary_array<T>/process_unary_array<T> above, differing only by
// element type. The wrappers below now instantiate the unified template
// with T=int8_t instead.

// --- Int8 Binary Operation Wrappers ---
py::array_t<int8_t> tadd_int8_array(py::array_t<int8_t> A, py::array_t<int8_t> B) {
    return process_binary_array<int8_t, SANITIZE>(A, B, tadd_int8_simd<SANITIZE>, tadd_int8_scalar);
}

py::array_t<int8_t> tmul_int8_array(py::array_t<int8_t> A, py::array_t<int8_t> B) {
    return process_binary_array<int8_t, SANITIZE>(A, B, tmul_int8_simd<SANITIZE>, tmul_int8_scalar);
}

py::array_t<int8_t> tmin_int8_array(py::array_t<int8_t> A, py::array_t<int8_t> B) {
    return process_binary_array<int8_t, SANITIZE>(A, B, tmin_int8_simd<SANITIZE>, tmin_int8_scalar);
}

py::array_t<int8_t> tmax_int8_array(py::array_t<int8_t> A, py::array_t<int8_t> B) {
    return process_binary_array<int8_t, SANITIZE>(A, B, tmax_int8_simd<SANITIZE>, tmax_int8_scalar);
}

// --- Int8 Unary Operation Wrapper ---
py::array_t<int8_t> tnot_int8_array(py::array_t<int8_t> A) {
    return process_unary_array<int8_t, SANITIZE>(A, tnot_int8_simd<SANITIZE>, tnot_int8_scalar);
}

// --- Int8 Fused Operations (Level 3: Bridge + Fusion) ---
py::array_t<int8_t> fused_tnot_tadd_int8_array(py::array_t<int8_t> A, py::array_t<int8_t> B) {
    return process_binary_array<int8_t, SANITIZE>(A, B, fused_tnot_tadd_int8_simd<SANITIZE>,
        [](int8_t a, int8_t b) { return tnot_int8_scalar(tadd_int8_scalar(a, b)); });
}

py::array_t<int8_t> fused_tnot_tmul_int8_array(py::array_t<int8_t> A, py::array_t<int8_t> B) {
    return process_binary_array<int8_t, SANITIZE>(A, B, fused_tnot_tmul_int8_simd<SANITIZE>,
        [](int8_t a, int8_t b) { return tnot_int8_scalar(tmul_int8_scalar(a, b)); });
}

py::array_t<int8_t> fused_tnot_tmin_int8_array(py::array_t<int8_t> A, py::array_t<int8_t> B) {
    return process_binary_array<int8_t, SANITIZE>(A, B, fused_tnot_tmin_int8_simd<SANITIZE>,
        [](int8_t a, int8_t b) { return tnot_int8_scalar(tmin_int8_scalar(a, b)); });
}

py::array_t<int8_t> fused_tnot_tmax_int8_array(py::array_t<int8_t> A, py::array_t<int8_t> B) {
    return process_binary_array<int8_t, SANITIZE>(A, B, fused_tnot_tmax_int8_simd<SANITIZE>,
        [](int8_t a, int8_t b) { return tnot_int8_scalar(tmax_int8_scalar(a, b)); });
}

PYBIND11_MODULE(ternary_simd_engine, m) {
    // FIX: Check for AVX2 support at module initialization
    // This module requires AVX2 (Intel Haswell 2013+, AMD Excavator 2015+)
    if (!has_avx2()) {
        throw std::runtime_error(
            "Ternary SIMD Engine requires AVX2 instruction set support.\n"
            "Your CPU does not support AVX2. Detected ISA: " +
            std::string(simd_level_name(detect_best_simd())) + "\n"
            "Minimum requirement: Intel Haswell (2013+) or AMD Excavator (2015+)"
        );
    }

    // Export basic operation functions
    m.def("tadd", &tadd_array);
    m.def("tmul", &tmul_array);
    m.def("tmin", &tmin_array);
    m.def("tmax", &tmax_array);
    m.def("tnot", &tnot_array);

    // Export fused operations (Phase 4.1 - Validated)
    m.def("fused_tnot_tadd", &fused_tnot_tadd_array, py::arg("a"), py::arg("b"),
          "Fused operation: tnot(tadd(a, b)) - Validated 1.62-1.95× speedup");
    m.def("fused_tnot_tmul", &fused_tnot_tmul_array, py::arg("a"), py::arg("b"),
          "Fused operation: tnot(tmul(a, b)) - Validated 1.53-1.86× speedup");
    m.def("fused_tnot_tmin", &fused_tnot_tmin_array, py::arg("a"), py::arg("b"),
          "Fused operation: tnot(tmin(a, b)) - Validated 1.61-11.26× speedup");
    m.def("fused_tnot_tmax", &fused_tnot_tmax_array, py::arg("a"), py::arg("b"),
          "Fused operation: tnot(tmax(a, b)) - Validated 1.65-9.50× speedup");

    // Export CPU capability detection functions
    m.def("has_avx2", &has_avx2, "Check if CPU supports AVX2");
    m.def("detect_simd_level", []() { return static_cast<int>(detect_best_simd()); },
          "Detect best available SIMD level (0=None, 1=AVX2, 2=AVX512BW, 3=NEON, 4=SVE)");
    m.def("simd_level_string", []() { return std::string(simd_level_name(detect_best_simd())); },
          "Get human-readable SIMD level name");

    // =========================================================================
    // BRIDGE LAYER: Int8 Operations (Zero Conversion Overhead)
    // =========================================================================
    //
    // These functions accept int8 arrays directly (values -1, 0, +1) and return
    // int8 arrays. Format conversion is fused with the kernel, eliminating
    // all NumPy conversion overhead (~97% of previous pipeline time).
    //
    // Usage:
    //   import numpy as np
    //   a = np.array([-1, 0, 1, -1, 0], dtype=np.int8)
    //   b = np.array([1, 0, -1, 1, 0], dtype=np.int8)
    //   result = te.tadd_int8(a, b)  # Direct int8 → int8
    //
    // Performance: ~30x speedup over naive pipeline (uint8 + conversion)
    //

    // Basic int8 operations
    m.def("tadd_int8", &tadd_int8_array, py::arg("a"), py::arg("b"),
          "Ternary addition on int8 arrays. Direct int8 input/output, zero conversion overhead.");
    m.def("tmul_int8", &tmul_int8_array, py::arg("a"), py::arg("b"),
          "Ternary multiplication on int8 arrays. Direct int8 input/output, zero conversion overhead.");
    m.def("tmin_int8", &tmin_int8_array, py::arg("a"), py::arg("b"),
          "Ternary minimum on int8 arrays. Direct int8 input/output, zero conversion overhead.");
    m.def("tmax_int8", &tmax_int8_array, py::arg("a"), py::arg("b"),
          "Ternary maximum on int8 arrays. Direct int8 input/output, zero conversion overhead.");
    m.def("tnot_int8", &tnot_int8_array, py::arg("a"),
          "Ternary negation on int8 arrays. Direct int8 input/output, zero conversion overhead.");

    // Fused int8 operations (Level 3: Bridge + Fusion)
    m.def("fused_tnot_tadd_int8", &fused_tnot_tadd_int8_array, py::arg("a"), py::arg("b"),
          "Fused tnot(tadd(a, b)) on int8 arrays. Maximum optimization: bridge + operation fusion.");
    m.def("fused_tnot_tmul_int8", &fused_tnot_tmul_int8_array, py::arg("a"), py::arg("b"),
          "Fused tnot(tmul(a, b)) on int8 arrays. Maximum optimization: bridge + operation fusion.");
    m.def("fused_tnot_tmin_int8", &fused_tnot_tmin_int8_array, py::arg("a"), py::arg("b"),
          "Fused tnot(tmin(a, b)) on int8 arrays. Maximum optimization: bridge + operation fusion.");
    m.def("fused_tnot_tmax_int8", &fused_tnot_tmax_int8_array, py::arg("a"), py::arg("b"),
          "Fused tnot(tmax(a, b)) on int8 arrays. Maximum optimization: bridge + operation fusion.");

    // =========================================================================
    // Perfetto profiler control
    // =========================================================================
    //
    // Only meaningful when this module is built with -DTERNARY_ENABLE_PERFETTO
    // (build/build.py's --enable-perfetto flag; default builds don't define
    // it -- zero overhead, matching this project's stated no-op-by-default
    // design). When built with it, the TERNARY_PROFILE_TASK_BEGIN/END call
    // sites already wired into the tadd/tmul/etc. array functions above
    // (OpenMP_Parallel / Serial_SIMD / Scalar_Tail) emit real Perfetto
    // trace events; these two functions let Python code start/stop the
    // session that captures them, since Perfetto's in-process backend
    // needs the traced program itself to do that (unlike VTune/NVTX,
    // which attach externally). See
    // src/core/profiling/ternary_profiler_perfetto.cc and
    // reports/2026-08-25/PERFETTO_PROFILER_INTEGRATION.md.
#ifdef TERNARY_ENABLE_PERFETTO
    m.def("perfetto_start", [](const std::string& trace_path) {
        return ternary_profiler_perfetto_start(trace_path.c_str());
    }, py::arg("trace_path"),
       "Start a Perfetto tracing session, writing events to trace_path as "
       "they occur. Call before any tadd/tmul/tmin/tmax/tnot calls you "
       "want captured. Returns True on success.");
    m.def("perfetto_stop", &ternary_profiler_perfetto_stop,
          "Stop the active Perfetto tracing session and flush/close the "
          "trace file. No-op if no session is active.");
    m.attr("has_perfetto") = true;
#else
    m.attr("has_perfetto") = false;
#endif
}
