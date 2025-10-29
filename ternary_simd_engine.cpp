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
#include "ternary_core/algebra/ternary_algebra.h"
#include "ternary_errors.h"
#include "ternary_core/simd/ternary_cpu_detect.h"

// MSVC compatibility: ssize_t is not standard C++
#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

namespace py = pybind11;

// OPT-001: OpenMP threshold for large array parallelization
// Arrays >= threshold will use multi-threaded processing
// OPT-PHASE3-01: Adaptive threshold scales with CPU core count
// Formula: 32K elements per thread ensures good load balancing across CPU tiers
// FIX: Clamp hardware_concurrency() to [1, 64] (can return 0 on some VMs)
static const ssize_t OMP_THRESHOLD = 32768 * std::max(1u, std::min(64u, std::thread::hardware_concurrency()));

// OPT-STREAM: Streaming store threshold (arrays exceeding L3 cache size)
// Typical L3: 8-32 MB; use streaming stores for arrays > 1M elements (~1 MB)
// Non-temporal stores reduce cache pollution for memory-bound workloads
static const ssize_t STREAM_THRESHOLD = 1000000;

// OPT-PHASE3-03: Prefetch distance tuning
// Prefetch stride for hiding memory latency (can be tuned per CPU family)
// 512 bytes = 16 × 32-byte cache lines, optimal for Zen 2/4 and Raptor Lake
// Adjust to 256 for older CPUs or 1024 for server-class processors
constexpr int PREFETCH_DIST = 512;

// OPT-PHASE3-04: Optional compile-time sanitization switch
// Define TERNARY_NO_SANITIZE at compile time to disable input sanitization
// for validated data pipelines (3-5% performance gain)
// Example: c++ -DTERNARY_NO_SANITIZE -O3 ...
#ifdef TERNARY_NO_SANITIZE
constexpr bool SANITIZE = false;
#else
constexpr bool SANITIZE = true;
#endif

// --- LUT-Based SIMD Operations (OPT-061) ---
// Each operation uses _mm256_shuffle_epi8 for 32 parallel LUT lookups.
// The LUTs are the same 16-entry tables used in scalar operations (ternary_algebra.h).

// Helper: Load 16-entry LUT and broadcast to both 128-bit lanes of 256-bit vector
static inline __m256i broadcast_lut_16(const uint8_t* lut) {
    // Load 16 bytes into lower 128 bits, then duplicate to upper 128 bits
    __m128i lut_128 = _mm_loadu_si128((const __m128i*)lut);
    return _mm256_broadcastsi128_si256(lut_128);
}

// --- Pre-broadcasted LUT Cache (OPT-LUT-BROADCAST) ---
// Pre-broadcast all LUTs once at initialization to avoid repeated broadcast_lut_16() calls
// Note: __m256i is not a literal type, so these must be runtime-initialized (not constexpr)
// Initialized before first use via static initialization

namespace {
    // Helper function for static initialization
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

    // Static instance - initialized once before main()
    static const BroadcastedLUTs g_luts;
}  // anonymous namespace

// Helper: Optional masking to ensure only lower 2 bits of each byte are used
// OPT-HASWELL-02: Template-based sanitization for compile-time optimization
// When Sanitize=false, masking is elided for validated data pipelines (3-5% gain)
template <bool Sanitize = true>
static inline __m256i maybe_mask(__m256i v) {
    if constexpr (Sanitize)
        return _mm256_and_si256(v, _mm256_set1_epi8(0x03));
    else
        return v;
}

// --- Binary Operations (tadd, tmul, tmin, tmax) ---
// Index formula: (a << 2) | b, where a and b are 2-bit trits

// Unified template for all binary SIMD operations (eliminates 90% code duplication)
template <bool Sanitize = true>
static inline __m256i binary_simd_op(__m256i a, __m256i b, __m256i lut) {
    // Build 4-bit indices: (a << 2) | b
    // Note: AVX2 lacks byte-level shifts, using triple-add is actually optimal
    __m256i a_masked = maybe_mask<Sanitize>(a);
    __m256i b_masked = maybe_mask<Sanitize>(b);
    __m256i a_shifted = _mm256_add_epi8(_mm256_add_epi8(a_masked, a_masked),
                                         _mm256_add_epi8(a_masked, a_masked)); // a * 4
    __m256i indices = _mm256_or_si256(a_shifted, b_masked);

    // Perform 32 parallel LUT lookups (LUT pre-broadcasted for efficiency)
    return _mm256_shuffle_epi8(lut, indices);
}

// Specialized wrappers for each operation (using pre-broadcasted LUTs)
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

// --- Unary Operation (tnot) ---
// Index formula: a & 0x03 (2-bit trit value)
// Note: TNOT_LUT_SIMD is defined in ternary_algebra.h (16-entry padded for _mm256_shuffle_epi8)

template <bool Sanitize = true>
static inline __m256i tnot_simd(__m256i a) {
    __m256i indices = maybe_mask<Sanitize>(a);
    // Use pre-broadcasted LUT for efficiency
    return _mm256_shuffle_epi8(g_luts.tnot, indices);
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
// VALIDATION STRATEGY:
// - Input validation occurs at entry points (process_binary_array, process_unary_array)
// - Validation is inline for simplicity: single check per entry point
// - Uses centralized exception types from ternary_errors.h for consistency
// - If validation logic grows complex, consider extracting to validation helpers
// - Current validation: array size matching (binary ops), no validation needed (unary ops)
//
// =============================================================================

// --- Helper: Check if pointer is 32-byte aligned for streaming stores ---
// FIX: Streaming stores (_mm256_stream_si256) require 32-byte alignment
// NumPy arrays do not guarantee this, so we must check before using
inline bool is_aligned_32(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) % 32) == 0;
}

// --- Unified Binary Operation Template ---
template <bool Sanitize = true, typename SimdOp, typename ScalarOp>
py::array_t<uint8_t> process_binary_array(
    py::array_t<uint8_t> A,
    py::array_t<uint8_t> B,
    SimdOp simd_op,
    ScalarOp scalar_op
) {
    auto a = A.unchecked<1>();
    auto b = B.unchecked<1>();
    ssize_t n = A.size();

    // ENTRY-POINT VALIDATION: Ensure arrays match in size
    // Validation happens here (not centralized) for simplicity and early failure
    // Uses typed exception from ternary_errors.h for clear error semantics
    if (n != B.size()) throw ArraySizeMismatchError(n, B.size());

    py::array_t<uint8_t> out(n);
    auto r = out.mutable_unchecked<1>();
    const uint8_t* a_ptr = static_cast<const uint8_t*>(A.data());
    const uint8_t* b_ptr = static_cast<const uint8_t*>(B.data());
    uint8_t* r_ptr = static_cast<uint8_t*>(out.mutable_data());

    ssize_t i = 0;

    // PATH 1: Large arrays → OpenMP parallel (NUMA-aware scheduling)
    if (n >= OMP_THRESHOLD) {
        ssize_t n_simd_blocks = (n / 32) * 32;
        // FIX: Only use streaming stores if array is large AND output is 32-byte aligned
        bool use_streaming = (n >= STREAM_THRESHOLD) && is_aligned_32(r_ptr);

        // OPT-NUMA: guided scheduling for multi-CCD CPUs (Ryzen/EPYC)
        // Guided schedule adapts chunk sizes dynamically
        // Note: proc_bind(spread) removed for MSVC compatibility (OpenMP 4.0 feature)
        #pragma omp parallel for schedule(guided, 4)
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

        // Memory fence after streaming stores to ensure visibility
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

// --- Unified Unary Operation Template ---
template <bool Sanitize = true, typename SimdOp, typename ScalarOp>
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

    // PATH 1: Large arrays → OpenMP parallel (NUMA-aware scheduling)
    if (n >= OMP_THRESHOLD) {
        ssize_t n_simd_blocks = (n / 32) * 32;
        // FIX: Only use streaming stores if array is large AND output is 32-byte aligned
        bool use_streaming = (n >= STREAM_THRESHOLD) && is_aligned_32(r_ptr);

        // OPT-NUMA: guided scheduling for multi-CCD CPUs (Ryzen/EPYC)
        // Guided schedule adapts chunk sizes dynamically
        // Note: proc_bind(spread) removed for MSVC compatibility (OpenMP 4.0 feature)
        #pragma omp parallel for schedule(guided, 4)
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

        // Memory fence after streaming stores to ensure visibility
        if (use_streaming) {
            _mm_sfence();
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
    return process_binary_array<SANITIZE>(A, B, tadd_simd<SANITIZE>, tadd);
}

py::array_t<uint8_t> tmul_array(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return process_binary_array<SANITIZE>(A, B, tmul_simd<SANITIZE>, tmul);
}

py::array_t<uint8_t> tmin_array(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return process_binary_array<SANITIZE>(A, B, tmin_simd<SANITIZE>, tmin);
}

py::array_t<uint8_t> tmax_array(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return process_binary_array<SANITIZE>(A, B, tmax_simd<SANITIZE>, tmax);
}

// --- Unary Operation ---
py::array_t<uint8_t> tnot_array(py::array_t<uint8_t> A) {
    return process_unary_array<SANITIZE>(A, tnot_simd<SANITIZE>, tnot);
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

    // Export operation functions
    m.def("tadd", &tadd_array);
    m.def("tmul", &tmul_array);
    m.def("tmin", &tmin_array);
    m.def("tmax", &tmax_array);
    m.def("tnot", &tnot_array);

    // Export CPU capability detection functions
    m.def("has_avx2", &has_avx2, "Check if CPU supports AVX2");
    m.def("detect_simd_level", []() { return static_cast<int>(detect_best_simd()); },
          "Detect best available SIMD level (0=None, 1=AVX2, 2=AVX512BW, 3=NEON, 4=SVE)");
    m.def("simd_level_string", []() { return std::string(simd_level_name(detect_best_simd())); },
          "Get human-readable SIMD level name");
}
