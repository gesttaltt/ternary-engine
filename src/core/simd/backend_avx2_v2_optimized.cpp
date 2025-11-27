/**
 * backend_avx2_v2_optimized.cpp - AVX2 optimized backend (35-45 Gops/s)
 *
 * Copyright 2025 Ternary Engine Contributors
 * Licensed under the Apache License, Version 2.0
 *
 * AVX2 backend featuring v1.2.0 optimizations:
 * - Canonical indexing (✅ ENABLED - eliminates shift/OR arithmetic)
 * - Dual-shuffle XOR (future enhancement)
 * - LUT-256B support (future)
 *
 * Performance target: 35-45 Gops/s stable (30-40% over v1)
 *
 * v1.3.0 Changes:
 * - Switched from traditional indexing idx=(a<<2)|b to canonical idx=(a*3)+b
 * - Using canonical LUTs from ternary_canonical_lut.h
 * - Expected 12-18% performance improvement over traditional indexing
 */

#include "backend_plugin_api.h"

#ifdef __AVX2__

#include "opt_canonical_index.h"
#include "opt_dual_shuffle_xor.h"
#include "fused_binary_unary_ops.h"            // Phase 4.1 fusion operations
#include "cpu_simd_capability.h"
#include "../algebra/ternary_algebra.h"        // Scalar operations for fallback
#include "../algebra/ternary_canonical_lut.h"  // Canonical LUTs
#include "../config/optimization_config.h"      // OMP_THRESHOLD, STREAM_THRESHOLD, PREFETCH_DIST
#include <immintrin.h>
#include <xmmintrin.h>                         // _mm_prefetch
#include <omp.h>                               // OpenMP parallelization
#include <stdbool.h>
#include <string.h>

// ============================================================================
// Pre-broadcasted LUTs (Canonical Indexing)
// ============================================================================

static __m256i g_tadd_canonical_lut_256;
static __m256i g_tmul_canonical_lut_256;
static __m256i g_tmax_canonical_lut_256;
static __m256i g_tmin_canonical_lut_256;
static __m256i g_tnot_canonical_lut_256;

static bool g_canonical_luts_initialized = false;

static void init_canonical_luts(void) {
    if (g_canonical_luts_initialized) return;

    // Canonical 16-byte LUTs from ternary_canonical_lut.h
    // These LUTs are organized for idx=(a*3)+b indexing
    g_tadd_canonical_lut_256 = broadcast_lut_16(TADD_CANONICAL_LUT.data());
    g_tmul_canonical_lut_256 = broadcast_lut_16(TMUL_CANONICAL_LUT.data());
    g_tmax_canonical_lut_256 = broadcast_lut_16(TMAX_CANONICAL_LUT.data());
    g_tmin_canonical_lut_256 = broadcast_lut_16(TMIN_CANONICAL_LUT.data());
    g_tnot_canonical_lut_256 = broadcast_lut_16(TNOT_CANONICAL_LUT.data());

    // Initialize dual-shuffle LUTs (future enhancement)
    // init_dual_shuffle_luts();  // TODO: Enable for additional performance

    g_canonical_luts_initialized = true;
}

// ============================================================================
// Binary Operation Using Canonical Indexing
// ============================================================================

/**
 * Generic binary operation with canonical indexing
 *
 * Uses: idx = (a*3)+b via canonical_index_avx2()
 *
 * This eliminates the shift/OR arithmetic bottleneck:
 * - Old: idx = (a<<2)|b  (dependent chain: shift → OR → shuffle)
 * - New: idx = canonical_index_avx2(a,b)  (parallel shuffles → add)
 *
 * Expected performance gain: 12-18% improvement
 */
static inline __m256i binary_op_canonical(__m256i a, __m256i b, __m256i lut) {
    // Mask to 2-bit trit values
    __m256i mask = _mm256_set1_epi8(0x03);
    __m256i a_masked = _mm256_and_si256(a, mask);
    __m256i b_masked = _mm256_and_si256(b, mask);

    // Canonical indexing: idx = (a*3)+b
    // This function from ternary_canonical_index.h performs:
    // - dual-shuffle to get components: contrib_a = shuffle(CANON_A, a)
    //                                   contrib_b = shuffle(CANON_B, b)
    // - combine with ADD: indices = contrib_a + contrib_b
    __m256i indices = canonical_index_avx2(a_masked, b_masked);

    // Lookup with canonical index
    __m256i result = _mm256_shuffle_epi8(lut, indices);

    return result;
}

// ============================================================================
// Unary Operations
// ============================================================================

static void avx2_v2_tnot(uint8_t* dst, const uint8_t* src, size_t n) {
    init_canonical_luts();

    __m256i mask = _mm256_set1_epi8(0x03);
    size_t i = 0;

    // PATH 1: Large arrays → OpenMP parallel with prefetch and streaming
    if (n >= OMP_THRESHOLD) {
        ssize_t n_simd_blocks = static_cast<ssize_t>((n / 32) * 32);
        bool use_streaming = (n >= STREAM_THRESHOLD) && is_aligned_32(dst);

        #pragma omp parallel for schedule(guided, 4)
        for (ssize_t idx = 0; idx < n_simd_blocks; idx += 32) {
            // Prefetch for unary operation (only one input)
            if (idx + PREFETCH_DIST < n_simd_blocks) {
                _mm_prefetch((const char*)(src + idx + PREFETCH_DIST), _MM_HINT_T0);
            }

            __m256i a = _mm256_loadu_si256((const __m256i*)(src + idx));
            __m256i indices = _mm256_and_si256(a, mask);
            __m256i result = _mm256_shuffle_epi8(g_tnot_canonical_lut_256, indices);

            if (use_streaming) {
                _mm256_stream_si256((__m256i*)(dst + idx), result);
            } else {
                _mm256_storeu_si256((__m256i*)(dst + idx), result);
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
            __m256i a = _mm256_loadu_si256((const __m256i*)(src + i));
            __m256i indices = _mm256_and_si256(a, mask);
            __m256i result = _mm256_shuffle_epi8(g_tnot_canonical_lut_256, indices);
            _mm256_storeu_si256((__m256i*)(dst + i), result);
        }
    }

    // PATH 3: Scalar tail
    for (; i < n; i++) {
        dst[i] = tnot(src[i]);
    }
}

// ============================================================================
// Binary Operations with Canonical Indexing
// ============================================================================

static void avx2_v2_tadd(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    init_canonical_luts();

    size_t i = 0;

    // PATH 1: Large arrays → OpenMP parallel with prefetch and streaming
    // Migrated from bindings_core_ops.cpp:process_binary_array
    if (n >= OMP_THRESHOLD) {
        ssize_t n_simd_blocks = static_cast<ssize_t>((n / 32) * 32);
        bool use_streaming = (n >= STREAM_THRESHOLD) && is_aligned_32(dst);

        #pragma omp parallel for schedule(guided, 4)
        for (ssize_t idx = 0; idx < n_simd_blocks; idx += 32) {
            // OPT-PREFETCH: Hide memory latency
            if (idx + PREFETCH_DIST < n_simd_blocks) {
                _mm_prefetch((const char*)(a + idx + PREFETCH_DIST), _MM_HINT_T0);
                _mm_prefetch((const char*)(b + idx + PREFETCH_DIST), _MM_HINT_T0);
            }

            // SIMD operation with canonical indexing
            __m256i va = _mm256_loadu_si256((const __m256i*)(a + idx));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(b + idx));
            __m256i result = binary_op_canonical(va, vb, g_tadd_canonical_lut_256);

            // OPT-STREAM: Reduce cache pollution on large arrays
            if (use_streaming) {
                _mm256_stream_si256((__m256i*)(dst + idx), result);
            } else {
                _mm256_storeu_si256((__m256i*)(dst + idx), result);
            }
        }

        // Memory fence after streaming stores
        if (use_streaming) {
            _mm_sfence();
        }

        i = n_simd_blocks;
    }
    // PATH 2: Small arrays → Serial SIMD
    else {
        for (; i + 32 <= n; i += 32) {
            __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
            __m256i result = binary_op_canonical(va, vb, g_tadd_canonical_lut_256);
            _mm256_storeu_si256((__m256i*)(dst + i), result);
        }
    }

    // PATH 3: Scalar tail
    for (; i < n; i++) {
        dst[i] = tadd(a[i], b[i]);
    }
}

static void avx2_v2_tmul(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    init_canonical_luts();

    size_t i = 0;

    // PATH 1: Large arrays → OpenMP parallel with prefetch and streaming
    if (n >= OMP_THRESHOLD) {
        ssize_t n_simd_blocks = static_cast<ssize_t>((n / 32) * 32);
        bool use_streaming = (n >= STREAM_THRESHOLD) && is_aligned_32(dst);

        #pragma omp parallel for schedule(guided, 4)
        for (ssize_t idx = 0; idx < n_simd_blocks; idx += 32) {
            if (idx + PREFETCH_DIST < n_simd_blocks) {
                _mm_prefetch((const char*)(a + idx + PREFETCH_DIST), _MM_HINT_T0);
                _mm_prefetch((const char*)(b + idx + PREFETCH_DIST), _MM_HINT_T0);
            }

            __m256i va = _mm256_loadu_si256((const __m256i*)(a + idx));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(b + idx));
            __m256i result = binary_op_canonical(va, vb, g_tmul_canonical_lut_256);

            if (use_streaming) {
                _mm256_stream_si256((__m256i*)(dst + idx), result);
            } else {
                _mm256_storeu_si256((__m256i*)(dst + idx), result);
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
            __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
            __m256i result = binary_op_canonical(va, vb, g_tmul_canonical_lut_256);
            _mm256_storeu_si256((__m256i*)(dst + i), result);
        }
    }

    // PATH 3: Scalar tail
    for (; i < n; i++) {
        dst[i] = tmul(a[i], b[i]);
    }
}

static void avx2_v2_tmax(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    init_canonical_luts();

    size_t i = 0;

    // PATH 1: Large arrays → OpenMP parallel with prefetch and streaming
    if (n >= OMP_THRESHOLD) {
        ssize_t n_simd_blocks = static_cast<ssize_t>((n / 32) * 32);
        bool use_streaming = (n >= STREAM_THRESHOLD) && is_aligned_32(dst);

        #pragma omp parallel for schedule(guided, 4)
        for (ssize_t idx = 0; idx < n_simd_blocks; idx += 32) {
            if (idx + PREFETCH_DIST < n_simd_blocks) {
                _mm_prefetch((const char*)(a + idx + PREFETCH_DIST), _MM_HINT_T0);
                _mm_prefetch((const char*)(b + idx + PREFETCH_DIST), _MM_HINT_T0);
            }

            __m256i va = _mm256_loadu_si256((const __m256i*)(a + idx));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(b + idx));
            __m256i result = binary_op_canonical(va, vb, g_tmax_canonical_lut_256);

            if (use_streaming) {
                _mm256_stream_si256((__m256i*)(dst + idx), result);
            } else {
                _mm256_storeu_si256((__m256i*)(dst + idx), result);
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
            __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
            __m256i result = binary_op_canonical(va, vb, g_tmax_canonical_lut_256);
            _mm256_storeu_si256((__m256i*)(dst + i), result);
        }
    }

    // PATH 3: Scalar tail
    for (; i < n; i++) {
        dst[i] = tmax(a[i], b[i]);
    }
}

static void avx2_v2_tmin(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    init_canonical_luts();

    size_t i = 0;

    // PATH 1: Large arrays → OpenMP parallel with prefetch and streaming
    if (n >= OMP_THRESHOLD) {
        ssize_t n_simd_blocks = static_cast<ssize_t>((n / 32) * 32);
        bool use_streaming = (n >= STREAM_THRESHOLD) && is_aligned_32(dst);

        #pragma omp parallel for schedule(guided, 4)
        for (ssize_t idx = 0; idx < n_simd_blocks; idx += 32) {
            if (idx + PREFETCH_DIST < n_simd_blocks) {
                _mm_prefetch((const char*)(a + idx + PREFETCH_DIST), _MM_HINT_T0);
                _mm_prefetch((const char*)(b + idx + PREFETCH_DIST), _MM_HINT_T0);
            }

            __m256i va = _mm256_loadu_si256((const __m256i*)(a + idx));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(b + idx));
            __m256i result = binary_op_canonical(va, vb, g_tmin_canonical_lut_256);

            if (use_streaming) {
                _mm256_stream_si256((__m256i*)(dst + idx), result);
            } else {
                _mm256_storeu_si256((__m256i*)(dst + idx), result);
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
            __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
            __m256i result = binary_op_canonical(va, vb, g_tmin_canonical_lut_256);
            _mm256_storeu_si256((__m256i*)(dst + i), result);
        }
    }

    // PATH 3: Scalar tail
    for (; i < n; i++) {
        dst[i] = tmin(a[i], b[i]);
    }
}

// ============================================================================
// Dual-Shuffle XOR Operations (Experimental)
// ============================================================================

/**
 * Alternative implementation using dual-shuffle XOR
 *
 * This is commented out until dual-shuffle LUTs are properly validated
 * Expected: 1.5-1.7× speedup on Zen2/3/4 once enabled
 */
/*
static void avx2_v2_tadd_dual_shuffle(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    init_luts();

    size_t i = 0;

    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i result = tadd_dual_shuffle(va, vb);  // From ternary_dual_shuffle.h
        _mm256_storeu_si256((__m256i*)(dst + i), result);
    }

    for (; i < n; i++) {
        dst[i] = tadd_lut(a[i], b[i]);
    }
}
*/

// ============================================================================
// Fusion Operations (Phase 4.1 - Validated 2025-10-29)
// ============================================================================

/**
 * Fused tnot(tadd(a, b)) - eliminates intermediate array
 * Performance: 1.76× average speedup (validated)
 */
static void avx2_v2_fused_tnot_tadd(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    size_t i = 0;

    // PATH 1: Large arrays → OpenMP parallel with prefetch and streaming
    if (n >= OMP_THRESHOLD) {
        ssize_t n_simd_blocks = static_cast<ssize_t>((n / 32) * 32);
        bool use_streaming = (n >= STREAM_THRESHOLD) && is_aligned_32(dst);

        #pragma omp parallel for schedule(guided, 4)
        for (ssize_t idx = 0; idx < n_simd_blocks; idx += 32) {
            if (idx + PREFETCH_DIST < n_simd_blocks) {
                _mm_prefetch((const char*)(a + idx + PREFETCH_DIST), _MM_HINT_T0);
                _mm_prefetch((const char*)(b + idx + PREFETCH_DIST), _MM_HINT_T0);
            }

            __m256i va = _mm256_loadu_si256((const __m256i*)(a + idx));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(b + idx));
            __m256i result = fused_tnot_tadd_simd<false>(va, vb);

            if (use_streaming) {
                _mm256_stream_si256((__m256i*)(dst + idx), result);
            } else {
                _mm256_storeu_si256((__m256i*)(dst + idx), result);
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
            __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
            __m256i result = fused_tnot_tadd_simd<false>(va, vb);
            _mm256_storeu_si256((__m256i*)(dst + i), result);
        }
    }

    // PATH 3: Scalar tail
    for (; i < n; i++) {
        dst[i] = fused_tnot_tadd_scalar(a[i], b[i]);
    }
}

/**
 * Fused tnot(tmul(a, b)) - eliminates intermediate array
 * Performance: 1.71× average speedup (validated)
 */
static void avx2_v2_fused_tnot_tmul(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    size_t i = 0;

    // PATH 1: Large arrays → OpenMP parallel with prefetch and streaming
    if (n >= OMP_THRESHOLD) {
        ssize_t n_simd_blocks = static_cast<ssize_t>((n / 32) * 32);
        bool use_streaming = (n >= STREAM_THRESHOLD) && is_aligned_32(dst);

        #pragma omp parallel for schedule(guided, 4)
        for (ssize_t idx = 0; idx < n_simd_blocks; idx += 32) {
            if (idx + PREFETCH_DIST < n_simd_blocks) {
                _mm_prefetch((const char*)(a + idx + PREFETCH_DIST), _MM_HINT_T0);
                _mm_prefetch((const char*)(b + idx + PREFETCH_DIST), _MM_HINT_T0);
            }

            __m256i va = _mm256_loadu_si256((const __m256i*)(a + idx));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(b + idx));
            __m256i result = fused_tnot_tmul_simd<false>(va, vb);

            if (use_streaming) {
                _mm256_stream_si256((__m256i*)(dst + idx), result);
            } else {
                _mm256_storeu_si256((__m256i*)(dst + idx), result);
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
            __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
            __m256i result = fused_tnot_tmul_simd<false>(va, vb);
            _mm256_storeu_si256((__m256i*)(dst + i), result);
        }
    }

    // PATH 3: Scalar tail
    for (; i < n; i++) {
        dst[i] = fused_tnot_tmul_scalar(a[i], b[i]);
    }
}

/**
 * Fused tnot(tmin(a, b)) - eliminates intermediate array
 * Performance: 4.06× average speedup (validated)
 */
static void avx2_v2_fused_tnot_tmin(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    size_t i = 0;

    // PATH 1: Large arrays → OpenMP parallel with prefetch and streaming
    if (n >= OMP_THRESHOLD) {
        ssize_t n_simd_blocks = static_cast<ssize_t>((n / 32) * 32);
        bool use_streaming = (n >= STREAM_THRESHOLD) && is_aligned_32(dst);

        #pragma omp parallel for schedule(guided, 4)
        for (ssize_t idx = 0; idx < n_simd_blocks; idx += 32) {
            if (idx + PREFETCH_DIST < n_simd_blocks) {
                _mm_prefetch((const char*)(a + idx + PREFETCH_DIST), _MM_HINT_T0);
                _mm_prefetch((const char*)(b + idx + PREFETCH_DIST), _MM_HINT_T0);
            }

            __m256i va = _mm256_loadu_si256((const __m256i*)(a + idx));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(b + idx));
            __m256i result = fused_tnot_tmin_simd<false>(va, vb);

            if (use_streaming) {
                _mm256_stream_si256((__m256i*)(dst + idx), result);
            } else {
                _mm256_storeu_si256((__m256i*)(dst + idx), result);
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
            __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
            __m256i result = fused_tnot_tmin_simd<false>(va, vb);
            _mm256_storeu_si256((__m256i*)(dst + i), result);
        }
    }

    // PATH 3: Scalar tail
    for (; i < n; i++) {
        dst[i] = fused_tnot_tmin_scalar(a[i], b[i]);
    }
}

/**
 * Fused tnot(tmax(a, b)) - eliminates intermediate array
 * Performance: 3.68× average speedup (validated)
 */
static void avx2_v2_fused_tnot_tmax(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    size_t i = 0;

    // PATH 1: Large arrays → OpenMP parallel with prefetch and streaming
    if (n >= OMP_THRESHOLD) {
        ssize_t n_simd_blocks = static_cast<ssize_t>((n / 32) * 32);
        bool use_streaming = (n >= STREAM_THRESHOLD) && is_aligned_32(dst);

        #pragma omp parallel for schedule(guided, 4)
        for (ssize_t idx = 0; idx < n_simd_blocks; idx += 32) {
            if (idx + PREFETCH_DIST < n_simd_blocks) {
                _mm_prefetch((const char*)(a + idx + PREFETCH_DIST), _MM_HINT_T0);
                _mm_prefetch((const char*)(b + idx + PREFETCH_DIST), _MM_HINT_T0);
            }

            __m256i va = _mm256_loadu_si256((const __m256i*)(a + idx));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(b + idx));
            __m256i result = fused_tnot_tmax_simd<false>(va, vb);

            if (use_streaming) {
                _mm256_stream_si256((__m256i*)(dst + idx), result);
            } else {
                _mm256_storeu_si256((__m256i*)(dst + idx), result);
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
            __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
            __m256i result = fused_tnot_tmax_simd<false>(va, vb);
            _mm256_storeu_si256((__m256i*)(dst + i), result);
        }
    }

    // PATH 3: Scalar tail
    for (; i < n; i++) {
        dst[i] = fused_tnot_tmax_scalar(a[i], b[i]);
    }
}

// ============================================================================
// Availability Check
// ============================================================================

static bool avx2_v2_is_available(void) {
    return has_avx2();  // From ternary_cpu_detect.h
}

// ============================================================================
// Backend Definition
// ============================================================================

static const TernaryBackend g_avx2_v2_backend = {
    .info = TERNARY_BACKEND_INFO(
        "AVX2_v2",
        "AVX2 with v1.2.0 optimizations (canonical + fusion)",
        TERNARY_VERSION(1, 3, 0),
        TERNARY_CAP_SIMD_256 | TERNARY_CAP_CANONICAL | TERNARY_CAP_FUSION,
        32,  // Process 32 trits at a time
        avx2_v2_is_available
    ),

    // Core operations
    .tnot = avx2_v2_tnot,
    .tadd = avx2_v2_tadd,
    .tmul = avx2_v2_tmul,
    .tmax = avx2_v2_tmax,
    .tmin = avx2_v2_tmin,

    // Fusion operations (Phase 4.1 validated)
    .fused_tnot_tadd = avx2_v2_fused_tnot_tadd,
    .fused_tnot_tmul = avx2_v2_fused_tnot_tmul,
    .fused_tnot_tmin = avx2_v2_fused_tnot_tmin,
    .fused_tnot_tmax = avx2_v2_fused_tnot_tmax,

    // Advanced operations
    .tand = NULL,
    .tor = NULL
};

// ============================================================================
// Registration
// ============================================================================

void ternary_register_avx2_v2_backend(void) {
    ternary_backend_register(&g_avx2_v2_backend);
}

#endif  // __AVX2__
