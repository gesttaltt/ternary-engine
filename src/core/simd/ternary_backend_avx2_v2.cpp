/**
 * ternary_backend_avx2_v2.cpp - AVX2 Backend with v1.2.0 Optimizations
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

#include "ternary_backend_interface.h"

#ifdef __AVX2__

#include "ternary_canonical_index.h"
#include "ternary_dual_shuffle.h"
#include "../simd/ternary_cpu_detect.h"
#include "../algebra/ternary_algebra.h"        // Scalar operations for fallback
#include "../algebra/ternary_canonical_lut.h"  // Canonical LUTs
#include <immintrin.h>
#include <stdbool.h>
#include <string.h>

// ============================================================================
// Helper: Load 16-byte LUT and broadcast to 256-bit
// ============================================================================

static inline __m256i broadcast_lut_16(const uint8_t* lut) {
    __m128i lut_128 = _mm_loadu_si128((const __m128i*)lut);
    return _mm256_broadcastsi128_si256(lut_128);
}

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

    // Process 32 trits at a time
    for (; i + 32 <= n; i += 32) {
        __m256i a = _mm256_loadu_si256((const __m256i*)(src + i));
        __m256i indices = _mm256_and_si256(a, mask);
        __m256i result = _mm256_shuffle_epi8(g_tnot_canonical_lut_256, indices);
        _mm256_storeu_si256((__m256i*)(dst + i), result);
    }

    // Handle remaining elements with scalar fallback
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

    // Process 32 trits at a time with canonical indexing
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i result = binary_op_canonical(va, vb, g_tadd_canonical_lut_256);
        _mm256_storeu_si256((__m256i*)(dst + i), result);
    }

    // Scalar fallback for remainder
    for (; i < n; i++) {
        dst[i] = tadd(a[i], b[i]);
    }
}

static void avx2_v2_tmul(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    init_canonical_luts();

    size_t i = 0;

    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i result = binary_op_canonical(va, vb, g_tmul_canonical_lut_256);
        _mm256_storeu_si256((__m256i*)(dst + i), result);
    }

    for (; i < n; i++) {
        dst[i] = tmul(a[i], b[i]);
    }
}

static void avx2_v2_tmax(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    init_canonical_luts();

    size_t i = 0;

    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i result = binary_op_canonical(va, vb, g_tmax_canonical_lut_256);
        _mm256_storeu_si256((__m256i*)(dst + i), result);
    }

    for (; i < n; i++) {
        dst[i] = tmax(a[i], b[i]);
    }
}

static void avx2_v2_tmin(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    init_canonical_luts();

    size_t i = 0;

    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i result = binary_op_canonical(va, vb, g_tmin_canonical_lut_256);
        _mm256_storeu_si256((__m256i*)(dst + i), result);
    }

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
        "AVX2 with v1.2.0 optimizations (canonical indexing)",
        TERNARY_VERSION(1, 2, 0),
        TERNARY_CAP_SIMD_256 | TERNARY_CAP_CANONICAL,
        32,  // Process 32 trits at a time
        avx2_v2_is_available
    ),

    // Core operations
    .tnot = avx2_v2_tnot,
    .tadd = avx2_v2_tadd,
    .tmul = avx2_v2_tmul,
    .tmax = avx2_v2_tmax,
    .tmin = avx2_v2_tmin,

    // Fusion operations (not yet implemented)
    .tadd_tmul = NULL,
    .tmul_tadd = NULL,

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
