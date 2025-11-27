/**
 * backend_avx2_v1_baseline.cpp - AVX2 v1 baseline backend (28-35 Gops/s)
 *
 * Copyright 2025 Ternary Engine Contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Wrapper for existing AVX2 v1 implementation (current production kernel)
 * This provides the existing performance as a baseline for comparison.
 *
 * Performance: 28-35 Gops/s (validated v1.1.0)
 */

#include "backend_plugin_api.h"

#ifdef __AVX2__

#include "simd_avx2_32trit_ops.h"
#include "cpu_simd_capability.h"
#include "../algebra/ternary_algebra.h"
#include <immintrin.h>
#include <stdbool.h>

// ============================================================================
// AVX2 v1 Operations (Using Existing Kernels)
// ============================================================================

static void avx2_v1_tnot(uint8_t* dst, const uint8_t* src, size_t n) {
    size_t i = 0;

    // Process 32 trits at a time
    for (; i + 32 <= n; i += 32) {
        __m256i a = _mm256_loadu_si256((const __m256i*)(src + i));
        __m256i result = tnot_simd<true>(a);  // From simd_avx2_32trit_ops.h
        _mm256_storeu_si256((__m256i*)(dst + i), result);
    }

    // Handle remaining elements with scalar fallback
    for (; i < n; i++) {
        dst[i] = tnot(src[i]);
    }
}

static void avx2_v1_tadd(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    size_t i = 0;

    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i result = tadd_simd<true>(va, vb);  // From simd_avx2_32trit_ops.h
        _mm256_storeu_si256((__m256i*)(dst + i), result);
    }

    for (; i < n; i++) {
        dst[i] = tadd(a[i], b[i]);
    }
}

static void avx2_v1_tmul(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    size_t i = 0;

    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i result = tmul_simd<true>(va, vb);
        _mm256_storeu_si256((__m256i*)(dst + i), result);
    }

    for (; i < n; i++) {
        dst[i] = tmul(a[i], b[i]);
    }
}

static void avx2_v1_tmax(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    size_t i = 0;

    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i result = tmax_simd<true>(va, vb);
        _mm256_storeu_si256((__m256i*)(dst + i), result);
    }

    for (; i < n; i++) {
        dst[i] = tmax(a[i], b[i]);
    }
}

static void avx2_v1_tmin(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    size_t i = 0;

    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i result = tmin_simd<true>(va, vb);
        _mm256_storeu_si256((__m256i*)(dst + i), result);
    }

    for (; i < n; i++) {
        dst[i] = tmin(a[i], b[i]);
    }
}

// ============================================================================
// Availability Check
// ============================================================================

static bool avx2_v1_is_available(void) {
    return has_avx2();  // From ternary_cpu_detect.h
}

// ============================================================================
// Backend Definition
// ============================================================================

static const TernaryBackend g_avx2_v1_backend = {
    .info = TERNARY_BACKEND_INFO(
        "AVX2_v1",
        "AVX2 baseline implementation (v1.1.0)",
        TERNARY_VERSION(1, 1, 0),
        TERNARY_CAP_SIMD_256,
        32,  // Process 32 trits at a time
        avx2_v1_is_available
    ),

    // Core operations
    .tnot = avx2_v1_tnot,
    .tadd = avx2_v1_tadd,
    .tmul = avx2_v1_tmul,
    .tmax = avx2_v1_tmax,
    .tmin = avx2_v1_tmin,

    // Fusion operations (not supported in AVX2_v1)
    .fused_tnot_tadd = NULL,
    .fused_tnot_tmul = NULL,
    .fused_tnot_tmin = NULL,
    .fused_tnot_tmax = NULL,

    // Advanced operations
    .tand = NULL,
    .tor = NULL
};

// ============================================================================
// Registration
// ============================================================================

void ternary_register_avx2_v1_backend(void) {
    ternary_backend_register(&g_avx2_v1_backend);
}

#endif  // __AVX2__
