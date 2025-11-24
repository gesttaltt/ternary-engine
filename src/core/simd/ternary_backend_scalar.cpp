/**
 * ternary_backend_scalar.c - Scalar Reference Backend Implementation
 *
 * Copyright 2025 Ternary Engine Contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Portable scalar reference implementation for all platforms.
 * This is the baseline for correctness and portability.
 *
 * Features:
 * - Pure C implementation
 * - No SIMD dependencies
 * - 100% portable
 * - Reference for all other backends
 *
 * Performance: ~100-500 Mops/s (depending on CPU and compiler)
 */

#include "ternary_backend_interface.h"
#include "../algebra/ternary_algebra.h"
#include <stdbool.h>

// ============================================================================
// Scalar Operations
// ============================================================================

static void scalar_tnot(uint8_t* dst, const uint8_t* src, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = tnot(src[i]);
    }
}

static void scalar_tadd(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = tadd(a[i], b[i]);
    }
}

static void scalar_tmul(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = tmul(a[i], b[i]);
    }
}

static void scalar_tmax(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = tmax(a[i], b[i]);
    }
}

static void scalar_tmin(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = tmin(a[i], b[i]);
    }
}

// ============================================================================
// Availability Check
// ============================================================================

static bool scalar_is_available(void) {
    // Scalar backend is always available
    return true;
}

// ============================================================================
// Backend Definition
// ============================================================================

static const TernaryBackend g_scalar_backend = {
    .info = TERNARY_BACKEND_INFO(
        "Scalar",
        "Portable scalar reference implementation",
        TERNARY_VERSION(1, 2, 0),
        TERNARY_CAP_SCALAR,
        1,  // Process 1 trit at a time
        scalar_is_available
    ),

    // Core operations
    .tnot = scalar_tnot,
    .tadd = scalar_tadd,
    .tmul = scalar_tmul,
    .tmax = scalar_tmax,
    .tmin = scalar_tmin,

    // Fusion operations (not implemented for scalar)
    .tadd_tmul = NULL,
    .tmul_tadd = NULL,

    // Advanced operations (not implemented)
    .tand = NULL,
    .tor = NULL
};

// ============================================================================
// Registration
// ============================================================================

void ternary_register_scalar_backend(void) {
    ternary_backend_register(&g_scalar_backend);
}
