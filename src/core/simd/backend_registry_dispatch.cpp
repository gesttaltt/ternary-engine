/**
 * backend_registry_dispatch.cpp - Backend registration, selection, and dispatch
 *
 * Copyright 2025 Ternary Engine Contributors
 * Licensed under the Apache License, Version 2.0
 */

#include "backend_plugin_api.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>

// ============================================================================
// Global LUT Definitions (extern from headers)
// ============================================================================

// From opt_lut_256byte_expanded.h
alignas(256) uint8_t TADD_LUT_256B[4096] = {0};
alignas(256) uint8_t TMUL_LUT_256B[4096] = {0};
alignas(256) uint8_t TMAX_LUT_256B[4096] = {0};
alignas(256) uint8_t TMIN_LUT_256B[4096] = {0};
alignas(256) uint8_t TNOT_LUT_256B[256] = {0};

// From opt_dual_shuffle_xor.h
alignas(32) uint8_t TNOT_DUAL_A[32] = {0};
alignas(32) uint8_t TNOT_DUAL_B[32] = {0};
alignas(32) uint8_t TADD_DUAL_A[32] = {0};
alignas(32) uint8_t TADD_DUAL_B[32] = {0};
alignas(32) uint8_t TMUL_DUAL_A[32] = {0};
alignas(32) uint8_t TMUL_DUAL_B[32] = {0};

// ============================================================================
// Backend Registry
// ============================================================================

static const TernaryBackend* g_backends[TERNARY_MAX_BACKENDS] = {NULL};
static size_t g_backend_count = 0;
static const TernaryBackend* g_active_backend = NULL;
static bool g_initialized = false;

// ============================================================================
// Backend Registration
// ============================================================================

bool ternary_backend_register(const TernaryBackend* backend) {
    if (!backend) {
        return false;
    }

    if (g_backend_count >= TERNARY_MAX_BACKENDS) {
        fprintf(stderr, "Error: Maximum number of backends (%d) reached\n", TERNARY_MAX_BACKENDS);
        return false;
    }

    // Check if backend is available on this system
    if (backend->info.is_available && !backend->info.is_available()) {
        // Backend not available on this CPU
        return false;
    }

    g_backends[g_backend_count++] = backend;
    return true;
}

size_t ternary_backend_count(void) {
    return g_backend_count;
}

const TernaryBackend* ternary_backend_get(size_t index) {
    if (index >= g_backend_count) {
        return NULL;
    }
    return g_backends[index];
}

const TernaryBackend* ternary_backend_find(const char* name) {
    if (!name) {
        return NULL;
    }

    for (size_t i = 0; i < g_backend_count; i++) {
        if (g_backends[i] && strcmp(g_backends[i]->info.name, name) == 0) {
            return g_backends[i];
        }
    }

    return NULL;
}

// ============================================================================
// Backend Selection
// ============================================================================

const TernaryBackend* ternary_backend_select_best(void) {
    if (g_backend_count == 0) {
        return NULL;
    }

    const TernaryBackend* best = NULL;
    uint32_t best_score = 0;

    for (size_t i = 0; i < g_backend_count; i++) {
        const TernaryBackend* backend = g_backends[i];
        if (!backend) continue;

        // Score based on capabilities
        uint32_t score = 0;

        // Prioritize SIMD width
        if (backend->info.capabilities & TERNARY_CAP_SIMD_512) {
            score += 1000;
        } else if (backend->info.capabilities & TERNARY_CAP_SIMD_256) {
            score += 500;
        } else if (backend->info.capabilities & TERNARY_CAP_SIMD_128) {
            score += 250;
        }

        // Add points for optimizations
        if (backend->info.capabilities & TERNARY_CAP_DUAL_SHUFFLE) {
            score += 100;
        }
        if (backend->info.capabilities & TERNARY_CAP_CANONICAL) {
            score += 50;
        }
        if (backend->info.capabilities & TERNARY_CAP_LUT_256B) {
            score += 50;
        }
        if (backend->info.capabilities & TERNARY_CAP_FUSION) {
            score += 25;
        }
        if (backend->info.capabilities & TERNARY_CAP_OPENMP) {
            score += 25;
        }

        // Update best if this backend scores higher
        if (score > best_score) {
            best_score = score;
            best = backend;
        }
    }

    return best;
}

void ternary_backend_set_active(const TernaryBackend* backend) {
    g_active_backend = backend;
}

const TernaryBackend* ternary_backend_get_active(void) {
    return g_active_backend;
}

// ============================================================================
// Dispatch Functions
// ============================================================================

void ternary_dispatch_tnot(uint8_t* dst, const uint8_t* src, size_t n) {
    if (!g_active_backend || !g_active_backend->tnot) {
        fprintf(stderr, "Error: No active backend or tnot not implemented\n");
        return;
    }
    g_active_backend->tnot(dst, src, n);
}

void ternary_dispatch_tadd(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    if (!g_active_backend || !g_active_backend->tadd) {
        fprintf(stderr, "Error: No active backend or tadd not implemented\n");
        return;
    }
    g_active_backend->tadd(dst, a, b, n);
}

void ternary_dispatch_tmul(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    if (!g_active_backend || !g_active_backend->tmul) {
        fprintf(stderr, "Error: No active backend or tmul not implemented\n");
        return;
    }
    g_active_backend->tmul(dst, a, b, n);
}

void ternary_dispatch_tmax(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    if (!g_active_backend || !g_active_backend->tmax) {
        fprintf(stderr, "Error: No active backend or tmax not implemented\n");
        return;
    }
    g_active_backend->tmax(dst, a, b, n);
}

void ternary_dispatch_tmin(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    if (!g_active_backend || !g_active_backend->tmin) {
        fprintf(stderr, "Error: No active backend or tmin not implemented\n");
        return;
    }
    g_active_backend->tmin(dst, a, b, n);
}

// ============================================================================
// Fusion Operations Dispatch (Phase 4.1)
// ============================================================================

void ternary_dispatch_fused_tnot_tadd(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    if (!g_active_backend || !g_active_backend->fused_tnot_tadd) {
        fprintf(stderr, "Error: No active backend or fused_tnot_tadd not implemented\n");
        return;
    }
    g_active_backend->fused_tnot_tadd(dst, a, b, n);
}

void ternary_dispatch_fused_tnot_tmul(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    if (!g_active_backend || !g_active_backend->fused_tnot_tmul) {
        fprintf(stderr, "Error: No active backend or fused_tnot_tmul not implemented\n");
        return;
    }
    g_active_backend->fused_tnot_tmul(dst, a, b, n);
}

void ternary_dispatch_fused_tnot_tmin(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    if (!g_active_backend || !g_active_backend->fused_tnot_tmin) {
        fprintf(stderr, "Error: No active backend or fused_tnot_tmin not implemented\n");
        return;
    }
    g_active_backend->fused_tnot_tmin(dst, a, b, n);
}

void ternary_dispatch_fused_tnot_tmax(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    if (!g_active_backend || !g_active_backend->fused_tnot_tmax) {
        fprintf(stderr, "Error: No active backend or fused_tnot_tmax not implemented\n");
        return;
    }
    g_active_backend->fused_tnot_tmax(dst, a, b, n);
}

// ============================================================================
// Initialization
// ============================================================================

// Forward declarations for backend registration functions
extern void ternary_register_scalar_backend(void);
#ifdef __AVX2__
extern void ternary_register_avx2_v1_backend(void);
extern void ternary_register_avx2_v2_backend(void);
#endif

bool ternary_backend_init(void) {
    if (g_initialized) {
        return true;  // Already initialized
    }

    // Register all available backends
    ternary_register_scalar_backend();

#ifdef __AVX2__
    ternary_register_avx2_v1_backend();
    ternary_register_avx2_v2_backend();
#endif

    // Select best available backend
    const TernaryBackend* best = ternary_backend_select_best();
    if (!best) {
        fprintf(stderr, "Error: No backends available\n");
        return false;
    }

    ternary_backend_set_active(best);
    g_initialized = true;

    return true;
}

void ternary_backend_shutdown(void) {
    g_active_backend = NULL;
    g_backend_count = 0;
    g_initialized = false;

    // Clear backend registry
    for (size_t i = 0; i < TERNARY_MAX_BACKENDS; i++) {
        g_backends[i] = NULL;
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

void ternary_backend_print_all(void) {
    printf("===========================================\n");
    printf("Registered Ternary Backends (%zu)\n", g_backend_count);
    printf("===========================================\n\n");

    for (size_t i = 0; i < g_backend_count; i++) {
        const TernaryBackend* backend = g_backends[i];
        if (!backend) continue;

        printf("[%zu] %s\n", i, backend->info.name);
        printf("    Description: %s\n", backend->info.description);
        printf("    Version: %d.%d.%d\n",
               (backend->info.version >> 16) & 0xFF,
               (backend->info.version >> 8) & 0xFF,
               backend->info.version & 0xFF);
        printf("    Capabilities: 0x%08X\n", backend->info.capabilities);
        printf("    Batch size: %zu\n", backend->info.preferred_batch_size);
        printf("    Active: %s\n\n", (backend == g_active_backend) ? "YES" : "no");
    }
}

void ternary_backend_capabilities_to_string(uint32_t capabilities, char* buffer, size_t buffer_size) {
    if (!buffer || buffer_size == 0) return;

    buffer[0] = '\0';
    size_t offset = 0;

#define APPEND_CAP(cap, str) \
    if (capabilities & cap) { \
        if (offset > 0) { \
            offset += snprintf(buffer + offset, buffer_size - offset, ", "); \
        } \
        offset += snprintf(buffer + offset, buffer_size - offset, "%s", str); \
    }

    APPEND_CAP(TERNARY_CAP_SCALAR, "Scalar");
    APPEND_CAP(TERNARY_CAP_SIMD_128, "SIMD-128");
    APPEND_CAP(TERNARY_CAP_SIMD_256, "SIMD-256");
    APPEND_CAP(TERNARY_CAP_SIMD_512, "SIMD-512");
    APPEND_CAP(TERNARY_CAP_OPENMP, "OpenMP");
    APPEND_CAP(TERNARY_CAP_FUSION, "Fusion");
    APPEND_CAP(TERNARY_CAP_CANONICAL, "Canonical");
    APPEND_CAP(TERNARY_CAP_DUAL_SHUFFLE, "Dual-Shuffle");
    APPEND_CAP(TERNARY_CAP_LUT_256B, "LUT-256B");

#undef APPEND_CAP
}
