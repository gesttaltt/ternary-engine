/**
 * backend_plugin_api.h - Pluggable backend interface for ternary operations
 *
 * Copyright 2025 Ternary Engine Contributors
 * Licensed under the Apache License, Version 2.0
 *
 * This defines the abstract interface for ternary operation backends.
 * Multiple implementations can coexist:
 * - Scalar reference (portable, baseline)
 * - AVX2 v1 (current implementation)
 * - AVX2 v2 (with v1.2.0 optimizations)
 * - Future: AVX-512, ARM NEON/SVE, RISC-V Vector, etc.
 *
 * Architecture principles:
 * - Backends operate on canonical 2-bit internal format only
 * - Encodings (Sixtet/Octet) are separate I/O layers
 * - CPU detection and dispatch happen at runtime
 * - Semantics are defined by scalar reference backend
 *
 * This is the key to portability: adding a new platform requires
 * only implementing this interface, not rewriting the entire engine.
 */

#ifndef BACKEND_PLUGIN_API_H
#define BACKEND_PLUGIN_API_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Backend Capabilities
// ============================================================================

/**
 * Backend capability flags
 *
 * Describes what features a backend supports
 */
typedef enum {
    TERNARY_CAP_SCALAR      = 0x0001,  // Scalar operations
    TERNARY_CAP_SIMD_128    = 0x0002,  // 128-bit SIMD (SSE/NEON)
    TERNARY_CAP_SIMD_256    = 0x0004,  // 256-bit SIMD (AVX2)
    TERNARY_CAP_SIMD_512    = 0x0008,  // 512-bit SIMD (AVX-512)
    TERNARY_CAP_OPENMP      = 0x0010,  // OpenMP multi-threading
    TERNARY_CAP_FUSION      = 0x0020,  // Operation fusion
    TERNARY_CAP_CANONICAL   = 0x0040,  // Canonical indexing
    TERNARY_CAP_DUAL_SHUFFLE = 0x0080, // Dual-shuffle XOR
    TERNARY_CAP_LUT_256B    = 0x0100   // 256-byte LUTs
} TernaryBackendCapabilities;

/**
 * Backend metadata
 *
 * Identifies a backend implementation
 */
typedef struct {
    const char* name;                    // Backend name (e.g., "AVX2_v2")
    const char* description;             // Short description
    uint32_t version;                    // Version number (e.g., 0x010200 for v1.2.0)
    uint32_t capabilities;               // Capability flags (OR of TernaryBackendCapabilities)
    size_t preferred_batch_size;         // Preferred batch size for this backend
    bool (*is_available)(void);          // Runtime availability check
} TernaryBackendInfo;

// ============================================================================
// Core Operation Function Types
// ============================================================================

/**
 * Unary operation signature
 *
 * @param dst Output array (2-bit encoding, 1 byte per trit)
 * @param src Input array (2-bit encoding, 1 byte per trit)
 * @param n Number of trits to process
 */
typedef void (*TernaryUnaryOp)(uint8_t* dst, const uint8_t* src, size_t n);

/**
 * Binary operation signature
 *
 * @param dst Output array (2-bit encoding, 1 byte per trit)
 * @param a First input array (2-bit encoding, 1 byte per trit)
 * @param b Second input array (2-bit encoding, 1 byte per trit)
 * @param n Number of trits to process
 */
typedef void (*TernaryBinaryOp)(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n);

// ============================================================================
// Backend Interface Structure
// ============================================================================

/**
 * Complete backend interface
 *
 * All backends must implement this interface
 * NULL pointers indicate unsupported operations
 */
typedef struct {
    // Backend metadata
    TernaryBackendInfo info;

    // Core unary operations
    TernaryUnaryOp tnot;  // Negation

    // Core binary operations
    TernaryBinaryOp tadd;  // Addition (saturating)
    TernaryBinaryOp tmul;  // Multiplication
    TernaryBinaryOp tmax;  // Maximum
    TernaryBinaryOp tmin;  // Minimum

    // Fusion operations (Phase 4.1 - validated 2025-10-29)
    // Pattern: unary(binary(a, b)) - eliminates intermediate array
    // Performance: 1.5-11× speedup depending on operation and array size
    TernaryBinaryOp fused_tnot_tadd;   // tnot(tadd(a, b)) - 1.76× avg
    TernaryBinaryOp fused_tnot_tmul;   // tnot(tmul(a, b)) - 1.71× avg
    TernaryBinaryOp fused_tnot_tmin;   // tnot(tmin(a, b)) - 4.06× avg
    TernaryBinaryOp fused_tnot_tmax;   // tnot(tmax(a, b)) - 3.68× avg

    // Advanced operations (optional)
    TernaryBinaryOp tand;  // Ternary AND (consensus)
    TernaryBinaryOp tor;   // Ternary OR

} TernaryBackend;

// ============================================================================
// Backend Registration and Discovery
// ============================================================================

/**
 * Maximum number of registered backends
 */
#define TERNARY_MAX_BACKENDS 16

/**
 * Register a backend
 *
 * @param backend Backend to register
 * @return true if registered successfully, false if table is full
 */
bool ternary_backend_register(const TernaryBackend* backend);

/**
 * Get number of registered backends
 */
size_t ternary_backend_count(void);

/**
 * Get backend by index
 *
 * @param index Backend index (0 to count-1)
 * @return Backend pointer, or NULL if index out of range
 */
const TernaryBackend* ternary_backend_get(size_t index);

/**
 * Find backend by name
 *
 * @param name Backend name
 * @return Backend pointer, or NULL if not found
 */
const TernaryBackend* ternary_backend_find(const char* name);

// ============================================================================
// Backend Selection and Dispatch
// ============================================================================

/**
 * Select best available backend
 *
 * Uses CPU detection to find the fastest backend available on this system
 * Priority: AVX2_v2 > AVX2_v1 > Scalar
 *
 * @return Best backend, or NULL if no backends available
 */
const TernaryBackend* ternary_backend_select_best(void);

/**
 * Set active backend
 *
 * Override automatic selection with a specific backend
 *
 * @param backend Backend to use, or NULL to re-enable automatic selection
 */
void ternary_backend_set_active(const TernaryBackend* backend);

/**
 * Get active backend
 *
 * @return Currently active backend, or NULL if none selected
 */
const TernaryBackend* ternary_backend_get_active(void);

// ============================================================================
// Dispatch Functions (High-Level API)
// ============================================================================

/**
 * Dispatch ternary operations through active backend
 *
 * These functions automatically use the best available backend
 * Users should call these instead of backend functions directly
 */

void ternary_dispatch_tnot(uint8_t* dst, const uint8_t* src, size_t n);
void ternary_dispatch_tadd(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n);
void ternary_dispatch_tmul(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n);
void ternary_dispatch_tmax(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n);
void ternary_dispatch_tmin(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n);

/**
 * Fusion operations dispatch (Phase 4.1)
 * These eliminate intermediate arrays for performance
 */
void ternary_dispatch_fused_tnot_tadd(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n);
void ternary_dispatch_fused_tnot_tmul(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n);
void ternary_dispatch_fused_tnot_tmin(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n);
void ternary_dispatch_fused_tnot_tmax(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n);

// ============================================================================
// Backend Initialization
// ============================================================================

/**
 * Initialize backend system
 *
 * Call once at program startup to:
 * - Detect CPU capabilities
 * - Register all available backends
 * - Select best backend
 *
 * @return true if successful, false on error
 */
bool ternary_backend_init(void);

/**
 * Shutdown backend system
 *
 * Cleanup resources (if any)
 */
void ternary_backend_shutdown(void);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Print all available backends (for debugging)
 */
void ternary_backend_print_all(void);

/**
 * Get backend name string for capabilities
 *
 * @param capabilities Capability flags
 * @param buffer Output buffer
 * @param buffer_size Size of output buffer
 */
void ternary_backend_capabilities_to_string(uint32_t capabilities, char* buffer, size_t buffer_size);

// ============================================================================
// Backend Implementation Helpers
// ============================================================================

/**
 * Helper macro to define backend info
 */
#define TERNARY_BACKEND_INFO(name_, desc_, version_, caps_, batch_size_, available_fn_) \
    { \
        .name = name_, \
        .description = desc_, \
        .version = version_, \
        .capabilities = caps_, \
        .preferred_batch_size = batch_size_, \
        .is_available = available_fn_ \
    }

/**
 * Helper macro to define version number
 */
#define TERNARY_VERSION(major, minor, patch) \
    (((major) << 16) | ((minor) << 8) | (patch))

// ============================================================================
// Example Usage
// ============================================================================

/*
// At program startup:
ternary_backend_init();  // Auto-detects and selects best backend

// Use dispatch functions (automatically uses best backend):
uint8_t a[1000], b[1000], result[1000];
ternary_dispatch_tadd(result, a, b, 1000);

// Or manually select a backend:
const TernaryBackend* avx2_backend = ternary_backend_find("AVX2_v2");
if (avx2_backend) {
    ternary_backend_set_active(avx2_backend);
}

// Query backend info:
const TernaryBackend* active = ternary_backend_get_active();
printf("Using backend: %s\n", active->info.name);
printf("Capabilities: 0x%08X\n", active->info.capabilities);

// At program shutdown:
ternary_backend_shutdown();
*/

#ifdef __cplusplus
}
#endif

#endif  // TERNARY_BACKEND_INTERFACE_H
