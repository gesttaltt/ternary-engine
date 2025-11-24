/**
 * sixtet_pack.h - Sixtet Encoding (3 trits → 6 bits → 1 byte)
 *
 * Copyright 2025 Ternary Engine Contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Sixtet encoding provides 3× compression over 2-bit internal format:
 * - 3 trits require 3 bytes in 2-bit format (1 byte per trit)
 * - 3 trits require 1 byte in Sixtet format (6 bits used, 2 bits wasted)
 *
 * Encoding scheme:
 * - Each trit uses 2 bits: 00 = -1, 01 = 0, 10 = +1, 11 = invalid
 * - Pack format: [t2:t2][t1:t1][t0:t0] (bits 5-0, bits 7-6 unused)
 *
 * Valid states: 27 out of 64 possible (3^3 valid ternary combinations)
 *
 * Use cases:
 * - Input loading
 * - Output storage
 * - Cache-friendly strip-mining
 * - Stream compression
 */

#ifndef SIXTET_PACK_H
#define SIXTET_PACK_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>  // For memset

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

// Trit encoding (2 bits per trit)
#define SIXTET_MINUS_ONE  0x00  // 00
#define SIXTET_ZERO       0x01  // 01
#define SIXTET_PLUS_ONE   0x02  // 10
#define SIXTET_INVALID    0x03  // 11

// Bit positions for each trit in packed byte
#define SIXTET_T0_SHIFT   0
#define SIXTET_T1_SHIFT   2
#define SIXTET_T2_SHIFT   4

// Masks
#define SIXTET_TRIT_MASK  0x03  // 2-bit mask for single trit
#define SIXTET_VALID_MASK 0x3F  // 6-bit mask for valid Sixtet

// ============================================================================
// Compile-Time Unpack LUT Generation
// ============================================================================

// Generate unpack LUT at compile time
// Maps packed byte (0-63) to 3 trits {-1, 0, +1}
// Invalid encodings (containing 11) map to {0, 0, 0}
static const int8_t SIXTET_UNPACK_LUT[64][3] = {
    // Index = [t2:t2][t1:t1][t0:t0]
    {-1, -1, -1}, {0, -1, -1}, {1, -1, -1}, {0, 0, 0},  // 0x00-0x03
    {-1, 0, -1}, {0, 0, -1}, {1, 0, -1}, {0, 0, 0},     // 0x04-0x07
    {-1, 1, -1}, {0, 1, -1}, {1, 1, -1}, {0, 0, 0},     // 0x08-0x0B
    {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},          // 0x0C-0x0F (invalid t1=11)

    {-1, -1, 0}, {0, -1, 0}, {1, -1, 0}, {0, 0, 0},     // 0x10-0x13
    {-1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {0, 0, 0},        // 0x14-0x17
    {-1, 1, 0}, {0, 1, 0}, {1, 1, 0}, {0, 0, 0},        // 0x18-0x1B
    {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},          // 0x1C-0x1F

    {-1, -1, 1}, {0, -1, 1}, {1, -1, 1}, {0, 0, 0},     // 0x20-0x23
    {-1, 0, 1}, {0, 0, 1}, {1, 0, 1}, {0, 0, 0},        // 0x24-0x27
    {-1, 1, 1}, {0, 1, 1}, {1, 1, 1}, {0, 0, 0},        // 0x28-0x2B
    {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},          // 0x2C-0x2F

    {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},          // 0x30-0x33 (invalid t2=11)
    {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},          // 0x34-0x37
    {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},          // 0x38-0x3B
    {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}           // 0x3C-0x3F
};

// Validation LUT: true if byte encodes valid ternary triple
static const bool SIXTET_VALID_LUT[64] = {
    true, true, true, false,    // 0x00-0x03
    true, true, true, false,    // 0x04-0x07
    true, true, true, false,    // 0x08-0x0B
    false, false, false, false, // 0x0C-0x0F

    true, true, true, false,    // 0x10-0x13
    true, true, true, false,    // 0x14-0x17
    true, true, true, false,    // 0x18-0x1B
    false, false, false, false, // 0x1C-0x1F

    true, true, true, false,    // 0x20-0x23
    true, true, true, false,    // 0x24-0x27
    true, true, true, false,    // 0x28-0x2B
    false, false, false, false, // 0x2C-0x2F

    false, false, false, false, // 0x30-0x33
    false, false, false, false, // 0x34-0x37
    false, false, false, false, // 0x38-0x3B
    false, false, false, false  // 0x3C-0x3F
};

// ============================================================================
// Core Pack/Unpack Functions
// ============================================================================

/**
 * Pack 3 trits into 6 bits (stored in uint8_t)
 *
 * @param t0 First trit {-1, 0, +1}
 * @param t1 Second trit {-1, 0, +1}
 * @param t2 Third trit {-1, 0, +1}
 * @return Packed byte with 6 bits used (bits 5-0), upper 2 bits zero
 *
 * Invalid inputs default to zero (01 encoding)
 */
static inline uint8_t sixtet_pack(int8_t t0, int8_t t1, int8_t t2) {
    // Convert trits to 2-bit encoding (branchless)
    uint8_t b0 = (t0 == -1) ? SIXTET_MINUS_ONE : ((t0 == 0) ? SIXTET_ZERO : SIXTET_PLUS_ONE);
    uint8_t b1 = (t1 == -1) ? SIXTET_MINUS_ONE : ((t1 == 0) ? SIXTET_ZERO : SIXTET_PLUS_ONE);
    uint8_t b2 = (t2 == -1) ? SIXTET_MINUS_ONE : ((t2 == 0) ? SIXTET_ZERO : SIXTET_PLUS_ONE);

    // Pack into byte: [t2:t2][t1:t1][t0:t0]
    return (b2 << SIXTET_T2_SHIFT) | (b1 << SIXTET_T1_SHIFT) | (b0 << SIXTET_T0_SHIFT);
}

/**
 * Unpack 6 bits into 3 trits using LUT
 *
 * @param packed Packed byte (only lower 6 bits used)
 * @param t0 Output: first trit
 * @param t1 Output: second trit
 * @param t2 Output: third trit
 *
 * Invalid encodings decode to {0, 0, 0}
 */
static inline void sixtet_unpack(uint8_t packed, int8_t* t0, int8_t* t1, int8_t* t2) {
    uint8_t idx = packed & SIXTET_VALID_MASK;  // Ensure only lower 6 bits
    *t0 = SIXTET_UNPACK_LUT[idx][0];
    *t1 = SIXTET_UNPACK_LUT[idx][1];
    *t2 = SIXTET_UNPACK_LUT[idx][2];
}

/**
 * Check if packed byte is valid ternary encoding
 *
 * @param packed Packed byte
 * @return true if valid, false if contains invalid encoding (11 bits)
 */
static inline bool sixtet_is_valid(uint8_t packed) {
    uint8_t idx = packed & SIXTET_VALID_MASK;
    return SIXTET_VALID_LUT[idx];
}

// ============================================================================
// Array Batch Operations
// ============================================================================

/**
 * Pack array of 2-bit trits (1 byte per trit) into Sixtet format
 *
 * @param trits Input array of trits in 2-bit format (1 byte per trit)
 * @param n_trits Number of trits in input (must be multiple of 3)
 * @param packed Output array (size = n_trits / 3)
 * @return Number of bytes written to packed array, or -1 on error
 *
 * Note: Input must have n_trits % 3 == 0. Use padding if needed.
 */
static inline int sixtet_pack_array(const int8_t* trits, size_t n_trits, uint8_t* packed) {
    if (n_trits % 3 != 0) {
        return -1;  // Invalid input size
    }

    size_t n_sextets = n_trits / 3;
    for (size_t i = 0; i < n_sextets; i++) {
        packed[i] = sixtet_pack(trits[i * 3], trits[i * 3 + 1], trits[i * 3 + 2]);
    }

    return (int)n_sextets;
}

/**
 * Unpack Sixtet array into 2-bit trit format (1 byte per trit)
 *
 * @param packed Input array of packed Sextets
 * @param n_sextets Number of Sextets in input
 * @param trits Output array (size = n_sextets * 3)
 * @return Number of trits written, or -1 on error
 */
static inline int sixtet_unpack_array(const uint8_t* packed, size_t n_sextets, int8_t* trits) {
    for (size_t i = 0; i < n_sextets; i++) {
        int8_t t0, t1, t2;
        sixtet_unpack(packed[i], &t0, &t1, &t2);
        trits[i * 3] = t0;
        trits[i * 3 + 1] = t1;
        trits[i * 3 + 2] = t2;
    }

    return (int)(n_sextets * 3);
}

/**
 * Validate entire Sixtet array
 *
 * @param packed Input array of packed Sextets
 * @param n_sextets Number of Sextets
 * @return Number of valid Sextets, or -1 if any invalid encoding found
 */
static inline int sixtet_validate_array(const uint8_t* packed, size_t n_sextets) {
    for (size_t i = 0; i < n_sextets; i++) {
        if (!sixtet_is_valid(packed[i])) {
            return -1;  // Found invalid encoding
        }
    }
    return (int)n_sextets;
}

// ============================================================================
// AVX2 Batch Operations (Optional, for future optimization)
// ============================================================================

#ifdef __AVX2__

/**
 * AVX2-accelerated Sixtet pack (future optimization)
 *
 * Processes 32 trits (10.67 Sextets) at once using AVX2
 * Currently placeholder for future implementation
 */
static inline void sixtet_pack_avx2(const int8_t* trits, size_t n_trits, uint8_t* packed) {
    // TODO: Implement AVX2 version in Phase 6
    // For now, fall back to scalar
    sixtet_pack_array(trits, n_trits, packed);
}

/**
 * AVX2-accelerated Sixtet unpack (future optimization)
 *
 * Processes 32 Sextets at once using AVX2 shuffles
 * Currently placeholder for future implementation
 */
static inline void sixtet_unpack_avx2(const uint8_t* packed, size_t n_sextets, int8_t* trits) {
    // TODO: Implement AVX2 version in Phase 6
    // For now, fall back to scalar
    sixtet_unpack_array(packed, n_sextets, trits);
}

#endif  // __AVX2__

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Calculate number of Sextets needed for n trits (with padding)
 *
 * @param n_trits Number of trits
 * @return Number of Sextets needed (rounded up to multiple of 3)
 */
static inline size_t sixtet_required_bytes(size_t n_trits) {
    return (n_trits + 2) / 3;  // Round up division by 3
}

/**
 * Calculate compression ratio
 *
 * @param n_trits Number of trits
 * @return Compression ratio (original bytes / packed bytes)
 */
static inline float sixtet_compression_ratio(size_t n_trits) {
    return (float)n_trits / (float)sixtet_required_bytes(n_trits);
}

#ifdef __cplusplus
}
#endif

#endif  // SIXTET_PACK_H
