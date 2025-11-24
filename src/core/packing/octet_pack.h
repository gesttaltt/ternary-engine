/**
 * octet_pack.h - Octet Encoding (2 trits → 4 bits → 1 byte)
 *
 * Copyright 2025 Ternary Engine Contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Octet encoding provides byte-aligned storage for trit pairs:
 * - 2 trits require 2 bytes in 2-bit format (1 byte per trit)
 * - 2 trits require 1 byte in Octet format (4 bits used, 4 bits wasted)
 *
 * Encoding scheme:
 * - Each trit uses 2 bits: 00 = -1, 01 = 0, 10 = +1, 11 = invalid
 * - Pack format: [0000][t1:t1][t0:t0] (bits 3-0 used, bits 7-4 zero)
 *
 * Valid states: 9 out of 16 possible (3^2 valid ternary combinations)
 *
 * Use cases:
 * - CPU↔GPU transfer format
 * - Disk I/O (byte-aligned)
 * - Network transport (no bit-level packing)
 * - DMA transfers (hardware-friendly alignment)
 * - Partial data packing (when N mod 3 != 0)
 */

#ifndef OCTET_PACK_H
#define OCTET_PACK_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

// Trit encoding (2 bits per trit, same as Sixtet)
#define OCTET_MINUS_ONE  0x00  // 00
#define OCTET_ZERO       0x01  // 01
#define OCTET_PLUS_ONE   0x02  // 10
#define OCTET_INVALID    0x03  // 11

// Bit positions for each trit in packed byte
#define OCTET_T0_SHIFT   0
#define OCTET_T1_SHIFT   2

// Masks
#define OCTET_TRIT_MASK  0x03  // 2-bit mask for single trit
#define OCTET_VALID_MASK 0x0F  // 4-bit mask for valid Octet

// ============================================================================
// Compile-Time Unpack LUT Generation
// ============================================================================

// Generate unpack LUT at compile time
// Maps packed nibble (0-15) to 2 trits {-1, 0, +1}
// Invalid encodings (containing 11) map to {0, 0}
static const int8_t OCTET_UNPACK_LUT[16][2] = {
    // Index = [0000][t1:t1][t0:t0]
    {-1, -1}, {0, -1}, {1, -1}, {0, 0},  // 0x0-0x3 (t1=00)
    {-1, 0}, {0, 0}, {1, 0}, {0, 0},     // 0x4-0x7 (t1=01)
    {-1, 1}, {0, 1}, {1, 1}, {0, 0},     // 0x8-0xB (t1=10)
    {0, 0}, {0, 0}, {0, 0}, {0, 0}       // 0xC-0xF (t1=11, invalid)
};

// Validation LUT: true if nibble encodes valid ternary pair
static const bool OCTET_VALID_LUT[16] = {
    true, true, true, false,    // 0x0-0x3
    true, true, true, false,    // 0x4-0x7
    true, true, true, false,    // 0x8-0xB
    false, false, false, false  // 0xC-0xF
};

// ============================================================================
// Core Pack/Unpack Functions
// ============================================================================

/**
 * Pack 2 trits into 4 bits (stored in lower nibble of uint8_t)
 *
 * @param t0 First trit {-1, 0, +1}
 * @param t1 Second trit {-1, 0, +1}
 * @return Packed byte with 4 bits used (bits 3-0), upper 4 bits zero
 *
 * Invalid inputs default to zero (01 encoding)
 */
static inline uint8_t octet_pack(int8_t t0, int8_t t1) {
    // Convert trits to 2-bit encoding (branchless)
    uint8_t b0 = (t0 == -1) ? OCTET_MINUS_ONE : ((t0 == 0) ? OCTET_ZERO : OCTET_PLUS_ONE);
    uint8_t b1 = (t1 == -1) ? OCTET_MINUS_ONE : ((t1 == 0) ? OCTET_ZERO : OCTET_PLUS_ONE);

    // Pack into lower nibble: [0000][t1:t1][t0:t0]
    return (b1 << OCTET_T1_SHIFT) | (b0 << OCTET_T0_SHIFT);
}

/**
 * Unpack 4 bits (lower nibble) into 2 trits using LUT
 *
 * @param packed Packed byte (only lower 4 bits used)
 * @param t0 Output: first trit
 * @param t1 Output: second trit
 *
 * Invalid encodings decode to {0, 0}
 */
static inline void octet_unpack(uint8_t packed, int8_t* t0, int8_t* t1) {
    uint8_t idx = packed & OCTET_VALID_MASK;  // Ensure only lower 4 bits
    *t0 = OCTET_UNPACK_LUT[idx][0];
    *t1 = OCTET_UNPACK_LUT[idx][1];
}

/**
 * Check if packed nibble is valid ternary encoding
 *
 * @param packed Packed byte
 * @return true if valid, false if contains invalid encoding (11 bits)
 */
static inline bool octet_is_valid(uint8_t packed) {
    uint8_t idx = packed & OCTET_VALID_MASK;
    return OCTET_VALID_LUT[idx];
}

// ============================================================================
// Array Batch Operations
// ============================================================================

/**
 * Pack array of 2-bit trits (1 byte per trit) into Octet format
 *
 * @param trits Input array of trits in 2-bit format (1 byte per trit)
 * @param n_trits Number of trits in input (must be multiple of 2)
 * @param packed Output array (size = n_trits / 2)
 * @return Number of bytes written to packed array, or -1 on error
 *
 * Note: Input must have n_trits % 2 == 0. Use padding if needed.
 */
static inline int octet_pack_array(const int8_t* trits, size_t n_trits, uint8_t* packed) {
    if (n_trits % 2 != 0) {
        return -1;  // Invalid input size
    }

    size_t n_octets = n_trits / 2;
    for (size_t i = 0; i < n_octets; i++) {
        packed[i] = octet_pack(trits[i * 2], trits[i * 2 + 1]);
    }

    return (int)n_octets;
}

/**
 * Unpack Octet array into 2-bit trit format (1 byte per trit)
 *
 * @param packed Input array of packed Octets
 * @param n_octets Number of Octets in input
 * @param trits Output array (size = n_octets * 2)
 * @return Number of trits written, or -1 on error
 */
static inline int octet_unpack_array(const uint8_t* packed, size_t n_octets, int8_t* trits) {
    for (size_t i = 0; i < n_octets; i++) {
        int8_t t0, t1;
        octet_unpack(packed[i], &t0, &t1);
        trits[i * 2] = t0;
        trits[i * 2 + 1] = t1;
    }

    return (int)(n_octets * 2);
}

/**
 * Validate entire Octet array
 *
 * @param packed Input array of packed Octets
 * @param n_octets Number of Octets
 * @return Number of valid Octets, or -1 if any invalid encoding found
 */
static inline int octet_validate_array(const uint8_t* packed, size_t n_octets) {
    for (size_t i = 0; i < n_octets; i++) {
        if (!octet_is_valid(packed[i])) {
            return -1;  // Found invalid encoding
        }
    }
    return (int)n_octets;
}

// ============================================================================
// AVX2 Batch Operations (Optional, for future optimization)
// ============================================================================

#ifdef __AVX2__

/**
 * AVX2-accelerated Octet pack (future optimization)
 *
 * Processes 64 trits (32 Octets) at once using AVX2
 * Currently placeholder for future implementation
 */
static inline void octet_pack_avx2(const int8_t* trits, size_t n_trits, uint8_t* packed) {
    // TODO: Implement AVX2 version in Phase 6
    // For now, fall back to scalar
    octet_pack_array(trits, n_trits, packed);
}

/**
 * AVX2-accelerated Octet unpack (future optimization)
 *
 * Processes 32 Octets at once using AVX2 shuffles
 * Currently placeholder for future implementation
 */
static inline void octet_unpack_avx2(const uint8_t* packed, size_t n_octets, int8_t* trits) {
    // TODO: Implement AVX2 version in Phase 6
    // For now, fall back to scalar
    octet_unpack_array(packed, n_octets, trits);
}

#endif  // __AVX2__

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Calculate number of Octets needed for n trits (with padding)
 *
 * @param n_trits Number of trits
 * @return Number of Octets needed (rounded up to multiple of 2)
 */
static inline size_t octet_required_bytes(size_t n_trits) {
    return (n_trits + 1) / 2;  // Round up division by 2
}

/**
 * Calculate compression ratio
 *
 * @param n_trits Number of trits
 * @return Compression ratio (original bytes / packed bytes)
 */
static inline float octet_compression_ratio(size_t n_trits) {
    return (float)n_trits / (float)octet_required_bytes(n_trits);
}

/**
 * Convert between Sixtet and Octet formats
 *
 * Useful for handling N mod 3 != 0 cases:
 * - Use Sixtet for bulk (multiples of 3)
 * - Use Octet for remainder (1-2 trits)
 */

/**
 * Get compression efficiency comparison
 *
 * @param n_trits Number of trits to encode
 * @param use_sixtet Output: true if Sixtet is more efficient
 * @return Bytes saved by choosing better format
 */
static inline int octet_vs_sixtet_efficiency(size_t n_trits, bool* use_sixtet) {
    size_t sixtet_bytes = (n_trits + 2) / 3;  // Round up
    size_t octet_bytes = (n_trits + 1) / 2;   // Round up

    *use_sixtet = (sixtet_bytes <= octet_bytes);
    return (int)(octet_bytes - sixtet_bytes);
}

#ifdef __cplusplus
}
#endif

#endif  // OCTET_PACK_H
