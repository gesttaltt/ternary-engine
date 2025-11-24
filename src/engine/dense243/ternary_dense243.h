// ternary_dense243.h — T5-Dense243 high-density ternary packing (5 trits/byte)
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
// DESIGN OVERVIEW
// =============================================================================
//
// T5-Dense243 encoding packs 5 balanced trits into 1 byte using base-243
// representation, achieving 95.3% density (243/256 states).
//
// ENCODING: b = Σ(tᵢ + 1) × 3ⁱ for i ∈ [0,4]
// DECODING: tᵢ = ((b / 3ⁱ) mod 3) - 1
//
// MEMORY DENSITY:
//   Current (2-bit): 4 trits/byte = 25% density (16/256 states)
//   T5-Dense243:     5 trits/byte = 95.3% density (243/256 states)
//   Improvement:     +25% capacity, -20% memory bandwidth
//
// TARGET WORKLOADS:
//   - Memory-bound operations (1M+ element arrays)
//   - Large-scale storage (files, databases)
//   - Network transmission (bandwidth-limited)
//
// IMPLEMENTATION STATUS:
//   Phase 1: Scalar proof-of-concept (this file)
//   Phase 2: SIMD extraction kernels (future)
//   Phase 3: Full SIMD pipeline (future)
//
// =============================================================================

#ifndef TERNARY_DENSE243_H
#define TERNARY_DENSE243_H

#include <stdint.h>
#include <array>
#include "core/algebra/ternary_lut_gen.h"  // For int_to_trit_constexpr, trit_to_int_constexpr

// =============================================================================
// Compile-time Helpers
// =============================================================================

// Constexpr integer power function (for 3^n calculations)
constexpr uint32_t ipow(uint32_t base, uint32_t exp) {
    uint32_t result = 1;
    for (uint32_t i = 0; i < exp; ++i) {
        result *= base;
    }
    return result;
}

// Compile-time validation
static_assert(ipow(3, 0) == 1, "ipow(3, 0) should be 1");
static_assert(ipow(3, 1) == 3, "ipow(3, 1) should be 3");
static_assert(ipow(3, 2) == 9, "ipow(3, 2) should be 9");
static_assert(ipow(3, 3) == 27, "ipow(3, 3) should be 27");
static_assert(ipow(3, 4) == 81, "ipow(3, 4) should be 81");
static_assert(ipow(3, 5) == 243, "ipow(3, 5) should be 243");

// =============================================================================
// Extraction LUT Generation (243/256 entries, padded to 256 for SIMD)
// =============================================================================

// Generate extraction LUT for trit position i (0-4)
// Maps: packed byte [0-242] → 2-bit trit encoding (0b00=-1, 0b01=0, 0b10=+1)
// Invalid bytes [243-255] → 0b01 (neutral zero trit)
template <size_t Position>
constexpr std::array<uint8_t, 256> make_dense243_extract_lut() {
    static_assert(Position < 5, "Position must be 0-4 for 5-trit packing");
    std::array<uint8_t, 256> lut{};

    constexpr uint32_t divisor = ipow(3, Position);  // 3^Position

    for (size_t packed_byte = 0; packed_byte < 256; ++packed_byte) {
        if (packed_byte < 243) {
            // Valid T5-Dense243 value: decode trit at position
            int trit_offset = (packed_byte / divisor) % 3;  // ∈ {0, 1, 2}
            int trit_value = trit_offset - 1;               // ∈ {-1, 0, +1}
            lut[packed_byte] = int_to_trit_constexpr(trit_value);  // → 2-bit
        } else {
            // Invalid value (243-255): map to neutral zero trit
            lut[packed_byte] = 0b01;  // 0 trit (neutral element)
        }
    }

    return lut;
}

// Pregenerate all 5 extraction LUTs at compile time
constexpr auto DENSE243_EXTRACT_T0_LUT = make_dense243_extract_lut<0>();
constexpr auto DENSE243_EXTRACT_T1_LUT = make_dense243_extract_lut<1>();
constexpr auto DENSE243_EXTRACT_T2_LUT = make_dense243_extract_lut<2>();
constexpr auto DENSE243_EXTRACT_T3_LUT = make_dense243_extract_lut<3>();
constexpr auto DENSE243_EXTRACT_T4_LUT = make_dense243_extract_lut<4>();

// Compile-time LUT size validation
static_assert(DENSE243_EXTRACT_T0_LUT.size() == 256, "Extraction LUTs must be 256 bytes for AVX2 compatibility");
static_assert(DENSE243_EXTRACT_T1_LUT.size() == 256, "Extraction LUTs must be 256 bytes for AVX2 compatibility");
static_assert(DENSE243_EXTRACT_T2_LUT.size() == 256, "Extraction LUTs must be 256 bytes for AVX2 compatibility");
static_assert(DENSE243_EXTRACT_T3_LUT.size() == 256, "Extraction LUTs must be 256 bytes for AVX2 compatibility");
static_assert(DENSE243_EXTRACT_T4_LUT.size() == 256, "Extraction LUTs must be 256 bytes for AVX2 compatibility");

// =============================================================================
// Scalar Encoding/Decoding Functions
// =============================================================================

// --- Encode: Pack 5 trits into 1 byte ---
// Input: 5 trits in 2-bit encoding (0b00=-1, 0b01=0, 0b10=+1)
// Output: Dense243 byte [0-242]
static inline uint8_t dense243_pack(uint8_t t0, uint8_t t1, uint8_t t2, uint8_t t3, uint8_t t4) {
    // Convert 2-bit encoding to offset {0, 1, 2}
    int o0 = trit_to_int_constexpr(t0) + 1;  // -1→0, 0→1, +1→2
    int o1 = trit_to_int_constexpr(t1) + 1;
    int o2 = trit_to_int_constexpr(t2) + 1;
    int o3 = trit_to_int_constexpr(t3) + 1;
    int o4 = trit_to_int_constexpr(t4) + 1;

    // Encode as base-243: Σ(oᵢ × 3ⁱ)
    return static_cast<uint8_t>(o0 + o1*3 + o2*9 + o3*27 + o4*81);
}

// --- Decode: Unpack 1 byte into 5 trits ---
// Input: Dense243 byte [0-242] (values 243-255 treated as 0s)
// Output: 5 trits in 2-bit encoding via output parameters

struct Dense243Unpacked {
    uint8_t t0, t1, t2, t3, t4;
};

static inline Dense243Unpacked dense243_unpack(uint8_t packed_byte) {
    Dense243Unpacked result;

    // Use precomputed extraction LUTs for efficiency
    result.t0 = DENSE243_EXTRACT_T0_LUT[packed_byte];
    result.t1 = DENSE243_EXTRACT_T1_LUT[packed_byte];
    result.t2 = DENSE243_EXTRACT_T2_LUT[packed_byte];
    result.t3 = DENSE243_EXTRACT_T3_LUT[packed_byte];
    result.t4 = DENSE243_EXTRACT_T4_LUT[packed_byte];

    return result;
}

// --- Extract single trit from packed byte ---
// Position ∈ [0, 4], returns 2-bit trit encoding
static inline uint8_t dense243_extract_trit(uint8_t packed_byte, size_t position) {
    switch (position) {
        case 0: return DENSE243_EXTRACT_T0_LUT[packed_byte];
        case 1: return DENSE243_EXTRACT_T1_LUT[packed_byte];
        case 2: return DENSE243_EXTRACT_T2_LUT[packed_byte];
        case 3: return DENSE243_EXTRACT_T3_LUT[packed_byte];
        case 4: return DENSE243_EXTRACT_T4_LUT[packed_byte];
        default: return 0b01;  // Invalid position → neutral zero
    }
}

// =============================================================================
// Array Size Conversions
// =============================================================================

// Calculate number of Dense243 bytes needed for N trits
constexpr size_t dense243_bytes_for_trits(size_t num_trits) {
    return (num_trits + 4) / 5;  // Ceiling division: ⌈N/5⌉
}

// Calculate number of trits stored in N Dense243 bytes
constexpr size_t dense243_trits_in_bytes(size_t num_bytes) {
    return num_bytes * 5;
}

// =============================================================================
// Validation and Diagnostics
// =============================================================================

// Check if byte is valid Dense243 encoding (0-242)
static inline bool dense243_is_valid(uint8_t packed_byte) {
    return packed_byte < 243;
}

// Validate entire Dense243 array
// Returns: Number of invalid bytes found (0 = all valid)
static inline size_t dense243_validate_array(const uint8_t* data, size_t num_bytes) {
    size_t invalid_count = 0;
    for (size_t i = 0; i < num_bytes; ++i) {
        if (!dense243_is_valid(data[i])) {
            invalid_count++;
        }
    }
    return invalid_count;
}

// =============================================================================
// Compile-time Tests (Proof of Correctness)
// =============================================================================

// Test encoding/decoding round-trip
namespace {
    constexpr bool test_dense243_roundtrip() {
        // Test case: (+1, 0, -1, +1, 0)
        uint8_t t0 = 0b10;  // +1
        uint8_t t1 = 0b01;  //  0
        uint8_t t2 = 0b00;  // -1
        uint8_t t3 = 0b10;  // +1
        uint8_t t4 = 0b01;  //  0

        // Encode
        int o0 = trit_to_int_constexpr(t0) + 1;  // +1 → 2
        int o1 = trit_to_int_constexpr(t1) + 1;  //  0 → 1
        int o2 = trit_to_int_constexpr(t2) + 1;  // -1 → 0
        int o3 = trit_to_int_constexpr(t3) + 1;  // +1 → 2
        int o4 = trit_to_int_constexpr(t4) + 1;  //  0 → 1

        uint8_t packed = o0 + o1*3 + o2*9 + o3*27 + o4*81;
        // = 2 + 1*3 + 0*9 + 2*27 + 1*81
        // = 2 + 3 + 0 + 54 + 81 = 140

        // Decode using LUTs
        uint8_t r0 = DENSE243_EXTRACT_T0_LUT[packed];
        uint8_t r1 = DENSE243_EXTRACT_T1_LUT[packed];
        uint8_t r2 = DENSE243_EXTRACT_T2_LUT[packed];
        uint8_t r3 = DENSE243_EXTRACT_T3_LUT[packed];
        uint8_t r4 = DENSE243_EXTRACT_T4_LUT[packed];

        // Verify round-trip
        return (r0 == t0) && (r1 == t1) && (r2 == t2) && (r3 == t3) && (r4 == t4);
    }

    static_assert(test_dense243_roundtrip(), "Dense243 round-trip encoding/decoding failed");
}

#endif // TERNARY_DENSE243_H
