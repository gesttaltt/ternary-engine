// ternary_triadsextet.h — TriadSextet interface layer (3 trits/6-bit unit)
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
// TriadSextet is a lightweight arithmetic interface layer that encodes
// 3 balanced trits into a 6-bit unit (sextet), providing a clean binary↔ternary
// bridge for external systems, debuggers, and arithmetic co-processors.
//
// ENCODING: s = Σ(tᵢ + 1) × 3ⁱ for i ∈ [0,2]
// DECODING: tᵢ = ((s / 3ⁱ) mod 3) - 1
//
// DENSITY:
//   TriadSextet:  3 trits/6-bit = 42% density (27/64 states)
//   Current 2-bit: 4 trits/byte = 25% density (16/256 states)
//   Dense243:     5 trits/byte = 95.3% density (243/256 states)
//
// DESIGN PHILOSOPHY:
//   - Algebraic elegance over density (clean 3^n structure)
//   - Zero-cost interface layer (reinterpretation functions)
//   - Multipurpose interpretation (FFI, debugging, arithmetic hardware)
//
// USE CASES:
//   - External API boundaries (C/Rust/Zig/Python FFI)
//   - Debugger extensions (GDB/LLDB pretty-printers)
//   - Network protocols (human-readable packet inspection)
//   - Arithmetic co-processors (FPGA ternary ALUs)
//   - Educational tools (simplified ternary representation)
//
// STORAGE STRATEGIES:
//   1. Virtual (zero-cost): Only interpretation functions, never materialized
//   2. Byte-aligned: 1 sextet = 1 byte (37.5% density, simple)
//   3. Packed 4:3: 4 sextets in 3 bytes (50% density, complex)
//
// IMPLEMENTATION STATUS:
//   Phase 1: Core LUTs and functions (this file)
//   Phase 2: C API layer (future)
//   Phase 3: Language bindings (future)
//
// =============================================================================

#ifndef TERNARY_TRIADSEXTET_H
#define TERNARY_TRIADSEXTET_H

#include <stdint.h>
#include <array>
#include "core/algebra/ternary_lut_gen.h"  // For int_to_trit_constexpr, trit_to_int_constexpr

// =============================================================================
// Type Definitions
// =============================================================================

// TriadSextet: 3 trits encoded in 6 bits (stored in byte, upper 2 bits unused)
typedef uint8_t triadsextet_t;

// Mask for extracting 6-bit sextet from byte
constexpr uint8_t SEXTET_MASK = 0x3F;  // 0b00111111

// Maximum valid sextet value (27 states: 0-26)
constexpr uint8_t SEXTET_MAX_VALID = 26;

// =============================================================================
// Compile-time Helpers
// =============================================================================

// Note: ipow is already defined in ternary_dense243.h if included
// Define it here for standalone use
#ifndef TERNARY_DENSE243_H
constexpr uint32_t ipow(uint32_t base, uint32_t exp) {
    uint32_t result = 1;
    for (uint32_t i = 0; i < exp; ++i) {
        result *= base;
    }
    return result;
}
#endif

// Compile-time validation
static_assert(ipow(3, 0) == 1, "ipow(3, 0) should be 1");
static_assert(ipow(3, 1) == 3, "ipow(3, 1) should be 3");
static_assert(ipow(3, 2) == 9, "ipow(3, 2) should be 9");
static_assert(ipow(3, 3) == 27, "ipow(3, 3) should be 27");

// =============================================================================
// Extraction LUT Generation (27/64 entries, padded to 64 for simplicity)
// =============================================================================

// Generate extraction LUT for trit position i (0-2)
// Maps: sextet [0-26] → 2-bit trit encoding (0b00=-1, 0b01=0, 0b10=+1)
// Invalid sextets [27-63] → 0b01 (neutral zero trit)
template <size_t Position>
constexpr std::array<uint8_t, 64> make_triadsextet_extract_lut() {
    static_assert(Position < 3, "Position must be 0-2 for 3-trit packing");
    std::array<uint8_t, 64> lut{};

    constexpr uint32_t divisor = ipow(3, Position);  // 3^Position

    for (size_t sextet = 0; sextet < 64; ++sextet) {
        if (sextet <= SEXTET_MAX_VALID) {
            // Valid TriadSextet value: decode trit at position
            int trit_offset = (sextet / divisor) % 3;  // ∈ {0, 1, 2}
            int trit_value = trit_offset - 1;           // ∈ {-1, 0, +1}
            lut[sextet] = int_to_trit_constexpr(trit_value);  // → 2-bit
        } else {
            // Invalid value (27-63): map to neutral zero trit
            lut[sextet] = 0b01;  // 0 trit (neutral element)
        }
    }

    return lut;
}

// Pregenerate all 3 extraction LUTs at compile time
constexpr auto TRIADSEXTET_EXTRACT_T0_LUT = make_triadsextet_extract_lut<0>();
constexpr auto TRIADSEXTET_EXTRACT_T1_LUT = make_triadsextet_extract_lut<1>();
constexpr auto TRIADSEXTET_EXTRACT_T2_LUT = make_triadsextet_extract_lut<2>();

// Compile-time LUT size validation
static_assert(TRIADSEXTET_EXTRACT_T0_LUT.size() == 64, "Extraction LUTs must be 64 bytes");
static_assert(TRIADSEXTET_EXTRACT_T1_LUT.size() == 64, "Extraction LUTs must be 64 bytes");
static_assert(TRIADSEXTET_EXTRACT_T2_LUT.size() == 64, "Extraction LUTs must be 64 bytes");

// =============================================================================
// Core Encoding/Decoding Functions
// =============================================================================

// --- Encode: Pack 3 trits into 1 sextet ---
// Input: 3 trits in 2-bit encoding (0b00=-1, 0b01=0, 0b10=+1)
// Output: TriadSextet [0-26] in 6-bit encoding
static inline triadsextet_t triadsextet_pack(uint8_t t0, uint8_t t1, uint8_t t2) {
    // Convert 2-bit encoding to offset {0, 1, 2}
    int o0 = trit_to_int_constexpr(t0) + 1;  // -1→0, 0→1, +1→2
    int o1 = trit_to_int_constexpr(t1) + 1;
    int o2 = trit_to_int_constexpr(t2) + 1;

    // Encode as base-27: Σ(oᵢ × 3ⁱ)
    return static_cast<triadsextet_t>(o0 + o1*3 + o2*9);
}

// --- Decode: Unpack 1 sextet into 3 trits ---
// Input: TriadSextet [0-26] (values 27-63 treated as zeros)
// Output: 3 trits in 2-bit encoding

struct TriadSextetUnpacked {
    uint8_t t0, t1, t2;
};

static inline TriadSextetUnpacked triadsextet_unpack(triadsextet_t sextet) {
    // Mask to ensure only 6 bits are used
    uint8_t masked = sextet & SEXTET_MASK;

    return {
        TRIADSEXTET_EXTRACT_T0_LUT[masked],
        TRIADSEXTET_EXTRACT_T1_LUT[masked],
        TRIADSEXTET_EXTRACT_T2_LUT[masked]
    };
}

// --- Extract single trit from sextet ---
// Position ∈ [0, 2], returns 2-bit trit encoding
static inline uint8_t triadsextet_extract_trit(triadsextet_t sextet, size_t position) {
    uint8_t masked = sextet & SEXTET_MASK;

    switch (position) {
        case 0: return TRIADSEXTET_EXTRACT_T0_LUT[masked];
        case 1: return TRIADSEXTET_EXTRACT_T1_LUT[masked];
        case 2: return TRIADSEXTET_EXTRACT_T2_LUT[masked];
        default: return 0b01;  // Invalid position → neutral zero
    }
}

// =============================================================================
// Validation and Sanitization
// =============================================================================

// Check if sextet is valid TriadSextet encoding (0-26)
static inline bool triadsextet_is_valid(triadsextet_t sextet) {
    return (sextet & SEXTET_MASK) <= SEXTET_MAX_VALID;
}

// Sanitize sextet: map invalid values to 0 (encoding of {-1, -1, -1})
static inline triadsextet_t triadsextet_sanitize(triadsextet_t sextet) {
    return triadsextet_is_valid(sextet) ? (sextet & SEXTET_MASK) : 0;
}

// Validate entire TriadSextet array
// Returns: Number of invalid sextets found (0 = all valid)
static inline size_t triadsextet_validate_array(const triadsextet_t* data, size_t num_sextets) {
    size_t invalid_count = 0;
    for (size_t i = 0; i < num_sextets; ++i) {
        if (!triadsextet_is_valid(data[i])) {
            invalid_count++;
        }
    }
    return invalid_count;
}

// =============================================================================
// Conversion from Integer Trits (FFI-friendly)
// =============================================================================

// Pack 3 integer trits (∈ {-1, 0, +1}) into sextet
// Input: Integer trits (for external API convenience)
// Output: TriadSextet
static inline triadsextet_t triadsextet_pack_int(int8_t t0, int8_t t1, int8_t t2) {
    // Clamp to valid range and convert to offsets
    int o0 = clamp_ternary(t0) + 1;  // -1→0, 0→1, +1→2
    int o1 = clamp_ternary(t1) + 1;
    int o2 = clamp_ternary(t2) + 1;

    return static_cast<triadsextet_t>(o0 + o1*3 + o2*9);
}

// Unpack sextet into 3 integer trits (∈ {-1, 0, +1})
// Input: TriadSextet
// Output: Integer trits via output parameters
static inline void triadsextet_unpack_int(triadsextet_t sextet, int8_t* t0, int8_t* t1, int8_t* t2) {
    auto unpacked = triadsextet_unpack(sextet);

    *t0 = trit_to_int_constexpr(unpacked.t0);
    *t1 = trit_to_int_constexpr(unpacked.t1);
    *t2 = trit_to_int_constexpr(unpacked.t2);
}

// =============================================================================
// Array Size Conversions
// =============================================================================

// Calculate number of sextets needed for N trits (byte-aligned storage)
constexpr size_t triadsextet_sextets_for_trits(size_t num_trits) {
    return (num_trits + 2) / 3;  // Ceiling division: ⌈N/3⌉
}

// Calculate number of trits stored in N sextets
constexpr size_t triadsextet_trits_in_sextets(size_t num_sextets) {
    return num_sextets * 3;
}

// Calculate bytes needed for N sextets (byte-aligned storage: 1 sextet = 1 byte)
constexpr size_t triadsextet_bytes_for_sextets(size_t num_sextets) {
    return num_sextets;  // 1:1 mapping in byte-aligned storage
}

// Calculate bytes needed for N trits (byte-aligned sextet storage)
constexpr size_t triadsextet_bytes_for_trits(size_t num_trits) {
    return triadsextet_sextets_for_trits(num_trits);
}

// =============================================================================
// Ternary Operations on TriadSextet (via 2-bit transcoding)
// =============================================================================

#include "core/algebra/ternary_algebra.h"  // For tadd, tmul, etc.

// TriadSextet addition (via unpack → 2-bit operation → repack)
static inline triadsextet_t triadsextet_tadd(triadsextet_t sa, triadsextet_t sb) {
    auto a = triadsextet_unpack(sa);
    auto b = triadsextet_unpack(sb);

    uint8_t r0 = tadd(a.t0, b.t0);
    uint8_t r1 = tadd(a.t1, b.t1);
    uint8_t r2 = tadd(a.t2, b.t2);

    return triadsextet_pack(r0, r1, r2);
}

// TriadSextet multiplication
static inline triadsextet_t triadsextet_tmul(triadsextet_t sa, triadsextet_t sb) {
    auto a = triadsextet_unpack(sa);
    auto b = triadsextet_unpack(sb);

    uint8_t r0 = tmul(a.t0, b.t0);
    uint8_t r1 = tmul(a.t1, b.t1);
    uint8_t r2 = tmul(a.t2, b.t2);

    return triadsextet_pack(r0, r1, r2);
}

// TriadSextet minimum
static inline triadsextet_t triadsextet_tmin(triadsextet_t sa, triadsextet_t sb) {
    auto a = triadsextet_unpack(sa);
    auto b = triadsextet_unpack(sb);

    uint8_t r0 = tmin(a.t0, b.t0);
    uint8_t r1 = tmin(a.t1, b.t1);
    uint8_t r2 = tmin(a.t2, b.t2);

    return triadsextet_pack(r0, r1, r2);
}

// TriadSextet maximum
static inline triadsextet_t triadsextet_tmax(triadsextet_t sa, triadsextet_t sb) {
    auto a = triadsextet_unpack(sa);
    auto b = triadsextet_unpack(sb);

    uint8_t r0 = tmax(a.t0, b.t0);
    uint8_t r1 = tmax(a.t1, b.t1);
    uint8_t r2 = tmax(a.t2, b.t2);

    return triadsextet_pack(r0, r1, r2);
}

// TriadSextet negation
static inline triadsextet_t triadsextet_tnot(triadsextet_t sa) {
    auto a = triadsextet_unpack(sa);

    uint8_t r0 = tnot(a.t0);
    uint8_t r1 = tnot(a.t1);
    uint8_t r2 = tnot(a.t2);

    return triadsextet_pack(r0, r1, r2);
}

// =============================================================================
// Compile-time Tests (Proof of Correctness)
// =============================================================================

namespace {
    // Test encoding/decoding round-trip
    constexpr bool test_triadsextet_roundtrip() {
        // Test case: (+1, 0, -1)
        uint8_t t0 = 0b10;  // +1
        uint8_t t1 = 0b01;  //  0
        uint8_t t2 = 0b00;  // -1

        // Encode
        int o0 = trit_to_int_constexpr(t0) + 1;  // +1 → 2
        int o1 = trit_to_int_constexpr(t1) + 1;  //  0 → 1
        int o2 = trit_to_int_constexpr(t2) + 1;  // -1 → 0

        uint8_t sextet = o0 + o1*3 + o2*9;
        // = 2 + 1*3 + 0*9 = 2 + 3 + 0 = 5

        // Decode using LUTs
        uint8_t r0 = TRIADSEXTET_EXTRACT_T0_LUT[sextet];
        uint8_t r1 = TRIADSEXTET_EXTRACT_T1_LUT[sextet];
        uint8_t r2 = TRIADSEXTET_EXTRACT_T2_LUT[sextet];

        // Verify round-trip
        return (r0 == t0) && (r1 == t1) && (r2 == t2);
    }

    // Test all valid sextets (27 states)
    constexpr bool test_triadsextet_exhaustive() {
        for (int o0 = 0; o0 < 3; ++o0) {
            for (int o1 = 0; o1 < 3; ++o1) {
                for (int o2 = 0; o2 < 3; ++o2) {
                    // Compute sextet
                    uint8_t sextet = o0 + o1*3 + o2*9;

                    // Verify sextet is in valid range
                    if (sextet > SEXTET_MAX_VALID) return false;

                    // Decode
                    uint8_t r0 = TRIADSEXTET_EXTRACT_T0_LUT[sextet];
                    uint8_t r1 = TRIADSEXTET_EXTRACT_T1_LUT[sextet];
                    uint8_t r2 = TRIADSEXTET_EXTRACT_T2_LUT[sextet];

                    // Convert back to offsets
                    int r0_offset = trit_to_int_constexpr(r0) + 1;
                    int r1_offset = trit_to_int_constexpr(r1) + 1;
                    int r2_offset = trit_to_int_constexpr(r2) + 1;

                    // Verify round-trip
                    if (r0_offset != o0 || r1_offset != o1 || r2_offset != o2) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    static_assert(test_triadsextet_roundtrip(), "TriadSextet round-trip encoding/decoding failed");
    static_assert(test_triadsextet_exhaustive(), "TriadSextet exhaustive validation failed");
}

#endif // TERNARY_TRIADSEXTET_H
