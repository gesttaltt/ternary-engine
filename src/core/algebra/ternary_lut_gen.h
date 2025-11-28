// ternary_lut_gen.h — Constexpr compile-time LUT generation for ternary operations
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
// DESIGN RATIONALE
// =============================================================================
//
// Constexpr LUT generation provides:
// - Single source of truth: Algorithm IS the documentation
// - Compile-time verification: No runtime overhead
// - Maintainability: Adding operations requires only algebraic lambda
// - AVX2 compatibility: Generates same 16-byte LUTs for _mm256_shuffle_epi8
//
// Target: AVX2-compatible CPUs (Intel Haswell 2013+, AMD Excavator 2015+)
// - Binary LUTs: 16 entries for (a << 2) | b indexing
// - Unary LUTs: 4 entries (padded to 16 for SIMD compatibility if needed)
//
// =============================================================================

#ifndef TERNARY_LUT_GEN_H
#define TERNARY_LUT_GEN_H

#include <stdint.h>
#include <array>

// --- Constexpr conversion functions for compile-time evaluation ---

constexpr uint8_t int_to_trit_constexpr(int v) {
    return (v < 0) ? 0b00 : (v > 0) ? 0b10 : 0b01;
}

constexpr int trit_to_int_constexpr(uint8_t t) {
    return (t == 0b00) ? -1 : (t == 0b10) ? 1 : 0;
}

// --- Binary LUT Generator (16 entries) ---
// Generates lookup tables for binary operations: op(a, b)
// Index format: (a << 2) | b, where a,b ∈ {0b00, 0b01, 0b10, 0b11}
// AVX2 shuffle requires 16-byte tables for _mm256_shuffle_epi8 compatibility

template <typename Func>
constexpr std::array<uint8_t, 16> make_binary_lut(Func op) {
    std::array<uint8_t, 16> lut{};

    // Generate all 16 entries: 4 values of a × 4 values of b
    for (size_t a = 0; a < 4; ++a) {
        for (size_t b = 0; b < 4; ++b) {
            size_t index = (a << 2) | b;
            lut[index] = op(static_cast<uint8_t>(a), static_cast<uint8_t>(b));
        }
    }

    return lut;
}

// --- Unary LUT Generator (4 entries) ---
// Generates lookup tables for unary operations: op(a)
// Index format: a & 0x03, where a ∈ {0b00, 0b01, 0b10, 0b11}
// Result is 4 entries (can be padded to 16 for SIMD if needed)

template <typename Func>
constexpr std::array<uint8_t, 4> make_unary_lut(Func op) {
    std::array<uint8_t, 4> lut{};

    // Generate all 4 entries
    for (size_t a = 0; a < 4; ++a) {
        lut[a] = op(static_cast<uint8_t>(a));
    }

    return lut;
}

// --- Unary LUT Generator with SIMD Padding (16 entries) ---
// Generates 16-entry LUT by replicating 4-entry pattern
// Used in SIMD code where _mm256_shuffle_epi8 requires 16-byte LUTs

template <typename Func>
constexpr std::array<uint8_t, 16> make_unary_lut_padded(Func op) {
    std::array<uint8_t, 16> lut{};

    // Generate base 4 entries and replicate pattern 4 times
    for (size_t i = 0; i < 16; ++i) {
        size_t base_index = i & 0x03;  // Wrap to 0-3
        lut[i] = op(static_cast<uint8_t>(base_index));
    }

    return lut;
}

// --- Helper: Clamp integer to ternary range [-1, 0, +1] ---

constexpr int clamp_ternary(int value) {
    if (value > 1) return 1;
    if (value < -1) return -1;
    return value;
}

// =============================================================================
// CANONICAL INDEXING LUT GENERATORS
// =============================================================================
//
// Canonical indexing uses idx = a*3 + b instead of (a << 2) | b
//
// Benefits:
// - Compact: Uses indices 0-8 (no gaps) vs 0,1,2,4,5,6,8,9,10 (with gaps)
// - Algebraic: Preserves Z_3 group structure
// - SIMD-friendly: Dual-shuffle + ADD instead of shift + OR
//
// Layout for 16-entry LUT (padded for AVX2 shuffle):
//   [0] = op(-1, -1)   [1] = op(-1, 0)   [2] = op(-1, +1)
//   [3] = op( 0, -1)   [4] = op( 0, 0)   [5] = op( 0, +1)
//   [6] = op(+1, -1)   [7] = op(+1, 0)   [8] = op(+1, +1)
//   [9-15] = padding (copies of [0] for safety)
//
// =============================================================================

// --- Canonical Binary LUT Generator (16 entries, indices 0-8 used) ---
// Generates lookup tables for binary operations with canonical indexing
// Index formula: a*3 + b, where a,b ∈ {0, 1, 2} representing {-1, 0, +1}

template <typename Func>
constexpr std::array<uint8_t, 16> make_canonical_binary_lut(Func op) {
    std::array<uint8_t, 16> lut{};

    // Generate 9 valid entries (3x3 combinations)
    for (size_t a = 0; a < 3; ++a) {
        for (size_t b = 0; b < 3; ++b) {
            size_t index = a * 3 + b;  // Canonical index: 0-8
            // Call op with 2-bit encoded trits (0=-1, 1=0, 2=+1)
            lut[index] = op(static_cast<uint8_t>(a), static_cast<uint8_t>(b));
        }
    }

    // Pad entries 9-15 with safe fallback (copy of entry 0)
    // This handles any out-of-range indices gracefully
    for (size_t i = 9; i < 16; ++i) {
        lut[i] = lut[0];
    }

    return lut;
}

// --- Canonical Unary LUT Generator (16 entries, indices 0-2 used) ---
// For unary operations with canonical indexing
// Index formula: a, where a ∈ {0, 1, 2} representing {-1, 0, +1}

template <typename Func>
constexpr std::array<uint8_t, 16> make_canonical_unary_lut(Func op) {
    std::array<uint8_t, 16> lut{};

    // Generate 3 valid entries
    for (size_t a = 0; a < 3; ++a) {
        lut[a] = op(static_cast<uint8_t>(a));
    }

    // Pad entries 3-15 by replicating the pattern
    for (size_t i = 3; i < 16; ++i) {
        lut[i] = lut[i % 3];
    }

    return lut;
}

#endif // TERNARY_LUT_GEN_H
