// ternary_algebra.h — optimized ternary algebra core header
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

#ifndef TERNARY_ALGEBRA_H
#define TERNARY_ALGEBRA_H

#include <stdint.h>
#include "ternary_lut_gen.h"  // Constexpr LUT generation

// each trit occupies 2 bits → 00 = -1, 01 = 0, 10 = +1
typedef uint8_t trit;

// Platform-specific force inline
#ifdef _MSC_VER
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE __attribute__((always_inline)) inline
#endif

// --- conversions (kept for compatibility/reference) ---
static inline trit int_to_trit(int v) { return (v < 0) ? 0b00 : (v > 0) ? 0b10 : 0b01; }
static inline int  trit_to_int(trit t){ return (t==0b00)?-1:(t==0b10)?1:0; }

// --- Constexpr-generated lookup tables (OPT-AUTO-LUT) ---
// Index format: (a << 2) | b, where a,b are 2-bit trit values
//
// LUT SIZE STRATEGY:
// - Binary operations (TADD, TMUL, TMIN, TMAX): 16 entries
//   * Cover all 4×4 combinations of 2-bit trit pairs
//   * Directly compatible with AVX2 _mm256_shuffle_epi8 (requires 16-byte LUTs)
//   * Used by both scalar operations and SIMD operations without conversion
//
// - Unary operations: Dual LUT approach
//   * TNOT_LUT: 4 entries (scalar/minimal use)
//   * TNOT_LUT_SIMD: 16 entries (padded/replicated for AVX2 compatibility)
//   * Both generated from same algebraic lambda, ensuring semantic consistency

// TADD: Saturated ternary addition
// Algorithm: clamp(a + b, -1, +1)
constexpr auto TADD_LUT = make_binary_lut([](uint8_t a, uint8_t b) -> uint8_t {
    int sa = trit_to_int_constexpr(a);
    int sb = trit_to_int_constexpr(b);
    int sum = sa + sb;
    // Saturate to [-1, +1] range
    return int_to_trit_constexpr(clamp_ternary(sum));
});

// TMUL: Ternary multiplication
// Algorithm: a * b
constexpr auto TMUL_LUT = make_binary_lut([](uint8_t a, uint8_t b) -> uint8_t {
    int sa = trit_to_int_constexpr(a);
    int sb = trit_to_int_constexpr(b);
    int product = sa * sb;
    return int_to_trit_constexpr(product);
});

// TMIN: Ternary minimum
// Algorithm: min(a, b)
constexpr auto TMIN_LUT = make_binary_lut([](uint8_t a, uint8_t b) -> uint8_t {
    int sa = trit_to_int_constexpr(a);
    int sb = trit_to_int_constexpr(b);
    int minimum = (sa < sb) ? sa : sb;
    return int_to_trit_constexpr(minimum);
});

// TMAX: Ternary maximum
// Algorithm: max(a, b)
constexpr auto TMAX_LUT = make_binary_lut([](uint8_t a, uint8_t b) -> uint8_t {
    int sa = trit_to_int_constexpr(a);
    int sb = trit_to_int_constexpr(b);
    int maximum = (sa > sb) ? sa : sb;
    return int_to_trit_constexpr(maximum);
});

// TNOT: Ternary negation
// Algorithm: -a (sign flip, 0 unchanged)
constexpr auto TNOT_LUT = make_unary_lut([](uint8_t a) -> uint8_t {
    int sa = trit_to_int_constexpr(a);
    int negated = -sa;
    return int_to_trit_constexpr(negated);
});

// TNOT_SIMD: Padded version for SIMD operations (16 entries for _mm256_shuffle_epi8)
constexpr auto TNOT_LUT_SIMD = make_unary_lut_padded([](uint8_t a) -> uint8_t {
    int sa = trit_to_int_constexpr(a);
    int negated = -sa;
    return int_to_trit_constexpr(negated);
});

// =============================================================================
// CANONICAL INDEXING LUTs (for SIMD dual-shuffle optimization)
// =============================================================================
//
// These LUTs are organized for canonical indexing: idx = a*3 + b
// Used by SIMD kernels with dual-shuffle + ADD instead of shift + OR
//
// Layout:
//   Index 0 = op(0,0) = op(-1,-1)    Index 1 = op(0,1) = op(-1, 0)
//   Index 2 = op(0,2) = op(-1,+1)    Index 3 = op(1,0) = op( 0,-1)
//   Index 4 = op(1,1) = op( 0, 0)    Index 5 = op(1,2) = op( 0,+1)
//   Index 6 = op(2,0) = op(+1,-1)    Index 7 = op(2,1) = op(+1, 0)
//   Index 8 = op(2,2) = op(+1,+1)    Index 9-15 = padding
//
// =============================================================================

// TADD_CANONICAL: Saturated ternary addition (canonical layout)
constexpr auto TADD_LUT_CANONICAL = make_canonical_binary_lut([](uint8_t a, uint8_t b) -> uint8_t {
    // a,b are in canonical encoding: 0=-1, 1=0, 2=+1
    int sa = static_cast<int>(a) - 1;  // Convert to integer: -1, 0, +1
    int sb = static_cast<int>(b) - 1;
    int sum = sa + sb;
    // Saturate to [-1, +1] and convert back to encoding
    return static_cast<uint8_t>(clamp_ternary(sum) + 1);
});

// TMUL_CANONICAL: Ternary multiplication (canonical layout)
constexpr auto TMUL_LUT_CANONICAL = make_canonical_binary_lut([](uint8_t a, uint8_t b) -> uint8_t {
    int sa = static_cast<int>(a) - 1;
    int sb = static_cast<int>(b) - 1;
    int product = sa * sb;
    return static_cast<uint8_t>(product + 1);
});

// TMIN_CANONICAL: Ternary minimum (canonical layout)
constexpr auto TMIN_LUT_CANONICAL = make_canonical_binary_lut([](uint8_t a, uint8_t b) -> uint8_t {
    int sa = static_cast<int>(a) - 1;
    int sb = static_cast<int>(b) - 1;
    int minimum = (sa < sb) ? sa : sb;
    return static_cast<uint8_t>(minimum + 1);
});

// TMAX_CANONICAL: Ternary maximum (canonical layout)
constexpr auto TMAX_LUT_CANONICAL = make_canonical_binary_lut([](uint8_t a, uint8_t b) -> uint8_t {
    int sa = static_cast<int>(a) - 1;
    int sb = static_cast<int>(b) - 1;
    int maximum = (sa > sb) ? sa : sb;
    return static_cast<uint8_t>(maximum + 1);
});

// TNOT_CANONICAL: Ternary negation (canonical layout)
constexpr auto TNOT_LUT_CANONICAL = make_canonical_unary_lut([](uint8_t a) -> uint8_t {
    int sa = static_cast<int>(a) - 1;
    int negated = -sa;
    return static_cast<uint8_t>(negated + 1);
});

// --- Compile-time LUT size validation ---
// Ensures LUTs have correct sizes for their respective use cases
static_assert(TADD_LUT.size() == 16, "Binary LUTs must have 16 entries for AVX2 compatibility");
static_assert(TMUL_LUT.size() == 16, "Binary LUTs must have 16 entries for AVX2 compatibility");
static_assert(TMIN_LUT.size() == 16, "Binary LUTs must have 16 entries for AVX2 compatibility");
static_assert(TMAX_LUT.size() == 16, "Binary LUTs must have 16 entries for AVX2 compatibility");
static_assert(TNOT_LUT.size() == 4, "Scalar unary LUT must have 4 entries");
static_assert(TNOT_LUT_SIMD.size() == 16, "SIMD unary LUT must have 16 entries (padded)");

// --- Optimized operations using constexpr-generated lookup tables ---
// OPT-051: Force inline + OPT-AUTO-LUT: Compile-time generation

static FORCE_INLINE trit tnot(trit a) {
    return TNOT_LUT[a & 0b11];
}

static FORCE_INLINE trit tmin(trit a, trit b) {
    return TMIN_LUT[(a << 2) | b];
}

static FORCE_INLINE trit tmax(trit a, trit b) {
    return TMAX_LUT[(a << 2) | b];
}

static FORCE_INLINE trit tadd(trit a, trit b) {
    return TADD_LUT[(a << 2) | b];
}

static FORCE_INLINE trit tmul(trit a, trit b) {
    return TMUL_LUT[(a << 2) | b];
}

// --- packing of 4 trits into 1 byte ---
static inline uint8_t pack_trits(trit t0,trit t1,trit t2,trit t3){
    return (t0) | (t1<<2) | (t2<<4) | (t3<<6);
}
static inline trit unpack_trit(uint8_t pack,int idx){
    return (pack>>(2*idx)) & 0b11;
}

#endif // TERNARY_ALGEBRA_H
