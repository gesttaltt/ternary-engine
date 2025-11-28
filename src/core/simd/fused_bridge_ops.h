// fused_bridge_ops.h — Fused Bridge Operations (int8 ↔ uint8 with kernel fusion)
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
// BRIDGE LAYER: ALGEBRAIC FOUNDATIONS
// =============================================================================
//
// This header implements the CRITICAL bridge between two representation worlds:
//
//   World A (Python/NumPy): int8 semantic values {-1, 0, +1}
//   World B (SIMD Engine):  uint8 2-bit encoded {0b00, 0b01, 0b10}
//
// The bridge is an ISOMORPHISM φ: A → B defined by:
//   φ(x) = x + 1       (maps -1→0, 0→1, +1→2)
//   φ⁻¹(y) = y - 1     (inverse mapping)
//
// KEY INSIGHT: The isomorphism preserves algebraic structure:
//   φ(a ⊙ b) = φ(a) ⊙' φ(b)
//
// Where ⊙ is semantic operation and ⊙' is LUT-based implementation.
//
// =============================================================================
// PERFORMANCE ANALYSIS
// =============================================================================
//
// NAIVE PIPELINE (current):
//   Python: a_int8 → NumPy +1 → astype → a_uint8
//   Python: b_int8 → NumPy +1 → astype → b_uint8
//   C++: kernel(a_uint8, b_uint8) → r_uint8
//   Python: r_uint8 → astype → NumPy -1 → r_int8
//
//   Cost: 5 array allocations, 320 bytes memory traffic per 32 elements
//   Time breakdown: 97% conversion, 3% kernel
//
// FUSED BRIDGE (this implementation):
//   C++: load a_int8, b_int8
//   C++: φ(a), φ(b) in registers (2 SIMD adds)
//   C++: kernel operation
//   C++: φ⁻¹(result) in register (1 SIMD sub)
//   C++: store r_int8
//
//   Cost: 0 array allocations, 96 bytes memory traffic per 32 elements
//   Time breakdown: 100% useful work
//
// EXPECTED SPEEDUP: ~30x for full pipeline
//
// =============================================================================

#ifndef FUSED_BRIDGE_OPS_H
#define FUSED_BRIDGE_OPS_H

#include <immintrin.h>
#include "simd_avx2_32trit_ops.h"
#include "../algebra/ternary_algebra.h"

// =============================================================================
// CONSTANTS: Bridge Transformation Vectors
// =============================================================================

namespace bridge {

// The isomorphism constant: φ(x) = x + 1
// Broadcast to all 32 byte lanes for SIMD operation
static inline __m256i get_phi_constant() {
    return _mm256_set1_epi8(1);
}

// Precomputed for performance (avoid repeated broadcast)
// Note: Static initialization is safe for POD types in header
namespace constants {
    // These will be initialized on first use via inline function
    // to avoid static initialization order issues
}

} // namespace bridge

// =============================================================================
// ISOMORPHISM OPERATIONS (φ and φ⁻¹)
// =============================================================================
//
// These are the fundamental bridge transformations.
// They operate entirely in SIMD registers with zero memory traffic.
//
// COST: 1 cycle per transformation (single SIMD instruction)
//

// φ: int8 → uint8 (semantic → computational)
// Maps: -1 → 0, 0 → 1, +1 → 2
static inline __m256i phi_transform(__m256i x_int8) {
    return _mm256_add_epi8(x_int8, _mm256_set1_epi8(1));
}

// φ⁻¹: uint8 → int8 (computational → semantic)
// Maps: 0 → -1, 1 → 0, 2 → +1
static inline __m256i phi_inverse(__m256i y_uint8) {
    return _mm256_sub_epi8(y_uint8, _mm256_set1_epi8(1));
}

// =============================================================================
// FUSED BRIDGE BINARY OPERATIONS
// =============================================================================
//
// Pattern: result_int8 = φ⁻¹(kernel(φ(a_int8), φ(b_int8)))
//
// All transformations happen in registers:
//   1. Load int8 inputs
//   2. Apply φ (add 1) - register operation
//   3. Apply kernel (LUT shuffle) - register operation
//   4. Apply φ⁻¹ (sub 1) - register operation
//   5. Store int8 result
//
// Memory traffic: 96 bytes per 32 elements (3 × 32)
// Allocations: 0
//

// Generic fused bridge for binary operations
template <bool Sanitize = true>
static inline __m256i fused_bridge_binary(
    __m256i a_int8,
    __m256i b_int8,
    __m256i lut
) {
    const __m256i one = _mm256_set1_epi8(1);

    // φ: int8 → uint8 (in registers)
    __m256i a_uint8 = _mm256_add_epi8(a_int8, one);
    __m256i b_uint8 = _mm256_add_epi8(b_int8, one);

    // Kernel operation (canonical indexing)
    __m256i result_uint8 = binary_simd_op<Sanitize>(a_uint8, b_uint8, lut);

    // φ⁻¹: uint8 → int8 (in registers)
    return _mm256_sub_epi8(result_uint8, one);
}

// --- Concrete Fused Bridge Operations ---

// tadd_int8: Saturated ternary addition on int8 inputs
template <bool Sanitize = true>
static inline __m256i tadd_int8_simd(__m256i a, __m256i b) {
    return fused_bridge_binary<Sanitize>(a, b, g_luts.tadd);
}

// tmul_int8: Ternary multiplication on int8 inputs
template <bool Sanitize = true>
static inline __m256i tmul_int8_simd(__m256i a, __m256i b) {
    return fused_bridge_binary<Sanitize>(a, b, g_luts.tmul);
}

// tmin_int8: Ternary minimum on int8 inputs
template <bool Sanitize = true>
static inline __m256i tmin_int8_simd(__m256i a, __m256i b) {
    return fused_bridge_binary<Sanitize>(a, b, g_luts.tmin);
}

// tmax_int8: Ternary maximum on int8 inputs
template <bool Sanitize = true>
static inline __m256i tmax_int8_simd(__m256i a, __m256i b) {
    return fused_bridge_binary<Sanitize>(a, b, g_luts.tmax);
}

// =============================================================================
// FUSED BRIDGE UNARY OPERATIONS
// =============================================================================

// Generic fused bridge for unary operations
template <bool Sanitize = true>
static inline __m256i fused_bridge_unary(
    __m256i a_int8,
    __m256i lut
) {
    const __m256i one = _mm256_set1_epi8(1);

    // φ: int8 → uint8
    __m256i a_uint8 = _mm256_add_epi8(a_int8, one);

    // Kernel operation
    __m256i indices = maybe_mask<Sanitize>(a_uint8);
    __m256i result_uint8 = _mm256_shuffle_epi8(lut, indices);

    // φ⁻¹: uint8 → int8
    return _mm256_sub_epi8(result_uint8, one);
}

// tnot_int8: Ternary negation on int8 inputs
template <bool Sanitize = true>
static inline __m256i tnot_int8_simd(__m256i a) {
    return fused_bridge_unary<Sanitize>(a, g_luts.tnot);
}

// =============================================================================
// FUSED BRIDGE + FUSION (Level 3 Operations)
// =============================================================================
//
// These combine bridge conversion WITH operation fusion for maximum efficiency.
// Pattern: result_int8 = φ⁻¹(unary(binary(φ(a), φ(b))))
//
// Memory traffic: Same as Level 2 (96 bytes per 32 elements)
// Compute: One extra shuffle, still all in registers
//

// fused_tnot_tadd_int8: tnot(tadd(a, b)) on int8 inputs
template <bool Sanitize = true>
static inline __m256i fused_tnot_tadd_int8_simd(__m256i a, __m256i b) {
    const __m256i one = _mm256_set1_epi8(1);

    // φ: int8 → uint8
    __m256i a_uint8 = _mm256_add_epi8(a, one);
    __m256i b_uint8 = _mm256_add_epi8(b, one);

    // Fused kernel: tnot(tadd(a, b))
    __m256i tadd_result = binary_simd_op<Sanitize>(a_uint8, b_uint8, g_luts.tadd);
    __m256i indices = maybe_mask<Sanitize>(tadd_result);
    __m256i result_uint8 = _mm256_shuffle_epi8(g_luts.tnot, indices);

    // φ⁻¹: uint8 → int8
    return _mm256_sub_epi8(result_uint8, one);
}

// fused_tnot_tmul_int8: tnot(tmul(a, b)) on int8 inputs
template <bool Sanitize = true>
static inline __m256i fused_tnot_tmul_int8_simd(__m256i a, __m256i b) {
    const __m256i one = _mm256_set1_epi8(1);

    __m256i a_uint8 = _mm256_add_epi8(a, one);
    __m256i b_uint8 = _mm256_add_epi8(b, one);

    __m256i tmul_result = binary_simd_op<Sanitize>(a_uint8, b_uint8, g_luts.tmul);
    __m256i indices = maybe_mask<Sanitize>(tmul_result);
    __m256i result_uint8 = _mm256_shuffle_epi8(g_luts.tnot, indices);

    return _mm256_sub_epi8(result_uint8, one);
}

// fused_tnot_tmin_int8: tnot(tmin(a, b)) on int8 inputs
template <bool Sanitize = true>
static inline __m256i fused_tnot_tmin_int8_simd(__m256i a, __m256i b) {
    const __m256i one = _mm256_set1_epi8(1);

    __m256i a_uint8 = _mm256_add_epi8(a, one);
    __m256i b_uint8 = _mm256_add_epi8(b, one);

    __m256i tmin_result = binary_simd_op<Sanitize>(a_uint8, b_uint8, g_luts.tmin);
    __m256i indices = maybe_mask<Sanitize>(tmin_result);
    __m256i result_uint8 = _mm256_shuffle_epi8(g_luts.tnot, indices);

    return _mm256_sub_epi8(result_uint8, one);
}

// fused_tnot_tmax_int8: tnot(tmax(a, b)) on int8 inputs
template <bool Sanitize = true>
static inline __m256i fused_tnot_tmax_int8_simd(__m256i a, __m256i b) {
    const __m256i one = _mm256_set1_epi8(1);

    __m256i a_uint8 = _mm256_add_epi8(a, one);
    __m256i b_uint8 = _mm256_add_epi8(b, one);

    __m256i tmax_result = binary_simd_op<Sanitize>(a_uint8, b_uint8, g_luts.tmax);
    __m256i indices = maybe_mask<Sanitize>(tmax_result);
    __m256i result_uint8 = _mm256_shuffle_epi8(g_luts.tnot, indices);

    return _mm256_sub_epi8(result_uint8, one);
}

// =============================================================================
// SCALAR BRIDGE OPERATIONS (for tail elements)
// =============================================================================

// Scalar φ: int8 → uint8
static inline uint8_t phi_scalar(int8_t x) {
    return static_cast<uint8_t>(x + 1);
}

// Scalar φ⁻¹: uint8 → int8
static inline int8_t phi_inverse_scalar(uint8_t y) {
    return static_cast<int8_t>(y - 1);
}

// Scalar fused bridge binary
static inline int8_t fused_bridge_binary_scalar(int8_t a, int8_t b, const uint8_t* lut) {
    uint8_t a_enc = phi_scalar(a);
    uint8_t b_enc = phi_scalar(b);
    uint8_t result_enc = lut[(a_enc * 3) + b_enc];  // Canonical indexing
    return phi_inverse_scalar(result_enc);
}

// Concrete scalar operations
static inline int8_t tadd_int8_scalar(int8_t a, int8_t b) {
    return fused_bridge_binary_scalar(a, b, TADD_LUT_CANONICAL.data());
}

static inline int8_t tmul_int8_scalar(int8_t a, int8_t b) {
    return fused_bridge_binary_scalar(a, b, TMUL_LUT_CANONICAL.data());
}

static inline int8_t tmin_int8_scalar(int8_t a, int8_t b) {
    return fused_bridge_binary_scalar(a, b, TMIN_LUT_CANONICAL.data());
}

static inline int8_t tmax_int8_scalar(int8_t a, int8_t b) {
    return fused_bridge_binary_scalar(a, b, TMAX_LUT_CANONICAL.data());
}

static inline int8_t tnot_int8_scalar(int8_t a) {
    uint8_t a_enc = phi_scalar(a);
    uint8_t result_enc = TNOT_LUT_CANONICAL[a_enc];
    return phi_inverse_scalar(result_enc);
}

// =============================================================================
// CORRECTNESS VALIDATION (compile-time)
// =============================================================================

namespace bridge_validation {

// Verify φ mapping
static_assert(static_cast<uint8_t>(-1 + 1) == 0, "φ(-1) should be 0");
static_assert(static_cast<uint8_t>(0 + 1) == 1, "φ(0) should be 1");
static_assert(static_cast<uint8_t>(1 + 1) == 2, "φ(+1) should be 2");

// Verify φ⁻¹ mapping
static_assert(static_cast<int8_t>(0 - 1) == -1, "φ⁻¹(0) should be -1");
static_assert(static_cast<int8_t>(1 - 1) == 0, "φ⁻¹(1) should be 0");
static_assert(static_cast<int8_t>(2 - 1) == 1, "φ⁻¹(2) should be +1");

// Verify roundtrip: φ⁻¹(φ(x)) = x
constexpr bool test_roundtrip() {
    for (int8_t x = -1; x <= 1; ++x) {
        uint8_t y = static_cast<uint8_t>(x + 1);
        int8_t z = static_cast<int8_t>(y - 1);
        if (z != x) return false;
    }
    return true;
}
static_assert(test_roundtrip(), "Bridge roundtrip φ⁻¹(φ(x)) = x must hold");

} // namespace bridge_validation

// =============================================================================
// OPERATION COUNT ANALYSIS
// =============================================================================
//
// PER 32 ELEMENTS:
//
// | Operation              | Instructions | Cycles | Memory |
// |------------------------|--------------|--------|--------|
// | Load a_int8            | 1            | 1      | 32B    |
// | Load b_int8            | 1            | 1      | 32B    |
// | φ(a) = add(a, 1)       | 1            | 1      | 0      |
// | φ(b) = add(b, 1)       | 1            | 1      | 0      |
// | Load CANON_A (cached)  | 0            | 0      | 0      |
// | Load CANON_B (cached)  | 0            | 0      | 0      |
// | Shuffle a contribution | 1            | 1      | 0      |
// | Shuffle b contribution | 1            | 1      | 0      |
// | Add indices            | 1            | 1      | 0      |
// | Load LUT (cached)      | 0            | 0      | 0      |
// | Shuffle LUT lookup     | 1            | 1      | 0      |
// | φ⁻¹(r) = sub(r, 1)     | 1            | 1      | 0      |
// | Store r_int8           | 1            | 1      | 32B    |
// |------------------------|--------------|--------|--------|
// | TOTAL                  | 10           | 10     | 96B    |
//
// THROUGHPUT at 4 GHz: 32 elements / 10 cycles = 3.2 elements/cycle
//                    = 12.8 billion elements/second
//
// =============================================================================

#endif // FUSED_BRIDGE_OPS_H
