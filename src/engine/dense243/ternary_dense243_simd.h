// ternary_dense243_simd.h — SIMD kernels for T5-Dense243 encoding
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
// SIMD acceleration for Dense243-encoded arrays using AVX2 instructions.
//
// CHALLENGE:
//   Dense243 packs 5 trits per byte. Loading 32 bytes with AVX2 gives us
//   160 trits that need to be:
//   1. Extracted into 5 separate trit streams (32 trits each)
//   2. Operated on using existing 2-bit LUT operations
//   3. Repacked back into Dense243 format
//
// STRATEGY:
//   - Use 5× _mm256_shuffle_epi8 with extraction LUTs to unpack
//   - Perform operations on extracted 2-bit trits (existing SIMD kernels)
//   - Repack using arithmetic or SIMD operations
//
// PERFORMANCE CONSIDERATIONS:
//   - Extraction: 5 shuffles per 32 bytes = 5 cycles overhead
//   - Operation: 1 cycle (existing SIMD LUT operations)
//   - Insertion: ~10-15 cycles (arithmetic packing)
//   - Total: ~20 cycles per 32 bytes (vs 1 cycle for direct 2-bit)
//   - Breakeven: Only when memory bandwidth is bottleneck (1M+ elements)
//
// IMPLEMENTATION STATUS:
//   Phase 1: SIMD extraction kernels (this file)
//   Phase 2: Array processing functions (separate implementation file)
//   Phase 3: Integration with main engine (future)
//
// =============================================================================

#ifndef TERNARY_DENSE243_SIMD_H
#define TERNARY_DENSE243_SIMD_H

#include <immintrin.h>
#include <stdint.h>
#include "ternary_dense243.h"
#include "core/simd/simd_avx2_32trit_ops.h"  // For existing 2-bit SIMD operations

// =============================================================================
// Pre-broadcasted Extraction LUTs for AVX2
// =============================================================================

namespace {
    // Helper: Broadcast 256-byte LUT to both 128-bit lanes of AVX2 register
    // Note: Dense243 extraction LUTs are already 256 bytes (full coverage)
    static inline __m256i broadcast_dense243_lut(const uint8_t* lut) {
        // For 256-byte LUTs, we need two 128-bit loads to cover all indices
        // However, _mm256_shuffle_epi8 only uses lower 4 bits for indexing,
        // so we can just load the first 128 bytes and broadcast
        // (indices 128-255 wrap around due to 4-bit index limitation)

        // Actually, _mm256_shuffle_epi8 uses bit 7 to determine lane,
        // and bits 3:0 for the shuffle index within each 128-bit lane.
        // So we need to broadcast the first 128 bytes to both lanes.
        __m128i lut_128 = _mm_loadu_si128((const __m128i*)lut);
        return _mm256_broadcastsi128_si256(lut_128);
    }

    // Pre-broadcasted extraction LUTs for Dense243
    struct Dense243BroadcastedLUTs {
        __m256i extract_t0;
        __m256i extract_t1;
        __m256i extract_t2;
        __m256i extract_t3;
        __m256i extract_t4;

        Dense243BroadcastedLUTs()
            : extract_t0(broadcast_dense243_lut(DENSE243_EXTRACT_T0_LUT.data()))
            , extract_t1(broadcast_dense243_lut(DENSE243_EXTRACT_T1_LUT.data()))
            , extract_t2(broadcast_dense243_lut(DENSE243_EXTRACT_T2_LUT.data()))
            , extract_t3(broadcast_dense243_lut(DENSE243_EXTRACT_T3_LUT.data()))
            , extract_t4(broadcast_dense243_lut(DENSE243_EXTRACT_T4_LUT.data()))
        {}
    };

    // Static instance - initialized once before main()
    static const Dense243BroadcastedLUTs g_dense243_luts;
}  // anonymous namespace

// =============================================================================
// SIMD Extraction Kernels
// =============================================================================

// Extract trit position 0 from 32 Dense243-packed bytes
// Input:  32 packed bytes (each containing 5 trits)
// Output: 32 trits at position 0 (2-bit encoding)
static inline __m256i dense243_extract_t0_simd(__m256i packed) {
    return _mm256_shuffle_epi8(g_dense243_luts.extract_t0, packed);
}

// Extract trit position 1 from 32 Dense243-packed bytes
static inline __m256i dense243_extract_t1_simd(__m256i packed) {
    return _mm256_shuffle_epi8(g_dense243_luts.extract_t1, packed);
}

// Extract trit position 2 from 32 Dense243-packed bytes
static inline __m256i dense243_extract_t2_simd(__m256i packed) {
    return _mm256_shuffle_epi8(g_dense243_luts.extract_t2, packed);
}

// Extract trit position 3 from 32 Dense243-packed bytes
static inline __m256i dense243_extract_t3_simd(__m256i packed) {
    return _mm256_shuffle_epi8(g_dense243_luts.extract_t3, packed);
}

// Extract trit position 4 from 32 Dense243-packed bytes
static inline __m256i dense243_extract_t4_simd(__m256i packed) {
    return _mm256_shuffle_epi8(g_dense243_luts.extract_t4, packed);
}

// =============================================================================
// SIMD Insertion Kernels (Packing 5 Trits → Dense243)
// =============================================================================

// Pack 5 trit positions back into Dense243 format
// Input:  5× __m256i vectors (each containing 32 trits in 2-bit encoding)
// Output: 32 packed Dense243 bytes
//
// STRATEGY: Arithmetic approach using multiply-add
// Formula: result = o0 + o1*3 + o2*9 + o3*27 + o4*81
//
// CHALLENGE: AVX2 doesn't have byte-level multiplication
// SOLUTION: Use _mm256_maddubs_epi16 for multiply-add, then collapse

static inline __m256i dense243_pack_simd(
    __m256i t0,  // 32 trits at position 0 (2-bit encoding)
    __m256i t1,  // 32 trits at position 1
    __m256i t2,  // 32 trits at position 2
    __m256i t3,  // 32 trits at position 3
    __m256i t4   // 32 trits at position 4
) {
    // Step 1: Convert 2-bit trits to offsets {0, 1, 2}
    // Trit encoding: 0b00=-1, 0b01=0, 0b10=+1
    // We need: 0b00→0, 0b01→1, 0b10→2

    // Create conversion LUT: input (2-bit trit) → output (offset)
    // 0b00 (0) → 0, 0b01 (1) → 1, 0b10 (2) → 2, 0b11 (3) → invalid (use 1)
    const __m256i trit_to_offset_lut = _mm256_setr_epi8(
        0, 1, 2, 1,  0, 1, 2, 1,  0, 1, 2, 1,  0, 1, 2, 1,  // Lane 0
        0, 1, 2, 1,  0, 1, 2, 1,  0, 1, 2, 1,  0, 1, 2, 1   // Lane 1
    );

    // Convert all trits to offsets
    __m256i o0 = _mm256_shuffle_epi8(trit_to_offset_lut, t0);
    __m256i o1 = _mm256_shuffle_epi8(trit_to_offset_lut, t1);
    __m256i o2 = _mm256_shuffle_epi8(trit_to_offset_lut, t2);
    __m256i o3 = _mm256_shuffle_epi8(trit_to_offset_lut, t3);
    __m256i o4 = _mm256_shuffle_epi8(trit_to_offset_lut, t4);

    // Step 2: Compute result = o0 + o1*3 + o2*9 + o3*27 + o4*81
    // Use scalar-like arithmetic on each byte

    // Multiply by creating constants
    const __m256i three = _mm256_set1_epi8(3);
    const __m256i nine = _mm256_set1_epi8(9);
    const __m256i twenty_seven = _mm256_set1_epi8(27);
    const __m256i eighty_one = _mm256_set1_epi8(81);

    // AVX2 doesn't have byte multiplication, so we use a trick:
    // For small values (0-2), we can use addition instead

    // o1_times_3 = o1 + o1 + o1
    __m256i o1_times_3 = _mm256_add_epi8(o1, _mm256_add_epi8(o1, o1));

    // o2_times_9 = o2 * 9 (using shift and add: o2*9 = o2*8 + o2 = (o2<<3) + o2)
    __m256i o2_shifted = _mm256_slli_epi16(o2, 3);  // Shift 16-bit words, not bytes
    // For byte-level, we need to use addition chains
    // o2*9 = o2*8 + o2, but we can't shift bytes directly
    // Alternative: o2*9 = o2*3*3 = (o2 + o2 + o2) + (o2 + o2 + o2) + (o2 + o2 + o2)
    __m256i o2_times_3 = _mm256_add_epi8(o2, _mm256_add_epi8(o2, o2));
    __m256i o2_times_9 = _mm256_add_epi8(o2_times_3, _mm256_add_epi8(o2_times_3, o2_times_3));

    // o3*27 = o3*3*9
    __m256i o3_times_3 = _mm256_add_epi8(o3, _mm256_add_epi8(o3, o3));
    __m256i o3_times_9 = _mm256_add_epi8(o3_times_3, _mm256_add_epi8(o3_times_3, o3_times_3));
    __m256i o3_times_27 = _mm256_add_epi8(o3_times_9, _mm256_add_epi8(o3_times_9, o3_times_9));

    // o4*81 = o4*3*3*3*3 (four times multiply by 3)
    // Follow same pattern as o3: build up using addition chains
    __m256i o4_times_3 = _mm256_add_epi8(o4, _mm256_add_epi8(o4, o4));
    __m256i o4_times_9 = _mm256_add_epi8(o4_times_3, _mm256_add_epi8(o4_times_3, o4_times_3));
    __m256i o4_times_27 = _mm256_add_epi8(o4_times_9, _mm256_add_epi8(o4_times_9, o4_times_9));
    __m256i o4_times_81 = _mm256_add_epi8(o4_times_27, _mm256_add_epi8(o4_times_27, o4_times_27));

    // Step 3: Sum all components
    __m256i result = o0;
    result = _mm256_add_epi8(result, o1_times_3);
    result = _mm256_add_epi8(result, o2_times_9);
    result = _mm256_add_epi8(result, o3_times_27);
    result = _mm256_add_epi8(result, o4_times_81);

    return result;
}

// =============================================================================
// Simplified Scalar Fallback for Packing
// =============================================================================

// For tail processing or verification, provide scalar packing
// This is simpler than the SIMD version and easier to understand/debug
static inline void dense243_pack_scalar_block(
    const uint8_t* t0_array,  // 32 trits at position 0
    const uint8_t* t1_array,
    const uint8_t* t2_array,
    const uint8_t* t3_array,
    const uint8_t* t4_array,
    uint8_t* output,          // 32 packed Dense243 bytes
    size_t count              // Number of bytes to pack (≤ 32)
) {
    for (size_t i = 0; i < count; ++i) {
        output[i] = dense243_pack(
            t0_array[i],
            t1_array[i],
            t2_array[i],
            t3_array[i],
            t4_array[i]
        );
    }
}

// =============================================================================
// Complete Unpack-Operate-Repack Pipeline
// =============================================================================

// Perform binary operation on Dense243-encoded data (SIMD kernel)
// Input:  2× __m256i packed Dense243 vectors (32 bytes each = 160 trits)
// Output: 1× __m256i packed Dense243 result
//
// Template parameter: SimdOp should be a 2-bit SIMD operation function
// Example: tadd_simd, tmul_simd, etc.

template <typename SimdOp>
static inline __m256i dense243_binary_op_simd(
    __m256i packed_a,
    __m256i packed_b,
    SimdOp simd_op
) {
    // Step 1: Extract all 5 trit positions from both inputs (10 extractions total)
    __m256i a_t0 = dense243_extract_t0_simd(packed_a);
    __m256i a_t1 = dense243_extract_t1_simd(packed_a);
    __m256i a_t2 = dense243_extract_t2_simd(packed_a);
    __m256i a_t3 = dense243_extract_t3_simd(packed_a);
    __m256i a_t4 = dense243_extract_t4_simd(packed_a);

    __m256i b_t0 = dense243_extract_t0_simd(packed_b);
    __m256i b_t1 = dense243_extract_t1_simd(packed_b);
    __m256i b_t2 = dense243_extract_t2_simd(packed_b);
    __m256i b_t3 = dense243_extract_t3_simd(packed_b);
    __m256i b_t4 = dense243_extract_t4_simd(packed_b);

    // Step 2: Perform operations on each trit position (5 operations)
    __m256i r_t0 = simd_op(a_t0, b_t0);
    __m256i r_t1 = simd_op(a_t1, b_t1);
    __m256i r_t2 = simd_op(a_t2, b_t2);
    __m256i r_t3 = simd_op(a_t3, b_t3);
    __m256i r_t4 = simd_op(a_t4, b_t4);

    // Step 3: Pack results back into Dense243 format
    return dense243_pack_simd(r_t0, r_t1, r_t2, r_t3, r_t4);
}

// Unary operation variant
template <typename SimdOp>
static inline __m256i dense243_unary_op_simd(
    __m256i packed_a,
    SimdOp simd_op
) {
    // Extract all 5 trit positions
    __m256i a_t0 = dense243_extract_t0_simd(packed_a);
    __m256i a_t1 = dense243_extract_t1_simd(packed_a);
    __m256i a_t2 = dense243_extract_t2_simd(packed_a);
    __m256i a_t3 = dense243_extract_t3_simd(packed_a);
    __m256i a_t4 = dense243_extract_t4_simd(packed_a);

    // Perform operations
    __m256i r_t0 = simd_op(a_t0);
    __m256i r_t1 = simd_op(a_t1);
    __m256i r_t2 = simd_op(a_t2);
    __m256i r_t3 = simd_op(a_t3);
    __m256i r_t4 = simd_op(a_t4);

    // Pack results
    return dense243_pack_simd(r_t0, r_t1, r_t2, r_t3, r_t4);
}

// =============================================================================
// Specialized Operation Wrappers
// =============================================================================

// Dense243 SIMD addition
static inline __m256i dense243_tadd_simd(__m256i a, __m256i b) {
    return dense243_binary_op_simd(a, b, tadd_simd<>);
}

// Dense243 SIMD multiplication
static inline __m256i dense243_tmul_simd(__m256i a, __m256i b) {
    return dense243_binary_op_simd(a, b, tmul_simd<>);
}

// Dense243 SIMD minimum
static inline __m256i dense243_tmin_simd(__m256i a, __m256i b) {
    return dense243_binary_op_simd(a, b, tmin_simd<>);
}

// Dense243 SIMD maximum
static inline __m256i dense243_tmax_simd(__m256i a, __m256i b) {
    return dense243_binary_op_simd(a, b, tmax_simd<>);
}

// Dense243 SIMD negation
static inline __m256i dense243_tnot_simd(__m256i a) {
    return dense243_unary_op_simd(a, tnot_simd<>);
}

// =============================================================================
// Performance Notes
// =============================================================================
//
// INSTRUCTION COUNT per 32-byte SIMD block (160 trits):
//
// Binary operations (e.g., dense243_tadd_simd):
//   - Extraction:  10× _mm256_shuffle_epi8 = ~10 cycles
//   - Operations:   5× existing 2-bit SIMD = ~5 cycles
//   - Packing:      1× dense243_pack_simd = ~30 cycles (many additions)
//   - Total:       ~45 cycles per 32 bytes
//
// Comparison with direct 2-bit SIMD:
//   - Current:     ~1 cycle per 32 bytes (single shuffle)
//   - Dense243:   ~45 cycles per 32 bytes
//   - Slowdown:    45× in compute
//
// Memory bandwidth savings:
//   - 2-bit:      4 trits/byte
//   - Dense243:   5 trits/byte
//   - Savings:    20% less memory traffic
//
// BREAKEVEN ANALYSIS:
//   Compute cost increases 45×, memory decreases 20%
//   Only beneficial when memory bandwidth is severe bottleneck
//   (very large arrays on memory-starved systems)
//
// RECOMMENDATION:
//   Use Dense243 for:
//     - Storage/serialization (disk, network)
//     - Very large arrays (10M+ elements) on low-bandwidth systems
//
//   Use 2-bit SIMD for:
//     - Active computation
//     - Most array sizes
//     - Systems with good memory bandwidth
//
// HYBRID STRATEGY:
//   Best performance: Store in Dense243, transcode to 2-bit for compute,
//   transcode back for storage. Amortizes transcoding cost across multiple
//   operations on the same data.
//
// =============================================================================

#endif // TERNARY_DENSE243_SIMD_H
