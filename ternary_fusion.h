// ternary_fusion.h — Operation Fusion Infrastructure (Phase 4.0 PoC)
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
// PHASE 4.0: PROOF OF CONCEPT
// =============================================================================
//
// IMPLEMENTATION STATUS: Minimal viable implementation
//   - Single fused operation: tnot(tadd(a, b))
//   - Goal: Validate fusion concept with rigorous benchmarking
//   - Success criteria: Measurable speedup with statistical significance
//
// DESIGN PHILOSOPHY: Truth over claims
//   - Measure actual performance, not theoretical
//   - Report failures honestly
//   - Adapt design based on real-world results
//
// PERFORMANCE HYPOTHESIS (TO BE VALIDATED):
//   - Memory traffic reduction: 40% (5N → 3N bytes)
//   - Expected speedup: 1.5-2.5× on large arrays (>1M elements)
//   - Breakeven point: TBD via benchmarking
//
// =============================================================================

#ifndef TERNARY_FUSION_H
#define TERNARY_FUSION_H

#include "ternary_simd_kernels.h"
#include "ternary_algebra.h"

// =============================================================================
// PHASE 4.0: SINGLE FUSED OPERATION (Proof of Concept)
// =============================================================================

// Fused operation: tnot(tadd(a, b))
// Combines saturated addition with negation in single pass

// SIMD kernel (AVX2: 32 trits per vector)
template <bool Sanitize = true>
static inline __m256i fused_tnot_tadd_simd(__m256i a, __m256i b) {
    // Step 1: tadd_simd(a, b) - saturated addition
    __m256i temp = tadd_simd<Sanitize>(a, b);

    // Step 2: tnot_simd(temp) - negation
    // Note: temp stays in register, no memory store/load
    return tnot_simd<Sanitize>(temp);
}

// Scalar fallback (for tail processing)
static inline uint8_t fused_tnot_tadd_scalar(uint8_t a, uint8_t b) {
    // Both operations force-inline to single LUT lookups
    return tnot(tadd(a, b));
}

// =============================================================================
// INSTRUCTION COUNT ANALYSIS (Theoretical)
// =============================================================================
//
// UNFUSED (2 separate operations on 32 elements):
//   Operation 1 (tadd):
//     - Load A: 1 instruction (_mm256_loadu_si256)
//     - Load B: 1 instruction
//     - Compute: 1 instruction (_mm256_shuffle_epi8)
//     - Store temp: 1 instruction (_mm256_storeu_si256)
//     Total: 4 instructions
//
//   Operation 2 (tnot):
//     - Load temp: 1 instruction
//     - Compute: 1 instruction
//     - Store result: 1 instruction
//     Total: 3 instructions
//
//   Combined: 7 instructions, 5 memory ops (2 loads + 1 store + 1 load + 1 store)
//
// FUSED (single operation on 32 elements):
//   - Load A: 1 instruction
//   - Load B: 1 instruction
//   - Compute tadd: 1 instruction
//   - Compute tnot: 1 instruction (temp in register)
//   - Store result: 1 instruction
//   Total: 5 instructions, 3 memory ops (2 loads + 1 store)
//
// THEORETICAL GAIN:
//   - Instructions: 7 → 5 (-29%)
//   - Memory operations: 5 → 3 (-40%)
//   - Cache pollution: Intermediate array eliminated
//
// ACTUAL PERFORMANCE: TO BE MEASURED VIA BENCHMARKING
//
// =============================================================================

// =============================================================================
// MEMORY TRAFFIC ANALYSIS (Theoretical)
// =============================================================================
//
// For N-element arrays:
//
// UNFUSED:
//   tadd(A, B) → temp:
//     - Read A: N bytes
//     - Read B: N bytes
//     - Write temp: N bytes
//     Subtotal: 3N bytes
//
//   tnot(temp) → result:
//     - Read temp: N bytes
//     - Write result: N bytes
//     Subtotal: 2N bytes
//
//   Total: 5N bytes
//
// FUSED:
//   fused_tnot_tadd(A, B) → result:
//     - Read A: N bytes
//     - Read B: N bytes
//     - Write result: N bytes
//     Total: 3N bytes
//
// REDUCTION: 5N → 3N = 40% less memory traffic
//
// IMPACT ON PERFORMANCE:
//   - Small arrays (<100K): Compute-bound, minimal impact (~1.1-1.3×)
//   - Medium arrays (100K-1M): L3-bound, moderate impact (~1.5-1.8×)
//   - Large arrays (>1M): DRAM-bound, significant impact (~1.8-2.5×)
//
// HYPOTHESIS VALIDATION: See benchmarks/bench_fusion_poc.py
//
// =============================================================================

#endif // TERNARY_FUSION_H
