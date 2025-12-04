/**
 * load_profiling_optimized.cpp - Optimized Load Profile Classification
 *
 * Performance optimizations while maintaining correctness:
 * - Remove memset overhead via direct initialization
 * - Branchless bit manipulation for trit packing
 * - Better SIMD instruction scheduling
 * - Prefetching for reduced cache misses
 *
 * Target: 2.0 ms for 1M intervals (3× speedup vs C# baseline)
 *
 * Copyright (c) 2025 NavierLib Technologies
 */

#include "load_profiling.h"
#include "../core/algebra/ternary_algebra.h"
#include <immintrin.h>
#include <cmath>
#include <cstring>

// Ternary trit values (2-bit encoding)
#define TRIT_MINUS_ONE  0b00  // -1
#define TRIT_ZERO       0b01  //  0
#define TRIT_PLUS_ONE   0b10  // +1

/**
 * Optimized batch classification with branchless packing
 */
static void classify_batch_optimized(
    const double* consumption,
    const double* baseline,
    uint8_t* categories,
    int64_t count,
    double low_threshold,
    double high_threshold
) {
    __m256d low_thresh_vec = _mm256_set1_pd(low_threshold);
    __m256d high_thresh_vec = _mm256_set1_pd(high_threshold);

    // Process 4 doubles at a time (1 AVX2 register)
    int64_t simd_count = (count / 4) * 4;

    for (int64_t i = 0; i < simd_count; i += 4) {
        // Prefetch next iteration data (64 bytes ahead = 8 doubles)
        _mm_prefetch((const char*)&consumption[i + 8], _MM_HINT_T0);
        _mm_prefetch((const char*)&baseline[i + 8], _MM_HINT_T0);

        // Load consumption and baseline
        __m256d cons = _mm256_loadu_pd(&consumption[i]);
        __m256d base = _mm256_loadu_pd(&baseline[i]);

        // Compute ratio
        __m256d ratio = _mm256_div_pd(cons, base);

        // Compare against thresholds
        __m256d is_low = _mm256_cmp_pd(ratio, low_thresh_vec, _CMP_LT_OQ);
        __m256d is_high = _mm256_cmp_pd(ratio, high_thresh_vec, _CMP_GT_OQ);

        // Extract comparison masks as integer bits
        int low_mask = _mm256_movemask_pd(is_low);
        int high_mask = _mm256_movemask_pd(is_high);

        // Branchless trit computation
        // Formula: trit = (high_bit << 1) | (!low_bit & !high_bit)
        // Expanded: trit = (high_bit << 1) | ((~low_mask & ~high_mask) >> bit_pos) & 1)

        // Extract each bit and compute trit value
        // For bit position j: trit[j] = ((high_mask >> j) & 1) << 1) | ((~low_mask & ~high_mask) >> j) & 1)

        // Compute inverted masks once
        int not_low_or_high = ~low_mask & ~high_mask;

        // Extract and pack all 4 trits in parallel (branchless)
        uint8_t t0 = (((high_mask >> 0) & 1) << 1) | ((not_low_or_high >> 0) & 1);
        uint8_t t1 = (((high_mask >> 1) & 1) << 1) | ((not_low_or_high >> 1) & 1);
        uint8_t t2 = (((high_mask >> 2) & 1) << 1) | ((not_low_or_high >> 2) & 1);
        uint8_t t3 = (((high_mask >> 3) & 1) << 1) | ((not_low_or_high >> 3) & 1);

        // Pack into single byte: (t3 << 6) | (t2 << 4) | (t1 << 2) | t0
        // Direct assignment (no memset needed, no |= needed)
        categories[i / 4] = (t3 << 6) | (t2 << 4) | (t1 << 2) | t0;
    }

    // Handle tail elements (not divisible by 4)
    // Initialize first, then use |= for subsequent trits in same byte
    for (int64_t i = simd_count; i < count; i++) {
        double ratio = consumption[i] / baseline[i];

        uint8_t category;
        if (ratio < low_threshold) {
            category = TRIT_MINUS_ONE;
        } else if (ratio > high_threshold) {
            category = TRIT_PLUS_ONE;
        } else {
            category = TRIT_ZERO;
        }

        int byte_idx = i / 4;
        int trit_idx = i % 4;

        if (trit_idx == 0) {
            // First trit in byte: direct assignment
            categories[byte_idx] = category;
        } else {
            // Subsequent trits: OR into existing byte
            categories[byte_idx] |= (category << (trit_idx * 2));
        }
    }
}

/**
 * Optimized classification (replaces nv_classify_load_profile)
 */
int nv_classify_load_profile_optimized(
    const double* consumption,
    const double* baseline,
    uint8_t* categories,
    int64_t count,
    double low_ratio,
    double high_ratio
) {
    // Validate inputs
    if (!consumption || !baseline || !categories || count <= 0) {
        return -1;
    }

    if (low_ratio >= high_ratio) {
        return -2;
    }

    // NO memset - direct initialization in loop
    // Batch classify using optimized SIMD
    classify_batch_optimized(
        consumption,
        baseline,
        categories,
        count,
        low_ratio,
        high_ratio
    );

    return 0;
}

/**
 * Ultra-optimized: Process 8 doubles per iteration
 *
 * Doubles instruction-level parallelism by processing 2 AVX2 registers
 * simultaneously, allowing CPU to pipeline operations better.
 */
static void classify_batch_8wide(
    const double* consumption,
    const double* baseline,
    uint8_t* categories,
    int64_t count,
    double low_threshold,
    double high_threshold
) {
    __m256d low_thresh_vec = _mm256_set1_pd(low_threshold);
    __m256d high_thresh_vec = _mm256_set1_pd(high_threshold);

    // Process 8 doubles at a time (2 AVX2 registers)
    int64_t simd_count = (count / 8) * 8;

    for (int64_t i = 0; i < simd_count; i += 8) {
        // Prefetch 2 cache lines ahead
        _mm_prefetch((const char*)&consumption[i + 16], _MM_HINT_T0);
        _mm_prefetch((const char*)&baseline[i + 16], _MM_HINT_T0);

        // Load first 4 doubles
        __m256d cons0 = _mm256_loadu_pd(&consumption[i]);
        __m256d base0 = _mm256_loadu_pd(&baseline[i]);

        // Load second 4 doubles
        __m256d cons1 = _mm256_loadu_pd(&consumption[i + 4]);
        __m256d base1 = _mm256_loadu_pd(&baseline[i + 4]);

        // Compute ratios (pipeline-able)
        __m256d ratio0 = _mm256_div_pd(cons0, base0);
        __m256d ratio1 = _mm256_div_pd(cons1, base1);

        // Compare first register
        __m256d is_low0 = _mm256_cmp_pd(ratio0, low_thresh_vec, _CMP_LT_OQ);
        __m256d is_high0 = _mm256_cmp_pd(ratio0, high_thresh_vec, _CMP_GT_OQ);

        // Compare second register (pipeline-able)
        __m256d is_low1 = _mm256_cmp_pd(ratio1, low_thresh_vec, _CMP_LT_OQ);
        __m256d is_high1 = _mm256_cmp_pd(ratio1, high_thresh_vec, _CMP_GT_OQ);

        // Extract masks
        int low_mask0 = _mm256_movemask_pd(is_low0);
        int high_mask0 = _mm256_movemask_pd(is_high0);
        int low_mask1 = _mm256_movemask_pd(is_low1);
        int high_mask1 = _mm256_movemask_pd(is_high1);

        // Branchless packing for first 4 trits
        int not_low_or_high0 = ~low_mask0 & ~high_mask0;
        uint8_t byte0 =
            ((((high_mask0 >> 0) & 1) << 1) | ((not_low_or_high0 >> 0) & 1)) |
            ((((high_mask0 >> 1) & 1) << 1) | ((not_low_or_high0 >> 1) & 1)) << 2 |
            ((((high_mask0 >> 2) & 1) << 1) | ((not_low_or_high0 >> 2) & 1)) << 4 |
            ((((high_mask0 >> 3) & 1) << 1) | ((not_low_or_high0 >> 3) & 1)) << 6;

        // Branchless packing for second 4 trits
        int not_low_or_high1 = ~low_mask1 & ~high_mask1;
        uint8_t byte1 =
            ((((high_mask1 >> 0) & 1) << 1) | ((not_low_or_high1 >> 0) & 1)) |
            ((((high_mask1 >> 1) & 1) << 1) | ((not_low_or_high1 >> 1) & 1)) << 2 |
            ((((high_mask1 >> 2) & 1) << 1) | ((not_low_or_high1 >> 2) & 1)) << 4 |
            ((((high_mask1 >> 3) & 1) << 1) | ((not_low_or_high1 >> 3) & 1)) << 6;

        // Write 2 bytes (8 trits)
        categories[i / 4] = byte0;
        categories[i / 4 + 1] = byte1;
    }

    // Handle remaining elements with 4-wide SIMD
    int64_t remaining = count - simd_count;
    if (remaining >= 4) {
        classify_batch_optimized(
            &consumption[simd_count],
            &baseline[simd_count],
            &categories[simd_count / 4],
            remaining,
            low_threshold,
            high_threshold
        );
    } else {
        // Handle final 1-3 elements
        for (int64_t i = simd_count; i < count; i++) {
            double ratio = consumption[i] / baseline[i];

            uint8_t category;
            if (ratio < low_threshold) {
                category = TRIT_MINUS_ONE;
            } else if (ratio > high_threshold) {
                category = TRIT_PLUS_ONE;
            } else {
                category = TRIT_ZERO;
            }

            int byte_idx = i / 4;
            int trit_idx = i % 4;

            if (trit_idx == 0) {
                categories[byte_idx] = category;
            } else {
                categories[byte_idx] |= (category << (trit_idx * 2));
            }
        }
    }
}

/**
 * Ultra-optimized classification (8-wide SIMD)
 */
int nv_classify_load_profile_8wide(
    const double* consumption,
    const double* baseline,
    uint8_t* categories,
    int64_t count,
    double low_ratio,
    double high_ratio
) {
    // Validate inputs
    if (!consumption || !baseline || !categories || count <= 0) {
        return -1;
    }

    if (low_ratio >= high_ratio) {
        return -2;
    }

    // NO memset - direct initialization in loop
    classify_batch_8wide(
        consumption,
        baseline,
        categories,
        count,
        low_ratio,
        high_ratio
    );

    return 0;
}
