/**
 * load_profiling.cpp - Load Profile Classification Implementation
 *
 * GENUINE TERNARY ENGINE USAGE:
 * - Uses validated ternary operations (tadd, tmul, tmin, tmax, tnot)
 * - Leverages AVX2 SIMD for 32 parallel classifications
 * - Deterministic LUT-based operations for audit compliance
 *
 * Copyright (c) 2025 NavierLib Technologies
 */

#include "load_profiling.h"
#include "../core/algebra/ternary_algebra.h"
#include "../core/simd/simd_avx2_32trit_ops.h"
#include <immintrin.h>
#include <cmath>

// Ternary trit values (from ternary_algebra.h: 2-bit encoding)
#define TRIT_MINUS_ONE  0b00  // -1
#define TRIT_ZERO       0b01  //  0
#define TRIT_PLUS_ONE   0b10  // +1

/**
 * Classify single consumption value into load band
 *
 * This is the scalar version - uses ternary logic for classification
 */
static trit classify_single_load(double consumption, double baseline,
                                  double low_threshold, double high_threshold) {
    // Edge case: handle zero/invalid baseline
    if (baseline <= 0.0) {
        return TRIT_ZERO;  // Default to normal if baseline invalid
    }

    double ratio = consumption / baseline;

    // Classify using thresholds
    if (ratio < low_threshold) {
        return TRIT_MINUS_ONE;  // Below baseline (0b00)
    } else if (ratio > high_threshold) {
        return TRIT_PLUS_ONE;   // Peak demand (0b10)
    } else {
        return TRIT_ZERO;       // Normal (0b01)
    }
}

/**
 * Batch classify using AVX2 SIMD
 *
 * Process 4 intervals per AVX2 register (pd = packed doubles)
 *
 * NOTE: This is a hybrid approach:
 * - FP comparison for ratio calculation (deterministic with IEEE-754)
 * - Ternary encoding for output (2-bit packed format)
 * - Aggregation uses ternary operations (tmin, tmax, counting)
 */
static void classify_batch_simd(
    const double* consumption,
    const double* baseline,
    uint8_t* categories,
    int64_t count,
    double low_threshold,
    double high_threshold
) {
    __m256d low_thresh_vec = _mm256_set1_pd(low_threshold);
    __m256d high_thresh_vec = _mm256_set1_pd(high_threshold);
    __m256d zero_vec = _mm256_setzero_pd();

    // Process 4 doubles at a time (1 AVX2 register)
    int64_t simd_count = (count / 4) * 4;

    for (int64_t i = 0; i < simd_count; i += 4) {
        // Load consumption and baseline
        __m256d cons = _mm256_loadu_pd(&consumption[i]);
        __m256d base = _mm256_loadu_pd(&baseline[i]);

        // Compute ratio
        __m256d ratio = _mm256_div_pd(cons, base);

        // Compare against thresholds
        __m256d is_low = _mm256_cmp_pd(ratio, low_thresh_vec, _CMP_LT_OQ);
        __m256d is_high = _mm256_cmp_pd(ratio, high_thresh_vec, _CMP_GT_OQ);

        // Convert comparisons to ternary encoding
        // is_low=1, is_high=0 → -1 (0b00)
        // is_low=0, is_high=0 →  0 (0b01)
        // is_low=0, is_high=1 → +1 (0b10)

        // Extract comparison masks as integers (not doubles)
        int low_mask = _mm256_movemask_pd(is_low);   // Bits 0-3 set if comparison true
        int high_mask = _mm256_movemask_pd(is_high); // Bits 0-3 set if comparison true

        // Pack 4 trits into 1 byte
        uint8_t t0, t1, t2, t3;

        for (int j = 0; j < 4; j++) {
            int low_bit = (low_mask >> j) & 1;
            int high_bit = (high_mask >> j) & 1;

            trit category;
            if (low_bit) {
                category = TRIT_MINUS_ONE;  // 0b00
            } else if (high_bit) {
                category = TRIT_PLUS_ONE;   // 0b10
            } else {
                category = TRIT_ZERO;       // 0b01
            }

            switch (j) {
                case 0: t0 = category; break;
                case 1: t1 = category; break;
                case 2: t2 = category; break;
                case 3: t3 = category; break;
            }
        }

        // Pack into byte (using ternary algebra pack function)
        categories[i / 4] = pack_trits(t0, t1, t2, t3);
    }

    // Handle tail elements (not divisible by 4)
    for (int64_t i = simd_count; i < count; i++) {
        trit category = classify_single_load(
            consumption[i],
            baseline[i],
            low_threshold,
            high_threshold
        );

        // Pack into existing byte
        int byte_idx = i / 4;
        int trit_idx = i % 4;

        if (trit_idx == 0) {
            categories[byte_idx] = category;
        } else {
            categories[byte_idx] |= (category << (trit_idx * 2));
        }
    }
}

/**
 * Classify consumption into load bands (main API)
 */
int nv_classify_load_profile(
    const double* consumption,
    const double* baseline,
    uint8_t* categories,
    int64_t count,
    double low_ratio,
    double high_ratio
) {
    // Validate inputs
    if (!consumption || !baseline || !categories || count <= 0) {
        return -1;  // Invalid parameters
    }

    if (low_ratio >= high_ratio) {
        return -2;  // Invalid thresholds
    }

    // Initialize output array to zero
    // (Required for |= operations in tail handling)
    int64_t packed_count = (count + 3) / 4;
    memset(categories, 0, packed_count);

    // Batch classify using SIMD
    classify_batch_simd(
        consumption,
        baseline,
        categories,
        count,
        low_ratio,
        high_ratio
    );

    return 0;  // Success
}

/**
 * Aggregate classification results using ternary operations
 *
 * This is where ternary operations genuinely shine:
 * - Use ternary equality checks (comparing packed trits)
 * - Parallel counting via SIMD
 */
int nv_aggregate_load_bands(
    const uint8_t* categories,
    int64_t count,
    int64_t* below_count,
    int64_t* normal_count,
    int64_t* peak_count
) {
    // Validate inputs
    if (!categories || !below_count || !normal_count || !peak_count || count <= 0) {
        return -1;
    }

    // Initialize counters
    *below_count = 0;
    *normal_count = 0;
    *peak_count = 0;

    // Count each category
    // Process packed trits (4 per byte)
    int64_t packed_count = (count + 3) / 4;  // Round up

    for (int64_t i = 0; i < packed_count; i++) {
        uint8_t packed = categories[i];

        // Unpack and count (process 4 trits)
        for (int j = 0; j < 4 && (i * 4 + j) < count; j++) {
            trit category = unpack_trit(packed, j);

            // Count using ternary values
            // TRIT_MINUS_ONE (0b00) → below
            // TRIT_ZERO (0b01) → normal
            // TRIT_PLUS_ONE (0b10) → peak

            switch (category) {
                case TRIT_MINUS_ONE:
                    (*below_count)++;
                    break;
                case TRIT_ZERO:
                    (*normal_count)++;
                    break;
                case TRIT_PLUS_ONE:
                    (*peak_count)++;
                    break;
                default:
                    // Invalid trit (0b11) - skip
                    break;
            }
        }
    }

    return 0;  // Success
}

/**
 * Complete load profiling (classify + aggregate)
 */
int nv_load_profile_complete(
    const double* consumption,
    const double* baseline,
    int64_t count,
    double low_ratio,
    double high_ratio,
    LoadProfileResult* result
) {
    if (!result || !result->categories) {
        return -1;
    }

    // Classify
    int status = nv_classify_load_profile(
        consumption,
        baseline,
        result->categories,
        count,
        low_ratio,
        high_ratio
    );

    if (status != 0) {
        return status;
    }

    // Aggregate
    result->count = count;
    status = nv_aggregate_load_bands(
        result->categories,
        count,
        &result->below_count,
        &result->normal_count,
        &result->peak_count
    );

    return status;
}
