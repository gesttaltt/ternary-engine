/**
 * navierlib_impl.cpp - NavierLib Implementation
 *
 * High-performance deterministic calculation engine for .NET
 * Copyright (c) 2025 NavierLib Technologies
 */

#include "navierlib_api.h"
#include <immintrin.h>
#include <cmath>
#include <cstring>

// Physical constants for natural gas calculations
namespace NavierConstants {
    // Standard conditions (Nm³)
    constexpr double T_STANDARD = 273.15;  // K (0°C)
    constexpr double P_STANDARD = 1.01325; // bar

    // Typical natural gas calorific value
    constexpr double CALORIFIC_VALUE_KWH_PER_NM3 = 10.55; // kWh/Nm³ (typical range: 9.5-11.5)

    // Gas constant
    constexpr double R_GAS = 8.314; // J/(mol·K)
}

// Thread-local error storage
static thread_local char g_last_error[256] = {0};

// Set error message
static void set_error(const char* msg) {
    strncpy_s(g_last_error, sizeof(g_last_error), msg, _TRUNCATE);
}

// Clear error
static void clear_error() {
    g_last_error[0] = '\0';
}

/**
 * CPU Feature Detection
 */
NAVIERLIB_API int nv_detect_cpu_features(void) {
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);

    int features = 0;

    // Check AVX2 (EAX=7, ECX=0, EBX bit 5)
    __cpuidex(cpuInfo, 7, 0);
    if (cpuInfo[1] & (1 << 5)) {
        features |= 0x01; // AVX2
    }

    // Check FMA (ECX bit 12)
    __cpuid(cpuInfo, 1);
    if (cpuInfo[2] & (1 << 12)) {
        features |= 0x02; // FMA
    }

    return features;
}

/**
 * Gas Volume Conversion Implementation (AVX2-accelerated)
 *
 * Physics:
 * V_actual = V_standard * (T_actual / T_standard) * (P_standard / P_actual) * Z
 * Energy = V_actual * CalorificValue
 *
 * Optimized with AVX2 FMA (Fused Multiply-Add)
 */
NAVIERLIB_API void nv_convert_gas_volume_batch(
    const double* input,
    double* output_kwh,
    int64_t count
) {
    clear_error();

    if (!input || !output_kwh || count <= 0) {
        set_error("Invalid input parameters");
        return;
    }

    // Check AVX2 availability
    int features = nv_detect_cpu_features();
    if (!(features & 0x01)) {
        set_error("AVX2 not supported on this CPU");
        return;
    }

    // Constants in SIMD registers
    __m256d t_std = _mm256_set1_pd(NavierConstants::T_STANDARD);
    __m256d p_std = _mm256_set1_pd(NavierConstants::P_STANDARD);
    __m256d calorific = _mm256_set1_pd(NavierConstants::CALORIFIC_VALUE_KWH_PER_NM3);
    __m256d celsius_to_kelvin = _mm256_set1_pd(273.15);

    // Process 4 records per iteration (AVX2 = 4 doubles)
    int64_t simd_count = count & ~3;  // Round down to multiple of 4

    for (int64_t i = 0; i < simd_count; i += 4) {
        // Load 4 records (16 doubles total)
        // Record format: [Volume, Temp, Pressure, Z]

        // Deinterleave into separate vectors
        __m256d vol   = _mm256_set_pd(input[i*4 + 12], input[i*4 + 8], input[i*4 + 4], input[i*4 + 0]);
        __m256d temp  = _mm256_set_pd(input[i*4 + 13], input[i*4 + 9], input[i*4 + 5], input[i*4 + 1]);
        __m256d press = _mm256_set_pd(input[i*4 + 14], input[i*4 + 10], input[i*4 + 6], input[i*4 + 2]);
        __m256d z     = _mm256_set_pd(input[i*4 + 15], input[i*4 + 11], input[i*4 + 7], input[i*4 + 3]);

        // Convert Celsius to Kelvin
        __m256d temp_k = _mm256_add_pd(temp, celsius_to_kelvin);

        // Temperature correction: T_actual / T_standard
        __m256d temp_ratio = _mm256_div_pd(temp_k, t_std);

        // Pressure correction: P_standard / P_actual
        __m256d press_ratio = _mm256_div_pd(p_std, press);

        // Combined correction: vol * temp_ratio * press_ratio * Z
        __m256d corrected_vol = _mm256_mul_pd(vol, temp_ratio);
        corrected_vol = _mm256_mul_pd(corrected_vol, press_ratio);
        corrected_vol = _mm256_mul_pd(corrected_vol, z);

        // Convert to kWh
        __m256d energy = _mm256_mul_pd(corrected_vol, calorific);

        // Store results
        _mm256_storeu_pd(&output_kwh[i], energy);
    }

    // Scalar tail for remaining records
    for (int64_t i = simd_count; i < count; ++i) {
        double vol = input[i * 4 + 0];
        double temp_c = input[i * 4 + 1];
        double press = input[i * 4 + 2];
        double z = input[i * 4 + 3];

        double temp_k = temp_c + 273.15;
        double temp_ratio = temp_k / NavierConstants::T_STANDARD;
        double press_ratio = NavierConstants::P_STANDARD / press;

        double corrected_vol = vol * temp_ratio * press_ratio * z;
        double energy = corrected_vol * NavierConstants::CALORIFIC_VALUE_KWH_PER_NM3;

        output_kwh[i] = energy;
    }
}

/**
 * 15-Minute Aggregation Implementation (AVX2-accelerated)
 *
 * Assumes input contains 900 records (15 minutes * 60 seconds) per output slot
 * Uses horizontal summation for maximum throughput
 */
NAVIERLIB_API void nv_aggregate_15min(
    const double* src,
    double* dst,
    int64_t count
) {
    clear_error();

    if (!src || !dst || count <= 0) {
        set_error("Invalid input parameters");
        return;
    }

    // 15-minute aggregation: 900 records → 1 output (for 1-second data)
    constexpr int64_t AGG_FACTOR = 900;
    int64_t output_count = count / AGG_FACTOR;

    if (count % AGG_FACTOR != 0) {
        set_error("Input count not multiple of aggregation factor (900)");
        return;
    }

    for (int64_t out_idx = 0; out_idx < output_count; ++out_idx) {
        int64_t start_idx = out_idx * AGG_FACTOR;

        // AVX2 horizontal sum
        __m256d sum_vec = _mm256_setzero_pd();

        // Process 4 doubles at a time
        int64_t simd_count = AGG_FACTOR & ~3;
        for (int64_t i = 0; i < simd_count; i += 4) {
            __m256d values = _mm256_loadu_pd(&src[start_idx + i]);
            sum_vec = _mm256_add_pd(sum_vec, values);
        }

        // Horizontal sum of vector
        __m128d sum_high = _mm256_extractf128_pd(sum_vec, 1);
        __m128d sum_low = _mm256_castpd256_pd128(sum_vec);
        __m128d sum_128 = _mm_add_pd(sum_high, sum_low);
        __m128d sum_64 = _mm_hadd_pd(sum_128, sum_128);

        double total = _mm_cvtsd_f64(sum_64);

        // Scalar tail
        for (int64_t i = simd_count; i < AGG_FACTOR; ++i) {
            total += src[start_idx + i];
        }

        dst[out_idx] = total;
    }
}

/**
 * Version Information
 */
NAVIERLIB_API const char* nv_get_version(void) {
    return "1.2.0";
}

/**
 * Error Handling
 */
NAVIERLIB_API const char* nv_get_last_error(void) {
    return g_last_error[0] ? g_last_error : nullptr;
}
