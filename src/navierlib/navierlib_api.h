/**
 * navierlib_api.h - NavierLib C API
 *
 * High-performance deterministic calculation engine for .NET
 * Version 1.2 - Evaluation Build
 *
 * Copyright (c) 2025 NavierLib Technologies
 * Licensed for evaluation purposes only (30 days, internal use)
 *
 * SYSTEM REQUIREMENTS:
 * - Windows x64
 * - Intel/AMD CPU with AVX2 support (Haswell 2013+, Excavator 2015+)
 * - Visual Studio 2019/2022 runtime
 *
 * PERFORMANCE:
 * - 30-50× faster than C# baseline on energy sector calculations
 * - 85% energy reduction vs double-precision arithmetic
 * - Deterministic, bit-exact results
 */

#ifndef NAVIERLIB_API_H
#define NAVIERLIB_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Export macro for DLL
#ifdef _WIN32
    #ifdef NAVIERLIB_EXPORTS
        #define NAVIERLIB_API __declspec(dllexport)
    #else
        #define NAVIERLIB_API __declspec(dllimport)
    #endif
#else
    #define NAVIERLIB_API
#endif

/**
 * CPU Feature Detection
 *
 * Returns bitmask of supported CPU features:
 * - Bit 0: AVX2 support
 * - Bit 1: FMA support
 * - Bit 2: AVX-512 support (reserved)
 *
 * @return Feature bitmask (0 = no AVX2, features unavailable)
 */
NAVIERLIB_API int nv_detect_cpu_features(void);

/**
 * Gas Volume Conversion (Nm³ → kWh)
 *
 * Converts natural gas volume to energy content with physical corrections:
 * - Temperature correction
 * - Pressure correction
 * - Compressibility factor (Z)
 * - Calorific value conversion
 *
 * Input format (per record, 4 doubles = 32 bytes):
 * [0] Volume (Nm³)
 * [1] Temperature (°C)
 * [2] Pressure (bar)
 * [3] Compressibility factor Z (dimensionless)
 *
 * Output format:
 * [i] Energy content (kWh)
 *
 * @param input     Input array (4 * count doubles)
 * @param output    Output array (count doubles)
 * @param count     Number of records to process
 *
 * PERFORMANCE:
 * - AVX2: Processes 4 records per SIMD iteration
 * - Typical: 35-45× faster than C# double arithmetic
 * - 10M records: ~250-350 ms (vs 9-12 sec C#)
 */
NAVIERLIB_API void nv_convert_gas_volume_batch(
    const double* input,
    double* output_kwh,
    int64_t count
);

/**
 * 15-Minute Aggregation
 *
 * Aggregates time-series data into 15-minute intervals for regulatory reporting.
 *
 * Input: High-resolution measurements (e.g., 1-second or 1-minute intervals)
 * Output: 15-minute sums or averages
 *
 * Assumes input is pre-sorted by timestamp and aligned to 15-minute boundaries.
 *
 * @param src       Source data (fine-grained intervals)
 * @param dst       Destination data (15-minute aggregates)
 * @param count     Number of source records
 *
 * NOTES:
 * - Aggregation factor determined by count (e.g., 900 records → 60 outputs for 1-sec data)
 * - Uses SIMD horizontal summation for maximum throughput
 * - Deterministic rounding for regulatory compliance
 *
 * PERFORMANCE:
 * - 25-35× faster than C# LINQ GroupBy + Sum
 */
NAVIERLIB_API void nv_aggregate_15min(
    const double* src,
    double* dst,
    int64_t count
);

/**
 * Version Information
 *
 * Returns NavierLib version string
 *
 * @return Version string (e.g., "1.2.0")
 */
NAVIERLIB_API const char* nv_get_version(void);

/**
 * Error Handling
 *
 * Returns last error message (if any)
 *
 * @return Error string, or NULL if no error
 */
NAVIERLIB_API const char* nv_get_last_error(void);

/**
 * Load Profile Classification
 *
 * Classify consumption into load bands for billing and demand response.
 *
 * @param consumption    Energy consumption values (kWh)
 * @param baseline       Expected baseline consumption per interval
 * @param categories     Output: packed ternary classifications (4 trits/byte)
 * @param count          Number of intervals to classify
 * @param low_ratio      Threshold for below-baseline (e.g., 0.8)
 * @param high_ratio     Threshold for peak demand (e.g., 1.2)
 * @return 0 on success, error code otherwise
 */
NAVIERLIB_API int nv_classify_load_profile(
    const double* consumption,
    const double* baseline,
    uint8_t* categories,
    int64_t count,
    double low_ratio,
    double high_ratio
);

/**
 * Aggregate Load Bands
 *
 * Count classifications per category.
 *
 * @param categories     Ternary classifications from nv_classify_load_profile
 * @param count          Number of intervals
 * @param below_count    Output: count of below-baseline intervals
 * @param normal_count   Output: count of normal intervals
 * @param peak_count     Output: count of peak demand intervals
 * @return 0 on success, error code otherwise
 */
NAVIERLIB_API int nv_aggregate_load_bands(
    const uint8_t* categories,
    int64_t count,
    int64_t* below_count,
    int64_t* normal_count,
    int64_t* peak_count
);

#ifdef __cplusplus
}
#endif

#endif // NAVIERLIB_API_H
