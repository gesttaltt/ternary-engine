/*
 * bench_memory.h - Memory Efficiency Benchmark Framework
 *
 * Copyright 2025 Ternary Engine Contributors
 * Licensed under the Apache License, Version 2.0
 *
 * PURPOSE:
 * Provides a C-compatible API for measuring memory efficiency of ternary
 * operations compared to standard numeric formats (INT8, INT4, FP16, FP32).
 *
 * KEY METRICS:
 * - Memory footprint (bytes per weight)
 * - Effective bandwidth (GB/s achieved)
 * - Ops per byte (computational density)
 * - Cache efficiency (throughput at different cache levels)
 *
 * TERNARY VALUE PROPOSITION:
 * - 2-bit encoding: 4 trits per byte
 * - Theoretical 4x compression vs INT8
 * - Theoretical 8x compression vs FP16
 * - Theoretical 16x compression vs FP32
 */

#ifndef BENCH_MEMORY_H
#define BENCH_MEMORY_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Numeric Format Definitions
 * ============================================================================ */

typedef enum {
    FORMAT_TERNARY_2BIT,    /* 2-bit ternary: {-1, 0, +1}, 4 values/byte */
    FORMAT_TERNARY_PACKED,  /* Dense243: 5 trits/byte (95.3% density) */
    FORMAT_BINARY_INT2,     /* 2-bit integer: {0,1,2,3}, 4 values/byte */
    FORMAT_BINARY_INT4,     /* 4-bit integer: 0-15, 2 values/byte */
    FORMAT_BINARY_INT8,     /* 8-bit integer: -128 to 127, 1 value/byte */
    FORMAT_BINARY_FP16,     /* 16-bit float: half precision, 0.5 values/byte */
    FORMAT_BINARY_FP32,     /* 32-bit float: single precision, 0.25 values/byte */
    FORMAT_BINARY_FP64,     /* 64-bit float: double precision, 0.125 values/byte */
} NumericFormat;

/* Bits per element for each format */
static const double FORMAT_BITS_PER_ELEMENT[] = {
    2.0,    /* TERNARY_2BIT: 2 bits */
    1.585,  /* TERNARY_PACKED: log2(243) = 7.92 bits for 5 trits = 1.585 bits/trit */
    2.0,    /* INT2: 2 bits */
    4.0,    /* INT4: 4 bits */
    8.0,    /* INT8: 8 bits */
    16.0,   /* FP16: 16 bits */
    32.0,   /* FP32: 32 bits */
    64.0,   /* FP64: 64 bits */
};

/* Human-readable names */
static const char* FORMAT_NAMES[] = {
    "Ternary-2bit",
    "Ternary-Dense243",
    "INT2",
    "INT4",
    "INT8",
    "FP16",
    "FP32",
    "FP64",
};

/* ============================================================================
 * Memory Efficiency Result
 * ============================================================================ */

typedef struct {
    /* Format identification */
    NumericFormat format;
    const char* format_name;

    /* Array configuration */
    size_t num_elements;            /* Number of logical elements */
    size_t bytes_allocated;         /* Actual bytes used */
    double bits_per_element;        /* Bits per logical element */

    /* Memory metrics */
    double compression_ratio;       /* vs FP32 baseline (32 bits/element) */
    double memory_footprint_mb;     /* Total memory in MB */

    /* Bandwidth metrics */
    double bandwidth_gbps;          /* Achieved bandwidth GB/s */
    double bandwidth_efficiency;    /* Achieved / Theoretical peak (0-1) */

    /* Computational density */
    double ops_per_byte;            /* Operations per byte transferred */
    double ops_per_second;          /* Total Gops/s */

    /* Cache behavior (throughput at different sizes) */
    double throughput_l1_gops;      /* Throughput for L1-resident data */
    double throughput_l2_gops;      /* Throughput for L2-resident data */
    double throughput_l3_gops;      /* Throughput for L3-resident data */
    double throughput_ram_gops;     /* Throughput for RAM-resident data */

} MemoryEfficiencyResult;

/* ============================================================================
 * Model Size Comparison
 * ============================================================================ */

typedef struct {
    const char* model_name;         /* e.g., "LLaMA-7B", "Phi-2" */
    size_t num_parameters;          /* Total parameters */

    /* Memory per format */
    double size_fp32_gb;            /* Size in FP32 */
    double size_fp16_gb;            /* Size in FP16 */
    double size_int8_gb;            /* Size in INT8 */
    double size_int4_gb;            /* Size in INT4 */
    double size_ternary_gb;         /* Size in Ternary-2bit */
    double size_dense243_gb;        /* Size in Dense243 */

    /* Compression ratios vs FP32 */
    double compression_fp16;        /* 2x */
    double compression_int8;        /* 4x */
    double compression_int4;        /* 8x */
    double compression_ternary;     /* 16x */
    double compression_dense243;    /* ~20x */

} ModelSizeComparison;

/* ============================================================================
 * Common Model Sizes (for reference)
 * ============================================================================ */

/* LLM model parameter counts */
#define MODEL_TINYLLAMA_PARAMS      1100000000ULL   /* 1.1B */
#define MODEL_PHI2_PARAMS           2700000000ULL   /* 2.7B */
#define MODEL_LLAMA_7B_PARAMS       7000000000ULL   /* 7B */
#define MODEL_LLAMA_13B_PARAMS     13000000000ULL   /* 13B */
#define MODEL_LLAMA_70B_PARAMS     70000000000ULL   /* 70B */

/* Calculate model size in GB for a given format */
#define CALC_MODEL_SIZE_GB(params, bits_per_elem) \
    ((double)(params) * (bits_per_elem) / 8.0 / 1024.0 / 1024.0 / 1024.0)

/* ============================================================================
 * Cache Size Definitions (typical desktop/laptop)
 * ============================================================================ */

#define CACHE_L1_SIZE_KB        32      /* Typical L1 data cache */
#define CACHE_L2_SIZE_KB        256     /* Typical L2 cache */
#define CACHE_L3_SIZE_MB        16      /* Typical L3 cache */

/* Array sizes for cache-level testing */
#define TEST_SIZE_L1            (CACHE_L1_SIZE_KB * 1024 / 4)       /* ~8K elements */
#define TEST_SIZE_L2            (CACHE_L2_SIZE_KB * 1024 / 4)       /* ~64K elements */
#define TEST_SIZE_L3            (CACHE_L3_SIZE_MB * 1024 * 1024 / 4) /* ~4M elements */
#define TEST_SIZE_RAM           (64 * 1024 * 1024)                   /* 64M elements */

/* ============================================================================
 * Bandwidth Calculation Helpers
 * ============================================================================ */

/* Calculate bytes transferred for a binary operation
 * read_a + read_b + write_result = 3 * elements * bytes_per_element */
#define CALC_BYTES_TRANSFERRED(elements, bytes_per_elem) \
    ((size_t)(elements) * (bytes_per_elem) * 3)

/* Calculate bandwidth in GB/s
 * bandwidth = bytes_transferred / time_seconds / 1e9 */
#define CALC_BANDWIDTH_GBPS(bytes, time_ns) \
    ((double)(bytes) / ((time_ns) / 1e9) / 1e9)

/* Calculate ops per byte
 * For binary operations: 1 op per element pair, 3 bytes transferred */
#define CALC_OPS_PER_BYTE(elements, bytes_transferred) \
    ((double)(elements) / (double)(bytes_transferred))

/* ============================================================================
 * Theoretical Peak Bandwidth (DDR4/DDR5)
 * ============================================================================ */

/* DDR4-3200: 25.6 GB/s per channel, dual channel = 51.2 GB/s */
#define DDR4_3200_PEAK_GBPS     51.2

/* DDR5-4800: 38.4 GB/s per channel, dual channel = 76.8 GB/s */
#define DDR5_4800_PEAK_GBPS     76.8

/* ============================================================================
 * Function Prototypes
 * ============================================================================ */

/* Calculate memory efficiency for a specific format and size */
MemoryEfficiencyResult bench_memory_efficiency(
    NumericFormat format,
    size_t num_elements,
    double throughput_gops,
    double time_ns
);

/* Compare memory usage across formats for a model size */
ModelSizeComparison bench_model_size_comparison(
    const char* model_name,
    size_t num_parameters
);

/* Print memory efficiency result */
void bench_print_memory_result(const MemoryEfficiencyResult* result);

/* Print model size comparison */
void bench_print_model_comparison(const ModelSizeComparison* comparison);

/* Export results to JSON */
void bench_export_memory_json(
    const MemoryEfficiencyResult* results,
    size_t num_results,
    const char* filename
);

#ifdef __cplusplus
}
#endif

#endif /* BENCH_MEMORY_H */
