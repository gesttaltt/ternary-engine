/*
 * bench_throughput.h - C-compatible benchmark framework for Gops/s measurement
 *
 * Copyright 2025 Ternary Engine Contributors
 * Licensed under the Apache License, Version 2.0
 *
 * PURPOSE:
 * Provides a standardized API for measuring true kernel throughput in Gops/s
 * (billions of operations per second) without Python interpreter overhead.
 *
 * DESIGN PHILOSOPHY:
 * - Honest measurement: No self-deception, measure what actually runs
 * - Fair comparison: Ternary vs binary ops with equivalent workload
 * - Statistical rigor: Mean, stddev, min, max, percentiles
 * - Reproducibility: Deterministic RNG, documented methodology
 */

#ifndef BENCH_THROUGHPUT_H
#define BENCH_THROUGHPUT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Throughput Units
 * ============================================================================ */

typedef enum {
    THROUGHPUT_OPS_PER_SEC,     /* Raw operations/second */
    THROUGHPUT_MOPS,            /* Million ops/second (1e6) */
    THROUGHPUT_GOPS,            /* Billion ops/second (1e9) - PRIMARY METRIC */
    THROUGHPUT_TOPS,            /* Trillion ops/second (1e12) */
    THROUGHPUT_GBPS,            /* Memory bandwidth: GB/s */
} ThroughputUnit;

/* ============================================================================
 * Benchmark Result Structure
 * ============================================================================ */

typedef struct {
    /* Identification */
    const char* operation_name;     /* e.g., "ternary_tadd", "binary_int8_add" */
    const char* operation_class;    /* "ternary" or "binary" */
    size_t array_size;              /* Number of elements processed */
    size_t iterations;              /* Number of benchmark iterations */

    /* Timing (nanoseconds) */
    double total_time_ns;           /* Total benchmark time */
    double mean_time_ns;            /* Mean time per iteration */
    double stddev_time_ns;          /* Standard deviation */
    double min_time_ns;             /* Minimum (best case) */
    double max_time_ns;             /* Maximum (worst case) */
    double p50_time_ns;             /* Median */
    double p95_time_ns;             /* 95th percentile */
    double p99_time_ns;             /* 99th percentile */

    /* Throughput (PRIMARY METRICS) */
    double throughput_gops;         /* Billion operations/second */
    double throughput_elements_ps;  /* Elements/second */
    double memory_bandwidth_gbps;   /* Memory bandwidth GB/s */

    /* Efficiency metrics */
    double coefficient_of_variation; /* CV = stddev/mean (lower is more stable) */
    double ops_per_cycle_estimate;  /* Estimated ops/cycle (requires CPU freq) */

} BenchmarkResult;

/* ============================================================================
 * Comparison Result (Ternary vs Binary)
 * ============================================================================ */

typedef struct {
    BenchmarkResult ternary;        /* Ternary operation result */
    BenchmarkResult binary;         /* Binary baseline result */

    /* Comparison metrics */
    double throughput_ratio;        /* ternary_gops / binary_gops */
    double memory_efficiency_ratio; /* ternary_bw / binary_bw */
    const char* winner;             /* "ternary", "binary", or "tie" */
    const char* analysis;           /* Human-readable analysis */

} ComparisonResult;

/* ============================================================================
 * Benchmark Configuration
 * ============================================================================ */

typedef struct {
    /* Iteration control */
    size_t warmup_iterations;       /* Warmup runs (not measured) */
    size_t benchmark_iterations;    /* Measured iterations */

    /* Array sizes to test */
    size_t* sizes;                  /* Array of sizes */
    size_t num_sizes;               /* Number of sizes */

    /* Output options */
    int output_json;                /* 1 = JSON output */
    int output_csv;                 /* 1 = CSV output */
    int verbose;                    /* 1 = detailed output */

    /* Reproducibility */
    uint32_t random_seed;           /* RNG seed for data generation */

    /* Advanced options */
    int disable_turbo_check;        /* Skip turbo boost warning */
    double cpu_freq_ghz;            /* CPU frequency for ops/cycle (0 = auto) */

} BenchmarkConfig;

/* ============================================================================
 * Default Configuration
 * ============================================================================ */

#define BENCH_DEFAULT_WARMUP        100
#define BENCH_DEFAULT_ITERATIONS    1000
#define BENCH_DEFAULT_SEED          42

/* Default array sizes for comprehensive testing */
static const size_t BENCH_DEFAULT_SIZES[] = {
    32,           /* Single AVX2 vector */
    256,          /* L1 cache resident */
    4096,         /* L1 cache boundary */
    32768,        /* L2 cache resident */
    262144,       /* L2 cache boundary */
    1048576,      /* L3 cache resident (1M) */
    10485760,     /* Memory bound (10M) */
    104857600,    /* Large scale (100M) */
};
#define BENCH_DEFAULT_NUM_SIZES 8

/* ============================================================================
 * Throughput Calculation Macros
 * ============================================================================ */

/* Convert elements and time to Gops/s */
#define CALC_GOPS(elements, iterations, time_ns) \
    (((double)(elements) * (double)(iterations)) / ((time_ns) / 1e9) / 1e9)

/* Convert elements and time to memory bandwidth GB/s
 * bytes_per_element: 1 for int8, 2 for int16, 4 for int32, etc.
 * For ternary: effectively 0.25 bytes (2 bits), but stored as 1 byte
 * read_write_factor: 2 for read+write, 3 for read+read+write, etc. */
#define CALC_BANDWIDTH_GBPS(elements, iterations, time_ns, bytes_per_elem, rw_factor) \
    (((double)(elements) * (double)(iterations) * (bytes_per_elem) * (rw_factor)) / ((time_ns) / 1e9) / 1e9)

/* ============================================================================
 * Function Prototypes (implemented in bench_gops_comparative.cpp)
 * ============================================================================ */

/* Initialize benchmark configuration with defaults */
void bench_config_init(BenchmarkConfig* config);

/* Run a single benchmark */
BenchmarkResult bench_run_single(
    const char* name,
    void (*benchmark_fn)(const uint8_t*, const uint8_t*, uint8_t*, size_t),
    size_t array_size,
    const BenchmarkConfig* config
);

/* Run ternary vs binary comparison */
ComparisonResult bench_compare(
    const char* ternary_name,
    void (*ternary_fn)(const uint8_t*, const uint8_t*, uint8_t*, size_t),
    const char* binary_name,
    void (*binary_fn)(const uint8_t*, const uint8_t*, uint8_t*, size_t),
    size_t array_size,
    const BenchmarkConfig* config
);

/* Print result in human-readable format */
void bench_print_result(const BenchmarkResult* result, ThroughputUnit unit);

/* Print comparison in human-readable format */
void bench_print_comparison(const ComparisonResult* result);

/* Export results to JSON */
void bench_export_json(const BenchmarkResult* results, size_t num_results, const char* filename);

/* Export results to CSV */
void bench_export_csv(const BenchmarkResult* results, size_t num_results, const char* filename);

#ifdef __cplusplus
}
#endif

#endif /* BENCH_THROUGHPUT_H */
