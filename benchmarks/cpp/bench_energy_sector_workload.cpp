/**
 * bench_energy_sector_workload.cpp - eBase/Eneva Production Workload Benchmark
 *
 * Copyright (c) 2025 NavierLib Technologies
 * Licensed under the Apache License, Version 2.0
 *
 * PURPOSE:
 * Validate NavierLib performance for real-world energy sector workloads:
 * - End-to-end workflow (classify + aggregate)
 * - Realistic consumption patterns (residential/commercial/industrial)
 * - Multiple scales (hourly, daily, monthly, annual batches)
 * - Determinism validation (billing compliance)
 * - Memory efficiency measurement
 *
 * TARGET: eBase (energy management platform) and Eneva (Brazilian utility)
 *
 * USAGE:
 *   python benchmarks/cpp/build_energy_sector_bench.py
 *   dist/EnergySectorBench/EnergySectorWorkload.exe
 */

#include "../../src/navierlib/load_profiling.h"
#include "../../src/core/algebra/ternary_algebra.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

// =============================================================================
// Test Configuration
// =============================================================================

#define HOURLY_BATCH      400000    // 4 intervals × 100K customers
#define DAILY_BATCH       1000000   // Quick daily batch
#define MONTHLY_BATCH     10000000  // 10M for testing (full: 2.88B too large)
#define ANNUAL_BATCH      100000000 // 100M for stress testing

#define WARMUP_ITERATIONS 3
#define BENCHMARK_ITERATIONS 10
#define DETERMINISM_ITERATIONS 1000

#define LOW_THRESHOLD 0.8
#define HIGH_THRESHOLD 1.2

// =============================================================================
// Energy Consumption Pattern Generators
// =============================================================================

/**
 * Residential pattern: 25% below, 60% normal, 15% peak
 * Typical household with evening peaks, weekend off-peaks
 */
static void generate_residential_pattern(
    double* consumption,
    double* baseline,
    int64_t count,
    unsigned int seed
) {
    srand(seed);

    for (int64_t i = 0; i < count; i++) {
        // Baseline varies ±10%
        baseline[i] = 100.0 * (0.9 + ((double)rand() / RAND_MAX) * 0.2);

        double pattern = (double)rand() / RAND_MAX;

        if (pattern < 0.25) {
            // Below baseline (night, weekends)
            consumption[i] = baseline[i] * (0.5 + ((double)rand() / RAND_MAX) * 0.3);
        } else if (pattern < 0.85) {
            // Normal (typical usage)
            consumption[i] = baseline[i] * (0.8 + ((double)rand() / RAND_MAX) * 0.4);
        } else {
            // Peak (evening, AC usage)
            consumption[i] = baseline[i] * (1.2 + ((double)rand() / RAND_MAX) * 0.8);
        }
    }
}

/**
 * Commercial pattern: 10% below, 75% normal, 15% peak
 * Businesses with consistent weekday usage, lower weekend
 */
static void generate_commercial_pattern(
    double* consumption,
    double* baseline,
    int64_t count,
    unsigned int seed
) {
    srand(seed);

    for (int64_t i = 0; i < count; i++) {
        baseline[i] = 100.0 * (0.9 + ((double)rand() / RAND_MAX) * 0.2);

        double pattern = (double)rand() / RAND_MAX;

        if (pattern < 0.10) {
            // Below (weekends, holidays)
            consumption[i] = baseline[i] * (0.5 + ((double)rand() / RAND_MAX) * 0.3);
        } else if (pattern < 0.85) {
            // Normal (business hours)
            consumption[i] = baseline[i] * (0.8 + ((double)rand() / RAND_MAX) * 0.4);
        } else {
            // Peak (summer cooling, winter heating)
            consumption[i] = baseline[i] * (1.2 + ((double)rand() / RAND_MAX) * 0.8);
        }
    }
}

/**
 * Industrial pattern: 5% below, 85% normal, 10% peak
 * Factories with continuous operation, occasional peaks
 */
static void generate_industrial_pattern(
    double* consumption,
    double* baseline,
    int64_t count,
    unsigned int seed
) {
    srand(seed);

    for (int64_t i = 0; i < count; i++) {
        baseline[i] = 100.0 * (0.9 + ((double)rand() / RAND_MAX) * 0.2);

        double pattern = (double)rand() / RAND_MAX;

        if (pattern < 0.05) {
            // Below (shutdown periods)
            consumption[i] = baseline[i] * (0.5 + ((double)rand() / RAND_MAX) * 0.3);
        } else if (pattern < 0.90) {
            // Normal (continuous operation)
            consumption[i] = baseline[i] * (0.8 + ((double)rand() / RAND_MAX) * 0.4);
        } else {
            // Peak (production peaks)
            consumption[i] = baseline[i] * (1.2 + ((double)rand() / RAND_MAX) * 0.8);
        }
    }
}

/**
 * Mixed pattern: 20% below, 65% normal, 15% peak
 * Weighted average of residential (60%), commercial (30%), industrial (10%)
 */
static void generate_mixed_pattern(
    double* consumption,
    double* baseline,
    int64_t count,
    unsigned int seed
) {
    srand(seed);

    for (int64_t i = 0; i < count; i++) {
        baseline[i] = 100.0 * (0.9 + ((double)rand() / RAND_MAX) * 0.2);

        double pattern = (double)rand() / RAND_MAX;

        if (pattern < 0.20) {
            // Below
            consumption[i] = baseline[i] * (0.5 + ((double)rand() / RAND_MAX) * 0.3);
        } else if (pattern < 0.85) {
            // Normal
            consumption[i] = baseline[i] * (0.8 + ((double)rand() / RAND_MAX) * 0.4);
        } else {
            // Peak
            consumption[i] = baseline[i] * (1.2 + ((double)rand() / RAND_MAX) * 0.8);
        }
    }
}

// =============================================================================
// Benchmark Result Structure
// =============================================================================

typedef struct {
    double classification_ms;
    double aggregation_ms;
    double total_ms;
    double throughput_millions_per_sec;
    int64_t below_count;
    int64_t normal_count;
    int64_t peak_count;
    double below_pct;
    double normal_pct;
    double peak_pct;
} WorkloadResult;

// =============================================================================
// End-to-End Workflow Benchmark
// =============================================================================

WorkloadResult benchmark_workload(
    const char* pattern_name,
    double* consumption,
    double* baseline,
    uint8_t* categories,
    int64_t count
) {
    WorkloadResult result;

    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        nv_classify_load_profile(
            consumption, baseline, categories,
            count, LOW_THRESHOLD, HIGH_THRESHOLD
        );
    }

    // Benchmark classification
    clock_t class_start = clock();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        nv_classify_load_profile(
            consumption, baseline, categories,
            count, LOW_THRESHOLD, HIGH_THRESHOLD
        );
    }
    clock_t class_end = clock();
    result.classification_ms = ((double)(class_end - class_start) / CLOCKS_PER_SEC) / BENCHMARK_ITERATIONS * 1000;

    // Benchmark aggregation
    clock_t agg_start = clock();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        nv_aggregate_load_bands(
            categories, count,
            &result.below_count,
            &result.normal_count,
            &result.peak_count
        );
    }
    clock_t agg_end = clock();
    result.aggregation_ms = ((double)(agg_end - agg_start) / CLOCKS_PER_SEC) / BENCHMARK_ITERATIONS * 1000;

    // Calculate total and throughput
    result.total_ms = result.classification_ms + result.aggregation_ms;
    result.throughput_millions_per_sec = count / result.total_ms / 1000.0;

    // Calculate percentages
    result.below_pct = 100.0 * result.below_count / count;
    result.normal_pct = 100.0 * result.normal_count / count;
    result.peak_pct = 100.0 * result.peak_count / count;

    return result;
}

// =============================================================================
// Determinism Test
// =============================================================================

typedef struct {
    int passed;
    int failed;
    double variance_pct;
} DeterminismResult;

DeterminismResult test_determinism(
    double* consumption,
    double* baseline,
    int64_t count,
    int iterations
) {
    DeterminismResult result = {0, 0, 0.0};

    // Allocate reference and test arrays
    int64_t packed_count = (count + 3) / 4;
    uint8_t* reference = (uint8_t*)malloc(packed_count);
    uint8_t* test = (uint8_t*)malloc(packed_count);

    // First run as reference
    nv_classify_load_profile(
        consumption, baseline, reference,
        count, LOW_THRESHOLD, HIGH_THRESHOLD
    );

    // Compare against multiple runs
    for (int i = 0; i < iterations; i++) {
        nv_classify_load_profile(
            consumption, baseline, test,
            count, LOW_THRESHOLD, HIGH_THRESHOLD
        );

        // Compare byte-by-byte
        int mismatches = 0;
        for (int64_t j = 0; j < packed_count; j++) {
            if (reference[j] != test[j]) {
                mismatches++;
            }
        }

        if (mismatches == 0) {
            result.passed++;
        } else {
            result.failed++;
        }
    }

    result.variance_pct = 100.0 * result.failed / iterations;

    free(reference);
    free(test);

    return result;
}

// =============================================================================
// Main Benchmark Suite
// =============================================================================

int main() {
    printf("\n");
    printf("========================================================================\n");
    printf("  Energy Sector Workload Benchmark - eBase/Eneva\n");
    printf("========================================================================\n");
    printf("Platform:   Windows x64\n");
    printf("Compiler:   MSVC /O2 /arch:AVX2\n");
    printf("Target:     Load profiling for energy utilities\n");
    printf("\n");

    // =======================================================================
    // Test 1: Realistic Data Patterns
    // =======================================================================

    printf("========================================================================\n");
    printf("  Test 1: Realistic Energy Consumption Patterns\n");
    printf("========================================================================\n");
    printf("Testing workflow: classify + aggregate\n");
    printf("Size: 1M intervals (typical daily batch)\n");
    printf("\n");

    int64_t test_size = DAILY_BATCH;

    // Allocate arrays
    double* consumption = (double*)malloc(test_size * sizeof(double));
    double* baseline = (double*)malloc(test_size * sizeof(double));
    uint8_t* categories = (uint8_t*)malloc((test_size + 3) / 4);

    if (!consumption || !baseline || !categories) {
        printf("ERROR: Memory allocation failed\n");
        return 1;
    }

    printf("Pattern              |  Classify |  Aggregate |    Total | Throughput | Distribution\n");
    printf("---------------------|-----------|------------|----------|------------|------------------\n");

    // Test residential pattern
    generate_residential_pattern(consumption, baseline, test_size, 42);
    WorkloadResult res = benchmark_workload("Residential", consumption, baseline, categories, test_size);
    printf("Residential (25/60/15)| %7.2f ms| %8.2f ms| %7.2f ms| %7.2f M/s| %5.1f/%5.1f/%5.1f\n",
           res.classification_ms, res.aggregation_ms, res.total_ms,
           res.throughput_millions_per_sec,
           res.below_pct, res.normal_pct, res.peak_pct);

    // Test commercial pattern
    generate_commercial_pattern(consumption, baseline, test_size, 42);
    res = benchmark_workload("Commercial", consumption, baseline, categories, test_size);
    printf("Commercial (10/75/15) | %7.2f ms| %8.2f ms| %7.2f ms| %7.2f M/s| %5.1f/%5.1f/%5.1f\n",
           res.classification_ms, res.aggregation_ms, res.total_ms,
           res.throughput_millions_per_sec,
           res.below_pct, res.normal_pct, res.peak_pct);

    // Test industrial pattern
    generate_industrial_pattern(consumption, baseline, test_size, 42);
    res = benchmark_workload("Industrial", consumption, baseline, categories, test_size);
    printf("Industrial (5/85/10)  | %7.2f ms| %8.2f ms| %7.2f ms| %7.2f M/s| %5.1f/%5.1f/%5.1f\n",
           res.classification_ms, res.aggregation_ms, res.total_ms,
           res.throughput_millions_per_sec,
           res.below_pct, res.normal_pct, res.peak_pct);

    // Test mixed pattern
    generate_mixed_pattern(consumption, baseline, test_size, 42);
    res = benchmark_workload("Mixed", consumption, baseline, categories, test_size);
    printf("Mixed (20/65/15)      | %7.2f ms| %8.2f ms| %7.2f ms| %7.2f M/s| %5.1f/%5.1f/%5.1f\n",
           res.classification_ms, res.aggregation_ms, res.total_ms,
           res.throughput_millions_per_sec,
           res.below_pct, res.normal_pct, res.peak_pct);

    printf("\n");

    // =======================================================================
    // Test 2: Scale Testing
    // =======================================================================

    printf("========================================================================\n");
    printf("  Test 2: Scale Testing (Mixed Pattern)\n");
    printf("========================================================================\n");
    printf("Validating linear scaling across workload sizes\n");
    printf("\n");

    printf("Size       | Batch Type        |  Classify |  Aggregate |    Total | Throughput\n");
    printf("-----------|-------------------|-----------|------------|----------|------------\n");

    // Test scales
    int64_t scales[] = {HOURLY_BATCH, DAILY_BATCH, MONTHLY_BATCH, ANNUAL_BATCH};
    const char* batch_names[] = {"Hourly (real-time)", "Daily", "Monthly (partial)", "Annual (stress)"};
    int num_scales = 4;

    for (int s = 0; s < num_scales; s++) {
        int64_t size = scales[s];

        // Reallocate arrays if needed
        if (size != test_size) {
            free(consumption);
            free(baseline);
            free(categories);

            consumption = (double*)malloc(size * sizeof(double));
            baseline = (double*)malloc(size * sizeof(double));
            categories = (uint8_t*)malloc((size + 3) / 4);

            if (!consumption || !baseline || !categories) {
                printf("ERROR: Memory allocation failed for size %lld\n", size);
                return 1;
            }

            test_size = size;
        }

        generate_mixed_pattern(consumption, baseline, size, 42);
        res = benchmark_workload("Mixed", consumption, baseline, categories, size);

        printf("%10lld | %-17s | %7.2f ms| %8.2f ms| %7.2f ms| %7.2f M/s\n",
               size, batch_names[s],
               res.classification_ms, res.aggregation_ms, res.total_ms,
               res.throughput_millions_per_sec);
    }

    printf("\n");

    // =======================================================================
    // Test 3: Determinism Validation (Billing Compliance)
    // =======================================================================

    printf("========================================================================\n");
    printf("  Test 3: Determinism Validation (EU/Brazil Billing Compliance)\n");
    printf("========================================================================\n");
    printf("Testing: %d iterations with identical input\n", DETERMINISM_ITERATIONS);
    printf("Requirement: 0%% variance (all outputs bit-identical)\n");
    printf("\n");

    // Use 1M intervals for determinism test
    int64_t det_size = DAILY_BATCH;
    if (test_size != det_size) {
        free(consumption);
        free(baseline);
        free(categories);

        consumption = (double*)malloc(det_size * sizeof(double));
        baseline = (double*)malloc(det_size * sizeof(double));
        categories = (uint8_t*)malloc((det_size + 3) / 4);
    }

    generate_mixed_pattern(consumption, baseline, det_size, 42);

    DeterminismResult det_result = test_determinism(consumption, baseline, det_size, DETERMINISM_ITERATIONS);

    printf("Iterations:     %d\n", DETERMINISM_ITERATIONS);
    printf("Passed:         %d\n", det_result.passed);
    printf("Failed:         %d\n", det_result.failed);
    printf("Variance:       %.4f%%\n", det_result.variance_pct);
    printf("\n");

    if (det_result.variance_pct == 0.0) {
        printf("✓ PASS: 100%% determinism (billing compliant)\n");
    } else {
        printf("✗ FAIL: Non-deterministic output detected\n");
    }

    printf("\n");

    // =======================================================================
    // Test 4: Memory Efficiency
    // =======================================================================

    printf("========================================================================\n");
    printf("  Test 4: Memory Efficiency\n");
    printf("========================================================================\n");

    int64_t mem_size = DAILY_BATCH;
    double input_mb = (mem_size * 2 * sizeof(double)) / (1024.0 * 1024.0);
    double output_mb = ((mem_size + 3) / 4) / (1024.0 * 1024.0);
    double total_mb = input_mb + output_mb;

    printf("Array size:     %lld intervals\n", mem_size);
    printf("Input:          %.2f MB (2× double for consumption/baseline)\n", input_mb);
    printf("Output:         %.2f MB (packed trits, 4 intervals/byte)\n", output_mb);
    printf("Total working:  %.2f MB\n", total_mb);
    printf("Compression:    %.1f× (vs 2× double output)\n", input_mb / output_mb);
    printf("\n");

    // =======================================================================
    // Summary
    // =======================================================================

    printf("========================================================================\n");
    printf("  Summary: eBase/Eneva Production Readiness\n");
    printf("========================================================================\n");

    // Use mixed pattern result from scale test
    generate_mixed_pattern(consumption, baseline, DAILY_BATCH, 42);
    res = benchmark_workload("Mixed", consumption, baseline, categories, DAILY_BATCH);

    double baseline_throughput = 169.0;  // C# baseline: 169 M intervals/sec
    double speedup = res.throughput_millions_per_sec / baseline_throughput;

    printf("Performance:\n");
    printf("  Throughput:         %.2f M intervals/sec\n", res.throughput_millions_per_sec);
    printf("  vs C# baseline:     %.2f× speedup\n", speedup);
    printf("  Classification:     %.2f ms per 1M intervals\n", res.classification_ms);
    printf("  Aggregation:        %.2f ms per 1M intervals (%.1f%% overhead)\n",
           res.aggregation_ms, 100.0 * res.aggregation_ms / res.classification_ms);
    printf("\n");

    printf("Compliance:\n");
    printf("  Determinism:        %.4f%% variance ", det_result.variance_pct);
    if (det_result.variance_pct == 0.0) {
        printf("✓\n");
    } else {
        printf("✗\n");
    }
    printf("  Billing compliant:  %s\n", det_result.variance_pct == 0.0 ? "YES" : "NO");
    printf("\n");

    printf("Scaling:\n");
    printf("  Hourly (400K):      < 1 sec (real-time)\n");
    printf("  Daily (1M):         %.2f ms\n", res.total_ms);
    printf("  Monthly (2.88B):    ~%.1f sec (estimated)\n", res.total_ms * 2880);
    printf("  Annual (35B):       ~%.1f sec (estimated)\n", res.total_ms * 35000);
    printf("\n");

    printf("Production Status:\n");
    if (res.throughput_millions_per_sec > 500 && det_result.variance_pct == 0.0) {
        printf("  ✓ READY FOR PRODUCTION\n");
    } else {
        printf("  ✗ NOT READY (performance or determinism issues)\n");
    }
    printf("\n");

    printf("========================================================================\n");

    // Cleanup
    free(consumption);
    free(baseline);
    free(categories);

    return 0;
}
