/**
 * optimization_benchmark.cpp - Comprehensive optimization comparison
 *
 * Compares three implementations:
 * 1. Original (correct): memset + movemask
 * 2. Optimized: no memset + branchless packing
 * 3. 8-wide: 2× AVX2 registers per iteration
 *
 * Copyright (c) 2025 NavierLib Technologies
 */

#include "../../src/navierlib/load_profiling.h"
#include "../../src/navierlib/load_profiling_diagnostics.h"
#include "../../src/core/algebra/ternary_algebra.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define TEST_SIZE 1000000
#define ITERATIONS 20
#define LOW_THRESHOLD 0.8
#define HIGH_THRESHOLD 1.2

// Forward declarations for optimized versions
extern int nv_classify_load_profile_optimized(
    const double* consumption,
    const double* baseline,
    uint8_t* categories,
    int64_t count,
    double low_ratio,
    double high_ratio
);

extern int nv_classify_load_profile_8wide(
    const double* consumption,
    const double* baseline,
    uint8_t* categories,
    int64_t count,
    double low_ratio,
    double high_ratio
);

/**
 * Generate test data
 */
static void generate_test_data(double* consumption, double* baseline, int64_t count) {
    srand(42);
    for (int64_t i = 0; i < count; i++) {
        baseline[i] = 100.0 * (0.9 + ((double)rand() / RAND_MAX) * 0.2);
        double pattern = (double)rand() / RAND_MAX;
        if (pattern < 0.2) {
            consumption[i] = baseline[i] * (0.5 + ((double)rand() / RAND_MAX) * 0.3);
        } else if (pattern < 0.8) {
            consumption[i] = baseline[i] * (0.8 + ((double)rand() / RAND_MAX) * 0.4);
        } else {
            consumption[i] = baseline[i] * (1.2 + ((double)rand() / RAND_MAX) * 0.8);
        }
    }
}

/**
 * C++ reference implementation
 */
static void classify_reference(
    const double* consumption,
    const double* baseline,
    uint8_t* categories,
    int64_t count
) {
    int64_t packed_count = (count + 3) / 4;
    memset(categories, 0, packed_count);

    for (int64_t i = 0; i < count; i++) {
        double ratio = consumption[i] / baseline[i];
        uint8_t category;
        if (ratio < LOW_THRESHOLD) {
            category = 0b00;
        } else if (ratio > HIGH_THRESHOLD) {
            category = 0b10;
        } else {
            category = 0b01;
        }
        int byte_idx = i / 4;
        int trit_idx = i % 4;
        categories[byte_idx] |= (category << (trit_idx * 2));
    }
}

/**
 * Compare two results
 */
static int compare_results(const uint8_t* a, const uint8_t* b, int64_t count) {
    int64_t packed_count = (count + 3) / 4;
    for (int64_t i = 0; i < packed_count; i++) {
        if (a[i] != b[i]) {
            return -1;
        }
    }
    return 0;
}

/**
 * Benchmark function
 */
typedef int (*classify_func_t)(const double*, const double*, uint8_t*, int64_t, double, double);

static double benchmark_function(
    const char* name,
    classify_func_t func,
    const double* consumption,
    const double* baseline,
    uint8_t* categories,
    uint8_t* reference,
    int64_t count
) {
    printf("%-25s", name);
    fflush(stdout);

    // Warmup
    for (int i = 0; i < 3; i++) {
        func(consumption, baseline, categories, count, LOW_THRESHOLD, HIGH_THRESHOLD);
    }

    // Benchmark
    clock_t start = clock();
    for (int i = 0; i < ITERATIONS; i++) {
        func(consumption, baseline, categories, count, LOW_THRESHOLD, HIGH_THRESHOLD);
    }
    clock_t end = clock();

    double total_time = (double)(end - start) / CLOCKS_PER_SEC;
    double avg_time = total_time / ITERATIONS;
    double throughput = count / avg_time / 1000000.0;

    // Verify correctness
    int matches = compare_results(categories, reference, count);
    const char* status = (matches == 0) ? "✓" : "✗";

    printf(" %6.2f ms | %7.2f M/s | %s\n", avg_time * 1000, throughput, status);

    return avg_time;
}

int main() {
    printf("\n");
    printf("================================================================================\n");
    printf("  NavierLib Load Profiling Optimization Benchmark\n");
    printf("================================================================================\n");
    printf("Platform:   Windows x64\n");
    printf("Compiler:   MSVC with AVX2\n");
    printf("Test size:  %d intervals\n", TEST_SIZE);
    printf("Iterations: %d\n", ITERATIONS);
    printf("\n");

    // Allocate arrays
    double* consumption = (double*)malloc(TEST_SIZE * sizeof(double));
    double* baseline = (double*)malloc(TEST_SIZE * sizeof(double));
    uint8_t* categories = (uint8_t*)malloc((TEST_SIZE + 3) / 4);
    uint8_t* reference = (uint8_t*)malloc((TEST_SIZE + 3) / 4);

    if (!consumption || !baseline || !categories || !reference) {
        printf("ERROR: Memory allocation failed\n");
        return 1;
    }

    // Generate test data
    printf("Generating test data...\n");
    generate_test_data(consumption, baseline, TEST_SIZE);

    // Generate reference results
    printf("Computing reference results...\n");
    classify_reference(consumption, baseline, reference, TEST_SIZE);
    printf("\n");

    printf("Implementation           |   Time   | Throughput | Status\n");
    printf("-------------------------|----------|------------|-------\n");

    // Benchmark C++ reference (manual loop)
    printf("%-25s", "C++ Reference");
    fflush(stdout);

    // Warmup
    for (int i = 0; i < 3; i++) {
        classify_reference(consumption, baseline, categories, TEST_SIZE);
    }

    clock_t start = clock();
    for (int i = 0; i < ITERATIONS; i++) {
        classify_reference(consumption, baseline, categories, TEST_SIZE);
    }
    clock_t end = clock();
    double ref_time = ((double)(end - start) / CLOCKS_PER_SEC) / ITERATIONS;
    double ref_throughput = TEST_SIZE / ref_time / 1000000.0;
    printf(" %6.2f ms | %7.2f M/s | ✓\n", ref_time * 1000, ref_throughput);

    // Benchmark original (correct)
    double orig_time = benchmark_function(
        "Original (memset+mask)",
        nv_classify_load_profile,
        consumption, baseline, categories, reference, TEST_SIZE
    );

    // Benchmark optimized (no memset)
    double opt_time = benchmark_function(
        "Optimized (branchless)",
        nv_classify_load_profile_optimized,
        consumption, baseline, categories, reference, TEST_SIZE
    );

    // Benchmark 8-wide
    double wide_time = benchmark_function(
        "8-Wide SIMD",
        nv_classify_load_profile_8wide,
        consumption, baseline, categories, reference, TEST_SIZE
    );

    // Summary
    printf("\n");
    printf("================================================================================\n");
    printf("  Performance Summary\n");
    printf("================================================================================\n");
    printf("C++ Reference:         %.2f ms (baseline)\n", ref_time * 1000);
    printf("Original (correct):    %.2f ms (%.2f× vs reference)\n",
           orig_time * 1000, ref_time / orig_time);
    printf("Optimized:             %.2f ms (%.2f× vs reference, %.2f× vs original)\n",
           opt_time * 1000, ref_time / opt_time, orig_time / opt_time);
    printf("8-Wide SIMD:           %.2f ms (%.2f× vs reference, %.2f× vs original)\n",
           wide_time * 1000, ref_time / wide_time, orig_time / wide_time);
    printf("\n");

    // Speedup table
    printf("Speedup Analysis:\n");
    printf("  Original → Optimized:  %.2f× faster (saved %.2f ms)\n",
           orig_time / opt_time, (orig_time - opt_time) * 1000);
    printf("  Original → 8-Wide:     %.2f× faster (saved %.2f ms)\n",
           orig_time / wide_time, (orig_time - wide_time) * 1000);
    printf("  Optimized → 8-Wide:    %.2f× faster (saved %.2f ms)\n",
           opt_time / wide_time, (opt_time - wide_time) * 1000);
    printf("\n");

    // Memory bandwidth analysis
    double data_size_mb = (TEST_SIZE * 2 * sizeof(double)) / (1024.0 * 1024.0);
    double bandwidth_gbps = data_size_mb / (wide_time * 1024.0);
    printf("Memory Bandwidth:\n");
    printf("  Data size:             %.2f MB (input)\n", data_size_mb);
    printf("  Effective bandwidth:   %.2f GB/s\n", bandwidth_gbps);
    printf("  Theoretical limit:     ~2.0 ms (for %.2f MB @ 8 GB/s)\n",
           data_size_mb);
    printf("\n");

    printf("================================================================================\n");

    // Cleanup
    free(consumption);
    free(baseline);
    free(categories);
    free(reference);

    return 0;
}
