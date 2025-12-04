/**
 * load_profiling_test.cpp - C++ test suite for load profiling
 *
 * Comprehensive diagnostic and benchmarking suite for investigating
 * classification mismatches and performance characteristics.
 *
 * USAGE:
 *   LoadProfilingTest.exe [test]
 *
 * Tests:
 *   --pack-unpack       Test pack/unpack roundtrip
 *   --single            Test single classification
 *   --batch-small       Test small batch (100 intervals)
 *   --batch-large       Test large batch (1M intervals)
 *   --benchmark         Run performance benchmark
 *   --all               Run all tests (default)
 *
 * Copyright (c) 2025 NavierLib Technologies
 */

#include "../../src/navierlib/load_profiling.h"
#include "../../src/navierlib/load_profiling_diagnostics.h"
#include "../../src/core/algebra/ternary_algebra.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// Test configuration
#define SMALL_BATCH_SIZE 100
#define LARGE_BATCH_SIZE 1000000
#define BENCHMARK_ITERATIONS 10

#define LOW_THRESHOLD 0.8
#define HIGH_THRESHOLD 1.2

/**
 * Generate realistic energy consumption data
 */
static void generate_test_data(
    double* consumption,
    double* baseline,
    int64_t count,
    unsigned int seed
) {
    srand(seed);

    for (int64_t i = 0; i < count; i++) {
        // Baseline varies ±10%
        baseline[i] = 100.0 * (0.9 + ((double)rand() / RAND_MAX) * 0.2);

        // Consumption distribution:
        // 20% below (0.5-0.8× baseline)
        // 60% normal (0.8-1.2× baseline)
        // 20% peak (1.2-2.0× baseline)

        double pattern = (double)rand() / RAND_MAX;

        if (pattern < 0.2) {
            // Below baseline
            consumption[i] = baseline[i] * (0.5 + ((double)rand() / RAND_MAX) * 0.3);
        } else if (pattern < 0.8) {
            // Normal
            consumption[i] = baseline[i] * (0.8 + ((double)rand() / RAND_MAX) * 0.4);
        } else {
            // Peak
            consumption[i] = baseline[i] * (1.2 + ((double)rand() / RAND_MAX) * 0.8);
        }
    }
}

/**
 * Reference C++ classification (matches C# baseline)
 */
static void classify_reference_cpp(
    const double* consumption,
    const double* baseline,
    uint8_t* categories,
    int64_t count,
    double low_threshold,
    double high_threshold
) {
    // Initialize all bytes to zero
    int64_t packed_count = (count + 3) / 4;
    memset(categories, 0, packed_count);

    for (int64_t i = 0; i < count; i++) {
        double ratio = consumption[i] / baseline[i];

        uint8_t category;
        if (ratio < low_threshold) {
            category = 0b00;  // Below baseline
        } else if (ratio > high_threshold) {
            category = 0b10;  // Peak demand
        } else {
            category = 0b01;  // Normal
        }

        // Pack into byte
        int byte_idx = i / 4;
        int trit_idx = i % 4;

        categories[byte_idx] |= (category << (trit_idx * 2));
    }
}

/**
 * Test 1: Pack/Unpack Roundtrip
 */
static int test_pack_unpack() {
    printf("\n");
    printf("=============================================================================\n");
    printf("  TEST 1: Pack/Unpack Roundtrip\n");
    printf("=============================================================================\n");

    int result = test_pack_unpack_roundtrip(stdout);

    if (result == 0) {
        printf("✓ PASS\n");
    } else {
        printf("✗ FAIL\n");
    }

    return result;
}

/**
 * Test 2: Single Classification
 */
static int test_single_classification() {
    printf("\n");
    printf("=============================================================================\n");
    printf("  TEST 2: Single Classification\n");
    printf("=============================================================================\n");

    struct {
        double consumption;
        double baseline;
        uint8_t expected;
        const char* description;
    } tests[] = {
        {70.0, 100.0, 0b00, "Below baseline (70% < 80%)"},
        {90.0, 100.0, 0b01, "Normal (90% between 80%-120%)"},
        {130.0, 100.0, 0b10, "Peak (130% > 120%)"},
        {80.0, 100.0, 0b01, "Boundary: exactly 80%"},
        {120.0, 100.0, 0b01, "Boundary: exactly 120%"},
        {79.9, 100.0, 0b00, "Just below 80%"},
        {120.1, 100.0, 0b10, "Just above 120%"},
    };

    int num_tests = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    int failed = 0;

    printf("\nTest cases:\n");
    printf("-----------------------------------------------------------------------------\n");

    for (int i = 0; i < num_tests; i++) {
        ClassificationDiagnostic diag;
        diag.index = i;

        uint8_t result = classify_single_with_diagnostics(
            tests[i].consumption,
            tests[i].baseline,
            LOW_THRESHOLD,
            HIGH_THRESHOLD,
            &diag
        );

        diag.category_expected = tests[i].expected;
        diag.matches = (result == tests[i].expected);

        printf("[%d] %s\n", i, tests[i].description);
        printf("    consumption=%.2f, baseline=%.2f, ratio=%.4f\n",
               tests[i].consumption, tests[i].baseline,
               tests[i].consumption / tests[i].baseline);
        printf("    Expected: 0b%d%d, Got: 0b%d%d -> %s\n",
               (tests[i].expected >> 1) & 1, tests[i].expected & 1,
               (result >> 1) & 1, result & 1,
               diag.matches ? "PASS" : "FAIL");

        if (diag.matches) {
            passed++;
        } else {
            failed++;
        }
    }

    printf("-----------------------------------------------------------------------------\n");
    printf("Results: %d passed, %d failed\n", passed, failed);

    if (failed == 0) {
        printf("✓ PASS\n");
        return 0;
    } else {
        printf("✗ FAIL\n");
        return -1;
    }
}

/**
 * Test 3: Small Batch
 */
static int test_small_batch() {
    printf("\n");
    printf("=============================================================================\n");
    printf("  TEST 3: Small Batch (%d intervals)\n", SMALL_BATCH_SIZE);
    printf("=============================================================================\n");

    // Allocate arrays
    double* consumption = (double*)malloc(SMALL_BATCH_SIZE * sizeof(double));
    double* baseline = (double*)malloc(SMALL_BATCH_SIZE * sizeof(double));
    uint8_t* categories_navierlib = (uint8_t*)malloc((SMALL_BATCH_SIZE + 3) / 4);
    uint8_t* categories_reference = (uint8_t*)malloc((SMALL_BATCH_SIZE + 3) / 4);

    if (!consumption || !baseline || !categories_navierlib || !categories_reference) {
        printf("ERROR: Memory allocation failed\n");
        return -1;
    }

    // Generate test data
    printf("Generating test data (seed=42)...\n");
    generate_test_data(consumption, baseline, SMALL_BATCH_SIZE, 42);

    // Classify with NavierLib
    printf("Classifying with NavierLib...\n");
    int status = nv_classify_load_profile(
        consumption, baseline, categories_navierlib,
        SMALL_BATCH_SIZE, LOW_THRESHOLD, HIGH_THRESHOLD
    );

    if (status != 0) {
        printf("ERROR: nv_classify_load_profile failed with status %d\n", status);
        return -1;
    }

    // Classify with reference implementation
    printf("Classifying with C++ reference...\n");
    classify_reference_cpp(
        consumption, baseline, categories_reference,
        SMALL_BATCH_SIZE, LOW_THRESHOLD, HIGH_THRESHOLD
    );

    // Dump first 20 intervals for inspection
    printf("\nNavierLib output (first 20 intervals):\n");
    dump_unpacked_intervals(categories_navierlib, SMALL_BATCH_SIZE, 20, stdout);

    printf("\nReference output (first 20 intervals):\n");
    dump_unpacked_intervals(categories_reference, SMALL_BATCH_SIZE, 20, stdout);

    printf("\nNavierLib packed bytes (first 10 bytes):\n");
    dump_packed_bytes(categories_navierlib, SMALL_BATCH_SIZE, 10, stdout);

    printf("\nReference packed bytes (first 10 bytes):\n");
    dump_packed_bytes(categories_reference, SMALL_BATCH_SIZE, 10, stdout);

    // Compare classifications
    DiagnosticReport* report = alloc_diagnostic_report(20);
    compare_classifications(
        consumption, baseline,
        categories_navierlib,
        categories_reference,
        SMALL_BATCH_SIZE,
        20,
        report
    );

    print_diagnostic_report(report, stdout);

    int result = (report->mismatch_count == 0) ? 0 : -1;

    if (result == 0) {
        printf("\n✓ PASS: All classifications match\n");
    } else {
        printf("\n✗ FAIL: %lld mismatches detected\n", report->mismatch_count);
    }

    // Cleanup
    free_diagnostic_report(report);
    free(consumption);
    free(baseline);
    free(categories_navierlib);
    free(categories_reference);

    return result;
}

/**
 * Test 4: Large Batch
 */
static int test_large_batch() {
    printf("\n");
    printf("=============================================================================\n");
    printf("  TEST 4: Large Batch (%d intervals)\n", LARGE_BATCH_SIZE);
    printf("=============================================================================\n");

    // Allocate arrays
    double* consumption = (double*)malloc(LARGE_BATCH_SIZE * sizeof(double));
    double* baseline = (double*)malloc(LARGE_BATCH_SIZE * sizeof(double));
    uint8_t* categories_navierlib = (uint8_t*)malloc((LARGE_BATCH_SIZE + 3) / 4);
    uint8_t* categories_reference = (uint8_t*)malloc((LARGE_BATCH_SIZE + 3) / 4);

    if (!consumption || !baseline || !categories_navierlib || !categories_reference) {
        printf("ERROR: Memory allocation failed\n");
        return -1;
    }

    // Generate test data
    printf("Generating test data (seed=42)...\n");
    generate_test_data(consumption, baseline, LARGE_BATCH_SIZE, 42);

    // Classify with NavierLib
    printf("Classifying with NavierLib...\n");
    int status = nv_classify_load_profile(
        consumption, baseline, categories_navierlib,
        LARGE_BATCH_SIZE, LOW_THRESHOLD, HIGH_THRESHOLD
    );

    if (status != 0) {
        printf("ERROR: nv_classify_load_profile failed with status %d\n", status);
        return -1;
    }

    // Classify with reference implementation
    printf("Classifying with C++ reference...\n");
    classify_reference_cpp(
        consumption, baseline, categories_reference,
        LARGE_BATCH_SIZE, LOW_THRESHOLD, HIGH_THRESHOLD
    );

    // Compare classifications
    DiagnosticReport* report = alloc_diagnostic_report(100);
    compare_classifications(
        consumption, baseline,
        categories_navierlib,
        categories_reference,
        LARGE_BATCH_SIZE,
        100,
        report
    );

    print_diagnostic_report(report, stdout);

    int result = (report->mismatch_count == 0) ? 0 : -1;

    if (result == 0) {
        printf("\n✓ PASS: All classifications match\n");
    } else {
        printf("\n✗ FAIL: %lld mismatches detected (%.2f%%)\n",
               report->mismatch_count,
               100.0 * report->mismatch_count / LARGE_BATCH_SIZE);
    }

    // Cleanup
    free_diagnostic_report(report);
    free(consumption);
    free(baseline);
    free(categories_navierlib);
    free(categories_reference);

    return result;
}

/**
 * Test 5: Performance Benchmark
 */
static int test_benchmark() {
    printf("\n");
    printf("=============================================================================\n");
    printf("  TEST 5: Performance Benchmark\n");
    printf("=============================================================================\n");

    // Allocate arrays
    double* consumption = (double*)malloc(LARGE_BATCH_SIZE * sizeof(double));
    double* baseline = (double*)malloc(LARGE_BATCH_SIZE * sizeof(double));
    uint8_t* categories = (uint8_t*)malloc((LARGE_BATCH_SIZE + 3) / 4);

    if (!consumption || !baseline || !categories) {
        printf("ERROR: Memory allocation failed\n");
        return -1;
    }

    // Generate test data
    generate_test_data(consumption, baseline, LARGE_BATCH_SIZE, 42);

    // Warmup
    printf("Warming up...\n");
    for (int i = 0; i < 3; i++) {
        nv_classify_load_profile(
            consumption, baseline, categories,
            LARGE_BATCH_SIZE, LOW_THRESHOLD, HIGH_THRESHOLD
        );
    }

    // Benchmark
    printf("Running %d iterations...\n", BENCHMARK_ITERATIONS);
    clock_t start = clock();

    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        nv_classify_load_profile(
            consumption, baseline, categories,
            LARGE_BATCH_SIZE, LOW_THRESHOLD, HIGH_THRESHOLD
        );
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double avg_time = elapsed / BENCHMARK_ITERATIONS;
    double throughput = LARGE_BATCH_SIZE / avg_time / 1000000.0;

    printf("\nResults:\n");
    printf("  Total time:       %.3f sec\n", elapsed);
    printf("  Avg per iteration: %.2f ms\n", avg_time * 1000);
    printf("  Throughput:       %.2f M intervals/sec\n", throughput);

    // Cleanup
    free(consumption);
    free(baseline);
    free(categories);

    printf("\n✓ Benchmark complete\n");
    return 0;
}

/**
 * Main
 */
int main(int argc, char** argv) {
    printf("NavierLib Load Profiling Test Suite\n");
    printf("=============================================================================\n");
    printf("Platform: Windows x64\n");
    printf("Compiler: MSVC\n");
    printf("\n");

    int run_all = 1;
    int run_pack_unpack = 0;
    int run_single = 0;
    int run_batch_small = 0;
    int run_batch_large = 0;
    int run_benchmark = 0;

    // Parse arguments
    if (argc > 1) {
        run_all = 0;
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--pack-unpack") == 0) run_pack_unpack = 1;
            else if (strcmp(argv[i], "--single") == 0) run_single = 1;
            else if (strcmp(argv[i], "--batch-small") == 0) run_batch_small = 1;
            else if (strcmp(argv[i], "--batch-large") == 0) run_batch_large = 1;
            else if (strcmp(argv[i], "--benchmark") == 0) run_benchmark = 1;
            else if (strcmp(argv[i], "--all") == 0) run_all = 1;
            else {
                printf("Unknown option: %s\n", argv[i]);
                printf("\nUsage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  --pack-unpack    Test pack/unpack roundtrip\n");
                printf("  --single         Test single classification\n");
                printf("  --batch-small    Test small batch (100 intervals)\n");
                printf("  --batch-large    Test large batch (1M intervals)\n");
                printf("  --benchmark      Run performance benchmark\n");
                printf("  --all            Run all tests (default)\n");
                return 1;
            }
        }
    }

    if (run_all) {
        run_pack_unpack = 1;
        run_single = 1;
        run_batch_small = 1;
        run_batch_large = 1;
        run_benchmark = 1;
    }

    int failures = 0;

    if (run_pack_unpack) {
        if (test_pack_unpack() != 0) failures++;
    }

    if (run_single) {
        if (test_single_classification() != 0) failures++;
    }

    if (run_batch_small) {
        if (test_small_batch() != 0) failures++;
    }

    if (run_batch_large) {
        if (test_large_batch() != 0) failures++;
    }

    if (run_benchmark) {
        if (test_benchmark() != 0) failures++;
    }

    printf("\n");
    printf("=============================================================================\n");
    if (failures == 0) {
        printf("  ALL TESTS PASSED\n");
    } else {
        printf("  %d TEST(S) FAILED\n", failures);
    }
    printf("=============================================================================\n");

    return (failures == 0) ? 0 : 1;
}
