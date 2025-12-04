/**
 * load_profiling_diagnostics.h - Diagnostic utilities for load profiling
 *
 * Debugging and validation tools for investigating classification mismatches
 *
 * Copyright (c) 2025 NavierLib Technologies
 */

#ifndef NAVIERLIB_LOAD_PROFILING_DIAGNOSTICS_H
#define NAVIERLIB_LOAD_PROFILING_DIAGNOSTICS_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Diagnostic result for single classification
 */
typedef struct {
    int64_t index;
    double consumption;
    double baseline;
    double ratio;
    uint8_t category_computed;
    uint8_t category_expected;
    int matches;
} ClassificationDiagnostic;

/**
 * Diagnostic report for batch classification
 */
typedef struct {
    int64_t total_count;
    int64_t match_count;
    int64_t mismatch_count;
    double match_rate;
    ClassificationDiagnostic* mismatches;  // Array of first N mismatches
    int64_t mismatch_capacity;
} DiagnosticReport;

/**
 * Print single classification diagnostic
 */
void print_classification_diagnostic(const ClassificationDiagnostic* diag, FILE* out);

/**
 * Print diagnostic report summary
 */
void print_diagnostic_report(const DiagnosticReport* report, FILE* out);

/**
 * Classify single interval with detailed diagnostics
 *
 * @param consumption    Consumption value (kWh)
 * @param baseline       Baseline value (kWh)
 * @param low_threshold  Low threshold ratio
 * @param high_threshold High threshold ratio
 * @param diag           Output: diagnostic information
 * @return Computed category
 */
uint8_t classify_single_with_diagnostics(
    double consumption,
    double baseline,
    double low_threshold,
    double high_threshold,
    ClassificationDiagnostic* diag
);

/**
 * Compare two classification results and generate diagnostic report
 *
 * @param consumption      Input: consumption values
 * @param baseline         Input: baseline values
 * @param categories_a     First classification result
 * @param categories_b     Second classification result
 * @param count            Number of intervals
 * @param max_mismatches   Maximum mismatches to capture
 * @param report           Output: diagnostic report
 * @return 0 on success
 */
int compare_classifications(
    const double* consumption,
    const double* baseline,
    const uint8_t* categories_a,
    const uint8_t* categories_b,
    int64_t count,
    int64_t max_mismatches,
    DiagnosticReport* report
);

/**
 * Allocate diagnostic report
 */
DiagnosticReport* alloc_diagnostic_report(int64_t max_mismatches);

/**
 * Free diagnostic report
 */
void free_diagnostic_report(DiagnosticReport* report);

/**
 * Dump first N packed bytes as binary
 */
void dump_packed_bytes(const uint8_t* packed, int64_t count, int64_t limit, FILE* out);

/**
 * Dump first N intervals with unpacked trits
 */
void dump_unpacked_intervals(
    const uint8_t* categories,
    int64_t count,
    int64_t limit,
    FILE* out
);

/**
 * Verify pack/unpack roundtrip
 *
 * Tests that pack_trits(t0,t1,t2,t3) followed by unpack_trit()
 * recovers original values
 *
 * @return 0 if pass, -1 if fail
 */
int test_pack_unpack_roundtrip(FILE* out);

#ifdef __cplusplus
}
#endif

#endif // NAVIERLIB_LOAD_PROFILING_DIAGNOSTICS_H
