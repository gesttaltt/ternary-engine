/**
 * load_profiling_diagnostics.cpp - Diagnostic utilities implementation
 *
 * Copyright (c) 2025 NavierLib Technologies
 */

#include "load_profiling_diagnostics.h"
#include "load_profiling.h"
#include "../core/algebra/ternary_algebra.h"
#include <stdlib.h>
#include <string.h>

// Ternary trit values (2-bit encoding)
#define TRIT_MINUS_ONE  0b00  // -1
#define TRIT_ZERO       0b01  //  0
#define TRIT_PLUS_ONE   0b10  // +1

/**
 * Classify single interval (for diagnostics)
 */
static uint8_t classify_single_internal(double consumption, double baseline,
                                        double low_threshold, double high_threshold) {
    if (baseline <= 0.0) {
        return TRIT_ZERO;
    }

    double ratio = consumption / baseline;

    if (ratio < low_threshold) {
        return TRIT_MINUS_ONE;
    } else if (ratio > high_threshold) {
        return TRIT_PLUS_ONE;
    } else {
        return TRIT_ZERO;
    }
}

/**
 * Convert trit to string
 */
static const char* trit_to_string(uint8_t trit) {
    switch (trit) {
        case TRIT_MINUS_ONE: return "BELOW (-1, 0b00)";
        case TRIT_ZERO:      return "NORMAL (0, 0b01)";
        case TRIT_PLUS_ONE:  return "PEAK (+1, 0b10)";
        default:             return "INVALID (0b11)";
    }
}

void print_classification_diagnostic(const ClassificationDiagnostic* diag, FILE* out) {
    fprintf(out, "[%6lld] consumption=%.2f, baseline=%.2f, ratio=%.4f\n",
            diag->index, diag->consumption, diag->baseline, diag->ratio);
    fprintf(out, "         Computed: %s\n", trit_to_string(diag->category_computed));
    fprintf(out, "         Expected: %s\n", trit_to_string(diag->category_expected));
    fprintf(out, "         Match: %s\n", diag->matches ? "YES" : "NO");
}

void print_diagnostic_report(const DiagnosticReport* report, FILE* out) {
    fprintf(out, "\n");
    fprintf(out, "=============================================================================\n");
    fprintf(out, "  DIAGNOSTIC REPORT\n");
    fprintf(out, "=============================================================================\n");
    fprintf(out, "Total intervals:  %lld\n", report->total_count);
    fprintf(out, "Matches:          %lld (%.2f%%)\n",
            report->match_count,
            100.0 * report->match_count / report->total_count);
    fprintf(out, "Mismatches:       %lld (%.2f%%)\n",
            report->mismatch_count,
            100.0 * report->mismatch_count / report->total_count);
    fprintf(out, "\n");

    if (report->mismatch_count > 0) {
        int64_t shown = report->mismatch_count < report->mismatch_capacity
                       ? report->mismatch_count
                       : report->mismatch_capacity;

        fprintf(out, "First %lld mismatches:\n", shown);
        fprintf(out, "-----------------------------------------------------------------------------\n");

        for (int64_t i = 0; i < shown; i++) {
            print_classification_diagnostic(&report->mismatches[i], out);
            if (i < shown - 1) {
                fprintf(out, "\n");
            }
        }
    }
    fprintf(out, "=============================================================================\n");
}

uint8_t classify_single_with_diagnostics(
    double consumption,
    double baseline,
    double low_threshold,
    double high_threshold,
    ClassificationDiagnostic* diag
) {
    uint8_t category = classify_single_internal(consumption, baseline,
                                                 low_threshold, high_threshold);

    if (diag) {
        diag->consumption = consumption;
        diag->baseline = baseline;
        diag->ratio = (baseline > 0.0) ? (consumption / baseline) : 0.0;
        diag->category_computed = category;
    }

    return category;
}

int compare_classifications(
    const double* consumption,
    const double* baseline,
    const uint8_t* categories_a,
    const uint8_t* categories_b,
    int64_t count,
    int64_t max_mismatches,
    DiagnosticReport* report
) {
    if (!consumption || !baseline || !categories_a || !categories_b || !report) {
        return -1;
    }

    report->total_count = count;
    report->match_count = 0;
    report->mismatch_count = 0;

    int64_t captured = 0;

    for (int64_t i = 0; i < count; i++) {
        // Unpack trits from packed bytes
        int byte_idx = i / 4;
        int trit_idx = i % 4;

        uint8_t cat_a = (categories_a[byte_idx] >> (trit_idx * 2)) & 0b11;
        uint8_t cat_b = (categories_b[byte_idx] >> (trit_idx * 2)) & 0b11;

        if (cat_a == cat_b) {
            report->match_count++;
        } else {
            report->mismatch_count++;

            // Capture details if space available
            if (captured < max_mismatches && captured < report->mismatch_capacity) {
                ClassificationDiagnostic* diag = &report->mismatches[captured];
                diag->index = i;
                diag->consumption = consumption[i];
                diag->baseline = baseline[i];
                diag->ratio = (baseline[i] > 0.0) ? (consumption[i] / baseline[i]) : 0.0;
                diag->category_computed = cat_a;
                diag->category_expected = cat_b;
                diag->matches = 0;
                captured++;
            }
        }
    }

    report->match_rate = (double)report->match_count / report->total_count;
    return 0;
}

DiagnosticReport* alloc_diagnostic_report(int64_t max_mismatches) {
    DiagnosticReport* report = (DiagnosticReport*)malloc(sizeof(DiagnosticReport));
    if (!report) return NULL;

    memset(report, 0, sizeof(DiagnosticReport));
    report->mismatch_capacity = max_mismatches;

    if (max_mismatches > 0) {
        report->mismatches = (ClassificationDiagnostic*)malloc(
            max_mismatches * sizeof(ClassificationDiagnostic)
        );
        if (!report->mismatches) {
            free(report);
            return NULL;
        }
    }

    return report;
}

void free_diagnostic_report(DiagnosticReport* report) {
    if (report) {
        if (report->mismatches) {
            free(report->mismatches);
        }
        free(report);
    }
}

void dump_packed_bytes(const uint8_t* packed, int64_t count, int64_t limit, FILE* out) {
    fprintf(out, "\nPacked bytes (first %lld):\n", limit);
    fprintf(out, "Byte# | Binary     | Hex | t0 t1 t2 t3\n");
    fprintf(out, "------|------------|-----|------------\n");

    int64_t byte_count = (count + 3) / 4;
    int64_t show = (limit < byte_count) ? limit : byte_count;

    for (int64_t i = 0; i < show; i++) {
        uint8_t byte = packed[i];

        // Extract 4 trits
        uint8_t t0 = (byte >> 0) & 0b11;
        uint8_t t1 = (byte >> 2) & 0b11;
        uint8_t t2 = (byte >> 4) & 0b11;
        uint8_t t3 = (byte >> 6) & 0b11;

        fprintf(out, "%5lld | ", i);

        // Print binary
        for (int b = 7; b >= 0; b--) {
            fprintf(out, "%d", (byte >> b) & 1);
        }

        fprintf(out, " | %02X  | %02X %02X %02X %02X\n",
                byte, t0, t1, t2, t3);
    }
}

void dump_unpacked_intervals(
    const uint8_t* categories,
    int64_t count,
    int64_t limit,
    FILE* out
) {
    fprintf(out, "\nUnpacked intervals (first %lld):\n", limit);
    fprintf(out, "Index | Category | Description\n");
    fprintf(out, "------|----------|-------------------\n");

    int64_t show = (limit < count) ? limit : count;

    for (int64_t i = 0; i < show; i++) {
        int byte_idx = i / 4;
        int trit_idx = i % 4;

        uint8_t category = (categories[byte_idx] >> (trit_idx * 2)) & 0b11;

        fprintf(out, "%5lld | 0b%d%d     | %s\n",
                i,
                (category >> 1) & 1,
                category & 1,
                trit_to_string(category));
    }
}

int test_pack_unpack_roundtrip(FILE* out) {
    fprintf(out, "\nTesting pack/unpack roundtrip...\n");
    fprintf(out, "-----------------------------------\n");

    // Test all combinations of 4 trits
    uint8_t trits[3] = {TRIT_MINUS_ONE, TRIT_ZERO, TRIT_PLUS_ONE};
    int failures = 0;
    int tests = 0;

    for (int i0 = 0; i0 < 3; i0++) {
        for (int i1 = 0; i1 < 3; i1++) {
            for (int i2 = 0; i2 < 3; i2++) {
                for (int i3 = 0; i3 < 3; i3++) {
                    uint8_t t0 = trits[i0];
                    uint8_t t1 = trits[i1];
                    uint8_t t2 = trits[i2];
                    uint8_t t3 = trits[i3];

                    // Pack
                    uint8_t packed = pack_trits(t0, t1, t2, t3);

                    // Unpack
                    uint8_t u0 = unpack_trit(packed, 0);
                    uint8_t u1 = unpack_trit(packed, 1);
                    uint8_t u2 = unpack_trit(packed, 2);
                    uint8_t u3 = unpack_trit(packed, 3);

                    tests++;

                    if (u0 != t0 || u1 != t1 || u2 != t2 || u3 != t3) {
                        fprintf(out, "FAIL: pack(%02X,%02X,%02X,%02X) = 0x%02X -> unpack(%02X,%02X,%02X,%02X)\n",
                                t0, t1, t2, t3, packed, u0, u1, u2, u3);
                        failures++;
                    }
                }
            }
        }
    }

    fprintf(out, "Tested %d combinations: ", tests);
    if (failures == 0) {
        fprintf(out, "ALL PASSED\n");
        return 0;
    } else {
        fprintf(out, "%d FAILURES\n", failures);
        return -1;
    }
}
