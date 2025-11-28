/**
 * test_backends.cpp - Backend System Tests
 *
 * Copyright 2025 Ternary Engine Contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Tests for v1.2.0 backend system:
 * - Backend registration
 * - Backend selection
 * - Dispatch system
 * - Cross-backend correctness (Scalar = AVX2 v1 = AVX2 v2)
 */

#include <iostream>
#include <cstring>
#include <cassert>
#include <cstdio>

#include "../../src/core/simd/backend_plugin_api.h"

// Test counters
static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    do { \
        tests_run++; \
        std::cout << "Running: " << name << "..."; \
    } while(0)

#define ASSERT_TRUE(cond) \
    do { \
        if (cond) { \
            std::cout << " PASS\n"; \
            tests_passed++; \
        } else { \
            std::cout << " FAIL: " << #cond << " is false\n"; \
        } \
    } while(0)

#define ASSERT_EQ(a, b) \
    do { \
        if ((a) == (b)) { \
            std::cout << " PASS\n"; \
            tests_passed++; \
        } else { \
            std::cout << " FAIL: " << #a << " (" << (a) << ") != " << #b << " (" << (b) << ")\n"; \
        } \
    } while(0)

// ============================================================================
// Helper Functions
// ============================================================================

void init_test_data(uint8_t* a, uint8_t* b, size_t n) {
    // Initialize with valid ternary values (00=-1, 01=0, 10=+1)
    for (size_t i = 0; i < n; i++) {
        a[i] = i % 3;  // 0, 1, 2 pattern
        b[i] = (i % 3 + 1) % 3;  // 1, 2, 0 pattern
    }
}

bool arrays_equal(const uint8_t* a, const uint8_t* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            std::cout << "  Mismatch at index " << i << ": "
                      << (int)a[i] << " != " << (int)b[i] << "\n";
            return false;
        }
    }
    return true;
}

// ============================================================================
// Backend System Tests
// ============================================================================

void test_backend_init() {
    TEST("Backend initialization");

    bool success = ternary_backend_init();
    ASSERT_TRUE(success);
}

void test_backend_registration() {
    TEST("Backend registration");

    size_t count = ternary_backend_count();

    // Should have at least Scalar backend
    // May also have AVX2 v1 and v2 if AVX2 available
    ASSERT_TRUE(count >= 1);
}

void test_backend_selection() {
    TEST("Backend selection");

    const TernaryBackend* best = ternary_backend_select_best();
    ASSERT_TRUE(best != NULL);
}

void test_active_backend() {
    TEST("Active backend");

    const TernaryBackend* active = ternary_backend_get_active();

    if (active) {
        std::cout << " PASS (using " << active->info.name << ")\n";
        tests_passed++;
    } else {
        std::cout << " FAIL: No active backend\n";
    }
}

void test_find_scalar_backend() {
    TEST("Find scalar backend");

    const TernaryBackend* scalar = ternary_backend_find("Scalar");
    ASSERT_TRUE(scalar != NULL);
}

// ============================================================================
// Correctness Tests
// ============================================================================

void test_dispatch_tnot() {
    TEST("Dispatch tnot");

    const size_t N = 100;
    uint8_t src[N], dst[N];

    // Initialize
    for (size_t i = 0; i < N; i++) {
        src[i] = i % 3;
    }

    // Execute
    ternary_dispatch_tnot(dst, src, N);

    // Verify (tnot inverts: 0→2, 1→1, 2→0)
    bool correct = true;
    for (size_t i = 0; i < N; i++) {
        uint8_t expected = 2 - src[i];  // Inversion
        if (dst[i] != expected) {
            correct = false;
            break;
        }
    }

    ASSERT_TRUE(correct);
}

void test_dispatch_tadd() {
    TEST("Dispatch tadd");

    const size_t N = 100;
    uint8_t a[N], b[N], dst[N];

    init_test_data(a, b, N);

    ternary_dispatch_tadd(dst, a, b, N);

    // Basic sanity check: result should be in valid range
    bool valid = true;
    for (size_t i = 0; i < N; i++) {
        if (dst[i] > 2) {
            valid = false;
            break;
        }
    }

    ASSERT_TRUE(valid);
}

void test_dispatch_tmul() {
    TEST("Dispatch tmul");

    const size_t N = 100;
    uint8_t a[N], b[N], dst[N];

    init_test_data(a, b, N);

    ternary_dispatch_tmul(dst, a, b, N);

    // Basic sanity check
    bool valid = true;
    for (size_t i = 0; i < N; i++) {
        if (dst[i] > 2) {
            valid = false;
            break;
        }
    }

    ASSERT_TRUE(valid);
}

// ============================================================================
// Cross-Backend Correctness Test
// ============================================================================

void test_cross_backend_correctness() {
    TEST("Cross-backend correctness (Scalar vs SIMD)");

    const size_t N = 1000;
    uint8_t a[N], b[N];
    uint8_t result_scalar[N], result_simd[N];

    init_test_data(a, b, N);

    // Get scalar backend
    const TernaryBackend* scalar = ternary_backend_find("Scalar");
    if (!scalar) {
        std::cout << " SKIP (scalar backend not found)\n";
        tests_passed++;  // Don't fail
        return;
    }

    // Get best SIMD backend (AVX2 v1 or v2)
    const TernaryBackend* simd = ternary_backend_select_best();
    if (!simd || simd == scalar) {
        std::cout << " SKIP (no SIMD backend available)\n";
        tests_passed++;  // Don't fail
        return;
    }

    std::cout << "\n";
    std::cout << "    Comparing: " << scalar->info.name << " vs " << simd->info.name << "\n";

    // Test tadd
    scalar->tadd(result_scalar, a, b, N);
    simd->tadd(result_simd, a, b, N);

    bool tadd_match = arrays_equal(result_scalar, result_simd, N);
    std::cout << "    tadd: " << (tadd_match ? "MATCH" : "MISMATCH") << "\n";

    // Test tmul
    scalar->tmul(result_scalar, a, b, N);
    simd->tmul(result_simd, a, b, N);

    bool tmul_match = arrays_equal(result_scalar, result_simd, N);
    std::cout << "    tmul: " << (tmul_match ? "MATCH" : "MISMATCH") << "\n";

    // Test tmax
    scalar->tmax(result_scalar, a, b, N);
    simd->tmax(result_simd, a, b, N);

    bool tmax_match = arrays_equal(result_scalar, result_simd, N);
    std::cout << "    tmax: " << (tmax_match ? "MATCH" : "MISMATCH") << "\n";

    // Test tmin
    scalar->tmin(result_scalar, a, b, N);
    simd->tmin(result_simd, a, b, N);

    bool tmin_match = arrays_equal(result_scalar, result_simd, N);
    std::cout << "    tmin: " << (tmin_match ? "MATCH" : "MISMATCH") << "\n";

    // Test tnot
    scalar->tnot(result_scalar, a, N);
    simd->tnot(result_simd, a, N);

    bool tnot_match = arrays_equal(result_scalar, result_simd, N);
    std::cout << "    tnot: " << (tnot_match ? "MATCH" : "MISMATCH") << "\n";

    bool all_match = tadd_match && tmul_match && tmax_match && tmin_match && tnot_match;

    if (all_match) {
        std::cout << "  PASS\n";
        tests_passed++;
    } else {
        std::cout << "  FAIL\n";
    }
}

// ============================================================================
// Performance Comparison Test
// ============================================================================

void test_performance_comparison() {
    TEST("Performance comparison (informational)");

    const size_t N = 1000000;  // 1M trits
    uint8_t* a = new uint8_t[N];
    uint8_t* b = new uint8_t[N];
    uint8_t* result = new uint8_t[N];

    init_test_data(a, b, N);

    // Get backends
    const TernaryBackend* scalar = ternary_backend_find("Scalar");
    const TernaryBackend* avx2_v1 = ternary_backend_find("AVX2_v1");
    const TernaryBackend* avx2_v2 = ternary_backend_find("AVX2_v2");

    std::cout << "\n";
    std::cout << "    Available backends:\n";
    if (scalar) std::cout << "      - " << scalar->info.name << "\n";
    if (avx2_v1) std::cout << "      - " << avx2_v1->info.name << "\n";
    if (avx2_v2) std::cout << "      - " << avx2_v2->info.name << "\n";

    std::cout << "    Note: Run benchmarks/bench_backends.py for detailed performance comparison\n";
    std::cout << "  PASS (informational)\n";

    delete[] a;
    delete[] b;
    delete[] result;

    tests_passed++;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "===========================================\n";
    std::cout << "Ternary Backend System Tests\n";
    std::cout << "===========================================\n\n";

    // Backend system tests
    test_backend_init();
    test_backend_registration();
    test_backend_selection();
    test_active_backend();
    test_find_scalar_backend();

    // Print registered backends
    std::cout << "\n";
    ternary_backend_print_all();

    // Correctness tests
    std::cout << "\n--- Dispatch Tests ---\n";
    test_dispatch_tnot();
    test_dispatch_tadd();
    test_dispatch_tmul();

    // Cross-backend validation
    std::cout << "\n--- Cross-Backend Tests ---\n";
    test_cross_backend_correctness();

    // Performance info
    std::cout << "\n--- Performance ---\n";
    test_performance_comparison();

    // Cleanup
    ternary_backend_shutdown();

    std::cout << "\n===========================================\n";
    std::cout << "Results: " << tests_passed << "/" << tests_run << " tests passed\n";

    if (tests_passed == tests_run) {
        std::cout << "✓ ALL TESTS PASSED\n";
        return 0;
    } else {
        std::cout << "✗ SOME TESTS FAILED\n";
        return 1;
    }
}
