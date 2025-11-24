// test_triadsextet.cpp — Unit tests for TriadSextet encoding
//
// Copyright (c) 2025 Jonathan Verdun (Ternary Engine Project)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <iomanip>
#include <cassert>
#include <vector>
#include <chrono>
#include "../src/engine/dense243/ternary_triadsextet.h"

// ANSI color codes for terminal output
#define COLOR_GREEN "\033[32m"
#define COLOR_RED "\033[31m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_RESET "\033[0m"

// Test result tracking
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST(name) \
    void test_##name(); \
    struct TestRegistrar_##name { \
        TestRegistrar_##name() { \
            std::cout << "Running test: " << #name << " ... "; \
            try { \
                test_##name(); \
                std::cout << COLOR_GREEN << "PASSED" << COLOR_RESET << "\n"; \
                g_tests_passed++; \
            } catch (const std::exception& e) { \
                std::cout << COLOR_RED << "FAILED: " << e.what() << COLOR_RESET << "\n"; \
                g_tests_failed++; \
            } \
        } \
    } g_test_registrar_##name; \
    void test_##name()

#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { \
        throw std::runtime_error(std::string("Assertion failed: ") + #a + " == " + #b); \
    }

// =============================================================================
// Test Cases
// =============================================================================

TEST(basic_pack_unpack) {
    // Test case: (+1, 0, -1)
    uint8_t t0 = 0b10;  // +1
    uint8_t t1 = 0b01;  //  0
    uint8_t t2 = 0b00;  // -1

    // Pack
    triadsextet_t sextet = triadsextet_pack(t0, t1, t2);
    // Expected: (2, 1, 0) in base-3 offset = 2 + 1*3 + 0*9 = 5
    ASSERT_EQ(sextet, 5);

    // Unpack
    TriadSextetUnpacked result = triadsextet_unpack(sextet);

    // Verify round-trip
    ASSERT_EQ(result.t0, t0);
    ASSERT_EQ(result.t1, t1);
    ASSERT_EQ(result.t2, t2);
}

TEST(all_zeros) {
    // Test case: (0, 0, 0)
    uint8_t t0 = 0b01, t1 = 0b01, t2 = 0b01;

    triadsextet_t sextet = triadsextet_pack(t0, t1, t2);
    // Expected: (1, 1, 1) in base-3 offset = 1 + 1*3 + 1*9 = 13
    ASSERT_EQ(sextet, 13);

    TriadSextetUnpacked result = triadsextet_unpack(sextet);
    ASSERT_EQ(result.t0, 0b01);
    ASSERT_EQ(result.t1, 0b01);
    ASSERT_EQ(result.t2, 0b01);
}

TEST(all_positive) {
    // Test case: (+1, +1, +1)
    uint8_t t0 = 0b10, t1 = 0b10, t2 = 0b10;

    triadsextet_t sextet = triadsextet_pack(t0, t1, t2);
    // Expected: (2, 2, 2) in base-3 offset = 2 + 2*3 + 2*9 = 2 + 6 + 18 = 26
    ASSERT_EQ(sextet, 26);
    ASSERT_EQ(triadsextet_is_valid(sextet), true);

    TriadSextetUnpacked result = triadsextet_unpack(sextet);
    ASSERT_EQ(result.t0, 0b10);
    ASSERT_EQ(result.t1, 0b10);
    ASSERT_EQ(result.t2, 0b10);
}

TEST(all_negative) {
    // Test case: (-1, -1, -1)
    uint8_t t0 = 0b00, t1 = 0b00, t2 = 0b00;

    triadsextet_t sextet = triadsextet_pack(t0, t1, t2);
    // Expected: (0, 0, 0) in base-3 offset = 0
    ASSERT_EQ(sextet, 0);

    TriadSextetUnpacked result = triadsextet_unpack(sextet);
    ASSERT_EQ(result.t0, 0b00);
    ASSERT_EQ(result.t1, 0b00);
    ASSERT_EQ(result.t2, 0b00);
}

TEST(exhaustive_roundtrip) {
    // Test all 27 valid sextet encodings
    int errors = 0;

    for (int o0 = 0; o0 < 3; ++o0) {
        for (int o1 = 0; o1 < 3; ++o1) {
            for (int o2 = 0; o2 < 3; ++o2) {
                // Convert offsets {0,1,2} to trits {-1,0,+1}
                uint8_t t0 = int_to_trit(o0 - 1);
                uint8_t t1 = int_to_trit(o1 - 1);
                uint8_t t2 = int_to_trit(o2 - 1);

                // Pack
                triadsextet_t sextet = triadsextet_pack(t0, t1, t2);

                // Verify sextet is in valid range
                if (sextet > SEXTET_MAX_VALID) {
                    errors++;
                    continue;
                }

                // Unpack
                TriadSextetUnpacked result = triadsextet_unpack(sextet);

                // Verify round-trip
                if (result.t0 != t0 || result.t1 != t1 || result.t2 != t2) {
                    errors++;
                }
            }
        }
    }

    ASSERT_EQ(errors, 0);
}

TEST(invalid_sextets) {
    // Test that invalid sextets (27-63) map to zeros
    for (uint8_t invalid_sextet = 27; invalid_sextet < 64; ++invalid_sextet) {
        TriadSextetUnpacked result = triadsextet_unpack(invalid_sextet);

        // All should be neutral zero trits (0b01)
        ASSERT_EQ(result.t0, 0b01);
        ASSERT_EQ(result.t1, 0b01);
        ASSERT_EQ(result.t2, 0b01);

        // Also test validation function
        ASSERT_EQ(triadsextet_is_valid(invalid_sextet), false);
    }
}

TEST(extract_single_trit) {
    // Test case: (+1, 0, -1) → sextet = 5
    triadsextet_t sextet = triadsextet_pack(0b10, 0b01, 0b00);

    ASSERT_EQ(triadsextet_extract_trit(sextet, 0), 0b10);  // +1
    ASSERT_EQ(triadsextet_extract_trit(sextet, 1), 0b01);  //  0
    ASSERT_EQ(triadsextet_extract_trit(sextet, 2), 0b00);  // -1

    // Invalid position should return neutral zero
    ASSERT_EQ(triadsextet_extract_trit(sextet, 3), 0b01);
}

TEST(integer_pack_unpack) {
    // Test FFI-friendly integer interface
    int8_t t0 = +1, t1 = 0, t2 = -1;

    triadsextet_t sextet = triadsextet_pack_int(t0, t1, t2);
    ASSERT_EQ(sextet, 5);  // Same as 2-bit test

    int8_t r0, r1, r2;
    triadsextet_unpack_int(sextet, &r0, &r1, &r2);

    ASSERT_EQ(r0, +1);
    ASSERT_EQ(r1, 0);
    ASSERT_EQ(r2, -1);
}

TEST(array_size_conversions) {
    // Test sextet count calculations
    ASSERT_EQ(triadsextet_sextets_for_trits(0), 0);
    ASSERT_EQ(triadsextet_sextets_for_trits(1), 1);   // 1 trit  → 1 sextet (2 unused)
    ASSERT_EQ(triadsextet_sextets_for_trits(3), 1);   // 3 trits → 1 sextet (exact)
    ASSERT_EQ(triadsextet_sextets_for_trits(4), 2);   // 4 trits → 2 sextets
    ASSERT_EQ(triadsextet_sextets_for_trits(6), 2);   // 6 trits → 2 sextets (exact)
    ASSERT_EQ(triadsextet_sextets_for_trits(7), 3);   // 7 trits → 3 sextets

    // Test trit count calculations
    ASSERT_EQ(triadsextet_trits_in_sextets(0), 0);
    ASSERT_EQ(triadsextet_trits_in_sextets(1), 3);
    ASSERT_EQ(triadsextet_trits_in_sextets(2), 6);
    ASSERT_EQ(triadsextet_trits_in_sextets(100), 300);

    // Test byte count (byte-aligned storage)
    ASSERT_EQ(triadsextet_bytes_for_trits(3), 1);
    ASSERT_EQ(triadsextet_bytes_for_trits(6), 2);
    ASSERT_EQ(triadsextet_bytes_for_trits(9), 3);
}

TEST(array_validation) {
    // Valid array
    triadsextet_t valid_data[] = {0, 1, 2, 13, 25, 26};
    ASSERT_EQ(triadsextet_validate_array(valid_data, 6), 0);

    // Mixed array
    triadsextet_t mixed_data[] = {0, 1, 27, 13, 63, 26};
    ASSERT_EQ(triadsextet_validate_array(mixed_data, 6), 2);  // 2 invalid

    // All invalid
    triadsextet_t invalid_data[] = {27, 30, 40, 63};
    ASSERT_EQ(triadsextet_validate_array(invalid_data, 4), 4);
}

TEST(sanitization) {
    // Valid sextets pass through
    ASSERT_EQ(triadsextet_sanitize(0), 0);
    ASSERT_EQ(triadsextet_sanitize(13), 13);
    ASSERT_EQ(triadsextet_sanitize(26), 26);

    // Invalid sextets → 0
    ASSERT_EQ(triadsextet_sanitize(27), 0);
    ASSERT_EQ(triadsextet_sanitize(63), 0);

    // High bits are masked
    ASSERT_EQ(triadsextet_sanitize(0xFF), 0);  // 0xFF & 0x3F = 63 → invalid → 0
}

// =============================================================================
// Ternary Operations Tests
// =============================================================================

TEST(operations_tadd) {
    // (+1) + (+1) = (+1) (saturated)
    triadsextet_t s1 = triadsextet_pack_int(+1, 0, -1);
    triadsextet_t s2 = triadsextet_pack_int(+1, +1, +1);
    triadsextet_t result = triadsextet_tadd(s1, s2);

    int8_t r0, r1, r2;
    triadsextet_unpack_int(result, &r0, &r1, &r2);

    ASSERT_EQ(r0, +1);  // +1 + +1 = +1 (saturated)
    ASSERT_EQ(r1, +1);  //  0 + +1 = +1
    ASSERT_EQ(r2, 0);   // -1 + +1 =  0
}

TEST(operations_tmul) {
    // Multiplication
    triadsextet_t s1 = triadsextet_pack_int(+1, 0, -1);
    triadsextet_t s2 = triadsextet_pack_int(+1, +1, +1);
    triadsextet_t result = triadsextet_tmul(s1, s2);

    int8_t r0, r1, r2;
    triadsextet_unpack_int(result, &r0, &r1, &r2);

    ASSERT_EQ(r0, +1);  // +1 * +1 = +1
    ASSERT_EQ(r1, 0);   //  0 * +1 =  0
    ASSERT_EQ(r2, -1);  // -1 * +1 = -1
}

TEST(operations_tnot) {
    // Negation
    triadsextet_t s = triadsextet_pack_int(+1, 0, -1);
    triadsextet_t result = triadsextet_tnot(s);

    int8_t r0, r1, r2;
    triadsextet_unpack_int(result, &r0, &r1, &r2);

    ASSERT_EQ(r0, -1);  // -( +1) = -1
    ASSERT_EQ(r1, 0);   // -(  0) =  0
    ASSERT_EQ(r2, +1);  // -( -1) = +1
}

TEST(operations_tmin_tmax) {
    triadsextet_t s1 = triadsextet_pack_int(+1, 0, -1);
    triadsextet_t s2 = triadsextet_pack_int(0, +1, 0);

    // Min
    triadsextet_t min_result = triadsextet_tmin(s1, s2);
    int8_t m0, m1, m2;
    triadsextet_unpack_int(min_result, &m0, &m1, &m2);
    ASSERT_EQ(m0, 0);   // min(+1, 0) =  0
    ASSERT_EQ(m1, 0);   // min( 0, +1) = 0
    ASSERT_EQ(m2, -1);  // min(-1, 0) = -1

    // Max
    triadsextet_t max_result = triadsextet_tmax(s1, s2);
    int8_t x0, x1, x2;
    triadsextet_unpack_int(max_result, &x0, &x1, &x2);
    ASSERT_EQ(x0, +1);  // max(+1, 0) = +1
    ASSERT_EQ(x1, +1);  // max( 0, +1) = +1
    ASSERT_EQ(x2, 0);   // max(-1, 0) =  0
}

// =============================================================================
// Performance Benchmark
// =============================================================================

TEST(benchmark_pack_unpack) {
    const size_t iterations = 1000000;

    // Prepare test data: 3 trits per iteration
    std::vector<uint8_t> t0_data(iterations), t1_data(iterations), t2_data(iterations);

    for (size_t i = 0; i < iterations; ++i) {
        t0_data[i] = int_to_trit((i % 3) - 1);
        t1_data[i] = int_to_trit(((i / 3) % 3) - 1);
        t2_data[i] = int_to_trit(((i / 9) % 3) - 1);
    }

    std::vector<triadsextet_t> sextet_data(iterations);

    // Benchmark packing
    auto start_pack = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        sextet_data[i] = triadsextet_pack(t0_data[i], t1_data[i], t2_data[i]);
    }
    auto end_pack = std::chrono::high_resolution_clock::now();

    // Benchmark unpacking
    size_t verify_sum = 0;
    auto start_unpack = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        TriadSextetUnpacked result = triadsextet_unpack(sextet_data[i]);
        verify_sum += result.t0 + result.t1 + result.t2;
    }
    auto end_unpack = std::chrono::high_resolution_clock::now();

    // Calculate timings
    auto pack_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_pack - start_pack).count();
    auto unpack_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_unpack - start_unpack).count();

    double pack_ns_per_op = static_cast<double>(pack_ns) / iterations;
    double unpack_ns_per_op = static_cast<double>(unpack_ns) / iterations;

    std::cout << "\n";
    std::cout << "  Pack:   " << std::fixed << std::setprecision(2) << pack_ns_per_op << " ns/op\n";
    std::cout << "  Unpack: " << std::fixed << std::setprecision(2) << unpack_ns_per_op << " ns/op\n";
    std::cout << "  Total:  " << std::fixed << std::setprecision(2) << (pack_ns_per_op + unpack_ns_per_op) << " ns/op\n";
    std::cout << "  (verify_sum = " << verify_sum << " to prevent optimization)\n";
}

// =============================================================================
// Main Test Runner
// =============================================================================

int main() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << " TriadSextet Unit Tests\n";
    std::cout << "========================================\n\n";

    // Tests are auto-registered and run via static initializers
    // Results are already printed

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << " Test Results\n";
    std::cout << "========================================\n";
    std::cout << "Passed: " << COLOR_GREEN << g_tests_passed << COLOR_RESET << "\n";
    std::cout << "Failed: " << (g_tests_failed > 0 ? COLOR_RED : COLOR_GREEN)
              << g_tests_failed << COLOR_RESET << "\n";
    std::cout << "========================================\n\n";

    return (g_tests_failed == 0) ? 0 : 1;
}
