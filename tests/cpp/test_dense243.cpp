// test_dense243.cpp — Unit tests for T5-Dense243 encoding
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
#include "../src/engine/dense243/ternary_dense243.h"
#include "../src/core/algebra/ternary_algebra.h"

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
    // Test case: (+1, 0, -1, +1, 0)
    uint8_t t0 = 0b10;  // +1
    uint8_t t1 = 0b01;  //  0
    uint8_t t2 = 0b00;  // -1
    uint8_t t3 = 0b10;  // +1
    uint8_t t4 = 0b01;  //  0

    // Pack
    uint8_t packed = dense243_pack(t0, t1, t2, t3, t4);

    // Unpack
    Dense243Unpacked result = dense243_unpack(packed);

    // Verify round-trip
    ASSERT_EQ(result.t0, t0);
    ASSERT_EQ(result.t1, t1);
    ASSERT_EQ(result.t2, t2);
    ASSERT_EQ(result.t3, t3);
    ASSERT_EQ(result.t4, t4);
}

TEST(all_zeros) {
    // Test case: (0, 0, 0, 0, 0)
    uint8_t t0 = 0b01, t1 = 0b01, t2 = 0b01, t3 = 0b01, t4 = 0b01;

    uint8_t packed = dense243_pack(t0, t1, t2, t3, t4);
    Dense243Unpacked result = dense243_unpack(packed);

    ASSERT_EQ(result.t0, 0b01);
    ASSERT_EQ(result.t1, 0b01);
    ASSERT_EQ(result.t2, 0b01);
    ASSERT_EQ(result.t3, 0b01);
    ASSERT_EQ(result.t4, 0b01);
}

TEST(all_positive) {
    // Test case: (+1, +1, +1, +1, +1)
    uint8_t t0 = 0b10, t1 = 0b10, t2 = 0b10, t3 = 0b10, t4 = 0b10;

    uint8_t packed = dense243_pack(t0, t1, t2, t3, t4);
    // Expected: (2, 2, 2, 2, 2) in base-3 offset
    // = 2 + 2*3 + 2*9 + 2*27 + 2*81 = 2 + 6 + 18 + 54 + 162 = 242
    ASSERT_EQ(packed, 242);

    Dense243Unpacked result = dense243_unpack(packed);
    ASSERT_EQ(result.t0, 0b10);
    ASSERT_EQ(result.t1, 0b10);
    ASSERT_EQ(result.t2, 0b10);
    ASSERT_EQ(result.t3, 0b10);
    ASSERT_EQ(result.t4, 0b10);
}

TEST(all_negative) {
    // Test case: (-1, -1, -1, -1, -1)
    uint8_t t0 = 0b00, t1 = 0b00, t2 = 0b00, t3 = 0b00, t4 = 0b00;

    uint8_t packed = dense243_pack(t0, t1, t2, t3, t4);
    // Expected: (0, 0, 0, 0, 0) in base-3 offset = 0
    ASSERT_EQ(packed, 0);

    Dense243Unpacked result = dense243_unpack(packed);
    ASSERT_EQ(result.t0, 0b00);
    ASSERT_EQ(result.t1, 0b00);
    ASSERT_EQ(result.t2, 0b00);
    ASSERT_EQ(result.t3, 0b00);
    ASSERT_EQ(result.t4, 0b00);
}

TEST(exhaustive_roundtrip) {
    // Test all 243 valid encodings
    int errors = 0;

    for (int o0 = 0; o0 < 3; ++o0) {
        for (int o1 = 0; o1 < 3; ++o1) {
            for (int o2 = 0; o2 < 3; ++o2) {
                for (int o3 = 0; o3 < 3; ++o3) {
                    for (int o4 = 0; o4 < 3; ++o4) {
                        // Convert offsets {0,1,2} to trits {-1,0,+1}
                        uint8_t t0 = int_to_trit(o0 - 1);
                        uint8_t t1 = int_to_trit(o1 - 1);
                        uint8_t t2 = int_to_trit(o2 - 1);
                        uint8_t t3 = int_to_trit(o3 - 1);
                        uint8_t t4 = int_to_trit(o4 - 1);

                        // Pack
                        uint8_t packed = dense243_pack(t0, t1, t2, t3, t4);

                        // Verify packed value is in valid range
                        if (packed >= 243) {
                            errors++;
                            continue;
                        }

                        // Unpack
                        Dense243Unpacked result = dense243_unpack(packed);

                        // Verify round-trip
                        if (result.t0 != t0 || result.t1 != t1 || result.t2 != t2 ||
                            result.t3 != t3 || result.t4 != t4) {
                            errors++;
                        }
                    }
                }
            }
        }
    }

    ASSERT_EQ(errors, 0);
}

TEST(invalid_bytes) {
    // Test that invalid bytes (243-255) map to zeros
    for (uint8_t invalid_byte = 243; invalid_byte != 0; ++invalid_byte) {
        Dense243Unpacked result = dense243_unpack(invalid_byte);

        // All should be neutral zero trits (0b01)
        ASSERT_EQ(result.t0, 0b01);
        ASSERT_EQ(result.t1, 0b01);
        ASSERT_EQ(result.t2, 0b01);
        ASSERT_EQ(result.t3, 0b01);
        ASSERT_EQ(result.t4, 0b01);

        // Also test validation function
        ASSERT_EQ(dense243_is_valid(invalid_byte), false);
    }
}

TEST(extract_single_trit) {
    // Test case: (+1, 0, -1, +1, 0) → packed = 140
    uint8_t packed = dense243_pack(0b10, 0b01, 0b00, 0b10, 0b01);

    ASSERT_EQ(dense243_extract_trit(packed, 0), 0b10);  // +1
    ASSERT_EQ(dense243_extract_trit(packed, 1), 0b01);  //  0
    ASSERT_EQ(dense243_extract_trit(packed, 2), 0b00);  // -1
    ASSERT_EQ(dense243_extract_trit(packed, 3), 0b10);  // +1
    ASSERT_EQ(dense243_extract_trit(packed, 4), 0b01);  //  0

    // Invalid position should return neutral zero
    ASSERT_EQ(dense243_extract_trit(packed, 5), 0b01);
}

TEST(array_size_conversions) {
    // Test byte count calculations
    ASSERT_EQ(dense243_bytes_for_trits(0), 0);
    ASSERT_EQ(dense243_bytes_for_trits(1), 1);   // 1 trit  → 1 byte (4 unused slots)
    ASSERT_EQ(dense243_bytes_for_trits(5), 1);   // 5 trits → 1 byte (exact)
    ASSERT_EQ(dense243_bytes_for_trits(6), 2);   // 6 trits → 2 bytes
    ASSERT_EQ(dense243_bytes_for_trits(10), 2);  // 10 trits → 2 bytes (5 each)
    ASSERT_EQ(dense243_bytes_for_trits(11), 3);  // 11 trits → 3 bytes

    // Test trit count calculations
    ASSERT_EQ(dense243_trits_in_bytes(0), 0);
    ASSERT_EQ(dense243_trits_in_bytes(1), 5);
    ASSERT_EQ(dense243_trits_in_bytes(2), 10);
    ASSERT_EQ(dense243_trits_in_bytes(100), 500);
}

TEST(array_validation) {
    // Valid array
    uint8_t valid_data[] = {0, 1, 2, 100, 200, 242};
    ASSERT_EQ(dense243_validate_array(valid_data, 6), 0);

    // Mixed array
    uint8_t mixed_data[] = {0, 1, 243, 100, 255, 242};
    ASSERT_EQ(dense243_validate_array(mixed_data, 6), 2);  // 2 invalid

    // All invalid
    uint8_t invalid_data[] = {243, 244, 245, 255};
    ASSERT_EQ(dense243_validate_array(invalid_data, 4), 4);
}

// =============================================================================
// Performance Benchmark
// =============================================================================

TEST(benchmark_pack_unpack) {
    const size_t iterations = 1000000;

    // Prepare test data: 5 trits per iteration
    std::vector<uint8_t> t0_data(iterations), t1_data(iterations), t2_data(iterations);
    std::vector<uint8_t> t3_data(iterations), t4_data(iterations);

    for (size_t i = 0; i < iterations; ++i) {
        t0_data[i] = int_to_trit((i % 3) - 1);
        t1_data[i] = int_to_trit(((i / 3) % 3) - 1);
        t2_data[i] = int_to_trit(((i / 9) % 3) - 1);
        t3_data[i] = int_to_trit(((i / 27) % 3) - 1);
        t4_data[i] = int_to_trit(((i / 81) % 3) - 1);
    }

    std::vector<uint8_t> packed_data(iterations);

    // Benchmark packing
    auto start_pack = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        packed_data[i] = dense243_pack(t0_data[i], t1_data[i], t2_data[i], t3_data[i], t4_data[i]);
    }
    auto end_pack = std::chrono::high_resolution_clock::now();

    // Benchmark unpacking
    size_t verify_sum = 0;
    auto start_unpack = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        Dense243Unpacked result = dense243_unpack(packed_data[i]);
        verify_sum += result.t0 + result.t1 + result.t2 + result.t3 + result.t4;
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
    std::cout << " T5-Dense243 Unit Tests\n";
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
