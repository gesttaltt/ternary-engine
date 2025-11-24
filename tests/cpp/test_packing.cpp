/**
 * test_packing.cpp - Unit tests for Sixtet and Octet encoding
 *
 * Copyright 2025 Ternary Engine Contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Comprehensive tests for packing layer:
 * - Sixtet encoding (3 trits → 6 bits)
 * - Octet encoding (2 trits → 4 bits)
 * - Round-trip validation
 * - Edge cases and error handling
 * - Compression ratio validation
 */

#include <iostream>
#include <cassert>
#include <cstring>
#include <cstdio>

// Include both packing headers
#include "../src/core/packing/sixtet_pack.h"
#include "../src/core/packing/octet_pack.h"

// Test counters
static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    do { \
        tests_run++; \
        std::cout << "Running: " << name << "..."; \
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

#define ASSERT_TRUE(cond) \
    do { \
        if (cond) { \
            std::cout << " PASS\n"; \
            tests_passed++; \
        } else { \
            std::cout << " FAIL: " << #cond << " is false\n"; \
        } \
    } while(0)

// ============================================================================
// Sixtet Tests
// ============================================================================

void test_sixtet_pack_basic() {
    TEST("Sixtet pack basic trits");

    // Test all 27 valid combinations
    uint8_t packed;

    // (-1, -1, -1) → 0b00'00'00 = 0x00
    packed = sixtet_pack(-1, -1, -1);
    ASSERT_EQ(packed, 0x00);
}

void test_sixtet_unpack_basic() {
    TEST("Sixtet unpack basic");

    int8_t t0, t1, t2;

    // 0x00 → (-1, -1, -1)
    sixtet_unpack(0x00, &t0, &t1, &t2);
    ASSERT_TRUE(t0 == -1 && t1 == -1 && t2 == -1);
}

void test_sixtet_round_trip() {
    TEST("Sixtet round-trip all 27 states");

    int8_t trits[3] = {-1, 0, 1};
    bool all_passed = true;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                int8_t t0 = trits[i];
                int8_t t1 = trits[j];
                int8_t t2 = trits[k];

                // Pack
                uint8_t packed = sixtet_pack(t0, t1, t2);

                // Unpack
                int8_t u0, u1, u2;
                sixtet_unpack(packed, &u0, &u1, &u2);

                // Verify
                if (u0 != t0 || u1 != t1 || u2 != t2) {
                    all_passed = false;
                    std::cout << " FAIL at (" << (int)t0 << "," << (int)t1 << "," << (int)t2 << ")\n";
                    return;
                }
            }
        }
    }

    ASSERT_TRUE(all_passed);
}

void test_sixtet_validation() {
    TEST("Sixtet validation");

    // Valid encodings
    bool valid = true;
    valid &= sixtet_is_valid(0x00);  // (-1, -1, -1)
    valid &= sixtet_is_valid(0x15);  // (0, 0, 0)
    valid &= sixtet_is_valid(0x2A);  // (1, 1, 1)

    // Invalid encodings (contain 11 bits)
    valid &= !sixtet_is_valid(0x03);  // t0 = 11
    valid &= !sixtet_is_valid(0x0C);  // t1 = 11
    valid &= !sixtet_is_valid(0x30);  // t2 = 11

    ASSERT_TRUE(valid);
}

void test_sixtet_array_pack() {
    TEST("Sixtet array pack");

    // Input: 6 trits (2 Sextets)
    int8_t trits[6] = {-1, 0, 1, 1, 0, -1};
    uint8_t packed[2];

    int result = sixtet_pack_array(trits, 6, packed);

    ASSERT_EQ(result, 2);
}

void test_sixtet_array_unpack() {
    TEST("Sixtet array unpack");

    // Pack then unpack
    int8_t trits_in[9] = {-1, 0, 1, 0, 0, 0, 1, 0, -1};
    uint8_t packed[3];
    int8_t trits_out[9];

    sixtet_pack_array(trits_in, 9, packed);
    int result = sixtet_unpack_array(packed, 3, trits_out);

    bool match = (result == 9);
    for (int i = 0; i < 9; i++) {
        match &= (trits_in[i] == trits_out[i]);
    }

    ASSERT_TRUE(match);
}

void test_sixtet_compression_ratio() {
    TEST("Sixtet compression ratio");

    // 3 trits: 3 bytes uncompressed → 1 byte compressed = 3.0× ratio
    float ratio = sixtet_compression_ratio(3);
    ASSERT_TRUE(ratio > 2.99f && ratio < 3.01f);
}

// ============================================================================
// Octet Tests
// ============================================================================

void test_octet_pack_basic() {
    TEST("Octet pack basic trits");

    // Test all 9 valid combinations
    uint8_t packed;

    // (-1, -1) → 0b0000'00'00 = 0x00
    packed = octet_pack(-1, -1);
    ASSERT_EQ(packed, 0x00);
}

void test_octet_unpack_basic() {
    TEST("Octet unpack basic");

    int8_t t0, t1;

    // 0x00 → (-1, -1)
    octet_unpack(0x00, &t0, &t1);
    ASSERT_TRUE(t0 == -1 && t1 == -1);
}

void test_octet_round_trip() {
    TEST("Octet round-trip all 9 states");

    int8_t trits[3] = {-1, 0, 1};
    bool all_passed = true;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int8_t t0 = trits[i];
            int8_t t1 = trits[j];

            // Pack
            uint8_t packed = octet_pack(t0, t1);

            // Unpack
            int8_t u0, u1;
            octet_unpack(packed, &u0, &u1);

            // Verify
            if (u0 != t0 || u1 != t1) {
                all_passed = false;
                std::cout << " FAIL at (" << (int)t0 << "," << (int)t1 << ")\n";
                return;
            }
        }
    }

    ASSERT_TRUE(all_passed);
}

void test_octet_validation() {
    TEST("Octet validation");

    // Valid encodings
    bool valid = true;
    valid &= octet_is_valid(0x00);  // (-1, -1)
    valid &= octet_is_valid(0x05);  // (0, 0)
    valid &= octet_is_valid(0x0A);  // (1, 1)

    // Invalid encodings (contain 11 bits)
    valid &= !octet_is_valid(0x03);  // t0 = 11
    valid &= !octet_is_valid(0x0C);  // t1 = 11

    ASSERT_TRUE(valid);
}

void test_octet_array_pack() {
    TEST("Octet array pack");

    // Input: 4 trits (2 Octets)
    int8_t trits[4] = {-1, 0, 1, 1};
    uint8_t packed[2];

    int result = octet_pack_array(trits, 4, packed);

    ASSERT_EQ(result, 2);
}

void test_octet_array_unpack() {
    TEST("Octet array unpack");

    // Pack then unpack
    int8_t trits_in[6] = {-1, 0, 0, 0, 1, -1};
    uint8_t packed[3];
    int8_t trits_out[6];

    octet_pack_array(trits_in, 6, packed);
    int result = octet_unpack_array(packed, 3, trits_out);

    bool match = (result == 6);
    for (int i = 0; i < 6; i++) {
        match &= (trits_in[i] == trits_out[i]);
    }

    ASSERT_TRUE(match);
}

void test_octet_compression_ratio() {
    TEST("Octet compression ratio");

    // 2 trits: 2 bytes uncompressed → 1 byte compressed = 2.0× ratio
    float ratio = octet_compression_ratio(2);
    ASSERT_TRUE(ratio > 1.99f && ratio < 2.01f);
}

// ============================================================================
// Comparison Tests
// ============================================================================

void test_sixtet_vs_octet_efficiency() {
    TEST("Sixtet vs Octet efficiency comparison");

    bool use_sixtet;

    // For 6 trits: Sixtet needs 2 bytes, Octet needs 3 bytes
    int saved = octet_vs_sixtet_efficiency(6, &use_sixtet);
    ASSERT_TRUE(use_sixtet && saved == 1);
}

void test_mixed_encoding() {
    TEST("Mixed Sixtet/Octet encoding");

    // Scenario: 5 trits
    // Option 1: Pure Sixtet → 2 bytes (pad to 6 trits)
    // Option 2: Sixtet (3) + Octet (2) → 2 bytes (no padding)

    int8_t trits[5] = {-1, 0, 1, 1, 0};

    // Pack first 3 with Sixtet
    uint8_t sixtet_packed = sixtet_pack(trits[0], trits[1], trits[2]);

    // Pack last 2 with Octet
    uint8_t octet_packed = octet_pack(trits[3], trits[4]);

    // Unpack and verify
    int8_t s0, s1, s2, o0, o1;
    sixtet_unpack(sixtet_packed, &s0, &s1, &s2);
    octet_unpack(octet_packed, &o0, &o1);

    bool match = (s0 == trits[0] && s1 == trits[1] && s2 == trits[2] &&
                  o0 == trits[3] && o1 == trits[4]);

    ASSERT_TRUE(match);
}

// ============================================================================
// Edge Cases
// ============================================================================

void test_sixtet_invalid_input_size() {
    TEST("Sixtet invalid input size");

    int8_t trits[5] = {0, 0, 0, 0, 0};  // Not multiple of 3
    uint8_t packed[2];

    int result = sixtet_pack_array(trits, 5, packed);

    ASSERT_EQ(result, -1);  // Should return error
}

void test_octet_invalid_input_size() {
    TEST("Octet invalid input size");

    int8_t trits[5] = {0, 0, 0, 0, 0};  // Not multiple of 2
    uint8_t packed[3];

    int result = octet_pack_array(trits, 5, packed);

    ASSERT_EQ(result, -1);  // Should return error
}

void test_large_array_packing() {
    TEST("Large array packing (1000 trits)");

    const size_t N = 1002;  // Multiple of both 2 and 3 (LCM = 6)
    int8_t trits[N];
    uint8_t sixtet_packed[N/3];
    uint8_t octet_packed[N/2];
    int8_t trits_out[N];

    // Initialize with pattern
    for (size_t i = 0; i < N; i++) {
        trits[i] = (int8_t)((i % 3) - 1);  // -1, 0, 1, -1, 0, 1, ...
    }

    // Test Sixtet round-trip
    sixtet_pack_array(trits, N, sixtet_packed);
    sixtet_unpack_array(sixtet_packed, N/3, trits_out);

    bool sixtet_match = true;
    for (size_t i = 0; i < N; i++) {
        if (trits[i] != trits_out[i]) {
            sixtet_match = false;
            break;
        }
    }

    // Test Octet round-trip
    octet_pack_array(trits, N, octet_packed);
    octet_unpack_array(octet_packed, N/2, trits_out);

    bool octet_match = true;
    for (size_t i = 0; i < N; i++) {
        if (trits[i] != trits_out[i]) {
            octet_match = false;
            break;
        }
    }

    ASSERT_TRUE(sixtet_match && octet_match);
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "===========================================\n";
    std::cout << "Ternary Packing Tests (Sixtet + Octet)\n";
    std::cout << "===========================================\n\n";

    // Sixtet tests
    std::cout << "--- Sixtet Tests ---\n";
    test_sixtet_pack_basic();
    test_sixtet_unpack_basic();
    test_sixtet_round_trip();
    test_sixtet_validation();
    test_sixtet_array_pack();
    test_sixtet_array_unpack();
    test_sixtet_compression_ratio();
    test_sixtet_invalid_input_size();

    std::cout << "\n--- Octet Tests ---\n";
    test_octet_pack_basic();
    test_octet_unpack_basic();
    test_octet_round_trip();
    test_octet_validation();
    test_octet_array_pack();
    test_octet_array_unpack();
    test_octet_compression_ratio();
    test_octet_invalid_input_size();

    std::cout << "\n--- Comparison & Mixed Tests ---\n";
    test_sixtet_vs_octet_efficiency();
    test_mixed_encoding();
    test_large_array_packing();

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
