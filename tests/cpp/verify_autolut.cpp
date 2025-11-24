// verify_autolut.cpp - Verify constexpr-generated LUTs match manual baseline
//
// Copyright (c) 2025 Jonathan Verdun (Ternary Core Experimental Project)
// Licensed under the Apache License, Version 2.0
//
// Compile: g++ -std=c++20 -O0 -I.. verify_autolut.cpp -o verify_autolut && ./verify_autolut

#include <iostream>
#include <iomanip>
#include <bitset>
#include "../src/core/algebra/ternary_algebra.h"

// Manual baseline LUTs (from BASELINE-PRE-AUTOLUT.md)
static const uint8_t TADD_LUT_MANUAL[16] = {
    0b00, 0b00, 0b01, 0b00,
    0b00, 0b01, 0b10, 0b00,
    0b01, 0b10, 0b10, 0b00,
    0b00, 0b00, 0b00, 0b00
};

static const uint8_t TMUL_LUT_MANUAL[16] = {
    0b10, 0b01, 0b00, 0b00,
    0b01, 0b01, 0b01, 0b00,
    0b00, 0b01, 0b10, 0b00,
    0b00, 0b00, 0b00, 0b00
};

static const uint8_t TMIN_LUT_MANUAL[16] = {
    0b00, 0b00, 0b00, 0b00,
    0b00, 0b01, 0b01, 0b00,
    0b00, 0b01, 0b10, 0b00,
    0b00, 0b00, 0b00, 0b00
};

static const uint8_t TMAX_LUT_MANUAL[16] = {
    0b00, 0b01, 0b10, 0b00,
    0b01, 0b01, 0b10, 0b00,
    0b10, 0b10, 0b10, 0b00,
    0b00, 0b00, 0b00, 0b00
};

static const uint8_t TNOT_LUT_MANUAL[4] = {
    0b10,  // tnot(-1) = +1
    0b01,  // tnot(0) = 0
    0b00,  // tnot(+1) = -1
    0b00   // tnot(invalid) = undefined
};

bool verify_binary_lut(const char* name, const auto& generated, const uint8_t* manual) {
    std::cout << "\n=== Verifying " << name << " ===" << std::endl;
    bool all_match = true;

    for (size_t i = 0; i < 16; ++i) {
        if (generated[i] != manual[i]) {
            all_match = false;
            std::cout << "  MISMATCH at index " << i << ": "
                     << "generated=0b" << std::bitset<2>(generated[i])
                     << " manual=0b" << std::bitset<2>(manual[i]) << std::endl;
        }
    }

    if (all_match) {
        std::cout << "  ✓ All 16 entries match!" << std::endl;
    }

    return all_match;
}

bool verify_unary_lut(const char* name, const auto& generated, const uint8_t* manual) {
    std::cout << "\n=== Verifying " << name << " ===" << std::endl;
    bool all_match = true;

    for (size_t i = 0; i < 4; ++i) {
        if (generated[i] != manual[i]) {
            all_match = false;
            std::cout << "  MISMATCH at index " << i << ": "
                     << "generated=0b" << std::bitset<2>(generated[i])
                     << " manual=0b" << std::bitset<2>(manual[i]) << std::endl;
        }
    }

    if (all_match) {
        std::cout << "  ✓ All 4 entries match!" << std::endl;
    }

    return all_match;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Constexpr LUT Verification Suite" << std::endl;
    std::cout << "========================================" << std::endl;

    int total = 0;
    int passed = 0;

    // Verify binary LUTs
    if (verify_binary_lut("TADD_LUT", TADD_LUT, TADD_LUT_MANUAL)) passed++;
    total++;

    if (verify_binary_lut("TMUL_LUT", TMUL_LUT, TMUL_LUT_MANUAL)) passed++;
    total++;

    if (verify_binary_lut("TMIN_LUT", TMIN_LUT, TMIN_LUT_MANUAL)) passed++;
    total++;

    if (verify_binary_lut("TMAX_LUT", TMAX_LUT, TMAX_LUT_MANUAL)) passed++;
    total++;

    // Verify unary LUT
    if (verify_unary_lut("TNOT_LUT", TNOT_LUT, TNOT_LUT_MANUAL)) passed++;
    total++;

    // Summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Test Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Total LUTs tested: " << total << std::endl;
    std::cout << "  Passed:            " << passed << " ✓" << std::endl;
    std::cout << "  Failed:            " << (total - passed) << std::endl;

    if (passed == total) {
        std::cout << "\n  🎉 ALL LUTS VERIFIED! 🎉" << std::endl;
        std::cout << "  Constexpr generation produces identical results." << std::endl;
        return 0;
    } else {
        std::cout << "\n  ❌ VERIFICATION FAILED" << std::endl;
        std::cout << "  Generated LUTs differ from manual baseline." << std::endl;
        return 1;
    }
}
