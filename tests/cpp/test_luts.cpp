// test_luts.cpp - Validate Phase 0 LUT optimizations
//
// Copyright (c) 2025 Jonathan Verdun (Ternary Core Experimental Project)
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
//
// Compile: g++ -std=c++17 -O0 test_luts.cpp -o test_luts && ./test_luts

#include <iostream>
#include <iomanip>
#include "../src/core/algebra/ternary_algebra.h"

// Reference implementations (pre-optimization) for comparison
namespace reference {
    inline int trit_to_int(uint8_t t) {
        return (t==0b00)?-1:(t==0b10)?1:0;
    }

    inline uint8_t int_to_trit(int v) {
        return (v < 0) ? 0b00 : (v > 0) ? 0b10 : 0b01;
    }

    inline uint8_t tadd(uint8_t a, uint8_t b) {
        int s = trit_to_int(a) + trit_to_int(b);
        if (s>1) s=1; if (s<-1) s=-1;
        return int_to_trit(s);
    }

    inline uint8_t tmul(uint8_t a, uint8_t b) {
        return int_to_trit(trit_to_int(a) * trit_to_int(b));
    }

    inline uint8_t tmin(uint8_t a, uint8_t b) {
        return (trit_to_int(a) < trit_to_int(b)) ? a : b;
    }

    inline uint8_t tmax(uint8_t a, uint8_t b) {
        return (trit_to_int(a) > trit_to_int(b)) ? a : b;
    }

    inline uint8_t tnot(uint8_t a) {
        return (a==0b00)?0b10:(a==0b10)?0b00:0b01;
    }
}

// Helper function to print trit value
const char* trit_name(uint8_t t) {
    switch(t) {
        case 0b00: return "-1 (0b00)";
        case 0b01: return " 0 (0b01)";
        case 0b10: return "+1 (0b10)";
        default:   return "?? (0b11)";
    }
}

// Test counters
int total_tests = 0;
int passed_tests = 0;
int failed_tests = 0;

void test_binary_op(const char* name,
                   uint8_t (*opt_func)(uint8_t, uint8_t),
                   uint8_t (*ref_func)(uint8_t, uint8_t)) {
    std::cout << "\n=== Testing " << name << " ===" << std::endl;

    uint8_t valid_trits[] = {0b00, 0b01, 0b10};
    bool all_passed = true;

    for (uint8_t a : valid_trits) {
        for (uint8_t b : valid_trits) {
            total_tests++;

            uint8_t opt_result = opt_func(a, b);
            uint8_t ref_result = ref_func(a, b);

            bool passed = (opt_result == ref_result);

            if (passed) {
                passed_tests++;
            } else {
                failed_tests++;
                all_passed = false;
                std::cout << "  FAIL: " << name << "(" << trit_name(a) << ", "
                         << trit_name(b) << ") = " << trit_name(opt_result)
                         << ", expected " << trit_name(ref_result) << std::endl;
            }
        }
    }

    if (all_passed) {
        std::cout << "  ✓ All 9 test cases passed" << std::endl;
    }
}

void test_unary_op(const char* name,
                  uint8_t (*opt_func)(uint8_t),
                  uint8_t (*ref_func)(uint8_t)) {
    std::cout << "\n=== Testing " << name << " ===" << std::endl;

    uint8_t valid_trits[] = {0b00, 0b01, 0b10};
    bool all_passed = true;

    for (uint8_t a : valid_trits) {
        total_tests++;

        uint8_t opt_result = opt_func(a);
        uint8_t ref_result = ref_func(a);

        bool passed = (opt_result == ref_result);

        if (passed) {
            passed_tests++;
        } else {
            failed_tests++;
            all_passed = false;
            std::cout << "  FAIL: " << name << "(" << trit_name(a)
                     << ") = " << trit_name(opt_result)
                     << ", expected " << trit_name(ref_result) << std::endl;
        }
    }

    if (all_passed) {
        std::cout << "  ✓ All 3 test cases passed" << std::endl;
    }
}

void print_operation_table(const char* name,
                          uint8_t (*func)(uint8_t, uint8_t)) {
    std::cout << "\n" << name << " truth table:" << std::endl;
    std::cout << "     -1(00)  0(01) +1(10)" << std::endl;
    std::cout << "    ────────────────────" << std::endl;

    uint8_t valid_trits[] = {0b00, 0b01, 0b10};
    const char* names[] = {"-1", " 0", "+1"};

    for (int i = 0; i < 3; i++) {
        std::cout << names[i] << " │ ";
        for (int j = 0; j < 3; j++) {
            uint8_t result = func(valid_trits[i], valid_trits[j]);
            const char* result_name;
            if (result == 0b00) result_name = "-1";
            else if (result == 0b01) result_name = " 0";
            else if (result == 0b10) result_name = "+1";
            else result_name = "??";
            std::cout << std::setw(6) << result_name << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Phase 0 LUT Optimization Test Suite  " << std::endl;
    std::cout << "========================================" << std::endl;

    // Test all operations
    test_binary_op("tadd", tadd, reference::tadd);
    test_binary_op("tmul", tmul, reference::tmul);
    test_binary_op("tmin", tmin, reference::tmin);
    test_binary_op("tmax", tmax, reference::tmax);
    test_unary_op("tnot", tnot, reference::tnot);

    // Print summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Test Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Total tests:  " << total_tests << std::endl;
    std::cout << "  Passed:       " << passed_tests << " ✓" << std::endl;
    std::cout << "  Failed:       " << failed_tests << (failed_tests > 0 ? " ✗" : "") << std::endl;

    if (failed_tests == 0) {
        std::cout << "\n  🎉 ALL TESTS PASSED! 🎉" << std::endl;
        std::cout << "  Phase 0 LUT optimizations are correct." << std::endl;
    } else {
        std::cout << "\n  ❌ TESTS FAILED" << std::endl;
        std::cout << "  Please review the LUT implementations." << std::endl;
    }

    // Print operation tables for verification
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Operation Truth Tables (Optimized)" << std::endl;
    std::cout << "========================================" << std::endl;

    print_operation_table("tadd", tadd);
    print_operation_table("tmul", tmul);
    print_operation_table("tmin", tmin);
    print_operation_table("tmax", tmax);

    std::cout << "\ntnot truth table:" << std::endl;
    std::cout << "  tnot(-1) = " << trit_name(tnot(0b00)) << std::endl;
    std::cout << "  tnot( 0) = " << trit_name(tnot(0b01)) << std::endl;
    std::cout << "  tnot(+1) = " << trit_name(tnot(0b10)) << std::endl;

    std::cout << "\n========================================\n" << std::endl;

    return (failed_tests > 0) ? 1 : 0;
}
