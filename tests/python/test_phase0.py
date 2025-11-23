"""
test_phase0.py - Phase 0 Optimization Validation

Copyright 2025 Ternary Core Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tests the LUT-based scalar operations against expected ternary arithmetic rules.
Run after compiling the optimized module.

Usage:
    python test_phase0.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

try:
    import ternary_simd_engine as tc
except ImportError:
    print("ERROR: Module not compiled. Please compile first:")
    print("  python build.py")
    exit(1)

# Encoding: 0b00=-1, 0b01=0, 0b10=+1
MINUS_ONE = 0b00
ZERO = 0b01
PLUS_ONE = 0b10

def trit_name(t):
    """Convert trit encoding to readable name"""
    if t == MINUS_ONE: return "-1"
    if t == ZERO: return " 0"
    if t == PLUS_ONE: return "+1"
    return "??"

def test_operation(name, op_func, test_cases):
    """Test an operation against expected results"""
    print(f"\n=== Testing {name} ===")
    all_passed = True

    for inputs, expected in test_cases:
        if len(inputs) == 1:
            # Unary operation
            result = op_func(np.array([inputs[0]], dtype=np.uint8))[0]
            input_str = f"{name}({trit_name(inputs[0])})"
        else:
            # Binary operation
            a, b = inputs
            result = op_func(
                np.array([a], dtype=np.uint8),
                np.array([b], dtype=np.uint8)
            )[0]
            input_str = f"{name}({trit_name(a)}, {trit_name(b)})"

        passed = (result == expected)
        status = "[PASS]" if passed else "[FAIL]"

        if not passed:
            print(f"  {status} {input_str} = {trit_name(result)}, expected {trit_name(expected)}")
            all_passed = False

    if all_passed:
        print(f"  [OK] All {len(test_cases)} test cases passed")

    return all_passed

def print_truth_table(name, op_func, is_unary=False):
    """Print truth table for an operation"""
    print(f"\n{name} truth table:")

    if is_unary:
        for t in [MINUS_ONE, ZERO, PLUS_ONE]:
            result = op_func(np.array([t], dtype=np.uint8))[0]
            print(f"  {name}({trit_name(t)}) = {trit_name(result)}")
    else:
        print("     -1   0  +1")
        print("    -----------")
        for a in [MINUS_ONE, ZERO, PLUS_ONE]:
            print(f"{trit_name(a)} | ", end="")
            for b in [MINUS_ONE, ZERO, PLUS_ONE]:
                result = op_func(
                    np.array([a], dtype=np.uint8),
                    np.array([b], dtype=np.uint8)
                )[0]
                print(f"{trit_name(result)} ", end="")
            print()

def main():
    print("=" * 50)
    print("  Phase 0 LUT Optimization Test Suite (Python)")
    print("=" * 50)

    # Define test cases for each operation
    # Format: ([inputs], expected_output)

    tadd_tests = [
        # Addition with saturation [-1, +1]
        ([MINUS_ONE, MINUS_ONE], MINUS_ONE),  # -1 + -1 = -1 (saturated)
        ([MINUS_ONE, ZERO], MINUS_ONE),        # -1 + 0 = -1
        ([MINUS_ONE, PLUS_ONE], ZERO),         # -1 + 1 = 0
        ([ZERO, MINUS_ONE], MINUS_ONE),        # 0 + -1 = -1
        ([ZERO, ZERO], ZERO),                  # 0 + 0 = 0
        ([ZERO, PLUS_ONE], PLUS_ONE),          # 0 + 1 = +1
        ([PLUS_ONE, MINUS_ONE], ZERO),         # 1 + -1 = 0
        ([PLUS_ONE, ZERO], PLUS_ONE),          # 1 + 0 = +1
        ([PLUS_ONE, PLUS_ONE], PLUS_ONE),      # 1 + 1 = +1 (saturated)
    ]

    tmul_tests = [
        # Multiplication
        ([MINUS_ONE, MINUS_ONE], PLUS_ONE),    # -1 * -1 = +1
        ([MINUS_ONE, ZERO], ZERO),             # -1 * 0 = 0
        ([MINUS_ONE, PLUS_ONE], MINUS_ONE),    # -1 * 1 = -1
        ([ZERO, MINUS_ONE], ZERO),             # 0 * -1 = 0
        ([ZERO, ZERO], ZERO),                  # 0 * 0 = 0
        ([ZERO, PLUS_ONE], ZERO),              # 0 * 1 = 0
        ([PLUS_ONE, MINUS_ONE], MINUS_ONE),    # 1 * -1 = -1
        ([PLUS_ONE, ZERO], ZERO),              # 1 * 0 = 0
        ([PLUS_ONE, PLUS_ONE], PLUS_ONE),      # 1 * 1 = +1
    ]

    tmin_tests = [
        # Minimum
        ([MINUS_ONE, MINUS_ONE], MINUS_ONE),   # min(-1, -1) = -1
        ([MINUS_ONE, ZERO], MINUS_ONE),        # min(-1, 0) = -1
        ([MINUS_ONE, PLUS_ONE], MINUS_ONE),    # min(-1, +1) = -1
        ([ZERO, MINUS_ONE], MINUS_ONE),        # min(0, -1) = -1
        ([ZERO, ZERO], ZERO),                  # min(0, 0) = 0
        ([ZERO, PLUS_ONE], ZERO),              # min(0, +1) = 0
        ([PLUS_ONE, MINUS_ONE], MINUS_ONE),    # min(+1, -1) = -1
        ([PLUS_ONE, ZERO], ZERO),              # min(+1, 0) = 0
        ([PLUS_ONE, PLUS_ONE], PLUS_ONE),      # min(+1, +1) = +1
    ]

    tmax_tests = [
        # Maximum
        ([MINUS_ONE, MINUS_ONE], MINUS_ONE),   # max(-1, -1) = -1
        ([MINUS_ONE, ZERO], ZERO),             # max(-1, 0) = 0
        ([MINUS_ONE, PLUS_ONE], PLUS_ONE),     # max(-1, +1) = +1
        ([ZERO, MINUS_ONE], ZERO),             # max(0, -1) = 0
        ([ZERO, ZERO], ZERO),                  # max(0, 0) = 0
        ([ZERO, PLUS_ONE], PLUS_ONE),          # max(0, +1) = +1
        ([PLUS_ONE, MINUS_ONE], PLUS_ONE),     # max(+1, -1) = +1
        ([PLUS_ONE, ZERO], PLUS_ONE),          # max(+1, 0) = +1
        ([PLUS_ONE, PLUS_ONE], PLUS_ONE),      # max(+1, +1) = +1
    ]

    tnot_tests = [
        # Negation
        ([MINUS_ONE], PLUS_ONE),               # not(-1) = +1
        ([ZERO], ZERO),                        # not(0) = 0
        ([PLUS_ONE], MINUS_ONE),               # not(+1) = -1
    ]

    # Run tests
    results = []
    results.append(test_operation("tadd", tc.tadd, tadd_tests))
    results.append(test_operation("tmul", tc.tmul, tmul_tests))
    results.append(test_operation("tmin", tc.tmin, tmin_tests))
    results.append(test_operation("tmax", tc.tmax, tmax_tests))
    results.append(test_operation("tnot", tc.tnot, tnot_tests))

    # Summary
    total_ops = len(results)
    passed_ops = sum(results)

    print("\n" + "=" * 50)
    print("  Test Summary")
    print("=" * 50)
    print(f"  Operations tested: {total_ops}")
    print(f"  Passed: {passed_ops} [OK]")
    print(f"  Failed: {total_ops - passed_ops}" + (" [FAIL]" if passed_ops < total_ops else ""))

    if all(results):
        print("\n  [SUCCESS] ALL TESTS PASSED!")
        print("  Phase 0 LUT optimizations are correct.")
    else:
        print("\n  [FAIL] TESTS FAILED")
        print("  Please review the LUT implementations.")

    # Print truth tables
    print("\n" + "=" * 50)
    print("  Operation Truth Tables")
    print("=" * 50)

    print_truth_table("tadd", tc.tadd)
    print_truth_table("tmul", tc.tmul)
    print_truth_table("tmin", tc.tmin)
    print_truth_table("tmax", tc.tmax)
    print_truth_table("tnot", tc.tnot, is_unary=True)

    print("\n" + "=" * 50 + "\n")

    return 0 if all(results) else 1

if __name__ == "__main__":
    exit(main())
