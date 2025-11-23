#!/usr/bin/env python3
"""
test_fusion.py - Test suite for operation fusion engine

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Tests correctness and behavior of fused operations in ternary_simd_engine.
Validates that fused operations produce identical results to separate operations.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path (modules are in project root)
# File is in tests/python/, so go up 3 levels: test_fusion.py -> python/ -> tests/ -> project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Import engine (fusion operations now integrated into main module)
import ternary_simd_engine as base
import ternary_simd_engine as fusion  # Fusion ops are now in main module

# Encoding constants
MINUS_ONE = 0b00
ZERO = 0b01
PLUS_ONE = 0b10

def print_header(message):
    """Print test section header"""
    print("\n" + "="*70)
    print(f"  {message}")
    print("="*70 + "\n")

def test_fused_tnot_tadd():
    """Test fused_tnot_tadd correctness"""
    print("=== Testing fused_tnot_tadd ===")

    # Test all combinations
    test_cases = [
        ([MINUS_ONE], [MINUS_ONE], "tnot(tadd(-1, -1)) = tnot(-1) = +1"),
        ([MINUS_ONE], [ZERO], "tnot(tadd(-1, 0)) = tnot(-1) = +1"),
        ([MINUS_ONE], [PLUS_ONE], "tnot(tadd(-1, +1)) = tnot(0) = 0"),
        ([ZERO], [ZERO], "tnot(tadd(0, 0)) = tnot(0) = 0"),
        ([ZERO], [PLUS_ONE], "tnot(tadd(0, +1)) = tnot(+1) = -1"),
        ([PLUS_ONE], [PLUS_ONE], "tnot(tadd(+1, +1)) = tnot(+1) = -1"),
    ]

    passed = 0
    for a_val, b_val, desc in test_cases:
        a = np.array(a_val, dtype=np.uint8)
        b = np.array(b_val, dtype=np.uint8)

        # Compute using separate operations
        expected = base.tnot(base.tadd(a, b))

        # Compute using fused operation
        result = fusion.fused_tnot_tadd(a, b)

        if np.array_equal(result, expected):
            passed += 1
        else:
            print(f"  [FAIL] {desc}")
            print(f"    Expected: {expected[0]:02b}, Got: {result[0]:02b}")
            return False

    print(f"  [OK] All {passed} test cases passed")
    return True

def test_fused_tnot_tmul():
    """Test fused_tnot_tmul correctness"""
    print("\n=== Testing fused_tnot_tmul ===")

    test_cases = [
        ([MINUS_ONE], [MINUS_ONE], "tnot(tmul(-1, -1)) = tnot(+1) = -1"),
        ([MINUS_ONE], [ZERO], "tnot(tmul(-1, 0)) = tnot(0) = 0"),
        ([MINUS_ONE], [PLUS_ONE], "tnot(tmul(-1, +1)) = tnot(-1) = +1"),
        ([ZERO], [PLUS_ONE], "tnot(tmul(0, +1)) = tnot(0) = 0"),
        ([PLUS_ONE], [PLUS_ONE], "tnot(tmul(+1, +1)) = tnot(+1) = -1"),
    ]

    passed = 0
    for a_val, b_val, desc in test_cases:
        a = np.array(a_val, dtype=np.uint8)
        b = np.array(b_val, dtype=np.uint8)

        expected = base.tnot(base.tmul(a, b))
        result = fusion.fused_tnot_tmul(a, b)

        if np.array_equal(result, expected):
            passed += 1
        else:
            print(f"  [FAIL] {desc}")
            print(f"    Expected: {expected[0]:02b}, Got: {result[0]:02b}")
            return False

    print(f"  [OK] All {passed} test cases passed")
    return True

def test_fused_tnot_tmin():
    """Test fused_tnot_tmin correctness"""
    print("\n=== Testing fused_tnot_tmin ===")

    test_cases = [
        ([MINUS_ONE], [MINUS_ONE], "tnot(tmin(-1, -1)) = tnot(-1) = +1"),
        ([MINUS_ONE], [ZERO], "tnot(tmin(-1, 0)) = tnot(-1) = +1"),
        ([MINUS_ONE], [PLUS_ONE], "tnot(tmin(-1, +1)) = tnot(-1) = +1"),
        ([ZERO], [PLUS_ONE], "tnot(tmin(0, +1)) = tnot(0) = 0"),
        ([PLUS_ONE], [PLUS_ONE], "tnot(tmin(+1, +1)) = tnot(+1) = -1"),
    ]

    passed = 0
    for a_val, b_val, desc in test_cases:
        a = np.array(a_val, dtype=np.uint8)
        b = np.array(b_val, dtype=np.uint8)

        expected = base.tnot(base.tmin(a, b))
        result = fusion.fused_tnot_tmin(a, b)

        if np.array_equal(result, expected):
            passed += 1
        else:
            print(f"  [FAIL] {desc}")
            print(f"    Expected: {expected[0]:02b}, Got: {result[0]:02b}")
            return False

    print(f"  [OK] All {passed} test cases passed")
    return True

def test_fused_tnot_tmax():
    """Test fused_tnot_tmax correctness"""
    print("\n=== Testing fused_tnot_tmax ===")

    test_cases = [
        ([MINUS_ONE], [MINUS_ONE], "tnot(tmax(-1, -1)) = tnot(-1) = +1"),
        ([ZERO], [ZERO], "tnot(tmax(0, 0)) = tnot(0) = 0"),
        ([MINUS_ONE], [PLUS_ONE], "tnot(tmax(-1, +1)) = tnot(+1) = -1"),
        ([ZERO], [PLUS_ONE], "tnot(tmax(0, +1)) = tnot(+1) = -1"),
        ([PLUS_ONE], [PLUS_ONE], "tnot(tmax(+1, +1)) = tnot(+1) = -1"),
    ]

    passed = 0
    for a_val, b_val, desc in test_cases:
        a = np.array(a_val, dtype=np.uint8)
        b = np.array(b_val, dtype=np.uint8)

        expected = base.tnot(base.tmax(a, b))
        result = fusion.fused_tnot_tmax(a, b)

        if np.array_equal(result, expected):
            passed += 1
        else:
            print(f"  [FAIL] {desc}")
            print(f"    Expected: {expected[0]:02b}, Got: {result[0]:02b}")
            return False

    print(f"  [OK] All {passed} test cases passed")
    return True

def test_array_sizes():
    """Test fused operations with various array sizes"""
    print("\n=== Testing Array Sizes ===")

    sizes = [1, 10, 32, 33, 100, 1000, 10000]

    for size in sizes:
        # Create random test arrays
        np.random.seed(42)
        a = np.random.randint(0, 3, size, dtype=np.uint8)
        b = np.random.randint(0, 3, size, dtype=np.uint8)

        # Test all fusion operations
        ops = [
            ("fused_tnot_tadd", fusion.fused_tnot_tadd, lambda a, b: base.tnot(base.tadd(a, b))),
            ("fused_tnot_tmul", fusion.fused_tnot_tmul, lambda a, b: base.tnot(base.tmul(a, b))),
            ("fused_tnot_tmin", fusion.fused_tnot_tmin, lambda a, b: base.tnot(base.tmin(a, b))),
            ("fused_tnot_tmax", fusion.fused_tnot_tmax, lambda a, b: base.tnot(base.tmax(a, b))),
        ]

        for name, fused_op, separate_op in ops:
            expected = separate_op(a, b)
            result = fused_op(a, b)

            if not np.array_equal(result, expected):
                print(f"  [FAIL] {name} failed at size {size}")
                mismatch_idx = np.where(result != expected)[0]
                print(f"    Mismatches at indices: {mismatch_idx[:10]}")
                return False

    print(f"  [OK] All fusion operations correct for sizes: {sizes}")
    return True

def test_error_handling():
    """Test error handling for fused operations"""
    print("\n=== Testing Error Handling ===")

    # Size mismatch
    try:
        a = np.array([ZERO, ZERO], dtype=np.uint8)
        b = np.array([ZERO], dtype=np.uint8)
        fusion.fused_tnot_tadd(a, b)
        print("  [FAIL] Size mismatch did not raise exception")
        return False
    except RuntimeError as e:
        if "size mismatch" in str(e).lower() or "array" in str(e).lower():
            print(f"  [OK] Size mismatch correctly raised exception")
        else:
            print(f"  [WARN] Exception raised but unexpected message: {e}")

    # Empty arrays
    a = np.array([], dtype=np.uint8)
    b = np.array([], dtype=np.uint8)
    result = fusion.fused_tnot_tadd(a, b)
    if len(result) == 0:
        print("  [OK] Empty array handling correct")
    else:
        print("  [FAIL] Empty array handling incorrect")
        return False

    return True

def test_simd_boundaries():
    """Test SIMD boundary conditions"""
    print("\n=== Testing SIMD Boundaries ===")

    # Test around 32-element SIMD boundary
    boundary_sizes = [31, 32, 33, 63, 64, 65]

    for size in boundary_sizes:
        a = np.ones(size, dtype=np.uint8) * PLUS_ONE
        b = np.ones(size, dtype=np.uint8) * MINUS_ONE

        # fused_tnot_tadd(+1, -1) = tnot(0) = 0
        result = fusion.fused_tnot_tadd(a, b)
        expected = np.ones(size, dtype=np.uint8) * ZERO

        if np.array_equal(result, expected):
            print(f"  [OK] Size {size}: correct")
        else:
            print(f"  [FAIL] Size {size}: incorrect")
            return False

    print(f"  [OK] All SIMD boundary sizes correct")
    return True

def main():
    print_header("FUSION ENGINE TEST SUITE")

    tests = [
        ("Basic Correctness - fused_tnot_tadd", test_fused_tnot_tadd),
        ("Basic Correctness - fused_tnot_tmul", test_fused_tnot_tmul),
        ("Basic Correctness - fused_tnot_tmin", test_fused_tnot_tmin),
        ("Basic Correctness - fused_tnot_tmax", test_fused_tnot_tmax),
        ("Array Sizes", test_array_sizes),
        ("Error Handling", test_error_handling),
        ("SIMD Boundaries", test_simd_boundaries),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n[EXCEPTION] {name}: {e}")
            failed += 1

    print_header("TEST SUMMARY")
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n[SUCCESS] ALL FUSION TESTS PASSED!")
        print("Fusion operations produce identical results to separate operations.")
        return 0
    else:
        print(f"\n[FAIL] {failed} TEST(S) FAILED")
        return 1

if __name__ == '__main__':
    sys.exit(main())
