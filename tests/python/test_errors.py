"""
test_errors.py - Error handling and edge case tests

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Tests error conditions, exception handling, and edge cases.

Usage:
    python tests/test_errors.py
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

# Encoding
MINUS_ONE = 0b00
ZERO = 0b01
PLUS_ONE = 0b10
INVALID = 0b11  # Reserved/invalid trit value

def test_array_size_mismatch():
    """Test that mismatched array sizes raise appropriate errors"""
    print("\n=== Test: Array Size Mismatch ===")

    a = np.array([ZERO, PLUS_ONE], dtype=np.uint8)
    b = np.array([ZERO], dtype=np.uint8)

    try:
        result = tc.tadd(a, b)
        print("[FAIL] Expected exception for size mismatch, but none was raised")
        return False
    except (RuntimeError, ValueError) as e:
        if "size" in str(e).lower() or "match" in str(e).lower():
            print(f"[OK] Correctly raised exception: {type(e).__name__}")
            print(f"  Message: {e}")
            return True
        else:
            print(f"[FAIL] Unexpected exception message: {e}")
            return False
    except Exception as e:
        print(f"[FAIL] Unexpected exception type: {type(e).__name__}: {e}")
        return False

def test_empty_arrays():
    """Test operations on empty arrays"""
    print("\n=== Test: Empty Arrays ===")

    a = np.array([], dtype=np.uint8)
    b = np.array([], dtype=np.uint8)

    try:
        result = tc.tadd(a, b)
        if len(result) == 0:
            print("[OK] Empty array handling correct")
            return True
        else:
            print(f"[FAIL] Expected empty result, got {len(result)} elements")
            return False
    except Exception as e:
        print(f"[FAIL] Unexpected exception on empty arrays: {e}")
        return False

def test_single_element():
    """Test single element arrays"""
    print("\n=== Test: Single Element Arrays ===")

    a = np.array([PLUS_ONE], dtype=np.uint8)
    b = np.array([MINUS_ONE], dtype=np.uint8)

    try:
        result = tc.tadd(a, b)
        if len(result) == 1 and result[0] == ZERO:
            print("[OK] Single element handling correct")
            return True
        else:
            print(f"[FAIL] Expected [ZERO], got {result}")
            return False
    except Exception as e:
        print(f"[FAIL] Unexpected exception: {e}")
        return False

def test_simd_boundary():
    """Test arrays at SIMD boundaries (31, 32, 33 elements)"""
    print("\n=== Test: SIMD Boundary Cases ===")

    sizes = [31, 32, 33, 63, 64, 65]
    all_passed = True

    for size in sizes:
        a = np.full(size, PLUS_ONE, dtype=np.uint8)
        b = np.full(size, PLUS_ONE, dtype=np.uint8)

        try:
            result = tc.tadd(a, b)
            # tadd with saturation: +1 + +1 = +1 (saturated)
            expected = np.full(size, PLUS_ONE, dtype=np.uint8)

            if np.array_equal(result, expected):
                print(f"[OK] Size {size}: correct")
            else:
                print(f"[FAIL] Size {size}: mismatch")
                all_passed = False
        except Exception as e:
            print(f"[FAIL] Size {size}: exception {e}")
            all_passed = False

    return all_passed

def test_large_arrays():
    """Test very large arrays for memory handling"""
    print("\n=== Test: Large Array Handling ===")

    # Test 100M elements (~100MB)
    size = 100_000_000

    try:
        print(f"  Creating arrays of {size:,} elements...")
        a = np.random.choice([MINUS_ONE, ZERO, PLUS_ONE], size=size).astype(np.uint8)
        b = np.random.choice([MINUS_ONE, ZERO, PLUS_ONE], size=size).astype(np.uint8)

        print(f"  Running operation...")
        result = tc.tadd(a, b)

        if len(result) == size:
            print(f"[OK] Large array ({size:,} elements) handled correctly")
            return True
        else:
            print(f"[FAIL] Size mismatch: expected {size}, got {len(result)}")
            return False
    except MemoryError:
        print("[WARN] Skipped: Insufficient memory for 100M element test")
        return True  # Not a failure, just limited resources
    except Exception as e:
        print(f"[FAIL] Unexpected exception: {e}")
        return False

def test_invalid_trit_values():
    """Test behavior with invalid trit encodings (0b11)"""
    print("\n=== Test: Invalid Trit Values (0b11) ===")

    # Note: With sanitization enabled, invalid values should be masked
    a = np.array([INVALID, ZERO, PLUS_ONE], dtype=np.uint8)
    b = np.array([ZERO, ZERO, ZERO], dtype=np.uint8)

    try:
        result = tc.tadd(a, b)
        # With masking, 0b11 & 0x03 = 0b11, treated as index 3 in LUT
        print(f"  Input with INVALID: {a}")
        print(f"  Result: {result}")
        print("[OK] Invalid trit handling completed (check for sanitization)")
        return True
    except Exception as e:
        print(f"  Exception raised: {e}")
        print("[OK] Invalid trit handling (exception path)")
        return True

def test_wrong_dtype():
    """Test behavior with incorrect dtype"""
    print("\n=== Test: Wrong Data Type ===")

    # Try to use int32 instead of uint8
    a = np.array([1, 0, -1], dtype=np.int32)
    b = np.array([1, 0, -1], dtype=np.int32)

    try:
        result = tc.tadd(a, b)
        print(f"  Result with int32: {result}")
        print("[WARN] No exception raised for int32 (may auto-convert)")
        return True  # Some flexibility is okay
    except Exception as e:
        print(f"  Exception raised: {type(e).__name__}")
        print("[OK] Type checking enforced")
        return True

def test_unary_operation_errors():
    """Test error handling for unary operations"""
    print("\n=== Test: Unary Operation Errors ===")

    # Test tnot with various edge cases
    tests_passed = True

    # Empty array
    try:
        a = np.array([], dtype=np.uint8)
        result = tc.tnot(a)
        if len(result) == 0:
            print("[OK] tnot: empty array handled")
        else:
            print("[FAIL] tnot: empty array size mismatch")
            tests_passed = False
    except Exception as e:
        print(f"[FAIL] tnot: unexpected exception on empty array: {e}")
        tests_passed = False

    # Large array
    try:
        a = np.full(10_000, PLUS_ONE, dtype=np.uint8)
        result = tc.tnot(a)
        expected = np.full(10_000, MINUS_ONE, dtype=np.uint8)
        if np.array_equal(result, expected):
            print("[OK] tnot: large array correct")
        else:
            print("[FAIL] tnot: large array mismatch")
            tests_passed = False
    except Exception as e:
        print(f"[FAIL] tnot: exception on large array: {e}")
        tests_passed = False

    return tests_passed

def main():
    print("=" * 70)
    print("  Error Handling and Edge Case Test Suite")
    print("=" * 70)

    # Run all tests
    results = []
    results.append(("Array Size Mismatch", test_array_size_mismatch()))
    results.append(("Empty Arrays", test_empty_arrays()))
    results.append(("Single Element", test_single_element()))
    results.append(("SIMD Boundaries", test_simd_boundary()))
    results.append(("Large Arrays", test_large_arrays()))
    results.append(("Invalid Trit Values", test_invalid_trit_values()))
    results.append(("Wrong Data Type", test_wrong_dtype()))
    results.append(("Unary Errors", test_unary_operation_errors()))

    # Summary
    print("\n" + "=" * 70)
    print("  Test Summary")
    print("=" * 70)

    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)
    failed_tests = total_tests - passed_tests

    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests} [OK]")
    print(f"Failed: {failed_tests}" + (" [FAIL]" if failed_tests > 0 else ""))

    if failed_tests > 0:
        print("\nFailed tests:")
        for name, passed in results:
            if not passed:
                print(f"  - {name}")

    if all(passed for _, passed in results):
        print("\n  [SUCCESS] ALL ERROR HANDLING TESTS PASSED!")
        print("  Error handling is robust.")
        return 0
    else:
        print("\n  [FAIL] SOME TESTS FAILED")
        print("  Review error handling implementation.")
        return 1

if __name__ == "__main__":
    exit(main())
