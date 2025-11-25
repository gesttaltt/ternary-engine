"""
test_dual_shuffle_validation.py - Validate XOR-Decomposability of Dual-Shuffle LUTs

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

This test validates whether ternary operations can be correctly decomposed as:
    result = LUT_A(a) XOR LUT_B(b)

For each operation, we test all 9 valid input combinations and verify that
the XOR decomposition produces the correct result.

If validation fails, dual-shuffle XOR cannot be used for that operation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def ternary_to_2bit(trit):
    """Convert trit {-1, 0, +1} to 2-bit encoding"""
    if trit == -1:
        return 0b00
    elif trit == 0:
        return 0b01
    elif trit == 1:
        return 0b10
    else:
        raise ValueError(f"Invalid trit: {trit}")


def bit2_to_ternary(bits):
    """Convert 2-bit encoding to trit {-1, 0, +1}"""
    bits = bits & 0x03
    if bits == 0b00:
        return -1
    elif bits == 0b01:
        return 0
    elif bits == 0b10:
        return 1
    else:
        return 0  # Invalid, treat as 0


def ternary_not(a):
    """Ternary NOT operation"""
    return -a


def ternary_add(a, b):
    """Ternary addition with saturation"""
    result = a + b
    if result > 1:
        return 1
    elif result < -1:
        return -1
    return result


def ternary_mul(a, b):
    """Ternary multiplication"""
    return a * b


def generate_tnot_dual_luts():
    """Generate dual-shuffle LUTs for ternary NOT"""
    negation_map = {
        0b00: 0b10,  # -1 → +1
        0b01: 0b01,  # 0 → 0
        0b10: 0b00,  # +1 → -1
        0b11: 0b01,  # invalid → 0
    }

    lut_a = [negation_map[i & 0x03] for i in range(256)]
    lut_b = [0x00 for i in range(256)]  # Identity

    return lut_a, lut_b


def generate_tadd_dual_luts():
    """
    Generate dual-shuffle LUTs for ternary addition

    This is the critical function that needs validation.
    The current implementation from ternary_dual_shuffle.h is:
    - LUT_A: mapping based on first operand
    - LUT_B: mapping based on second operand
    - Result: LUT_A(a) XOR LUT_B(b)
    """
    trit_decode = {0b00: -1, 0b01: 0, 0b10: 1, 0b11: 0}

    lut_a = []
    for i in range(256):
        ta = trit_decode[i & 0x03]
        if ta == -1:
            lut_a.append(0x00)
        elif ta == 0:
            lut_a.append(0x01)
        else:  # ta == 1
            lut_a.append(0x02)

    lut_b = []
    for i in range(256):
        tb = trit_decode[i & 0x03]
        if tb == -1:
            lut_b.append(0x00)
        elif tb == 0:
            lut_b.append(0x00)  # Zero doesn't change result
        else:  # tb == 1
            lut_b.append(0x01)

    return lut_a, lut_b


def generate_tmul_dual_luts():
    """Generate dual-shuffle LUTs for ternary multiplication"""
    trit_decode = {0b00: -1, 0b01: 0, 0b10: 1, 0b11: 0}

    lut_a = []
    for i in range(256):
        ta = trit_decode[i & 0x03]
        if ta == -1:
            lut_a.append(0x02)  # Negation pattern
        elif ta == 0:
            lut_a.append(0x01)  # Zero pattern
        else:  # ta == 1
            lut_a.append(0x00)  # Identity pattern

    lut_b = []
    for i in range(256):
        tb = trit_decode[i & 0x03]
        lut_b.append((tb + 1) & 0xFF)  # Standard encoding

    return lut_a, lut_b


def validate_operation(op_name, op_func, lut_a, lut_b):
    """
    Validate that op_func(a, b) == LUT_A(a) XOR LUT_B(b)

    Returns: (success, error_count, total_tests)
    """
    errors = []
    valid_trits = [-1, 0, 1]

    for ta in valid_trits:
        for tb in valid_trits:
            # Compute expected result
            expected = op_func(ta, tb) if op_func.__code__.co_argcount == 2 else op_func(ta)
            expected_bits = ternary_to_2bit(expected)

            # Compute via dual-shuffle XOR
            a_bits = ternary_to_2bit(ta)
            b_bits = ternary_to_2bit(tb) if op_func.__code__.co_argcount == 2 else 0

            comp_a = lut_a[a_bits]
            comp_b = lut_b[b_bits]
            result_bits = comp_a ^ comp_b
            result_trit = bit2_to_ternary(result_bits)

            if result_trit != expected:
                errors.append({
                    'ta': ta,
                    'tb': tb,
                    'expected': expected,
                    'got': result_trit,
                    'a_bits': bin(a_bits),
                    'b_bits': bin(b_bits),
                    'comp_a': bin(comp_a),
                    'comp_b': bin(comp_b),
                    'result_bits': bin(result_bits),
                })

    total_tests = len(valid_trits) ** (op_func.__code__.co_argcount)
    success = len(errors) == 0

    return success, errors, total_tests


def main():
    print("=" * 80)
    print("DUAL-SHUFFLE XOR VALIDATION")
    print("=" * 80)
    print("\nTesting XOR-decomposability: result = LUT_A(a) XOR LUT_B(b)\n")

    tests = [
        ("tnot", ternary_not, generate_tnot_dual_luts()),
        ("tadd", ternary_add, generate_tadd_dual_luts()),
        ("tmul", ternary_mul, generate_tmul_dual_luts()),
    ]

    all_passed = True

    for op_name, op_func, (lut_a, lut_b) in tests:
        print(f"Testing {op_name}...")
        success, errors, total = validate_operation(op_name, op_func, lut_a, lut_b)

        if success:
            print(f"  ✅ PASS ({total}/{total} tests)\n")
        else:
            print(f"  ❌ FAIL ({total - len(errors)}/{total} tests passed)")
            print(f"  Errors: {len(errors)}\n")

            # Show first few errors
            for i, err in enumerate(errors[:3]):
                print(f"  Error {i+1}:")
                print(f"    Input: a={err['ta']}, b={err['tb']}")
                print(f"    Expected: {err['expected']}, Got: {err['got']}")
                print(f"    LUT_A({err['a_bits']}) = {err['comp_a']}")
                print(f"    LUT_B({err['b_bits']}) = {err['comp_b']}")
                print(f"    XOR result: {err['result_bits']}\n")

            if len(errors) > 3:
                print(f"  ... and {len(errors) - 3} more errors\n")

            all_passed = False

    print("=" * 80)
    if all_passed:
        print("✅ ALL OPERATIONS ARE XOR-DECOMPOSABLE")
        print("\nDual-shuffle XOR can be safely implemented for all operations.")
        return 0
    else:
        print("❌ SOME OPERATIONS ARE NOT XOR-DECOMPOSABLE")
        print("\nDual-shuffle XOR requires different LUT encoding or alternative approach.")
        print("\nOptions:")
        print("1. Fix LUT generation to make operations XOR-decomposable")
        print("2. Use dual-shuffle with ADD instead of XOR for non-decomposable ops")
        print("3. Skip dual-shuffle optimization for failing operations")
        return 1


if __name__ == '__main__':
    sys.exit(main())
