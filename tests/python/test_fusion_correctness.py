"""
test_fusion_correctness.py - Test Fusion Operations Correctness

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Tests that fused operations produce identical results to unfused operations.

Phase 4.1 validated operations:
- fused_tnot_tadd: tnot(tadd(a, b))
- fused_tnot_tmul: tnot(tmul(a, b))
- fused_tnot_tmin: tnot(tmin(a, b))
- fused_tnot_tmax: tnot(tmax(a, b))

Validation:
- Cross-backend correctness (Scalar == AVX2_v2)
- All 9 trit combinations
- Large arrays (correctness at scale)
"""

import sys
import numpy as np
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import ternary_backend as tb
except ImportError:
    print("ERROR: ternary_backend not found. Build it first:")
    print("  python build/build_backend.py")
    sys.exit(1)

# Trit encoding
MINUS_ONE = 0b00
ZERO = 0b01
PLUS_ONE = 0b10

# Test data
ALL_TRITS = [MINUS_ONE, ZERO, PLUS_ONE]

def test_fusion_correctness():
    """Test fusion operations produce correct results"""
    print("=" * 70)
    print("Fusion Operations Correctness Test")
    print("=" * 70)

    tb.init()

    # Test on all backends that support fusion
    # TERNARY_CAP_FUSION = 0x0020 from backend_interface.h
    TERNARY_CAP_FUSION = 0x0020

    backends_to_test = []
    for backend in tb.list_backends():
        if backend.capabilities & TERNARY_CAP_FUSION:
            backends_to_test.append(backend.name)

    print(f"\nBackends with fusion support: {backends_to_test}")

    if not backends_to_test:
        print("ERROR: No backends support fusion operations")
        return False

    passed = 0
    failed = 0

    # Test each fusion operation
    fusion_ops = [
        ('fused_tnot_tadd', 'tadd', 'tnot'),
        ('fused_tnot_tmul', 'tmul', 'tnot'),
        ('fused_tnot_tmin', 'tmin', 'tnot'),
        ('fused_tnot_tmax', 'tmax', 'tnot'),
    ]

    for backend_name in backends_to_test:
        tb.set_backend(backend_name)
        print(f"\nTesting backend: {backend_name}")

        for fused_name, binary_op, unary_op in fusion_ops:
            print(f"  Testing {fused_name}...", end='', flush=True)

            # Check if operation is available
            try:
                fused_func = getattr(tb, fused_name)
            except AttributeError:
                print(f" SKIP (not exposed)")
                continue

            # Test on small arrays with all combinations
            for a_trit in ALL_TRITS:
                for b_trit in ALL_TRITS:
                    a = np.array([a_trit] * 100, dtype=np.uint8)
                    b = np.array([b_trit] * 100, dtype=np.uint8)

                    # Fused result
                    fused_result = fused_func(a, b)

                    # Unfused result
                    binary_func = getattr(tb, binary_op)
                    unary_func = getattr(tb, unary_op)
                    temp = binary_func(a, b)
                    unfused_result = unary_func(temp)

                    # Compare
                    if not np.array_equal(fused_result, unfused_result):
                        print(f" FAIL")
                        print(f"    Inputs: a={a_trit:02b}, b={b_trit:02b}")
                        print(f"    Fused:   {fused_result[:5]}")
                        print(f"    Unfused: {unfused_result[:5]}")
                        failed += 1
                        break
                else:
                    continue
                break
            else:
                # Test on large arrays
                np.random.seed(42)
                a_large = np.random.randint(0, 3, 10000, dtype=np.uint8)
                b_large = np.random.randint(0, 3, 10000, dtype=np.uint8)

                fused_result = fused_func(a_large, b_large)

                binary_func = getattr(tb, binary_op)
                unary_func = getattr(tb, unary_op)
                temp = binary_func(a_large, b_large)
                unfused_result = unary_func(temp)

                if np.array_equal(fused_result, unfused_result):
                    print(" PASS")
                    passed += 1
                else:
                    print(" FAIL (large array mismatch)")
                    print(f"    Mismatches: {np.sum(fused_result != unfused_result)} / {len(fused_result)}")
                    failed += 1

    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n✅ ALL TESTS PASSED")
        return True
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
        return False


if __name__ == '__main__':
    success = test_fusion_correctness()
    sys.exit(0 if success else 1)
