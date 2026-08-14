#!/usr/bin/env python3
"""
test_tritnet_inference_bindings.py - TritNet Python bindings correctness tests

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Validates the ternary_tritnet_inference module (src/engine/bindings_tritnet_inference.cpp,
Phase 3's Python-reachable C++ inference engine) against the same full-input-space
ground truth used by tests/cpp/test_tritnet_inference.cpp and
tests/python/test_tritnet_export.py: 243 samples for tnot, 59,049 for the binary ops.

Also checks input validation (shape, dtype range) and the has_avx2()/
built_with_avx2 introspection attributes.

Exits 0 with a [SKIP] notice if the module is not built (it is optional);
build it with: python build/build_tritnet_inference.py

USAGE: python tests/python/test_tritnet_inference_bindings.py
"""

import sys

import numpy as np

try:
    import ternary_tritnet_inference as tn
except ImportError:
    print("[SKIP] ternary_tritnet_inference not built (optional).")
    print("       Build with: python build/build_tritnet_inference.py")
    sys.exit(0)

# Recorded checkpoint accuracy, same figures used by test_tritnet_export.py
# and tests/cpp/test_tritnet_inference.cpp -- cross-checked a third way here.
EXPECTED_ACC = {
    'tnot': 1.0,
    'tadd': 1.0,
    'tmul': 0.99491947889328,
    'tmin': 0.9988822937011719,
    'tmax': 0.9985097050666809,
}

_TRITS = (-1, 0, 1)


def _all_trit_vectors(n: int):
    if n == 0:
        yield []
        return
    for t in _TRITS:
        for rest in _all_trit_vectors(n - 1):
            yield [t] + rest


SCALAR_OPS = {
    'tnot': lambda a: -a,
    'tadd': lambda a, b: max(-1, min(1, a + b)),
    'tmul': lambda a, b: a * b,
    'tmin': min,
    'tmax': max,
}


def int_to_trit(v: int) -> int:
    """Match ternary_algebra.h's encoding: 0b00=-1, 0b01=0, 0b10=+1."""
    return {-1: 0b00, 0: 0b01, 1: 0b10}[v]


def trit_to_int(t: int) -> int:
    return {0b00: -1, 0b01: 0, 0b10: 1}[t]


def make_dataset(op_name: str):
    """Full input space, trit-encoded uint8: [N, 5] (unary) or two [N, 5] (binary)."""
    op = SCALAR_OPS[op_name]
    if op_name == 'tnot':
        rows_x, rows_y = [], []
        for vec_a in _all_trit_vectors(5):
            rows_x.append([int_to_trit(t) for t in vec_a])
            rows_y.append([int_to_trit(op(t)) for t in vec_a])
        return np.array(rows_x, dtype=np.uint8), None, np.array(rows_y, dtype=np.uint8)

    rows_a, rows_b, rows_y = [], [], []
    for vec_a in _all_trit_vectors(5):
        for vec_b in _all_trit_vectors(5):
            rows_a.append([int_to_trit(t) for t in vec_a])
            rows_b.append([int_to_trit(t) for t in vec_b])
            rows_y.append([int_to_trit(op(a, b)) for a, b in zip(vec_a, vec_b)])
    return (np.array(rows_a, dtype=np.uint8), np.array(rows_b, dtype=np.uint8),
            np.array(rows_y, dtype=np.uint8))


def check_op(op_name: str) -> bool:
    X_a, X_b, Y = make_dataset(op_name)

    if op_name == 'tnot':
        pred = tn.tnot(X_a)
    else:
        pred = getattr(tn, op_name)(X_a, X_b)

    n_total = Y.shape[0]
    exact_match = np.all(pred == Y, axis=1)
    n_match = int(exact_match.sum())
    acc = n_match / n_total

    expected_acc = EXPECTED_ACC[op_name]
    expected_match = round(expected_acc * n_total)
    ok = (n_match == expected_match)

    print(f"  {op_name:<6} acc={acc*100:.6f}%  expected={expected_acc*100:.6f}%  "
          f"n_match={n_match}/{n_total} (expected {expected_match})  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_full_input_space() -> bool:
    print("-- Full input space vs recorded checkpoint accuracy --")
    return all(check_op(op) for op in ['tnot', 'tadd', 'tmul', 'tmin', 'tmax'])


def test_introspection() -> bool:
    print("\n-- Introspection --")
    print(f"  built_with_avx2 = {tn.built_with_avx2}")
    print(f"  has_avx2()      = {tn.has_avx2()}")
    ok = isinstance(tn.built_with_avx2, bool) and isinstance(tn.has_avx2(), bool)
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_input_validation() -> bool:
    print("\n-- Input validation --")
    ok = True

    # Wrong shape (not [N, 5])
    try:
        tn.tnot(np.zeros((3, 4), dtype=np.uint8))
        print("  wrong shape: FAIL (should have raised)")
        ok = False
    except Exception:
        print("  wrong shape: PASS (raised as expected)")

    # Mismatched batch size between a and b
    try:
        tn.tadd(np.zeros((3, 5), dtype=np.uint8), np.zeros((4, 5), dtype=np.uint8))
        print("  mismatched batch size: FAIL (should have raised)")
        ok = False
    except Exception:
        print("  mismatched batch size: PASS (raised as expected)")

    # Invalid trit value (0b11 = 3 is not a valid 2-bit trit encoding)
    try:
        tn.tnot(np.array([[3, 0, 0, 0, 0]], dtype=np.uint8))
        print("  invalid trit value: FAIL (should have raised)")
        ok = False
    except Exception:
        print("  invalid trit value: PASS (raised as expected)")

    return ok


def main() -> int:
    print("=" * 70)
    print("TritNet Python Bindings -- Correctness Tests")
    print("=" * 70)

    results = [
        test_full_input_space(),
        test_introspection(),
        test_input_validation(),
    ]

    all_ok = all(results)
    print(f"\n{'[SUCCESS] All checks passed' if all_ok else '[FAIL] At least one check failed'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
