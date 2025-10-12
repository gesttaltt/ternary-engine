"""
Reference implementations for ternary operations.

These implementations mimic the pre-optimization (Phase 0) behavior
using conversion-based approach with branches. They serve as a baseline
for benchmark comparisons.

Encoding:
  -1 → 0 (0b00)
   0 → 1 (0b01)
  +1 → 2 (0b10)
"""

import numpy as np


def ref_tadd(a, b):
    """
    Reference implementation of saturating ternary addition.

    Uses integer conversion with branches (pre-LUT optimization).
    Expected to be 3-10x slower than optimized LUT version.
    """
    def to_int(t):
        return -1 if t == 0 else (1 if t == 2 else 0)

    def to_trit(i):
        return 0 if i < 0 else (2 if i > 0 else 1)

    s = to_int(a) + to_int(b)
    s = max(-1, min(1, s))  # Saturate to [-1, 1]
    return to_trit(s)


def ref_tmul(a, b):
    """
    Reference implementation of ternary multiplication.

    Sign-based multiplication with zero absorption.
    """
    def to_int(t):
        return -1 if t == 0 else (1 if t == 2 else 0)

    def to_trit(i):
        return 0 if i < 0 else (2 if i > 0 else 1)

    return to_trit(to_int(a) * to_int(b))


def ref_tmin(a, b):
    """
    Reference implementation of ternary minimum.

    Logical order: -1 < 0 < +1
    """
    def to_int(t):
        return -1 if t == 0 else (1 if t == 2 else 0)

    return a if to_int(a) < to_int(b) else b


def ref_tmax(a, b):
    """
    Reference implementation of ternary maximum.

    Logical order: -1 < 0 < +1
    """
    def to_int(t):
        return -1 if t == 0 else (1 if t == 2 else 0)

    return a if to_int(a) > to_int(b) else b


def ref_tnot(a):
    """
    Reference implementation of ternary negation.

    Arithmetic negation: -(-1) = +1, -(0) = 0, -(+1) = -1
    """
    def to_int(t):
        return -1 if t == 0 else (1 if t == 2 else 0)

    def to_trit(i):
        return 0 if i < 0 else (2 if i > 0 else 1)

    return to_trit(-to_int(a))


def ref_op_array(func, A, B=None):
    """
    Vectorize a scalar reference function for array inputs.

    This is intentionally slow (Python loop) to represent
    pre-optimization performance.

    Args:
        func: Scalar reference function (ref_tadd, ref_tmul, etc.)
        A: Input array (numpy array of uint8)
        B: Second input array (optional, for binary operations)

    Returns:
        Output array (numpy array of uint8)
    """
    if B is None:
        # Unary operation
        return np.array([func(a) for a in A], dtype=np.uint8)
    else:
        # Binary operation
        return np.array([func(a, b) for a, b in zip(A, B)], dtype=np.uint8)


# Correctness check functions
def verify_encoding():
    """Verify that reference implementations use correct encoding."""
    # Test conversions
    assert ref_tadd(0, 0) == 0  # -1 + -1 = -1 (saturated)
    assert ref_tadd(1, 1) == 1  # 0 + 0 = 0
    assert ref_tadd(2, 2) == 2  # +1 + +1 = +1 (saturated)

    assert ref_tmul(0, 0) == 2  # -1 * -1 = +1
    assert ref_tmul(2, 2) == 2  # +1 * +1 = +1
    assert ref_tmul(0, 2) == 0  # -1 * +1 = -1

    assert ref_tnot(0) == 2  # -(-1) = +1
    assert ref_tnot(1) == 1  # -(0) = 0
    assert ref_tnot(2) == 0  # -(+1) = -1

    print("✓ Reference implementations verified")


if __name__ == "__main__":
    # Quick verification
    verify_encoding()

    # Example usage
    import numpy as np
    A = np.array([0, 1, 2], dtype=np.uint8)
    B = np.array([0, 1, 2], dtype=np.uint8)

    print("\nExample outputs:")
    print(f"tadd: {ref_op_array(ref_tadd, A, B)}")
    print(f"tmul: {ref_op_array(ref_tmul, A, B)}")
    print(f"tnot: {ref_op_array(ref_tnot, A)}")
