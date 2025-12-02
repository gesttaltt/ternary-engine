"""
test_fused_op_bug.py - Minimal test case to demonstrate the fused op bug.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import ternary_simd_engine as te

def numpy_fused_tnot_add_original_buggy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Original buggy baseline for fused_tnot_tadd_int8."""
    tnot_a = (a * -1).astype(np.int8)
    return np.clip(tnot_a.astype(np.int16) + b.astype(np.int16), -1, 1).astype(np.int8)

def numpy_fused_tnot_add_correct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Corrected baseline for fused_tnot_tadd_int8."""
    # This is tnot(a+b)
    sum_val = np.clip(a.astype(np.int16) + b.astype(np.int16), -1, 1).astype(np.int8)
    return (sum_val * -1).astype(np.int8)

def main():
    """Run the test case."""
    a = np.array([1, 1, 0, -1, -1, 1], dtype=np.int8)
    b = np.array([1, 0, -1, 1, 0, -1], dtype=np.int8)

    print("Input a: ", a)
    print("Input b: ", b)
    print("-" * 30)

    ternary_result = te.fused_tnot_tadd_int8(a, b)
    print("C++ Engine Result:      ", ternary_result)

    numpy_buggy_result = numpy_fused_tnot_add_original_buggy(a, b)
    print("NumPy Buggy Baseline:   ", numpy_buggy_result)

    numpy_correct_result = numpy_fused_tnot_add_correct(a, b)
    print("NumPy Correct Baseline: ", numpy_correct_result)

    print("-" * 30)

    bug_confirmed = not np.array_equal(ternary_result, numpy_buggy_result)
    fix_confirmed = np.array_equal(ternary_result, numpy_correct_result)

    if bug_confirmed:
        print("✅ Bug confirmed: C++ engine result differs from the buggy NumPy baseline.")
    else:
        print("❌ Bug not confirmed: C++ engine result matches the buggy NumPy baseline.")

    if fix_confirmed:
        print("✅ Fix confirmed: C++ engine result matches the corrected NumPy baseline.")
    else:
        print("❌ Fix not confirmed: C++ engine result does not match the corrected NumPy baseline.")

    print("-" * 30)
    if bug_confirmed and fix_confirmed:
        print("Conclusion: The bug is in the NumPy baseline `numpy_fused_tnot_add`.")
        return 0
    else:
        print("Conclusion: The bug is elsewhere, or the fix is incorrect.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
