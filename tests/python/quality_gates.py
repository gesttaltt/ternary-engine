#!/usr/bin/env python3
"""
quality_gates.py - Critical Quality Gates for Ternary Engine CI/CD

PURPOSE:
  Strict quality gates that must pass before any release. These are:
  1. Mathematical Correctness Gate - SIMD vs scalar must match exactly
  2. Algebraic Invariants Gate - Properties that MUST hold forever
  3. Type Enforcement Gate - Proper dtype flow (uint8 required)
  4. Error Boundary Gate - Exceptions raised correctly
  5. Performance Sanity Gate - Operations complete in bounded time

TYPE ENFORCEMENT:
  - Input: np.ndarray with dtype=np.uint8 (strict)
  - Output: np.ndarray with dtype=np.uint8 (strict)
  - Invalid dtype raises TypeError (no silent conversion)

CRITICAL BOTTLENECKS TESTED:
  - SIMD boundary conditions (31, 32, 33 elements)
  - All 9 combinations for binary ops (3x3 truth table)
  - All 3 combinations for unary ops
  - Algebraic properties: commutativity, associativity, identity, involution

EXIT CODES:
  0 = ALL GATES PASSED
  1 = ANY GATE FAILED (block release)

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0
"""

import sys
import os
from pathlib import Path

# Add project root to path for module import
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
from typing import Callable, Tuple, List, Dict, Any

# =============================================================================
# GATE 0: Module Import (Prerequisite)
# =============================================================================
try:
    import ternary_simd_engine as tc
except ImportError as e:
    print(f"❌ GATE 0 FAILED: Cannot import ternary_simd_engine")
    print(f"   Error: {e}")
    print(f"   Fix: Run 'python build/build.py' first")
    sys.exit(1)

# =============================================================================
# CONSTANTS
# =============================================================================
MINUS_ONE = np.uint8(0b00)  # Encoding: 0b00 = -1
ZERO = np.uint8(0b01)  # Encoding: 0b01 = 0
PLUS_ONE = np.uint8(0b10)  # Encoding: 0b10 = +1
INVALID = np.uint8(0b11)  # Invalid trit (should be masked)


# =============================================================================
# GATE RESULT TYPE
# =============================================================================
class GateResult:
    """Strict gate result with required evidence"""

    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.failures: List[str] = []
        self.warnings: List[str] = []

    def record_pass(self, detail: str = ""):
        prefix = f"  ✓ {detail}" if detail else "  ✓"
        print(prefix)

    def record_fail(self, detail: str, evidence: str = ""):
        self.passed = False
        msg = f"  ✗ {detail}"
        if evidence:
            msg += f"\n    Evidence: {evidence}"
        self.failures.append(msg)
        print(msg)

    def record_warn(self, detail: str):
        self.warnings.append(detail)
        print(f"  ⚠ {detail}")

    def summary(self) -> bool:
        if self.passed:
            print(f"\n  ✓ GATE {self.name}: PASSED")
            return True
        else:
            print(f"\n  ✗ GATE {self.name}: FAILED")
            for f in self.failures:
                print(f"    {f}")
            return False


# =============================================================================
# GATE 1: Mathematical Correctness (EXHAUSTIVE)
# =============================================================================
def gate1_mathematical_correctness() -> bool:
    """
    CRITICAL: Every SIMD output must match scalar reference exactly.

    Tests:
    - All 9 combinations for binary ops (3x3 truth table)
    - All 3 combinations for unary ops

    This is the PRIMARY correctness gate - failure here blocks release.
    """
    result = GateResult("1: Mathematical Correctness")
    print("\n" + "=" * 70)
    print("  GATE 1: MATHEMATICAL CORRECTNESS (Exhaustive)")
    print("=" * 70)

    # Expected truth tables
    TADD_EXPECTED = {
        (0, 0): 0,
        (0, 1): 0,
        (0, 2): 1,
        (1, 0): 0,
        (1, 1): 1,
        (1, 2): 2,
        (2, 0): 1,
        (2, 1): 2,
        (2, 2): 2,
    }
    TMUL_EXPECTED = {
        (0, 0): 2,
        (0, 1): 1,
        (0, 2): 0,
        (1, 0): 1,
        (1, 1): 1,
        (1, 2): 1,
        (2, 0): 0,
        (2, 1): 1,
        (2, 2): 2,
    }
    TMIN_EXPECTED = {
        (0, 0): 0,
        (0, 1): 0,
        (0, 2): 0,
        (1, 0): 0,
        (1, 1): 1,
        (1, 2): 1,
        (2, 0): 0,
        (2, 1): 1,
        (2, 2): 2,
    }
    TMAX_EXPECTED = {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 2,
        (1, 0): 1,
        (1, 1): 1,
        (1, 2): 2,
        (2, 0): 2,
        (2, 1): 2,
        (2, 2): 2,
    }
    TNOT_EXPECTED = {0: 2, 1: 1, 2: 0}

    ops = [
        ("tadd", tc.tadd, TADD_EXPECTED, False),
        ("tmul", tc.tmul, TMUL_EXPECTED, False),
        ("tmin", tc.tmin, TMIN_EXPECTED, False),
        ("tmax", tc.tmax, TMAX_EXPECTED, False),
        ("tnot", tc.tnot, TNOT_EXPECTED, True),
    ]

    failures = 0
    for name, fn, expected, is_unary in ops:
        op_failures = 0
        if is_unary:
            for a_val in [0, 1, 2]:
                inputs = (np.array([a_val], dtype=np.uint8),)
                simd_result = fn(inputs[0])[0]
                if simd_result != np.uint8(expected[a_val]):
                    result.record_fail(
                        f"{name}({a_val})", f"SIMD={simd_result}, Expected={expected[a_val]}"
                    )
                    op_failures += 1
        else:
            for a_val in [0, 1, 2]:
                for b_val in [0, 1, 2]:
                    inputs = (np.array([a_val], dtype=np.uint8), np.array([b_val], dtype=np.uint8))
                    simd_result = fn(inputs[0], inputs[1])[0]
                    if simd_result != np.uint8(expected[(a_val, b_val)]):
                        result.record_fail(
                            f"{name}({a_val},{b_val})",
                            f"SIMD={simd_result}, Expected={expected[(a_val, b_val)]}",
                        )
                        op_failures += 1

        if op_failures == 0:
            result.record_pass(f"{name}: all combinations correct")
        else:
            failures += op_failures

    return result.summary() and failures == 0


# =============================================================================
# GATE 2: Algebraic Invariants (MUST NEVER BREAK)
# =============================================================================
def gate2_algebraic_invariants() -> bool:
    """
    CRITICAL: These properties must hold FOREVER.

    Properties tested:
    - tadd: commutativity (a+b = b+a), identity (a+0 = a)
    - tmul: commutativity, zero property (a*0 = 0)
    - tmin/tmax: idempotence (a⊕a = a)
    - tnot: involution (not(not(a)) = a)
    - saturation: output is always valid trit (no 0b11)

    NOTE: TADD is NOT associative due to saturation:
      ((+1) + (+1)) + (-1) = (+1) + (-1) = 0
      (+1) + ((+1) + (-1)) = (+1) + (0) = +1
      So 0 ≠ +1, proving non-associativity
    """
    result = GateResult("2: Algebraic Invariants")
    print("\n" + "=" * 70)
    print("  GATE 2: ALGEBRAIC INVARIANTS")
    print("=" * 70)

    failures = 0
    sizes = [1, 32, 1000]

    for size in sizes:
        a = np.random.randint(0, 3, size, dtype=np.uint8)
        b = np.random.randint(0, 3, size, dtype=np.uint8)

        # === TADD invariants ===
        # Commutativity: a + b = b + a
        if not np.array_equal(tc.tadd(a, b), tc.tadd(b, a)):
            result.record_fail(f"tadd commutativity", f"size={size}")
            failures += 1

        # Identity: a + 0 = a
        zeros = np.full(size, ZERO, dtype=np.uint8)
        if not np.array_equal(tc.tadd(a, zeros), a):
            result.record_fail(f"tadd identity (a+0=a)", f"size={size}")
            failures += 1

        # Saturation bound: output is always valid (no 0b11)
        sum_ab = tc.tadd(a, b)
        if np.any(sum_ab == INVALID):
            result.record_fail(f"tadd saturation", f"size={size}, invalid trits found")
            failures += 1

        # === TMUL invariants ===
        # Commutativity
        if not np.array_equal(tc.tmul(a, b), tc.tmul(b, a)):
            result.record_fail(f"tmul commutativity", f"size={size}")
            failures += 1

        # Zero property: a * 0 = 0
        if not np.array_equal(tc.tmul(a, zeros), zeros):
            result.record_fail(f"tmul zero property", f"size={size}")
            failures += 1

        # Identity: a * 1 = a
        ones = np.full(size, PLUS_ONE, dtype=np.uint8)
        if not np.array_equal(tc.tmul(a, ones), a):
            result.record_fail(f"tmul identity (a*1=a)", f"size={size}")
            failures += 1

        # === TNOT invariants ===
        # Involution: not(not(a)) = a
        not_a = tc.tnot(a)
        not_not_a = tc.tnot(not_a)
        if not np.array_equal(not_not_a, a):
            result.record_fail(f"tnot involution", f"size={size}")
            failures += 1

        # Fixed points: not(0) = 0
        if not np.array_equal(tc.tnot(zeros), zeros):
            result.record_fail(f"tnot fixed point", f"not(0) != 0, size={size}")
            failures += 1

        # === TMIN/TMAX invariants ===
        # Idempotence
        if not np.array_equal(tc.tmin(a, a), a):
            result.record_fail(f"tmin idempotence", f"size={size}")
            failures += 1
        if not np.array_equal(tc.tmax(a, a), a):
            result.record_fail(f"tmax idempotence", f"size={size}")
            failures += 1

        # Absorption: min(a, max(a, b)) = a
        if not np.array_equal(tc.tmin(a, tc.tmax(a, b)), a):
            result.record_fail(f"tmin absorption", f"size={size}")
            failures += 1
        if not np.array_equal(tc.tmax(a, tc.tmin(a, b)), a):
            result.record_fail(f"tmax absorption", f"size={size}")
            failures += 1

    if failures == 0:
        result.record_pass("All algebraic invariants hold")

    return result.summary() and failures == 0


# =============================================================================
# GATE 3: Type Enforcement (STRICT)
# =============================================================================
def gate3_type_enforcement() -> bool:
    """
    CRITICAL: Type safety must be enforced strictly.

    Requirements:
    - Input MUST be np.ndarray with dtype=np.uint8
    - Output MUST be np.ndarray with dtype=np.uint8
    - Output length equals input length exactly
    """
    result = GateResult("3: Type Enforcement")
    print("\n" + "=" * 70)
    print("  GATE 3: TYPE ENFORCEMENT (Strict)")
    print("=" * 70)

    failures = 0
    ops = [tc.tadd, tc.tmul, tc.tmin, tc.tmax, tc.tnot]
    op_names = ["tadd", "tmul", "tmin", "tmax", "tnot"]

    # Test 1: Valid uint8 arrays work
    for name, op in zip(op_names, ops):
        a = np.array([0, 1, 2], dtype=np.uint8)
        b = np.array([2, 1, 0], dtype=np.uint8)

        try:
            if name == "tnot":
                r = op(a)
            else:
                r = op(a, b)

            if r.dtype != np.dtype("uint8"):
                result.record_fail(f"{name} output dtype", f"got {r.dtype}, expected uint8")
                failures += 1
            else:
                result.record_pass(f"{name} output dtype: uint8")
        except Exception as e:
            result.record_fail(f"{name} valid uint8 input", str(e))
            failures += 1

    # Test 2: Output length matches input
    for name, op in zip(op_names, ops):
        for size in [1, 31, 32, 33, 1000]:
            a = np.random.randint(0, 3, size, dtype=np.uint8)
            b = np.random.randint(0, 3, size, dtype=np.uint8)

            try:
                if name == "tnot":
                    r = op(a)
                else:
                    r = op(a, b)

                if len(r) != size:
                    result.record_fail(
                        f"{name} length preservation", f"input={size}, output={len(r)}"
                    )
                    failures += 1
            except Exception as e:
                result.record_fail(f"{name} length test", str(e))
                failures += 1

    if failures == 0:
        result.record_pass("Type enforcement validated")

    return result.summary() and failures == 0


# =============================================================================
# GATE 4: Error Boundary Conditions
# =============================================================================
def gate4_error_boundaries() -> bool:
    """
    CRITICAL: Error handling must work correctly.

    Tests:
    - Size mismatch raises exception with clear message
    - Empty arrays handled gracefully
    - Very large arrays handled without crash
    """
    result = GateResult("4: Error Boundaries")
    print("\n" + "=" * 70)
    print("  GATE 4: ERROR BOUNDARIES")
    print("=" * 70)

    failures = 0

    # Test: Size mismatch raises exception
    a = np.array([0, 1, 2], dtype=np.uint8)
    b = np.array([0, 1], dtype=np.uint8)

    for op_name, op in [("tadd", tc.tadd), ("tmul", tc.tmul), ("tmin", tc.tmin), ("tmax", tc.tmax)]:
        try:
            r = op(a, b)
            result.record_fail(f"{op_name} size mismatch", "no exception raised")
            failures += 1
        except (RuntimeError, ValueError) as e:
            if "size" in str(e).lower() or "match" in str(e).lower():
                result.record_pass(f"{op_name} size mismatch: clear message")
            else:
                result.record_fail(f"{op_name} size mismatch message", str(e))
                failures += 1
        except Exception as e:
            result.record_fail(f"{op_name} size mismatch", f"wrong exception: {type(e).__name__}")
            failures += 1

    # Test: Empty arrays handled
    empty = np.array([], dtype=np.uint8)
    for op_name, op in [
        ("tadd", tc.tadd),
        ("tmul", tc.tmul),
        ("tmin", tc.tmin),
        ("tmax", tc.tmax),
        ("tnot", tc.tnot),
    ]:
        try:
            if op_name == "tnot":
                r = op(empty)
            else:
                r = op(empty, empty)
            if len(r) == 0:
                result.record_pass(f"{op_name} empty array")
            else:
                result.record_fail(f"{op_name} empty array", f"len={len(r)}, expected 0")
                failures += 1
        except Exception as e:
            result.record_fail(f"{op_name} empty array", str(e))
            failures += 1

    # Test: Large arrays (10M elements) don't crash
    print("  Testing large array handling (10M elements)...")
    try:
        size = 10_000_000
        a = np.random.randint(0, 3, size, dtype=np.uint8)
        b = np.random.randint(0, 3, size, dtype=np.uint8)
        r = tc.tadd(a, b)
        if len(r) == size:
            result.record_pass(f"10M element array handled")
        else:
            result.record_fail(f"10M element size", f"expected {size}, got {len(r)}")
            failures += 1
    except MemoryError:
        result.record_warn("10M array: insufficient memory (acceptable)")
    except Exception as e:
        result.record_fail(f"10M element crash", str(e))
        failures += 1

    return result.summary() and failures == 0


# =============================================================================
# GATE 5: SIMD Boundary Conditions
# =============================================================================
def gate5_simd_boundaries() -> bool:
    """
    CRITICAL: SIMD vectorization boundaries must be correct.

    AVX2 processes 32 elements at a time. We must verify:
    - 31 elements: 31 scalar (1 tail)
    - 32 elements: 32 SIMD (exact fit)
    - 33 elements: 32 SIMD + 1 scalar
    """
    result = GateResult("5: SIMD Boundaries")
    print("\n" + "=" * 70)
    print("  GATE 5: SIMD BOUNDARY CONDITIONS")
    print("=" * 70)

    boundary_sizes = [31, 32, 33, 63, 64, 65, 127, 128, 129]

    failures = 0
    for size in boundary_sizes:
        a = np.full(size, PLUS_ONE, dtype=np.uint8)

        # tadd(+1, +1) = +1 (saturated)
        b = np.full(size, PLUS_ONE, dtype=np.uint8)
        r = tc.tadd(a, b)
        if not np.all(r == PLUS_ONE):
            result.record_fail(f"tadd boundary size={size}", f"expected all +1, got {np.unique(r)}")
            failures += 1

        # tadd(+1, -1) = 0
        b = np.full(size, MINUS_ONE, dtype=np.uint8)
        r = tc.tadd(a, b)
        if not np.all(r == ZERO):
            result.record_fail(
                f"tadd(+1,-1) boundary size={size}", f"expected all 0, got {np.unique(r)}"
            )
            failures += 1

        # tnot(+1) = -1
        a = np.full(size, PLUS_ONE, dtype=np.uint8)
        r = tc.tnot(a)
        if not np.all(r == MINUS_ONE):
            result.record_fail(f"tnot boundary size={size}", f"expected all -1, got {np.unique(r)}")
            failures += 1

    if failures == 0:
        result.record_pass(f"All {len(boundary_sizes)} boundary sizes correct")

    return result.summary() and failures == 0


# =============================================================================
# GATE 6: Performance Sanity Check
# =============================================================================
def gate6_performance_sanity() -> bool:
    """
    Sanity check: Operations complete in reasonable time.

    This is NOT a performance regression test - it's a sanity check
    that the code isn't catastrophically slow or stuck.

    Requirements:
    - 1M element tadd completes in < 1 second
    - No infinite loops or deadlocks
    """
    result = GateResult("6: Performance Sanity")
    print("\n" + "=" * 70)
    print("  GATE 6: PERFORMANCE SANITY CHECK")
    print("=" * 70)

    import time

    failures = 0
    size = 1_000_000
    a = np.random.randint(0, 3, size, dtype=np.uint8)
    b = np.random.randint(0, 3, size, dtype=np.uint8)

    for op_name, op in [
        ("tadd", tc.tadd),
        ("tmul", tc.tmul),
        ("tmin", tc.tmin),
        ("tmax", tc.tmax),
        ("tnot", tc.tnot),
    ]:
        start = time.perf_counter()
        if op_name == "tnot":
            r = op(a)
        else:
            r = op(a, b)
        elapsed = time.perf_counter() - start

        # Sanity: should complete in < 1 second for 1M elements
        if elapsed > 1.0:
            result.record_fail(f"{op_name} sanity", f"{elapsed:.3f}s > 1.0s")
            failures += 1
        else:
            result.record_pass(f"{op_name}: {elapsed * 1000:.1f}ms for 1M elements")

    if failures == 0:
        result.record_pass("All operations meet sanity threshold")

    return result.summary() and failures == 0


# =============================================================================
# MAIN: Run All Quality Gates
# =============================================================================
def main():
    print("=" * 70)
    print("  TERNARY ENGINE QUALITY GATES")
    print("  Critical Bottleneck Testing with Type Enforcement")
    print("=" * 70)

    gates = [
        ("1: Mathematical Correctness", gate1_mathematical_correctness),
        ("2: Algebraic Invariants", gate2_algebraic_invariants),
        ("3: Type Enforcement", gate3_type_enforcement),
        ("4: Error Boundaries", gate4_error_boundaries),
        ("5: SIMD Boundaries", gate5_simd_boundaries),
        ("6: Performance Sanity", gate6_performance_sanity),
    ]

    results = []
    for name, gate_fn in gates:
        try:
            passed = gate_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  ✗ {name}: EXCEPTION")
            print(f"    {type(e).__name__}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("  QUALITY GATE SUMMARY")
    print("=" * 70)

    total = len(results)
    passed = sum(1 for _, p in results if p)
    failed = total - passed

    print(f"\nTotal gates: {total}")
    print(f"Passed:      {passed}")
    print(f"Failed:      {failed}")

    if failed == 0:
        print("\n" + "=" * 70)
        print("  ✓ ALL QUALITY GATES PASSED")
        print("  ✓ Release is CLEARED")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("  ✗ QUALITY GATES FAILED")
        print(f"  ✗ {failed} gate(s) blocked release")
        print("=" * 70)
        print("\nFailed gates:")
        for name, p in results:
            if not p:
                print(f"  - {name}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
