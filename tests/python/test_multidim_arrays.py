#!/usr/bin/env python3
"""
test_multidim_arrays.py - Multi-dimensional array support correctness tests

Copyright 2026 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Validates the multi-dimensional array support added 2026-08-18 to
process_binary_array/process_unary_array in bindings_core_ops.cpp (previously
1-D only -- CLAUDE.md's own "Nice to Have" list documented this as a known
limitation). Covers:
  1. N-D output matches the flattened 1-D reference, for every op family
     (core, unary, fused) and both element-type bridges (uint8, int8)
  2. Shapes are preserved exactly (not flattened) in the output
  3. Error paths: shape mismatch (same total size, different shape),
     size mismatch (1-D, unchanged pre-existing behavior), non-contiguous
     input (transposed view, Fortran-order array)
  4. The OMP-parallel path (arrays >= OMP_THRESHOLD) is exercised with a
     genuinely multi-dimensional array, not just inferred from the 1-D case

Full writeup: reports/2026-08-18/MULTIDIM_ARRAY_SUPPORT.md

USAGE: python tests/python/test_multidim_arrays.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import ternary_simd_engine as tc
except ImportError:
    print("[SKIP] ternary_simd_engine not built (required module).")
    print("       Build with: python build/build.py")
    sys.exit(0)

SEED = 2026


def _ref(fn, *arrays):
    """Flatten every array, call fn on the 1-D views, reshape back."""
    shape = arrays[0].shape
    flat = [a.ravel() for a in arrays]
    return fn(*flat).reshape(shape)


def test_2d_matches_flattened_reference() -> bool:
    """2-D inputs produce the same result as calling the op on the flattened 1-D arrays."""
    rng = np.random.default_rng(SEED)
    ok = True
    shapes = [(1, 1), (5, 7), (3, 100), (17, 13)]
    for shape in shapes:
        a = rng.integers(0, 3, shape, dtype=np.uint8)
        b = rng.integers(0, 3, shape, dtype=np.uint8)
        for name in ('tadd', 'tmul', 'tmin', 'tmax'):
            fn = getattr(tc, name)
            got = fn(a, b)
            if got.shape != shape:
                print(f"  [FAIL] {name}{shape}: output shape {got.shape} != {shape}")
                ok = False
                continue
            if not np.array_equal(got, _ref(fn, a, b)):
                print(f"  [FAIL] {name}{shape}: doesn't match flattened reference")
                ok = False
        got_unary = tc.tnot(a)
        if got_unary.shape != shape or not np.array_equal(got_unary, _ref(tc.tnot, a)):
            print(f"  [FAIL] tnot{shape}: mismatch")
            ok = False
    if ok:
        print(f"  [OK] 2-D binary + unary ops match flattened reference for {len(shapes)} shapes")
    return ok


def test_3d_and_higher() -> bool:
    """3-D and 4-D inputs work the same way."""
    rng = np.random.default_rng(SEED)
    ok = True
    for shape in [(2, 3, 4), (2, 2, 2, 2), (1, 1, 50)]:
        a = rng.integers(0, 3, shape, dtype=np.uint8)
        b = rng.integers(0, 3, shape, dtype=np.uint8)
        got = tc.tadd(a, b)
        if got.shape != shape or not np.array_equal(got, _ref(tc.tadd, a, b)):
            print(f"  [FAIL] tadd{shape}: mismatch")
            ok = False
    if ok:
        print("  [OK] 3-D and 4-D inputs match flattened reference")
    return ok


def test_fused_ops() -> bool:
    """Fused ops (fused_tnot_tadd etc.) also support N-D, and still equal tnot(op(a,b))."""
    rng = np.random.default_rng(SEED)
    shape = (6, 9)
    a = rng.integers(0, 3, shape, dtype=np.uint8)
    b = rng.integers(0, 3, shape, dtype=np.uint8)
    ok = True
    for op, fused in (('tadd', 'fused_tnot_tadd'), ('tmul', 'fused_tnot_tmul'),
                      ('tmin', 'fused_tnot_tmin'), ('tmax', 'fused_tnot_tmax')):
        got = getattr(tc, fused)(a, b)
        expected = tc.tnot(getattr(tc, op)(a, b))
        if got.shape != shape or not np.array_equal(got, expected):
            print(f"  [FAIL] {fused}{shape}: doesn't equal tnot({op}(a,b))")
            ok = False
    if ok:
        print(f"  [OK] fused ops match tnot(op(a,b)) for shape {shape}")
    return ok


def test_int8_bridge() -> bool:
    """The int8 bridge functions (tadd_int8 etc.) support N-D too."""
    rng = np.random.default_rng(SEED)
    shape = (4, 6)
    a = rng.integers(-1, 2, shape, dtype=np.int8)
    b = rng.integers(-1, 2, shape, dtype=np.int8)
    ok = True
    for name in ('tadd_int8', 'tmul_int8', 'tmin_int8', 'tmax_int8'):
        fn = getattr(tc, name)
        got = fn(a, b)
        if got.shape != shape or not np.array_equal(got, _ref(fn, a, b)):
            print(f"  [FAIL] {name}{shape}: mismatch")
            ok = False
    got_unary = tc.tnot_int8(a)
    if got_unary.shape != shape or not np.array_equal(got_unary, _ref(tc.tnot_int8, a)):
        print(f"  [FAIL] tnot_int8{shape}: mismatch")
        ok = False
    if ok:
        print(f"  [OK] int8 bridge ops match flattened reference for shape {shape}")
    return ok


def test_fused_int8_ops() -> bool:
    """The int8 bridge's fused ops (fused_tnot_tadd_int8 etc.) support N-D
    too, and still equal tnot_int8(op_int8(a,b)) -- found missing from this
    file's own initial coverage on a follow-up pass (all 4 uint8 fused ops
    were covered, all 4 int8 core ops were covered, but the 4 ops at the
    intersection -- int8 AND fused -- were not)."""
    rng = np.random.default_rng(SEED)
    shape = (5, 6)
    a = rng.integers(-1, 2, shape, dtype=np.int8)
    b = rng.integers(-1, 2, shape, dtype=np.int8)
    ok = True
    for op, fused in (('tadd_int8', 'fused_tnot_tadd_int8'), ('tmul_int8', 'fused_tnot_tmul_int8'),
                      ('tmin_int8', 'fused_tnot_tmin_int8'), ('tmax_int8', 'fused_tnot_tmax_int8')):
        got = getattr(tc, fused)(a, b)
        expected = tc.tnot_int8(getattr(tc, op)(a, b))
        if got.shape != shape or not np.array_equal(got, expected):
            print(f"  [FAIL] {fused}{shape}: doesn't equal tnot_int8({op}(a,b))")
            ok = False
    if ok:
        print(f"  [OK] int8 fused ops match tnot_int8(op_int8(a,b)) for shape {shape}")
    return ok


def test_noncontiguous_1d_regression() -> bool:
    """Regression test for a real latent bug found while examining this
    feature further (2026-08-18, same day): the pre-multi-dimensional-
    array-support code never validated C-contiguity for 1-D input at all
    (only N-D inputs got a check, since that was new code written for this
    feature). Confirmed by testing the actual prior commit's binding file:
    tc.tadd(a[::2], b) computed a WRONG result silently (no exception) --
    the flat-pointer SIMD path indexed the strided view as if it were
    densely packed. This test locks in the fix (now: rejected with a clear
    ValueError) so it can't regress back to the old silent-wrong-answer
    behavior. See reports/2026-08-18/MULTIDIM_ARRAY_SUPPORT.md."""
    rng = np.random.default_rng(SEED)
    a = rng.integers(0, 3, 100, dtype=np.uint8)
    b = rng.integers(0, 3, 50, dtype=np.uint8)
    ok = True

    def expect_value_error(label, fn):
        nonlocal ok
        try:
            fn()
            print(f"  [FAIL] {label}: no exception raised (would silently compute wrong results)")
            ok = False
        except ValueError:
            pass
        except Exception as e:
            print(f"  [FAIL] {label}: wrong exception type {type(e).__name__}: {e}")
            ok = False

    expect_value_error("step-sliced 1-D (a[::2])", lambda: tc.tadd(a[::2], b))
    expect_value_error("reversed 1-D (a[::-1])", lambda: tc.tadd(a[::-1], a[::-1]))
    expect_value_error("unary step-sliced 1-D", lambda: tc.tnot(a[::2]))

    if ok:
        print("  [OK] non-contiguous 1-D input rejected (regression-tests the latent bug fix)")
    return ok


def test_omp_path_multidim() -> bool:
    """A 2-D array large enough to cross OMP_THRESHOLD (262,144 on 8 cores)
    exercises the parallel path with a genuinely multi-dimensional shape,
    not just size inferred from the 1-D case."""
    rng = np.random.default_rng(SEED)
    shape = (600, 500)  # 300,000 elements
    a = rng.integers(0, 3, shape, dtype=np.uint8)
    b = rng.integers(0, 3, shape, dtype=np.uint8)
    got = tc.tadd(a, b)
    ok = got.shape == shape and np.array_equal(got, _ref(tc.tadd, a, b))
    if ok:
        print(f"  [OK] OMP-parallel path correct for shape {shape} ({a.size:,} elements)")
    else:
        print(f"  [FAIL] OMP-parallel path incorrect for shape {shape}")
    return ok


def test_1d_unchanged() -> bool:
    """1-D behavior (the pre-existing, extensively benchmarked path) is untouched:
    same results, same error type/message for size mismatch."""
    rng = np.random.default_rng(SEED)
    a = rng.integers(0, 3, 1000, dtype=np.uint8)
    b = rng.integers(0, 3, 1000, dtype=np.uint8)
    ok = True
    if not np.array_equal(tc.tadd(a, b), _ref(tc.tadd, a, b)):
        print("  [FAIL] 1-D tadd changed behavior")
        ok = False

    try:
        tc.tadd(np.zeros(3, dtype=np.uint8), np.zeros(5, dtype=np.uint8))
        print("  [FAIL] 1-D size mismatch: no exception raised")
        ok = False
    except RuntimeError as e:
        expected = ("Array size mismatch: array A has 3 elements, array B has 5 "
                    "elements. Binary operations require equal-sized arrays.")
        if str(e) != expected:
            print(f"  [FAIL] 1-D size mismatch message changed: {e!r}")
            ok = False
    except Exception as e:
        print(f"  [FAIL] 1-D size mismatch: wrong exception type {type(e).__name__}")
        ok = False

    if ok:
        print("  [OK] 1-D behavior and error message unchanged")
    return ok


def test_shape_edge_cases() -> bool:
    """0-d arrays (numpy scalars), empty N-D arrays, singleton dimensions,
    and higher-dimensional (6-D) arrays -- boundary shapes worth checking
    explicitly rather than assuming the general N-D path handles them."""
    rng = np.random.default_rng(SEED)
    ok = True

    a0 = np.array(1, dtype=np.uint8)  # ndim=0
    b0 = np.array(2, dtype=np.uint8)
    r0 = tc.tadd(a0, b0)
    if r0.shape != () or int(r0) != 2:
        print(f"  [FAIL] 0-d scalar: shape={r0.shape} value={r0}")
        ok = False

    ae = np.zeros((0, 5), dtype=np.uint8)
    be = np.zeros((0, 5), dtype=np.uint8)
    re = tc.tadd(ae, be)
    if re.shape != (0, 5):
        print(f"  [FAIL] empty N-D (0,5): shape={re.shape}")
        ok = False

    a1 = rng.integers(0, 3, (1, 5, 1), dtype=np.uint8)
    b1 = rng.integers(0, 3, (1, 5, 1), dtype=np.uint8)
    r1 = tc.tadd(a1, b1)
    if r1.shape != (1, 5, 1) or not np.array_equal(r1, _ref(tc.tadd, a1, b1)):
        print(f"  [FAIL] singleton dims (1,5,1): mismatch")
        ok = False

    a6 = rng.integers(0, 3, (2, 2, 2, 2, 2, 2), dtype=np.uint8)  # 64 elements, 6-D
    b6 = rng.integers(0, 3, (2, 2, 2, 2, 2, 2), dtype=np.uint8)
    r6 = tc.tadd(a6, b6)
    if r6.shape != (2, 2, 2, 2, 2, 2) or not np.array_equal(r6, _ref(tc.tadd, a6, b6)):
        print(f"  [FAIL] 6-D array: mismatch")
        ok = False

    if ok:
        print("  [OK] 0-d, empty, singleton-dim, and 6-D shapes all correct")
    return ok


def test_shape_mismatch_errors() -> bool:
    """Shape mismatch (same total size, different shape) raises
    ArrayShapeMismatchError; different total size still raises
    ArraySizeMismatchError even for N-D inputs."""
    rng = np.random.default_rng(SEED)
    ok = True

    a = rng.integers(0, 3, (3, 4), dtype=np.uint8)  # 12 elements
    b = rng.integers(0, 3, (4, 3), dtype=np.uint8)  # also 12 elements, different shape
    try:
        tc.tadd(a, b)
        print("  [FAIL] same-size different-shape: no exception raised")
        ok = False
    except RuntimeError as e:
        if "shape mismatch" not in str(e).lower():
            print(f"  [FAIL] same-size different-shape: unexpected message {e!r}")
            ok = False
    except Exception as e:
        print(f"  [FAIL] same-size different-shape: wrong type {type(e).__name__}")
        ok = False

    c = rng.integers(0, 3, (3, 4), dtype=np.uint8)   # 12 elements
    d = rng.integers(0, 3, (3, 5), dtype=np.uint8)   # 15 elements
    try:
        tc.tadd(c, d)
        print("  [FAIL] different-size N-D: no exception raised")
        ok = False
    except RuntimeError as e:
        if "size mismatch" not in str(e).lower():
            print(f"  [FAIL] different-size N-D: unexpected message {e!r}")
            ok = False
    except Exception as e:
        print(f"  [FAIL] different-size N-D: wrong type {type(e).__name__}")
        ok = False

    if ok:
        print("  [OK] shape/size mismatch errors correctly distinguished")
    return ok


def test_noncontiguous_rejected() -> bool:
    """Non-contiguous input (transposed view, Fortran-order array) is
    rejected with a clear error instead of silently reading wrong data --
    the flat SIMD/OMP paths require linear memory layout."""
    rng = np.random.default_rng(SEED)
    ok = True

    a = rng.integers(0, 3, (4, 6), dtype=np.uint8)
    b = rng.integers(0, 3, (6, 4), dtype=np.uint8)

    def expect_value_error(label, fn):
        nonlocal ok
        try:
            fn()
            print(f"  [FAIL] {label}: no exception raised")
            ok = False
        except ValueError:
            pass
        except Exception as e:
            print(f"  [FAIL] {label}: wrong exception type {type(e).__name__}: {e}")
            ok = False

    expect_value_error("transposed A", lambda: tc.tadd(a.T, b))
    expect_value_error("Fortran-order A", lambda: tc.tadd(np.asfortranarray(a), b))
    expect_value_error("unary non-contiguous", lambda: tc.tnot(np.asfortranarray(a)))

    if ok:
        print("  [OK] non-contiguous inputs correctly rejected")
    return ok


def main() -> int:
    print("=" * 70)
    print("Multi-Dimensional Array Support Tests")
    print("=" * 70)

    results = [
        ("2-D matches flattened reference", test_2d_matches_flattened_reference()),
        ("3-D and higher", test_3d_and_higher()),
        ("Fused ops", test_fused_ops()),
        ("Int8 bridge", test_int8_bridge()),
        ("Int8 fused ops", test_fused_int8_ops()),
        ("Shape edge cases (0-d, empty, singleton, 6-D)", test_shape_edge_cases()),
        ("OMP-parallel path (multi-dim)", test_omp_path_multidim()),
        ("1-D behavior unchanged", test_1d_unchanged()),
        ("Shape/size mismatch errors", test_shape_mismatch_errors()),
        ("Non-contiguous inputs rejected", test_noncontiguous_rejected()),
        ("Non-contiguous 1-D regression (latent bug fix)", test_noncontiguous_1d_regression()),
    ]

    failed = [name for name, passed in results if not passed]
    print("-" * 70)
    if failed:
        print(f"[FAIL] {len(failed)} test group(s) failed: {', '.join(failed)}")
        return 1
    print(f"[SUCCESS] All {len(results)} test groups passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
