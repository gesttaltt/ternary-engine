#!/usr/bin/env python3
"""
test_zero_skip_gemm.py - Zero-skip ternary GEMM correctness tests

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Validates the ternary_zero_skip_gemm module against a NumPy float64
reference (C = A @ B):
  1. ZeroSkipWeights.gemm (CSC j-parallel) across shapes and sparsities,
     including SIMD-unfriendly sizes (M=1, K=1, odd dimensions)
  2. gemm_tiled (CSR k-parallel) agrees with the reference
  3. Scalar path (use_avx2=False) agrees with the AVX2 path
  4. Convenience module-level gemm()
  5. info()/sparsity_info() report the true zero fraction

Reference is computed in float64 and compared with a tolerance that
covers float32 accumulation-order differences.

Exits 0 with a SKIP notice if the module is not built (it is optional);
build it with: python build/build_zero_skip_gemm.py

USAGE: python tests/python/test_zero_skip_gemm.py
"""

import sys
from pathlib import Path

# Add project root to path (compiled modules live there, not under tests/python/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import ternary_zero_skip_gemm as zs
except ImportError:
    print("[SKIP] ternary_zero_skip_gemm not built (optional).")
    print("       Build with: python build/build_zero_skip_gemm.py")
    sys.exit(0)

SEED = 42
RTOL = 1e-4
ATOL = 1e-4

# (M, K, N) shapes: typical, SIMD-unfriendly, and degenerate edges
SHAPES = [
    (1, 1, 1),
    (1, 17, 3),
    (7, 33, 5),
    (16, 64, 32),
    (64, 128, 96),
    (33, 129, 65),   # all odd, exercises SIMD remainder loops
]

# Weight densities: dense, balanced-ternary typical (~1/3 zeros), sparse
ZERO_FRACTIONS = [0.0, 0.33, 0.9]


def make_inputs(rng, m, k, n, zero_frac):
    A = rng.standard_normal((m, k), dtype=np.float32)
    p = zero_frac
    B = rng.choice(
        np.array([-1, 0, 1], dtype=np.int8),
        size=(k, n),
        p=[(1 - p) / 2, p, (1 - p) / 2],
    ).astype(np.int8)
    return A, B


def reference(A, B):
    return (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)


def check(name, got, ref):
    if got.shape != ref.shape:
        print(f"  [FAIL] {name}: shape {got.shape} != {ref.shape}")
        return False
    if not np.allclose(got, ref, rtol=RTOL, atol=ATOL):
        bad = np.abs(got - ref).max()
        print(f"  [FAIL] {name}: max abs diff {bad}")
        return False
    return True


def test_gemm_vs_reference() -> bool:
    """ZeroSkipWeights.gemm matches NumPy across shapes and sparsities."""
    rng = np.random.default_rng(SEED)
    ok = True
    for m, k, n in SHAPES:
        for zf in ZERO_FRACTIONS:
            A, B = make_inputs(rng, m, k, n, zf)
            ref = reference(A, B)
            got = zs.ZeroSkipWeights(B).gemm(A)
            ok &= check(f"gemm M={m} K={k} N={n} zero={zf}", got, ref)
    if ok:
        print(f"  [OK] gemm matches reference for {len(SHAPES)} shapes "
              f"x {len(ZERO_FRACTIONS)} sparsities")
    return ok


def test_extreme_weights() -> bool:
    """All-zero and no-zero weight matrices are handled correctly."""
    rng = np.random.default_rng(SEED)
    A = rng.standard_normal((8, 32), dtype=np.float32)
    ok = True

    B_zero = np.zeros((32, 16), dtype=np.int8)
    ok &= check("all-zero B", zs.ZeroSkipWeights(B_zero).gemm(A),
                np.zeros((8, 16), dtype=np.float32))

    B_dense = rng.choice(np.array([-1, 1], dtype=np.int8), size=(32, 16))
    B_dense = B_dense.astype(np.int8)
    ok &= check("no-zero B", zs.ZeroSkipWeights(B_dense).gemm(A),
                reference(A, B_dense))
    if ok:
        print("  [OK] all-zero and no-zero weight matrices")
    return ok


def test_tiled_and_scalar_paths() -> bool:
    """gemm_tiled and use_avx2=False agree with the reference."""
    rng = np.random.default_rng(SEED)
    ok = True
    for m, k, n in ((7, 33, 5), (64, 128, 96), (33, 129, 65)):
        A, B = make_inputs(rng, m, k, n, 0.33)
        ref = reference(A, B)
        w = zs.ZeroSkipWeights(B)
        ok &= check(f"gemm_tiled M={m} K={k} N={n}", w.gemm_tiled(A), ref)
        ok &= check(f"gemm scalar M={m} K={k} N={n}",
                    w.gemm(A, use_avx2=False), ref)
    if ok:
        print("  [OK] tiled (CSR) and scalar paths match reference")
    return ok


def test_module_level_gemm() -> bool:
    """Convenience zs.gemm(A, B) matches the reference."""
    rng = np.random.default_rng(SEED)
    A, B = make_inputs(rng, 16, 64, 32, 0.33)
    ok = check("module-level gemm", zs.gemm(A, B), reference(A, B))
    if ok:
        print("  [OK] module-level gemm")
    return ok


def test_sparsity_info() -> bool:
    """info() and sparsity_info() report the true zero fraction."""
    rng = np.random.default_rng(SEED)
    _, B = make_inputs(rng, 1, 128, 64, 0.33)
    true_zero = float((B == 0).mean())
    true_nnz = int((B != 0).sum())

    info = zs.ZeroSkipWeights(B).info()
    sinfo = zs.sparsity_info(B)
    ok = True
    for name, d in (("info", info), ("sparsity_info", sinfo)):
        if abs(d["zero_fraction"] - true_zero) > 1e-9 or d["nnz"] != true_nnz:
            print(f"  [FAIL] {name}: {d} vs zero={true_zero}, nnz={true_nnz}")
            ok = False
    if ok:
        print(f"  [OK] sparsity reporting (zero_fraction={true_zero:.3f}, "
              f"nnz={true_nnz})")
    return ok


def test_dense_gemm_vs_reference() -> bool:
    """DenseWeights.gemm matches NumPy across shapes and sparsities.

    Added 2026-08-20 alongside ternary_gemm_dense.h/.cpp (see
    src/engine/bindings_zero_skip_gemm.cpp module docstring): the
    dense-packed kernel added because CSC/CSR index overhead exceeds the
    dense array's own size at ternary's actual ~33-40% zero density.
    Same shapes/sparsities as test_gemm_vs_reference, including M=1
    (pure GEMV, the exact case that motivated this kernel), so a
    regression here would be caught the same way a ZeroSkipWeights
    regression would.
    """
    rng = np.random.default_rng(SEED)
    ok = True
    for m, k, n in SHAPES:
        for zf in ZERO_FRACTIONS:
            A, B = make_inputs(rng, m, k, n, zf)
            ref = reference(A, B)
            got = zs.DenseWeights(B).gemm(A)
            ok &= check(f"dense gemm M={m} K={k} N={n} zero={zf}", got, ref)
    if ok:
        print(f"  [OK] DenseWeights.gemm matches reference for {len(SHAPES)} "
              f"shapes x {len(ZERO_FRACTIONS)} sparsities")
    return ok


def test_dense_extreme_weights() -> bool:
    """All-zero and no-zero weight matrices are handled correctly (dense kernel)."""
    rng = np.random.default_rng(SEED)
    A = rng.standard_normal((8, 32), dtype=np.float32)
    ok = True

    B_zero = np.zeros((32, 16), dtype=np.int8)
    ok &= check("dense all-zero B", zs.DenseWeights(B_zero).gemm(A),
                np.zeros((8, 16), dtype=np.float32))

    B_dense = rng.choice(np.array([-1, 1], dtype=np.int8), size=(32, 16))
    B_dense = B_dense.astype(np.int8)
    ok &= check("dense no-zero B", zs.DenseWeights(B_dense).gemm(A),
                reference(A, B_dense))
    if ok:
        print("  [OK] DenseWeights: all-zero and no-zero weight matrices")
    return ok


def test_dense_scalar_path() -> bool:
    """DenseWeights.gemm(use_avx2=False) agrees with the AVX2 path and reference."""
    rng = np.random.default_rng(SEED)
    ok = True
    for m, k, n in ((7, 33, 5), (64, 128, 96), (33, 129, 65)):
        A, B = make_inputs(rng, m, k, n, 0.33)
        ref = reference(A, B)
        w = zs.DenseWeights(B)
        ok &= check(f"dense scalar M={m} K={k} N={n}",
                    w.gemm(A, use_avx2=False), ref)
    if ok:
        print("  [OK] DenseWeights scalar path matches reference")
    return ok


def test_dense_info() -> bool:
    """DenseWeights.info() reports consistent K, N, and packed_bytes.

    packed_bytes should equal K * ceil(N/8) * 8 regardless of sparsity
    (unlike ZeroSkipWeights.info()'s nnz-dependent size) -- that
    sparsity-independence is the whole point (see module docstring).
    """
    rng = np.random.default_rng(SEED)
    ok = True
    for m, k, n in ((1, 128, 64), (1, 128, 65), (1, 128, 71)):  # incl. N % 8 != 0
        _, B = make_inputs(rng, m, k, n, 0.33)
        info = zs.DenseWeights(B).info()
        expected_bytes = k * ((n + 7) // 8) * 8
        if info["K"] != k or info["N"] != n or info["packed_bytes"] != expected_bytes:
            print(f"  [FAIL] dense info K={k} N={n}: {info} "
                  f"(expected packed_bytes={expected_bytes})")
            ok = False
    if ok:
        print("  [OK] DenseWeights.info() (K, N, packed_bytes incl. N%8!=0 tail)")
    return ok


def test_input_validation() -> bool:
    """Malformed inputs raise ValueError instead of crashing or reading garbage.

    Added 2026-08-18 alongside the CLAUDE.md gap #6 Cluster C fix
    (src/engine/py_array_validate.h): these validation paths previously had
    zero test coverage in this file, and the fix itself changed real
    behavior in one case (sparsity_info() gained a C-contiguity check it
    was missing -- see the comment at that call site in
    bindings_zero_skip_gemm.cpp for why the old code was a latent bug, not
    just "less validated").
    """
    rng = np.random.default_rng(SEED)
    ok = True

    def expect_value_error(label, fn):
        nonlocal ok
        try:
            fn()
            print(f"  [FAIL] {label}: expected ValueError, none raised")
            ok = False
        except ValueError:
            pass
        except Exception as e:
            print(f"  [FAIL] {label}: expected ValueError, got {type(e).__name__}: {e}")
            ok = False

    A, B = make_inputs(rng, 8, 32, 16, 0.33)
    B_noncontig = np.asfortranarray(B)          # valid values, wrong layout
    A_noncontig = np.asfortranarray(A)
    B_1d = B.ravel()
    A_1d = A.ravel()

    # ZeroSkipWeights constructor: B must be 2-D and C-contiguous
    expect_value_error("ZeroSkipWeights(1-D B)", lambda: zs.ZeroSkipWeights(B_1d))
    expect_value_error("ZeroSkipWeights(non-contiguous B)",
                        lambda: zs.ZeroSkipWeights(B_noncontig))

    # gemm()/gemm_tiled(): A must be 2-D, C-contiguous, and K-compatible
    w = zs.ZeroSkipWeights(B)
    expect_value_error("gemm(1-D A)", lambda: w.gemm(A_1d))
    expect_value_error("gemm(non-contiguous A)", lambda: w.gemm(A_noncontig))
    expect_value_error("gemm(wrong K)", lambda: w.gemm(A[:, :16]))  # K=16 != weight K=32
    expect_value_error("gemm_tiled(non-contiguous A)", lambda: w.gemm_tiled(A_noncontig))

    # Module-level gemm(): both A and B validated
    expect_value_error("gemm(1-D A, B)", lambda: zs.gemm(A_1d, B))
    expect_value_error("gemm(A, non-contiguous B)", lambda: zs.gemm(A, B_noncontig))
    expect_value_error("gemm(A, 1-D B)", lambda: zs.gemm(A, B_1d))

    # sparsity_info(): the fix itself -- previously accepted non-contiguous B
    # silently (and would have read wrong values off a transposed/strided
    # view, since the loop indexes linearly assuming row-major layout).
    expect_value_error("sparsity_info(1-D B)", lambda: zs.sparsity_info(B_1d))
    expect_value_error("sparsity_info(non-contiguous B)",
                        lambda: zs.sparsity_info(B_noncontig))

    # DenseWeights: same validation surface as ZeroSkipWeights
    expect_value_error("DenseWeights(1-D B)", lambda: zs.DenseWeights(B_1d))
    expect_value_error("DenseWeights(non-contiguous B)",
                        lambda: zs.DenseWeights(B_noncontig))
    dw = zs.DenseWeights(B)
    expect_value_error("DenseWeights.gemm(1-D A)", lambda: dw.gemm(A_1d))
    expect_value_error("DenseWeights.gemm(non-contiguous A)", lambda: dw.gemm(A_noncontig))
    expect_value_error("DenseWeights.gemm(wrong K)", lambda: dw.gemm(A[:, :16]))

    if ok:
        print("  [OK] malformed inputs correctly rejected (2-D, contiguous, "
              "shape-compatible checks)")
    return ok


def main() -> int:
    print("=" * 70)
    print("Zero-Skip Ternary GEMM Correctness Tests")
    print(f"AVX2: {zs.has_avx2}  OpenMP: {zs.has_openmp}")
    print("=" * 70)

    results = [
        ("GEMM vs NumPy reference", test_gemm_vs_reference()),
        ("Extreme weight matrices", test_extreme_weights()),
        ("Tiled and scalar paths", test_tiled_and_scalar_paths()),
        ("Module-level gemm", test_module_level_gemm()),
        ("Sparsity reporting", test_sparsity_info()),
        ("Dense: GEMM vs NumPy reference", test_dense_gemm_vs_reference()),
        ("Dense: extreme weight matrices", test_dense_extreme_weights()),
        ("Dense: scalar path", test_dense_scalar_path()),
        ("Dense: info reporting", test_dense_info()),
        ("Input validation", test_input_validation()),
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
