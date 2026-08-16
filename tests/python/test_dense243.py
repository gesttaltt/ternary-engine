#!/usr/bin/env python3
"""
test_dense243.py - Dense243 encoding correctness tests

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Validates the ternary_dense243_module (5 trits/byte encoding):
  1. Pack/unpack round-trip at multiple sizes, including non-multiples of 5
  2. All operations (tadd, tmul, tmin, tmax, tnot) against the reference
     ternary_simd_engine on random inputs
  3. Density constant and byte-count helper

Exits 0 with a SKIP notice if the module is not built (it is optional);
build it with: python build/build_dense243.py

USAGE: python tests/python/test_dense243.py
"""

import sys
from pathlib import Path

# Add project root to path (compiled modules live there, not under tests/python/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import ternary_dense243_module as td
except ImportError:
    print("[SKIP] ternary_dense243_module not built (optional).")
    print("       Build with: python build/build_dense243.py")
    sys.exit(0)

import ternary_simd_engine as eng

SEED = 42
BINARY_OPS = ("tadd", "tmul", "tmin", "tmax")


def test_roundtrip() -> bool:
    """Pack/unpack must be lossless, including sizes not divisible by 5."""
    rng = np.random.default_rng(SEED)
    ok = True
    for n in (1, 4, 5, 6, 243, 1_000, 99_998, 100_000):
        a = rng.integers(0, 3, n, dtype=np.uint8)
        packed = td.pack(a)
        expected_bytes = td.bytes_for_trits(n)
        if len(packed) != expected_bytes:
            print(f"  [FAIL] n={n}: packed {len(packed)} bytes, "
                  f"expected {expected_bytes}")
            ok = False
            continue
        unpacked = td.unpack(packed, n)
        if not np.array_equal(a, unpacked):
            print(f"  [FAIL] n={n}: round-trip mismatch")
            ok = False
    if ok:
        print("  [OK] round-trip lossless for all tested sizes")
    return ok


def test_ops_match_reference() -> bool:
    """Dense243 ops must agree with the reference SIMD engine."""
    rng = np.random.default_rng(SEED)
    n = 100_000
    a = rng.integers(0, 3, n, dtype=np.uint8)
    b = rng.integers(0, 3, n, dtype=np.uint8)
    pa, pb = td.pack(a), td.pack(b)

    ok = True
    for op in BINARY_OPS:
        ref = getattr(eng, op)(a, b)
        got = td.unpack(getattr(td, op)(pa, pb), n)
        if np.array_equal(ref, got):
            print(f"  [OK] {op} matches reference ({n:,} trits)")
        else:
            print(f"  [FAIL] {op} mismatch vs reference")
            ok = False

    ref = eng.tnot(a)
    got = td.unpack(td.tnot(pa), n)
    if np.array_equal(ref, got):
        print(f"  [OK] tnot matches reference ({n:,} trits)")
    else:
        print("  [FAIL] tnot mismatch vs reference")
        ok = False
    return ok


def test_density() -> bool:
    """Advertised density is 5 trits/byte."""
    if td.DENSITY == 5 and td.bytes_for_trits(100_000) == 20_000:
        print(f"  [OK] density = {td.DENSITY} trits/byte")
        return True
    print(f"  [FAIL] density = {td.DENSITY}, bytes_for_trits(100000) = "
          f"{td.bytes_for_trits(100_000)}")
    return False


def main() -> int:
    print("=" * 70)
    print("Dense243 Correctness Tests")
    print(f"Backend: {td.get_backend()}")
    print("=" * 70)

    results = [
        ("Pack/unpack round-trip", test_roundtrip()),
        ("Operations vs reference", test_ops_match_reference()),
        ("Density constants", test_density()),
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
