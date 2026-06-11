#!/usr/bin/env python3
"""
build_zero_skip_gemm.py - Build Zero-Skip Ternary GEMM Python Module

USAGE:
    python3 build/build_zero_skip_gemm.py
    python3 build/build_zero_skip_gemm.py --verbose
    python3 build/build_zero_skip_gemm.py --clean

OUTPUT:
    ternary_zero_skip_gemm.so (Linux/macOS) in project root

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0
"""

import os
import sys
import platform
import argparse
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR   = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))


def get_compiler_flags():
    is_windows = platform.system() == "Windows"
    is_msvc = is_windows and "mingw" not in sys.platform.lower()

    if is_msvc:
        compile_args = [
            "/O2", "/GL", "/arch:AVX2", "/std:c++17", "/fp:fast",
            "/GS-", "/Oi", "/Ot", "/openmp",
        ]
        link_args = ["/LTCG"]
    else:
        compile_args = [
            "-O3", "-march=native", "-mavx2", "-mfma",
            "-std=c++17", "-flto", "-ffast-math", "-funroll-loops",
            "-fopenmp",
        ]
        link_args = ["-flto", "-fopenmp"]

    return compile_args, link_args


def build_module(verbose=False, clean=False):
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    from setuptools import setup

    print("=" * 60)
    print("Building ternary_zero_skip_gemm")
    print("=" * 60)

    if clean:
        for pattern in ["ternary_zero_skip_gemm*.so", "ternary_zero_skip_gemm*.pyd"]:
            for f in ROOT_DIR.glob(pattern):
                f.unlink()
                print(f"  Removed: {f.name}")

    compile_args, link_args = get_compiler_flags()

    ext = Pybind11Extension(
        "ternary_zero_skip_gemm",
        sources=[
            "src/core/simd/ternary_gemm_zero_skip.cpp",
            "src/engine/bindings_zero_skip_gemm.cpp",
        ],
        include_dirs=["src/core/simd"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language="c++",
    )

    print(f"Platform:  {platform.system()} {platform.machine()}")
    print(f"Python:    {sys.version.split()[0]}")
    print(f"Flags:     {' '.join(compile_args[:4])} ...")
    print()

    import subprocess
    stdout = None if verbose else subprocess.DEVNULL
    stderr = None if verbose else subprocess.STDOUT

    setup(
        name="ternary_zero_skip_gemm",
        ext_modules=[ext],
        cmdclass={"build_ext": build_ext},
        zip_safe=False,
        script_args=["build_ext", "--inplace"],
        options={"build_ext": {"verbose": 1 if verbose else 0}},
    )

    built = list(ROOT_DIR.glob("ternary_zero_skip_gemm*.so")) + \
            list(ROOT_DIR.glob("ternary_zero_skip_gemm*.pyd"))

    if built:
        sz = built[0].stat().st_size / 1024
        print(f"\nOutput: {built[0].name}  ({sz:.1f} KB)")
    else:
        print("\nWarning: could not locate output file")

    print("=" * 60)


def validate():
    print("\nValidating...")
    import numpy as np
    import ternary_zero_skip_gemm as zs

    print(f"  AVX2:   {zs.has_avx2}")
    print(f"  OpenMP: {zs.has_openmp}")

    rng = np.random.default_rng(0)
    M, K, N = 32, 60, 40
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.choice([-1, 0, 1], size=(K, N)).astype(np.int8)

    info = zs.sparsity_info(B)
    print(f"  Sparsity: zero_fraction={info['zero_fraction']:.3f}")

    # Reference
    C_ref = A @ B.astype(np.float32)

    # ZeroSkipWeights class
    W = zs.ZeroSkipWeights(B)
    C_skip = W.gemm(A)
    err = float(abs(C_skip - C_ref).max())
    print(f"  AVX2 max_error: {err:.2e}", "PASS" if err < 1e-4 else "FAIL")

    C_scalar = W.gemm(A, use_avx2=False)
    err2 = float(abs(C_scalar - C_ref).max())
    print(f"  Scalar max_error: {err2:.2e}", "PASS" if err2 < 1e-4 else "FAIL")

    C_tiled = W.gemm_tiled(A)
    err3 = float(abs(C_tiled - C_ref).max())
    print(f"  Tiled max_error:  {err3:.2e}", "PASS" if err3 < 1e-4 else "FAIL")

    print("Validation done.")
    return err < 1e-4 and err2 < 1e-4 and err3 < 1e-4


def main():
    parser = argparse.ArgumentParser(description="Build ternary_zero_skip_gemm")
    parser.add_argument("--verbose",  action="store_true")
    parser.add_argument("--clean",    action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT_DIR)

    build_module(verbose=args.verbose, clean=args.clean)

    if not args.no_validate:
        ok = validate()
        if not ok:
            sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
