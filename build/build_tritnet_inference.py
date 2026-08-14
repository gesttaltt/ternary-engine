#!/usr/bin/env python3
"""
build_tritnet_inference.py - Build TritNet Inference Engine Python Module

Compiles the Phase 3 TritNet inference engine (models/tritnet/inference/) with
AVX2 SIMD optimization and exposes it to Python as ternary_tritnet_inference.
Mirrors build_tritnet_gemm.py's structure and flags.

USAGE:
    python build/build_tritnet_inference.py [options]

OPTIONS:
    --verbose       Show detailed compilation output
    --clean         Clean before building
    --no-validate   Skip post-build validation

OUTPUT:
    ternary_tritnet_inference.pyd (Windows) or .so (Linux/macOS)
    Located in project root for direct import

VALIDATION:
    python -c "import ternary_tritnet_inference as tn; print(tn.has_avx2())"

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0
"""

import os
import sys
import platform
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))


def get_compiler_flags():
    """Platform-specific compiler flags. Always includes AVX2 -- this
    project's stated baseline is "AVX2 required" (CLAUDE.md Platform
    Requirements), and tritnet_inference_avx2.h hard-errors at compile time
    without __AVX2__, so unlike build_tritnet_gemm.py there is no
    --naive-only equivalent here; the module always needs AVX2 codegen
    available. Runtime dispatch (has_avx2()) still degrades gracefully on a
    CPU that lacks it at *execution* time -- see bindings_tritnet_inference.cpp.

    Uses a FIXED baseline (-march=haswell / /arch:AVX2), not -march=native,
    matching build.py's and build_dense243.py's convention -- found
    2026-08-14 via code review: this module's docstring specifically
    promises graceful degradation via has_avx2() on any AVX2-only machine,
    but -march=native would let the compiler auto-vectorize the "scalar"
    fallback path (tritnet_inference.h, compiled in the same translation
    unit as the AVX2 kernel) with whatever ISA extensions happen to be
    present on the BUILD machine. A binary built on a newer-than-AVX2 host
    and run/redistributed to an AVX2-only one would then SIGILL on the
    supposedly-safe fallback path -- the exact crash class already fixed
    once in this repo for bindings_dense243.cpp (2026-08-12). Other sibling
    scripts (build_tritnet_gemm.py, build_zero_skip_gemm.py,
    build_backend.py) still use -march=native; none of them make this
    module's specific runtime-portability promise, so they're a separate,
    pre-existing inconsistency not fixed here.
    """
    is_windows = platform.system() == "Windows"
    is_msvc = is_windows and "mingw" not in sys.platform.lower()

    if is_msvc:
        compile_args = [
            "/O2", "/GL", "/arch:AVX2", "/std:c++17",
            "/fp:fast", "/GS-", "/Oi", "/Ot", "/favor:AMD64",
        ]
        link_args = ["/LTCG"]
    else:
        compile_args = [
            "-O3", "-march=haswell", "-mavx2", "-mfma", "-std=c++17",
            "-flto", "-ffast-math", "-funroll-loops",
        ]
        link_args = ["-flto"]

    return compile_args, link_args


def build_module(verbose=False, clean=False):
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    from setuptools import setup

    print("=" * 70)
    print("Building TritNet Inference Engine Python Module")
    print("=" * 70)

    sources = ["src/engine/bindings_tritnet_inference.cpp"]

    # Header-only engine (tritnet_inference.h, tritnet_inference_avx2.h,
    # tritnet_weights.h) -- nothing else to compile, matches the project's
    # existing pattern for compile-time-generated-weights headers.
    include_dirs = [
        "src",
        ".",  # for "models/tritnet/inference/..." and "core/..." includes
    ]

    compile_args, link_args = get_compiler_flags()

    print(f"\nPlatform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Compiler flags: {' '.join(compile_args[:3])}...")
    print(f"Source files: {len(sources)} files")

    ext_modules = [
        Pybind11Extension(
            "ternary_tritnet_inference",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            language="c++",
        ),
    ]

    if clean:
        print("\n Cleaning build artifacts...")
        import shutil
        artifacts_dir = ROOT_DIR / "build" / "artifacts" / "tritnet_inference"
        if artifacts_dir.exists():
            shutil.rmtree(artifacts_dir)
            print(f"   Removed: {artifacts_dir}")
        for ext in [".pyd", ".so"]:
            for old_file in ROOT_DIR.glob(f"ternary_tritnet_inference*{ext}"):
                old_file.unlink()
                print(f"   Removed: {old_file.name}")

    print("\nCompiling...")
    stdout = None if verbose else subprocess.DEVNULL

    setup(
        name="ternary_tritnet_inference",
        version="1.0.0",
        description="TritNet C++ Inference Engine (Phase 3) - Python bindings",
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
        zip_safe=False,
        script_args=["build_ext", "--inplace"],
        options={"build_ext": {"verbose": 1 if verbose else 0}},
    )

    print("\nBuild complete!")

    built_files = list(ROOT_DIR.glob("ternary_tritnet_inference*.pyd")) + \
                  list(ROOT_DIR.glob("ternary_tritnet_inference*.so"))
    if built_files:
        output_file = built_files[0]
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"\nOutput: {output_file.name} ({file_size_mb:.2f} MB)")
        print(f"   Location: {output_file.parent}")
    else:
        print("\nWarning: Could not locate built module file")

    print("\n" + "=" * 70)


def validate_module():
    print("\nValidating module...")
    try:
        import ternary_tritnet_inference as tn
        print("Module imported successfully")

        for func_name in ["tnot", "tadd", "tmul", "tmin", "tmax", "has_avx2"]:
            status = "OK" if hasattr(tn, func_name) else "MISSING"
            print(f"  {func_name}(): {status}")

        print(f"\n  built_with_avx2 = {tn.built_with_avx2}")
        print(f"  has_avx2()      = {tn.has_avx2()}")

        import numpy as np
        a = np.array([[0, 1, 2, 0, 1]], dtype=np.uint8)
        out = tn.tnot(a)
        expected = np.array([[2, 1, 0, 2, 1]], dtype=np.uint8)  # negate each trit
        if np.array_equal(out, expected):
            print("\nSmoke test (tnot) PASSED")
        else:
            print(f"\nSmoke test (tnot) FAILED: got {out.tolist()}, expected {expected.tolist()}")
            return False

    except ImportError as e:
        print(f"Failed to import module: {e}")
        return False
    except Exception as e:
        print(f"Validation error: {e}")
        return False

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build TritNet Inference Engine Python module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--no-validate", action="store_true")

    args = parser.parse_args()
    os.chdir(ROOT_DIR)

    try:
        if not args.validate_only:
            build_module(verbose=args.verbose, clean=args.clean)

        if not args.no_validate:
            if not validate_module():
                sys.exit(1)

        print("\nAll done!")

    except Exception as e:
        print(f"\nBuild failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
