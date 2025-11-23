#!/usr/bin/env python3
"""
build_tritnet_gemm.py - Build TritNet GEMM Python Module

Compiles the ternary matrix multiplication module with AVX2 SIMD optimizations.
Builds both naive and AVX2-optimized implementations.

USAGE:
    python scripts/build/build_tritnet_gemm.py [options]

OPTIONS:
    --naive-only    Build only naive implementation (no AVX2)
    --verbose       Show detailed compilation output
    --clean         Clean before building

OUTPUT:
    ternary_tritnet_gemm.pyd (Windows) or .so (Linux/macOS)
    Located in project root for direct import

VALIDATION:
    python -c "import ternary_tritnet_gemm as gemm; print(gemm.__doc__)"

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0
"""

import os
import sys
import platform
import subprocess
from pathlib import Path

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))

def get_compiler_flags():
    """
    Get platform-specific compiler flags for optimized build.

    Returns:
        tuple: (compile_args, link_args, extra_objects)
    """
    is_windows = platform.system() == "Windows"

    # On Windows, assume MSVC unless explicitly using MinGW
    is_msvc = is_windows and "mingw" not in sys.platform.lower()

    if is_msvc:
        # MSVC compiler flags
        compile_args = [
            "/O2",              # Maximum optimization
            "/GL",              # Whole program optimization
            "/arch:AVX2",       # Enable AVX2 SIMD
            "/std:c++17",       # C++17 standard
            "/fp:fast",         # Fast floating-point model
            "/GS-",             # Disable security checks (for perf)
            "/Oi",              # Enable intrinsics
            "/Ot",              # Favor fast code
            "/favor:AMD64",     # Optimize for 64-bit
        ]

        link_args = [
            "/LTCG",            # Link-time code generation
        ]

        extra_objects = []

    else:
        # GCC/Clang compiler flags
        compile_args = [
            "-O3",              # Maximum optimization
            "-march=native",    # Use all CPU features
            "-mavx2",           # Enable AVX2 explicitly
            "-std=c++17",       # C++17 standard
            "-flto",            # Link-time optimization
            "-ffast-math",      # Fast math (careful with this)
            "-funroll-loops",   # Loop unrolling
        ]

        link_args = [
            "-flto",            # Link-time optimization
        ]

        extra_objects = []

    return compile_args, link_args, extra_objects


def build_module(naive_only=False, verbose=False, clean=False):
    """
    Build the TritNet GEMM Python module.

    Args:
        naive_only: If True, only build naive implementation (no AVX2)
        verbose: Show detailed compilation output
        clean: Clean before building
    """
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    from setuptools import setup

    print("=" * 70)
    print("Building TritNet GEMM Python Module")
    print("=" * 70)

    # Source files
    sources = [
        "ternary_engine/ternary_tritnet_gemm_module.cpp",
        "src/tritnet_gemm_naive.cpp",
    ]

    # Add AVX2 implementation if not naive-only
    if not naive_only:
        sources.append("src/tritnet_gemm_avx2.cpp")
        print("✓ Including AVX2 SIMD optimization")
    else:
        print("⚠ Building naive-only (no SIMD)")

    # Include directories
    include_dirs = [
        "include",
        "ternary_core",
        "ternary_engine",
        "ternary_engine/experimental/dense243",
    ]

    # Get compiler flags
    compile_args, link_args, extra_objects = get_compiler_flags()

    # Add preprocessor defines
    define_macros = []
    if not naive_only:
        define_macros.append(("__AVX2__", "1"))

    # Print build configuration
    print(f"\nPlatform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Compiler flags: {' '.join(compile_args[:3])}...")
    print(f"Include dirs: {len(include_dirs)} directories")
    print(f"Source files: {len(sources)} files")

    # Create extension module
    ext_modules = [
        Pybind11Extension(
            "ternary_tritnet_gemm",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            extra_objects=extra_objects,
            define_macros=define_macros,
            language="c++",
        ),
    ]

    # Clean if requested
    if clean:
        print("\n🧹 Cleaning build artifacts...")
        import shutil
        build_dir = ROOT_DIR / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir)

        # Remove old .pyd/.so files
        for ext in [".pyd", ".so"]:
            for old_file in ROOT_DIR.glob(f"ternary_tritnet_gemm*{ext}"):
                old_file.unlink()
                print(f"   Removed: {old_file.name}")

    # Build configuration
    print("\n🔨 Compiling...")

    # Capture output if not verbose
    stdout = None if verbose else subprocess.DEVNULL
    stderr = None if verbose else subprocess.STDOUT

    setup(
        name="ternary_tritnet_gemm",
        version="1.0.0",
        description="TritNet Direct Ternary GEMM - Optimized Matrix Multiplication",
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
        zip_safe=False,
        script_args=["build_ext", "--inplace"],
        options={
            "build_ext": {
                "verbose": 1 if verbose else 0,
            }
        },
    )

    # Find built module
    print("\n✅ Build complete!")

    # Locate output file
    built_files = list(ROOT_DIR.glob("ternary_tritnet_gemm*.pyd")) + \
                  list(ROOT_DIR.glob("ternary_tritnet_gemm*.so"))

    if built_files:
        output_file = built_files[0]
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"\n📦 Output: {output_file.name} ({file_size_mb:.2f} MB)")
        print(f"   Location: {output_file.parent}")
    else:
        print("\n⚠ Warning: Could not locate built module file")

    print("\n" + "=" * 70)


def validate_module():
    """
    Validate the built module by importing and testing basic functionality.
    """
    print("\n🔍 Validating module...")

    try:
        import ternary_tritnet_gemm as gemm
        print("✓ Module imported successfully")

        # Check available functions
        expected_funcs = [
            "gemm", "gemm_scaled", "convert_from_bitnet",
            "set_num_threads", "get_tile_size", "benchmark", "validate"
        ]

        for func_name in expected_funcs:
            if hasattr(gemm, func_name):
                print(f"✓ {func_name}() available")
            else:
                print(f"✗ {func_name}() MISSING")

        # Run quick validation test
        print("\n🧪 Running correctness test...")
        error = gemm.validate(8, 8, 160)  # Tiny test
        print(f"   Max error: {error:.2e} (should be < 1e-6)")

        if error < 1e-6:
            print("✅ Validation PASSED")
        else:
            print("❌ Validation FAILED - error too large")

        # Quick benchmark
        print("\n⚡ Quick benchmark (8×8×160)...")
        time_ms = gemm.benchmark(8, 8, 160, num_runs=100)
        if time_ms > 0:
            print(f"   Time: {time_ms:.3f} ms")
        else:
            print("   ⚠ Benchmark not yet implemented (returns 0)")

    except ImportError as e:
        print(f"❌ Failed to import module: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

    return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build TritNet GEMM Python module with SIMD optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build/build_tritnet_gemm.py
      Build with AVX2 optimization (default)

  python scripts/build/build_tritnet_gemm.py --naive-only
      Build reference implementation only

  python scripts/build/build_tritnet_gemm.py --clean --verbose
      Clean rebuild with detailed output

  python scripts/build/build_tritnet_gemm.py --validate-only
      Skip build, just validate existing module
        """
    )

    parser.add_argument("--naive-only", action="store_true",
                       help="Build only naive implementation (no AVX2)")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed compilation output")
    parser.add_argument("--clean", action="store_true",
                       help="Clean before building")
    parser.add_argument("--validate-only", action="store_true",
                       help="Skip build, just validate existing module")
    parser.add_argument("--no-validate", action="store_true",
                       help="Skip validation after build")

    args = parser.parse_args()

    # Change to root directory
    os.chdir(ROOT_DIR)

    try:
        # Build module (unless validate-only)
        if not args.validate_only:
            build_module(
                naive_only=args.naive_only,
                verbose=args.verbose,
                clean=args.clean
            )

        # Validate module (unless --no-validate)
        if not args.no_validate:
            if not validate_module():
                sys.exit(1)

        print("\n✅ All done!")

    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
