"""
build_fusion.py - Build script for ternary_fusion_engine (Phase 4.0 PoC)

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Builds the operation fusion module for benchmarking and validation.

USAGE: Run from project root directory:
    python build_fusion.py build_ext --inplace

OUTPUT: ternary_fusion_engine.pyd (Windows) or .so (Linux/macOS)
"""

import sys
import platform
from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# Determine compiler and platform-specific flags
is_windows = sys.platform == "win32"
is_msvc = is_windows and platform.python_compiler().startswith("MSC")

# Base compilation flags
compile_args = []
link_args = []

if is_msvc:
    # MSVC flags
    compile_args.extend([
        "/O2",          # Optimize for speed
        "/arch:AVX2",   # Enable AVX2 instructions
        "/openmp",      # Enable OpenMP
        "/std:c++17",   # C++17 standard
        "/W3",          # Warning level 3
    ])
    link_args.extend([
        "/openmp"
    ])
else:
    # GCC/Clang flags
    compile_args.extend([
        "-O3",          # Maximum optimization
        "-mavx2",       # Enable AVX2 instructions
        "-fopenmp",     # Enable OpenMP
        "-std=c++17",   # C++17 standard
        "-Wall",        # All warnings
        "-Wextra",      # Extra warnings
    ])
    link_args.extend([
        "-fopenmp"
    ])

# Extension module definition
ext_modules = [
    Pybind11Extension(
        "ternary_fusion_engine",
        ["ternary_simd_engine_fusion.cpp"],
        include_dirs=[
            str(Path(__file__).parent),  # Project root
        ],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language="c++"
    ),
]

def print_build_info():
    """Print build configuration"""
    print("\n" + "="*70)
    print("  TERNARY FUSION ENGINE BUILD (Phase 4.0 PoC)")
    print("="*70)
    print(f"Platform: {sys.platform}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Compiler: {'MSVC' if is_msvc else 'GCC/Clang'}")
    print(f"Optimizations: AVX2, OpenMP, C++17")
    print("="*70 + "\n")

if __name__ == "__main__":
    print_build_info()

    setup(
        name="ternary_fusion_engine",
        version="0.1.0-poc",
        description="Ternary Operation Fusion Engine - Phase 4.0 Proof of Concept",
        author="Jonathan Verdun (Ternary Engine Project)",
        license="Apache License 2.0",
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
        python_requires=">=3.7",
        install_requires=[
            "numpy>=1.20.0",
            "pybind11>=2.6.0"
        ]
    )

    print("\n" + "="*70)
    print("  BUILD COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Test the module:")
    print("     python -c \"import ternary_fusion_engine; print('SUCCESS')\"")
    print("\n  2. Run rigorous benchmarks:")
    print("     python benchmarks/bench_fusion_poc.py")
    print("\n  3. Validate results against theoretical predictions")
    print("="*70 + "\n")
