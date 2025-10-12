"""
setup.py - Build script for ternary_core_simd_full module (Standard Optimized)

Copyright 2025 Ternary Core Contributors
Licensed under the Apache License, Version 2.0

This provides an organized build system with timestamped artifacts.

USAGE: Run from project root directory:
    python build/scripts/setup.py

Artifacts are organized in: build/artifacts/standard/{timestamp}/
Latest build is copied to: build/artifacts/standard/latest/
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# Get project root (two levels up from this script)
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "build" / "artifacts"

# Generate timestamp for this build
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Build directories
BUILD_TYPE_DIR = ARTIFACTS_DIR / "standard"
BUILD_TIMESTAMP_DIR = BUILD_TYPE_DIR / TIMESTAMP
BUILD_TEMP_DIR = BUILD_TIMESTAMP_DIR / "temp"
BUILD_OUTPUT_DIR = BUILD_TIMESTAMP_DIR / "output"
BUILD_LATEST_DIR = BUILD_TYPE_DIR / "latest"

def print_header():
    """Print build header"""
    print("\n" + "="*70)
    print("  STANDARD OPTIMIZED BUILD")
    print(f"  Timestamp: {TIMESTAMP}")
    print("="*70 + "\n")

def setup_directories():
    """Create build directory structure"""
    BUILD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created build directories:")
    print(f"  Temp:   {BUILD_TEMP_DIR}")
    print(f"  Output: {BUILD_OUTPUT_DIR}\n")

def build_module():
    """Build the module using setuptools"""
    print("Building ternary_core_simd_full module...\n")

    # Generate setup code
    setup_code = f'''
from setuptools import setup, Extension
import pybind11
import os

PROJECT_ROOT = r"{PROJECT_ROOT}"

ext_modules = [
    Extension(
        'ternary_core_simd_full',
        [os.path.join(PROJECT_ROOT, 'ternary_core_simd_full.cpp')],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            PROJECT_ROOT
        ],
        language='c++',
        extra_compile_args=[
            '/O2',           # MSVC: Maximum optimization
            '/GL',           # MSVC: Whole program optimization
            '/arch:AVX2',    # MSVC: Enable AVX2
            '/openmp',       # MSVC: Enable OpenMP (OPT-001)
            '/std:c++17',    # C++17 standard
            '/EHsc',         # Exception handling
        ],
        extra_link_args=[
            '/LTCG',         # Link-time code generation
        ],
    ),
]

setup(
    name='ternary_core_simd_full',
    version='0.1.0',
    author='Ternary Core Team',
    description='AVX2-optimized ternary logic operations with Phase 0 LUT optimizations',
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires='>=3.7',
)
'''

    # Write temporary setup file
    setup_temp_path = BUILD_TEMP_DIR / "setup_temp.py"
    with open(setup_temp_path, "w") as f:
        f.write(setup_code)

    # Run build
    result = subprocess.run(
        [sys.executable, str(setup_temp_path), "build_ext",
         "--build-temp", str(BUILD_TEMP_DIR),
         "--build-lib", str(BUILD_OUTPUT_DIR)],
        cwd=str(PROJECT_ROOT),
        capture_output=False
    )

    setup_temp_path.unlink()

    if result.returncode != 0:
        print("\n❌ Build failed")
        sys.exit(1)

    return True

def copy_to_latest():
    """Copy build output to latest directory"""
    print(f"\nCopying to latest directory...")

    # Remove old latest directory
    if BUILD_LATEST_DIR.exists():
        shutil.rmtree(BUILD_LATEST_DIR)

    # Copy entire timestamp directory
    shutil.copytree(BUILD_TIMESTAMP_DIR, BUILD_LATEST_DIR)

    # Also copy .pyd to project root for convenience
    for pyd_file in BUILD_OUTPUT_DIR.glob("*.pyd"):
        dest = PROJECT_ROOT / pyd_file.name
        shutil.copy2(pyd_file, dest)
        print(f"  ✓ {pyd_file.name} → {dest}")

def print_summary():
    """Print build summary"""
    print("\n" + "="*70)
    print("  ✅ BUILD COMPLETE")
    print("="*70)
    print(f"\nBuild artifacts:")
    print(f"  Timestamped: {BUILD_TIMESTAMP_DIR}")
    print(f"  Latest:      {BUILD_LATEST_DIR}")

    # Show file sizes
    pyd_files = list(BUILD_OUTPUT_DIR.glob("*.pyd"))
    if pyd_files:
        print(f"\nGenerated modules:")
        for pyd_file in pyd_files:
            size_kb = pyd_file.stat().st_size / 1024
            print(f"  - {pyd_file.name} ({size_kb:.1f} KB)")

def main():
    print_header()
    setup_directories()
    build_module()
    copy_to_latest()
    print_summary()

if __name__ == "__main__":
    main()
