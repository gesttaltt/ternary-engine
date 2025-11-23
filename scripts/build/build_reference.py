"""
build_reference.py - Build unoptimized C++ reference for benchmarking

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

This builds a baseline C++ implementation WITHOUT optimizations:
- No LUTs (uses conversion-based operations)
- No SIMD
- No force inline
- Minimal compiler optimizations (/O1 instead of /O2)

PURPOSE: Fair performance comparison to measure actual optimization impact,
not Python vs C++ differences.

USAGE: Run from project root directory:
    python scripts/build/build_reference.py

Artifacts are organized in: build/artifacts/reference/{timestamp}/
Latest build is copied to: build/artifacts/reference/latest/
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# Get project root (script is in scripts/build/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
ARTIFACTS_DIR = PROJECT_ROOT / "build" / "artifacts"

# Generate timestamp for this build
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Build directories
BUILD_TYPE_DIR = ARTIFACTS_DIR / "reference"
BUILD_TIMESTAMP_DIR = BUILD_TYPE_DIR / TIMESTAMP
BUILD_TEMP_DIR = BUILD_TIMESTAMP_DIR / "temp"
BUILD_OUTPUT_DIR = BUILD_TIMESTAMP_DIR / "output"
BUILD_LATEST_DIR = BUILD_TYPE_DIR / "latest"

def print_header():
    """Print build header"""
    print("\n" + "="*70)
    print("  REFERENCE UNOPTIMIZED BUILD")
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
    print("Building reference_cpp module...\n")

    # Generate setup code
    setup_code = f'''
from setuptools import setup, Extension
import pybind11
import os

PROJECT_ROOT = r"{PROJECT_ROOT}"

ext_modules = [
    Extension(
        'reference_cpp',
        [os.path.join(PROJECT_ROOT, 'benchmarks', 'reference_cpp.cpp')],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            PROJECT_ROOT
        ],
        language='c++',
        extra_compile_args=[
            '/O1',           # MSVC: Basic optimization only (NOT /O2)
            '/std:c++17',    # C++17 standard
            '/EHsc',         # Exception handling
            # NO /GL, /arch:AVX2, /openmp - minimal optimizations
        ],
        # NO /LTCG - no link-time optimization
    ),
]

setup(
    name='reference_cpp',
    version='0.1.0',
    author='Ternary Engine Team',
    description='Unoptimized C++ reference for fair benchmarking',
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
        print("\n[FAIL] Build failed")
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
        print(f"  [OK] {pyd_file.name} -> {dest}")

def print_summary():
    """Print build summary"""
    print("\n" + "="*70)
    print("  [SUCCESS] BUILD COMPLETE")
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
