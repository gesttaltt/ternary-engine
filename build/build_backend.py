"""
build_backend.py - Build script for ternary_backend module (Backend API)

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Builds the ternary_backend Python module that exposes the v1.2.0 backend system.

USAGE: Run from project root directory:
    python build/build_backend.py

Artifacts are organized in: build/artifacts/backend/{timestamp}/
Latest build is copied to: build/artifacts/backend/latest/
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# Get project root (script is in build/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
ARTIFACTS_DIR = PROJECT_ROOT / "build" / "artifacts"

# Generate timestamp for this build
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Build directories
BUILD_TYPE_DIR = ARTIFACTS_DIR / "backend"
BUILD_TIMESTAMP_DIR = BUILD_TYPE_DIR / TIMESTAMP
BUILD_TEMP_DIR = BUILD_TIMESTAMP_DIR / "t"
BUILD_OUTPUT_DIR = BUILD_TIMESTAMP_DIR / "o"
BUILD_LATEST_DIR = BUILD_TYPE_DIR / "latest"

def print_header():
    """Print build header"""
    print("\n" + "="*70)
    print("  TERNARY BACKEND MODULE BUILD")
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
    print("Building ternary_backend module...\n")

    # Generate setup code
    setup_code = f'''
from setuptools import setup, Extension
import pybind11
import os
import platform

PROJECT_ROOT = r"{PROJECT_ROOT}"

# Platform-specific compiler flags
system = platform.system()
is_windows = system == 'Windows'

if is_windows:
    # MSVC flags (Windows)
    compile_args = [
        '/O2',           # Maximum optimization
        '/GL',           # Whole program optimization
        '/arch:AVX2',    # Enable AVX2
        '/openmp',       # OpenMP parallelization (CRITICAL - enables 14× speedup)
        '/std:c++20',    # C++20 standard (required for designated initializers)
        '/EHsc',         # Exception handling
    ]
    link_args = ['/LTCG']  # Link-time code generation
else:
    # GCC/Clang flags (Linux/macOS)
    compile_args = [
        '-O3',           # Maximum optimization
        '-march=native', # Native CPU optimization
        '-mavx2',        # AVX2 support
        '-fopenmp',      # OpenMP parallelization (CRITICAL - enables 14× speedup)
        '-std=c++17',    # C++17 standard
        '-flto',         # Link-time optimization
    ]
    link_args = ['-flto', '-fopenmp']  # Link OpenMP library

# Backend module sources
sources = [
    os.path.join(PROJECT_ROOT, "src/engine/bindings_backend_api.cpp"),
    os.path.join(PROJECT_ROOT, "src/core/simd/ternary_backend_dispatch.cpp"),
    os.path.join(PROJECT_ROOT, "src/core/simd/ternary_backend_scalar.cpp"),
    os.path.join(PROJECT_ROOT, "src/core/simd/ternary_backend_avx2_v1.cpp"),
    os.path.join(PROJECT_ROOT, "src/core/simd/ternary_backend_avx2_v2.cpp"),
]

ext_modules = [
    Extension(
        'ternary_backend',
        sources=sources,
        include_dirs=[
            pybind11.get_include(),
            os.path.join(PROJECT_ROOT, "src"),
        ],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language='c++',
    ),
]

setup(
    name='ternary_backend',
    version='1.2.0',
    author='Ternary Engine Team',
    description='Backend system for Ternary Engine v1.2.0',
    ext_modules=ext_modules,
    zip_safe=False,
)
'''

    # Write setup file
    setup_file = BUILD_TEMP_DIR / "setup_backend.py"
    with open(setup_file, 'w') as f:
        f.write(setup_code)

    # Build command
    cmd = [
        sys.executable,
        str(setup_file),
        "build_ext",
        f"--build-temp={BUILD_TEMP_DIR}",
        f"--build-lib={BUILD_OUTPUT_DIR}"
    ]

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        print("\n❌ Build failed")
        return False

    print("\n✅ Build successful")
    return True

def copy_to_project_root():
    """Copy built module to project root for easy import"""
    print("\nCopying module to project root...")

    # Find the built module (.pyd on Windows, .so on Linux/macOS)
    built_files = list(BUILD_OUTPUT_DIR.glob("ternary_backend*.pyd")) + \
                  list(BUILD_OUTPUT_DIR.glob("ternary_backend*.so"))

    if not built_files:
        print("❌ No built module found")
        return False

    for built_file in built_files:
        target = PROJECT_ROOT / built_file.name
        shutil.copy2(built_file, target)
        print(f"  Copied: {built_file.name} → project root")

        # Also create/update 'latest' symlink
        try:
            latest_target = BUILD_LATEST_DIR / built_file.name
            BUILD_LATEST_DIR.mkdir(parents=True, exist_ok=True)
            if latest_target.exists():
                latest_target.unlink()
            shutil.copy2(built_file, latest_target)
            print(f"  Updated: build/artifacts/backend/latest/{built_file.name}")
        except Exception as e:
            print(f"  Warning: Could not update latest symlink: {e}")

    return True

def print_summary():
    """Print build summary"""
    print("\n" + "="*70)
    print("  BUILD COMPLETE")
    print("="*70)
    print(f"\nModule location: {PROJECT_ROOT}")
    print("\nUsage:")
    print("  import ternary_backend")
    print("  ternary_backend.init()")
    print("  print(ternary_backend.list_backends())")
    print("  ternary_backend.set_backend('AVX2_v2')")
    print("  result = ternary_backend.tadd(a, b)")
    print()

def main():
    print_header()
    setup_directories()

    if not build_module():
        return 1

    if not copy_to_project_root():
        return 1

    print_summary()
    return 0

if __name__ == "__main__":
    sys.exit(main())
