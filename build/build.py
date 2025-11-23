"""
build.py - Build script for ternary_simd_engine module (Standard Optimized)

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

This provides an organized build system with timestamped artifacts.

USAGE: Run from project root directory:
    python build/build.py

Artifacts are organized in: build/artifacts/standard/{timestamp}/
Latest build is copied to: build/artifacts/standard/latest/
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

# Build directories (use shorter paths to avoid Windows MAX_PATH)
BUILD_TYPE_DIR = ARTIFACTS_DIR / "standard"
BUILD_TIMESTAMP_DIR = BUILD_TYPE_DIR / TIMESTAMP
# Use "t" and "o" for temp/output to keep paths short
BUILD_TEMP_DIR = BUILD_TIMESTAMP_DIR / "t"
BUILD_OUTPUT_DIR = BUILD_TIMESTAMP_DIR / "o"
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
    print("Building ternary_simd_engine module...\n")

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
is_macos = system == 'Darwin'
is_linux = system == 'Linux'

if is_windows:
    # MSVC flags (Windows)
    compile_args = [
        '/O2',           # Maximum optimization
        '/GL',           # Whole program optimization
        '/arch:AVX2',    # Enable AVX2
        '/openmp',       # Enable OpenMP (OPT-001)
        '/std:c++17',    # C++17 standard
        '/EHsc',         # Exception handling
    ]
    link_args = ['/LTCG']  # Link-time code generation
elif is_macos:
    # Clang flags (macOS) - OpenMP not supported by Apple Clang
    # Note: macOS can be ARM64 (Apple Silicon) or x86_64 (Intel)
    import platform as plat
    machine = plat.machine()

    if machine == 'arm64':
        # Apple Silicon (M1/M2/M3)
        compile_args = [
            '-O3',           # Maximum optimization
            '-mcpu=apple-m1',# Apple Silicon optimization
            '-std=c++17',    # C++17 standard
            '-flto',         # Link-time optimization
        ]
        print("Note: Building for Apple Silicon (ARM64)")
        print("Note: OpenMP and AVX2 not available on ARM (tests will be skipped)")
    else:
        # Intel macOS
        compile_args = [
            '-O3',           # Maximum optimization
            '-march=haswell',# Haswell architecture (AVX2 support)
            '-mavx2',        # Explicit AVX2
            '-std=c++17',    # C++17 standard
            '-flto',         # Link-time optimization
        ]
        print("Note: Building for Intel macOS (x86_64)")
        print("Note: OpenMP disabled (Apple Clang does not support -fopenmp)")
    link_args = []
else:
    # GCC flags (Linux)
    compile_args = [
        '-O3',           # Maximum optimization
        '-march=haswell',# Haswell architecture (AVX2 support, safer than native for CI)
        '-mavx2',        # Explicit AVX2
        '-fopenmp',      # Enable OpenMP (OPT-001)
        '-std=c++17',    # C++17 standard
        '-flto',         # Link-time optimization
    ]
    link_args = ['-fopenmp']  # OpenMP linker flag

ext_modules = [
    Extension(
        'ternary_simd_engine',
        [os.path.join(PROJECT_ROOT, 'ternary_engine', 'bindings_core_ops.cpp')],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            PROJECT_ROOT,
            os.path.join(PROJECT_ROOT, 'ternary_core')
        ],
        language='c++',
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
]

setup(
    name='ternary_simd_engine',
    version='0.1.0',
    author='Ternary Engine Team',
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

    # Run build using --inplace with a short temp directory
    # Use C:\Temp on Windows to avoid MAX_PATH issues with deep project paths
    import tempfile
    import platform as plat

    if plat.system() == 'Windows':
        # Use a short path for Windows temp directory
        short_temp = Path("C:/Temp/ternary_build")
        short_temp.mkdir(parents=True, exist_ok=True)
        temp_arg = ["--build-temp", str(short_temp)]
    else:
        # On Unix, use system temp
        temp_arg = []

    result = subprocess.run(
        [sys.executable, str(setup_temp_path), "build_ext", "--inplace"] + temp_arg,
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
    print(f"\nCopying to output directory...")

    # Find module files in project root (built with --inplace)
    module_files = list(PROJECT_ROOT.glob("ternary_simd_engine*.pyd")) + \
                   list(PROJECT_ROOT.glob("ternary_simd_engine*.so"))

    if not module_files:
        print("  [ERROR] No module files found!")
        return

    # Copy to output directory
    BUILD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for module_file in module_files:
        dest = BUILD_OUTPUT_DIR / module_file.name
        shutil.copy2(module_file, dest)
        print(f"  [OK] {module_file.name} -> output directory")

    # Copy to latest directory
    if BUILD_LATEST_DIR.exists():
        shutil.rmtree(BUILD_LATEST_DIR)
    BUILD_LATEST_DIR.mkdir(parents=True, exist_ok=True)

    for module_file in module_files:
        dest = BUILD_LATEST_DIR / module_file.name
        shutil.copy2(module_file, dest)
        print(f"  [OK] {module_file.name} -> latest directory")

def print_summary():
    """Print build summary"""
    print("\n" + "="*70)
    print("  [SUCCESS] BUILD COMPLETE")
    print("="*70)
    print(f"\nBuild artifacts:")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Timestamped:  {BUILD_TIMESTAMP_DIR}")
    print(f"  Latest:       {BUILD_LATEST_DIR}")

    # Show file sizes for both .pyd (Windows) and .so (Linux/macOS)
    module_files = list(PROJECT_ROOT.glob("ternary_simd_engine*.pyd")) + \
                   list(PROJECT_ROOT.glob("ternary_simd_engine*.so"))
    if module_files:
        print(f"\nGenerated modules:")
        for module_file in module_files:
            size_kb = module_file.stat().st_size / 1024
            print(f"  - {module_file.name} ({size_kb:.1f} KB)")

def main():
    print_header()
    setup_directories()
    build_module()
    copy_to_latest()
    print_summary()

if __name__ == "__main__":
    main()
