"""
build_dense243.py - Build script for ternary_dense243_module

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Builds the Dense243 encoding module for high-density ternary storage.

USAGE: Run from project root directory:
    python scripts/build/build_dense243.py

OUTPUT: ternary_dense243_module.pyd (Windows) or .so (Linux/macOS)

DENSITY: 5 trits/byte (95.3% state utilization) vs 4 trits/byte (standard)
USE CASES: Persistent storage, network transmission, memory-bound workloads
FUTURE: TritNet integration (neural network-based operations)
"""

import sys
import platform
import shutil
from pathlib import Path
from datetime import datetime
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# =============================================================================
# Project Paths
# =============================================================================

# Get project root (script is in scripts/build/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
BUILD_DIR = PROJECT_ROOT / "build" / "artifacts" / "dense243"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BUILD_SUBDIR = BUILD_DIR / TIMESTAMP

# Create build directories
BUILD_SUBDIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = BUILD_SUBDIR / "temp"
OUTPUT_DIR = BUILD_SUBDIR / "output"
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Building ternary_dense243_module...")
print(f"Project root: {PROJECT_ROOT}")
print(f"Build directory: {BUILD_SUBDIR}")

# =============================================================================
# Compiler Configuration
# =============================================================================

# Platform-specific compiler flags
if sys.platform == "win32":
    # MSVC flags
    extra_compile_args = [
        "/O2",              # Maximum optimization
        "/GL",              # Whole program optimization
        "/std:c++17",       # C++17 standard
        "/EHsc",            # Exception handling
        "/W3",              # Warning level 3
    ]
    extra_link_args = [
        "/LTCG",            # Link-time code generation
    ]
else:
    # GCC/Clang flags
    extra_compile_args = [
        "-O3",              # Maximum optimization
        "-std=c++17",       # C++17 standard
        "-flto",            # Link-time optimization
        "-fvisibility=hidden",  # Hide symbols by default
    ]
    extra_link_args = [
        "-flto",            # Link-time optimization
    ]

# =============================================================================
# Extension Definition
# =============================================================================

ext_modules = [
    Pybind11Extension(
        "ternary_dense243_module",
        [str(PROJECT_ROOT / "ternary_engine" / "ternary_dense243_module.cpp")],
        include_dirs=[
            str(PROJECT_ROOT),
            str(PROJECT_ROOT / "ternary_core"),
            str(PROJECT_ROOT / "ternary_engine"),
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        cxx_std=17,
    ),
]

# =============================================================================
# Build Configuration
# =============================================================================

class CustomBuildExt(build_ext):
    """Custom build_ext to control output directory"""

    def build_extensions(self):
        # Set build temp directory
        self.build_temp = str(TEMP_DIR)
        super().build_extensions()

    def copy_extensions_to_source(self):
        """Override to copy to our custom output directory"""
        build_py = self.get_finalized_command('build_py')
        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)
            modpath = fullname.split('.')
            package = '.'.join(modpath[:-1])
            package_dir = build_py.get_package_dir(package)

            # Source: build/lib.*/module.pyd
            src = Path(self.build_lib) / filename

            # Destinations
            dest_output = OUTPUT_DIR / Path(filename).name
            dest_root = PROJECT_ROOT / Path(filename).name

            # Copy to output directory
            if src.exists():
                shutil.copy2(src, dest_output)
                print(f"Copied to: {dest_output}")

                # Also copy to project root for easy import
                shutil.copy2(src, dest_root)
                print(f"Copied to: {dest_root}")
            else:
                print(f"Warning: Built extension not found at {src}")

# =============================================================================
# Setup
# =============================================================================

setup(
    name="ternary_dense243_module",
    version="1.0.0",
    author="Jonathan Verdun",
    author_email="jonathan.verdun707@gmail.com",
    description="Dense243 encoding module (5 trits/byte) with TritNet-ready architecture",
    long_description=__doc__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
    python_requires=">=3.7",
)

# =============================================================================
# Post-Build Summary
# =============================================================================

print("\n" + "="*80)
print("Dense243 Module Build Complete")
print("="*80)
print(f"Output directory: {OUTPUT_DIR}")
print(f"Module also copied to project root for easy import")
print("\nTest the module:")
print("  python -c \"import ternary_dense243_module as td; print(td.__doc__)\"")
print("\nFeatures:")
print("  - Pack/unpack: 2-bit ↔ dense243 format")
print("  - Operations: tadd, tmul, tmin, tmax, tnot on dense243 data")
print("  - TritNet-ready: Backend selection for future NN integration")
print("\nDensity: 5 trits/byte (95.3% utilization) vs 4 trits/byte (standard)")
print("="*80 + "\n")

# Create symlink to latest build
latest_link = BUILD_DIR / "latest"
if latest_link.exists() or latest_link.is_symlink():
    latest_link.unlink()

try:
    latest_link.symlink_to(BUILD_SUBDIR, target_is_directory=True)
    print(f"Created symlink: {latest_link} -> {BUILD_SUBDIR}")
except OSError as e:
    # Symlink creation might fail on Windows without admin privileges
    print(f"Note: Could not create symlink (this is OK): {e}")
