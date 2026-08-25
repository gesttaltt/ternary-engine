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

def build_module(enable_perfetto=False):
    """Build the module using setuptools.

    enable_perfetto: adds -DTERNARY_ENABLE_PERFETTO and links the vendored
    Perfetto SDK (third_party/perfetto/) + its glue TU
    (src/core/profiling/ternary_profiler_perfetto.cc), activating the
    real TERNARY_PROFILE_TASK_BEGIN/END call sites already wired into
    bindings_core_ops.cpp's hot paths (see
    src/core/profiling/ternary_profiler.h and
    reports/2026-08-25/PERFETTO_PROFILER_INTEGRATION.md, which validated
    this same backend via a standalone native demo before it was wired
    in here). Default is False -- matching this project's stated
    "zero overhead when profiling disabled" design, the default build
    is unaffected either way and doesn't touch these files at all.
    Verified on Linux x64 only; Windows/macOS wiring is mechanically
    consistent with the rest of this script's per-platform branching but
    UNVERIFIED -- no Windows/macOS machine available to build and run it.
    """
    print("Building ternary_simd_engine module...\n")
    if enable_perfetto:
        print("  (with Perfetto profiler support: -DTERNARY_ENABLE_PERFETTO)\n")

    # Generate setup code
    setup_code = f'''
from setuptools import setup, Extension
import pybind11
import os
import platform

PROJECT_ROOT = r"{PROJECT_ROOT}"
ENABLE_PERFETTO = {enable_perfetto}

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

# Optional Perfetto profiler support (build/build.py --enable-perfetto).
# Default build never touches these files or flags -- zero overhead when
# disabled, matching this project's stated profiler design. See
# src/core/profiling/ternary_profiler.h and
# reports/2026-08-25/PERFETTO_PROFILER_INTEGRATION.md.
extra_sources = []
if ENABLE_PERFETTO:
    if is_windows:
        compile_args.append('/DTERNARY_ENABLE_PERFETTO')
        # UNVERIFIED: no Windows machine available to build/run this.
        # Perfetto's SDK is cross-platform and MSVC is a supported
        # compiler upstream, but this project's own verify-by-execution
        # discipline means this branch is mechanically consistent with
        # the rest of this script, not a tested claim.
    else:
        compile_args.append('-DTERNARY_ENABLE_PERFETTO')
        link_args.append('-lpthread')
    extra_sources = [
        os.path.join(PROJECT_ROOT, 'third_party', 'perfetto', 'perfetto.cc'),
        os.path.join(PROJECT_ROOT, 'src', 'core', 'profiling', 'ternary_profiler_perfetto.cc'),
    ]

ext_modules = [
    Extension(
        'ternary_simd_engine',
        [os.path.join(PROJECT_ROOT, 'src', 'engine', 'bindings_core_ops.cpp')] + extra_sources,
        include_dirs=[
            pybind11.get_include(),
            PROJECT_ROOT,
            os.path.join(PROJECT_ROOT, 'src')
        ],
        language='c++',
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
]

setup(
    name='ternary_simd_engine',
    version='1.1.0',
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

    # --force: distutils' build_ext skips the ENTIRE extension rebuild (not
    # just individual stale .o files) whenever the existing --inplace .so
    # already looks newer than every source file -- it has no way to know
    # compiler flags or the sources list changed since the last run.
    # Found 2026-08-25 while adding --enable-perfetto: running the default
    # build immediately followed by --enable-perfetto silently reused the
    # first build's .so untouched (has_perfetto stayed False) because the
    # new perfetto.cc/ternary_profiler_perfetto.cc sources' mtimes predate
    # the .so that had just been built moments earlier. This script always
    # intends a full rebuild, not incremental reuse, so --force makes that
    # actually true regardless of timestamps -- not specific to the
    # Perfetto flag, this was a latent bug in every prior invocation too,
    # just never visible because compiler flags/sources rarely changed
    # between two consecutive runs without also touching bindings_core_ops.cpp.
    result = subprocess.run(
        [sys.executable, str(setup_temp_path), "build_ext", "--inplace", "--force"] + temp_arg,
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
        return False

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

    return True

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
    import argparse
    parser = argparse.ArgumentParser(description="Build ternary_simd_engine (Standard Optimized)")
    parser.add_argument(
        "--enable-perfetto", action="store_true",
        help="Build with real Perfetto profiler tracing support "
             "(-DTERNARY_ENABLE_PERFETTO). Default build is unaffected -- "
             "zero overhead when this flag is omitted. See "
             "reports/2026-08-25/PERFETTO_PROFILER_INTEGRATION.md. "
             "Verified on Linux x64 only."
    )
    args = parser.parse_args()

    print_header()
    setup_directories()
    build_module(enable_perfetto=args.enable_perfetto)
    if not copy_to_latest():
        print("\n[FAIL] BUILD INCOMPLETE - no module files were produced")
        sys.exit(1)
    print_summary()
    if args.enable_perfetto:
        print("\nBuilt with Perfetto support. Drive tracing from Python:")
        print("  import ternary_simd_engine as tc")
        print("  tc.perfetto_start('trace.perfetto-trace')")
        print("  # ... tc.tadd(a, b), etc. ...")
        print("  tc.perfetto_stop()")

if __name__ == "__main__":
    main()
