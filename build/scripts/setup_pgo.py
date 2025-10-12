"""
setup_pgo.py - Profile-Guided Optimization build system for ternary_simd_engine

Copyright 2025 Ternary Core Contributors
Licensed under the Apache License, Version 2.0

This script implements a 3-phase PGO build process:
1. Instrumentation build (generate profiling instrumentation)
2. Profile collection (run benchmarks to collect runtime data)
3. Optimized build (use profile data for final optimization)

Usage (run from project root):
    python build/scripts/setup_pgo.py instrument    # Phase 1: Build with instrumentation
    python build/scripts/setup_pgo.py profile       # Phase 2: Run profiling workload
    python build/scripts/setup_pgo.py optimize      # Phase 3: Build optimized version
    python build/scripts/setup_pgo.py clean         # Clean PGO artifacts
    python build/scripts/setup_pgo.py full          # Run all phases automatically

Artifacts are organized in:
  - build/artifacts/pgo/instrumented/{timestamp}/
  - build/artifacts/pgo/optimized/{timestamp}/
  - build/artifacts/pgo/pgo_data/
  - build/artifacts/pgo/latest/ (final optimized build)
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

# PGO base directory
PGO_BASE_DIR = ARTIFACTS_DIR / "pgo"
PGO_DATA_DIR = PGO_BASE_DIR / "pgo_data"
PROFILE_DATA = PGO_DATA_DIR / "ternary_simd_engine.pgd"
PGO_LATEST_DIR = PGO_BASE_DIR / "latest"

def print_phase(phase_name, description):
    """Print phase header"""
    print("\n" + "="*70)
    print(f"  Phase: {phase_name}")
    print(f"  {description}")
    print("="*70 + "\n")

def clean_pgo():
    """Clean PGO artifacts and build directories"""
    print_phase("CLEAN", "Removing PGO data and build artifacts")

    # Remove entire PGO directory
    if PGO_BASE_DIR.exists():
        print(f"Removing {PGO_BASE_DIR}/")
        shutil.rmtree(PGO_BASE_DIR)

    # Remove compiled modules from project root
    for pyd_file in PROJECT_ROOT.glob("ternary_simd_engine*.pyd"):
        print(f"Removing {pyd_file}")
        pyd_file.unlink()

    for so_file in PROJECT_ROOT.glob("ternary_simd_engine*.so"):
        print(f"Removing {so_file}")
        so_file.unlink()

    print("\n✅ Clean complete")

def phase1_instrument():
    """Phase 1: Build with instrumentation"""
    print_phase("PHASE 1: INSTRUMENT",
                "Building with instrumentation to collect profiling data")

    # Generate timestamp for this build
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build directories for instrumented build
    instrumented_dir = PGO_BASE_DIR / "instrumented" / timestamp
    temp_dir = instrumented_dir / "temp"
    output_dir = instrumented_dir / "output"

    # Create directories
    PGO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Build directories:")
    print(f"  Temp:   {temp_dir}")
    print(f"  Output: {output_dir}")
    print(f"  PGO Data: {PGO_DATA_DIR}\n")

    # Build with instrumentation flags
    setup_code = f'''
from setuptools import setup, Extension
import pybind11
import os

PROJECT_ROOT = r"{PROJECT_ROOT}"
PGO_DATA_DIR = r"{PGO_DATA_DIR}"

ext_modules = [
    Extension(
        'ternary_simd_engine',
        [os.path.join(PROJECT_ROOT, 'ternary_simd_engine.cpp')],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            PROJECT_ROOT
        ],
        language='c++',
        extra_compile_args=[
            '/O2',                    # Maximum optimization
            '/GL',                    # Whole program optimization
            '/arch:AVX2',            # Enable AVX2
            '/openmp',               # Enable OpenMP
            '/std:c++17',            # C++17 standard
            '/EHsc',                 # Exception handling
        ],
        extra_link_args=[
            '/LTCG:PGI',             # OPT-114: Generate instrumented code for profiling
            f'/PGD:{{PGO_DATA_DIR}}\\ternary_simd_engine.pgd',  # Profile database location
        ],
    ),
]

setup(
    name='ternary_simd_engine',
    version='0.1.0',
    author='Ternary Core Team',
    description='AVX2-optimized ternary logic with PGO Phase 1 (Instrumentation)',
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires='>=3.7',
)
'''

    setup_temp_path = temp_dir / "setup_temp.py"
    with open(setup_temp_path, "w") as f:
        f.write(setup_code)

    # Run build from project root with custom build directories
    result = subprocess.run(
        [sys.executable, str(setup_temp_path), "build_ext",
         "--build-temp", str(temp_dir),
         "--build-lib", str(output_dir)],
        cwd=str(PROJECT_ROOT),
        capture_output=False
    )

    setup_temp_path.unlink()

    if result.returncode != 0:
        print("\n❌ Instrumentation build failed")
        sys.exit(1)

    # Copy .pyd to project root for profiling
    for pyd_file in output_dir.glob("*.pyd"):
        dest = PROJECT_ROOT / pyd_file.name
        shutil.copy2(pyd_file, dest)
        print(f"\n  ✓ Copied {pyd_file.name} to project root")

    print("\n✅ Phase 1 complete: Instrumented build ready")
    print(f"   Build artifacts: {instrumented_dir}")
    print(f"   Profile data will be written to: {PROFILE_DATA}")

def phase2_profile():
    """Phase 2: Run profiling workload"""
    print_phase("PHASE 2: PROFILE",
                "Running benchmarks to collect runtime profiling data")

    # Check if instrumented build exists in project root
    pyd_file = list(PROJECT_ROOT.glob("ternary_simd_engine*.pyd"))
    if not pyd_file:
        print("❌ No instrumented module found. Run 'python build/scripts/setup_pgo.py instrument' first.")
        sys.exit(1)

    print(f"Found instrumented module: {pyd_file[0]}")
    print("\nRunning profiling workload...")
    print("(This will take ~8 minutes - running full benchmark suite)\n")

    # Run the benchmark suite from project root
    benchmark_script = PROJECT_ROOT / "benchmarks" / "bench_phase0.py"
    result = subprocess.run(
        [sys.executable, str(benchmark_script)],
        cwd=str(PROJECT_ROOT),
        capture_output=False
    )

    if result.returncode != 0:
        print("\n❌ Profiling workload failed")
        sys.exit(1)

    # Check if profile data was generated
    pgc_files = list(PGO_DATA_DIR.glob("**/*.pgc")) if PGO_DATA_DIR.exists() else []
    if not pgc_files:
        print("\n⚠️  Warning: No .pgc files found in PGO data directory")
        print("   Profile data may not have been collected properly")
    else:
        print(f"\n✅ Found {len(pgc_files)} profile counter files:")
        for pgc in pgc_files:
            print(f"   - {pgc.name}")

    if PROFILE_DATA.exists():
        size_mb = PROFILE_DATA.stat().st_size / (1024 * 1024)
        print(f"\n✅ Phase 2 complete: Profile data collected ({size_mb:.2f} MB)")
        print(f"   Location: {PROFILE_DATA}")
    else:
        print(f"\n⚠️  Warning: Expected profile database not found at {PROFILE_DATA}")
        print("   Continuing to Phase 3 anyway (MSVC may have written it elsewhere)")

def phase3_optimize():
    """Phase 3: Build optimized version using profile data"""
    print_phase("PHASE 3: OPTIMIZE",
                "Building final optimized version using collected profile data")

    # Check for profile data
    pgc_files = list(PGO_DATA_DIR.glob("**/*.pgc")) if PGO_DATA_DIR.exists() else []
    if not pgc_files and not PROFILE_DATA.exists():
        print("⚠️  Warning: No profile data found")
        print("   Run 'python build/scripts/setup_pgo.py profile' first for best results")
        print("   Continuing with optimization anyway...\n")

    # Generate timestamp for this build
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build directories for optimized build
    optimized_dir = PGO_BASE_DIR / "optimized" / timestamp
    temp_dir = optimized_dir / "temp"
    output_dir = optimized_dir / "output"

    # Create directories
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Build directories:")
    print(f"  Temp:   {temp_dir}")
    print(f"  Output: {output_dir}\n")

    # Build with profile-guided optimization
    setup_code = f'''
from setuptools import setup, Extension
import pybind11
import os

PROJECT_ROOT = r"{PROJECT_ROOT}"
PGO_DATA_DIR = r"{PGO_DATA_DIR}"

ext_modules = [
    Extension(
        'ternary_simd_engine',
        [os.path.join(PROJECT_ROOT, 'ternary_simd_engine.cpp')],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            PROJECT_ROOT
        ],
        language='c++',
        extra_compile_args=[
            '/O2',                    # Maximum optimization
            '/GL',                    # Whole program optimization
            '/arch:AVX2',            # Enable AVX2
            '/openmp',               # Enable OpenMP
            '/std:c++17',            # C++17 standard
            '/EHsc',                 # Exception handling
        ],
        extra_link_args=[
            '/LTCG:PGO',             # OPT-114: Use profile data for optimization
            f'/PGD:{{PGO_DATA_DIR}}\\ternary_simd_engine.pgd',  # Profile database location
        ],
    ),
]

setup(
    name='ternary_simd_engine',
    version='0.1.0',
    author='Ternary Core Team',
    description='AVX2-optimized ternary logic with PGO Phase 3 (Optimized)',
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires='>=3.7',
)
'''

    setup_temp_path = temp_dir / "setup_temp.py"
    with open(setup_temp_path, "w") as f:
        f.write(setup_code)

    # Run build from project root with custom build directories
    result = subprocess.run(
        [sys.executable, str(setup_temp_path), "build_ext",
         "--build-temp", str(temp_dir),
         "--build-lib", str(output_dir)],
        cwd=str(PROJECT_ROOT),
        capture_output=False
    )

    setup_temp_path.unlink()

    if result.returncode != 0:
        print("\n❌ Optimized build failed")
        sys.exit(1)

    # Copy to latest directory
    if PGO_LATEST_DIR.exists():
        shutil.rmtree(PGO_LATEST_DIR)
    shutil.copytree(optimized_dir, PGO_LATEST_DIR)

    # Copy .pyd to project root
    for pyd_file in output_dir.glob("*.pyd"):
        dest = PROJECT_ROOT / pyd_file.name
        shutil.copy2(pyd_file, dest)
        print(f"\n  ✓ Copied {pyd_file.name} to project root")
        size_kb = pyd_file.stat().st_size / 1024
        print(f"    Size: {size_kb:.1f} KB")

    print("\n✅ Phase 3 complete: Profile-guided optimized build ready")
    print(f"   Build artifacts: {optimized_dir}")
    print(f"   Latest: {PGO_LATEST_DIR}")
    print("\n" + "="*70)
    print("  🎉 PGO BUILD COMPLETE!")
    print("="*70)
    print("\nYour module is now optimized based on actual runtime behavior.")
    print("Expected improvements: 5-15% in hot paths")
    print("\nTo verify improvements, run:")
    print("  python benchmarks/bench_phase0.py")

def run_full_pgo():
    """Run all PGO phases sequentially"""
    print("\n" + "="*70)
    print("  FULL PGO BUILD PROCESS")
    print("  This will take ~10-15 minutes")
    print("="*70)

    clean_pgo()
    phase1_instrument()
    phase2_profile()
    phase3_optimize()

    print("\n" + "="*70)
    print("  ✅ FULL PGO PROCESS COMPLETE")
    print("="*70)

def print_usage():
    """Print usage information"""
    print(__doc__)
    print("\nCurrent PGO Status:")
    print(f"  PGO Base Dir:     {PGO_BASE_DIR}/ {'✅ exists' if PGO_BASE_DIR.exists() else '❌ not found'}")
    print(f"  Profile Database: {PROFILE_DATA} {'✅ exists' if PROFILE_DATA.exists() else '❌ not found'}")

    # Check for instrumented builds
    instrumented_dir = PGO_BASE_DIR / "instrumented"
    if instrumented_dir.exists():
        instrumented_builds = sorted([d.name for d in instrumented_dir.iterdir() if d.is_dir()])
        if instrumented_builds:
            print(f"  Instrumented Builds: {len(instrumented_builds)} ({instrumented_builds[-1]} latest)")

    # Check for optimized builds
    optimized_dir = PGO_BASE_DIR / "optimized"
    if optimized_dir.exists():
        optimized_builds = sorted([d.name for d in optimized_dir.iterdir() if d.is_dir()])
        if optimized_builds:
            print(f"  Optimized Builds: {len(optimized_builds)} ({optimized_builds[-1]} latest)")

    # Check for profile data
    pgc_files = list(PGO_DATA_DIR.glob("**/*.pgc")) if PGO_DATA_DIR.exists() else []
    if pgc_files:
        print(f"  Profile Counters: {len(pgc_files)} .pgc files found")

    # Check for compiled module
    pyd_files = list(PROJECT_ROOT.glob("ternary_simd_engine*.pyd"))
    if pyd_files:
        print(f"  Compiled Module:  {pyd_files[0].name} ✅")

def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(0)

    command = sys.argv[1].lower()

    commands = {
        'instrument': phase1_instrument,
        'profile': phase2_profile,
        'optimize': phase3_optimize,
        'clean': clean_pgo,
        'full': run_full_pgo,
        'help': print_usage,
    }

    if command not in commands:
        print(f"❌ Unknown command: {command}")
        print_usage()
        sys.exit(1)

    commands[command]()

if __name__ == "__main__":
    main()
