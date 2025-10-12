"""
setup_pgo.py - Profile-Guided Optimization build system for ternary_core_simd_full

Copyright 2025 Ternary Core Contributors
Licensed under the Apache License, Version 2.0

This script implements a 3-phase PGO build process:
1. Instrumentation build (generate profiling instrumentation)
2. Profile collection (run benchmarks to collect runtime data)
3. Optimized build (use profile data for final optimization)

Usage:
    python setup_pgo.py instrument    # Phase 1: Build with instrumentation
    python setup_pgo.py profile       # Phase 2: Run profiling workload
    python setup_pgo.py optimize      # Phase 3: Build optimized version
    python setup_pgo.py clean         # Clean PGO artifacts
    python setup_pgo.py full          # Run all phases automatically
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path

# PGO directories and files
PGO_DIR = Path("pgo_data")
PROFILE_DATA = PGO_DIR / "ternary_core_simd_full.pgd"
BUILD_DIR = Path("build")
DIST_DIR = Path("dist")

def print_phase(phase_name, description):
    """Print phase header"""
    print("\n" + "="*70)
    print(f"  Phase: {phase_name}")
    print(f"  {description}")
    print("="*70 + "\n")

def clean_pgo():
    """Clean PGO artifacts and build directories"""
    print_phase("CLEAN", "Removing PGO data and build artifacts")

    dirs_to_clean = [PGO_DIR, BUILD_DIR, DIST_DIR]
    for dir_path in dirs_to_clean:
        if dir_path.exists():
            print(f"Removing {dir_path}/")
            shutil.rmtree(dir_path)

    # Remove .pgc files (profile counters)
    for pgc_file in Path(".").glob("**/*.pgc"):
        print(f"Removing {pgc_file}")
        pgc_file.unlink()

    # Remove compiled modules
    for pyd_file in Path(".").glob("*.pyd"):
        print(f"Removing {pyd_file}")
        pyd_file.unlink()

    for so_file in Path(".").glob("*.so"):
        print(f"Removing {so_file}")
        so_file.unlink()

    print("\n✅ Clean complete")

def phase1_instrument():
    """Phase 1: Build with instrumentation"""
    print_phase("PHASE 1: INSTRUMENT",
                "Building with instrumentation to collect profiling data")

    # Create PGO directory
    PGO_DIR.mkdir(exist_ok=True)

    # Build with instrumentation flags
    setup_code = '''
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'ternary_core_simd_full',
        ['ternary_core_simd_full.cpp'],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            '.'
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
            '/PGD:pgo_data/ternary_core_simd_full.pgd',  # Profile database location
        ],
    ),
]

setup(
    name='ternary_core_simd_full',
    version='0.1.0',
    author='Ternary Core Team',
    description='AVX2-optimized ternary logic with PGO Phase 1 (Instrumentation)',
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires='>=3.7',
)
'''

    with open("setup_temp.py", "w") as f:
        f.write(setup_code)

    # Run build
    result = subprocess.run([sys.executable, "setup_temp.py", "build_ext", "--inplace"],
                          capture_output=False)

    os.remove("setup_temp.py")

    if result.returncode != 0:
        print("\n❌ Instrumentation build failed")
        sys.exit(1)

    print("\n✅ Phase 1 complete: Instrumented build ready")
    print(f"   Profile data will be written to: {PROFILE_DATA}")

def phase2_profile():
    """Phase 2: Run profiling workload"""
    print_phase("PHASE 2: PROFILE",
                "Running benchmarks to collect runtime profiling data")

    # Check if instrumented build exists
    pyd_file = list(Path(".").glob("ternary_core_simd_full*.pyd"))
    if not pyd_file:
        print("❌ No instrumented module found. Run 'python setup_pgo.py instrument' first.")
        sys.exit(1)

    print(f"Found instrumented module: {pyd_file[0]}")
    print("\nRunning profiling workload...")
    print("(This will take ~8 minutes - running full benchmark suite)\n")

    # Run the benchmark suite
    result = subprocess.run([sys.executable, "benchmarks/bench_phase0.py"],
                          capture_output=False)

    if result.returncode != 0:
        print("\n❌ Profiling workload failed")
        sys.exit(1)

    # Check if profile data was generated
    pgc_files = list(Path(".").glob("**/*.pgc"))
    if not pgc_files:
        print("\n⚠️  Warning: No .pgc files found")
        print("   Profile data may not have been collected properly")
    else:
        print(f"\n✅ Found {len(pgc_files)} profile counter files:")
        for pgc in pgc_files:
            print(f"   - {pgc}")

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
    pgc_files = list(Path(".").glob("**/*.pgc"))
    if not pgc_files and not PROFILE_DATA.exists():
        print("⚠️  Warning: No profile data found")
        print("   Run 'python setup_pgo.py profile' first for best results")
        print("   Continuing with optimization anyway...\n")

    # Build with profile-guided optimization
    setup_code = '''
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'ternary_core_simd_full',
        ['ternary_core_simd_full.cpp'],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            '.'
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
            '/PGD:pgo_data/ternary_core_simd_full.pgd',  # Profile database location
        ],
    ),
]

setup(
    name='ternary_core_simd_full',
    version='0.1.0',
    author='Ternary Core Team',
    description='AVX2-optimized ternary logic with PGO Phase 3 (Optimized)',
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires='>=3.7',
)
'''

    with open("setup_temp.py", "w") as f:
        f.write(setup_code)

    # Run build
    result = subprocess.run([sys.executable, "setup_temp.py", "build_ext", "--inplace"],
                          capture_output=False)

    os.remove("setup_temp.py")

    if result.returncode != 0:
        print("\n❌ Optimized build failed")
        sys.exit(1)

    print("\n✅ Phase 3 complete: Profile-guided optimized build ready")
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
    print(f"  PGO Directory:    {PGO_DIR}/ {'✅ exists' if PGO_DIR.exists() else '❌ not found'}")
    print(f"  Profile Database: {PROFILE_DATA} {'✅ exists' if PROFILE_DATA.exists() else '❌ not found'}")

    pgc_files = list(Path(".").glob("**/*.pgc"))
    if pgc_files:
        print(f"  Profile Counters: {len(pgc_files)} .pgc files found")

    pyd_files = list(Path(".").glob("ternary_core_simd_full*.pyd"))
    if pyd_files:
        print(f"  Compiled Module:  {pyd_files[0]} ✅")

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
