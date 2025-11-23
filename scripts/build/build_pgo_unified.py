"""
build_pgo_unified.py - Unified PGO build system (Clang-first, MSVC fallback)

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Unified Profile-Guided Optimization build that:
- Prefers Clang PGO (works with Python extensions)
- Falls back to MSVC PGO (with known limitations)
- Integrates with benchmarking suite automatically

USAGE:
    python scripts/build/build_pgo_unified.py          # Auto-detect compiler
    python scripts/build/build_pgo_unified.py --clang  # Force Clang
    python scripts/build/build_pgo_unified.py --msvc   # Force MSVC

CLANG PGO WORKFLOW (recommended):
    Phase 1: Build with -fprofile-generate
    Phase 2: Run benchmarks (writes .profraw immediately)
    Phase 3: Merge profiles with llvm-profdata
    Phase 4: Build with -fprofile-use

MSVC PGO WORKFLOW (fallback, has limitations):
    Phase 1: Build with /LTCG:PGI
    Phase 2: Run benchmarks (may not write .pgc files)
    Phase 3: Build with /LTCG:PGO (if .pgc files exist)

See docs/pgo/README.md for MSVC limitations with Python extensions.
"""

import sys
import os
import shutil
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# Get project root (script is in scripts/build/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
BUILD_ARTIFACTS = PROJECT_ROOT / "build" / "artifacts"
PGO_DATA_DIR = PROJECT_ROOT / "pgo_data"

# Benchmarking integration
BENCHMARK_SCRIPT = PROJECT_ROOT / "benchmarks" / "bench_phase0.py"

def print_header(message):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {message}")
    print("="*80 + "\n")

def detect_clang():
    """Detect Clang availability"""
    try:
        # Try clang-cl (Windows)
        result = subprocess.run(
            ["clang-cl", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return "clang-cl", result.stdout.split('\n')[0]
    except:
        pass

    try:
        # Try clang (Linux/macOS)
        result = subprocess.run(
            ["clang", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return "clang", result.stdout.split('\n')[0]
    except:
        pass

    return None, None

def detect_llvm_profdata():
    """Detect llvm-profdata tool"""
    try:
        result = subprocess.run(
            ["llvm-profdata", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return True
    except:
        pass
    return False

def detect_msvc():
    """Detect MSVC availability"""
    # Check if build.py can find MSVC (it has environment setup)
    # MSVC requires specific environment variables, easiest to check via build.py
    try:
        # Check for common MSVC paths
        import platform
        if platform.system() == "Windows":
            # On Windows, if setuptools can find MSVC, we have it
            # Try importing distutils which sets up MSVC
            try:
                from distutils import msvc9compiler
                return True
            except:
                pass

            # Alternative: Check VS installation paths
            vs_paths = [
                r"C:\Program Files (x86)\Microsoft Visual Studio",
                r"C:\Program Files\Microsoft Visual Studio"
            ]
            for base in vs_paths:
                if Path(base).exists():
                    # Found VS installation
                    return True
        return False
    except:
        return False

def clean_pgo_data():
    """Clean PGO data directory"""
    if PGO_DATA_DIR.exists():
        print(f"Cleaning PGO data directory: {PGO_DATA_DIR}")
        shutil.rmtree(PGO_DATA_DIR)
    PGO_DATA_DIR.mkdir(parents=True, exist_ok=True)

def build_clang_pgo_phase1(clang_cmd):
    """Phase 1: Build with Clang profile generation"""
    print_header("CLANG PGO - PHASE 1: Instrumentation Build")

    # Set environment for Clang PGO
    env = os.environ.copy()

    # Clang flags for profile generation
    profile_dir = str(PGO_DATA_DIR / "profiles")
    os.makedirs(profile_dir, exist_ok=True)

    if "clang-cl" in clang_cmd:
        # Windows: clang-cl (MSVC-compatible interface)
        env['CPPFLAGS'] = f"-fprofile-generate={profile_dir}"
        env['LDFLAGS'] = f"-fprofile-generate={profile_dir}"
    else:
        # Linux/macOS: clang
        env['CPPFLAGS'] = f"-fprofile-generate={profile_dir}"
        env['LDFLAGS'] = f"-fprofile-generate={profile_dir}"

    # Build with setuptools
    print(f"Building instrumented module with {clang_cmd}...")
    print(f"Profile directory: {profile_dir}")

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "build" / "build.py")],
        cwd=str(PROJECT_ROOT),
        env=env
    )

    if result.returncode != 0:
        print("\n❌ Phase 1 failed: Instrumentation build error")
        return False

    print("\n✅ Phase 1 complete: Instrumented module built")
    return True

def run_profiling_benchmarks():
    """Phase 2: Run benchmarks to collect profile data"""
    print_header("PHASE 2: Profile Data Collection")

    print("Running benchmark suite to collect profile data...")
    print(f"Benchmark script: {BENCHMARK_SCRIPT}")

    result = subprocess.run(
        [sys.executable, str(BENCHMARK_SCRIPT), "--quick"],
        cwd=str(PROJECT_ROOT)
    )

    if result.returncode != 0:
        print("\n⚠️  Benchmarks failed, but may have collected some data")
        return False

    print("\n✅ Phase 2 complete: Benchmarks executed")
    return True

def merge_clang_profiles():
    """Phase 3: Merge Clang profile data"""
    print_header("CLANG PGO - PHASE 3: Profile Merging")

    profile_dir = PGO_DATA_DIR / "profiles"
    profraw_files = list(profile_dir.glob("**/*.profraw"))

    if not profraw_files:
        print(f"❌ No .profraw files found in {profile_dir}")
        print("Profile data was not generated during benchmarking")
        return False

    print(f"Found {len(profraw_files)} .profraw files:")
    for f in profraw_files[:5]:  # Show first 5
        print(f"  - {f.name}")
    if len(profraw_files) > 5:
        print(f"  ... and {len(profraw_files) - 5} more")

    # Merge profiles
    merged_profile = PGO_DATA_DIR / "merged.profdata"

    cmd = ["llvm-profdata", "merge"]
    cmd.extend([str(f) for f in profraw_files])
    cmd.extend(["-output", str(merged_profile)])

    print(f"\nMerging profiles to: {merged_profile}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\n❌ Profile merging failed")
        return False

    print(f"\n✅ Phase 3 complete: Profiles merged ({merged_profile.stat().st_size} bytes)")
    return True

def build_clang_pgo_phase4(clang_cmd):
    """Phase 4: Build optimized with Clang profile data"""
    print_header("CLANG PGO - PHASE 4: Optimized Build")

    merged_profile = PGO_DATA_DIR / "merged.profdata"

    if not merged_profile.exists():
        print(f"❌ Merged profile not found: {merged_profile}")
        return False

    # Set environment for optimized build
    env = os.environ.copy()

    if "clang-cl" in clang_cmd:
        # Windows: clang-cl
        env['CPPFLAGS'] = f"-fprofile-use={merged_profile}"
        env['LDFLAGS'] = f"-fprofile-use={merged_profile}"
    else:
        # Linux/macOS: clang
        env['CPPFLAGS'] = f"-fprofile-use={merged_profile}"
        env['LDFLAGS'] = f"-fprofile-use={merged_profile}"

    print(f"Building optimized module with {clang_cmd}...")
    print(f"Using profile: {merged_profile}")

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "build" / "build.py")],
        cwd=str(PROJECT_ROOT),
        env=env
    )

    if result.returncode != 0:
        print("\n❌ Phase 4 failed: Optimized build error")
        return False

    print("\n✅ Phase 4 complete: PGO-optimized module built")
    return True

def build_msvc_pgo():
    """Fallback: MSVC PGO (with known limitations)"""
    print_header("MSVC PGO BUILD (FALLBACK)")
    print("⚠️  WARNING: MSVC PGO has known limitations with Python extensions")
    print("See docs/pgo/PGO_LIMITATIONS.md for details\n")

    # Use existing MSVC PGO script
    msvc_script = PROJECT_ROOT / "scripts" / "build" / "build_pgo.py"

    result = subprocess.run(
        [sys.executable, str(msvc_script), "full"],
        cwd=str(PROJECT_ROOT)
    )

    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(
        description='Unified PGO build system (Clang-first, MSVC fallback)'
    )
    parser.add_argument('--clang', action='store_true',
                       help='Force Clang PGO (fail if not available)')
    parser.add_argument('--msvc', action='store_true',
                       help='Force MSVC PGO (fallback mode)')
    parser.add_argument('--clean', action='store_true',
                       help='Clean PGO data before starting')

    args = parser.parse_args()

    print_header("UNIFIED PGO BUILD SYSTEM")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Clean if requested
    if args.clean:
        clean_pgo_data()

    # Detect compilers
    clang_cmd, clang_version = detect_clang()
    has_llvm_profdata = detect_llvm_profdata()
    has_msvc = detect_msvc()

    print("\nCompiler Detection:")
    if clang_cmd:
        print(f"  Clang:         ✅ {clang_version}")
        print(f"  llvm-profdata: {'✅' if has_llvm_profdata else '❌ Not found'}")
    else:
        print("  Clang:         ❌ Not found")

    if has_msvc:
        print("  MSVC:          ✅ Available")
    else:
        print("  MSVC:          ❌ Not found")

    # Determine build strategy
    use_clang = False
    use_msvc = False

    if args.clang:
        # Force Clang
        if not clang_cmd or not has_llvm_profdata:
            print("\n❌ ERROR: Clang or llvm-profdata not found")
            print("\nTo install Clang on Windows:")
            print("  1. Download LLVM from https://releases.llvm.org/")
            print("  2. Install with 'Add LLVM to PATH' option")
            print("  3. Restart terminal")
            return 1
        use_clang = True
    elif args.msvc:
        # Force MSVC
        if not has_msvc:
            print("\n❌ ERROR: MSVC not found")
            return 1
        use_msvc = True
    else:
        # Auto-detect (prefer Clang)
        if clang_cmd and has_llvm_profdata:
            print("\n✅ Using Clang PGO (recommended)")
            use_clang = True
        elif has_msvc:
            print("\n⚠️  Using MSVC PGO (fallback - has known limitations)")
            use_msvc = True
        else:
            print("\n❌ ERROR: No compatible compiler found")
            return 1

    # Execute PGO workflow
    if use_clang:
        print_header("CLANG PGO WORKFLOW")

        # Phase 1: Instrumentation build
        if not build_clang_pgo_phase1(clang_cmd):
            return 1

        # Phase 2: Run benchmarks
        if not run_profiling_benchmarks():
            print("⚠️  Continuing despite benchmark errors...")

        # Phase 3: Merge profiles
        if not merge_clang_profiles():
            return 1

        # Phase 4: Optimized build
        if not build_clang_pgo_phase4(clang_cmd):
            return 1

        print_header("CLANG PGO BUILD COMPLETE")
        print("✅ PGO-optimized module built successfully")
        print(f"\nProfile data saved in: {PGO_DATA_DIR}")

    elif use_msvc:
        if not build_msvc_pgo():
            return 1

    print("\n" + "="*80)
    print("Next steps:")
    print("  1. Run benchmarks to validate performance gain:")
    print("     python benchmarks/bench_phase0.py")
    print("\n  2. Compare with standard build")
    print("  3. Review profile data (if using Clang)")
    print("="*80 + "\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
