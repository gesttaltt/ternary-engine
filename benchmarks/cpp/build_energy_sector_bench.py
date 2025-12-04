"""
build_energy_sector_bench.py - Build eBase/Eneva workload benchmark

Compiles EnergySectorWorkload.exe to test real-world energy utility workloads:
- End-to-end workflow (classify + aggregate)
- Realistic consumption patterns (residential/commercial/industrial)
- Scale testing (hourly, daily, monthly, annual batches)
- Determinism validation (billing compliance)
- Memory efficiency analysis

USAGE:
    python benchmarks/cpp/build_energy_sector_bench.py
"""

import sys
import subprocess
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"  / "navierlib"
BENCHMARK_FILE = PROJECT_ROOT / "benchmarks" / "cpp" / "bench_energy_sector_workload.cpp"
BUILD_DIR = PROJECT_ROOT / "build" / "energy_sector_bench_build"
OUTPUT_DIR = PROJECT_ROOT / "dist" / "EnergySectorBench"

# MSVC Build Tools paths
MSVC_ROOT = Path("C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools")
MSVC_BIN = MSVC_ROOT / "VC/Tools/MSVC/14.50.35717/bin/HostX86/x64"
MSVC_INCLUDE = MSVC_ROOT / "VC/Tools/MSVC/14.50.35717/include"
MSVC_LIB = MSVC_ROOT / "VC/Tools/MSVC/14.50.35717/lib/x64"

WIN_SDK_ROOT = Path("C:/Program Files (x86)/Windows Kits/10")
WIN_SDK_INCLUDE = WIN_SDK_ROOT / "include/10.0.26100.0"
WIN_SDK_LIB = WIN_SDK_ROOT / "Lib/10.0.26100.0"

CL_EXE = MSVC_BIN / "cl.exe"
LINK_EXE = MSVC_BIN / "link.exe"

def clean_build_dir():
    """Clean previous build artifacts"""
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def build_benchmark():
    """Build energy sector workload benchmark executable"""
    print("\n" + "="*80)
    print("  Building EnergySectorWorkload.exe (MSVC x64 Release)")
    print("="*80 + "\n")

    if not CL_EXE.exists():
        print(f"ERROR: cl.exe not found at {CL_EXE}")
        sys.exit(1)

    # Include directories
    include_dirs = [
        f"/I{PROJECT_ROOT / 'src'}",
        f"/I{MSVC_INCLUDE}",
        f"/I{WIN_SDK_INCLUDE / 'ucrt'}",
        f"/I{WIN_SDK_INCLUDE / 'um'}",
        f"/I{WIN_SDK_INCLUDE / 'shared'}",
    ]

    # Compiler flags - aggressive optimization
    compile_flags = [
        "/c",
        "/nologo",
        "/O2",                   # Maximum optimization
        "/Oi",                   # Intrinsic functions
        "/Ot",                   # Favor speed over size
        "/arch:AVX2",            # AVX2 required
        "/fp:fast",              # Fast FP
        "/GS-",                  # No security checks
        "/Gy",                   # Function-level linking
        "/MT",                   # Static runtime
        "/EHsc",
        "/std:c++17",
        "/DNAVIERLIB_EXPORTS",
        "/DNDEBUG",
        "/W3",
    ] + include_dirs

    # Library directories
    lib_dirs = [
        f"/LIBPATH:{MSVC_LIB}",
        f"/LIBPATH:{WIN_SDK_LIB / 'ucrt/x64'}",
        f"/LIBPATH:{WIN_SDK_LIB / 'um/x64'}",
    ]

    # Linker flags
    link_flags = [
        "/nologo",
        "/OPT:REF",
        "/OPT:ICF",
        "/MACHINE:X64",
        "/SUBSYSTEM:CONSOLE",
        f"/OUT:{OUTPUT_DIR / 'EnergySectorWorkload.exe'}",
    ] + lib_dirs

    # Source files
    sources = [
        BENCHMARK_FILE,
        SRC_DIR / "load_profiling.cpp",
        SRC_DIR / "load_profiling_optimized.cpp",
    ]

    # Compile step
    print("Compiling sources...")
    obj_files = []
    for src in sources:
        obj_file = BUILD_DIR / (src.stem + ".obj")
        obj_files.append(obj_file)

        cmd = [str(CL_EXE)] + compile_flags + [
            f"/Fo{obj_file}",
            str(src)
        ]

        print(f"  {src.name}...", end=" ", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=BUILD_DIR)

        if result.returncode != 0:
            print("FAILED")
            print("\nCompiler Output:")
            print(result.stdout)
            print(result.stderr)
            sys.exit(1)
        print("OK")

    # Link step
    print("\nLinking executable...")
    cmd = [str(LINK_EXE)] + link_flags + [str(obj) for obj in obj_files]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=BUILD_DIR)

    if result.returncode != 0:
        print("FAILED")
        print("\nLinker Output:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)

    print("✓ EnergySectorWorkload.exe built successfully")
    print(f"✓ Output: {OUTPUT_DIR}")

    exe_path = OUTPUT_DIR / "EnergySectorWorkload.exe"
    if exe_path.exists():
        size_kb = exe_path.stat().st_size / 1024
        print(f"  EnergySectorWorkload.exe: {size_kb:.1f} KB")

def main():
    print("Energy Sector Workload Benchmark Builder")
    print("Target: eBase/Eneva production workloads")
    print("Tests: Realistic patterns, scale, determinism, memory efficiency")

    clean_build_dir()
    build_benchmark()

    print("\n✓ Build complete!")
    print(f"   Output directory: {OUTPUT_DIR}")
    print(f"\n   Run: {OUTPUT_DIR / 'EnergySectorWorkload.exe'}")
    print("\n   TESTS:")
    print("   1. Realistic consumption patterns (residential/commercial/industrial)")
    print("   2. Scale testing (400K, 1M, 10M, 100M intervals)")
    print("   3. Determinism (1000 runs, billing compliance)")
    print("   4. Memory efficiency (compression ratio)")

if __name__ == "__main__":
    main()
