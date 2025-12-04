"""
build_cpp_test.py - Build C++ load profiling test suite

Builds LoadProfilingTest.exe with diagnostics for investigating
classification mismatches.

USAGE:
    python build/build_cpp_test.py
"""

import sys
import subprocess
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
SRC_DIR = PROJECT_ROOT / "src" / "navierlib"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks" / "cpp"
BUILD_DIR = PROJECT_ROOT / "build" / "cpp_test_build"
OUTPUT_DIR = PROJECT_ROOT / "dist" / "CppTest"

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

def build_test_exe():
    """Build C++ test executable"""
    print("\n" + "="*80)
    print("  Building LoadProfilingTest.exe (MSVC x64 Release)")
    print("="*80 + "\n")

    if not CL_EXE.exists():
        print(f"ERROR: cl.exe not found at {CL_EXE}")
        sys.exit(1)

    # Include directories
    include_dirs = [
        f"/I{PROJECT_ROOT / 'src'}",           # For ternary core headers
        f"/I{MSVC_INCLUDE}",                   # MSVC standard library
        f"/I{WIN_SDK_INCLUDE / 'ucrt'}",       # Universal CRT
        f"/I{WIN_SDK_INCLUDE / 'um'}",         # Windows SDK
        f"/I{WIN_SDK_INCLUDE / 'shared'}",     # Shared headers
    ]

    # Compiler flags
    compile_flags = [
        "/c",                    # Compile only
        "/nologo",               # Suppress banner
        "/O2",                   # Maximum optimization
        "/Oi",                   # Intrinsic functions
        "/arch:AVX2",            # Require AVX2
        "/fp:fast",              # Fast floating point
        "/GS-",                  # Disable security checks (performance)
        "/Gy",                   # Function-level linking
        "/MT",                   # Static runtime (no DLL dependencies)
        "/EHsc",                 # C++ exception handling
        "/std:c++17",            # C++17 standard
        "/DNAVIERLIB_EXPORTS",   # Export symbols
        "/DNDEBUG",              # Release mode
        "/W3",                   # Warning level 3
    ] + include_dirs

    # Library directories
    lib_dirs = [
        f"/LIBPATH:{MSVC_LIB}",
        f"/LIBPATH:{WIN_SDK_LIB / 'ucrt/x64'}",
        f"/LIBPATH:{WIN_SDK_LIB / 'um/x64'}",
    ]

    # Linker flags
    link_flags = [
        "/nologo",               # Suppress banner
        "/OPT:REF",              # Eliminate unreferenced code
        "/OPT:ICF",              # Identical COMDAT folding
        "/MACHINE:X64",          # x64 target
        "/SUBSYSTEM:CONSOLE",    # Console subsystem
        f"/OUT:{OUTPUT_DIR / 'LoadProfilingTest.exe'}",
    ] + lib_dirs

    # Source files
    sources = [
        BENCHMARKS_DIR / "load_profiling_test.cpp",
        SRC_DIR / "load_profiling.cpp",
        SRC_DIR / "load_profiling_optimized.cpp",
        SRC_DIR / "load_profiling_diagnostics.cpp",
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

    print("✓ LoadProfilingTest.exe built successfully")
    print(f"✓ Output: {OUTPUT_DIR}")

    # File size check
    exe_path = OUTPUT_DIR / "LoadProfilingTest.exe"
    if exe_path.exists():
        size_kb = exe_path.stat().st_size / 1024
        print(f"  LoadProfilingTest.exe: {size_kb:.1f} KB")

def main():
    print("C++ Load Profiling Test Suite Builder")
    print("Target: x64 Release with AVX2")

    clean_build_dir()
    build_test_exe()

    print("\n✓ Build complete!")
    print(f"   Output directory: {OUTPUT_DIR}")
    print(f"\n   Run: {OUTPUT_DIR / 'LoadProfilingTest.exe'}")

if __name__ == "__main__":
    main()
