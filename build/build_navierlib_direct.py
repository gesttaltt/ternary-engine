"""
build_navierlib_direct.py - Build NavierLib DLL directly with full MSVC paths

Builds navierlib.dll without requiring vcvarsall.bat environment setup
by using full paths to MSVC tools.
"""

import sys
import subprocess
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
NAVIERLIB_SRC = PROJECT_ROOT / "src" / "navierlib"
BUILD_DIR = PROJECT_ROOT / "build" / "navierlib_build"
OUTPUT_DIR = PROJECT_ROOT / "dist" / "NavierLib"

# MSVC Build Tools paths (VS 2022 Build Tools 18)
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

def build_dll():
    """Build NavierLib DLL with MSVC"""
    print("\n" + "="*80)
    print("  Building NavierLib.dll (Direct MSVC x64 Release)")
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
        "/GL",                   # Whole program optimization (LTCG)
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
        "/DLL",                  # Build DLL
        "/nologo",               # Suppress banner
        "/LTCG",                 # Link-time code generation
        "/OPT:REF",              # Eliminate unreferenced code
        "/OPT:ICF",              # Identical COMDAT folding
        "/MACHINE:X64",          # x64 target
        "/SUBSYSTEM:WINDOWS",    # Windows subsystem
        f"/OUT:{OUTPUT_DIR / 'navierlib.dll'}",
    ] + lib_dirs

    # Source files
    sources = [
        NAVIERLIB_SRC / "navierlib_impl.cpp",
        NAVIERLIB_SRC / "load_profiling.cpp"
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
    print("\nLinking DLL...")
    cmd = [str(LINK_EXE)] + link_flags + [str(obj) for obj in obj_files]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=BUILD_DIR)

    if result.returncode != 0:
        print("FAILED")
        print("\nLinker Output:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)

    print("✓ NavierLib.dll built successfully")

    # Copy header
    shutil.copy(NAVIERLIB_SRC / "navierlib_api.h", OUTPUT_DIR / "navierlib_api.h")
    print(f"✓ Output: {OUTPUT_DIR}")

    # File size check
    dll_path = OUTPUT_DIR / "navierlib.dll"
    if dll_path.exists():
        size_kb = dll_path.stat().st_size / 1024
        print(f"  navierlib.dll: {size_kb:.1f} KB")

def main():
    print("NavierLib Build System (Direct MSVC)")
    print("Target: x64 Release DLL with AVX2 + LTCG")

    clean_build_dir()
    build_dll()

    print("\n✓ Build complete!")
    print(f"   Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
