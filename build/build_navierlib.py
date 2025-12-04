"""
build_navierlib.py - Build NavierLib DLL for commercial distribution

Builds navierlib.dll with maximum optimization:
- x64 Release
- AVX2 required
- Link-Time Code Generation (LTCG)
- Static runtime (no dependencies)
- Profile-Guided Optimization (PGO) optional

USAGE:
    python build/build_navierlib.py         # Standard build
    python build/build_navierlib.py --pgo   # With PGO (requires 2-phase build)
"""

import sys
import subprocess
import shutil
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
NAVIERLIB_SRC = PROJECT_ROOT / "src" / "navierlib"
BUILD_DIR = PROJECT_ROOT / "build" / "navierlib_build"
OUTPUT_DIR = PROJECT_ROOT / "dist" / "NavierLib"

def clean_build_dir():
    """Clean previous build artifacts"""
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def detect_msvc():
    """Detect MSVC compiler"""
    try:
        result = subprocess.run(
            ["cl.exe"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "Microsoft (R) C/C++ Optimizing Compiler" in result.stderr:
            return True
    except:
        pass
    return False

def build_dll_msvc(use_pgo=False):
    """Build NavierLib DLL with MSVC"""
    print("\n" + "="*80)
    print("  Building NavierLib.dll (MSVC x64 Release)")
    print("="*80 + "\n")

    if not detect_msvc():
        print("ERROR: MSVC compiler not found. Please run from Visual Studio Developer Command Prompt.")
        sys.exit(1)

    # Compiler flags
    compile_flags = [
        "/c",                    # Compile only
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
    ]

    if use_pgo:
        compile_flags.append("/GL")  # Required for PGO

    # Linker flags
    link_flags = [
        "/DLL",                  # Build DLL
        "/LTCG",                 # Link-time code generation
        "/OPT:REF",              # Eliminate unreferenced code
        "/OPT:ICF",              # Identical COMDAT folding
        "/MACHINE:X64",          # x64 target
        "/SUBSYSTEM:WINDOWS",    # Windows subsystem
        f"/OUT:{OUTPUT_DIR / 'navierlib.dll'}",
    ]

    # Source files
    sources = [
        NAVIERLIB_SRC / "navierlib_impl.cpp"
    ]

    # Compile step
    print("Compiling sources...")
    for src in sources:
        obj_file = BUILD_DIR / (src.stem + ".obj")
        cmd = ["cl.exe"] + compile_flags + [
            f"/Fo{obj_file}",
            str(src)
        ]

        print(f"  {src.name}...", end=" ")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("FAILED")
            print(result.stderr)
            sys.exit(1)
        print("OK")

    # Link step
    print("\nLinking DLL...")
    obj_files = list(BUILD_DIR.glob("*.obj"))
    cmd = ["link.exe"] + link_flags + [str(obj) for obj in obj_files]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=BUILD_DIR)

    if result.returncode != 0:
        print("FAILED")
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
    parser = argparse.ArgumentParser(description="Build NavierLib DLL")
    parser.add_argument("--pgo", action="store_true", help="Enable Profile-Guided Optimization")
    args = parser.parse_args()

    print("NavierLib Build System")
    print("Target: x64 Release DLL with AVX2 + LTCG")

    clean_build_dir()
    build_dll_msvc(use_pgo=args.pgo)

    print("\n✓ Build complete!")
    print(f"   Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
