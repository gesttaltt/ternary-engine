#!/usr/bin/env python3
"""
build_gops_bench.py - Build the Gops/s comparative benchmark suite

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

USAGE:
    python build_gops_bench.py           # Build with auto-detected compiler
    python build_gops_bench.py --msvc    # Force MSVC
    python build_gops_bench.py --gcc     # Force GCC
    python build_gops_bench.py --clang   # Force Clang
    python build_gops_bench.py --run     # Build and run
"""

import subprocess
import sys
import os
import platform
import shutil
from pathlib import Path
from datetime import datetime

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = SCRIPT_DIR / "bin"
SOURCE_FILE = SCRIPT_DIR / "bench_gops_comparative.cpp"

def find_msvc():
    """Find MSVC compiler."""
    # Check for cl.exe in PATH
    cl_path = shutil.which("cl")
    if cl_path:
        return cl_path

    # Check common Visual Studio locations
    vs_paths = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC",
    ]

    for vs_path in vs_paths:
        if os.path.exists(vs_path):
            # Find the latest version
            versions = sorted(os.listdir(vs_path), reverse=True)
            if versions:
                cl = Path(vs_path) / versions[0] / "bin" / "Hostx64" / "x64" / "cl.exe"
                if cl.exists():
                    return str(cl)

    return None

def build_msvc():
    """Build with MSVC."""
    print("Building with MSVC...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_exe = OUTPUT_DIR / "bench_gops.exe"

    # MSVC command
    cmd = [
        "cl",
        "/O2",                    # Optimization
        "/arch:AVX2",             # AVX2 support
        "/std:c++17",             # C++17 standard
        "/EHsc",                  # Exception handling
        f"/I{SRC_DIR}",           # Include src directory
        f"/I{SCRIPT_DIR / 'include'}",  # Include local headers
        "/Fe:" + str(output_exe), # Output file
        str(SOURCE_FILE),
    ]

    print(f"Command: {' '.join(cmd)}")

    # Run from a Developer Command Prompt or use vcvarsall
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("MSVC compilation failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("\nTIP: Run this from a 'Developer Command Prompt for VS' or")
        print("     run vcvarsall.bat x64 first.")
        return None

    print(f"Build successful: {output_exe}")
    return output_exe

def build_gcc():
    """Build with GCC."""
    print("Building with GCC...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_exe = OUTPUT_DIR / "bench_gops"

    cmd = [
        "g++",
        "-O3",                    # Optimization
        "-march=native",          # Native CPU optimizations
        "-mavx2",                 # AVX2 support
        "-std=c++17",             # C++17 standard
        f"-I{SRC_DIR}",           # Include src directory
        f"-I{SCRIPT_DIR / 'include'}",  # Include local headers
        "-o", str(output_exe),
        str(SOURCE_FILE),
    ]

    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("GCC compilation failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return None

    print(f"Build successful: {output_exe}")
    return output_exe

def build_clang():
    """Build with Clang."""
    print("Building with Clang...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_exe = OUTPUT_DIR / "bench_gops"

    cmd = [
        "clang++",
        "-O3",                    # Optimization
        "-march=native",          # Native CPU optimizations
        "-mavx2",                 # AVX2 support
        "-std=c++17",             # C++17 standard
        f"-I{SRC_DIR}",           # Include src directory
        f"-I{SCRIPT_DIR / 'include'}",  # Include local headers
        "-o", str(output_exe),
        str(SOURCE_FILE),
    ]

    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Clang compilation failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return None

    print(f"Build successful: {output_exe}")
    return output_exe

def run_benchmark(exe_path, args=None):
    """Run the benchmark."""
    if args is None:
        args = []

    print(f"\nRunning benchmark: {exe_path}")
    print("=" * 80)

    cmd = [str(exe_path)] + args
    result = subprocess.run(cmd, capture_output=False)

    return result.returncode

def main():
    print("=" * 80)
    print("  Gops/s Comparative Benchmark - Build Script")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print()

    # Parse arguments
    force_msvc = "--msvc" in sys.argv
    force_gcc = "--gcc" in sys.argv
    force_clang = "--clang" in sys.argv
    run_after = "--run" in sys.argv
    quick_mode = "--quick" in sys.argv

    # Determine compiler
    exe_path = None

    if force_msvc or (platform.system() == "Windows" and not force_gcc and not force_clang):
        exe_path = build_msvc()
    elif force_gcc:
        exe_path = build_gcc()
    elif force_clang:
        exe_path = build_clang()
    else:
        # Try GCC first, then Clang
        if shutil.which("g++"):
            exe_path = build_gcc()
        elif shutil.which("clang++"):
            exe_path = build_clang()
        else:
            print("ERROR: No suitable compiler found!")
            print("Install GCC, Clang, or run from a Visual Studio Developer Prompt")
            return 1

    if exe_path is None:
        return 1

    # Run if requested
    if run_after:
        args = ["--quick"] if quick_mode else []
        return run_benchmark(exe_path, args)

    print("\nTo run the benchmark:")
    print(f"  {exe_path}")
    print(f"  {exe_path} --quick    # Quick test")
    print(f"  {exe_path} --csv      # CSV output")

    return 0

if __name__ == "__main__":
    sys.exit(main())
