#!/usr/bin/env python3
"""
build_kernels.py - Build C++ native kernel benchmarks

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

USAGE:
    python build_kernels.py           # Build with auto-detected compiler
    python build_kernels.py --msvc    # Force MSVC
    python build_kernels.py --gcc     # Force GCC
    python build_kernels.py --clang   # Force Clang
    python build_kernels.py --run     # Build and run
"""

import subprocess
import sys
import os
import platform
import shutil
from pathlib import Path
from datetime import datetime

def print_header():
    print("=" * 80)
    print("  C++ Native Kernel Benchmarks - Build Script")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print()

def find_compiler():
    """Auto-detect available C++ compiler"""
    if platform.system() == "Windows":
        if shutil.which("cl"):
            return "msvc"
        elif shutil.which("g++"):
            return "gcc"
        elif shutil.which("clang++"):
            return "clang"
    else:
        if shutil.which("clang++"):
            return "clang"
        elif shutil.which("g++"):
            return "gcc"

    return None

def get_paths():
    """Get project paths"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    src_dir = project_root / "src"
    bin_dir = script_dir / "bin"

    return {
        "script_dir": script_dir,
        "project_root": project_root,
        "src_dir": src_dir,
        "bin_dir": bin_dir,
        "source": script_dir / "bench_kernels.cpp",
        "output": bin_dir / ("bench_kernels.exe" if platform.system() == "Windows" else "bench_kernels")
    }

def build_msvc():
    """Build with MSVC"""
    paths = get_paths()
    paths["bin_dir"].mkdir(exist_ok=True)

    cmd = [
        "cl",
        "/O2",
        "/arch:AVX2",
        "/std:c++17",
        "/EHsc",
        f"/I{paths['src_dir']}",
        f"/Fe:{paths['output']}",
        str(paths['source'])
    ]

    print("Building with MSVC...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("\nERROR: Compilation failed!")
        print(result.stderr)
        return None

    print("\n✓ Build successful!")
    print(f"  Output: {paths['output']}")
    return paths['output']

def build_gcc():
    """Build with GCC"""
    paths = get_paths()
    paths["bin_dir"].mkdir(exist_ok=True)

    cmd = [
        "g++",
        "-O3",
        "-march=native",
        "-mavx2",
        "-std=c++17",
        f"-I{paths['src_dir']}",
        "-o", str(paths['output']),
        str(paths['source'])
    ]

    print("Building with GCC...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("\nERROR: Compilation failed!")
        print(result.stderr)
        return None

    print("\n✓ Build successful!")
    print(f"  Output: {paths['output']}")
    return paths['output']

def build_clang():
    """Build with Clang"""
    paths = get_paths()
    paths["bin_dir"].mkdir(exist_ok=True)

    cmd = [
        "clang++",
        "-O3",
        "-march=native",
        "-mavx2",
        "-std=c++17",
        f"-I{paths['src_dir']}",
        "-o", str(paths['output']),
        str(paths['source'])
    ]

    print("Building with Clang...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("\nERROR: Compilation failed!")
        print(result.stderr)
        return None

    print("\n✓ Build successful!")
    print(f"  Output: {paths['output']}")
    return paths['output']

def run_benchmark(exe_path, args=None):
    """Run the compiled benchmark"""
    cmd = [str(exe_path)]
    if args:
        cmd.extend(args)

    print(f"\nRunning benchmark: {' '.join(cmd)}")
    print("=" * 80)
    print()

    result = subprocess.run(cmd)
    return result.returncode

def main():
    print_header()

    # Parse arguments
    force_compiler = None
    should_run = False
    run_args = []

    for arg in sys.argv[1:]:
        if arg == "--msvc":
            force_compiler = "msvc"
        elif arg == "--gcc":
            force_compiler = "gcc"
        elif arg == "--clang":
            force_compiler = "clang"
        elif arg == "--run":
            should_run = True
        else:
            run_args.append(arg)

    # Detect or use forced compiler
    compiler = force_compiler or find_compiler()

    if not compiler:
        print("ERROR: No C++ compiler found!")
        print()
        print("Please install one of the following:")
        print("  - MSVC (Visual Studio 2022)")
        print("  - GCC (via MinGW or WSL)")
        print("  - Clang")
        return 1

    print(f"Using compiler: {compiler.upper()}")
    print()

    # Build
    if compiler == "msvc":
        exe_path = build_msvc()
    elif compiler == "gcc":
        exe_path = build_gcc()
    elif compiler == "clang":
        exe_path = build_clang()
    else:
        print(f"ERROR: Unknown compiler: {compiler}")
        return 1

    if not exe_path:
        return 1

    # Run if requested
    if should_run:
        return run_benchmark(exe_path, run_args)

    print()
    print("=" * 80)
    print("  Usage")
    print("=" * 80)
    print(f"  {exe_path}              # Run all benchmarks")
    print(f"  {exe_path} --csv        # CSV output")
    print()

    return 0

if __name__ == "__main__":
    sys.exit(main())
