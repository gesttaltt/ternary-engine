"""
build_test_packing.py - Build and run packing tests

Simple script to compile test_packing.cpp with MSVC
"""

import subprocess
import sys
from pathlib import Path

def find_msvc():
    """Find MSVC compiler"""
    possible_paths = [
        Path("C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvarsall.bat"),
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Auxiliary/Build/vcvarsall.bat"),
    ]

    for path in possible_paths:
        if path.exists():
            return path

    return None

def main():
    project_root = Path(__file__).parent.parent
    test_file = project_root / "tests" / "cpp" / "test_packing.cpp"
    output_exe = project_root / "test_packing.exe"

    # Check if test file exists
    if not test_file.exists():
        print(f"Error: {test_file} not found")
        return 1

    # Find MSVC
    vcvarsall = find_msvc()
    if not vcvarsall:
        print("Error: MSVC not found")
        return 1

    print(f"Using MSVC from: {vcvarsall}")

    # Build command (quote every interpolated path — repo paths may contain spaces)
    cmd = (
        f'"{vcvarsall}" x64 && cl.exe /std:c++17 /O2 /EHsc '
        f'"/Fe:{output_exe}" "{test_file}" "/I{project_root}"'
    )

    print(f"Building {test_file}...")
    result = subprocess.run(cmd, shell=True, cwd=project_root)

    if result.returncode != 0:
        print("Build failed")
        return 1

    print("\nRunning tests...")
    result = subprocess.run([str(output_exe)], cwd=project_root)

    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
