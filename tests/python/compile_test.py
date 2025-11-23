#!/usr/bin/env python3
"""
Simple script to compile test_simd_correctness.cpp using the same compiler
that Python extensions use.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent
    test_source = project_root / "tests" / "test_simd_correctness.cpp"
    output_exe = project_root / "tests" / "test_simd_correctness.exe"

    print("=" * 70)
    print("  Compiling SIMD Correctness Tests")
    print("=" * 70)
    print(f"Source: {test_source}")
    print(f"Output: {output_exe}")
    print()

    # Try to use MSVC via vcvarsall if available
    # This is what setuptools does
    import distutils.ccompiler
    import distutils.msvccompiler

    try:
        # Get the MSVC compiler
        compiler = distutils.ccompiler.new_compiler(compiler='msvc')
        compiler.initialize()

        # Compile with AVX2 support
        compile_args = [
            '/EHsc',  # Exception handling
            '/std:c++17',  # C++17 standard
            '/O2',  # Optimization
            '/arch:AVX2',  # AVX2 support
            f'/I{project_root}',  # Include path
        ]

        link_args = []

        print("Compiler: MSVC (via Python distutils)")
        print(f"Compile args: {' '.join(compile_args)}")
        print()

        # Compile
        obj_file = compiler.compile(
            [str(test_source)],
            extra_preargs=compile_args
        )

        print(f"Compiled: {obj_file}")

        # Link
        compiler.link_executable(
            obj_file,
            str(output_exe).replace('.exe', ''),
            extra_preargs=link_args
        )

        print(f"Linked: {output_exe}")
        print()
        print("=" * 70)
        print("  ✓ Compilation Successful!")
        print("=" * 70)
        print()
        print(f"Run with: {output_exe}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        print()
        print("Trying fallback method...")

        # Fallback: try direct compiler invocation
        # Try to find cl.exe
        import shutil
        cl_exe = shutil.which('cl')

        if cl_exe:
            cmd = [
                cl_exe,
                '/EHsc',
                '/std:c++17',
                '/O2',
                '/arch:AVX2',
                f'/I{project_root}',
                str(test_source),
                f'/Fe:{output_exe}',
                '/nologo'
            ]

            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print("✓ Compilation successful!")
                return 0
            else:
                print("✗ Compilation failed:")
                print(result.stderr)
                return 1
        else:
            print("✗ Could not find MSVC compiler")
            print("Please install Visual Studio Build Tools")
            return 1

if __name__ == '__main__':
    sys.exit(main())
