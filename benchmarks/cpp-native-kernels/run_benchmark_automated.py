#!/usr/bin/env python3
"""
run_benchmark_automated.py - Automated C++ benchmark compilation and execution

This script automatically:
1. Finds Visual Studio installation
2. Activates MSVC environment
3. Compiles the benchmark
4. Runs the benchmark
5. Saves results

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import json

def print_header():
    print("=" * 80)
    print("  Automated C++ Native Benchmark Runner")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

def find_vcvarsall():
    """Find vcvarsall.bat in Visual Studio installation"""
    possible_paths = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvarsall.bat",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None

def setup_msvc_env():
    """Set up MSVC environment variables"""
    vcvarsall = find_vcvarsall()

    if not vcvarsall:
        print("ERROR: Could not find Visual Studio installation")
        print()
        print("Please install Visual Studio 2022 or 2019 with C++ support")
        return None

    print(f"Found Visual Studio: {vcvarsall}")
    print()

    # Run vcvarsall.bat and capture environment
    cmd = f'"{vcvarsall}" x64 && set'

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("ERROR: Failed to activate MSVC environment")
            print(result.stderr)
            return None

        # Parse environment variables
        env = {}
        for line in result.stdout.split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                env[key] = value

        return env

    except Exception as e:
        print(f"ERROR: {e}")
        return None

def compile_benchmark(env):
    """Compile the benchmark with MSVC"""
    script_dir = Path(__file__).parent
    src_dir = script_dir.parent.parent / "src"
    source = script_dir / "bench_kernels.cpp"
    bin_dir = script_dir / "bin"
    output = bin_dir / "bench_kernels.exe"

    # Create bin directory
    bin_dir.mkdir(exist_ok=True)

    # Compile command
    cmd = [
        "cl",
        "/O2",
        "/arch:AVX2",
        "/std:c++17",
        "/EHsc",
        f"/I{src_dir}",
        f"/Fe:{output}",
        str(source)
    ]

    print("Compiling benchmark...")
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd=str(script_dir)
        )

        if result.returncode != 0:
            print("ERROR: Compilation failed!")
            print(result.stdout)
            print(result.stderr)
            return None

        print("✓ Compilation successful!")
        print(f"  Output: {output}")
        print()

        return output

    except Exception as e:
        print(f"ERROR: {e}")
        return None

def run_benchmark(exe_path, output_csv=True, output_txt=True):
    """Run the compiled benchmark"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = Path(__file__).parent

    results = {}

    # Run CSV output
    if output_csv:
        csv_file = script_dir / f"results_{timestamp}.csv"
        print(f"Running benchmark (CSV output)...")

        try:
            result = subprocess.run(
                [str(exe_path), "--csv"],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                with open(csv_file, 'w') as f:
                    f.write(result.stdout)
                print(f"✓ CSV results saved: {csv_file}")
                results['csv'] = str(csv_file)
            else:
                print(f"ERROR: Benchmark failed (CSV)")
                print(result.stderr)

        except subprocess.TimeoutExpired:
            print("ERROR: Benchmark timed out (CSV)")
        except Exception as e:
            print(f"ERROR: {e}")

    # Run human-readable output
    if output_txt:
        txt_file = script_dir / f"results_{timestamp}.txt"
        print(f"Running benchmark (human-readable output)...")
        print()

        try:
            result = subprocess.run(
                [str(exe_path)],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                # Print to console
                print(result.stdout)

                # Save to file
                with open(txt_file, 'w') as f:
                    f.write(result.stdout)
                print(f"✓ Text results saved: {txt_file}")
                results['txt'] = str(txt_file)
            else:
                print(f"ERROR: Benchmark failed (text)")
                print(result.stderr)

        except subprocess.TimeoutExpired:
            print("ERROR: Benchmark timed out (text)")
        except Exception as e:
            print(f"ERROR: {e}")

    return results

def analyze_results(csv_file):
    """Analyze benchmark results"""
    try:
        import pandas as pd

        df = pd.read_csv(csv_file)

        print()
        print("=" * 80)
        print("  Results Analysis")
        print("=" * 80)
        print()

        # Best SIMD throughput
        best_simd = df['throughput_simd_ME_s'].max()
        best_op = df.loc[df['throughput_simd_ME_s'].idxmax(), 'operation']
        best_size = df.loc[df['throughput_simd_ME_s'].idxmax(), 'size']

        print(f"Best SIMD throughput: {best_simd:.2f} Mops/s ({best_simd/1000:.2f} Gops/s)")
        print(f"  Operation: {best_op}")
        print(f"  Size: {best_size:,} elements")
        print()

        # Average speedup
        avg_speedup = df['speedup'].mean()
        print(f"Average SIMD speedup: {avg_speedup:.2f}×")
        print()

        # Per-operation summary
        print("Per-operation peak throughput:")
        for op in df['operation'].unique():
            op_data = df[df['operation'] == op]
            peak = op_data['throughput_simd_ME_s'].max()
            print(f"  {op:4s}: {peak:.2f} Mops/s ({peak/1000:.2f} Gops/s)")
        print()

        # Compare with Python
        python_best = 19570  # Mops/s from Python benchmarks
        overhead_pct = (best_simd - python_best) / python_best * 100

        print("Comparison with Python benchmarks:")
        print(f"  C++ best:     {best_simd:.2f} Mops/s ({best_simd/1000:.2f} Gops/s)")
        print(f"  Python best:  {python_best} Mops/s (19.57 Gops/s)")
        print(f"  Python overhead: {overhead_pct:+.1f}%")

        if overhead_pct > 0:
            print(f"  ✓ C++ is {overhead_pct:.1f}% faster (Python binding overhead quantified)")
        else:
            print(f"  ⚠ Unexpected: C++ is slower by {abs(overhead_pct):.1f}%")

        print()

        return {
            'best_throughput_mops': best_simd,
            'best_throughput_gops': best_simd / 1000,
            'best_operation': best_op,
            'best_size': int(best_size),
            'avg_speedup': avg_speedup,
            'python_overhead_pct': overhead_pct
        }

    except ImportError:
        print("Install pandas for detailed analysis: pip install pandas")
        return None
    except Exception as e:
        print(f"Analysis error: {e}")
        return None

def save_summary(results_files, analysis):
    """Save summary JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = Path(__file__).parent
    summary_file = script_dir / f"summary_{timestamp}.json"

    summary = {
        'timestamp': datetime.now().isoformat(),
        'platform': 'Windows x64 AVX2',
        'compiler': 'MSVC',
        'results_files': results_files,
        'analysis': analysis
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Summary saved: {summary_file}")
    print()

def main():
    print_header()

    # Step 1: Find and activate MSVC environment
    print("Step 1: Setting up MSVC environment...")
    env = setup_msvc_env()

    if not env:
        print()
        print("=" * 80)
        print("FAILED: Could not set up MSVC environment")
        print("=" * 80)
        print()
        print("Please install Visual Studio 2022 or 2019 with C++ support")
        return 1

    print("✓ MSVC environment activated")
    print()

    # Step 2: Compile benchmark
    print("Step 2: Compiling benchmark...")
    exe_path = compile_benchmark(env)

    if not exe_path:
        print()
        print("=" * 80)
        print("FAILED: Compilation error")
        print("=" * 80)
        return 1

    # Step 3: Run benchmark
    print("Step 3: Running benchmark...")
    print()
    results_files = run_benchmark(exe_path, output_csv=True, output_txt=True)

    if not results_files:
        print()
        print("=" * 80)
        print("FAILED: Benchmark execution error")
        print("=" * 80)
        return 1

    # Step 4: Analyze results
    analysis = None
    if 'csv' in results_files:
        analysis = analyze_results(results_files['csv'])

    # Step 5: Save summary
    if analysis:
        save_summary(results_files, analysis)

    # Final summary
    print("=" * 80)
    print("  SUCCESS: Benchmark Complete")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Review results in:")
    for file_type, file_path in results_files.items():
        print(f"   - {file_path}")
    print()
    print("2. Update README.md with validated performance claims")
    print("3. Commit results to repository")
    print()

    return 0

if __name__ == "__main__":
    sys.exit(main())
