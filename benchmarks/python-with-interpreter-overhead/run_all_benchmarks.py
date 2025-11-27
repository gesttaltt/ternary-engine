"""
run_all_benchmarks.py - Master benchmark orchestrator for Ternary Engine

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Orchestrates complete benchmarking workflow:
1. Build standard optimized version
2. Run benchmarks on standard build
3. Build PGO version (optional)
4. Run benchmarks on PGO build (optional)
5. Compare results
6. Generate comprehensive report

Usage:
    python benchmarks/run_all_benchmarks.py                      # Standard build only
    python benchmarks/run_all_benchmarks.py --with-pgo           # Include PGO build
    python benchmarks/run_all_benchmarks.py --quick              # Quick benchmarks
    python benchmarks/run_all_benchmarks.py --clean              # Clean before building
"""

import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import shutil


PROJECT_ROOT = Path(__file__).parent.parent
BUILD_SCRIPT = PROJECT_ROOT / "build" / "build.py"
BUILD_PGO_SCRIPT = PROJECT_ROOT / "build" / "build_pgo_unified.py"
BENCH_SCRIPT = PROJECT_ROOT / "benchmarks" / "bench_simd_core_ops.py"
COMPARE_SCRIPT = PROJECT_ROOT / "benchmarks" / "bench_regression_detect.py"
CLEAN_SCRIPT = PROJECT_ROOT / "build" / "clean_all.py"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"


def print_step(step_num: int, total: int, description: str):
    """Print step header"""
    print("\n" + "=" * 80)
    print(f"  STEP {step_num}/{total}: {description}")
    print("=" * 80 + "\n")


def run_command(cmd: list, description: str) -> bool:
    """Run a command and check for errors"""
    print(f"Running: {' '.join(str(x) for x in cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed with exit code {result.returncode}")
        return False

    print(f"\n✅ {description} completed successfully")
    return True


def clean_builds():
    """Clean all build artifacts using comprehensive cleanup utility"""
    print("Cleaning build artifacts...")

    # Use the comprehensive cleanup utility
    cmd = [sys.executable, str(CLEAN_SCRIPT)]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        print("⚠️  Cleanup had some issues, but continuing...")
    else:
        print("✅ Cleanup complete")


def build_standard() -> bool:
    """Build standard optimized version"""
    return run_command([sys.executable, str(BUILD_SCRIPT)], "Standard build")


def build_pgo() -> bool:
    """Build PGO optimized version"""
    return run_command([sys.executable, str(BUILD_PGO_SCRIPT), "full"], "PGO build")


def run_benchmarks(build_name: str, quick: bool = False) -> Path:
    """Run benchmarks and return results path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / build_name

    cmd = [sys.executable, str(BENCH_SCRIPT), f"--output={output_dir}"]
    if quick:
        cmd.append("--quick")

    if not run_command(cmd, f"Benchmarks for {build_name}"):
        return None

    # Find the most recent results file
    results_files = sorted(output_dir.glob("bench_results_*.json"))
    if results_files:
        return results_files[-1]

    return None


def compare_results(before_path: Path, after_path: Path, report_name: str):
    """Compare two benchmark results"""
    output_path = RESULTS_DIR / f"comparison_{report_name}.json"

    cmd = [sys.executable, str(COMPARE_SCRIPT),
           str(before_path), str(after_path),
           f"--output={output_path}"]

    run_command(cmd, f"Comparison: {report_name}")


def print_final_summary(standard_results: Path, pgo_results: Path = None):
    """Print final summary"""
    print("\n" + "=" * 80)
    print("  BENCHMARKING COMPLETE")
    print("=" * 80)

    print("\nResults:")
    print(f"  Standard build: {standard_results}")
    if pgo_results:
        print(f"  PGO build:      {pgo_results}")

    print(f"\nAll results saved in: {RESULTS_DIR}")

    print("\nNext steps:")
    print("  1. Review JSON results for detailed metrics")
    print("  2. Import CSV into spreadsheet for visualization")
    if pgo_results:
        print("  3. Check comparison report for PGO impact")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Run complete benchmarking workflow for Ternary Engine'
    )
    parser.add_argument('--with-pgo', action='store_true',
                       help='Include PGO build and comparison')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmarks (fewer test sizes)')
    parser.add_argument('--clean', action='store_true',
                       help='Clean build artifacts before starting')
    parser.add_argument('--skip-build', action='store_true',
                       help='Skip building, only run benchmarks')

    args = parser.parse_args()

    # Calculate total steps
    total_steps = 2  # Standard build + benchmark
    if args.with_pgo:
        total_steps += 2  # PGO build + benchmark + comparison
    if args.clean:
        total_steps += 1
    if args.with_pgo:
        total_steps += 1  # Comparison step

    current_step = 0

    print("\n" + "=" * 80)
    print("  TERNARY ENGINE - MASTER BENCHMARK SUITE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Standard build: Yes")
    print(f"  PGO build:      {'Yes' if args.with_pgo else 'No'}")
    print(f"  Quick mode:     {'Yes' if args.quick else 'No'}")
    print(f"  Clean first:    {'Yes' if args.clean else 'No'}")
    print(f"  Skip builds:    {'Yes' if args.skip_build else 'No'}")
    print(f"\nTotal steps: {total_steps}")

    # Step: Clean (optional)
    if args.clean:
        current_step += 1
        print_step(current_step, total_steps, "Clean build artifacts")
        clean_builds()

    # Step: Build standard
    if not args.skip_build:
        current_step += 1
        print_step(current_step, total_steps, "Build standard optimized version")
        if not build_standard():
            print("\n❌ Build failed. Aborting.")
            sys.exit(1)

    # Step: Benchmark standard
    current_step += 1
    print_step(current_step, total_steps, "Run benchmarks on standard build")
    standard_results = run_benchmarks("standard", args.quick)
    if not standard_results:
        print("\n❌ Standard benchmarks failed. Aborting.")
        sys.exit(1)

    pgo_results = None

    # Optional PGO workflow
    if args.with_pgo:
        # Step: Build PGO
        if not args.skip_build:
            current_step += 1
            print_step(current_step, total_steps, "Build PGO optimized version")
            if not build_pgo():
                print("\n⚠️  PGO build failed. Skipping PGO benchmarks.")
            else:
                # Step: Benchmark PGO
                current_step += 1
                print_step(current_step, total_steps, "Run benchmarks on PGO build")
                pgo_results = run_benchmarks("pgo", args.quick)

        # Step: Compare
        if pgo_results:
            current_step += 1
            print_step(current_step, total_steps, "Compare standard vs PGO")
            compare_results(standard_results, pgo_results, "standard_vs_pgo")

    # Final summary
    print_final_summary(standard_results, pgo_results)


if __name__ == '__main__':
    main()
