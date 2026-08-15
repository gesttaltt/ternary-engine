"""
build_all.py - Unified Build Script for Entire Project

Builds all components and runs validation:
1. Standard engine (ternary_simd_engine)
2. Dense243 experimental, TritNet GEMM, reference baseline, backend,
   zero-skip GEMM, TritNet inference (each individually skippable via --no-*)
3. Runs Phase 0 validation tests
4. Prepares competitive benchmarks
5. Generates build report

Usage:
    python build/build_all.py
    python build/build_all.py --quick    # Skip tests
    python build/build_all.py --clean    # Clean first
"""

import sys
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

def print_header(title):
    """Print section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def run_command(cmd, description, cwd=None):
    """Run command and report status"""
    print(f"[*] {description}...")
    print(f"    Command: {' '.join(cmd)}")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True
        )
        elapsed = time.time() - start
        print(f"    ✓ Success ({elapsed:.1f}s)")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print(f"    ✗ Failed ({elapsed:.1f}s)")
        print(f"    Error: {e.stderr}")
        return False

def clean_build():
    """Clean all build artifacts"""
    print_header("CLEANING BUILD ARTIFACTS")

    clean_script = PROJECT_ROOT / "build" / "clean_all.py"
    return run_command(
        [sys.executable, str(clean_script)],
        "Cleaning all artifacts"
    )

def build_standard():
    """Build standard optimized engine"""
    print_header("BUILDING STANDARD ENGINE")

    build_script = PROJECT_ROOT / "build" / "build.py"
    return run_command(
        [sys.executable, str(build_script)],
        "Building ternary_simd_engine"
    )

def build_dense243():
    """Build Dense243 experimental"""
    print_header("BUILDING DENSE243 EXPERIMENTAL")

    build_script = PROJECT_ROOT / "build" / "build_dense243.py"
    return run_command(
        [sys.executable, str(build_script)],
        "Building ternary_dense243"
    )

def build_tritnet_gemm():
    """Build TritNet GEMM module"""
    print_header("BUILDING TRITNET GEMM")

    build_script = PROJECT_ROOT / "build" / "build_tritnet_gemm.py"
    return run_command(
        [sys.executable, str(build_script)],
        "Building ternary_tritnet_gemm"
    )

def build_reference():
    """Build reference baseline (optional)"""
    print_header("BUILDING REFERENCE BASELINE")

    build_script = PROJECT_ROOT / "build" / "build_reference.py"
    return run_command(
        [sys.executable, str(build_script)],
        "Building reference_cpp"
    )

def build_backend():
    """Build pluggable backend module"""
    print_header("BUILDING BACKEND")

    build_script = PROJECT_ROOT / "build" / "build_backend.py"
    return run_command(
        [sys.executable, str(build_script)],
        "Building ternary_backend"
    )

def build_zero_skip_gemm():
    """Build zero-skip ternary GEMM module"""
    print_header("BUILDING ZERO-SKIP GEMM")

    build_script = PROJECT_ROOT / "build" / "build_zero_skip_gemm.py"
    return run_command(
        [sys.executable, str(build_script)],
        "Building ternary_zero_skip_gemm"
    )

def build_tritnet_inference():
    """Build TritNet inference module"""
    print_header("BUILDING TRITNET INFERENCE")

    build_script = PROJECT_ROOT / "build" / "build_tritnet_inference.py"
    return run_command(
        [sys.executable, str(build_script)],
        "Building ternary_tritnet_inference"
    )

def run_phase0_tests():
    """Run Phase 0 validation tests"""
    print_header("RUNNING PHASE 0 VALIDATION")

    test_script = PROJECT_ROOT / "tests" / "test_phase0.py"
    return run_command(
        [sys.executable, str(test_script)],
        "Running Phase 0 tests"
    )

def check_competitive_deps():
    """Check and report competitive benchmark dependencies"""
    print_header("CHECKING COMPETITIVE BENCHMARK DEPENDENCIES")

    deps = {
        'numpy': False,
        'torch': False,
        'transformers': False
    }

    for dep in deps.keys():
        try:
            __import__(dep)
            deps[dep] = True
            print(f"    ✓ {dep} available")
        except ImportError:
            print(f"    ✗ {dep} not available")

    if not all(deps.values()):
        print("\n    To install missing dependencies:")
        missing = [d for d, available in deps.items() if not available]
        print(f"    pip install {' '.join(missing)}")

    return all(deps.values())

def generate_build_report(results):
    """Generate build report"""
    print_header("BUILD REPORT")

    report = {
        'timestamp': datetime.now().isoformat(),
        'results': results
    }

    # Print summary
    print("Build Summary:")
    for step, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status:8} | {step}")

    # Overall status
    all_passed = all(results.values())
    print("\n" + "=" * 80)
    if all_passed:
        print("  ✓ ALL BUILDS SUCCESSFUL")
    else:
        print("  ✗ SOME BUILDS FAILED")
    print("=" * 80)

    # Save report
    report_dir = PROJECT_ROOT / "reports" / "builds"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_file = report_dir / f"build_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("UNIFIED BUILD REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {report['timestamp']}\n\n")
        f.write("Results:\n")
        for step, success in results.items():
            status = "PASS" if success else "FAIL"
            f.write(f"  {status:6} | {step}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Overall: {'SUCCESS' if all_passed else 'FAILURE'}\n")
        f.write("=" * 80 + "\n")

    print(f"\n✓ Report saved to: {report_file}")

    return all_passed

def main():
    """Main build orchestration"""
    import argparse

    parser = argparse.ArgumentParser(description='Unified Build Script')
    parser.add_argument('--clean', action='store_true', help='Clean before building')
    parser.add_argument('--quick', action='store_true', help='Skip tests')
    parser.add_argument('--no-dense243', action='store_true', help='Skip Dense243 build')
    parser.add_argument('--no-tritnet-gemm', action='store_true', help='Skip TritNet GEMM build')
    parser.add_argument('--no-reference', action='store_true', help='Skip reference baseline build')
    parser.add_argument('--no-backend', action='store_true', help='Skip backend build')
    parser.add_argument('--no-zero-skip-gemm', action='store_true', help='Skip zero-skip GEMM build')
    parser.add_argument('--no-tritnet-inference', action='store_true', help='Skip TritNet inference build')

    args = parser.parse_args()

    print("=" * 80)
    print("  UNIFIED BUILD SYSTEM")
    print("  Ternary Engine - Complete Build")
    print("=" * 80)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Python: {sys.executable}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # Step 1: Clean (optional)
    if args.clean:
        results['clean'] = clean_build()

    # Step 2: Build standard engine
    results['standard_build'] = build_standard()

    # Step 3: Build Dense243 (optional)
    if not args.no_dense243:
        results['dense243_build'] = build_dense243()

    # Step 4: Build TritNet GEMM (optional)
    if not args.no_tritnet_gemm:
        results['tritnet_gemm_build'] = build_tritnet_gemm()

    # Step 5: Build reference baseline (optional)
    if not args.no_reference:
        results['reference_build'] = build_reference()

    # Step 5b: Build backend module (optional)
    if not args.no_backend:
        results['backend_build'] = build_backend()

    # Step 5c: Build zero-skip GEMM module (optional)
    if not args.no_zero_skip_gemm:
        results['zero_skip_gemm_build'] = build_zero_skip_gemm()

    # Step 5d: Build TritNet inference module (optional)
    if not args.no_tritnet_inference:
        results['tritnet_inference_build'] = build_tritnet_inference()

    # Step 6: Run Phase 0 tests (optional)
    if not args.quick and results.get('standard_build', False):
        results['phase0_tests'] = run_phase0_tests()

    # Step 7: Check competitive dependencies
    results['competitive_deps'] = check_competitive_deps()

    # Step 8: Generate report
    success = generate_build_report(results)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
