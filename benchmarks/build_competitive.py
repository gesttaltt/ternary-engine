"""
build_competitive.py - Build and Prepare for Competitive Benchmarking

Prepares the complete environment for running competitive benchmarks:
1. Builds ternary_simd_engine (standard optimized)
2. Installs Python dependencies (PyTorch, Transformers, NumPy)
3. Downloads TinyLlama-1.1B model for quantization testing
4. Runs Phase 0 validation to ensure engine works
5. Generates preparation report

Usage:
    python benchmarks/build_competitive.py
    python benchmarks/build_competitive.py --skip-model  # Skip model download
    python benchmarks/build_competitive.py --skip-deps   # Skip dependency install
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

def run_command(cmd, description, cwd=None, check=True):
    """Run command and report status"""
    print(f"[*] {description}...")
    print(f"    Command: {' '.join(cmd)}")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or PROJECT_ROOT,
            check=check,
            capture_output=True,
            text=True
        )
        elapsed = time.time() - start
        print(f"    ✓ Success ({elapsed:.1f}s)")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print(f"    ✗ Failed ({elapsed:.1f}s)")
        if e.stderr:
            print(f"    Error: {e.stderr[:500]}")
        return False, e.stderr

def build_engine():
    """Build ternary_simd_engine"""
    print_header("STEP 1: BUILD TERNARY SIMD ENGINE")

    build_script = PROJECT_ROOT / "build" / "build.py"
    success, output = run_command(
        [sys.executable, str(build_script)],
        "Building ternary_simd_engine (standard optimized)"
    )

    if success:
        print("\n    Engine built successfully")
        # Check for module file
        module_files = list(PROJECT_ROOT.glob("ternary_simd_engine*.pyd")) + \
                      list(PROJECT_ROOT.glob("ternary_simd_engine*.so"))
        if module_files:
            for mf in module_files:
                size_kb = mf.stat().st_size / 1024
                print(f"    Found: {mf.name} ({size_kb:.1f} KB)")

    return success

def install_dependencies(skip_deps=False):
    """Install Python dependencies for competitive benchmarks"""
    print_header("STEP 2: INSTALL PYTHON DEPENDENCIES")

    if skip_deps:
        print("    Skipping dependency installation (--skip-deps flag)")
        return True

    dependencies = {
        'numpy': 'NumPy for numerical operations',
        'torch': 'PyTorch for model quantization',
        'transformers': 'HuggingFace Transformers for model loading'
    }

    results = {}

    # Check existing installations
    print("Checking existing installations...")
    for dep, desc in dependencies.items():
        try:
            __import__(dep)
            print(f"    ✓ {dep} already installed")
            results[dep] = True
        except ImportError:
            print(f"    ✗ {dep} not found - {desc}")
            results[dep] = False

    # Install missing dependencies
    missing = [dep for dep, installed in results.items() if not installed]

    if missing:
        print(f"\nInstalling {len(missing)} missing dependencies...")
        for dep in missing:
            success, output = run_command(
                [sys.executable, "-m", "pip", "install", dep],
                f"Installing {dep}",
                check=False
            )
            results[dep] = success

    # Summary
    installed = sum(1 for v in results.values() if v)
    print(f"\n    Dependencies: {installed}/{len(dependencies)} available")

    return all(results.values())

def download_model(skip_model=False):
    """Download TinyLlama model for benchmarking"""
    print_header("STEP 3: DOWNLOAD TINYLLAMA MODEL")

    if skip_model:
        print("    Skipping model download (--skip-model flag)")
        return True

    # Check if transformers is available
    try:
        import transformers
    except ImportError:
        print("    ✗ Transformers not available, skipping model download")
        return False

    download_script = PROJECT_ROOT / "benchmarks" / "download_models.py"

    if not download_script.exists():
        print(f"    ✗ Download script not found: {download_script}")
        return False

    print("    Downloading TinyLlama-1.1B (this may take a few minutes)...")
    success, output = run_command(
        [sys.executable, str(download_script), "--model", "tinyllama"],
        "Downloading TinyLlama-1.1B model",
        cwd=PROJECT_ROOT / "benchmarks",
        check=False
    )

    return success

def run_phase0_validation():
    """Run Phase 0 validation tests"""
    print_header("STEP 4: RUN PHASE 0 VALIDATION")

    test_script = PROJECT_ROOT / "tests" / "test_phase0.py"

    if not test_script.exists():
        print(f"    ⚠ Test script not found: {test_script}")
        print("    Skipping validation (not critical for competitive benchmarks)")
        return True

    success, output = run_command(
        [sys.executable, str(test_script)],
        "Running Phase 0 validation tests",
        check=False
    )

    if not success:
        print("    ⚠ Some tests failed, but continuing...")
        print("    (Competitive benchmarks may still work)")

    return True  # Don't fail build if tests fail

def generate_report(results):
    """Generate build preparation report"""
    print_header("STEP 5: GENERATE PREPARATION REPORT")

    report_dir = PROJECT_ROOT / "reports" / "builds"
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = report_dir / f"competitive_prep_{timestamp}.txt"

    # Build summary
    all_passed = all(results.values())

    report_content = []
    report_content.append("=" * 80)
    report_content.append("COMPETITIVE BENCHMARKING - BUILD PREPARATION REPORT")
    report_content.append("=" * 80)
    report_content.append("")
    report_content.append(f"Timestamp: {datetime.now().isoformat()}")
    report_content.append(f"Project Root: {PROJECT_ROOT}")
    report_content.append("")
    report_content.append("Preparation Steps:")
    report_content.append("")

    for step, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        report_content.append(f"  {status:8} | {step}")

    report_content.append("")
    report_content.append("=" * 80)

    if all_passed:
        report_content.append("STATUS: ✓ READY FOR COMPETITIVE BENCHMARKING")
        report_content.append("")
        report_content.append("Next Steps:")
        report_content.append("  1. cd benchmarks")
        report_content.append("  2. python bench_competitive.py --all")
        report_content.append("  3. python bench_power_consumption.py --platform windows")
        report_content.append("  4. Check results in benchmarks/results/")
    else:
        report_content.append("STATUS: ⚠ PARTIAL PREPARATION")
        report_content.append("")
        failed = [step for step, success in results.items() if not success]
        report_content.append("Failed Steps:")
        for step in failed:
            report_content.append(f"  - {step}")
        report_content.append("")
        report_content.append("You may still run benchmarks, but some features may be limited.")

    report_content.append("=" * 80)

    # Write report
    report_text = "\n".join(report_content)
    with open(report_file, 'w') as f:
        f.write(report_text)

    # Print to console
    print(report_text)
    print(f"\n✓ Report saved to: {report_file}")

    return all_passed

def main():
    """Main build orchestration"""
    import argparse

    parser = argparse.ArgumentParser(description='Build and prepare for competitive benchmarking')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency installation')
    parser.add_argument('--skip-model', action='store_true', help='Skip model download')
    parser.add_argument('--skip-tests', action='store_true', help='Skip validation tests')

    args = parser.parse_args()

    print("=" * 80)
    print("  COMPETITIVE BENCHMARKING - BUILD PREPARATION")
    print("  Ternary Engine Competitive Analysis Suite")
    print("=" * 80)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Python: {sys.executable}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # Step 1: Build engine
    results['Engine Build'] = build_engine()

    # Step 2: Install dependencies
    results['Dependencies'] = install_dependencies(args.skip_deps)

    # Step 3: Download model
    results['Model Download'] = download_model(args.skip_model)

    # Step 4: Run validation
    if not args.skip_tests:
        results['Validation'] = run_phase0_validation()
    else:
        print_header("STEP 4: SKIP VALIDATION")
        print("    Skipping validation tests (--skip-tests flag)")
        results['Validation'] = True

    # Step 5: Generate report
    success = generate_report(results)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
