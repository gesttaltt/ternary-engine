#!/usr/bin/env python3
"""
run_simd_harness.py — Meta-test harness for SIMD verification

This script orchestrates comprehensive SIMD testing across:
- Multiple compilers (MSVC, GCC, Clang if available)
- Multiple optimization levels (-O0, -O2, -O3, -Ofast)
- Multiple test configurations
- Output comparison and determinism verification

Usage:
    python run_simd_harness.py
    python run_simd_harness.py --compilers msvc,gcc
    python run_simd_harness.py --opt-levels O0,O2,O3
    python run_simd_harness.py --quick  (reduced trials for CI)
"""

import subprocess
import os
import sys
import argparse
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
import platform

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Test configuration"""
    PROJECT_ROOT = Path(__file__).parent.parent
    TEST_SOURCE = PROJECT_ROOT / "tests" / "test_simd_correctness.cpp"
    BUILD_DIR = PROJECT_ROOT / "build" / "simd_harness"
    RESULTS_DIR = PROJECT_ROOT / "local-reports" / "simd_verification"

    # Compiler configurations
    COMPILERS = {
        'msvc': {
            'compiler': 'cl.exe',
            'flags': ['/EHsc', '/std:c++17', '/I', str(PROJECT_ROOT)],
            'opt_levels': {
                'O0': ['/Od'],
                'O2': ['/O2'],
                'O3': ['/Ox'],  # MSVC max optimization
            },
            'output_flag': '/Fe:',
            'arch_flag': '/arch:AVX2',
        },
        'gcc': {
            'compiler': 'g++',
            'flags': ['-std=c++17', '-I', str(PROJECT_ROOT), '-mavx2'],
            'opt_levels': {
                'O0': ['-O0'],
                'O2': ['-O2'],
                'O3': ['-O3'],
                'Ofast': ['-Ofast', '-ffast-math'],
            },
            'output_flag': '-o',
            'arch_flag': '-march=haswell',
        },
        'clang': {
            'compiler': 'clang++',
            'flags': ['-std=c++17', '-I', str(PROJECT_ROOT), '-mavx2'],
            'opt_levels': {
                'O0': ['-O0'],
                'O2': ['-O2'],
                'O3': ['-O3'],
                'Ofast': ['-Ofast', '-ffast-math'],
            },
            'output_flag': '-o',
            'arch_flag': '-march=haswell',
        },
    }

config = Config()

# ============================================================================
# Utilities
# ============================================================================

def print_header(text: str):
    """Print a formatted header"""
    border = "=" * 70
    print(f"\n{border}")
    print(f"  {text}")
    print(f"{border}\n")

def print_section(text: str):
    """Print a formatted section"""
    print(f"\n{'─' * 70}")
    print(f"  {text}")
    print(f"{'─' * 70}")

def check_compiler_available(compiler_name: str, compiler_cmd: str) -> bool:
    """Check if a compiler is available"""
    try:
        result = subprocess.run(
            [compiler_cmd, '--version'],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def hash_file(filepath: Path) -> str:
    """Compute SHA256 hash of a file"""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

# ============================================================================
# Compiler Detection
# ============================================================================

def detect_available_compilers() -> List[str]:
    """Detect which compilers are available on the system"""
    available = []

    print_section("Detecting Available Compilers")

    for name, cfg in config.COMPILERS.items():
        compiler_cmd = cfg['compiler']
        if check_compiler_available(name, compiler_cmd):
            print(f"  ✓ {name:10s} ({compiler_cmd}) - AVAILABLE")
            available.append(name)
        else:
            print(f"  ✗ {name:10s} ({compiler_cmd}) - NOT FOUND")

    if not available:
        print("\n  ⚠️  WARNING: No compilers detected!")
        print("  Install at least one of: MSVC, GCC, Clang")
        sys.exit(1)

    return available

# ============================================================================
# Compilation
# ============================================================================

def compile_test(compiler: str, opt_level: str, output_path: Path) -> Tuple[bool, str]:
    """
    Compile test with specified compiler and optimization level
    Returns: (success, error_message)
    """
    cfg = config.COMPILERS[compiler]

    # Build command
    cmd = [cfg['compiler']]
    cmd.extend(cfg['flags'])
    cmd.extend(cfg['opt_levels'][opt_level])
    cmd.append(cfg['arch_flag'])
    cmd.append(str(config.TEST_SOURCE))

    # Handle output path (different flag format for MSVC vs GCC/Clang)
    if compiler == 'msvc':
        cmd.append(f"{cfg['output_flag']}{output_path}")
    else:
        cmd.extend([cfg['output_flag'], str(output_path)])

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compile
    try:
        print(f"  Compiling: {compiler} {opt_level} -> {output_path.name}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=config.PROJECT_ROOT
        )

        if result.returncode != 0:
            return False, result.stderr

        return True, ""

    except subprocess.TimeoutExpired:
        return False, "Compilation timeout (60s)"
    except Exception as e:
        return False, str(e)

# ============================================================================
# Test Execution
# ============================================================================

def run_test(executable: Path) -> Tuple[bool, str, str]:
    """
    Run test executable
    Returns: (success, stdout, stderr)
    """
    try:
        print(f"  Executing: {executable.name}")
        result = subprocess.run(
            [str(executable)],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes
            cwd=executable.parent
        )

        return result.returncode == 0, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        return False, "", "Test execution timeout (300s)"
    except Exception as e:
        return False, "", str(e)

# ============================================================================
# Result Analysis
# ============================================================================

class TestRun:
    """Represents a single test run with specific configuration"""
    def __init__(self, compiler: str, opt_level: str):
        self.compiler = compiler
        self.opt_level = opt_level
        self.compile_success = False
        self.compile_error = ""
        self.test_success = False
        self.test_stdout = ""
        self.test_stderr = ""
        self.output_hash = ""
        self.executable_hash = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'compiler': self.compiler,
            'opt_level': self.opt_level,
            'compile_success': self.compile_success,
            'compile_error': self.compile_error,
            'test_success': self.test_success,
            'output_hash': self.output_hash,
            'executable_hash': self.executable_hash,
        }

def analyze_determinism(runs: List[TestRun]) -> Dict:
    """Analyze determinism across different builds"""
    analysis = {
        'total_runs': len(runs),
        'successful_runs': sum(1 for r in runs if r.test_success),
        'failed_runs': sum(1 for r in runs if not r.test_success),
        'unique_outputs': len(set(r.output_hash for r in runs if r.test_success)),
        'deterministic': False,
        'issues': []
    }

    # Check if all successful runs produced identical output
    successful_runs = [r for r in runs if r.test_success]
    if successful_runs:
        output_hashes = [r.output_hash for r in successful_runs]
        unique_hashes = set(output_hashes)

        if len(unique_hashes) == 1:
            analysis['deterministic'] = True
        else:
            analysis['deterministic'] = False
            analysis['issues'].append(
                f"Non-deterministic: {len(unique_hashes)} different outputs"
            )

            # Group by output hash
            hash_groups = {}
            for run in successful_runs:
                h = run.output_hash
                if h not in hash_groups:
                    hash_groups[h] = []
                hash_groups[h].append(f"{run.compiler}/{run.opt_level}")

            for h, configs in hash_groups.items():
                analysis['issues'].append(
                    f"  Hash {h[:8]}...: {', '.join(configs)}"
                )

    return analysis

# ============================================================================
# Main Orchestration
# ============================================================================

def run_comprehensive_tests(compilers: List[str], opt_levels: List[str],
                           quick_mode: bool = False) -> List[TestRun]:
    """
    Run comprehensive test suite across all configurations
    """
    runs = []

    print_header("Running Comprehensive SIMD Test Suite")

    total_configs = len(compilers) * len(opt_levels)
    current = 0

    for compiler in compilers:
        for opt_level in opt_levels:
            current += 1

            print_section(f"Configuration {current}/{total_configs}: {compiler} {opt_level}")

            run = TestRun(compiler, opt_level)

            # Compile
            exe_name = f"test_simd_{compiler}_{opt_level}"
            if platform.system() == "Windows":
                exe_name += ".exe"

            exe_path = config.BUILD_DIR / compiler / opt_level / exe_name

            compile_success, compile_error = compile_test(compiler, opt_level, exe_path)
            run.compile_success = compile_success
            run.compile_error = compile_error

            if not compile_success:
                print(f"  ✗ Compilation FAILED")
                print(f"    Error: {compile_error[:200]}")
                runs.append(run)
                continue

            print(f"  ✓ Compilation succeeded")

            # Compute executable hash (binary determinism check)
            run.executable_hash = hash_file(exe_path)

            # Run test
            test_success, stdout, stderr = run_test(exe_path)
            run.test_success = test_success
            run.test_stdout = stdout
            run.test_stderr = stderr

            if not test_success:
                print(f"  ✗ Test execution FAILED")
                print(f"    Error: {stderr[:200]}")
            else:
                print(f"  ✓ Test execution succeeded")
                # Compute output hash for determinism check
                run.output_hash = hashlib.sha256(stdout.encode()).hexdigest()

            runs.append(run)

    return runs

def print_results(runs: List[TestRun], analysis: Dict):
    """Print comprehensive test results"""
    print_header("Test Results Summary")

    # Overall statistics
    print("Overall Statistics:")
    print(f"  Total configurations:  {analysis['total_runs']}")
    print(f"  Successful tests:      {analysis['successful_runs']} ✓")
    print(f"  Failed tests:          {analysis['failed_runs']} ✗")
    print(f"  Unique outputs:        {analysis['unique_outputs']}")
    print(f"  Deterministic:         {'YES ✓' if analysis['deterministic'] else 'NO ✗'}")

    # Per-configuration results
    print("\nPer-Configuration Results:")
    print(f"  {'Compiler':<10} {'Opt':<8} {'Compile':<10} {'Test':<10} {'Hash':<16}")
    print(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*16}")

    for run in runs:
        compile_status = "✓ PASS" if run.compile_success else "✗ FAIL"
        test_status = "✓ PASS" if run.test_success else "✗ FAIL"
        output_hash = run.output_hash[:12] if run.test_success else "N/A"

        print(f"  {run.compiler:<10} {run.opt_level:<8} "
              f"{compile_status:<10} {test_status:<10} {output_hash:<16}")

    # Determinism analysis
    if not analysis['deterministic'] and analysis['issues']:
        print("\n⚠️  Determinism Issues Detected:")
        for issue in analysis['issues']:
            print(f"  {issue}")

    # Final verdict
    print_header("Final Verdict")

    if analysis['failed_runs'] == 0 and analysis['deterministic']:
        print("🎉 ALL TESTS PASSED!")
        print("   ✓ All configurations compiled successfully")
        print("   ✓ All tests passed")
        print("   ✓ Output is deterministic across configurations")
        print("\n   SIMD layer is PRODUCTION-READY")
        return True
    else:
        print("❌ TESTS FAILED!")
        if analysis['failed_runs'] > 0:
            print(f"   ✗ {analysis['failed_runs']} test(s) failed")
        if not analysis['deterministic']:
            print("   ✗ Non-deterministic output detected")
        print("\n   DO NOT USE IN PRODUCTION until issues are resolved")
        return False

def save_results(runs: List[TestRun], analysis: Dict):
    """Save detailed results to JSON file"""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        'runs': [r.to_dict() for r in runs],
        'analysis': analysis,
        'timestamp': subprocess.check_output(['date', '/t']).decode().strip() if platform.system() == "Windows" else subprocess.check_output(['date']).decode().strip(),
        'system': platform.system(),
        'architecture': platform.machine(),
    }

    output_file = config.RESULTS_DIR / "simd_verification_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Detailed results saved to: {output_file}")

# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SIMD verification meta-harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--compilers', type=str,
                       help='Comma-separated list of compilers (msvc,gcc,clang)')
    parser.add_argument('--opt-levels', type=str,
                       help='Comma-separated list of optimization levels (O0,O2,O3,Ofast)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: reduced test trials for CI')

    args = parser.parse_args()

    # Detect available compilers
    available_compilers = detect_available_compilers()

    # Select compilers to use
    if args.compilers:
        selected_compilers = args.compilers.split(',')
        # Validate
        for c in selected_compilers:
            if c not in available_compilers:
                print(f"Error: Compiler '{c}' not available")
                sys.exit(1)
        compilers = selected_compilers
    else:
        compilers = available_compilers

    # Select optimization levels
    if args.opt_levels:
        opt_levels = args.opt_levels.split(',')
    else:
        # Use all available optimization levels for first compiler
        opt_levels = list(config.COMPILERS[compilers[0]]['opt_levels'].keys())

    print(f"\nSelected configuration:")
    print(f"  Compilers:      {', '.join(compilers)}")
    print(f"  Opt levels:     {', '.join(opt_levels)}")
    print(f"  Quick mode:     {args.quick}")

    # Run tests
    runs = run_comprehensive_tests(compilers, opt_levels, args.quick)

    # Analyze results
    analysis = analyze_determinism(runs)

    # Print results
    success = print_results(runs, analysis)

    # Save results
    save_results(runs, analysis)

    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
