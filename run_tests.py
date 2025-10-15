#!/usr/bin/env python3
"""
run_tests.py - Unified test runner for Ternary Engine

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Runs all test suites and provides unified reporting.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --quick      # Skip slow tests
    python run_tests.py --verbose    # Detailed output
    python run_tests.py --suite=<name>  # Run specific test suite
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.resolve()
TESTS_DIR = PROJECT_ROOT / "tests"

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(message):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}  {message}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")

def print_success(message):
    """Print success message"""
    print(f"{Colors.OKGREEN}[OK] {message}{Colors.ENDC}")

def print_error(message):
    """Print error message"""
    print(f"{Colors.FAIL}[FAIL] {message}{Colors.ENDC}")

def print_warning(message):
    """Print warning message"""
    print(f"{Colors.WARNING}[WARN] {message}{Colors.ENDC}")

def run_test_suite(script_path, name, verbose=False):
    """Run a single test suite"""
    print(f"\n{Colors.OKBLUE}Running: {name}{Colors.ENDC}")
    print(f"  Script: {script_path}")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=not verbose,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print_success(f"{name} passed")
            if verbose and result.stdout:
                print(result.stdout)
            return True
        else:
            print_error(f"{name} failed (exit code: {result.returncode})")
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print_error(f"{name} timed out (>60s)")
        return False
    except Exception as e:
        print_error(f"{name} raised exception: {e}")
        return False

def check_module_built():
    """Check if the ternary_simd_engine module is built"""
    pyd_files = list(PROJECT_ROOT.glob("ternary_simd_engine*.pyd"))
    so_files = list(PROJECT_ROOT.glob("ternary_simd_engine*.so"))

    if pyd_files or so_files:
        module_file = (pyd_files + so_files)[0]
        print_success(f"Module found: {module_file.name}")
        return True
    else:
        print_error("Module not found. Build the module first:")
        print("  python build.py")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run Ternary Engine test suite')
    parser.add_argument('--quick', action='store_true',
                       help='Skip slow tests')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--suite', type=str,
                       help='Run specific test suite (phase0, omp, all)')
    parser.add_argument('--no-color', action='store_true',
                       help='Disable colored output')

    args = parser.parse_args()

    # Disable colors if requested
    if args.no_color:
        for attr in dir(Colors):
            if not attr.startswith('_'):
                setattr(Colors, attr, '')

    print_header("TERNARY ENGINE TEST SUITE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")

    # Check if module is built
    print_header("Pre-flight Checks")
    if not check_module_built():
        return 1

    # Define test suites
    test_suites = {
        'phase0': {
            'name': 'Phase 0 Correctness Tests',
            'script': TESTS_DIR / 'test_phase0.py',
            'required': True
        },
        'omp': {
            'name': 'OpenMP Parallelization Tests',
            'script': TESTS_DIR / 'test_omp.py',
            'required': True
        },
        'errors': {
            'name': 'Error Handling & Edge Cases',
            'script': TESTS_DIR / 'test_errors.py',
            'required': True
        }
    }

    # Filter test suites based on arguments
    if args.suite and args.suite != 'all':
        if args.suite not in test_suites:
            print_error(f"Unknown test suite: {args.suite}")
            print(f"Available suites: {', '.join(test_suites.keys())}, all")
            return 1
        test_suites = {args.suite: test_suites[args.suite]}

    # Run tests
    print_header("Running Test Suites")
    results = {}

    for suite_id, suite_info in test_suites.items():
        if not suite_info['script'].exists():
            print_warning(f"Skipping {suite_info['name']}: script not found")
            continue

        success = run_test_suite(
            suite_info['script'],
            suite_info['name'],
            verbose=args.verbose
        )
        results[suite_id] = success

    # Print summary
    print_header("Test Summary")

    total_suites = len(results)
    passed_suites = sum(1 for r in results.values() if r)
    failed_suites = total_suites - passed_suites

    print(f"Total test suites: {total_suites}")
    print(f"Passed: {Colors.OKGREEN}{passed_suites}{Colors.ENDC}")
    print(f"Failed: {Colors.FAIL}{failed_suites}{Colors.ENDC}")

    if failed_suites > 0:
        print(f"\n{Colors.FAIL}{Colors.BOLD}FAILED TEST SUITES:{Colors.ENDC}")
        for suite_id, success in results.items():
            if not success:
                print(f"  - {test_suites[suite_id]['name']}")

    print("\n" + "="*70)

    if all(results.values()):
        print(f"{Colors.OKGREEN}{Colors.BOLD}[SUCCESS] ALL TESTS PASSED!{Colors.ENDC}")
        print("="*70 + "\n")
        return 0
    else:
        print(f"{Colors.FAIL}{Colors.BOLD}[FAIL] SOME TESTS FAILED{Colors.ENDC}")
        print("="*70 + "\n")
        return 1

if __name__ == '__main__':
    sys.exit(main())
