#!/usr/bin/env python3
"""
run_tests.py - Unified test runner for Ternary Engine

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Runs all test suites and provides unified reporting.

Usage:
    python tests/run_tests.py              # Run all tests
    python tests/run_tests.py --quick      # Skip slow tests
    python tests/run_tests.py --verbose    # Detailed output
    python tests/run_tests.py --suite=<name>  # Run specific test suite
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# Script is now in tests/ directory, go up one level to project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
TESTS_DIR = Path(__file__).parent.resolve()

# Import capability detection
sys.path.insert(0, str(TESTS_DIR / 'python'))
try:
    from test_capabilities import detect_capabilities
    CAPABILITIES_AVAILABLE = True
except ImportError:
    CAPABILITIES_AVAILABLE = False
    print("[WARN] Capability detection not available")

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

def print_skip(message):
    """Print skipped test message"""
    print(f"{Colors.WARNING}[SKIP] {message}{Colors.ENDC}")

def run_test_suite(script_path, name, verbose=False, can_skip=False, skip_reason=None):
    """Run a single test suite

    Args:
        script_path: Path to test script
        name: Test suite name
        verbose: Print verbose output
        can_skip: Whether this test can be skipped
        skip_reason: Reason for skipping

    Returns:
        tuple: (passed: bool, skipped: bool)
    """
    if can_skip and skip_reason:
        print(f"\n{Colors.WARNING}Skipping: {name}{Colors.ENDC}")
        print(f"  Reason: {skip_reason}")
        return (True, True)  # Treat skipped as passed

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
            return (True, False)  # Passed, not skipped
        else:
            print_error(f"{name} failed (exit code: {result.returncode})")
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return (False, False)  # Failed, not skipped
    except subprocess.TimeoutExpired:
        print_error(f"{name} timed out (>60s)")
        return (False, False)
    except Exception as e:
        print_error(f"{name} raised exception: {e}")
        return (False, False)

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

    # Detect capabilities
    capabilities = None
    if CAPABILITIES_AVAILABLE:
        try:
            capabilities = detect_capabilities()
            capabilities.print_report()
        except Exception as e:
            print_warning(f"Could not detect capabilities: {e}")

    # Check if module is built
    print_header("Pre-flight Checks")
    if not check_module_built():
        return 1

    # Define test suites (now in python/ subdirectory)
    test_suites = {
        'phase0': {
            'name': 'Phase 0 Correctness Tests',
            'script': TESTS_DIR / 'python' / 'test_phase0.py',
            'required': True,
            'optional': False,
            'requires_capability': None
        },
        'omp': {
            'name': 'OpenMP Parallelization Tests',
            'script': TESTS_DIR / 'python' / 'test_omp.py',
            'required': False,  # Optional on platforms without OpenMP
            'optional': True,
            'requires_capability': 'openmp'
        },
        'errors': {
            'name': 'Error Handling & Edge Cases',
            'script': TESTS_DIR / 'python' / 'test_errors.py',
            'required': True,
            'optional': False,
            'requires_capability': None
        },
        'fusion': {
            'name': 'Operation Fusion Tests',
            'script': TESTS_DIR / 'python' / 'test_fusion.py',
            'required': False,  # Experimental feature
            'optional': True,
            'requires_capability': 'fusion'
        },
        'tritnet_gemm': {
            'name': 'TritNet GEMM Integration Tests',
            'script': TESTS_DIR / 'python' / 'test_tritnet_gemm_integration.py',
            'required': False,  # Optional module
            'optional': True,
            'requires_capability': None  # Will check for module in test itself
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
    results = {}  # suite_id -> (passed, skipped)

    for suite_id, suite_info in test_suites.items():
        if not suite_info['script'].exists():
            print_warning(f"Skipping {suite_info['name']}: script not found")
            continue

        # Check if test can be skipped based on capabilities
        can_skip = False
        skip_reason = None

        if suite_info['optional'] and capabilities:
            required_cap = suite_info['requires_capability']
            if required_cap == 'openmp':
                skip_reason = capabilities.get_skip_reason('openmp')
                can_skip = skip_reason is not None

        passed, skipped = run_test_suite(
            suite_info['script'],
            suite_info['name'],
            verbose=args.verbose,
            can_skip=can_skip,
            skip_reason=skip_reason
        )
        results[suite_id] = (passed, skipped)

    # Print summary
    print_header("Test Summary")

    total_suites = len(results)
    passed_suites = sum(1 for passed, skipped in results.values() if passed and not skipped)
    skipped_suites = sum(1 for passed, skipped in results.values() if skipped)
    failed_suites = sum(1 for passed, skipped in results.values() if not passed and not skipped)

    print(f"Total test suites: {total_suites}")
    print(f"Passed: {Colors.OKGREEN}{passed_suites}{Colors.ENDC}")
    print(f"Skipped: {Colors.WARNING}{skipped_suites}{Colors.ENDC}")
    print(f"Failed: {Colors.FAIL}{failed_suites}{Colors.ENDC}")

    if skipped_suites > 0:
        print(f"\n{Colors.WARNING}{Colors.BOLD}SKIPPED TEST SUITES:{Colors.ENDC}")
        for suite_id, (passed, skipped) in results.items():
            if skipped:
                print(f"  - {test_suites[suite_id]['name']}")

    if failed_suites > 0:
        print(f"\n{Colors.FAIL}{Colors.BOLD}FAILED TEST SUITES:{Colors.ENDC}")
        for suite_id, (passed, skipped) in results.items():
            if not passed and not skipped:
                print(f"  - {test_suites[suite_id]['name']}")

    print("\n" + "="*70)

    if failed_suites == 0:
        if skipped_suites > 0:
            print(f"{Colors.OKGREEN}{Colors.BOLD}[SUCCESS] ALL REQUIRED TESTS PASSED!{Colors.ENDC}")
            print(f"{Colors.WARNING}(Some optional tests were skipped){Colors.ENDC}")
        else:
            print(f"{Colors.OKGREEN}{Colors.BOLD}[SUCCESS] ALL TESTS PASSED!{Colors.ENDC}")
        print("="*70 + "\n")
        return 0
    else:
        print(f"{Colors.FAIL}{Colors.BOLD}[FAIL] SOME TESTS FAILED{Colors.ENDC}")
        print("="*70 + "\n")
        return 1

if __name__ == '__main__':
    sys.exit(main())
