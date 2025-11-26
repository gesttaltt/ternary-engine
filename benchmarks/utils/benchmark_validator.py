"""
benchmark_validator.py - Automated benchmark validation and regression detection

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

MANDATORY VALIDATION POLICY:
- Run AFTER every feature addition (no exceptions)
- Auto-fail if regression > 5%
- Generate comparison reports
- Track performance history

USAGE:
    # Compare current build against baseline
    python benchmarks/utils/benchmark_validator.py \\
        --baseline benchmarks/results/baseline_v1.3.0.json \\
        --current benchmarks/results/bench_results_LATEST.json \\
        --threshold 0.05

    # Auto-run after build (in CI/CD)
    python build/build_experimental.py && \\
    python benchmarks/bench_phase0.py && \\
    python benchmarks/utils/benchmark_validator.py --auto

OUTPUT:
- validation_report_YYYY-MM-DD.md (human-readable)
- validation_results.json (machine-readable)
- Exit code 0 = PASS, 1 = FAIL (for CI/CD)
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import argparse

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
BASELINE_FILE = PROJECT_ROOT / "benchmarks" / "results" / "baseline_v1.3.0.json"
REPORTS_DIR = PROJECT_ROOT / "reports" / "benchmark_validation"

# v1.3.0 Baseline Performance (45.3 Gops/s effective)
BASELINE_PERFORMANCE = {
    "tadd_1M": 36110.99,   # Mops/s
    "tmul_1M": 35539.89,
    "tmin_1M": 31569.04,
    "tmax_1M": 29471.14,
    "tnot_1M": 39056.40,
    "fused_tnot_tadd_1M": 45300.0,  # Effective throughput
}

class BenchmarkValidator:
    def __init__(self, baseline_file: Path, current_file: Path, regression_threshold: float = 0.05):
        """
        Initialize validator with baseline and current benchmark results.

        Args:
            baseline_file: Path to baseline benchmark JSON
            current_file: Path to current benchmark JSON
            regression_threshold: Maximum acceptable regression (default: 5%)
        """
        self.baseline_file = baseline_file
        self.current_file = current_file
        self.regression_threshold = regression_threshold
        self.baseline_data = None
        self.current_data = None
        self.validation_results = {}

    def load_data(self) -> bool:
        """Load benchmark data from JSON files."""
        try:
            with open(self.baseline_file, 'r') as f:
                self.baseline_data = json.load(f)
            with open(self.current_file, 'r') as f:
                self.current_data = json.load(f)
            return True
        except FileNotFoundError as e:
            print(f"❌ ERROR: Benchmark file not found: {e}")
            return False
        except json.JSONDecodeError as e:
            print(f"❌ ERROR: Invalid JSON format: {e}")
            return False

    def extract_performance(self, data: Dict, operation: str, size: int) -> float:
        """
        Extract performance value from benchmark data.

        Args:
            data: Benchmark JSON data
            operation: Operation name (e.g., 'tadd')
            size: Array size (e.g., 1000000)

        Returns:
            Throughput in Mops/s
        """
        # Handle different JSON formats
        if 'benchmarks' in data:
            # Format 1: benchmarks array
            for bench in data['benchmarks']:
                if bench.get('operation') == operation and bench.get('size') == size:
                    return bench.get('throughput_mops', 0.0)
        elif 'results_optimized' in data:
            # Format 2: results_optimized array (bench_phase0.py output)
            for result in data['results_optimized']:
                if result.get('operation') == operation and result.get('size') == size:
                    return result.get('throughput_mops', 0.0)
        elif 'results' in data:
            # Format 3: results dict
            key = f"{operation}_{size}"
            return data['results'].get(key, {}).get('throughput_mops', 0.0)

        return 0.0

    def compare_performance(self) -> Dict[str, Dict]:
        """
        Compare current performance against baseline.

        Returns:
            Dictionary of comparisons with pass/fail status
        """
        comparisons = {}

        # Define critical benchmarks (must not regress)
        critical_benchmarks = [
            ('tadd', 1000000),
            ('tmul', 1000000),
            ('tmin', 1000000),
            ('tmax', 1000000),
            ('tnot', 1000000),
        ]

        for operation, size in critical_benchmarks:
            baseline_perf = self.extract_performance(self.baseline_data, operation, size)
            current_perf = self.extract_performance(self.current_data, operation, size)

            if baseline_perf == 0.0:
                # No baseline available, use hardcoded v1.3.0 values
                key = f"{operation}_{size//1000000}M"
                baseline_perf = BASELINE_PERFORMANCE.get(key, 0.0)

            if baseline_perf > 0.0:
                delta = current_perf - baseline_perf
                delta_pct = (delta / baseline_perf) * 100
                regression = delta_pct < -self.regression_threshold * 100

                comparisons[f"{operation}@{size}"] = {
                    'operation': operation,
                    'size': size,
                    'baseline_mops': baseline_perf,
                    'current_mops': current_perf,
                    'delta_mops': delta,
                    'delta_percent': delta_pct,
                    'regression': regression,
                    'status': 'FAIL' if regression else 'PASS',
                }

        return comparisons

    def validate(self) -> bool:
        """
        Run full validation suite.

        Returns:
            True if all checks pass, False if regression detected
        """
        print("\n" + "="*80)
        print("  BENCHMARK VALIDATION")
        print("="*80)
        print(f"Baseline: {self.baseline_file.name}")
        print(f"Current:  {self.current_file.name}")
        print(f"Regression Threshold: {self.regression_threshold * 100:.1f}%")
        print("="*80 + "\n")

        if not self.load_data():
            return False

        self.validation_results = self.compare_performance()

        # Print results table
        print(f"{'Operation':<20} {'Baseline':<12} {'Current':<12} {'Delta':<12} {'Status':<8}")
        print("-" * 80)

        all_passed = True
        for key, result in self.validation_results.items():
            op = result['operation']
            size = result['size']
            baseline = result['baseline_mops']
            current = result['current_mops']
            delta_pct = result['delta_percent']
            status = result['status']

            status_symbol = "✅" if status == "PASS" else "❌"
            print(f"{op}@{size:<10} {baseline:>10.2f} Mops {current:>10.2f} Mops "
                  f"{delta_pct:>+8.1f}% {status_symbol} {status}")

            if status == "FAIL":
                all_passed = False

        print("\n" + "="*80)
        if all_passed:
            print("✅ VALIDATION PASSED - No regressions detected")
        else:
            print("❌ VALIDATION FAILED - Regression detected!")
            print(f"   One or more benchmarks regressed by more than {self.regression_threshold * 100:.1f}%")
        print("="*80 + "\n")

        return all_passed

    def generate_report(self, output_path: Path) -> None:
        """Generate human-readable validation report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = f"""# Benchmark Validation Report

**Date:** {timestamp}
**Baseline:** {self.baseline_file.name}
**Current:** {self.current_file.name}
**Regression Threshold:** {self.regression_threshold * 100:.1f}%

---

## Summary

"""

        # Count passes and failures
        passes = sum(1 for r in self.validation_results.values() if r['status'] == 'PASS')
        failures = sum(1 for r in self.validation_results.values() if r['status'] == 'FAIL')

        report += f"- **Total Benchmarks:** {len(self.validation_results)}\n"
        report += f"- **Passed:** {passes}\n"
        report += f"- **Failed:** {failures}\n"
        report += f"- **Status:** {'✅ PASS' if failures == 0 else '❌ FAIL'}\n\n"

        report += "---\n\n## Detailed Results\n\n"
        report += "| Operation | Baseline (Mops/s) | Current (Mops/s) | Delta (%) | Status |\n"
        report += "|-----------|-------------------|------------------|-----------|--------|\n"

        for key, result in self.validation_results.items():
            op = result['operation']
            size = result['size']
            baseline = result['baseline_mops']
            current = result['current_mops']
            delta_pct = result['delta_percent']
            status = result['status']
            status_symbol = "✅" if status == "PASS" else "❌"

            report += f"| {op}@{size} | {baseline:.2f} | {current:.2f} | {delta_pct:+.1f} | {status_symbol} {status} |\n"

        report += "\n---\n\n## Interpretation\n\n"

        if failures == 0:
            report += "All benchmarks passed validation. No performance regressions detected.\n\n"
            report += "**Action:** Proceed with merge to main branch.\n"
        else:
            report += "Performance regression detected in one or more benchmarks.\n\n"
            report += "**Action:** Investigate regression before merging. Consider:\n"
            report += "1. Reviewing recent code changes\n"
            report += "2. Profiling to identify bottlenecks\n"
            report += "3. Reverting problematic commits\n"
            report += "4. Re-running benchmarks to rule out measurement noise\n"

        # Write report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)

        print(f"📄 Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Validate benchmark results against baseline')
    parser.add_argument('--baseline', type=Path, default=BASELINE_FILE,
                       help='Baseline benchmark JSON file')
    parser.add_argument('--current', type=Path, required=True,
                       help='Current benchmark JSON file to validate')
    parser.add_argument('--threshold', type=float, default=0.05,
                       help='Regression threshold (default: 0.05 = 5%%)')
    parser.add_argument('--output', type=Path,
                       default=REPORTS_DIR / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                       help='Output report path')
    parser.add_argument('--auto', action='store_true',
                       help='Auto mode: find latest benchmark in results/')

    args = parser.parse_args()

    # Auto mode: find latest benchmark
    if args.auto:
        results_dir = PROJECT_ROOT / "benchmarks" / "results"
        bench_files = sorted(results_dir.glob("bench_results_*.json"), reverse=True)
        if not bench_files:
            print("❌ ERROR: No benchmark files found in benchmarks/results/")
            return 1
        args.current = bench_files[0]
        print(f"📊 Auto-detected latest benchmark: {args.current.name}")

    # Run validation
    validator = BenchmarkValidator(args.baseline, args.current, args.threshold)
    passed = validator.validate()

    # Generate report
    validator.generate_report(args.output)

    # Save JSON results for machine parsing
    json_output = args.output.with_suffix('.json')
    with open(json_output, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'baseline': str(args.baseline),
            'current': str(args.current),
            'threshold': args.threshold,
            'results': validator.validation_results,
            'passed': passed,
        }, f, indent=2)

    print(f"📊 JSON results saved to: {json_output}")

    # Exit with appropriate code for CI/CD
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
