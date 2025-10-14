"""
bench_compare.py - Benchmark comparison tool for Ternary Engine

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Compares benchmark results between two builds to detect:
- Performance regressions
- Performance improvements
- Scaling behavior changes

Usage:
    python benchmarks/bench_compare.py baseline.json optimized.json
    python benchmarks/bench_compare.py --threshold=5.0 before.json after.json
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ComparisonResult:
    """Result of comparing two benchmarks"""
    operation: str
    size: int
    before_mops: float
    after_mops: float
    change_percent: float
    before_ns_per_elem: float
    after_ns_per_elem: float

    @property
    def is_regression(self) -> bool:
        """Check if this is a performance regression"""
        return self.change_percent < -5.0  # More than 5% slower

    @property
    def is_improvement(self) -> bool:
        """Check if this is a performance improvement"""
        return self.change_percent > 5.0  # More than 5% faster


def load_benchmark_results(filepath: Path) -> Dict:
    """Load benchmark results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def compare_benchmarks(before: Dict, after: Dict) -> List[ComparisonResult]:
    """Compare two benchmark result sets"""
    results = []

    # Create lookup for 'after' results
    after_lookup = {}
    for r in after['results_optimized']:
        key = (r['operation'], r['size'])
        after_lookup[key] = r

    # Compare each 'before' result with corresponding 'after' result
    for before_r in before['results_optimized']:
        key = (before_r['operation'], before_r['size'])

        if key in after_lookup:
            after_r = after_lookup[key]

            # Calculate change percentage
            before_mops = before_r['throughput_mops']
            after_mops = after_r['throughput_mops']
            change_percent = ((after_mops - before_mops) / before_mops) * 100

            comparison = ComparisonResult(
                operation=before_r['operation'],
                size=before_r['size'],
                before_mops=before_mops,
                after_mops=after_mops,
                change_percent=change_percent,
                before_ns_per_elem=before_r['time_ns_per_elem'],
                after_ns_per_elem=after_r['time_ns_per_elem']
            )
            results.append(comparison)

    return results


def print_comparison_table(comparisons: List[ComparisonResult]):
    """Print formatted comparison table"""
    print("\n" + "=" * 100)
    print("  BENCHMARK COMPARISON")
    print("=" * 100)
    print(f"\n{'Operation':<10} {'Size':>12} {'Before':>12} {'After':>12} {'Change':>10} {'Status':<15}")
    print("-" * 100)

    for comp in comparisons:
        # Color-code the change
        if comp.is_regression:
            status = "⚠️  REGRESSION"
        elif comp.is_improvement:
            status = "✅ IMPROVEMENT"
        else:
            status = "→ Neutral"

        print(f"{comp.operation:<10} {comp.size:>12,} "
              f"{comp.before_mops:>11.2f}M {comp.after_mops:>11.2f}M "
              f"{comp.change_percent:>9.1f}% {status:<15}")

    print("-" * 100)


def print_summary(comparisons: List[ComparisonResult]):
    """Print summary statistics"""
    regressions = [c for c in comparisons if c.is_regression]
    improvements = [c for c in comparisons if c.is_improvement]
    neutral = [c for c in comparisons if not c.is_regression and not c.is_improvement]

    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)

    print(f"\nTotal comparisons: {len(comparisons)}")
    print(f"  Improvements: {len(improvements)} ({len(improvements)/len(comparisons)*100:.1f}%)")
    print(f"  Regressions:  {len(regressions)} ({len(regressions)/len(comparisons)*100:.1f}%)")
    print(f"  Neutral:      {len(neutral)} ({len(neutral)/len(comparisons)*100:.1f}%)")

    if improvements:
        avg_improvement = sum(c.change_percent for c in improvements) / len(improvements)
        print(f"\nAverage improvement: +{avg_improvement:.1f}%")
        best = max(improvements, key=lambda x: x.change_percent)
        print(f"Best improvement: {best.operation} @ {best.size:,} elements: +{best.change_percent:.1f}%")

    if regressions:
        avg_regression = sum(c.change_percent for c in regressions) / len(regressions)
        print(f"\nAverage regression: {avg_regression:.1f}%")
        worst = min(regressions, key=lambda x: x.change_percent)
        print(f"Worst regression: {worst.operation} @ {worst.size:,} elements: {worst.change_percent:.1f}%")

    # Overall average change
    avg_change = sum(c.change_percent for c in comparisons) / len(comparisons)
    if avg_change > 0:
        print(f"\nOverall change: +{avg_change:.1f}% (faster)")
    else:
        print(f"\nOverall change: {avg_change:.1f}% (slower)")

    print("\n" + "=" * 100)


def save_comparison_report(comparisons: List[ComparisonResult], output_path: Path):
    """Save comparison report to JSON"""
    report = {
        'comparisons': [
            {
                'operation': c.operation,
                'size': c.size,
                'before_mops': c.before_mops,
                'after_mops': c.after_mops,
                'change_percent': c.change_percent,
                'is_regression': c.is_regression,
                'is_improvement': c.is_improvement,
            }
            for c in comparisons
        ],
        'summary': {
            'total': len(comparisons),
            'improvements': len([c for c in comparisons if c.is_improvement]),
            'regressions': len([c for c in comparisons if c.is_regression]),
            'neutral': len([c for c in comparisons if not c.is_regression and not c.is_improvement]),
            'avg_change_percent': sum(c.change_percent for c in comparisons) / len(comparisons),
        }
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nComparison report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare two benchmark results')
    parser.add_argument('before', type=str, help='Path to baseline/before benchmark JSON')
    parser.add_argument('after', type=str, help='Path to optimized/after benchmark JSON')
    parser.add_argument('--output', type=str, help='Output path for comparison report JSON')
    parser.add_argument('--threshold', type=float, default=5.0,
                       help='Threshold for regression/improvement detection (default: 5.0%%)')

    args = parser.parse_args()

    # Load benchmark results
    before_path = Path(args.before)
    after_path = Path(args.after)

    if not before_path.exists():
        print(f"ERROR: Baseline file not found: {before_path}")
        sys.exit(1)

    if not after_path.exists():
        print(f"ERROR: Comparison file not found: {after_path}")
        sys.exit(1)

    print(f"Loading baseline: {before_path}")
    before = load_benchmark_results(before_path)

    print(f"Loading comparison: {after_path}")
    after = load_benchmark_results(after_path)

    # Compare results
    comparisons = compare_benchmarks(before, after)

    if not comparisons:
        print("\nWARNING: No matching benchmarks found between the two files.")
        print("Make sure both files contain results for the same operations and sizes.")
        sys.exit(1)

    # Print results
    print_comparison_table(comparisons)
    print_summary(comparisons)

    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        save_comparison_report(comparisons, output_path)

    # Exit with error code if regressions detected
    regressions = [c for c in comparisons if c.is_regression]
    if regressions:
        print(f"\n⚠️  WARNING: {len(regressions)} performance regressions detected!")
        sys.exit(1)

    print("\n✅ No significant performance regressions detected.")
    sys.exit(0)


if __name__ == '__main__':
    main()
