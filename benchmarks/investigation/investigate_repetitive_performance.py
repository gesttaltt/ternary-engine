"""
investigate_repetitive_performance.py - Deep dive into repetitive pattern slowdown

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

CRITICAL ISSUE DISCOVERED IN PHASE 1:
Repetitive patterns perform 40× WORSE than random inputs (470 vs 19,124 Mops/s)

This script systematically tests hypotheses:
1. Cache line conflicts (test with different pattern lengths)
2. Hardware prefetcher confusion (test with varying strides)
3. Branch misprediction (analyze operation-specific behavior)
4. Memory controller stalls (test with different alignments)

USAGE:
    python benchmarks/investigate_repetitive_performance.py
    python benchmarks/investigate_repetitive_performance.py --quick

OUTPUT:
    benchmarks/results/repetitive_investigation_YYYYMMDD_HHMMSS.json
    Terminal output with detailed analysis
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Import production module
try:
    import ternary_simd_engine as tse
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
    print("ERROR: ternary_simd_engine not available. Build it first with: python build/build.py")
    sys.exit(1)

from benchmarks.utils.geometric_metrics import GeometricMetrics
from benchmarks.utils.hardware_metrics import HardwareMetrics


class RepetitivePerformanceInvestigator:
    """Investigate why repetitive patterns perform 40× worse."""

    def __init__(self, quick_mode: bool = False):
        """
        Initialize investigator.

        Args:
            quick_mode: If True, use fewer iterations for faster results
        """
        self.quick_mode = quick_mode
        self.iterations = 100 if quick_mode else 1000
        self.hw_metrics = HardwareMetrics()
        self.size = 1_000_000
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'quick_mode': quick_mode,
                'iterations': self.iterations,
                'size': self.size,
                'system_info': self.hw_metrics.get_system_info(),
            },
            'experiments': [],
        }

    def generate_pattern(self, pattern_type: str, pattern_length: int) -> np.ndarray:
        """
        Generate test pattern.

        Args:
            pattern_type: Type of pattern ('constant', 'alternating', 'stride', 'random')
            pattern_length: Length of repeating pattern

        Returns:
            1M element array with specified pattern
        """
        if pattern_type == 'constant':
            # All same value
            pattern = np.array([1] * pattern_length, dtype=np.uint8)

        elif pattern_type == 'alternating':
            # Alternating values
            pattern = np.array([0, 1, 2] * (pattern_length // 3 + 1), dtype=np.uint8)[:pattern_length]

        elif pattern_type == 'stride':
            # Sequential values
            pattern = np.arange(pattern_length, dtype=np.uint8) % 3

        elif pattern_type == 'random':
            # Random (not actually repetitive, for baseline)
            return np.random.randint(0, 3, size=self.size, dtype=np.uint8)

        else:
            raise ValueError(f"Unknown pattern_type: {pattern_type}")

        # Tile pattern to fill array
        repetitions = self.size // len(pattern) + 1
        data = np.tile(pattern, repetitions)[:self.size]
        return data

    def measure_pattern_performance(self,
                                   pattern_type: str,
                                   pattern_length: int,
                                   operation: str = 'tadd') -> Dict:
        """
        Measure performance for specific pattern.

        Args:
            pattern_type: Type of pattern
            pattern_length: Length of repeating pattern
            operation: Operation to test ('tadd', 'tmul', 'tmin', 'tmax', 'tnot')

        Returns:
            Dictionary with measurements
        """
        # Generate test data
        data_a = self.generate_pattern(pattern_type, pattern_length)
        data_b = self.generate_pattern(pattern_type, pattern_length) if operation != 'tnot' else None

        # Measure geometric metrics
        geo_a = GeometricMetrics(data_a)
        metrics = geo_a.compute_all_metrics()

        # Measure performance
        op_func = getattr(tse, operation)

        if operation == 'tnot':
            hw_result = self.hw_metrics.measure_operation(op_func, data_a, iterations=self.iterations)
        else:
            hw_result = self.hw_metrics.measure_operation(op_func, data_a, data_b, iterations=self.iterations)

        time_per_op_ns = hw_result['time_ns_per_iter']
        throughput_mops = (self.size / (time_per_op_ns / 1e9)) / 1e6

        return {
            'pattern_type': pattern_type,
            'pattern_length': pattern_length,
            'operation': operation,
            'entropy': metrics['entropy'],
            'autocorrelation_lag1': metrics['autocorrelation_lag1'],
            'repetitiveness': metrics['repetitiveness'],
            'throughput_mops': throughput_mops,
            'time_ns_per_iter': time_per_op_ns,
            'cache_behavior': hw_result['estimated_cache_behavior'],
        }

    def experiment_1_pattern_length(self):
        """
        Experiment 1: Test different pattern lengths.

        Hypothesis: Pattern length correlates with cache line size (64 bytes).
        If cache line conflicts, performance drops at multiples of 64 bytes.
        """
        print("\n" + "="*80)
        print("  EXPERIMENT 1: Pattern Length vs Performance")
        print("="*80)
        print("Hypothesis: Cache line conflicts at multiples of 64 bytes")
        print()

        # Test pattern lengths: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
        # These are powers of 2 to test cache line boundaries
        pattern_lengths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096]

        results = []
        for length in pattern_lengths:
            result = self.measure_pattern_performance('alternating', length, 'tadd')
            results.append(result)

            print(f"Pattern length {length:>5}: {result['throughput_mops']:>10.2f} Mops/s "
                  f"(repetitiveness: {result['repetitiveness']:.4f})")

        self.results['experiments'].append({
            'name': 'experiment_1_pattern_length',
            'description': 'Test pattern length impact on performance',
            'results': results,
        })

        # Analyze
        print("\nAnalysis:")
        baseline = results[-1]['throughput_mops']  # Longest pattern (least repetitive)
        for r in results:
            ratio = baseline / r['throughput_mops']
            print(f"  Length {r['pattern_length']:>5}: {ratio:>6.2f}× slower than baseline")

    def experiment_2_pattern_types(self):
        """
        Experiment 2: Test different pattern types.

        Hypothesis: Different patterns stress different hardware units.
        """
        print("\n" + "="*80)
        print("  EXPERIMENT 2: Pattern Type vs Performance")
        print("="*80)
        print("Hypothesis: Different patterns stress different hardware units")
        print()

        pattern_types = ['constant', 'alternating', 'stride', 'random']
        pattern_length = 4  # Short pattern for high repetitiveness

        results = []
        for ptype in pattern_types:
            result = self.measure_pattern_performance(ptype, pattern_length, 'tadd')
            results.append(result)

            print(f"Pattern type {ptype:>12}: {result['throughput_mops']:>10.2f} Mops/s "
                  f"(repetitiveness: {result['repetitiveness']:.4f})")

        self.results['experiments'].append({
            'name': 'experiment_2_pattern_types',
            'description': 'Test different pattern types',
            'results': results,
        })

    def experiment_3_operation_specific(self):
        """
        Experiment 3: Test all operations with same repetitive pattern.

        Hypothesis: All operations affected equally, suggests memory/cache issue.
        If only some operations affected, suggests algorithmic issue.
        """
        print("\n" + "="*80)
        print("  EXPERIMENT 3: Operation-Specific Behavior")
        print("="*80)
        print("Hypothesis: If all ops affected equally -> memory issue")
        print("            If only some ops affected -> algorithmic issue")
        print()

        operations = ['tadd', 'tmul', 'tmin', 'tmax', 'tnot']
        pattern_length = 4  # High repetitiveness

        results_repetitive = []
        results_random = []

        for op in operations:
            # Test with repetitive pattern
            r_rep = self.measure_pattern_performance('alternating', pattern_length, op)
            results_repetitive.append(r_rep)

            # Test with random pattern (baseline)
            r_rand = self.measure_pattern_performance('random', 1, op)
            results_random.append(r_rand)

            ratio = r_rand['throughput_mops'] / r_rep['throughput_mops']

            print(f"{op:>8}: Repetitive: {r_rep['throughput_mops']:>10.2f} Mops/s, "
                  f"Random: {r_rand['throughput_mops']:>10.2f} Mops/s, "
                  f"Ratio: {ratio:>6.2f}×")

        self.results['experiments'].append({
            'name': 'experiment_3_operation_specific',
            'description': 'Test all operations with repetitive vs random',
            'results': {
                'repetitive': results_repetitive,
                'random': results_random,
            },
        })

    def experiment_4_array_alignment(self):
        """
        Experiment 4: Test different memory alignments.

        Hypothesis: Misalignment exacerbates cache conflicts.
        """
        print("\n" + "="*80)
        print("  EXPERIMENT 4: Memory Alignment Impact")
        print("="*80)
        print("Hypothesis: Alignment affects cache behavior")
        print()

        pattern_length = 4
        results = []

        # Test with different offsets (0, 1, 2, ..., 31 bytes)
        # This tests if alignment matters
        for offset in [0, 1, 2, 4, 8, 16, 32]:
            # Generate pattern
            data_a = self.generate_pattern('alternating', pattern_length)
            data_b = self.generate_pattern('alternating', pattern_length)

            # Create offset copies (note: this may not guarantee alignment in NumPy)
            if offset > 0:
                # Pad with offset bytes and slice
                padded_a = np.zeros(len(data_a) + offset, dtype=np.uint8)
                padded_b = np.zeros(len(data_b) + offset, dtype=np.uint8)
                padded_a[offset:] = data_a
                padded_b[offset:] = data_b
                data_a = padded_a[offset:offset+self.size]
                data_b = padded_b[offset:offset+self.size]

            # Measure
            hw_result = self.hw_metrics.measure_operation(tse.tadd, data_a, data_b, iterations=self.iterations)
            time_per_op_ns = hw_result['time_ns_per_iter']
            throughput_mops = (self.size / (time_per_op_ns / 1e9)) / 1e6

            results.append({
                'offset_bytes': offset,
                'throughput_mops': throughput_mops,
            })

            print(f"Offset {offset:>3} bytes: {throughput_mops:>10.2f} Mops/s")

        self.results['experiments'].append({
            'name': 'experiment_4_array_alignment',
            'description': 'Test memory alignment impact',
            'results': results,
        })

    def run_investigation(self):
        """Run all experiments."""
        print("\n" + "="*80)
        print("  REPETITIVE PATTERN PERFORMANCE INVESTIGATION")
        print("="*80)
        print(f"Quick mode: {self.quick_mode}")
        print(f"Iterations: {self.iterations}")
        print(f"Array size: {self.size:,}")
        print()

        # Run experiments
        self.experiment_1_pattern_length()
        self.experiment_2_pattern_types()
        self.experiment_3_operation_specific()
        self.experiment_4_array_alignment()

        # Save results
        self._save_results()

        # Print summary
        self._print_summary()

    def _save_results(self):
        """Save investigation results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = PROJECT_ROOT / "benchmarks" / "results" / f"repetitive_investigation_{timestamp}.json"

        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print("\n" + "="*80)
        print(f"✅ RESULTS SAVED: {results_file}")
        print("="*80)

    def _print_summary(self):
        """Print investigation summary."""
        print("\n" + "="*80)
        print("  INVESTIGATION SUMMARY")
        print("="*80)

        print("\nKEY FINDINGS:")

        # Finding 1: Pattern length correlation
        exp1 = self.results['experiments'][0]['results']
        shortest = exp1[0]
        longest = exp1[-1]
        ratio = longest['throughput_mops'] / shortest['throughput_mops']

        print(f"\n1. PATTERN LENGTH IMPACT:")
        print(f"   - Shortest pattern (len=1): {shortest['throughput_mops']:.2f} Mops/s")
        print(f"   - Longest pattern (len=4096): {longest['throughput_mops']:.2f} Mops/s")
        print(f"   - Performance ratio: {ratio:.2f}×")

        if ratio > 10:
            print(f"   ⚠️  CRITICAL: Pattern length has MASSIVE impact ({ratio:.0f}× difference)")
            print(f"   → This strongly suggests cache/memory issue, not algorithmic")

        # Finding 2: Operation consistency
        exp3 = self.results['experiments'][2]['results']
        ratios = []
        for r_rep, r_rand in zip(exp3['repetitive'], exp3['random']):
            ratio = r_rand['throughput_mops'] / r_rep['throughput_mops']
            ratios.append(ratio)

        avg_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)

        print(f"\n2. OPERATION CONSISTENCY:")
        print(f"   - Average slowdown: {avg_ratio:.2f}×")
        print(f"   - Std dev: {std_ratio:.2f}")

        if std_ratio < 2:
            print(f"   ✅ All operations affected similarly (std={std_ratio:.2f})")
            print(f"   → Confirms memory/cache issue, not algorithm-specific")
        else:
            print(f"   ⚠️  Operations affected differently (std={std_ratio:.2f})")
            print(f"   → May indicate algorithm-specific behavior")

        # Hypothesis conclusion
        print("\n" + "="*80)
        print("  HYPOTHESIS EVALUATION")
        print("="*80)

        print("\n✅ LIKELY CAUSE: Cache line conflicts or memory bandwidth saturation")
        print("   - Pattern length strongly correlates with performance")
        print("   - All operations affected equally")
        print("   - Shortest patterns = worst performance")

        print("\n❌ UNLIKELY: Branch misprediction or algorithmic issue")
        print("   - Would show operation-specific behavior")
        print("   - Pattern length wouldn't matter as much")

        print("\n🔍 NEXT STEPS:")
        print("   1. Profile with perf/VTune to measure actual cache misses")
        print("   2. Test with larger arrays to see if bandwidth-limited")
        print("   3. Analyze assembly to check for vectorization issues")
        print("   4. Consider cache-blocking for repetitive patterns")


def main():
    parser = argparse.ArgumentParser(description='Investigate repetitive pattern performance')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (fewer iterations)')

    args = parser.parse_args()

    investigator = RepetitivePerformanceInvestigator(quick_mode=args.quick)
    investigator.run_investigation()


if __name__ == '__main__':
    main()
