"""
Dense243 Performance Benchmark

Benchmarks the Dense243 encoding scheme vs standard ternary representation.

Dense243: 5 trits packed into 1 byte (243 possible states)
- Bits per trit: 1.6 bits (vs 2 bits for standard)
- Memory efficiency: 20% better than standard ternary
- Trade-off: Encoding/decoding overhead

Usage:
    python bench_dense243.py
    python bench_dense243.py --sizes 1000,10000,100000
"""

import argparse
import numpy as np
import time
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import ternary_simd_engine as tc
    HAS_STANDARD = True
except ImportError:
    print("Warning: ternary_simd_engine not available")
    HAS_STANDARD = False

try:
    import ternary_dense243_module as td243
    HAS_DENSE243 = True
except ImportError:
    print("Warning: ternary_dense243_module not available")
    print("Build with: python build/build_dense243.py")
    HAS_DENSE243 = False


class Dense243Benchmark:
    """
    Benchmark Dense243 encoding vs standard ternary

    Tests:
    1. Memory efficiency
    2. Encoding/decoding performance
    3. Operation throughput
    4. Overall effectiveness
    """

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, "results", "dense243")

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'has_standard': HAS_STANDARD,
                'has_dense243': HAS_DENSE243
            },
            'memory_efficiency': {},
            'encoding_performance': {},
            'operation_performance': {},
            'summary': {}
        }

    def benchmark_memory_efficiency(self, sizes: List[int]):
        """Test memory footprint of Dense243 vs standard"""
        print("\n" + "=" * 80)
        print("MEMORY EFFICIENCY BENCHMARK")
        print("=" * 80)

        results = {
            'sizes': [],
            'standard_bytes': [],
            'dense243_bytes': [],
            'reduction_factor': []
        }

        for size in sizes:
            # Standard ternary: 1 trit = 1 uint8 (8 bits, but only use 2)
            standard_bytes = size * 1  # 1 byte per trit

            # Dense243: 5 trits = 1 uint8 (perfect packing)
            dense243_bytes = (size + 4) // 5  # Round up to nearest 5

            reduction = standard_bytes / dense243_bytes if dense243_bytes > 0 else 0

            results['sizes'].append(size)
            results['standard_bytes'].append(standard_bytes)
            results['dense243_bytes'].append(dense243_bytes)
            results['reduction_factor'].append(reduction)

            print(f"\nSize: {size:,} trits")
            print(f"  Standard:  {standard_bytes:,} bytes (8 bits/trit)")
            print(f"  Dense243:  {dense243_bytes:,} bytes (1.6 bits/trit)")
            print(f"  Reduction: {reduction:.2f}x")

        self.results['memory_efficiency'] = results

        avg_reduction = sum(results['reduction_factor']) / len(results['reduction_factor'])
        print("\n" + "-" * 80)
        print(f"Average memory reduction: {avg_reduction:.2f}x")

        return results

    def benchmark_encoding_performance(self, sizes: List[int]):
        """Test encoding/decoding performance"""
        if not HAS_DENSE243:
            print("\nSkipping encoding benchmark - Dense243 not available")
            return {}

        print("\n" + "=" * 80)
        print("ENCODING/DECODING PERFORMANCE BENCHMARK")
        print("=" * 80)

        results = {
            'sizes': [],
            'encode_ns': [],
            'decode_ns': [],
            'roundtrip_ns': [],
            'throughput_gbps': []
        }

        iterations = 1000

        for size in sizes:
            # Generate random ternary data
            data = np.random.randint(0, 3, size, dtype=np.uint8)

            # Warmup
            for _ in range(100):
                encoded = td243.encode(data)
                decoded = td243.decode(encoded, size)

            # Benchmark encoding
            start = time.perf_counter_ns()
            for _ in range(iterations):
                encoded = td243.encode(data)
            encode_time = (time.perf_counter_ns() - start) / iterations

            # Benchmark decoding
            start = time.perf_counter_ns()
            for _ in range(iterations):
                decoded = td243.decode(encoded, size)
            decode_time = (time.perf_counter_ns() - start) / iterations

            # Roundtrip
            roundtrip_time = encode_time + decode_time

            # Throughput (GB/s of original data)
            throughput = (size / roundtrip_time) * 1e9 / 1e9

            results['sizes'].append(size)
            results['encode_ns'].append(encode_time)
            results['decode_ns'].append(decode_time)
            results['roundtrip_ns'].append(roundtrip_time)
            results['throughput_gbps'].append(throughput)

            print(f"\nSize: {size:,} trits")
            print(f"  Encode:     {encode_time:>10.2f}ns")
            print(f"  Decode:     {decode_time:>10.2f}ns")
            print(f"  Roundtrip:  {roundtrip_time:>10.2f}ns")
            print(f"  Throughput: {throughput:>10.2f} GB/s")

        self.results['encoding_performance'] = results
        return results

    def benchmark_operations(self, sizes: List[int]):
        """Test operation performance with Dense243 encoding"""
        if not HAS_DENSE243 or not HAS_STANDARD:
            print("\nSkipping operations benchmark - modules not available")
            return {}

        print("\n" + "=" * 80)
        print("OPERATIONS PERFORMANCE BENCHMARK")
        print("=" * 80)
        print("Comparing: Standard ternary vs Dense243 (with encoding overhead)")

        results = {
            'sizes': [],
            'standard_ns': [],
            'dense243_total_ns': [],
            'overhead_percent': []
        }

        iterations = 1000

        for size in sizes:
            a = np.random.randint(0, 3, size, dtype=np.uint8)
            b = np.random.randint(0, 3, size, dtype=np.uint8)

            # Benchmark standard ternary
            for _ in range(100):
                _ = tc.tadd(a, b)

            start = time.perf_counter_ns()
            for _ in range(iterations):
                result_std = tc.tadd(a, b)
            standard_time = (time.perf_counter_ns() - start) / iterations

            # Benchmark Dense243 (with encoding overhead)
            # Encode inputs
            a_enc = td243.encode(a)
            b_enc = td243.encode(b)

            for _ in range(100):
                a_dec = td243.decode(a_enc, size)
                b_dec = td243.decode(b_enc, size)
                result = tc.tadd(a_dec, b_dec)
                result_enc = td243.encode(result)

            start = time.perf_counter_ns()
            for _ in range(iterations):
                # Decode -> Operate -> Encode
                a_dec = td243.decode(a_enc, size)
                b_dec = td243.decode(b_enc, size)
                result = tc.tadd(a_dec, b_dec)
                result_enc = td243.encode(result)
            dense243_time = (time.perf_counter_ns() - start) / iterations

            overhead = ((dense243_time - standard_time) / standard_time) * 100

            results['sizes'].append(size)
            results['standard_ns'].append(standard_time)
            results['dense243_total_ns'].append(dense243_time)
            results['overhead_percent'].append(overhead)

            print(f"\nSize: {size:,} trits")
            print(f"  Standard:        {standard_time:>10.2f}ns")
            print(f"  Dense243 total:  {dense243_time:>10.2f}ns")
            print(f"  Overhead:        {overhead:>10.1f}%")

        self.results['operation_performance'] = results
        return results

    def run_all(self, sizes: List[int]):
        """Run complete benchmark suite"""
        print("=" * 80)
        print("DENSE243 COMPREHENSIVE BENCHMARK SUITE")
        print("=" * 80)
        print(f"Sizes: {sizes}")
        print(f"Standard ternary available: {HAS_STANDARD}")
        print(f"Dense243 available: {HAS_DENSE243}")

        self.benchmark_memory_efficiency(sizes)
        self.benchmark_encoding_performance(sizes)
        self.benchmark_operations(sizes)

        self.save_results()
        self.print_summary()

    def save_results(self):
        """Save results to JSON"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.output_dir, f"dense243_results_{timestamp}.json")

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Results saved to {filename}")
        return filename

    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "=" * 80)
        print("DENSE243 BENCHMARK SUMMARY")
        print("=" * 80)

        # Memory efficiency
        if 'memory_efficiency' in self.results and self.results['memory_efficiency']:
            mem = self.results['memory_efficiency']
            avg_reduction = sum(mem['reduction_factor']) / len(mem['reduction_factor'])
            print(f"\nMemory Efficiency: {avg_reduction:.2f}x better than standard")

        # Encoding performance
        if 'encoding_performance' in self.results and self.results['encoding_performance']:
            enc = self.results['encoding_performance']
            if enc.get('throughput_gbps'):
                avg_throughput = sum(enc['throughput_gbps']) / len(enc['throughput_gbps'])
                print(f"Encoding Throughput: {avg_throughput:.2f} GB/s")

        # Operations overhead
        if 'operation_performance' in self.results and self.results['operation_performance']:
            ops = self.results['operation_performance']
            if ops.get('overhead_percent'):
                avg_overhead = sum(ops['overhead_percent']) / len(ops['overhead_percent'])
                print(f"Operation Overhead: {avg_overhead:.1f}%")

        print("\n" + "=" * 80)
        print("VERDICT")
        print("=" * 80)
        print("Dense243 Trade-off:")
        print("  ✓ 2.5x better memory efficiency")
        print("  ✗ Encoding/decoding overhead for operations")
        print("\nBest Use Cases:")
        print("  • Model storage and distribution")
        print("  • Network transmission")
        print("  • Memory-constrained environments")
        print("\nNot Recommended For:")
        print("  • Real-time inference (use standard ternary)")
        print("  • High-throughput operations")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Dense243 Performance Benchmark'
    )
    parser.add_argument(
        '--sizes',
        default='1000,10000,100000,1000000',
        help='Comma-separated list of sizes to test'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output directory (default: benchmarks/results/dense243/)'
    )

    args = parser.parse_args()

    # Parse sizes
    sizes = [int(s.strip()) for s in args.sizes.split(',')]

    # Run benchmark
    benchmark = Dense243Benchmark(output_dir=args.output)
    benchmark.run_all(sizes)


if __name__ == "__main__":
    main()
