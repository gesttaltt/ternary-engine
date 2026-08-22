"""
Competitive Benchmarking Suite
Tests ternary operations against industry standards

This suite implements the comprehensive benchmarks from real.md to prove
whether ternary has commercial value by comparing against:
- NumPy INT8 operations
- Memory efficiency (INT4/INT8/FP16)
- Throughput at equivalent bit-width
- Neural network workload patterns
- Real model quantization
- Power consumption

Usage:
    python bench_competitive.py
    python bench_competitive.py --phase 1
    python bench_competitive.py --all
"""

import numpy as np
import time
import json
import argparse
import gc
import statistics
from datetime import datetime
from typing import Dict, List, Any, Tuple
import sys
import os

# Add parent directory to path to import ternary engine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # fixed 2026-08-12: was 1 dirname() short of repo root

from benchmark_framework import compute_timing_statistics  # noqa: E402 (path insert above must run first)

try:
    import ternary_simd_engine as tc
except ImportError:
    print("Warning: ternary_simd_engine not available, using mock operations")
    print("Build the module first: python build.py")
    # Mock operations for testing the framework
    class MockTC:
        @staticmethod
        def tadd(a, b):
            return (a + b) % 3

        @staticmethod
        def tmul(a, b):
            return (a * b) % 3

        @staticmethod
        def tsub(a, b):
            return (a - b) % 3

    tc = MockTC()


def _packed_signed_reference(bits: int):
    """
    Build a bit-packed signed-integer reference (INT2/INT4) for Phase 3.

    NumPy has no native dtype below 8 bits, so a genuine `bits`-wide
    packed representation (`8 // bits` lanes per uint8 byte, signed range
    [-2**(bits-1), 2**(bits-1)-1]) must pay a real unpack -> add -> clip
    -> repack cost on every op. That IS the honest price of hitting that
    bit density without a dedicated compiled kernel -- same standard
    bench_fair_baseline.py already applies ("fastest reasonable NumPy
    implementation of the same semantics").

    Returns (lanes_per_byte, make_array(n_bytes, rng), saturating_add(a, b)).
    """
    lanes = 8 // bits
    mask = (1 << bits) - 1
    max_val = (1 << (bits - 1)) - 1
    min_val = -(1 << (bits - 1))

    def make_array(n_bytes: int, rng: np.random.Generator) -> np.ndarray:
        packed = np.zeros(n_bytes, dtype=np.uint8)
        for lane in range(lanes):
            vals = rng.integers(min_val, max_val + 1, n_bytes, dtype=np.int16)
            nib = (vals & mask).astype(np.uint8)
            packed |= (nib << (lane * bits)).astype(np.uint8)
        return packed

    def saturating_add(a_packed: np.ndarray, b_packed: np.ndarray) -> np.ndarray:
        out = np.zeros_like(a_packed)
        for lane in range(lanes):
            shift = lane * bits
            a_lane = ((a_packed >> shift) & mask).astype(np.int16)
            a_lane = np.where(a_lane > max_val, a_lane - (1 << bits), a_lane)
            b_lane = ((b_packed >> shift) & mask).astype(np.int16)
            b_lane = np.where(b_lane > max_val, b_lane - (1 << bits), b_lane)
            s = np.clip(a_lane + b_lane, min_val, max_val)
            nib = (s & mask).astype(np.uint8)
            out |= (nib << shift).astype(np.uint8)
        return out

    return lanes, make_array, saturating_add


def _adaptive_timing(fn, *args, target_total_s: float = 2.0, min_reps: int = 3, max_reps: int = 30):
    """
    Time fn(*args) with a rep count chosen so the WHOLE measurement takes
    roughly target_total_s, instead of a fixed iteration count. Per-op
    cost in Phase 3 spans two orders of magnitude across representations
    (compiled SIMD vs NumPy bit-unpacking), so a fixed count is either
    too slow for the cheap ops or too noisy for the expensive ones.

    Returns (median_ns, reps).
    """
    fn(*args)  # warmup (also pays any first-call JIT/cache cost)
    start = time.perf_counter_ns()
    fn(*args)
    one_ns = max(time.perf_counter_ns() - start, 1)
    reps = max(min_reps, min(max_reps, int(target_total_s * 1e9 / one_ns)))
    times = []
    for _ in range(reps):
        start = time.perf_counter_ns()
        fn(*args)
        times.append(time.perf_counter_ns() - start)
    return statistics.median(times), reps


class CompetitiveBenchmark:
    """
    Comprehensive competitive benchmark suite

    Tests ternary engine against industry standards across 6 phases:
    1. Arithmetic operations vs NumPy
    2. Memory efficiency analysis
    3. Throughput at equivalent bit-width
    4. Neural network workload patterns
    5. Real model quantization
    6. Power consumption
    """

    def __init__(self, output_dir: str = None):
        # Default to benchmarks/results/competitive/
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, "results", "competitive")

        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'platform': sys.platform,
                'numpy_version': np.__version__
            },
            'phase1_arithmetic_comparison': {},
            'phase2_memory_efficiency': {},
            'phase3_throughput_equivalent_bitwidth': {},
            'phase4_neural_workload_patterns': {},
            'phase5_model_quantization': {},
            'phase6_power_consumption': {},
        }
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_all(self):
        """Run complete benchmark suite"""
        print("=" * 80)
        print("TERNARY ENGINE COMPETITIVE BENCHMARK SUITE")
        print("=" * 80)
        print(f"Started: {self.results['metadata']['timestamp']}")
        print(f"Platform: {sys.platform}")
        print(f"NumPy: {np.__version__}")

        print("\n[1/6] Arithmetic Operations vs NumPy...")
        self.phase1_benchmark_vs_numpy()

        print("\n[2/6] Memory Efficiency Analysis...")
        self.phase2_benchmark_memory_efficiency()

        print("\n[3/6] Throughput at Equivalent Bit-Width...")
        self.phase3_benchmark_equivalent_bitwidth()

        print("\n[4/6] Neural Network Workload Patterns...")
        self.phase4_benchmark_nn_patterns()

        print("\n[5/6] Model Quantization Analysis...")
        self.phase5_model_quantization()

        print("\n[6/6] Power Consumption Framework...")
        self.phase6_power_consumption()

        self.save_results()
        self.print_summary()

    def phase1_benchmark_vs_numpy(self):
        """
        Phase 1: Fair Arithmetic Comparisons

        Direct comparison with NumPy INT8 operations to establish baseline
        performance for equivalent information density operations.
        """
        print("\n" + "=" * 80)
        print("PHASE 1: Ternary vs NumPy INT8 Operations")
        print("=" * 80)

        sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]

        results = {
            'size': [],
            'ternary_add_ns': [],
            'numpy_int8_add_ns': [],
            'ternary_mul_ns': [],
            'numpy_int8_mul_ns': [],
            'ternary_throughput_gbps': [],
            'numpy_throughput_gbps': [],
            'add_speedup': [],
            'mul_speedup': []
        }

        for size in sizes:
            print(f"\nTesting size: {size:,} elements")

            # Ternary (2 bits per element, stored in uint8)
            a_tern = np.random.randint(0, 3, size, dtype=np.uint8)
            b_tern = np.random.randint(0, 3, size, dtype=np.uint8)

            # NumPy INT8 (8 bits per element)
            a_np = np.random.randint(-1, 2, size, dtype=np.int8)
            b_np = np.random.randint(-1, 2, size, dtype=np.int8)

            # Warm up
            for _ in range(100):
                _ = tc.tadd(a_tern, b_tern)
                _ = np.add(a_np, b_np, dtype=np.int8)

            # Benchmark ternary addition
            iterations = 1000
            start = time.perf_counter_ns()
            for _ in range(iterations):
                result_tern = tc.tadd(a_tern, b_tern)
            ternary_add_time = (time.perf_counter_ns() - start) / iterations

            # Benchmark NumPy addition
            start = time.perf_counter_ns()
            for _ in range(iterations):
                result_np = np.add(a_np, b_np, dtype=np.int8)
            numpy_add_time = (time.perf_counter_ns() - start) / iterations

            # Benchmark ternary multiplication
            start = time.perf_counter_ns()
            for _ in range(iterations):
                result_tern = tc.tmul(a_tern, b_tern)
            ternary_mul_time = (time.perf_counter_ns() - start) / iterations

            # Benchmark NumPy multiplication
            start = time.perf_counter_ns()
            for _ in range(iterations):
                result_np = np.multiply(a_np, b_np, dtype=np.int8)
            numpy_mul_time = (time.perf_counter_ns() - start) / iterations

            # Calculate throughput (GB/s)
            # Ternary: 2 bits/element = 0.25 bytes/element
            ternary_bytes = size * 0.25 * 2  # 2 arrays
            ternary_gbps = (ternary_bytes / ternary_add_time) * 1e9 / 1e9

            # NumPy: 1 byte/element
            numpy_bytes = size * 1 * 2  # 2 arrays
            numpy_gbps = (numpy_bytes / numpy_add_time) * 1e9 / 1e9

            add_speedup = numpy_add_time / ternary_add_time
            mul_speedup = numpy_mul_time / ternary_mul_time

            results['size'].append(size)
            results['ternary_add_ns'].append(ternary_add_time)
            results['numpy_int8_add_ns'].append(numpy_add_time)
            results['ternary_mul_ns'].append(ternary_mul_time)
            results['numpy_int8_mul_ns'].append(numpy_mul_time)
            results['ternary_throughput_gbps'].append(ternary_gbps)
            results['numpy_throughput_gbps'].append(numpy_gbps)
            results['add_speedup'].append(add_speedup)
            results['mul_speedup'].append(mul_speedup)

            print(f"  Addition:")
            print(f"    Ternary: {ternary_add_time:>10.2f}ns ({ternary_gbps:>6.2f} GB/s)")
            print(f"    NumPy:   {numpy_add_time:>10.2f}ns ({numpy_gbps:>6.2f} GB/s)")
            print(f"    Speedup: {add_speedup:.2f}x")
            print(f"  Multiplication:")
            print(f"    Ternary: {ternary_mul_time:>10.2f}ns")
            print(f"    NumPy:   {numpy_mul_time:>10.2f}ns")
            print(f"    Speedup: {mul_speedup:.2f}x")

        self.results['phase1_arithmetic_comparison'] = results

        # Print summary
        avg_add_speedup = sum(results['add_speedup']) / len(results['add_speedup'])
        avg_mul_speedup = sum(results['mul_speedup']) / len(results['mul_speedup'])

        print("\n" + "-" * 80)
        print(f"Phase 1 Summary:")
        print(f"  Average addition speedup:       {avg_add_speedup:.2f}x")
        print(f"  Average multiplication speedup: {avg_mul_speedup:.2f}x")
        print(f"  Verdict: {'✓ COMPETITIVE' if avg_add_speedup > 1.0 else '✗ NEEDS WORK'}")

    def phase2_benchmark_memory_efficiency(self):
        """
        Phase 2: Memory Footprint Comparisons

        Compare storage efficiency at equivalent model capacity against
        FP16, INT8, INT4, and Dense243 encoding.
        """
        print("\n" + "=" * 80)
        print("PHASE 2: Memory Footprint Analysis")
        print("=" * 80)

        model_sizes = [
            ("Small (7B params)", 7_000_000_000),
            ("Medium (13B params)", 13_000_000_000),
            ("Large (70B params)", 70_000_000_000),
            ("XL (405B params)", 405_000_000_000),
        ]

        results = []

        for name, params in model_sizes:
            print(f"\n{name}:")

            # FP16 baseline
            fp16_bytes = params * 2
            print(f"  FP16:     {fp16_bytes / 1e9:>8.2f} GB (baseline)")

            # INT8 quantization
            int8_bytes = params * 1
            int8_reduction = fp16_bytes / int8_bytes
            print(f"  INT8:     {int8_bytes / 1e9:>8.2f} GB ({int8_reduction:.1f}x smaller)")

            # INT4 quantization
            int4_bytes = params * 0.5
            int4_reduction = fp16_bytes / int4_bytes
            print(f"  INT4:     {int4_bytes / 1e9:>8.2f} GB ({int4_reduction:.1f}x smaller)")

            # Ternary (2 bits per weight, stored inefficiently)
            ternary_naive_bytes = params * 0.25
            ternary_naive_reduction = fp16_bytes / ternary_naive_bytes
            print(f"  Ternary:  {ternary_naive_bytes / 1e9:>8.2f} GB ({ternary_naive_reduction:.1f}x smaller)")

            # Ternary Dense243 (5 trits per byte = 1.6 bits per trit)
            dense243_bytes = params * (1.6 / 8)
            dense243_reduction = fp16_bytes / dense243_bytes
            print(f"  Dense243: {dense243_bytes / 1e9:>8.2f} GB ({dense243_reduction:.1f}x smaller)")

            # Memory bandwidth savings
            bw_vs_int8 = int8_bytes / ternary_naive_bytes
            bw_vs_int4 = int4_bytes / ternary_naive_bytes

            print(f"  Memory bandwidth reduction vs INT8: {bw_vs_int8:.2f}x")
            print(f"  Memory bandwidth reduction vs INT4: {bw_vs_int4:.2f}x")

            results.append({
                'name': name,
                'params': params,
                'fp16_gb': fp16_bytes / 1e9,
                'int8_gb': int8_bytes / 1e9,
                'int4_gb': int4_bytes / 1e9,
                'ternary_gb': ternary_naive_bytes / 1e9,
                'dense243_gb': dense243_bytes / 1e9,
                'ternary_vs_fp16': ternary_naive_reduction,
                'ternary_vs_int8': bw_vs_int8,
                'ternary_vs_int4': bw_vs_int4
            })

        self.results['phase2_memory_efficiency'] = results

        print("\n" + "-" * 80)
        print("Phase 2 Summary:")
        print(f"  Ternary memory advantage over INT8: 4.0x")
        print(f"  Ternary memory advantage over INT4: 2.0x")
        print(f"  Dense243 memory advantage over INT4: 2.5x")
        print(f"  Verdict: ✓ SIGNIFICANT ADVANTAGE")

    def phase3_benchmark_equivalent_bitwidth(self):
        """
        Phase 3: Throughput at Equivalent Bit-Width

        Compare saturating-add throughput across FOUR representations that
        each genuinely occupy the same ~1GB footprint:

          - Ternary (engine native): tc.tadd on the SIMD engine's actual
            working format -- uint8, ONE BYTE per trit. Included for
            continuity with Phase 1/2, but honestly labeled: this is NOT
            2-bit packed, whatever the "2 bits of information" framing
            elsewhere implies. It gets 4x the elements of the packed
            representations below for the same byte budget.
          - Ternary (Dense243, ~1.6 bits/trit): ternary_dense243_module.tadd
            operating directly on packed dense243 bytes -- the actual
            sub-byte-dense ternary format in this codebase.
          - INT4 packed (2 signed nibbles/byte, NumPy reference): real
            unpack -> add -> clip -> repack per op, see
            _packed_signed_reference().
          - INT2 packed (4 signed 2-bit lanes/byte, NumPy reference): same.

        Fixed 2026-08-11: the prior version measured only the ternary side
        and printed "NEEDS INT2/INT4 REFERENCE FOR COMPARISON" -- it never
        built one. This version does, and its own array-size bug is fixed
        too: `np.uint8` arrays sized for "elements = bytes / 0.25" actually
        allocate 4x the intended 1GB (each element still costs 1 full byte
        in that dtype), which is what running Phase 3 previously spiked
        this machine to ~7.5GB RSS. See
        reports/2026-08-11/LINUX_VALIDATION_REPORT.md for the full gap.
        """
        print("\n" + "=" * 80)
        print("PHASE 3: Throughput at Equivalent Bit-Width")
        print("=" * 80)

        target_bytes = 1_000_000_000  # 1GB, genuinely enforced per representation below
        rng = np.random.default_rng(42)
        representations = []

        print(f"\nTesting with {target_bytes / 1e9:.1f}GB memory footprint per representation:")

        # --- Ternary, engine native format (1 byte/trit) ---
        n_elem = target_bytes
        a = rng.integers(0, 3, n_elem, dtype=np.uint8)
        b = rng.integers(0, 3, n_elem, dtype=np.uint8)
        t_ns, reps = _adaptive_timing(tc.tadd, a, b)
        gops = (n_elem / t_ns)
        representations.append({
            'name': 'Ternary (engine native, 1 byte/trit)', 'bytes': n_elem,
            'elements': n_elem, 'time_ns': t_ns, 'reps': reps, 'gops': gops,
        })
        print(f"\n  Ternary (engine native, 1 byte/trit): {n_elem:,} elements")
        print(f"    {t_ns/1e6:.2f}ms/op ({reps} reps), {gops:.2f} GOPS")
        del a, b
        gc.collect()

        # --- Ternary, Dense243 (~1.6 bits/trit, compiled, packed) ---
        try:
            import ternary_dense243_module as td
            n_bytes = target_bytes
            a = rng.integers(0, 243, n_bytes, dtype=np.uint8)  # 0-242 are all valid dense243 bytes
            b = rng.integers(0, 243, n_bytes, dtype=np.uint8)
            t_ns, reps = _adaptive_timing(td.tadd, a, b)
            n_trits = n_bytes * 5  # DENSITY = 5 trits/byte
            gops = (n_trits / t_ns)
            representations.append({
                'name': 'Ternary (Dense243, 1.6 bits/trit)', 'bytes': n_bytes,
                'elements': n_trits, 'time_ns': t_ns, 'reps': reps, 'gops': gops,
            })
            print(f"\n  Ternary (Dense243, ~1.6 bits/trit): {n_trits:,} trits in {n_bytes:,} bytes")
            print(f"    {t_ns/1e6:.2f}ms/op ({reps} reps), {gops:.2f} GOPS")
            del a, b
            gc.collect()
        except ImportError:
            print("\n  Dense243 module not built -- skipping (build/build_dense243.py)")

        # --- INT4 packed reference (2 lanes/byte) ---
        lanes4, make4, add4 = _packed_signed_reference(4)
        a = make4(target_bytes, rng)
        b = make4(target_bytes, rng)
        t_ns, reps = _adaptive_timing(add4, a, b)
        n_elem4 = target_bytes * lanes4
        gops = (n_elem4 / t_ns)
        representations.append({
            'name': 'INT4 packed (2 lanes/byte, NumPy ref)', 'bytes': target_bytes,
            'elements': n_elem4, 'time_ns': t_ns, 'reps': reps, 'gops': gops,
        })
        print(f"\n  INT4 packed (NumPy reference, 2 lanes/byte): {n_elem4:,} elements")
        print(f"    {t_ns/1e6:.2f}ms/op ({reps} reps), {gops:.2f} GOPS")
        del a, b
        gc.collect()

        # --- INT2 packed reference (4 lanes/byte) ---
        lanes2, make2, add2 = _packed_signed_reference(2)
        a = make2(target_bytes, rng)
        b = make2(target_bytes, rng)
        t_ns, reps = _adaptive_timing(add2, a, b)
        n_elem2 = target_bytes * lanes2
        gops = (n_elem2 / t_ns)
        representations.append({
            'name': 'INT2 packed (4 lanes/byte, NumPy ref)', 'bytes': target_bytes,
            'elements': n_elem2, 'time_ns': t_ns, 'reps': reps, 'gops': gops,
        })
        print(f"\n  INT2 packed (NumPy reference, 4 lanes/byte): {n_elem2:,} elements")
        print(f"    {t_ns/1e6:.2f}ms/op ({reps} reps), {gops:.2f} GOPS")
        del a, b
        gc.collect()

        self.results['phase3_throughput_equivalent_bitwidth'] = {
            'target_bytes': target_bytes,
            'representations': representations,
            'note': ('INT2/INT4 are NumPy bit-packed references (real unpack/add/repack '
                     'per op -- no dedicated compiled kernel exists for them here, same '
                     'standard bench_fair_baseline.py uses: "fastest reasonable NumPy '
                     'implementation"). Dense243 uses the compiled module directly on '
                     'packed bytes. Engine-native uses the SIMD engine\'s actual working '
                     'format, which is 1 byte/trit, not 2-bit packed.'),
        }

        # Verdict compares like-for-like: Dense243 (true packed ternary) vs
        # the INT2 NumPy reference (same nominal 2-bit width). Computed from
        # what was actually measured above, not hardcoded.
        by_name = {r['name']: r['gops'] for r in representations}
        dense_gops = by_name.get('Ternary (Dense243, 1.6 bits/trit)')
        int2_gops = by_name.get('INT2 packed (4 lanes/byte, NumPy ref)')

        print("Phase 3 Summary:")
        for r in representations:
            print(f"  {r['name']:<42s} {r['gops']:>8.3f} GOPS")
        if dense_gops is not None and int2_gops:
            ratio = dense_gops / int2_gops
            if ratio >= 1.0:
                verdict = f"✓ Dense243 is {ratio:.1f}x FASTER than the INT2 NumPy reference at equivalent (~2-bit) density"
            else:
                verdict = f"✗ Dense243 is {1/ratio:.1f}x SLOWER than the INT2 NumPy reference at equivalent (~2-bit) density"
        else:
            verdict = "⚠ Dense243 module unavailable -- comparison incomplete"
        print(f"  Verdict: {verdict}")
        self.results['phase3_throughput_equivalent_bitwidth']['verdict'] = verdict

    def phase4_benchmark_nn_patterns(self):
        """
        Phase 4: Neural Network Workload Patterns

        Simulate actual neural network operations:
        - Matrix-vector multiplication (inference)
        - Batch operations
        - Common layer sizes

        This is critical: AI is matrix multiplication. If ternary ops are fast
        but matmul is slow, there's no viable AI solution.

        Fixed 2026-08-11: the prior version benchmarked a per-row Python
        loop (`for i in range(M): tc.tmul(...); np.sum(...)`), which mostly
        measures CPython interpreter/dispatch overhead, not the engine's
        actual GEMM capability -- this repo already has a compiled,
        AVX2-vectorized ternary GEMM kernel
        (ternary_zero_skip_gemm.ZeroSkipWeights) with its own correctness
        suite (tests/python/test_zero_skip_gemm.py). This version calls
        that instead. The sparse index is built ONCE per weight matrix,
        outside the timed loop -- realistic for inference, where weights
        are fixed and only activations change per call; index-build cost
        would only belong in the timing if weights changed every call.
        Every ternary result is also checked against the NumPy reference
        (max abs error) so a speed number can't hide a correctness bug.

        Switched to DenseWeights 2026-08-20 (see
        src/core/simd/ternary_gemm_dense.h and
        reports/2026-08-20/GEMM_DENSE_PACKED_OPTIMIZATION.md): investigating
        this phase's "0.189x avg / TOO SLOW FOR AI" verdict found ZeroSkipWeights'
        CSC/CSR index storage is actually ~3.3x LARGER than the dense int8
        weight array at ternary's real ~33% zero density, and its kernels
        vectorize over the batch dimension -- which this phase's batch=1
        (single-token inference) never engages. DenseWeights fixes both;
        verified 19x-32x faster than the best ZeroSkipWeights kernel at
        these exact shapes, batch=1, native pybind-free
        (benchmarks/cpp-native-kernels/bench_gemm_dense.cpp). ZeroSkipWeights
        is kept and still exercised by its own test/build-validation, but is
        no longer the kernel this phase measures.
        """
        print("\n" + "=" * 80)
        print("PHASE 4: Neural Network Workload Patterns")
        print("=" * 80)

        try:
            import ternary_zero_skip_gemm as zsg
        except ImportError:
            print("\nternary_zero_skip_gemm not built -- skipping Phase 4")
            print("Build with: python build/build_zero_skip_gemm.py")
            self.results['phase4_neural_workload_patterns'] = {
                'error': 'ternary_zero_skip_gemm not available'
            }
            return

        # Common layer sizes in neural networks
        configs = [
            ("Small MLP", 512, 512),
            ("Medium Layer", 2048, 2048),
            ("Large Layer", 4096, 4096),
            ("Attention Head", 8192, 1024),
        ]

        batch = 1  # single-token inference, matches this phase's original intent
        rng = np.random.default_rng(42)
        results = []

        for name, M, N in configs:
            print(f"\n{name} ({M}x{N}):")

            # Ternary weights, shape (M, N): M output neurons, N input features
            weights = rng.integers(-1, 2, (M, N)).astype(np.int8)
            weights_f32 = weights.astype(np.float32)
            B = np.ascontiguousarray(weights.T)  # (N, M), zsg convention: C = A @ B
            inp = rng.standard_normal((batch, N)).astype(np.float32)

            # Precompute the packed dense structure once -- see docstring
            # on why this is excluded from the timed loop.
            zw = zsg.DenseWeights(B)
            sparsity = 1.0 - np.count_nonzero(weights) / weights.size

            # Warmup by wall-clock duration, not a fixed call count.
            #
            # Found 2026-08-20 while validating the DenseWeights switch above:
            # a fixed 3-call warmup was adequate for the old ZeroSkipWeights
            # kernel (~1-5ms/call -> several ms of warmup) but leaves this
            # machine's CPU cold for DenseWeights (~10-30us/call at this
            # scale -> ~0.0001ms of warmup). Reproduced directly: repeated
            # fresh-process runs of the identical shape/code with the old
            # 3-call warmup gave call times ranging 7.6us-267us (>30x) for
            # Small MLP, tracking this machine's already-documented
            # powersave-governor DVFS ramp-up (CV_SPIKE_ROOT_CAUSE.md), not
            # noise in the kernel itself. A fixed wall-clock warmup budget
            # gives the CPU time to reach a stable frequency regardless of
            # how fast the thing being warmed up is.
            warmup_deadline = time.perf_counter() + 0.05  # 50ms
            while time.perf_counter() < warmup_deadline:
                _ = zw.gemm(inp)
                _ = inp @ weights_f32.T

            # Interleaved rep-by-rep sampling (this project's own
            # `interleaved_timing` requirement): alternating ternary/NumPy
            # calls means both see the same clock/thermal drift pattern,
            # rather than two back-to-back blocks that can land on
            # different points of a drift curve. Median + CV via the
            # shared compute_timing_statistics() (gap #8's statistics
            # engine, already used by bench_fair_baseline.py /
            # bench_simd_fusion_ops.py) instead of one no-variance block.
            n_samples = 200
            tern_times_ns, np_times_ns = [], []
            for _ in range(n_samples):
                t0 = time.perf_counter_ns()
                out_tern = zw.gemm(inp)
                tern_times_ns.append(time.perf_counter_ns() - t0)

                t0 = time.perf_counter_ns()
                out_np = inp @ weights_f32.T
                np_times_ns.append(time.perf_counter_ns() - t0)

            tern_stats = compute_timing_statistics(tern_times_ns)
            np_stats = compute_timing_statistics(np_times_ns)
            ternary_time = tern_stats['median']
            numpy_time = np_stats['median']

            # Robustness check uses p90/median, not compute_timing_statistics()'s
            # mean/stdev-based CV. Found 2026-08-20: at this kernel's speed
            # (single-digit-to-tens of microseconds/call), a single OS
            # scheduling hiccup among 200 samples (e.g. one 3.2ms outlier
            # against a rock-steady ~13.8us median, p90 13.99us) inflates
            # mean/stdev CV past 300% even though the median itself is
            # reproducible to within a few percent across repeated
            # fresh-process runs (verified: 4 runs, 2.09x-2.12x for this
            # exact shape). Mean/stdev CV is the wrong tool for a
            # heavy-tailed, median-robust distribution like this one; a
            # spread that a few rare outliers can't dominate is.
            tern_p90_ratio = np.percentile(tern_times_ns, 90) / ternary_time
            np_p90_ratio = np.percentile(np_times_ns, 90) / numpy_time
            is_stable = tern_p90_ratio < 2.0 and np_p90_ratio < 2.0

            max_err = float(np.max(np.abs(out_tern - out_np)))

            ops_count = M * N * batch  # multiply-accumulate operations
            ternary_gops = (ops_count / ternary_time) * 1e9 / 1e9
            numpy_gops = (ops_count / numpy_time) * 1e9 / 1e9
            speedup = numpy_time / ternary_time

            print(f"  Ternary (dense-packed GEMM, {sparsity:.1%} zeros): "
                  f"{ternary_time/1e6:>8.3f}ms, {ternary_gops:>8.2f} GOPS "
                  f"(median; mean-CV={tern_stats['cv']:.0f}% -- outlier-inflated, "
                  f"p90/median={tern_p90_ratio:.2f}x)")
            print(f"  NumPy:                                    "
                  f"{numpy_time/1e6:>8.3f}ms, {numpy_gops:>8.2f} GOPS "
                  f"(median; mean-CV={np_stats['cv']:.0f}%, p90/median={np_p90_ratio:.2f}x)")
            print(f"  Speedup: {speedup:.3f}x   (correctness max err: {max_err:.2e})"
                  f"{'' if is_stable else '  [WARN] p90/median >= 2x -- bulk of the distribution is spread out, not just rare outliers'}")

            results.append({
                'name': name,
                'shape': (M, N),
                'batch': batch,
                'sparsity_zero_fraction': sparsity,
                'ternary_ms': ternary_time / 1e6,
                'numpy_ms': numpy_time / 1e6,
                'ternary_gops': ternary_gops,
                'numpy_gops': numpy_gops,
                'speedup': speedup,
                'max_abs_error': max_err,
                'ternary_cv_percent': tern_stats['cv'],
                'numpy_cv_percent': np_stats['cv'],
                'ternary_p90_median_ratio': tern_p90_ratio,
                'numpy_p90_median_ratio': np_p90_ratio,
                'is_stable': is_stable,
            })

        self.results['phase4_neural_workload_patterns'] = results

        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        max_err_overall = max(r['max_abs_error'] for r in results)
        any_unstable = any(not r['is_stable'] for r in results)

        print("\n" + "-" * 80)
        print("Phase 4 Summary:")
        print(f"  Average matmul speedup (compiled dense-packed GEMM vs NumPy): {avg_speedup:.3f}x")
        print(f"  Max correctness error across all configs: {max_err_overall:.2e}")
        if any_unstable:
            print(f"  [WARN] one or more configs had p90/median >= 2x -- this average "
                  f"includes a cell where the bulk of the distribution is spread out "
                  f"(not just rare outliers), do not cite it as a clean number without re-running")
        print(f"  Verdict: {'✓ VIABLE FOR AI' if avg_speedup > 0.5 else '✗ TOO SLOW FOR AI'}"
              f"{' (UNSTABLE -- see warning above)' if any_unstable else ''}")
        print(f"  Note: batch={batch} (single-token decode) with ~33% random-ternary "
              f"sparsity; NumPy's BLAS is extremely well-optimized at this scale, and "
              f"real trained-model sparsity (~40%, see CLAUDE.md falsification notes) "
              f"or larger batches would likely change this ratio -- re-run with those "
              f"before citing a production number.")

    def phase5_model_quantization(self):
        """
        Phase 5: Real Model Quantization

        Analysis framework for quantizing real models to ternary.
        This would be the PROOF - if a ternary-quantized model maintains
        reasonable accuracy and runs faster, we have a product.
        """
        print("\n" + "=" * 80)
        print("PHASE 5: Model Quantization Analysis")
        print("=" * 80)

        print("\nQuantization Strategy:")
        print("  Simple threshold-based:")
        print("    Values > threshold  → +1")
        print("    Values < -threshold → -1")
        print("    Values in between   → 0")

        print("\nTarget Models for Testing:")
        models = [
            ("TinyLlama-1.1B", "1.1B parameters", "Chat model"),
            ("Phi-2", "2.7B parameters", "Small but capable"),
            ("Gemma-2B", "2B parameters", "Google small model"),
        ]

        for name, size, description in models:
            print(f"  • {name}: {size} - {description}")

        print("\nQuantization Metrics to Measure:")
        metrics = [
            "Perplexity degradation",
            "Accuracy on benchmark tasks",
            "Inference latency",
            "Memory footprint",
            "Throughput (tokens/sec)",
        ]

        for metric in metrics:
            print(f"  • {metric}")

        print("\nSuccess Criteria:")
        print("  ✓ Accuracy loss < 5% on benchmarks")
        print("  ✓ Inference latency < 2x original")
        print("  ✓ Memory footprint < 25% of FP16")
        print("  ✓ Maintains coherent text generation")

        self.results['phase5_model_quantization'] = {
            'status': 'Framework defined - requires actual model implementation',
            'target_models': [m[0] for m in models],
            'metrics': metrics,
            'note': 'Requires PyTorch/Transformers integration'
        }

        print("\n" + "-" * 80)
        print("Phase 5 Summary:")
        print("  Status: ⚠ FRAMEWORK READY - NEEDS IMPLEMENTATION")
        print("  Next steps:")
        print("    1. Implement quantize_to_ternary() function")
        print("    2. Test on TinyLlama-1.1B")
        print("    3. Measure accuracy and performance")
        print("    4. Compare with INT8/INT4 quantized versions")

    def phase6_power_consumption(self):
        """
        Phase 6: Power Consumption

        Runs a real energy-efficiency comparison via
        bench_power_efficiency.py's PowerConsumptionBenchmark, using
        whichever hardware power monitor is genuinely available on this
        machine (Intel RAPL, NVIDIA nvidia-smi, Windows PowerShell) --
        auto-detected the same way that script's own standalone CLI does.

        Wired up 2026-08-22 -- this phase previously only printed a
        static "framework" description with the literal note "Requires
        actual hardware power monitoring", despite
        bench_power_efficiency.py already containing real, working
        monitor code. Fixed the same day, in that file: `IntelRAPLMonitor
        .is_available()` checked only that the RAPL directory existed,
        not that its energy_uj counter file was actually *readable* --
        that file is root-only by default on a stock Linux install (the
        common case for an unprivileged user, confirmed concretely on
        this machine: directory present, `energy_uj` permission denied),
        so the old check would have silently reported "available" and
        then measured 0.0 Joules for everything, behind one easy-to-miss
        warning per call. If no real hardware monitor is available, this
        phase falls through to `MockPowerMonitor` and says so loudly --
        it never reports a simulated number as if it were a real one.
        """
        print("\n" + "=" * 80)
        print("PHASE 6: Power Consumption")
        print("=" * 80)

        try:
            from bench_power_efficiency import PowerConsumptionBenchmark
        except ImportError as e:
            print(f"\nbench_power_efficiency.py not importable ({e}) -- skipping Phase 6")
            self.results['phase6_power_consumption'] = {'error': str(e)}
            return

        pb = PowerConsumptionBenchmark(platform='auto')
        monitor_name = type(pb.monitor).__name__
        is_real_hw = monitor_name != 'MockPowerMonitor'

        size = 1_000_000
        rng = np.random.default_rng(42)
        a_tern = rng.integers(0, 3, size, dtype=np.uint8)
        b_tern = rng.integers(0, 3, size, dtype=np.uint8)
        a_np = rng.integers(-1, 2, size, dtype=np.int8)
        b_np = rng.integers(-1, 2, size, dtype=np.int8)

        # Shorter than bench_power_efficiency.py's own 10s default --
        # this phase is one of six in a full suite run, not a dedicated
        # power-measurement session.
        duration = 3.0

        tern_result = None
        if tc is not None:
            tern_result = pb.benchmark_operation(
                "Ternary Addition", lambda: tc.tadd(a_tern, b_tern), duration_sec=duration
            )
        np_result = pb.benchmark_operation(
            "NumPy INT8 Addition", lambda: np.add(a_np, b_np, dtype=np.int8), duration_sec=duration
        )

        efficiency_advantage = None
        if tern_result is not None and np_result.get('ops_per_joule', 0) > 0:
            efficiency_advantage = tern_result['ops_per_joule'] / np_result['ops_per_joule']

        self.results['phase6_power_consumption'] = {
            'monitor_type': monitor_name,
            'is_real_hardware_measurement': is_real_hw,
            'ternary': tern_result,
            'numpy': np_result,
            'efficiency_advantage': efficiency_advantage,
        }

        print("\n" + "-" * 80)
        print("Phase 6 Summary:")
        if not is_real_hw:
            print("  [WARN] No real hardware power monitor available on this machine "
                  "(checked: Windows PowerShell, Intel RAPL, NVIDIA nvidia-smi).")
            print("  The numbers above are SIMULATED (MockPowerMonitor, fixed ~50W draw), "
                  "NOT a real measurement.")
            print("  Re-run on a machine with RAPL/nvidia-smi access (or with the needed "
                  "permissions) for genuine numbers.")
            print("  Status: ⚠ NO HARDWARE MONITOR -- results are simulated, not citable")
        else:
            print(f"  Real hardware measurement via {monitor_name}")
            if efficiency_advantage is not None:
                print(f"  Ternary power efficiency: {efficiency_advantage:.2f}x ops/Joule vs NumPy")
                if efficiency_advantage > 1.5:
                    print("  Status: ✓ SIGNIFICANT POWER ADVANTAGE")
                elif efficiency_advantage > 1.0:
                    print("  Status: ⚠ MODEST POWER ADVANTAGE")
                else:
                    print("  Status: ✗ NO POWER ADVANTAGE")
            else:
                print("  Status: ⚠ MEASUREMENT INCOMPLETE (missing engine or zero-energy reading)")

    def save_results(self):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.output_dir, f"competitive_results_{timestamp}.json")

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Results saved to {filename}")
        return filename

    def print_summary(self):
        """Print comprehensive summary of all benchmark phases"""
        print("\n" + "=" * 80)
        print("COMPETITIVE BENCHMARK COMPREHENSIVE SUMMARY")
        print("=" * 80)

        print("\nKey Findings:\n")

        # Phase 1
        phase1 = self.results['phase1_arithmetic_comparison']
        if phase1:
            avg_add_speedup = sum(phase1['add_speedup']) / len(phase1['add_speedup'])
            print(f"[1] Arithmetic vs NumPy:          {avg_add_speedup:>6.2f}x average speedup")

        # Phase 2
        print(f"[2] Memory Efficiency:            4.00x smaller than INT8")

        # Phase 3
        phase3 = self.results['phase3_throughput_equivalent_bitwidth']
        if 'ternary_gops' in phase3:
            print(f"[3] Throughput @ 1GB:             {phase3['ternary_gops']:>6.2f} GOPS")

        # Phase 4
        phase4 = self.results['phase4_neural_workload_patterns']
        if phase4:
            avg_nn_speedup = sum(r['speedup'] for r in phase4) / len(phase4)
            print(f"[4] Neural Network Patterns:      {avg_nn_speedup:>6.2f}x matmul speedup")

        # Phase 5 & 6
        print(f"[5] Model Quantization:           Framework ready")
        print(f"[6] Power Consumption:            Framework ready")

        print("\n" + "=" * 80)
        print("COMMERCIAL VIABILITY ASSESSMENT")
        print("=" * 80)

        checklist = [
            ("Memory efficiency at same capacity", True, "4x smaller than INT8"),
            ("Throughput at equivalent bit-width", True, "Measured baseline"),
            ("Inference latency in real models", False, "Needs implementation"),
            ("Power consumption on edge devices", False, "Needs hardware"),
            ("Accuracy retention after quantization", False, "Needs model testing"),
        ]

        completed = sum(1 for _, done, _ in checklist if done)
        total = len(checklist)

        for item, done, note in checklist:
            status = "✓" if done else "⚠"
            print(f"  {status} {item:<40} {note}")

        print(f"\nProgress: {completed}/{total} criteria validated")

        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)

        if completed < 3:
            print("  Priority: Complete Phase 5 (Model Quantization)")
            print("    1. Implement ternary quantization for TinyLlama")
            print("    2. Measure accuracy degradation")
            print("    3. Compare inference speed with INT8")
            print("\n  By Week 4, we'll know if we have a business or a hobby project.")
        else:
            print("  Looking good! Continue with:")
            print("    1. Model quantization testing")
            print("    2. Power consumption measurements")
            print("    3. Production deployment validation")


def main():
    """Main entry point for competitive benchmarking"""
    parser = argparse.ArgumentParser(
        description='Ternary Engine Competitive Benchmark Suite'
    )
    parser.add_argument(
        '--phase',
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help='Run specific phase only (1-6)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all benchmark phases'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output directory for results (default: benchmarks/results/competitive/)'
    )

    args = parser.parse_args()

    benchmark = CompetitiveBenchmark(output_dir=args.output)

    if args.all or not args.phase:
        benchmark.run_all()
    else:
        # Run specific phase
        phase_methods = {
            1: benchmark.phase1_benchmark_vs_numpy,
            2: benchmark.phase2_benchmark_memory_efficiency,
            3: benchmark.phase3_benchmark_equivalent_bitwidth,
            4: benchmark.phase4_benchmark_nn_patterns,
            5: benchmark.phase5_model_quantization,
            6: benchmark.phase6_power_consumption,
        }

        print(f"Running Phase {args.phase} only...\n")
        phase_methods[args.phase]()
        benchmark.save_results()
        benchmark.print_summary()


if __name__ == "__main__":
    main()
