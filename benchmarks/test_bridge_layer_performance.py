"""
test_bridge_layer_performance.py - Tests to validate the performance of the Bridge Layer

Philosophy: The Bridge Layer claims to be fast. Let's prove it.

USAGE:
    python benchmarks/test_bridge_layer_performance.py
"""

import sys
import time
import json
import platform
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable
import numpy as np

# Agregar src al path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@dataclass
class BenchmarkResult:
    """Resultado de un benchmark individual."""
    name: str
    size: int
    mean_ns: float
    std_ns: float
    min_ns: float
    max_ns: float
    p50_ns: float
    p99_ns: float
    ops_per_second: float
    samples: int


@dataclass
class ComparisonResult:
    """Resultado de comparacion entre implementaciones."""
    ternary: BenchmarkResult
    baseline: BenchmarkResult
    speedup: float
    winner: str
    verdict: str


@dataclass
class PerformanceTestResult:
    """Resultado de un test de performance."""
    test_name: str
    passed: bool
    expected: str
    actual: str
    details: Dict
    timestamp: str


class PerformanceBenchmark:
    """Framework de benchmarking de performance."""

    def __init__(self):
        self.results: List[PerformanceTestResult] = []
        self.timestamp = datetime.now().isoformat()

    def benchmark(
        self,
        func: Callable,
        size: int,
        repetitions: int = 100,
        warmup: int = 10,
        op_type: str = 'binary'
    ) -> BenchmarkResult:
        """
        Ejecuta benchmark con estadisticas completas.
        """
        # Warmup (descartado)
        for _ in range(warmup):
            if op_type == 'unary':
                a = np.random.randint(-1, 2, size, dtype=np.int8)
                _ = func(a)
            else:
                a = np.random.randint(-1, 2, size, dtype=np.int8)
                b = np.random.randint(-1, 2, size, dtype=np.int8)
                _ = func(a, b)


        times = []
        for _ in range(repetitions):
            # Datos FRESCOS cada iteracion
            if op_type == 'unary':
                a = np.random.randint(-1, 2, size, dtype=np.int8)
                start = time.perf_counter_ns()
                result = func(a)
            else:
                a = np.random.randint(-1, 2, size, dtype=np.int8)
                b = np.random.randint(-1, 2, size, dtype=np.int8)
                start = time.perf_counter_ns()
                result = func(a, b)

            # Forzar materializacion
            _ = result[0] if hasattr(result, '__getitem__') else result
            end = time.perf_counter_ns()

            times.append(end - start)

        times = np.array(times)

        return BenchmarkResult(
            name=func.__name__ if hasattr(func, '__name__') else str(func),
            size=size,
            mean_ns=float(np.mean(times)),
            std_ns=float(np.std(times)),
            min_ns=float(np.min(times)),
            max_ns=float(np.max(times)),
            p50_ns=float(np.percentile(times, 50)),
            p99_ns=float(np.percentile(times, 99)),
            ops_per_second=size / (np.mean(times) / 1e9),
            samples=repetitions
        )

    def compare(
        self,
        ternary_func: Callable,
        baseline_func: Callable,
        size: int,
        ternary_name: str = "ternary",
        baseline_name: str = "baseline",
        op_type: str = 'binary'
    ) -> ComparisonResult:
        """Compara dos implementaciones."""
        ternary_result = self.benchmark(ternary_func, size, op_type=op_type)
        ternary_result.name = ternary_name

        baseline_result = self.benchmark(baseline_func, size, op_type=op_type)
        baseline_result.name = baseline_name

        speedup = baseline_result.mean_ns / ternary_result.mean_ns
        winner = "ternary" if speedup > 1.0 else "baseline"

        return ComparisonResult(
            ternary=ternary_result,
            baseline=baseline_result,
            speedup=speedup,
            winner=winner,
            verdict=f"{speedup:.2f}x {'FASTER' if speedup > 1 else 'SLOWER'}"
        )

    def record_test(
        self,
        test_name: str,
        passed: bool,
        expected: str,
        actual: str,
        details: Dict = None
    ):
        """Registra resultado de un test."""
        result = PerformanceTestResult(
            test_name=test_name,
            passed=passed,
            expected=expected,
            actual=actual,
            details=details or {},
            timestamp=datetime.now().isoformat()
        )
        self.results.append(result)
        return result

    def save_results(self, output_dir: Path = None):
        """Guarda resultados a JSON."""
        if output_dir is None:
            output_dir = ROOT / "benchmarks" / "results" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir.mkdir(parents=True, exist_ok=True)

        system_info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "timestamp": self.timestamp,
        }

        results_data = {
            "system": system_info,
            "tests": [asdict(r) for r in self.results],
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.passed),
                "failed": sum(1 for r in self.results if not r.passed),
            }
        }

        output_file = output_dir / "bridge_layer_performance_results.json"
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"\nResultados guardados en: {output_file}")
        return output_file


# =============================================================================
# IMPLEMENTACIONES DE REFERENCIA (BASELINES)
# =============================================================================

def numpy_int8_add_saturated(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Suma saturada INT8 - baseline NumPy."""
    return np.clip(a.astype(np.int16) + b.astype(np.int16), -1, 1).astype(np.int8)

def numpy_fused_tnot_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Baseline para fused_tnot_tadd_int8."""
    # Corrected logic: tnot(a + b)
    sum_val = np.clip(a.astype(np.int16) + b.astype(np.int16), -1, 1).astype(np.int8)
    return (sum_val * -1).astype(np.int8)


# =============================================================================
# TESTS DE PERFORMANCE
# =============================================================================

def run_bridge_layer_performance_tests():
    """Ejecuta todos los tests de performance del Bridge Layer."""

    bench = PerformanceBenchmark()
    print("=" * 70)
    print("TESTS DE PERFORMANCE - Bridge Layer")
    print("=" * 70)
    print(f"Plataforma: {platform.platform()}")
    print(f"Procesador: {platform.processor()}")
    print("=" * 70)

    # Intentar cargar modulo ternario
    ternary_available = False
    try:
        import ternary_simd_engine as te
        ternary_available = True
        print("\n[OK] ternary_simd_engine cargado")

        def ternary_add_bridge(a, b):
            return te.tadd_int8(a, b)

        def ternary_fused_op_bridge(a, b):
            return te.fused_tnot_tadd_int8(a, b)

    except ImportError as e:
        print(f"\n[CRITICAL] ternary_simd_engine no disponible: {e}")
        return 1

    # =========================================================================
    # TEST 1: Crossover Point (tadd_int8 vs numpy)
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 1: Crossover Point (tadd_int8)")
    print("-" * 70)

    sizes = [64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]
    crossover_point = None
    crossover_results = []

    for size in sizes:
        comparison = bench.compare(
            ternary_add_bridge, numpy_int8_add_saturated, size,
            "ternary_add_bridge", "numpy_int8_add"
        )
        crossover_results.append({
            'size': size,
            'speedup': comparison.speedup,
            'winner': comparison.winner
        })

        status = "WIN" if comparison.speedup > 1.0 else "LOSE"
        print(f"  Size {size:>10}: {comparison.speedup:.2f}x [{status}]")

        if comparison.speedup > 1.0 and crossover_point is None:
            crossover_point = size

    if crossover_point is not None:
        passed = crossover_point < 20000 # Expect a low crossover
        bench.record_test(
            "crossover_tadd_int8",
            passed=passed,
            expected="crossover < 20,000 elementos",
            actual=f"crossover = {crossover_point}",
            details={'crossover_results': crossover_results}
        )
        print(f"\n  Crossover encontrado en: {crossover_point}")
        print(f"  Resultado: {'PASS' if passed else 'FAIL'}")
    else:
        bench.record_test(
            "crossover_tadd_int8",
            passed=False,
            expected="crossover < 20,000 elementos",
            actual="No existe crossover en rango testeado",
            details={'crossover_results': crossover_results}
        )
        print(f"\n  FAIL: No se encontró crossover point para tadd_int8.")

    # =========================================================================
    # TEST 2: Speedup en Arrays Grandes (tadd_int8)
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 2: Speedup vs NumPy (tadd_int8, arrays grandes)")
    print("-" * 70)

    large_sizes = [100_000, 1_000_000, 10_000_000]
    speedup_results_add = []

    for size in large_sizes:
        comparison = bench.compare(
            ternary_add_bridge, numpy_int8_add_saturated, size,
            "ternary_add_bridge", "numpy_int8_add"
        )
        speedup_results_add.append({
            'size': size,
            'speedup': comparison.speedup,
            'ternary_gops': comparison.ternary.ops_per_second / 1e9,
            'baseline_gops': comparison.baseline.ops_per_second / 1e9,
        })

        print(f"  Size {size:>10}:")
        print(f"    Ternary:  {comparison.ternary.ops_per_second/1e9:.2f} Gops/s")
        print(f"    NumPy:    {comparison.baseline.ops_per_second/1e9:.2f} Gops/s")
        print(f"    Speedup:  {comparison.speedup:.2f}x")

    best_speedup_add = max(r['speedup'] for r in speedup_results_add)
    passed_add = best_speedup_add > 10.0 # Expect significant speedup

    bench.record_test(
        "speedup_tadd_int8_large_arrays",
        passed=passed_add,
        expected="speedup > 10.0x en arrays > 100K",
        actual=f"mejor speedup = {best_speedup_add:.2f}x",
        details={'speedup_results': speedup_results_add}
    )
    print(f"\n  Resultado: {'PASS' if passed_add else 'FAIL'}")

    # =========================================================================
    # TEST 3: Speedup FUSED OPERATION (fused_tnot_tadd_int8)
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 3: Speedup FUSED OP vs NumPy (fused_tnot_tadd_int8)")
    print("-" * 70)

    speedup_results_fused = []

    for size in large_sizes:
        comparison = bench.compare(
            ternary_fused_op_bridge, numpy_fused_tnot_add, size,
            "ternary_fused_op", "numpy_fused"
        )
        speedup_results_fused.append({
            'size': size,
            'speedup': comparison.speedup,
            'ternary_gops': comparison.ternary.ops_per_second / 1e9,
            'baseline_gops': comparison.baseline.ops_per_second / 1e9,
        })

        print(f"  Size {size:>10}:")
        print(f"    Ternary Fused: {comparison.ternary.ops_per_second/1e9:.2f} Gops/s")
        print(f"    NumPy Fused:   {comparison.baseline.ops_per_second/1e9:.2f} Gops/s")
        print(f"    Speedup:       {comparison.speedup:.2f}x")

    best_speedup_fused = max(r['speedup'] for r in speedup_results_fused)
    passed_fused = best_speedup_fused > 20.0 # Expect even higher speedup for fused ops

    bench.record_test(
        "speedup_fused_op_large_arrays",
        passed=passed_fused,
        expected="speedup > 20.0x para op fusionada",
        actual=f"mejor speedup = {best_speedup_fused:.2f}x",
        details={'speedup_results': speedup_results_fused}
    )
    print(f"\n  Resultado: {'PASS' if passed_fused else 'FAIL'}")


    # =========================================================================
    # TEST 4: Consistencia de Resultados
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 4: Correctitud (ternary == baseline)")
    print("-" * 70)

    test_sizes = [100, 1000, 10000]
    correctness_passed = True

    for size in test_sizes:
        a = np.random.randint(-1, 2, size, dtype=np.int8)
        b = np.random.randint(-1, 2, size, dtype=np.int8)

        result_ternary_add = ternary_add_bridge(a, b)
        result_baseline_add = numpy_int8_add_saturated(a, b)
        matches_add = np.array_equal(result_ternary_add, result_baseline_add)
        print(f"  tadd_int8 Size {size}: {'MATCH' if matches_add else 'MISMATCH'}")

        result_ternary_fused = ternary_fused_op_bridge(a, b)
        result_baseline_fused = numpy_fused_tnot_add(a, b)
        matches_fused = np.array_equal(result_ternary_fused, result_baseline_fused)
        print(f"  fused_op Size {size}: {'MATCH' if matches_fused else 'MISMATCH'}")

        correctness_passed = correctness_passed and matches_add and matches_fused

    bench.record_test(
        "correctness_bridge_layer",
        passed=correctness_passed,
        expected="100% match con baseline",
        actual="PASS" if correctness_passed else "FAIL",
        details={}
    )
    print(f"\n  Resultado: {'PASS' if correctness_passed else 'FAIL'}")


    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESUMEN DE PERFORMANCE - BRIDGE LAYER")
    print("=" * 70)

    passed_count = sum(1 for r in bench.results if r.passed)
    total_count = len(bench.results)

    for result in bench.results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.test_name}: {result.actual}")

    print("-" * 70)
    print(f"  Total: {passed_count}/{total_count} tests pasados")

    if passed_count == total_count:
        verdict = "VALIDATED: El Bridge Layer rinde como se esperaba."
    else:
        verdict = f"FAIL: Solo {passed_count}/{total_count} - investigar fallos."

    print(f"\n  VEREDICTO: {verdict}")
    print("=" * 70)

    # Guardar resultados
    output_file = bench.save_results()

    return 0 if passed_count == total_count else 1


if __name__ == "__main__":
    sys.exit(run_bridge_layer_performance_tests())
