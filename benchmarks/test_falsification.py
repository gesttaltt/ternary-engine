"""
test_falsification.py - Tests que DEBEN pasar o el proyecto no tiene valor

Filosofia: Si no podemos refutarlo, no podemos probarlo.

USO:
    python benchmarks/test_falsification.py
    pytest benchmarks/test_falsification.py -v

RESULTADO:
    - PASS: El proyecto tiene valor practico demostrable
    - FAIL: Investigar o reconsiderar premisas
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
class FalsificationResult:
    """Resultado de un test de falsificacion."""
    test_name: str
    passed: bool
    expected: str
    actual: str
    details: Dict
    timestamp: str


class SkepticalBenchmark:
    """Framework de benchmarking esceptico."""

    def __init__(self):
        self.results: List[FalsificationResult] = []
        self.timestamp = datetime.now().isoformat()

    def benchmark(
        self,
        func: Callable,
        size: int,
        repetitions: int = 100,
        warmup: int = 10
    ) -> BenchmarkResult:
        """
        Ejecuta benchmark con estadisticas completas.
        NO usa cache caliente artificialmente.
        """
        # Warmup (descartado)
        for _ in range(warmup):
            a = np.random.randint(-1, 2, size, dtype=np.int8)
            b = np.random.randint(-1, 2, size, dtype=np.int8)
            _ = func(a, b)

        times = []
        for _ in range(repetitions):
            # Datos FRESCOS cada iteracion
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
        baseline_name: str = "baseline"
    ) -> ComparisonResult:
        """Compara dos implementaciones."""
        ternary_result = self.benchmark(ternary_func, size)
        ternary_result.name = ternary_name

        baseline_result = self.benchmark(baseline_func, size)
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

    def record_falsification(
        self,
        test_name: str,
        passed: bool,
        expected: str,
        actual: str,
        details: Dict = None
    ):
        """Registra resultado de test de falsificacion."""
        result = FalsificationResult(
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

        # Metadata del sistema
        system_info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "timestamp": self.timestamp,
        }

        # Resultados
        results_data = {
            "system": system_info,
            "tests": [asdict(r) for r in self.results],
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.passed),
                "failed": sum(1 for r in self.results if not r.passed),
            }
        }

        output_file = output_dir / "falsification_results.json"
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


def numpy_int8_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiplicacion INT8 - baseline NumPy."""
    return (a * b).astype(np.int8)


def numpy_int8_min(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimo INT8 - baseline NumPy."""
    return np.minimum(a, b)


def numpy_int8_max(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Maximo INT8 - baseline NumPy."""
    return np.maximum(a, b)


# =============================================================================
# TESTS DE FALSIFICACION
# =============================================================================

def run_falsification_tests():
    """Ejecuta todos los tests de falsificacion."""

    bench = SkepticalBenchmark()
    print("=" * 70)
    print("TESTS DE FALSIFICACION - Ternary Engine")
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

        def ternary_add(a, b):
            # Convertir a uint8 (formato 2-bit: 0=-1, 1=0, 2=+1)
            a_uint8 = (a + 1).astype(np.uint8)
            b_uint8 = (b + 1).astype(np.uint8)
            result = te.tadd(a_uint8, b_uint8)
            return result.astype(np.int8) - 1

        def ternary_mul(a, b):
            a_uint8 = (a + 1).astype(np.uint8)
            b_uint8 = (b + 1).astype(np.uint8)
            result = te.tmul(a_uint8, b_uint8)
            return result.astype(np.int8) - 1

    except ImportError as e:
        print(f"\n[WARN] ternary_simd_engine no disponible: {e}")
        print("[WARN] Ejecutando solo tests de baseline para establecer referencia")

        # Fallback: usar NumPy como "ternary" para testing del framework
        def ternary_add(a, b):
            return numpy_int8_add_saturated(a, b)

        def ternary_mul(a, b):
            return numpy_int8_mul(a, b)

    # =========================================================================
    # TEST 1: Crossover Point
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 1: Busqueda de Crossover Point")
    print("-" * 70)

    sizes = [64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]
    crossover_point = None
    crossover_results = []

    for size in sizes:
        comparison = bench.compare(
            ternary_add, numpy_int8_add_saturated, size,
            "ternary_add", "numpy_int8_add"
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

    # Evaluar resultado
    if crossover_point is not None:
        passed = crossover_point < 100_000
        bench.record_falsification(
            "crossover_exists",
            passed=passed,
            expected="crossover < 100,000 elementos",
            actual=f"crossover = {crossover_point}",
            details={'crossover_results': crossover_results}
        )
        print(f"\n  Crossover encontrado en: {crossover_point}")
        print(f"  Resultado: {'PASS' if passed else 'FAIL'}")
    else:
        bench.record_falsification(
            "crossover_exists",
            passed=False,
            expected="crossover < 100,000 elementos",
            actual="No existe crossover en rango testeado",
            details={'crossover_results': crossover_results}
        )
        print(f"\n  FALSIFICADO: No existe crossover point")

    # =========================================================================
    # TEST 2: Speedup en Arrays Grandes
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 2: Speedup vs NumPy INT8 (arrays grandes)")
    print("-" * 70)

    large_sizes = [100_000, 1_000_000]
    speedup_results = []

    for size in large_sizes:
        comparison = bench.compare(
            ternary_add, numpy_int8_add_saturated, size,
            "ternary_add", "numpy_int8_add"
        )
        speedup_results.append({
            'size': size,
            'speedup': comparison.speedup,
            'ternary_ops_s': comparison.ternary.ops_per_second,
            'baseline_ops_s': comparison.baseline.ops_per_second,
        })

        print(f"  Size {size:>10}:")
        print(f"    Ternary:  {comparison.ternary.ops_per_second/1e6:.1f} Mops/s")
        print(f"    NumPy:    {comparison.baseline.ops_per_second/1e6:.1f} Mops/s")
        print(f"    Speedup:  {comparison.speedup:.2f}x")

    # Evaluar: debe ganar en al menos un tamano grande
    best_speedup = max(r['speedup'] for r in speedup_results)
    passed = best_speedup > 1.5

    bench.record_falsification(
        "speedup_large_arrays",
        passed=passed,
        expected="speedup > 1.5x en arrays > 100K",
        actual=f"mejor speedup = {best_speedup:.2f}x",
        details={'speedup_results': speedup_results}
    )
    print(f"\n  Resultado: {'PASS' if passed else 'FAIL'}")

    # =========================================================================
    # TEST 3: Consistencia de Resultados
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 3: Correctitud (ternary == baseline)")
    print("-" * 70)

    test_sizes = [100, 1000, 10000]
    correctness_passed = True

    for size in test_sizes:
        a = np.random.randint(-1, 2, size, dtype=np.int8)
        b = np.random.randint(-1, 2, size, dtype=np.int8)

        result_ternary = ternary_add(a, b)
        result_baseline = numpy_int8_add_saturated(a, b)

        matches = np.array_equal(result_ternary, result_baseline)
        correctness_passed = correctness_passed and matches

        print(f"  Size {size}: {'MATCH' if matches else 'MISMATCH'}")

    bench.record_falsification(
        "correctness",
        passed=correctness_passed,
        expected="100% match con baseline",
        actual="PASS" if correctness_passed else "FAIL",
        details={}
    )
    print(f"\n  Resultado: {'PASS' if correctness_passed else 'FAIL'}")

    # =========================================================================
    # TEST 4: Varianza Aceptable
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 4: Estabilidad (baja varianza)")
    print("-" * 70)

    result = bench.benchmark(ternary_add, 100_000, repetitions=200)
    cv = result.std_ns / result.mean_ns  # Coeficiente de variacion

    passed = cv < 0.3  # Menos de 30% variacion
    print(f"  Mean:   {result.mean_ns/1e6:.2f} ms")
    print(f"  Std:    {result.std_ns/1e6:.2f} ms")
    print(f"  CV:     {cv:.2%}")
    print(f"  P99:    {result.p99_ns/1e6:.2f} ms")

    bench.record_falsification(
        "variance_acceptable",
        passed=passed,
        expected="CV < 30%",
        actual=f"CV = {cv:.2%}",
        details={'mean_ns': result.mean_ns, 'std_ns': result.std_ns}
    )
    print(f"\n  Resultado: {'PASS' if passed else 'FAIL'}")

    # =========================================================================
    # TEST 5: Overhead de Operaciones Multiples
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 5: Overhead de pipeline (pack + compute + unpack)")
    print("-" * 70)

    # Simular pipeline completo vs operacion aislada
    size = 100_000

    # Solo kernel
    a = np.random.randint(-1, 2, size, dtype=np.int8)
    b = np.random.randint(-1, 2, size, dtype=np.int8)

    times_kernel = []
    times_pipeline = []

    for _ in range(100):
        # Kernel aislado
        start = time.perf_counter_ns()
        _ = ternary_add(a, b)
        times_kernel.append(time.perf_counter_ns() - start)

        # Pipeline simulado (conversion + kernel + conversion)
        start = time.perf_counter_ns()
        a_converted = a.astype(np.float32)  # Simular conversion
        b_converted = b.astype(np.float32)
        result = ternary_add(a.astype(np.int8), b.astype(np.int8))
        _ = result.astype(np.float32)  # Simular conversion salida
        times_pipeline.append(time.perf_counter_ns() - start)

    kernel_mean = np.mean(times_kernel)
    pipeline_mean = np.mean(times_pipeline)
    overhead = pipeline_mean / kernel_mean

    passed = overhead < 2.0  # Menos de 2x overhead

    print(f"  Kernel solo:     {kernel_mean/1e6:.2f} ms")
    print(f"  Pipeline:        {pipeline_mean/1e6:.2f} ms")
    print(f"  Overhead:        {overhead:.2f}x")

    bench.record_falsification(
        "pipeline_overhead",
        passed=passed,
        expected="overhead < 2.0x",
        actual=f"overhead = {overhead:.2f}x",
        details={'kernel_ms': kernel_mean/1e6, 'pipeline_ms': pipeline_mean/1e6}
    )
    print(f"\n  Resultado: {'PASS' if passed else 'FAIL'}")

    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESUMEN DE FALSIFICACION")
    print("=" * 70)

    passed_count = sum(1 for r in bench.results if r.passed)
    total_count = len(bench.results)

    for result in bench.results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.test_name}: {result.actual}")

    print("-" * 70)
    print(f"  Total: {passed_count}/{total_count} tests pasados")

    if passed_count == total_count:
        verdict = "VALIDATED: El proyecto tiene valor practico demostrable"
    elif passed_count >= total_count * 0.6:
        verdict = f"PARTIAL: {passed_count}/{total_count} criterios - investigar gaps"
    else:
        verdict = f"FALSIFIED: Solo {passed_count}/{total_count} - reconsiderar premisas"

    print(f"\n  VEREDICTO: {verdict}")
    print("=" * 70)

    # Guardar resultados
    output_file = bench.save_results()

    # Retornar codigo de salida
    return 0 if passed_count == total_count else 1


if __name__ == "__main__":
    sys.exit(run_falsification_tests())
