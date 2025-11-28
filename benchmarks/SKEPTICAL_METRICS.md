# Sistema de Metricas Escepticas

**Doc-Type:** Framework de Validacion · Version 1.0 · 2025-11-28
**Filosofia:** Si no podemos refutarlo, no podemos probarlo.

---

## Principio Fundamental

> "Los 35 Gops/s no significan nada si no sabemos contra qué, en qué contexto, y a qué costo."

Este documento define métricas que **pueden falsificar** las claims del proyecto.
Si ternario no supera estos tests, el proyecto no tiene valor práctico.

---

## Anti-Patrones a Evitar

| Trampa | Ejemplo | Por qué es mentira |
|:-------|:--------|:-------------------|
| Cherry-picking | "35 Gops/s pico" | ¿Promedio? ¿P99? ¿Con qué input? |
| Strawman baseline | "8000x vs Python" | Python no es competencia real |
| Kernel aislado | "tadd_simd es rápido" | ¿Y el pack/unpack? ¿Y memory allocation? |
| Hardware favorable | "En mi i9..." | ¿Y en edge ARM? ¿En el target real? |
| Ignorar energía | "Más ops/segundo" | ¿A qué costo en watts? |
| Tamaño conveniente | "Con arrays de 1M" | ¿Y con 1K? ¿Y con 1B? |

---

## Nivel 1: Metricas de Verdad Absoluta

### 1.1 Throughput Real (No Vanity)

```
METRIC: effective_ops_per_second

Definicion:
  (operaciones_utiles_completadas) / (tiempo_total_wallclock)

Incluye:
  - Allocation de memoria
  - Conversion de formato (pack/unpack)
  - Operacion aritmetica
  - Verificacion de resultado (overhead real)

NO incluye:
  - Solo el kernel SIMD aislado
  - Warm cache artificialmente
  - Inputs pre-alineados convenientemente
```

**Test obligatorio:**
```python
def measure_effective_throughput(operation, size, repetitions=100):
    """
    Mide throughput REAL incluyendo todo el overhead.
    """
    results = []

    for _ in range(repetitions):
        # Genera datos FRESCOS cada vez (no reusar buffers)
        a = generate_random_ternary_array(size)
        b = generate_random_ternary_array(size)

        # Mide TODO el ciclo
        start = time.perf_counter_ns()

        # Incluir conversion si viene de otro formato
        a_packed = pack_dense243(a) if needs_packing else a
        b_packed = pack_dense243(b) if needs_packing else b

        # Operacion
        result = operation(a_packed, b_packed)

        # Forzar materializacion (no lazy)
        _ = result[0]

        end = time.perf_counter_ns()

        results.append(size / ((end - start) / 1e9))

    return {
        'mean': np.mean(results),
        'std': np.std(results),
        'p50': np.percentile(results, 50),
        'p99': np.percentile(results, 99),
        'min': np.min(results),  # El PEOR caso importa
    }
```

### 1.2 Comparacion Justa (No Strawman)

**Baselines obligatorios:**

| Baseline | Implementacion | Por que |
|:---------|:---------------|:--------|
| NumPy INT8 | `np.add(a, b)` con clip | Referencia industria |
| NumPy INT8 + Numba | JIT-compiled | Mejor caso NumPy |
| PyTorch INT8 | `torch.add` quantized | Framework ML real |
| ONNX Runtime INT8 | Optimizado para inference | Produccion real |
| llama.cpp Q2_K | 2-bit quantization | Competencia directa |

**Codigo de comparacion:**
```python
def compare_against_baselines(ternary_op, size):
    """
    Compara contra TODOS los baselines relevantes.
    Si ternario pierde contra alguno, REPORTARLO.
    """
    results = {}

    # Generar datos equivalentes para cada formato
    data_ternary = generate_ternary(size)
    data_int8 = ternary_to_int8(data_ternary)
    data_torch = torch.from_numpy(data_int8)

    # Ternary Engine
    results['ternary_engine'] = benchmark(
        lambda: ternary_op(data_ternary, data_ternary)
    )

    # NumPy INT8 (con saturacion equivalente)
    results['numpy_int8'] = benchmark(
        lambda: np.clip(np.add(data_int8, data_int8, dtype=np.int16), -1, 1).astype(np.int8)
    )

    # PyTorch quantized
    results['torch_int8'] = benchmark(
        lambda: torch.clamp(data_torch + data_torch, -1, 1)
    )

    # CRITICO: Reportar si perdemos
    ternary_speed = results['ternary_engine']['mean']
    for name, baseline in results.items():
        if name != 'ternary_engine':
            ratio = ternary_speed / baseline['mean']
            status = "WIN" if ratio > 1.0 else "LOSE"
            print(f"vs {name}: {ratio:.2f}x [{status}]")

    return results
```

### 1.3 Eficiencia Energetica (La Metrica Real)

```
METRIC: operations_per_joule

Definicion:
  (operaciones_completadas) / (energia_consumida_joules)

Metodos de medicion:
  1. Intel RAPL (software, ~1ms resolucion)
  2. nvidia-smi (GPU)
  3. Medidor externo USB (ground truth)
```

**Implementacion RAPL:**
```python
import subprocess

def read_rapl_energy():
    """Lee energia del package CPU via RAPL (Linux)."""
    try:
        with open('/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj', 'r') as f:
            return int(f.read().strip())
    except:
        return None

def measure_energy_efficiency(operation, size, duration_seconds=5):
    """
    Mide ops/joule durante un periodo sostenido.
    """
    energy_start = read_rapl_energy()
    if energy_start is None:
        return {'error': 'RAPL not available - USE EXTERNAL METER'}

    ops_completed = 0
    start_time = time.time()

    # Preparar datos
    a = generate_ternary(size)
    b = generate_ternary(size)

    while time.time() - start_time < duration_seconds:
        operation(a, b)
        ops_completed += size

    energy_end = read_rapl_energy()
    energy_joules = (energy_end - energy_start) / 1e6

    return {
        'ops_per_joule': ops_completed / energy_joules,
        'total_joules': energy_joules,
        'total_ops': ops_completed,
        'watts_average': energy_joules / duration_seconds,
    }
```

**Windows (sin RAPL directo):**
```python
def measure_energy_windows():
    """
    Windows no expone RAPL facilmente.
    OPCIONES:
    1. HWiNFO64 + logging
    2. Intel Power Gadget API
    3. Medidor USB externo (RECOMENDADO para ground truth)
    """
    # TODO: Implementar con Intel Power Gadget
    raise NotImplementedError(
        "CRITICO: Sin medicion de energia, no podemos validar eficiencia. "
        "Adquirir medidor USB o usar Linux con RAPL."
    )
```

---

## Nivel 2: Analisis de Crossover

### 2.1 Encontrar el Punto de Equilibrio

```
PREGUNTA: A que tamaño de problema ternario supera a INT8?

Si no hay crossover point, el proyecto no tiene caso de uso.
```

**Test de crossover:**
```python
def find_crossover_point(ternary_op, baseline_op, size_range):
    """
    Encuentra donde ternario empieza a ganar (si es que lo hace).
    """
    crossover = None
    results = []

    for size in size_range:  # [64, 128, 256, 512, 1K, 4K, 16K, 64K, 256K, 1M, 4M, 16M]
        ternary_time = benchmark(ternary_op, size)['mean']
        baseline_time = benchmark(baseline_op, size)['mean']

        ratio = baseline_time / ternary_time  # >1 = ternario gana

        results.append({
            'size': size,
            'ternary_ns': ternary_time,
            'baseline_ns': baseline_time,
            'speedup': ratio,
            'winner': 'ternary' if ratio > 1 else 'baseline'
        })

        if ratio > 1 and crossover is None:
            crossover = size

    return {
        'crossover_point': crossover,
        'results': results,
        'verdict': f"Ternario gana a partir de {crossover} elementos" if crossover
                   else "Ternario NUNCA gana en este rango"
    }
```

### 2.2 Analisis de Bandwidth

```
PREGUNTA: Estamos limitados por memoria o por computo?

Si estamos compute-bound, ternario no ayuda.
Si estamos memory-bound, ternario podria ayudar 4x.
```

**Test de roofline:**
```python
def roofline_analysis(operation, sizes):
    """
    Determina si estamos memory-bound o compute-bound.
    """
    results = []

    # Medir bandwidth pico del sistema
    peak_bandwidth = measure_memory_bandwidth()  # GB/s
    peak_compute = measure_peak_flops()  # GFLOPS

    for size in sizes:
        ops = size  # Operaciones
        bytes_moved = size * 2 * 1  # 2 inputs * 1 byte cada uno (ternario)
        # vs INT8: size * 2 * 1, vs FP32: size * 2 * 4

        time_taken = benchmark(operation, size)['mean'] / 1e9  # segundos

        actual_bandwidth = bytes_moved / time_taken / 1e9  # GB/s
        actual_compute = ops / time_taken / 1e9  # GOPS

        arithmetic_intensity = ops / bytes_moved  # ops/byte

        # Determinar limitante
        bandwidth_limit = peak_bandwidth * arithmetic_intensity
        is_memory_bound = actual_compute < bandwidth_limit

        results.append({
            'size': size,
            'arithmetic_intensity': arithmetic_intensity,
            'actual_bandwidth_GBs': actual_bandwidth,
            'bandwidth_utilization': actual_bandwidth / peak_bandwidth,
            'is_memory_bound': is_memory_bound,
            'bottleneck': 'MEMORY' if is_memory_bound else 'COMPUTE'
        })

    return results
```

---

## Nivel 3: Tests de Falsificacion

### 3.1 Tests que DEBEN Fallar si el Proyecto no Vale

```python
class FalsificationTests:
    """
    Si CUALQUIERA de estos tests falla, el proyecto no tiene valor practico.
    """

    def test_beats_numpy_int8_large_arrays(self):
        """Ternario DEBE ser mas rapido que NumPy INT8 para arrays >100K."""
        for size in [100_000, 1_000_000, 10_000_000]:
            result = compare_against_baselines(tadd, size)
            assert result['ternary_engine']['mean'] > result['numpy_int8']['mean'], \
                f"FALSIFICADO: Ternario pierde contra NumPy INT8 en size={size}"

    def test_energy_efficiency_advantage(self):
        """Ternario DEBE usar menos energia por operacion."""
        ternary_efficiency = measure_energy_efficiency(tadd_ternary, 1_000_000)
        int8_efficiency = measure_energy_efficiency(add_int8, 1_000_000)

        ratio = ternary_efficiency['ops_per_joule'] / int8_efficiency['ops_per_joule']
        assert ratio > 1.5, \
            f"FALSIFICADO: Eficiencia energetica solo {ratio:.2f}x (necesita >1.5x)"

    def test_memory_reduction_realized(self):
        """4x menos memoria DEBE traducirse en beneficio medible."""
        # Ternario: 2 bits/valor, INT8: 8 bits/valor = 4x teorico

        ternary_bandwidth = measure_effective_bandwidth(tadd_ternary)
        int8_bandwidth = measure_effective_bandwidth(add_int8)

        # Deberiamos ver al menos 2x mejora (4x teorico - overhead)
        ratio = int8_bandwidth / ternary_bandwidth
        assert ratio > 2.0, \
            f"FALSIFICADO: Solo {ratio:.2f}x reduccion de bandwidth (necesita >2x)"

    def test_real_workload_improvement(self):
        """DEBE mejorar un workload real, no solo microbenchmarks."""
        # Simular capa de red neuronal: matmul + activation

        # INT8 baseline (como lo hace TensorRT)
        int8_time = benchmark_layer_int8(input_size=1024, output_size=1024)

        # Ternario (nuestra implementacion)
        ternary_time = benchmark_layer_ternary(input_size=1024, output_size=1024)

        assert ternary_time < int8_time * 1.5, \
            f"FALSIFICADO: Ternario {ternary_time/int8_time:.2f}x mas lento en workload real"

    def test_crossover_exists(self):
        """DEBE existir un punto donde ternario gana."""
        result = find_crossover_point(
            tadd_ternary,
            add_numpy_int8,
            size_range=[2**i for i in range(6, 26)]  # 64 to 64M
        )

        assert result['crossover_point'] is not None, \
            "FALSIFICADO: No existe crossover point - ternario nunca gana"

        assert result['crossover_point'] < 1_000_000, \
            f"FALSIFICADO: Crossover en {result['crossover_point']} es demasiado alto para ser util"
```

### 3.2 Registro de Falsificaciones

```python
# benchmarks/falsification_log.json
{
    "tests_run": 0,
    "tests_passed": 0,
    "tests_failed": 0,
    "falsifications": [
        // Cada vez que un test falla, registrar:
        {
            "date": "2025-XX-XX",
            "test": "test_beats_numpy_int8_large_arrays",
            "expected": "speedup > 1.0",
            "actual": "speedup = 0.73",
            "verdict": "FALSIFICADO",
            "action_taken": "investigar overhead de pack/unpack"
        }
    ],
    "current_status": "UNVALIDATED"  // o "VALIDATED" si todos pasan
}
```

---

## Nivel 4: Metricas de Sistema Completo

### 4.1 End-to-End Latency

```
No medir kernels aislados. Medir el pipeline completo.
```

```python
def measure_e2e_latency(input_format, output_format, operation, size):
    """
    Mide latencia de un flujo de datos realista.
    """
    # Simular datos llegando en formato externo (como en produccion)
    if input_format == 'numpy_int8':
        raw_input = np.random.randint(-1, 2, size, dtype=np.int8)
    elif input_format == 'pytorch_tensor':
        raw_input = torch.randint(-1, 2, (size,), dtype=torch.int8)

    start = time.perf_counter_ns()

    # Paso 1: Conversion a formato ternario
    ternary_input = convert_to_ternary(raw_input)

    # Paso 2: Operacion
    ternary_output = operation(ternary_input, ternary_input)

    # Paso 3: Conversion a formato de salida
    if output_format == 'numpy_int8':
        final_output = ternary_to_numpy(ternary_output)
    elif output_format == 'pytorch_tensor':
        final_output = ternary_to_torch(ternary_output)

    # Paso 4: Forzar materializacion
    _ = final_output[0]

    end = time.perf_counter_ns()

    return {
        'e2e_latency_ns': end - start,
        'overhead_vs_kernel_only': (end - start) / kernel_only_time,
    }
```

### 4.2 Memory Footprint Real

```python
def measure_memory_footprint(operation, size):
    """
    Mide memoria REAL usada, no solo el array.
    """
    import tracemalloc

    tracemalloc.start()

    # Baseline antes de operacion
    snapshot1 = tracemalloc.take_snapshot()

    # Ejecutar operacion
    a = generate_ternary(size)
    b = generate_ternary(size)
    result = operation(a, b)

    # Despues de operacion
    snapshot2 = tracemalloc.take_snapshot()

    # Calcular diferencia
    stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_allocated = sum(stat.size for stat in stats)

    # Comparar con teorico
    theoretical_minimum = size * 2 / 8  # 2 bits por trit
    overhead_ratio = total_allocated / theoretical_minimum

    return {
        'bytes_allocated': total_allocated,
        'theoretical_minimum': theoretical_minimum,
        'overhead_ratio': overhead_ratio,
        'verdict': 'ACCEPTABLE' if overhead_ratio < 2.0 else 'EXCESSIVE'
    }
```

---

## Nivel 5: Hardware Diversity

### 5.1 No Solo tu Maquina

```
OBLIGATORIO: Probar en multiples configuraciones.
```

| Categoria | Hardware | Por que |
|:----------|:---------|:--------|
| Desktop | Intel i7/i9 AVX2 | Tu maquina actual |
| Desktop | AMD Ryzen AVX2 | Diferente microarquitectura |
| Server | Intel Xeon | Mas cache, mas cores |
| Laptop | Intel U-series | Thermal throttling |
| Edge | Raspberry Pi 4 | ARM NEON, el target real |
| Cloud | AWS c6i.xlarge | Reproducibilidad |

**Script de CI multi-hardware:**
```yaml
# .github/workflows/benchmark_matrix.yml
jobs:
  benchmark:
    strategy:
      matrix:
        os: [ubuntu-22.04, windows-2022]
        arch: [x64]
        include:
          - runner: self-hosted-arm64  # Raspberry Pi

    steps:
      - name: Run falsification tests
        run: python benchmarks/run_falsification_tests.py

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-${{ matrix.os }}-${{ matrix.arch }}
          path: benchmarks/results/
```

---

## Nivel 6: Criterios de Exito/Fracaso

### 6.1 Definicion Clara

```
EXITO = Todos estos criterios se cumplen:

1. Speedup > 1.5x vs NumPy INT8 para arrays > 100K elementos
2. Eficiencia energetica > 2x vs INT8 (ops/joule)
3. Crossover point < 100K elementos
4. E2E latency overhead < 20% vs kernel-only
5. Funciona en al menos 2 arquitecturas (x64 + ARM)
6. Todos los tests de falsificacion pasan

FRACASO = Cualquiera de estos:

1. No existe crossover point en rango practico
2. Eficiencia energetica < 1.0x (peor que INT8)
3. Solo funciona en hardware especifico
4. Overhead de conversion > 50%
```

### 6.2 Decision Final

```python
def project_verdict():
    """
    Ejecutar despues de todos los benchmarks.
    """
    results = load_all_benchmark_results()

    criteria = {
        'speedup_large_arrays': results['speedup_100K'] > 1.5,
        'energy_efficiency': results['ops_per_joule_ratio'] > 2.0,
        'crossover_practical': results['crossover_point'] < 100_000,
        'e2e_overhead_acceptable': results['e2e_overhead'] < 1.2,
        'multi_arch': len(results['passing_architectures']) >= 2,
        'falsification_tests': results['falsification_failures'] == 0,
    }

    passed = sum(criteria.values())
    total = len(criteria)

    if passed == total:
        return "VALIDATED: El proyecto tiene valor practico demostrado"
    elif passed >= 4:
        return f"PARTIAL: {passed}/{total} criterios cumplidos - investigar gaps"
    else:
        return f"FALSIFIED: Solo {passed}/{total} criterios - reconsiderar premisas"
```

---

## Implementacion Inmediata

### Paso 1: Crear el runner

```bash
# benchmarks/run_skeptical_benchmarks.py
python -m pytest benchmarks/test_falsification.py -v --tb=short
```

### Paso 2: Ejecutar y registrar

```bash
# Cada ejecucion genera:
# - benchmarks/results/YYYY-MM-DD_HH-MM-SS/
#   - raw_data.json
#   - falsification_results.json
#   - verdict.txt
```

### Paso 3: No avanzar sin validacion

```
REGLA: Ningun commit de "optimizacion" sin benchmark que lo respalde.
REGLA: Ningun claim de performance sin link a resultado reproducible.
REGLA: Si un test de falsificacion falla, PUBLICARLO, no ocultarlo.
```

---

## Conclusion

Este framework existe para responder UNA pregunta:

> **"¿El ternary engine tiene valor practico demostrable, o es elegancia teorica sin impacto?"**

Los numeros decidiran. No nosotros.

---

**Siguiente Accion:** Implementar `benchmarks/test_falsification.py` y ejecutar.

**Compromiso:** Publicar resultados sean cuales sean, incluyendo fracasos.
