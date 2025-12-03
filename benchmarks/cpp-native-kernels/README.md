# C++ Native Kernel Benchmarks

**Doc-Type:** Benchmark Suite · Version 2.0 · Updated 2025-12-03

**Purpose:** Pure C++ benchmarks that bypass Python/NumPy overhead for accurate kernel performance measurement.

---

## NEW: Gops/s and Memory Efficiency Benchmarks

### Gops/s Comparative Benchmark
- **File:** `bench_gops_comparative.cpp`
- **Purpose:** Compare ternary vs binary INT8 throughput in Gops/s
- **Build:** `build_gops.bat` (Windows) or see [README_GOPS.md](README_GOPS.md)

### Memory Efficiency Benchmark
- **File:** `bench_memory_efficiency.cpp`
- **Purpose:** Measure ternary's TRUE value: memory compression
- **Build:** `build_memory_bench.bat` (Windows) or see [README_MEMORY.md](README_MEMORY.md)

**Key Metrics:**
| Metric | Ternary | INT8 | FP32 |
|:-------|:--------|:-----|:-----|
| Bits/Element | 2 | 8 | 32 |
| Compression | 16x | 4x | 1x |
| LLaMA-7B Size | 1.75 GB | 7 GB | 28 GB |

---

## Source Code Targets

These benchmarks target two distinct codebases with clear separation of concerns:

### src/core/ — Production Kernel (Validated, Stable)

| Component | Path | Description |
|-----------|------|-------------|
| Algebra | `src/core/algebra/` | Scalar operations + compile-time LUTs |
| SIMD Kernels | `src/core/simd/` | AVX2-accelerated operations (32 parallel trits) |
| Fusion | `src/core/simd/fused_binary_unary_ops.h` | Fused operations (validated 1.5-3.0× speedup) |
| CPU Detection | `src/core/simd/cpu_simd_capability.h` | Runtime ISA detection |
| Unified API | `src/core/core_api.h` | Single include entry point |

**Status:** Windows x64 validated, 65/65 tests passing, 35,042 Mops/s peak throughput

### src/engine/ — Library Code (Experimental)

| Component | Path | Description |
|-----------|------|-------------|
| Dense243 | `src/engine/dense243/` | 5 trits/byte encoding (95.3% density) |
| Python Bindings | `src/engine/bindings_*.cpp` | pybind11 wrappers (NOT benchmarked here) |

**Status:** Functional, validated pack/unpack, SIMD extraction pending

### models/tritnet/ — TritNet Models (Experimental)

| Component | Path | Description |
|-----------|------|-------------|
| GEMM | `models/tritnet/gemm/` | Ternary matrix multiplication |

**Status:** Proof-of-concept, Phase 2 training in progress

---

## Benchmark Files

### Core Kernel Benchmarks (src/core/)

| File | Target | Description |
|------|--------|-------------|
| `bench_gops_comparative.cpp` | `core/simd/` | **NEW:** Gops/s throughput comparison |
| `bench_memory_efficiency.cpp` | `core/simd/` | **NEW:** Memory efficiency analysis |
| `benchmark_main.cpp` | `core/algebra/`, `core/simd/` | Production suite with JSON/CSV, OpenMP |
| `bench_kernels.cpp` | `core/algebra/`, `core/simd/` | SIMD vs scalar microbenchmarks |
| `bench_fusion.cpp` | `core/simd/fused_binary_unary_ops.h` | Fused operation speedup validation |

### Engine Library Benchmarks (src/engine/)

| File | Target | Description |
|------|--------|-------------|
| `bench_dense243.cpp` | `engine/dense243/` | Pack/unpack/extract throughput |

### TritNet Benchmarks (models/tritnet/)

| File | Target | Description |
|------|--------|-------------|
| `bench_tritnet_gemm.cpp` | `models/tritnet/gemm/` | GEMM performance vs BitNet |

### Utility Headers

| File | Description |
|------|-------------|
| `include/bench_throughput.h` | **NEW:** Gops/s measurement framework |
| `include/bench_memory.h` | **NEW:** Memory efficiency metrics |
| `include/cpu_info.h` | CPU detection (vendor, AVX2/512 support) |
| `include/timer.h` | High-resolution timing utilities |

### Reference Implementation

| File | Description |
|------|-------------|
| `reference_cpp.cpp` | Unoptimized baseline (pybind11 module) |

---

## Compilation

All benchmarks compile from the `benchmarks/cpp-native-kernels/` directory.

### Core Kernel Benchmarks

```bash
# bench_kernels.cpp (SIMD microbenchmarks)
g++ -O3 -march=native -mavx2 -std=c++17 -I../../src bench_kernels.cpp -o bench_kernels
clang++ -O3 -march=native -mavx2 -std=c++17 -I../../src bench_kernels.cpp -o bench_kernels

# benchmark_main.cpp (production suite with OpenMP)
g++ -O3 -march=native -mavx2 -fopenmp -std=c++17 -I../../src benchmark_main.cpp -o bench_main
clang++ -O3 -march=native -mavx2 -fopenmp -std=c++17 -I../../src benchmark_main.cpp -o bench_main

# bench_fusion.cpp (fused operations)
g++ -O3 -march=native -mavx2 -std=c++17 -I../../src bench_fusion.cpp -o bench_fusion
```

### Engine Library Benchmarks

```bash
# bench_dense243.cpp (Dense243 encoding)
g++ -O3 -march=native -mavx2 -std=c++17 -I../../src bench_dense243.cpp -o bench_dense243
```

### TritNet Benchmarks

```bash
# bench_tritnet_gemm.cpp (TritNet GEMM)
g++ -O3 -march=native -mavx2 -std=c++17 -I../../ bench_tritnet_gemm.cpp -o bench_gemm
```

### Windows (MSVC)

```cmd
:: Core benchmarks
cl /O2 /arch:AVX2 /std:c++17 /EHsc /I..\..\src bench_kernels.cpp
cl /O2 /arch:AVX2 /openmp /std:c++17 /EHsc /I..\..\src benchmark_main.cpp

:: Engine benchmarks
cl /O2 /arch:AVX2 /std:c++17 /EHsc /I..\..\src bench_dense243.cpp

:: TritNet benchmarks
cl /O2 /arch:AVX2 /std:c++17 /EHsc /I..\..\ bench_tritnet_gemm.cpp
```

---

## Usage

### bench_kernels (SIMD microbenchmarks)

```bash
./bench_kernels              # Human-readable output
./bench_kernels --csv        # CSV format for analysis
./bench_kernels --json       # JSON format
```

**Output:** SIMD vs scalar throughput (ME/s), speedup ratios

### benchmark_main (production suite)

```bash
./bench_main --repeat=5 --threads=12 --out=results/bench.json
```

**Output:** JSON + CSV telemetry for CI dashboards

### bench_fusion (fused operations)

```bash
./bench_fusion               # Human-readable output
./bench_fusion --csv         # CSV format
```

**Output:** Unfused vs fused timing, speedup ratios, coefficient of variation

### bench_dense243 (Dense243 encoding)

```bash
./bench_dense243             # Human-readable output
./bench_dense243 --csv       # CSV format
```

**Output:** Pack/unpack/extract throughput (ME/s)

### bench_tritnet_gemm (TritNet GEMM)

```bash
./bench_gemm                 # Full benchmark suite + BitNet comparison
```

**Output:** GEMM throughput (Gops/s), memory bandwidth analysis

---

## Why Native C++?

Python benchmarks include interpreter overhead, pybind11 marshalling, and NumPy allocation costs. These native benchmarks measure:

- **Raw SIMD throughput** without FFI overhead
- **True kernel latency** via std::chrono::steady_clock
- **Accurate SIMD vs scalar speedups** for optimization validation
- **Memory bandwidth** for cache and streaming analysis

The 35,042 Mops/s peak throughput claim requires native timing to be credible.

### Overhead Breakdown

| Source | Typical Overhead |
|--------|------------------|
| pybind11 array conversion | 5-15% |
| Python function dispatch | 1-5% |
| NumPy output allocation | 5-10% |
| GIL release/acquire | 1-3% |
| **Total** | **12-33%** |

---

## Output Formats

### JSON (benchmark_main.cpp)

```json
{
  "meta": {"compiler": "...", "threads": 12, "repeat": 5},
  "runs": [
    {"op": "tadd", "mode": "SIMD_san", "n": 1000000, "ns_per_elem": 0.28}
  ]
}
```

### CSV

```csv
operation,size,throughput_simd_ME_s,throughput_scalar_ME_s,speedup
tadd,1000000,3571.43,112.50,31.74
```

---

## Validation Against Python Benchmarks

Compare with `python-with-interpreter-overhead/` to validate overhead estimates:

```bash
# Run native benchmark
./bench_kernels --csv > native_results.csv

# Run Python benchmark
cd ../python-with-interpreter-overhead
python bench_simd_core_ops.py > python_results.csv

# Compare (Python should be 12-33% slower)
```

**Expected:** If Python shows **higher** throughput than native, something is wrong.

---

## Performance Expectations

### Core SIMD Operations (bench_kernels, benchmark_main)

| Operation | Scalar (ME/s) | SIMD (ME/s) | Speedup |
|-----------|---------------|-------------|---------|
| tadd | ~100 | ~3500 | ~35× |
| tmul | ~100 | ~3500 | ~35× |
| tmin | ~100 | ~3500 | ~35× |
| tmax | ~100 | ~3500 | ~35× |
| tnot | ~150 | ~4000 | ~27× |

### Fusion Operations (bench_fusion)

| Operation | Speedup Range | Average |
|-----------|---------------|---------|
| tnot(tadd) | 1.62-1.95× | 1.76× |
| tnot(tmul) | 1.53-1.86× | 1.71× |
| tnot(tmin) | 1.61-11.26× | 4.06× |
| tnot(tmax) | 1.65-9.50× | 3.68× |

### Dense243 Operations (bench_dense243)

| Operation | Expected (ME/s) |
|-----------|-----------------|
| pack | ~50-100 |
| unpack | ~100-200 |
| extract | ~50-100 |

---

## Dependencies

- C++17 compiler (MSVC 2019+, GCC 7+, Clang 6+)
- AVX2-capable CPU (Intel Haswell 2013+, AMD Excavator 2015+)
- OpenMP (optional, for benchmark_main.cpp threading)
- pybind11 (only for reference_cpp.cpp)

---

## Architecture Summary

```
benchmarks/cpp-native-kernels/
├── README.md                    # This file
├── include/
│   ├── cpu_info.h               # CPU detection utilities
│   └── timer.h                  # Timing utilities
│
├── # Core Kernel Benchmarks (src/core/)
├── benchmark_main.cpp           # Production suite (JSON/CSV, OpenMP)
├── bench_kernels.cpp            # SIMD vs scalar microbenchmarks
├── bench_fusion.cpp             # Fused operation validation
│
├── # Engine Library Benchmarks (src/engine/)
├── bench_dense243.cpp           # Dense243 pack/unpack/extract
│
├── # TritNet Benchmarks (models/tritnet/)
├── bench_tritnet_gemm.cpp       # TritNet GEMM vs BitNet
│
└── # Reference
    └── reference_cpp.cpp        # Unoptimized baseline (pybind11)
```

---

**Validated:** Windows x64, MSVC 2022, 2025-12-03

---

## Latest Benchmark Results (2025-12-03)

### Peak Throughput (Python benchmarks, 100K elements)
| Operation | Gops/s | Speedup vs Python |
|:----------|:-------|:------------------|
| tnot | 19.57 | 2,634x |
| tmin | 15.53 | 3,837x |
| tmax | 15.48 | 3,951x |
| tadd | 15.34 | 4,022x |
| tmul | 15.10 | 3,993x |

### Bridge Layer Performance
| Metric | Value |
|:-------|:------|
| Max Speedup vs NumPy | 44.58x |
| Fused Ops Max Speedup | 29.12x |
| Crossover Point | 64 elements |

### Memory Efficiency (Theoretical)
| Model | FP32 | Ternary | Compression |
|:------|:-----|:--------|:------------|
| LLaMA-7B | 28 GB | 1.75 GB | 16x |
| LLaMA-70B | 280 GB | 17.5 GB | 16x |
