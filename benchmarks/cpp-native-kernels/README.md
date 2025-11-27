# C++ Native Kernel Benchmarks

**Purpose:** Pure C++ benchmarks that bypass Python/NumPy overhead for accurate kernel performance measurement.

---

## Why Native C++?

Python benchmarks include interpreter overhead, pybind11 marshalling, and NumPy allocation costs. These native benchmarks measure:

- **Raw SIMD throughput** without FFI overhead
- **True kernel latency** via std::chrono::steady_clock
- **Accurate SIMD vs scalar speedups** for optimization validation
- **Memory bandwidth** for cache and streaming analysis

The 35,042 Mops/s peak throughput claim requires native timing to be credible.

---

## Files

| File | Purpose |
|------|---------|
| `benchmark_main.cpp` | Production suite with JSON/CSV output, OpenMP threading |
| `bench_kernels.cpp` | Kernel microbenchmarks (SIMD vs scalar comparison) |
| `bench_tritnet_gemm.cpp` | TritNet GEMM vs BitNet performance analysis |
| `reference_cpp.cpp` | Unoptimized baseline (pybind11 module) for fair comparison |
| `include/cpu_info.h` | CPU detection (vendor, AVX2/512 support) |
| `include/timer.h` | High-resolution timing utilities |

---

## Build Instructions

### benchmark_main.cpp (Production Suite)

```bash
# Windows (MSVC)
cl /O2 /arch:AVX2 /openmp /std:c++17 /EHsc benchmark_main.cpp

# Linux/macOS (GCC/Clang)
g++ -O3 -march=native -fopenmp -std=c++17 benchmark_main.cpp -o bench
clang++ -O3 -march=native -fopenmp -std=c++17 benchmark_main.cpp -o bench
```

**Usage:**
```bash
./bench --repeat=5 --threads=12 --out=results/bench.json
```

### bench_kernels.cpp (Microbenchmarks)

```bash
# GCC/Clang
g++ -O3 -march=native -mavx2 -std=c++17 bench_kernels.cpp -o bench_kernels

# Run
./bench_kernels              # Human-readable output
./bench_kernels --csv        # CSV format
./bench_kernels --json       # JSON format
```

### bench_tritnet_gemm.cpp (TritNet GEMM)

```bash
# Requires tritnet_gemm.h from include/
g++ -O3 -march=native -mavx2 -std=c++17 -I../../include bench_tritnet_gemm.cpp -o bench_gemm
```

### reference_cpp.cpp (Python Module)

This is a pybind11 module compiled with the build system:
```bash
python build/build.py  # Builds as part of standard build
```

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

## Dependencies

- C++17 compiler (MSVC 2019+, GCC 7+, Clang 6+)
- AVX2-capable CPU (Intel Haswell 2013+, AMD Excavator 2015+)
- OpenMP (optional, for benchmark_main.cpp threading)
- pybind11 (only for reference_cpp.cpp)

---

## Integration with Python Benchmarks

These native benchmarks validate the Python benchmarks in `benchmarks/`. Use them to:

1. Verify throughput claims (native should match or exceed Python results)
2. Isolate kernel performance from FFI overhead
3. Compare SIMD implementations across compilers

---

**Validated:** Windows x64, MSVC 2022, 2025-11-27
