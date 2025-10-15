# Benchmarks

Production-grade Python benchmark suite for the Ternary Engine library, measuring performance of ternary logic operations with AVX2 SIMD acceleration and OpenMP parallelization.

## Quick Start

```bash
# Run full benchmark suite
python benchmarks/bench_phase0.py

# Quick test (fewer array sizes)
python benchmarks/bench_phase0.py --quick

# Master orchestrator (build + benchmark + compare)
python benchmarks/run_all_benchmarks.py

# With PGO build comparison
python benchmarks/run_all_benchmarks.py --with-pgo
```

## Structure

```
benchmarks/
├── bench_phase0.py         # Main benchmark suite
├── bench_compare.py        # Regression detection tool
├── run_all_benchmarks.py   # Master orchestrator
└── results/                # Output directory (JSON + CSV)
    ├── standard/           # Standard build results
    ├── pgo/               # PGO build results
    └── validation/        # Test run results
```

## Benchmark Suite (bench_phase0.py)

### Features

- **Comprehensive testing**: All 5 ternary operations (tadd, tmul, tmin, tmax, tnot)
- **Multiple array sizes**: 32 to 10M elements
- **Warmup + measured runs**: Accurate timing with warmup iterations
- **Python baseline**: Pure Python reference for speedup calculations
- **JSON + CSV output**: CI/CD ready
- **Reproducible**: Fixed random seed for deterministic results

### Usage

```bash
# Full suite (7 array sizes: 32, 100, 1K, 10K, 100K, 1M, 10M)
python benchmarks/bench_phase0.py

# Quick test (4 sizes: 32, 1K, 100K, 1M)
python benchmarks/bench_phase0.py --quick

# Custom output directory
python benchmarks/bench_phase0.py --output=benchmarks/results/my_test

# Minimal output
python benchmarks/bench_phase0.py --quiet
```

### Output Format

#### JSON Output

```json
{
  "metadata": {
    "timestamp": "2025-10-14T01:35:57",
    "module": "ternary_simd_engine",
    "numpy_version": "1.26.4",
    "test_sizes": [32, 1000, 100000, 1000000],
    "warmup_iterations": 100,
    "measured_iterations": 1000
  },
  "results_optimized": [
    {
      "operation": "tadd",
      "size": 100000,
      "time_ns_total": 7615900,
      "iterations": 1000,
      "time_ns_per_op": 7615.9,
      "time_ns_per_elem": 0.076159,
      "throughput_mops": 13130.42
    }
  ],
  "results_baseline": [
    {
      "operation": "tadd",
      "size": 1000,
      "time_ns_total": 583964100,
      "iterations": 100,
      "time_ns_per_op": 5839641.0,
      "time_ns_per_elem": 5839.641,
      "throughput_mops": 0.171
    }
  ]
}
```

#### CSV Output

```csv
operation,size,time_ns_total,time_ns_per_elem,throughput_mops
tadd,32,1409200,44.0375,22.71
tadd,1000,1542900,1.5429,648.13
tadd,100000,7615900,0.0762,13130.42
tadd,1000000,291782600,0.2918,3427.21
```

### Performance Metrics

The benchmark measures:

1. **Throughput**: Operations per second (Mops/s)
2. **Latency**: Nanoseconds per element
3. **Speedup**: Compared to pure Python baseline
4. **Scaling**: Performance across array sizes

### Expected Results

**Small arrays (32 elements)**:
- Throughput: 20-30 Mops/s
- Speedup vs Python: ~137x

**Medium arrays (1,000 elements)**:
- Throughput: 640-920 Mops/s
- Speedup vs Python: ~3,800x

**Large arrays (100,000 elements)**:
- Throughput: 13,000-17,000 Mops/s
- OpenMP parallelization active

**Very large arrays (1,000,000 elements)**:
- Throughput: 3,400-8,400 Mops/s
- Memory bandwidth limited

## Comparison Tool (bench_compare.py)

Compares benchmark results to detect performance regressions or improvements.

### Usage

```bash
# Compare two benchmark results
python benchmarks/bench_compare.py \
    benchmarks/results/before/bench_results_20251014_013601.json \
    benchmarks/results/after/bench_results_20251014_020000.json

# Custom output
python benchmarks/bench_compare.py before.json after.json --output=comparison.json

# Custom regression threshold (default: 5%)
python benchmarks/bench_compare.py before.json after.json --threshold=10.0
```

### Output

```
================================================================================
  BENCHMARK COMPARISON
================================================================================

Comparing:
  Before: benchmarks/results/before/bench_results_20251014_013601.json
  After:  benchmarks/results/after/bench_results_20251014_020000.json

--------------------------------------------------------------------------------

Operation: tadd | Size: 100000
  Before: 13130.42 Mops/s
  After:  13850.67 Mops/s
  Change: +5.49% ✅ IMPROVEMENT

Operation: tmul | Size: 100000
  Before: 13458.77 Mops/s
  After:  12800.34 Mops/s
  Change: -4.89% ⚠️  REGRESSION

================================================================================
  SUMMARY
================================================================================

Total comparisons: 20
Improvements: 12
Regressions: 3
Unchanged: 5

Average change: +2.3%

Significant regressions (> 5%): 1
Significant improvements (> 5%): 8
```

## Master Orchestrator (run_all_benchmarks.py)

Automates the complete benchmarking workflow: build → benchmark → compare.

### Usage

```bash
# Standard build only
python benchmarks/run_all_benchmarks.py

# Include PGO build and comparison
python benchmarks/run_all_benchmarks.py --with-pgo

# Quick mode (fewer test sizes)
python benchmarks/run_all_benchmarks.py --quick

# Clean build artifacts first
python benchmarks/run_all_benchmarks.py --clean

# Skip builds (only run benchmarks)
python benchmarks/run_all_benchmarks.py --skip-build
```

### Workflow

1. **Clean** (optional): Remove old build artifacts
2. **Build standard**: Run `build.py` for optimized build
3. **Benchmark standard**: Run `bench_phase0.py` on standard build
4. **Build PGO** (optional): Run `build_pgo.py` for profile-guided optimization
5. **Benchmark PGO** (optional): Run `bench_phase0.py` on PGO build
6. **Compare** (optional): Compare standard vs PGO results

### Output

Results are organized in `benchmarks/results/`:

```
results/
├── standard/
│   └── bench_results_20251014_013601.json
├── pgo/
│   └── bench_results_20251014_020000.json
└── comparison_standard_vs_pgo.json
```

## Operations Tested

All 5 ternary operations are benchmarked:

1. **tadd** - Saturated ternary addition
2. **tmul** - Ternary multiplication
3. **tmin** - Ternary minimum
4. **tmax** - Ternary maximum
5. **tnot** - Ternary negation (unary)

## Test Sizes

### Full Suite (Default)

32, 100, 1,000, 10,000, 100,000, 1,000,000, 10,000,000 elements

### Quick Mode

32, 1,000, 100,000, 1,000,000 elements

### Size Categories

- **32 elements**: Micro-benchmark, tests call overhead
- **100-1K elements**: Small arrays, cache-resident
- **10K-100K elements**: Medium arrays, L3 cache boundary
- **100K+ elements**: Large arrays, OpenMP parallelization
- **1M+ elements**: Very large arrays, streaming stores

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Performance Benchmarks

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install pybind11 numpy

      - name: Run benchmarks
        run: |
          python build.py
          python benchmarks/bench_phase0.py --quick --output=results

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: results/
```

### Regression Detection

```bash
# Store baseline results
python benchmarks/bench_phase0.py --output=benchmarks/results/baseline

# After changes, run comparison
python benchmarks/bench_phase0.py --output=benchmarks/results/current
python benchmarks/bench_compare.py \
    benchmarks/results/baseline/bench_results_*.json \
    benchmarks/results/current/bench_results_*.json
```

## Performance Tuning

### Thread Count

Control OpenMP threads:

```bash
export OMP_NUM_THREADS=8
python benchmarks/bench_phase0.py
```

### Array Size Threshold

The OpenMP threshold is adaptive (32K elements per thread):

```python
# In ternary_simd_engine.cpp
OMP_THRESHOLD = 32768 * std::thread::hardware_concurrency()
```

For 8 cores: threshold = 262,144 elements

### Streaming Stores

Non-temporal stores activate at 1M elements to reduce cache pollution:

```cpp
// In ternary_simd_engine.cpp
STREAM_THRESHOLD = 1000000
```

## Interpreting Results

### Throughput Patterns

**Expected behavior**:
- Small arrays: Lower throughput (call overhead)
- Medium arrays: Peak throughput (cache-resident + SIMD)
- Large arrays: High throughput (OpenMP parallelization)
- Very large arrays: Moderate throughput (memory bandwidth limit)

### Speedup vs Python

**Typical speedups**:
- Small arrays (32): 100-200x
- Medium arrays (1K): 2,000-4,000x
- Large arrays (100K+): 10,000-20,000x

### OpenMP Scaling

Test OpenMP effectiveness:

```bash
export OMP_NUM_THREADS=1
python benchmarks/bench_phase0.py --quick --output=results/threads_1

export OMP_NUM_THREADS=8
python benchmarks/bench_phase0.py --quick --output=results/threads_8

python benchmarks/bench_compare.py \
    results/threads_1/bench_results_*.json \
    results/threads_8/bench_results_*.json
```

Expected: 5-8x speedup for 8 threads on large arrays (100K+ elements)

## Troubleshooting

### "ModuleNotFoundError: No module named 'ternary_simd_engine'"

Build the module first:

```bash
python build.py
```

### Segmentation Fault

Known issue with verbose output. Use `--quiet` flag:

```bash
python benchmarks/bench_phase0.py --quiet
```

### Inconsistent Results

Ensure system is idle during benchmarking:

1. Close background applications
2. Disable CPU frequency scaling (Linux):
   ```bash
   sudo cpupower frequency-set --governor performance
   ```
3. Pin to specific CPU cores (advanced):
   ```bash
   taskset -c 0-7 python benchmarks/bench_phase0.py
   ```

### Very Low Throughput

Check CPU supports AVX2:

```bash
# Linux
grep avx2 /proc/cpuinfo

# macOS
sysctl machdep.cpu.features | grep AVX2

# Windows PowerShell
Get-WmiObject -Class Win32_Processor | Select-Object -Property Name
```

## Related Documentation

- **[../build/README.md](../build/README.md)** - Build system documentation
- **[../tests/README.md](../tests/README.md)** - Test suite
- **[../docs/PGO_README.md](../docs/PGO_README.md)** - Profile-Guided Optimization
- **[../README.md](../README.md)** - Project overview

---

**Last Updated**: 2025-10-14
**Maintained by**: Jonathan Verdun (Ternary Engine Project)
