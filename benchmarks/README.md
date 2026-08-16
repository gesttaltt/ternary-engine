# Benchmarks

Production-grade Python benchmark suite for the Ternary Engine library, measuring performance of ternary logic operations with AVX2 SIMD acceleration and OpenMP parallelization.

**Note (2026-08-16):** the scripts below (`bench_phase0.py`, `bench_compare.py`)
were renamed in a Nov 2025 reorganization and moved under
`python-with-interpreter-overhead/`; this file hadn't caught up. See
[`python-with-interpreter-overhead/README.md`](python-with-interpreter-overhead/README.md)
for this project's current, more precise framing of what these Python-level
timings can and can't be trusted for (FFI/interpreter overhead — absolute
throughput numbers here are approximate; see also `SKEPTICAL_METRICS.md`).

## Quick Start

```bash
# Run full benchmark suite
python benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py

# Quick test (fewer array sizes)
python benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py --quick

# Master orchestrator (build + benchmark + compare)
python benchmarks/python-with-interpreter-overhead/run_all_benchmarks.py

# With PGO build comparison
python benchmarks/python-with-interpreter-overhead/run_all_benchmarks.py --with-pgo
```

## Structure

```
benchmarks/
├── python-with-interpreter-overhead/  # Core Python benchmark suite
│   ├── bench_simd_core_ops.py         # Main benchmark suite (was bench_phase0.py)
│   ├── bench_regression_detect.py     # Regression detection tool (was bench_compare.py)
│   ├── run_all_benchmarks.py          # Master orchestrator
│   └── ... (bench_competitive.py, bench_dense243.py, bench_fair_baseline.py, etc.)
├── cpp-native-kernels/    # Native C++ benchmarks (FFI-overhead-free, for absolute claims)
├── macro/                 # Macro workload benchmarks (layer forward, image pipeline)
├── utils/                 # Shared metrics/validation helpers
├── deprecated/            # Scripts using the pre-v1.2.0 backend architecture
├── investigation/         # One-time analysis scripts
├── prototype/             # Scripts awaiting further module support
└── results/               # Output directory (flat, timestamped filenames per
                            # script -- see "Output Format" below, not
                            # standard/pgo/validation subdirectories)
```

## Benchmark Suite (bench_simd_core_ops.py)

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
python benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py

# Quick test (4 sizes: 32, 1K, 100K, 1M)
python benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py --quick

# Custom output directory
python benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py --output=benchmarks/results/my_test

# Minimal output
python benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py --quiet
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

**Caveat (see the note at the top of this file):** treat the pure-Python
`results_baseline` comparison as illustrative, not a headline claim — this
project has since retired compiled-vs-interpreted speedup framing project-wide
(see `.claude/CLAUDE.md` `core_innovation`) in favor of fair NumPy-baseline
comparisons (`bench_fair_baseline.py` in the same directory).

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

### Example Results

The numbers below are illustrative sample output from when this file was
first written, not a current validated baseline — see
`benchmarks/results/*.json` and `.claude/CLAUDE.md`'s TritNet/performance
sections for actual dated, platform-labeled measurements.

**Small arrays (32 elements)**: ~20-30 Mops/s, dominated by call overhead
**Medium arrays (1,000 elements)**: ~640-920 Mops/s, cache-resident + SIMD
**Large arrays (100,000 elements)**: ~13,000-17,000 Mops/s, OpenMP active
**Very large arrays (1,000,000 elements)**: ~3,400-8,400 Mops/s, memory-bandwidth limited

## Comparison Tool (bench_regression_detect.py)

Compares benchmark results to detect performance regressions or improvements.

### Usage

```bash
# Compare two benchmark results
python benchmarks/python-with-interpreter-overhead/bench_regression_detect.py \
    benchmarks/results/before/bench_results_20251014_013601.json \
    benchmarks/results/after/bench_results_20251014_020000.json

# Custom output
python benchmarks/python-with-interpreter-overhead/bench_regression_detect.py before.json after.json --output=comparison.json

# Custom regression threshold (default: 5%)
python benchmarks/python-with-interpreter-overhead/bench_regression_detect.py before.json after.json --threshold=10.0
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
python benchmarks/python-with-interpreter-overhead/run_all_benchmarks.py

# Include PGO build and comparison
python benchmarks/python-with-interpreter-overhead/run_all_benchmarks.py --with-pgo

# Quick mode (fewer test sizes)
python benchmarks/python-with-interpreter-overhead/run_all_benchmarks.py --quick

# Clean build artifacts first
python benchmarks/python-with-interpreter-overhead/run_all_benchmarks.py --clean

# Skip builds (only run benchmarks)
python benchmarks/python-with-interpreter-overhead/run_all_benchmarks.py --skip-build
```

### Workflow

1. **Clean** (optional): Remove old build artifacts
2. **Build standard**: Run `build/build.py` for optimized build
3. **Benchmark standard**: Run `bench_simd_core_ops.py` on standard build
4. **Build PGO** (optional): Run `build/build_pgo_unified.py` for profile-guided optimization
5. **Benchmark PGO** (optional): Run `bench_simd_core_ops.py` on PGO build
6. **Compare** (optional): Compare standard vs PGO results

### Output

Results are written to `benchmarks/results/` as flat, timestamped JSON files
(e.g. `fair_baseline_20260812_125250.json`), not organized into
`standard/`/`pgo/`/`comparison_*.json` subpaths.

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

This project's actual CI workflow is `.github/workflows/ci.yml` (build +
correctness tests, not performance benchmarks — see `.claude/CLAUDE.md`
"Critical Gaps" #1/#3 for current benchmark-validation status). The example
below is illustrative of how a performance job could be wired up, not a
description of an existing one:

```yaml
name: Performance Benchmarks

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install pybind11 numpy

      - name: Run benchmarks
        run: |
          python build/build.py
          python benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py --quick --output=results

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: results/
```

### Regression Detection

```bash
# Store baseline results
python benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py --output=benchmarks/results/baseline

# After changes, run comparison
python benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py --output=benchmarks/results/current
python benchmarks/python-with-interpreter-overhead/bench_regression_detect.py \
    benchmarks/results/baseline/bench_results_*.json \
    benchmarks/results/current/bench_results_*.json
```

## Performance Tuning

### Thread Count

Control OpenMP threads:

```bash
export OMP_NUM_THREADS=8
python benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py
```

### Array Size Threshold

The OpenMP threshold is adaptive (32K elements per thread), defined in
`src/core/config/optimization_config.h` (`OMP_THRESHOLD`), consumed by
`src/engine/bindings_core_ops.cpp`:

```cpp
OMP_THRESHOLD = 32768 * std::thread::hardware_concurrency()
```

For 8 cores: threshold = 262,144 elements

### Streaming Stores

Non-temporal stores activate at 1M elements to reduce cache pollution
(`STREAM_THRESHOLD` in the same header):

```cpp
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

Treat pure-Python-baseline speedup figures as illustrative only — see the
caveat at the top of this file and `.claude/CLAUDE.md`'s `core_innovation`
note. For a fair, currently-maintained comparison, use
`bench_fair_baseline.py` (NumPy baseline) in the same directory instead.

### OpenMP Scaling

Test OpenMP effectiveness:

```bash
export OMP_NUM_THREADS=1
python benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py --quick --output=results/threads_1

export OMP_NUM_THREADS=8
python benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py --quick --output=results/threads_8

python benchmarks/python-with-interpreter-overhead/bench_regression_detect.py \
    results/threads_1/bench_results_*.json \
    results/threads_8/bench_results_*.json
```

Expected: some speedup for multiple threads on large arrays (100K+ elements) —
validate against an actual run rather than assuming a specific factor; this
project's own rigor rules (`SKEPTICAL_METRICS.md`) call for measured, not
assumed, scaling numbers.

## Troubleshooting

### "ModuleNotFoundError: No module named 'ternary_simd_engine'"

Build the module first:

```bash
python build/build.py
```

### Segmentation Fault

Known issue with verbose output. Use `--quiet` flag:

```bash
python benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py --quiet
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
   taskset -c 0-7 python benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py
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

- **[python-with-interpreter-overhead/README.md](python-with-interpreter-overhead/README.md)** - This project's current framing of Python-level timing reliability (FFI/interpreter overhead caveats)
- **[../docs/build-system/README.md](../docs/build-system/README.md)** - Build system documentation
- **[../tests/README.md](../tests/README.md)** - Test suite
- **[../docs/pgo/README.md](../docs/pgo/README.md)** - Profile-Guided Optimization
- **[../README.md](../README.md)** - Project overview

---

**Last Updated**: 2026-08-16 (paths/names corrected against the Nov 2025
reorganization; original content from 2025-10-14)
**Maintained by**: Jonathan Verdun (Ternary Engine Project)
