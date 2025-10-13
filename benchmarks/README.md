# Ternary SIMD Engine - Benchmark Suite

Production-grade benchmark suite for measuring performance of ternary logic operations with AVX2 SIMD acceleration.

## Structure

```
benchmarks/
├── benchmark_main.cpp    # Main benchmark harness
├── utils/
│   ├── timer.h           # High-resolution timing utilities
│   └── cpu_info.h        # CPU detection and info
└── results/              # Output directory (JSON + CSV)
```

## Building

### Linux/macOS (GCC/Clang)

```bash
cd benchmarks
g++ -O3 -march=native -fopenmp -std=c++17 benchmark_main.cpp -o bench
# or
clang++ -O3 -march=native -fopenmp -std=c++17 benchmark_main.cpp -o bench
```

### Windows (MSVC)

```bash
cd benchmarks
cl /O2 /arch:AVX2 /openmp /std:c++17 /EHsc benchmark_main.cpp
```

## Running

### Basic usage

```bash
./bench
```

### Custom parameters

```bash
./bench --repeat=10 --threads=8 --out=results/bench_custom.json
```

### Parameters

- `--repeat=N` : Number of repetitions per test (default: 5)
- `--threads=N` : Number of OpenMP threads (default: hardware concurrency)
- `--out=PATH` : Output JSON file path (default: results/bench.json)

## Output

### JSON telemetry

```json
{
  "meta": {
    "compiler": "clang++ -O3 -march=native",
    "threads": 12,
    "repeat": 5,
    "cpu_threads": 12,
    "timestamp": 1728825600
  },
  "runs": [
    {
      "op": "tadd",
      "mode": "SIMD_san",
      "sanitize": true,
      "n": 1000000,
      "ns_total": 215300.0,
      "ns_per_elem": 0.215
    }
  ]
}
```

### CSV output

```csv
op,mode,sanitize,n,ns_total,ns_per_elem
tadd,SIMD_san,true,1000000,215300.000,0.215
tadd,SIMD_raw,false,1000000,204800.000,0.205
tadd,Scalar,true,1000000,1250000.000,1.250
```

## Benchmark Modes

### SIMD_san
- Sanitized mode with input validation
- Masks input to ensure 2-bit trit values
- Production-safe mode (~3-5% overhead)

### SIMD_raw
- Raw mode without input validation
- Assumes pre-validated data
- Maximum performance mode

### Scalar
- Non-vectorized reference implementation
- Single-element processing
- Baseline for speedup calculations

## Operations Tested

All 5 ternary operations are benchmarked:

1. **tadd** - Saturated ternary addition
2. **tmul** - Ternary multiplication
3. **tmin** - Ternary minimum
4. **tmax** - Ternary maximum
5. **tnot** - Ternary negation (unary)

## Test Sizes

Default test sizes: 1,000 | 10,000 | 100,000 | 1,000,000 elements

These sizes test:
- Small arrays (cache-resident)
- Medium arrays (L3 cache boundary)
- Large arrays (OpenMP threshold: 100K)
- Very large arrays (streaming threshold: 1M)

## Integration

### CI/CD

Output JSON can be integrated with:
- GitHub Actions
- Grafana dashboards
- Plotly/matplotlib visualization
- Google Sheets (import CSV)

### Regression Detection

Compare JSON outputs across commits:

```python
import json

with open('results/bench_before.json') as f:
    before = json.load(f)
with open('results/bench_after.json') as f:
    after = json.load(f)

# Compare ns_per_elem for same op/mode/n
```

## Design Goals

1. **Deterministic**: Seeded RNG ensures reproducibility
2. **Minimal dependencies**: No external libraries except standard library
3. **Cross-platform**: Windows (MSVC) + Linux/macOS (GCC/Clang)
4. **CI-ready**: JSON output for automated pipelines
5. **Transparent**: Measures real kernel performance, not artifacts

## See Also

- `../local-reports/benchmark.md` - Full design specification
- `../local-reports/build.md` - Build system documentation
- `../build/scripts/` - Build automation scripts
