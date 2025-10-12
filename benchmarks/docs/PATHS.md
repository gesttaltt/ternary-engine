# Benchmark Directory Structure and Paths

**Last Updated**: 2025-10-11
**Purpose**: Document all benchmark-related paths and file locations

## Directory Structure

```
benchmarks/
├── bench_phase0.py           # Main Phase 0 benchmark runner
├── reference/                # Reference implementations (for comparison)
│   └── (reference.py will be created here)
├── results/                  # JSON output files from benchmark runs
│   └── phase0_YYYYMMDD_HHMMSS.json
└── docs/                     # Benchmark documentation
    ├── benchmark.md          # Benchmark design specification
    └── PATHS.md              # This file

```

## Path Dependencies

### bench_phase0.py
- **Imports from**: `../` (parent directory for `ternary_core_simd_full`)
- **Writes to**: `results/phase0_YYYYMMDD_HHMMSS.json`
- **Run from**: Project root with `python benchmarks/bench_phase0.py`

### Results Files
- **Location**: `benchmarks/results/`
- **Format**: JSON
- **Naming**: `phase0_YYYYMMDD_HHMMSS.json`
- **Content**: Timing data, speedup metrics, validation results

## Running Benchmarks

### From Project Root
```bash
python benchmarks/bench_phase0.py
```

### From Benchmarks Directory
```bash
cd benchmarks
python bench_phase0.py
```

## File Isolation

All benchmark-related files are now isolated in the `benchmarks/` directory:

1. **Executable scripts**: Top level of benchmarks/
2. **Reference implementations**: benchmarks/reference/
3. **Documentation**: benchmarks/docs/
4. **Output data**: benchmarks/results/

## External Dependencies

- `ternary_core_simd_full.pyd`: Compiled module in project root
- `numpy`: System Python package
- `test_phase0.py`: Correctness tests (in project root, separate from benchmarks)

## Path Resolution Strategy

The benchmark script uses:
```python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```

This ensures the compiled module can be found regardless of where the script is run from.

## Separation of Concerns

- **Correctness Testing**: `test_phase0.py` (project root)
- **Performance Benchmarking**: `benchmarks/bench_phase0.py`
- **Correctness Documentation**: `local-reports/test-suite.md`
- **Performance Documentation**: `benchmarks/docs/benchmark.md`
