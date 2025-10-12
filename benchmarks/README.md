# Ternary Core Benchmark Suite

**Purpose**: Performance validation for Phase 0 optimizations

**Status**: Research-grade validation tool (not production benchmarking)

---

## Quick Start

### Prerequisites

1. Compiled module must exist in project root:
   ```bash
   cd ..
   python setup.py build_ext --inplace
   ```

2. Verify compilation:
   ```bash
   python -c "import ternary_core_simd_full; print('✓ Module loaded')"
   ```

### Running Benchmarks

From project root:
```bash
python benchmarks/bench_phase0.py
```

Expected output:
```
============================================================
  Phase 0 Benchmark Suite - Minimal Validation
============================================================

Correctness Verification (Spot Check)
  ✅ All spot checks passed

Benchmark 1: Microbenchmarks (Scalar Speedup)
  tadd (n= 8):   5.2x speedup
  tmul (n= 8):   4.8x speedup
  ...

  Median speedup: 5.3x
  Target: 3-10x → ✅ PASS

Benchmark 2: Array Size Sweep (SIMD Efficiency)
  n=      8:    12.4 Mtrits/s,  5.12x,   0.0% SIMD
  n=  1048576:  287.3 Mtrits/s,  1.45x, 100.0% SIMD
  ...

  ✅ PHASE 0 VALIDATED - All targets met!
```

**Runtime**: ~30 seconds

---

## Directory Structure

```
benchmarks/
├── README.md              # This file
├── bench_phase0.py        # Main benchmark runner
├── reference.py           # Reference implementations for comparison
├── docs/                  # Benchmark documentation
│   ├── benchmark.md       # Original detailed benchmark spec
│   ├── DESIGN_REVIEW.md   # Realism and reproducibility analysis
│   └── PATHS.md           # Path documentation
└── results/               # JSON output files
    └── phase0_YYYYMMDD_HHMMSS.json
```

---

## What Phase 0 Benchmarks Test

### 1. Microbenchmarks (Scalar Speedup)

**Goal**: Validate LUT optimization speedup claim (3-10x)

**Method**:
- Small arrays (8, 16, 31 elements)
- Forces scalar code path (no SIMD)
- Compares optimized (LUT-based) vs reference (branch-based)

**What it actually measures**:
- ⚠️ Python+C++ system speedup (NOT pure C++ speedup)
- Includes Python binding overhead
- Valid for demonstrating overall system improvement

**Target**: 3-10x median speedup across all operations

### 2. Array Size Sweep (SIMD Efficiency)

**Goal**: Validate overall performance improvement (30-50%)

**Method**:
- Arrays from 8 to 1M elements
- Tests single operation (tadd) across scales
- Measures throughput and SIMD percentage

**Metrics**:
- Throughput (million trits/second)
- Speedup vs reference
- SIMD efficiency percentage

**Targets**:
- Overall geometric mean speedup: 1.3-1.5x (30-50%)
- Peak throughput: >30 Mtrits/s

---

## Reproducibility

### Random Seed

All benchmarks use fixed seed (42) for reproducible results:
```python
np.random.seed(42)
```

Multiple runs should produce identical results (within measurement noise).

### Correctness Verification

Benchmarks include automatic correctness spot-checks before running:
- Tests key operations with known inputs
- Aborts if any correctness error detected
- Ensures we measure "fast correct answers" not "fast wrong answers"

---

## Understanding Results

### Results Files

Located in `results/phase0_YYYYMMDD_HHMMSS.json`:

```json
{
  "timestamp": "2025-10-11T21:30:00",
  "random_seed": 42,
  "elapsed_sec": 28.3,
  "microbenchmarks": [...],
  "size_sweep": [...],
  "summary": {
    "scalar_speedup": 5.3,
    "overall_speedup": 1.42,
    "peak_throughput": 287300000.0,
    "validation": {
      "scalar_pass": true,
      "overall_pass": true,
      "throughput_pass": true,
      "all_pass": true
    }
  }
}
```

### Interpreting Speedups

**Microbenchmarks** (3-10x target):
- Dominated by scalar LUT optimization
- Includes Python overhead (inflates numbers slightly)
- Valid measure of small-array performance improvement

**Array Size Sweep** (1.3-1.5x target):
- Realistic mixed workload measurement
- Small arrays: High scalar speedup
- Large arrays: SIMD dominates (less benefit from LUT)
- Geometric mean represents overall improvement

### Throughput Numbers

**Peak throughput** (>30 Mtrits/s):
- Measured on large arrays (SIMD-saturated)
- Typical range: 100-300 Mtrits/s
- Depends on CPU, memory bandwidth, cache
- Target of 30 Mtrits/s is very conservative

---

## Limitations

### What This Benchmark Does NOT Test

1. **Pure C++ optimization**: Measures Python+C++ system, not isolated C++
2. **Operation chains**: Only single operations, no realistic multi-op sequences
3. **Statistical rigor**: Single median, no confidence intervals
4. **Memory patterns**: Random data, not realistic access patterns
5. **Multi-threading**: Phase 0 is single-threaded only

### Why These Limitations Are Acceptable

Phase 0 is a **research validation tool**:
- Goal: Prove the concept is viable
- Not goal: Production-grade performance analysis

If Phase 0 succeeds → Invest in comprehensive benchmarking (Phase 1+)
If Phase 0 fails → No need for expensive profiling infrastructure

See `docs/DESIGN_REVIEW.md` for full analysis.

---

## Reference Implementation

The `reference.py` file contains pre-optimization implementations:

```python
from reference import ref_tadd, ref_tmul, ref_tnot
```

These use:
- Integer conversion (trit → int → trit)
- Branches (if/else logic)
- Python loops (intentionally slow)

This represents the "before optimization" baseline.

---

## Troubleshooting

### "Module not compiled" Error

```bash
# From project root
python setup.py build_ext --inplace

# Verify
python -c "import ternary_core_simd_full"
```

### Import Errors

Benchmark must be run from project root:
```bash
# Correct
python benchmarks/bench_phase0.py

# Incorrect
cd benchmarks
python bench_phase0.py  # May fail to find module
```

### Slow Performance / Timeouts

If benchmarks take >2 minutes:
1. Check CPU throttling (thermal issues?)
2. Close other applications (reduce system noise)
3. Try on different machine

### Unexpected Results

If targets not met:
1. Check correctness tests first (`python test_phase0.py`)
2. Verify compiler optimizations were applied
3. Check CPU architecture (AVX2 required for SIMD)
4. Review `docs/DESIGN_REVIEW.md` for expected ranges

---

## Next Steps

### If Phase 0 Passes ✅

1. Document results in project README
2. Plan Phase 1 optimizations
3. Consider comprehensive benchmark suite
4. Explore production use cases

### If Phase 0 Fails ❌

1. Analyze which targets failed
2. Check for implementation bugs
3. Re-evaluate optimization approach
4. Consider alternative strategies

---

## Documentation

- `docs/benchmark.md` - Original detailed benchmark specification
- `docs/DESIGN_REVIEW.md` - Realism and reproducibility analysis
- `docs/PATHS.md` - Directory structure and path documentation

---

**Version**: 1.0
**Last Updated**: 2025-10-11
**Status**: Ready for Phase 0 validation
