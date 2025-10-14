# Reference Build Script (setup_reference.py)

## Overview

`build/scripts/setup_reference.py` builds an intentionally unoptimized baseline implementation for performance benchmarking. This "reference" build uses conversion-based operations without SIMD, LUTs, or aggressive compiler optimizations.

**Purpose:** Measure the **actual impact** of optimizations, not Python vs C++ differences

**Location:** `build/scripts/setup_reference.py`

**Module produced:** `reference_cpp.cp312-win_amd64.pyd`

**Typical build time:** 25-45 seconds

## Why a Reference Build?

### The Benchmarking Problem

When measuring optimization impact, comparing against pure Python creates misleading results:

```
Pure Python implementation: 1000 ms
Optimized C++:               10 ms

Reported speedup: 100x
```

**But what causes the speedup?**
- 90x from C++ vs Python (language difference)
- 10x from optimizations (SIMD, LUTs, etc.)

**Problem:** We can't isolate the optimization impact!

### The Solution: Reference C++ Build

Compare apples to apples:

```
Reference C++ (unoptimized): 100 ms  ← Baseline
Standard C++ (optimized):     10 ms

Actual optimization impact: 10x
```

**Now we know:**
- C++ itself: ~10x faster than Python
- Our optimizations: ~10x additional speedup
- **Total:** 100x combined

## Intentional Limitations

The reference build deliberately **disables** all optimizations:

| Feature | Reference | Standard | Reason for Exclusion |
|---------|-----------|----------|----------------------|
| SIMD (AVX2) | ❌ No | ✅ Yes | Isolate SIMD impact |
| LUT operations | ❌ No | ✅ Yes | Measure LUT benefit |
| OpenMP | ❌ No | ✅ Yes | Isolate parallelization |
| `/O2` optimization | ❌ No (`/O1`) | ✅ Yes | Minimal compiler opts |
| `/GL` + `/LTCG` | ❌ No | ✅ Yes | No whole-program opts |
| Force inline | ❌ No | ✅ Yes | Natural call overhead |

**Result:** Fair baseline for measuring **our** optimizations

## Quick Start

```bash
# From project root
python build/scripts/setup_reference.py
```

The script follows the same workflow as `setup.py`:
1. Generate timestamp
2. Create build directories
3. Compile with minimal optimizations
4. Copy to `latest/` and project root

## Usage

### Basic Build

```bash
python build/scripts/setup_reference.py
```

### Build Output

```
======================================================================
  REFERENCE UNOPTIMIZED BUILD
  Timestamp: 20251012_143022
======================================================================

Created build directories:
  Temp:   H:\...\build\artifacts\reference\20251012_143022\temp
  Output: H:\...\build\artifacts\reference\20251012_143022\output

Building reference_cpp module...

[Compiler output...]

Copying to latest directory...
  ✓ reference_cpp.cp312-win_amd64.pyd → H:\...\reference_cpp.cp312-win_amd64.pyd

======================================================================
  ✅ BUILD COMPLETE
======================================================================

Build artifacts:
  Timestamped: H:\...\build\artifacts\reference\20251012_143022
  Latest:      H:\...\build\artifacts\reference\latest

Generated modules:
  - reference_cpp.cp312-win_amd64.pyd (125.2 KB)
```

## Technical Details

### Implementation: Conversion-Based Operations

The reference build uses pre-Phase 0 conversion-based operations:

```cpp
// Reference implementation (conversion-based)
static trit ref_tadd(trit a, trit b) {
    int ai = trit_to_int(a);        // Convert to int: overhead
    int bi = trit_to_int(b);        // Convert to int: overhead
    int sum = ai + bi;              // Arithmetic operation
    if (sum > 1) sum = 1;           // Branching: overhead
    if (sum < -1) sum = -1;         // Branching: overhead
    return int_to_trit(sum);        // Convert back: overhead
}

// Optimized implementation (LUT-based)
static inline trit tadd(trit a, trit b) {
    return TADD_LUT[(a << 2) | b];  // Direct table lookup
}
```

**Overhead sources in reference:**
1. Conversion to/from int
2. Arithmetic operations
3. Conditional branches
4. Function call overhead (no force inline)

**Result:** ~2-3x slower than LUT-based scalar operations

### Compiler Flags

#### Compile Flags

| Flag | Value | Purpose |
|------|-------|---------|
| `/O1` | Basic optimization | **Not** `/O2` (speed over size) |
| `/std:c++17` | C++17 standard | Match optimized build |
| `/EHsc` | Exception handling | Standard C++ exceptions |

**Explicitly excluded:**
- `/O2` - Maximum optimization
- `/GL` - Whole program optimization
- `/arch:AVX2` - SIMD instructions
- `/openmp` - Multi-threading
- `/favor:...` - Any CPU-specific tuning

#### Link Flags

**Explicitly excluded:**
- `/LTCG` - Link-time code generation
- `/OPT:REF` - Dead code removal
- `/OPT:ICF` - Identical code folding

**Result:** Minimal compiler interference with performance measurement

### Source Files

| File | Description | Implementation |
|------|-------------|----------------|
| `benchmarks/reference_cpp.cpp` | Reference implementation | Conversion-based operations |
| `ternary_algebra.h` | Header (unused) | Contains LUTs (not used by reference) |

**Note:** Reference implementation is self-contained in `reference_cpp.cpp` and does NOT use `ternary_algebra.h` LUTs.

## Directory Structure

### After Build

```
build/artifacts/
└── reference/
    ├── 20251012_143022/
    │   ├── temp/
    │   │   └── Release/
    │   │       └── benchmarks/
    │   │           ├── reference_cpp.obj           (~8 MB)
    │   │           ├── reference_cpp.*.exp         (~865 bytes)
    │   │           └── reference_cpp.*.lib         (~2 KB)
    │   └── output/
    │       └── reference_cpp.cp312-win_amd64.pyd   (~125 KB)
    └── latest/
        ├── temp/
        └── output/
```

**Note:** Reference `.pyd` is typically 20-25 KB smaller than optimized builds (less code, no SIMD).

## Benchmarking Usage

### Basic Comparison

```python
import numpy as np
import reference_cpp as ref
import ternary_simd_engine as opt

# Test data
a = np.random.randint(0, 3, size=1000000, dtype=np.uint8)
b = np.random.randint(0, 3, size=1000000, dtype=np.uint8)

# Benchmark reference
import time
t0 = time.perf_counter()
result_ref = ref.tadd(a, b)
t_ref = time.perf_counter() - t0

# Benchmark optimized
t0 = time.perf_counter()
result_opt = opt.tadd(a, b)
t_opt = time.perf_counter() - t0

# Calculate speedup
speedup = t_ref / t_opt
print(f"Reference: {t_ref*1000:.2f} ms")
print(f"Optimized: {t_opt*1000:.2f} ms")
print(f"Speedup:   {speedup:.1f}x")
```

### Using Benchmark Suite

```bash
# Build both versions
python build/scripts/setup_reference.py
python build/scripts/setup.py

# Run benchmarks (compares all available implementations)
python benchmarks/bench_phase0.py
```

**Output:**
```
Benchmark Results
=================

Small arrays (100 elements):
  reference_cpp:          250 ns/op
  ternary_simd_engine:  50 ns/op  (5.0x faster)

Medium arrays (10,000 elements):
  reference_cpp:         2500 ns/op
  ternary_simd_engine: 500 ns/op  (5.0x faster)

Large arrays (1,000,000 elements):
  reference_cpp:       500000 ns/op
  ternary_simd_engine: 50000 ns/op  (10.0x faster)
```

## Performance Characteristics

### Expected Performance

| Array Size | Reference | Standard | Speedup |
|------------|-----------|----------|---------|
| 100 | 250 ns | 50 ns | ~5x |
| 1,000 | 2.5 µs | 500 ns | ~5x |
| 10,000 | 25 µs | 5 µs | ~5x |
| 100,000 | 250 µs | 25 µs | ~10x |
| 1,000,000 | 2.5 ms | 100 µs | ~25x |

**Why speedup increases with size:**
- Small: LUT vs conversion (5x)
- Medium: + SIMD vectorization (5x)
- Large: + OpenMP parallelization (2-4x)
- **Total:** 5x × 5x × 4x = **100x** potential

### Operation Comparison

| Operation | Reference | Standard | Reason for Difference |
|-----------|-----------|----------|----------------------|
| `tadd` | Slowest | 10-50x faster | Conversion + branches vs LUT |
| `tmul` | Medium | 10-50x faster | Same as tadd |
| `tmin` | Medium | 8-40x faster | Comparison + conversion |
| `tmax` | Medium | 8-40x faster | Comparison + conversion |
| `tnot` | Fastest | 5-20x faster | Simple but still has overhead |

**Note:** Even the "fastest" reference operation is 5x slower than optimized.

## Use Cases

### 1. Optimization Impact Measurement

**Scenario:** You added a new optimization. Did it help?

```bash
# Before optimization
python build/scripts/setup.py
python benchmarks/bench_phase0.py > before.txt

# Apply optimization to code...

# After optimization
python build/scripts/setup.py
python benchmarks/bench_phase0.py > after.txt

# Compare
diff before.txt after.txt

# Also compare against reference
python build/scripts/setup_reference.py
python benchmarks/bench_phase0.py > reference.txt
```

---

### 2. Regression Detection

**Scenario:** Did recent changes hurt performance?

```bash
# Build reference once
python build/scripts/setup_reference.py

# Before changes
python build/scripts/setup.py
python benchmarks/bench_phase0.py --compare-reference > baseline.txt

# After changes
git pull
python build/scripts/setup.py
python benchmarks/bench_phase0.py --compare-reference > current.txt

# If current slower than baseline: investigate!
```

---

### 3. Publication/Paper Results

**Scenario:** Publishing performance results

```python
# Fair comparison for academic paper
reference_time = benchmark(reference_cpp.tadd)
optimized_time = benchmark(ternary_simd_engine.tadd)

print(f"Our SIMD+LUT optimizations achieve {reference_time/optimized_time:.1f}x")
print(f"speedup over baseline C++ implementation.")
# ✅ Accurate claim about YOUR optimizations
# ❌ Not conflated with C++ vs Python differences
```

## Artifact Sizes

### Binary Sizes

| Module | Size | Difference | Reason |
|--------|------|------------|--------|
| `reference_cpp.pyd` | ~125 KB | Baseline | Minimal code |
| `ternary_simd_engine.pyd` | ~145 KB | +20 KB | SIMD code variants |

**Size breakdown:**
- Reference: Conversion functions + scalar loops
- Standard: + AVX2 intrinsics + LUT tables + OpenMP runtime

### Build Artifact Sizes

| Artifact | Reference | Standard | Difference |
|----------|-----------|----------|------------|
| `.obj` | ~8 MB | ~8 MB | Similar (debug info) |
| `.pyd` | 125 KB | 145 KB | +16% |
| Total | ~8.1 MB | ~8.2 MB | Negligible |

**Observation:** Most build size is debug info in `.obj`, final `.pyd` difference is small.

## Limitations

### What Reference Build Does NOT Test

1. **Python overhead**: Both ref and optimized are C++
2. **Memory bandwidth**: Arrays fit in cache for both
3. **System load**: Single-machine, controlled environment
4. **Other languages**: Only C++ baseline

### When Reference Build Is Insufficient

**Scenario 1: Cross-language comparison**
- Need to compare Python, C++, Rust, etc.
- Reference build won't help (C++ only)

**Scenario 2: Memory-bound workloads**
- Arrays too large for cache
- Both implementations bottlenecked by DRAM

**Scenario 3: I/O-bound workloads**
- Performance dominated by disk/network
- CPU optimizations irrelevant

## Troubleshooting

### Build Issues

#### Error: "reference_cpp.cpp not found"

**Symptom:**
```
fatal error C1083: Cannot open source file: 'reference_cpp.cpp'
```

**Cause:** Script cannot find `benchmarks/reference_cpp.cpp`

**Solution:**
```bash
# Check file exists
ls benchmarks/reference_cpp.cpp

# Run from project root
cd /path/to/ternary-engine
python build/scripts/setup_reference.py
```

---

### Runtime Issues

#### Error: "Module reference_cpp has no attribute 'tadd'"

**Symptom:**
```python
>>> import reference_cpp
>>> reference_cpp.tadd
AttributeError: module 'reference_cpp' has no attribute 'tadd'
```

**Cause:** Old or corrupted build

**Solution:**
```bash
# Rebuild
python build/scripts/setup_reference.py

# Verify
python -c "import reference_cpp; print(dir(reference_cpp))"
# Should show: ['tadd', 'tmax', 'tmin', 'tmul', 'tnot']
```

---

### Benchmarking Issues

#### Issue: Reference and optimized give different results

**Symptom:**
```python
assert np.array_equal(ref.tadd(a, b), opt.tadd(a, b))
AssertionError
```

**Causes:**
1. Input out of range (not 0, 1, 2)
2. Bug in one implementation
3. Saturation behavior difference

**Solution:**
```python
# Check input validity
assert np.all((a >= 0) & (a <= 2))
assert np.all((b >= 0) & (b <= 2))

# Test simple case
a = np.array([0, 1, 2], dtype=np.uint8)
b = np.array([0, 0, 0], dtype=np.uint8)
print(ref.tadd(a, b))  # Should be [0, 1, 2]
print(opt.tadd(a, b))  # Should be [0, 1, 2]
```

## Comparison Matrix

| Aspect | Reference | Standard | PGO |
|--------|-----------|----------|-----|
| **Purpose** | Baseline | Production | Performance-critical |
| **SIMD** | ❌ No | ✅ AVX2 | ✅ AVX2 |
| **OpenMP** | ❌ No | ✅ Yes | ✅ Yes |
| **LUTs** | ❌ No | ✅ Yes | ✅ Yes |
| **Optimization** | `/O1` | `/O2 /GL /LTCG` | `/O2 /GL /LTCG:PGO` |
| **Build time** | 25-45s | 30-60s | 8-10min |
| **Binary size** | 125 KB | 145 KB | 145-155 KB |
| **Performance** | 1x (baseline) | 10-50x | 15-60x |
| **Use case** | Benchmarking | Production | Critical performance |

## Integration with CI

### GitHub Actions

```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmarks

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install setuptools pybind11 numpy

      - name: Build reference
        run: python build/scripts/setup_reference.py

      - name: Build optimized
        run: python build/scripts/setup.py

      - name: Run benchmarks
        run: python benchmarks/bench_phase0.py

      - name: Check for regressions
        run: |
          python benchmarks/bench_phase0.py --check-regression
          # Fails if optimized < 5x faster than reference
```

## Best Practices

### 1. Build Reference Once

```bash
# Build reference at start of development
python build/scripts/setup_reference.py

# Keep using same reference for all comparisons
# Don't rebuild unless reference code changes
```

### 2. Always Compare Against Reference

```bash
# Not ideal: Compare two optimized builds
python build/scripts/setup.py  # Before
python benchmarks/bench_phase0.py
# Make changes...
python build/scripts/setup.py  # After
python benchmarks/bench_phase0.py

# Better: Compare against stable reference
python benchmarks/bench_phase0.py --baseline reference_cpp
```

### 3. Document Methodology

```python
# In your benchmark report/paper:
"""
Performance measured against unoptimized C++ reference implementation
using conversion-based operations (/O1 optimization only).
Reference build eliminates language differences and isolates the impact
of our SIMD+LUT+OpenMP optimizations.
"""
```

## See Also

- [Standard Build](./setup-standard.md) - Optimized production build
- [PGO Build](./setup-pgo.md) - Profile-guided optimization
- [Artifact Organization](./artifact-organization.md) - Build output structure
- [Build System Overview](./README.md) - Complete build documentation
