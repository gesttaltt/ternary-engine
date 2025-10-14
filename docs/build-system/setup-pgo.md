# Profile-Guided Optimization Build (setup_pgo.py)

## Overview

`build/scripts/setup_pgo.py` implements a sophisticated 3-phase Profile-Guided Optimization (PGO) build system. PGO uses runtime profiling data to guide compiler optimizations, resulting in 5-15% performance improvements in hot code paths.

**Location:** `build/scripts/setup_pgo.py`

**Module produced:** `ternary_simd_engine.cp312-win_amd64.pyd` (PGO-optimized)

**Total build time:** 8-10 minutes (instrumentation + profiling + optimization)

**Expected improvement:** 5-15% over standard build in hot paths

## What is Profile-Guided Optimization?

PGO is a compiler optimization technique that uses actual runtime behavior to make better optimization decisions:

**Traditional Optimization:**
```
Source Code → Compiler (guesses) → Optimized Binary
```

**Profile-Guided Optimization:**
```
Source Code → Compiler → Instrumented Binary
     ↓
Instrumented Binary + Real Workload → Profile Data
     ↓
Source Code + Profile Data → Compiler → Optimized Binary
```

**Key Benefits:**
- **Better inlining decisions**: Inline frequently-called functions
- **Improved branch prediction**: Optimize common code paths
- **Efficient code layout**: Hot code contiguous in memory
- **Register allocation**: Prioritize hot variables
- **Dead code elimination**: Remove unused paths

## Three-Phase Build Process

### Phase 1: Instrumentation

**Command:** `python build/scripts/setup_pgo.py instrument`

**What it does:**
1. Compiles module with profiling instrumentation (`/LTCG:PGI`)
2. Creates `build/artifacts/pgo/instrumented/{timestamp}/`
3. Copies instrumented `.pyd` to project root

**Output:** Instrumented binary that records execution statistics

**Time:** ~45 seconds

---

### Phase 2: Profiling

**Command:** `python build/scripts/setup_pgo.py profile`

**What it does:**
1. Runs benchmark suite (`benchmarks/bench_phase0.py`)
2. Instrumented binary collects execution data
3. Generates `.pgd` (profile database) and `.pgc` (profile counters)
4. Stores profile data in `build/artifacts/pgo/pgo_data/`

**Output:** Profile data files capturing runtime behavior

**Time:** ~8 minutes (full benchmark suite)

---

### Phase 3: Optimization

**Command:** `python build/scripts/setup_pgo.py optimize`

**What it does:**
1. Compiles module with profile data (`/LTCG:PGO`)
2. Compiler uses profile data to optimize hot paths
3. Creates `build/artifacts/pgo/optimized/{timestamp}/`
4. Copies optimized `.pyd` to `latest/` and project root

**Output:** PGO-optimized binary (5-15% faster)

**Time:** ~60 seconds

## Quick Start

### Full Automatic Build

```bash
# Run all 3 phases automatically
python build/scripts/setup_pgo.py full
```

This runs: `instrument` → `profile` → `optimize` in sequence.

### Manual Phase-by-Phase

```bash
# Phase 1: Build with instrumentation
python build/scripts/setup_pgo.py instrument

# Phase 2: Collect profile data (takes ~8 minutes)
python build/scripts/setup_pgo.py profile

# Phase 3: Build optimized version
python build/scripts/setup_pgo.py optimize
```

### Check PGO Status

```bash
# View current PGO state
python build/scripts/setup_pgo.py help
```

Output:
```
Current PGO Status:
  PGO Base Dir:     build/artifacts/pgo/ ✅ exists
  Profile Database: ...pgo_data/ternary_simd_engine.pgd ✅ exists
  Instrumented Builds: 2 (20251012_143022 latest)
  Optimized Builds: 1 (20251012_150530 latest)
  Profile Counters: 3 .pgc files found
  Compiled Module:  ternary_simd_engine.cp312-win_amd64.pyd ✅
```

### Clean PGO Artifacts

```bash
# Remove all PGO data and builds
python build/scripts/setup_pgo.py clean
```

## Technical Details

### Phase 1: Instrumentation Build

#### Compiler Flags

**Compile flags** (same as standard build):
```
/O2            # Maximum optimization
/GL            # Whole program optimization
/arch:AVX2     # Enable AVX2 SIMD
/openmp        # OpenMP multi-threading
/std:c++17     # C++17 standard
/EHsc          # Exception handling
```

**Link flags** (instrumentation-specific):
```
/LTCG:PGI      # Link-Time Code Gen: Profile-Guided Instrumentation
/PGD:path      # Specify profile database location
```

**What `/LTCG:PGI` does:**
- Inserts profiling instrumentation into generated code
- Adds counters to track:
  - Function call frequencies
  - Branch taken/not-taken ratios
  - Loop iteration counts
  - Code path execution frequencies

#### Instrumented Binary Characteristics

| Property | Value | Notes |
|----------|-------|-------|
| Size | ~150 KB | Slightly larger than standard |
| Performance | ~5-10% slower | Overhead from profiling counters |
| Output location | `pgo/instrumented/{timestamp}/output/` | |
| Also copied to | Project root | For profiling convenience |

---

### Phase 2: Profile Collection

#### Profiling Workload

The script runs `benchmarks/bench_phase0.py`, which exercises:

1. **Small arrays** (100 elements)
2. **Medium arrays** (10,000 elements)
3. **Large arrays** (1,000,000 elements - triggers OpenMP)
4. **All operations**: tadd, tmul, tmin, tmax, tnot
5. **Edge cases**: All-zero, all-one, all-two values

**Total operations profiled:** ~400 million ternary operations

#### Profile Data Files

**`.pgd` (Profile-Guided Database):**
- Master database file
- Location: `build/artifacts/pgo/pgo_data/ternary_simd_engine.pgd`
- Size: ~100-200 KB
- Contains aggregated profile information

**`.pgc` (Profile Counter Files):**
- Raw counter files from instrumented runs
- Location: `build/artifacts/pgo/pgo_data/*.pgc`
- Size: ~10-50 KB each
- One file per instrumented module run

**Profile Data Retention:**
- All `.pgc` files are merged into `.pgd`
- Multiple profiling runs accumulate data (weighted average)
- Use `clean` command to reset profile data

---

### Phase 3: Optimized Build

#### Compiler Flags

**Compile flags** (same as Phase 1):
```
/O2, /GL, /arch:AVX2, /openmp, /std:c++17, /EHsc
```

**Link flags** (optimization-specific):
```
/LTCG:PGO      # Link-Time Code Gen: Profile-Guided Optimization
/PGD:path      # Read profile database from this location
```

**What `/LTCG:PGO` does:**
- Reads profile data from `.pgd` file
- Makes optimization decisions based on runtime behavior:
  - **Inlining:** Inline hot functions, don't inline cold functions
  - **Code layout:** Group hot code together (better cache locality)
  - **Branch prediction:** Optimize for common branches
  - **Register allocation:** Prioritize hot variables in registers

#### Optimizations Applied

Based on profile data, the compiler can:

**1. Function Inlining**
```cpp
// Hot path (called millions of times)
inline __m256i tadd_simd(__m256i a, __m256i b) {
    // Force inline based on profile data
}

// Cold path (rarely called)
void error_handler() {
    // Don't inline to save code size
}
```

**2. Branch Optimization**
```cpp
// Profile shows: OpenMP path taken 80% of the time
if (n >= OMP_THRESHOLD) {  // Predicted TRUE
    // Hot path - optimized layout, no branch misprediction penalty
    #pragma omp parallel
    ...
} else {
    // Cold path - can be less optimized
    ...
}
```

**3. Loop Optimizations**
```cpp
// Profile shows loop iterates millions of times
for (i = 0; i + 32 <= n; i += 32) {  // Hot loop
    // Compiler unrolls based on profile data
    // Optimized register allocation
    // Prefetching inserted
}
```

## Directory Structure

### Before PGO Build

```
build/artifacts/
└── (no pgo/ directory)
```

### After Phase 1 (Instrumentation)

```
build/artifacts/
└── pgo/
    ├── instrumented/
    │   └── 20251012_143022/
    │       ├── temp/
    │       │   └── Release/
    │       │       ├── ternary_simd_engine.obj
    │       │       ├── *.exp
    │       │       └── *.lib
    │       └── output/
    │           └── ternary_simd_engine.cp312-win_amd64.pyd  (instrumented)
    └── pgo_data/                    # Created, but empty
```

### After Phase 2 (Profiling)

```
build/artifacts/
└── pgo/
    ├── instrumented/
    │   └── (same as Phase 1)
    └── pgo_data/
        ├── ternary_simd_engine.pgd         # Profile database (150 KB)
        ├── ternary_simd_engine!1.pgc       # Profile counters (25 KB)
        ├── ternary_simd_engine!2.pgc       # Profile counters (25 KB)
        └── ternary_simd_engine!3.pgc       # Profile counters (25 KB)
```

### After Phase 3 (Optimization)

```
build/artifacts/
└── pgo/
    ├── instrumented/
    │   └── (same)
    ├── optimized/
    │   └── 20251012_150530/
    │       ├── temp/
    │       └── output/
    │           └── ternary_simd_engine.cp312-win_amd64.pyd  (optimized)
    ├── pgo_data/
    │   └── (same)
    └── latest/                      # Copy of optimized/20251012_150530/
        ├── temp/
        └── output/
```

## Performance Analysis

### Expected Improvements

| Operation | Standard Build | PGO Build | Improvement |
|-----------|----------------|-----------|-------------|
| Small arrays (<1K) | 50 ns/op | 48 ns/op | ~4% |
| Medium arrays (10K) | 500 ns/op | 475 ns/op | ~5% |
| Large arrays (1M) | 50 µs/op | 43 µs/op | ~14% |
| **Overall** | - | - | **5-15%** |

**Why larger arrays benefit more:**
- OpenMP parallelization overhead reduced
- Better cache locality in hot loops
- Branch prediction optimized for parallel paths

### Measuring Impact

```bash
# Benchmark standard build
python build/scripts/setup.py
python benchmarks/bench_phase0.py > results_standard.txt

# Benchmark PGO build
python build/scripts/setup_pgo.py full
python benchmarks/bench_phase0.py > results_pgo.txt

# Compare
diff results_standard.txt results_pgo.txt
```

## Advanced Usage

### Custom Profiling Workload

Instead of using the default benchmark suite, you can profile with your own workload:

```bash
# Phase 1: Instrument
python build/scripts/setup_pgo.py instrument

# Phase 2: Run YOUR workload (instead of bench_phase0.py)
python my_custom_workload.py  # Uses instrumented .pyd

# Phase 3: Optimize with collected data
python build/scripts/setup_pgo.py optimize
```

**Example custom workload:**

```python
# my_custom_workload.py
import numpy as np
import ternary_simd_engine as tc

# Profile your actual production workload
for _ in range(1000):
    # Simulate real usage patterns
    a = np.random.randint(0, 3, size=50000, dtype=np.uint8)
    b = np.random.randint(0, 3, size=50000, dtype=np.uint8)

    # Operations you actually use most
    result = tc.tadd(tc.tmul(a, b), tc.tnot(a))
```

**Benefits:**
- Optimizes for YOUR specific usage patterns
- Better performance for your workload
- May not generalize to other workloads

### Multiple Profiling Runs

Profile data from multiple runs is accumulated:

```bash
# Phase 1: Instrument
python build/scripts/setup_pgo.py instrument

# Phase 2: Multiple profiling runs
python benchmarks/bench_phase0.py   # Run 1: benchmarks
python my_workload.py               # Run 2: custom workload
python another_test.py              # Run 3: more data

# Profile data is merged automatically

# Phase 3: Optimize with all collected data
python build/scripts/setup_pgo.py optimize
```

**Use cases:**
- Profile different use cases (small/large arrays)
- Profile different data distributions
- Combine synthetic and real-world workloads

### Incremental PGO Updates

You can add more profile data without rebuilding:

```bash
# Existing PGO build
python build/scripts/setup_pgo.py full

# Later: Add more profile data
python build/scripts/setup_pgo.py instrument  # Rebuild instrumented
python new_workload.py                        # Add new profile data
python build/scripts/setup_pgo.py optimize    # Rebuild with merged data
```

## Troubleshooting

### Phase 1 Issues

#### Error: "Instrumentation build failed"

**Symptom:**
```
❌ Instrumentation build failed
```

**Causes:**
1. Same as standard build issues (MSVC, dependencies)
2. Insufficient disk space for profile data

**Solution:**
```bash
# Check disk space (need ~500 MB for profile data)
df -h build/artifacts/

# Clear old PGO data
python build/scripts/setup_pgo.py clean

# Retry
python build/scripts/setup_pgo.py instrument
```

---

### Phase 2 Issues

#### Error: "No instrumented module found"

**Symptom:**
```
❌ No instrumented module found. Run 'python build/scripts/setup_pgo.py instrument' first.
```

**Solution:**
```bash
# Run Phase 1 first
python build/scripts/setup_pgo.py instrument
```

---

#### Warning: "No .pgc files found"

**Symptom:**
```
⚠️  Warning: No .pgc files found
   Profile data may not have been collected properly
```

**Causes:**
1. Benchmark didn't run successfully
2. Profile data written to wrong location
3. Permissions issue

**Solution:**
```bash
# Check if instrumented .pyd exists
ls -lh ternary_simd_engine*.pyd

# Manually run benchmark with verbose output
python benchmarks/bench_phase0.py 2>&1 | tee profile.log

# Check for .pgc files
find . -name "*.pgc" -ls

# If files exist but in wrong location, move them
mv *.pgc build/artifacts/pgo/pgo_data/
```

---

#### Error: "Profiling workload failed"

**Symptom:**
```
❌ Profiling workload failed
```

**Causes:**
1. Benchmark script has errors
2. Missing dependencies
3. Instrumented binary crashes

**Solution:**
```bash
# Test instrumented binary directly
python -c "import ternary_simd_engine; print('OK')"

# Run benchmark with error details
python benchmarks/bench_phase0.py

# Check if dependencies installed
pip install numpy pytest
```

---

### Phase 3 Issues

#### Warning: "No profile data found"

**Symptom:**
```
⚠️  Warning: No profile data found
   Run 'python build/scripts/setup_pgo.py profile' first for best results
   Continuing with optimization anyway...
```

**Impact:** Build succeeds but NO PGO optimization applied (equivalent to standard build)

**Solution:**
```bash
# Run Phase 2 first
python build/scripts/setup_pgo.py profile

# Then retry Phase 3
python build/scripts/setup_pgo.py optimize
```

---

#### Error: "Optimized build failed"

**Symptom:**
```
❌ Optimized build failed
```

**Causes:**
1. Corrupted profile data
2. MSVC linker error

**Solution:**
```bash
# Clean and rebuild
python build/scripts/setup_pgo.py clean
python build/scripts/setup_pgo.py full

# If still fails, try without PGO
python build/scripts/setup.py
```

## Comparison: Standard vs PGO

| Aspect | Standard Build | PGO Build |
|--------|----------------|-----------|
| Build time | 30-60 seconds | 8-10 minutes |
| Build complexity | Simple | Complex (3 phases) |
| Performance | Fast | Faster (5-15%) |
| Binary size | 145 KB | 145-155 KB |
| Optimizations | Static | Profile-guided |
| Maintenance | Easy | Medium |
| **Use case** | **General production** | **Performance-critical** |

**When to use PGO:**
- ✅ Performance is critical
- ✅ You have representative workloads to profile
- ✅ You can afford 8-10 minute builds
- ✅ You need that extra 5-15% performance

**When to skip PGO:**
- ❌ Development/iteration (too slow)
- ❌ No clear hot paths (PGO benefit minimal)
- ❌ Workload highly variable (hard to profile)
- ❌ Standard build already fast enough

## Integration with CI/CD

### GitHub Actions Example

```yaml
# .github/workflows/build-pgo.yml
name: PGO Build

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday

jobs:
  pgo-build:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install setuptools pybind11 numpy pytest

      - name: PGO Build
        run: python build/scripts/setup_pgo.py full
        timeout-minutes: 15

      - name: Archive PGO binary
        uses: actions/upload-artifact@v3
        with:
          name: ternary-engine-pgo-${{ github.sha }}
          path: build/artifacts/pgo/latest/output/*.pyd

      - name: Run benchmarks
        run: python benchmarks/bench_phase0.py

      - name: Compare with standard build
        run: |
          python build/scripts/setup.py
          python benchmarks/bench_phase0.py --compare
```

## Best Practices

### 1. Profile Representative Workloads

✅ **Do:**
```python
# Profile actual production patterns
for _ in range(1000):
    real_data = load_production_data()
    result = tc.tadd(real_data.a, real_data.b)
```

❌ **Don't:**
```python
# Don't profile trivial/unrealistic workloads
a = np.array([1], dtype=np.uint8)
b = np.array([2], dtype=np.uint8)
tc.tadd(a, b)  # Too simple, won't help PGO
```

### 2. Profile Different Scenarios

```bash
# Profile multiple workload sizes
python workload_small.py   # <1K elements
python workload_medium.py  # 10K elements
python workload_large.py   # 1M elements
```

### 3. Keep Profile Data Fresh

```bash
# Reprofile after significant code changes
git diff --stat ternary_simd_engine.cpp

# If changes are major:
python build/scripts/setup_pgo.py clean
python build/scripts/setup_pgo.py full
```

### 4. Version Control

```gitignore
# .gitignore
build/artifacts/pgo/       # Don't commit build artifacts
!build/artifacts/pgo/.gitkeep
```

## See Also

- [Standard Build](./setup-standard.md) - Simpler alternative without PGO
- [Artifact Organization](./artifact-organization.md) - PGO artifact structure
- [Build System Overview](./README.md) - Complete documentation index
- [Microsoft PGO Documentation](https://docs.microsoft.com/en-us/cpp/build/profile-guided-optimizations)
