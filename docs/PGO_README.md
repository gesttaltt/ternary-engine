# Profile-Guided Optimization (PGO) - Ternary Engine

**Status**: ✅ IMPLEMENTED
**Date**: 2025-10-11
**Phase**: 0 (Compiler Optimizations)

---

## What is PGO?

Profile-Guided Optimization uses **actual runtime behavior** to guide compiler optimizations:

1. **Instrument**: Build code with profiling instrumentation
2. **Profile**: Run representative workloads to collect runtime data
3. **Optimize**: Rebuild using collected data to optimize hot paths

**Expected Benefits**:
- 5-15% performance improvement in hot paths
- Better function inlining decisions
- Improved branch prediction
- Optimized register allocation based on actual usage

---

## Quick Start

### Option 1: Automated Full PGO Build

```bash
python build_pgo.py full
```

This runs all 3 phases automatically (~10-15 minutes total).

### Option 2: Manual Step-by-Step

```bash
# Phase 1: Build with instrumentation (~2 minutes)
python build_pgo.py instrument

# Phase 2: Run profiling workload (~8 minutes)
python build_pgo.py profile

# Phase 3: Build optimized version (~2 minutes)
python build_pgo.py optimize
```

### Clean PGO Data

```bash
python build_pgo.py clean
```

---

## Detailed Usage

### Phase 1: Instrumentation Build

```bash
python build_pgo.py instrument
```

**What it does**:
- Compiles code with `/LTCG:PGI` flag (Profile-Guided Instrumentation)
- Creates `pgo_data/` directory for profile storage
- Generates instrumented module that records runtime behavior

**Output**:
- `ternary_simd_engine.cp312-win_amd64.pyd` (instrumented)
- `pgo_data/ternary_simd_engine.pgd` (profile database, empty)

### Phase 2: Profile Collection

```bash
python build_pgo.py profile
```

**What it does**:
- Runs `benchmarks/bench_phase0.py` (full benchmark suite)
- Collects runtime profiling data during execution
- Stores profile counters in `.pgc` files
- Merges counters into profile database

**Duration**: ~8 minutes (full benchmark suite)

**Output**:
- `*.pgc` files (profile counter data)
- Updated `pgo_data/ternary_simd_engine.pgd` (contains profile data)

**Representative Workload**: The benchmark suite covers:
- Small arrays (8-31 elements) → scalar code paths
- Medium arrays (32-1K elements) → SIMD code paths
- Large arrays (1M+ elements) → OpenMP threaded paths
- All 5 operations (tadd, tmul, tmin, tmax, tnot)

### Phase 3: Optimized Build

```bash
python build_pgo.py optimize
```

**What it does**:
- Compiles code with `/LTCG:PGO` flag (Profile-Guided Optimization)
- Uses collected profile data to optimize:
  - Function inlining (inline frequently called functions)
  - Branch prediction (optimize for common code paths)
  - Code layout (arrange hot code together for cache locality)
  - Register allocation (prioritize frequently accessed variables)

**Output**:
- `ternary_simd_engine.cp312-win_amd64.pyd` (PGO-optimized)

---

## Verification

### Before PGO (Baseline)

```bash
# Build without PGO
python build.py

# Run benchmark
python benchmarks/bench_phase0.py
```

### After PGO

```bash
# Build with PGO
python build_pgo.py full

# Run benchmark again
python benchmarks/bench_phase0.py
```

### Expected Improvements

Based on MSVC PGO documentation:
- **Hot paths**: 5-15% speedup
- **Code layout**: Better instruction cache utilization
- **Branch prediction**: 10-30% fewer branch mispredictions
- **Overall**: 3-10% average performance improvement

**Caveat**: Improvements depend on workload predictability. If runtime behavior matches profiling workload, gains are significant. If behavior differs, gains are minimal.

---

## Technical Details

### MSVC PGO Flags

**Phase 1 (Instrumentation)**:
```python
extra_link_args=[
    '/LTCG:PGI',     # Link-Time Code Generation: Profile-Guided Instrumentation
    '/PGD:pgo_data/ternary_simd_engine.pgd',  # Profile database location
]
```

**Phase 3 (Optimization)**:
```python
extra_link_args=[
    '/LTCG:PGO',     # Link-Time Code Generation: Profile-Guided Optimization
    '/PGD:pgo_data/ternary_simd_engine.pgd',  # Use this profile database
]
```

### Profile Data Files

| File | Description |
|------|-------------|
| `*.pgc` | Profile counter data (generated during profiling) |
| `*.pgd` | Profile database (merged from .pgc files) |
| `pgo_data/` | Directory containing all profile data |

### PGO vs LTO

**Common Confusion**: `/GL` + `/LTCG` are **Link-Time Optimization (LTO)**, not PGO.

| Feature | LTO | PGO |
|---------|-----|-----|
| **What** | Optimizes across compilation units | Uses runtime profiling data |
| **Flags** | `/GL` + `/LTCG` | `/LTCG:PGI` → `/LTCG:PGO` |
| **Benefit** | 5-20% (compile-time analysis) | 5-15% (runtime-guided) |
| **Cost** | Longer build time | 3-phase build + profiling run |
| **Data** | None required | Requires representative workload |

**Both are enabled** in PGO builds (PGO builds on top of LTO).

---

## Troubleshooting

### Problem: No profile data collected

**Symptoms**:
- No `*.pgc` files after Phase 2
- Empty or missing `*.pgd` file

**Solutions**:
1. Check instrumented module was used:
   ```bash
   python -c "import ternary_simd_engine; print(ternary_simd_engine.__file__)"
   ```
2. Verify profiling workload ran successfully
3. Check PGO directory exists: `pgo_data/`

### Problem: Build fails in Phase 1 or 3

**Symptoms**:
- Linker errors about `/LTCG` or `/PGD`

**Solutions**:
1. Ensure MSVC is installed (not MinGW)
2. Check Visual Studio version supports PGO (2015+)
3. Try absolute path for `/PGD`: `/PGD:C:/path/to/pgo_data/...`

### Problem: No performance improvement

**Possible Reasons**:
1. **Workload mismatch**: Profiling workload differs from actual usage
   - Solution: Run Phase 2 with your actual use case
2. **Already optimal**: Code already well-optimized (Phase 0 + 0.5)
   - Expected: PGO gives diminishing returns on top of existing optimizations
3. **Cache-bound**: Performance limited by memory, not CPU
   - PGO cannot improve memory bandwidth

---

## Integration with Regular Build

### Option 1: Always use PGO (Recommended for releases)

Replace `build.py` with:
```bash
python build_pgo.py full
```

### Option 2: Conditional PGO

Add environment variable check:
```python
# In build.py
import os
use_pgo = os.environ.get('USE_PGO', '0') == '1'

if use_pgo:
    extra_link_args.append('/LTCG:PGO')
else:
    extra_link_args.append('/LTCG')
```

Build with PGO:
```bash
set USE_PGO=1
python build.py
```

### Option 3: Separate PGO builds

Keep `build.py` for development builds (fast iteration).
Use `build_pgo.py` for release builds (maximum performance).

---

## Benchmarking PGO Impact

### Methodology

1. **Baseline** (No PGO):
   ```bash
   python build.py
   python benchmarks/bench_phase0.py > results_baseline.json
   ```

2. **With PGO**:
   ```bash
   python build_pgo.py full
   python benchmarks/bench_phase0.py > results_pgo.json
   ```

3. **Compare**:
   ```python
   import json
   baseline = json.load(open("results_baseline.json"))
   pgo = json.load(open("results_pgo.json"))

   improvement = (baseline["peak_throughput"] - pgo["peak_throughput"]) / baseline["peak_throughput"]
   print(f"PGO improvement: {improvement*100:.1f}%")
   ```

### Expected Results

Based on similar projects with PGO:
- **Small arrays** (<32 elements): 0-5% (scalar, limited optimization potential)
- **Medium arrays** (32-1K): 5-15% (SIMD hot paths benefit most)
- **Large arrays** (>100K): 3-8% (memory-bound, less CPU optimization impact)

---

## Maintenance

### When to Re-profile

Re-run PGO after:
- ✅ **Major code changes** (new algorithms, loop restructuring)
- ✅ **New CPU architecture** (different branch prediction behavior)
- ✅ **Workload changes** (different array sizes, operation mixes)

Do NOT re-profile after:
- ❌ Minor bug fixes
- ❌ Comment changes
- ❌ Variable renames

### Profile Data Versioning

Profile data is **compiler-specific** and **code-specific**:
- Different MSVC versions → different profile format
- Different source code → profile data invalid

**Best Practice**: Regenerate profile data for each release build.

---

## GCC/Clang PGO (Future)

For cross-platform PGO support, add GCC/Clang flags:

**GCC/Clang Phase 1** (Instrumentation):
```python
extra_compile_args=['-fprofile-generate'],
extra_link_args=['-fprofile-generate']
```

**GCC/Clang Phase 3** (Optimization):
```python
extra_compile_args=['-fprofile-use'],
extra_link_args=['-fprofile-use']
```

---

## References

- [MSVC Profile-Guided Optimization](https://docs.microsoft.com/en-us/cpp/build/profile-guided-optimizations)
- [GCC Profile-Guided Optimization](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-fprofile-generate)
- [Clang Profile-Guided Optimization](https://clang.llvm.org/docs/UsersManual.html#profile-guided-optimization)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-11
**Status**: OPT-114 fully implemented and documented
