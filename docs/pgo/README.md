# Profile-Guided Optimization (PGO) for Ternary Engine

**Doc-Type:** PGO Documentation Hub · Version 2.0 · Date 2025-11-23

---

## Overview

Profile-Guided Optimization uses actual runtime behavior to guide compiler optimizations, providing 5-15% additional performance gains on top of the already-optimized baseline.

**recommended** - Use Clang PGO (works perfectly with Python extensions)
**fallback** - MSVC PGO available but has known DLL lifecycle limitations

---

## Quick Start

### Recommended: Clang PGO

```bash
# Install Clang (see CLANG_INSTALLATION.md)
# Then run unified PGO build
python build/build_pgo_unified.py --clang
```

**expected_gain** - 5-15% performance improvement
**works_with** - Python extensions (no DLL limitations)

### Auto-Detect

```bash
# Automatically prefers Clang if available, falls back to MSVC
python build/build_pgo_unified.py
```

### MSVC Fallback

```bash
# Has known limitations with Python extensions
python build/build_pgo.py full
```

**note** - See PGO_LIMITATIONS.md for why MSVC PGO doesn't work well with Python

---

## Documentation Files

### CLANG_INSTALLATION.md

Complete installation guide for Clang on Windows, Linux, and macOS.

**contents**:
- Why Clang for PGO
- Platform-specific installation instructions
- Verification steps
- Troubleshooting common issues

### PGO_LIMITATIONS.md

Technical analysis of MSVC PGO limitations with Python extensions.

**contents**:
- Root cause: DLL lifecycle and .pgc file generation
- Why MSVC PGO fails with Python extensions
- Attempted fixes and why they don't work
- Clang as the solution

### PGO_AND_PATHS_FIX_REPORT.md

Historical report documenting PGO infrastructure improvements.

**contents**:
- MSVC PGO enhancement attempts
- Path reorganization
- Build script consolidation

---

## Clang vs MSVC PGO

| Feature | Clang PGO | MSVC PGO |
|---------|-----------|----------|
| **Works with Python Extensions** | ✅ Yes | ❌ No (DLL unload issue) |
| **Profile Collection** | ✅ Immediate (.profraw) | ❌ Requires DLL unload (.pgc) |
| **Cross-Platform** | ✅ Windows, Linux, macOS | ❌ Windows only |
| **Performance Gain** | ✅ 5-15% | N/A (no profile data collected) |
| **Setup Complexity** | ✅ Simple | ⚠️ Complex |
| **Maintenance** | ✅ Active | ⚠️ Deprecated |

**recommendation** - Always use Clang PGO when possible

---

## How PGO Works

### Traditional Compilation

```
Source Code → Compiler Optimizations → Binary
             (based on static analysis)
```

### Profile-Guided Optimization

```
Source Code → Instrumented Binary → Run Workload → Profile Data
                                                         ↓
                              Optimized Binary ← Compiler (guided by profiles)
```

**key_benefit** - Compiler knows which code paths are hot, optimizes accordingly

---

## Clang PGO Workflow (4 Phases)

### Phase 1: Instrumentation Build

```bash
CPPFLAGS="-fprofile-generate=pgo_data/profiles" python build/build.py
```

**output** - `ternary_simd_engine.pyd` with profiling hooks
**overhead** - ~10% slower than standard build
**purpose** - Collect runtime statistics

### Phase 2: Profile Collection

```bash
python benchmarks/bench_phase0.py --quick
```

**output** - `.profraw` files in `pgo_data/profiles/`
**writes** - Immediately during execution (no DLL unload needed)
**duration** - ~2-5 minutes

### Phase 3: Profile Merging

```bash
llvm-profdata merge -output=pgo_data/merged.profdata pgo_data/profiles/*.profraw
```

**output** - `pgo_data/merged.profdata`
**format** - Binary profile database
**size** - Typically 100-500 KB

### Phase 4: Optimized Build

```bash
CPPFLAGS="-fprofile-use=pgo_data/merged.profdata" python build/build.py
```

**output** - PGO-optimized `ternary_simd_engine.pyd`
**optimization** - Hot paths inlined, cold paths optimized for size
**result** - 5-15% faster than standard build

---

## Unified Script (Automatic)

The `build_pgo_unified.py` script automates all 4 phases:

```bash
# Full workflow (auto-detect compiler)
python build/build_pgo_unified.py

# Force Clang
python build/build_pgo_unified.py --clang

# Force MSVC (not recommended)
python build/build_pgo_unified.py --msvc

# Clean PGO data before starting
python build/build_pgo_unified.py --clang --clean
```

---

## Performance Analysis

### Expected Improvements

**baseline** - Standard build: 7,000-19,000× vs Pure Python
**pgo_gain** - Additional 5-15% on top of baseline
**total** - ~7,350-21,850× vs Pure Python

### Hottest Code Paths (PGO Benefits Most)

1. **SIMD loop dispatch** - Branch prediction improvement
2. **Array size thresholds** - Better inlining decisions
3. **Scalar tail handling** - Optimized for rare case
4. **Type checking** - Reduced overhead in hot paths

### Benchmark Validation

```bash
# Build and benchmark standard
python build/build.py
python benchmarks/bench_phase0.py > standard.txt

# Build and benchmark PGO
python build/build_pgo_unified.py --clang
python benchmarks/bench_phase0.py > pgo.txt

# Compare results
diff standard.txt pgo.txt
```

---

## Troubleshooting

### Clang Not Found

**symptom** - `❌ ERROR: Clang or llvm-profdata not found`

**solution** - See CLANG_INSTALLATION.md for platform-specific instructions

**quick_install**:
- **Windows:** https://releases.llvm.org/download.html
- **Linux:** `sudo apt install clang llvm`
- **macOS:** `brew install llvm`

### No .profraw Files Generated

**symptom** - Phase 3 fails with "No .profraw files found"

**possible_causes**:
1. Benchmarks didn't run successfully
2. Profile directory doesn't exist
3. Insufficient disk space

**solution**:
```bash
# Check profile directory
ls pgo_data/profiles/

# Manually run benchmarks
python benchmarks/bench_phase0.py --quick

# Verify .profraw files
find pgo_data -name "*.profraw"
```

### Profile Merge Fails

**symptom** - `llvm-profdata merge` errors

**error**: `Malformed instrumentation profile data`

**solution**:
- Delete `pgo_data/` directory
- Ensure same Clang version for all phases
- Rebuild from Phase 1

```bash
rm -rf pgo_data/
python build/build_pgo_unified.py --clang --clean
```

### MSVC Used Instead of Clang

**symptom** - Script uses MSVC even though Clang is installed

**solution**:
```bash
# Force Clang
python build/build_pgo_unified.py --clang

# Verify Clang detected
clang-cl --version
llvm-profdata --version
```

---

## Advanced Usage

### Custom Profiling Workload

```bash
# Phase 1: Build instrumented
CPPFLAGS="-fprofile-generate=pgo_data/profiles" python build/build.py

# Phase 2: Run custom workload
python my_custom_benchmark.py
python another_important_workload.py

# Phase 3: Merge profiles
llvm-profdata merge -output=pgo_data/merged.profdata pgo_data/profiles/*.profraw

# Phase 4: Optimized build
CPPFLAGS="-fprofile-use=pgo_data/merged.profdata" python build/build.py
```

### Analyze Profile Data

```bash
# View function coverage
llvm-profdata show --all-functions pgo_data/merged.profdata

# Detailed summary
llvm-profdata show --detailed-summary pgo_data/merged.profdata

# Top hot functions
llvm-profdata show --topn=10 pgo_data/merged.profdata
```

### Profile-Specific Optimizations

Clang uses profiles to optimize:

1. **Function inlining** - Inline hot functions, don't inline cold ones
2. **Branch prediction** - Optimize likely branches, move unlikely code out of hot path
3. **Register allocation** - Prioritize registers for frequently-used variables
4. **Code layout** - Group hot code together for better cache locality

---

## Migration Guide

### From MSVC PGO (Old)

**old_workflow**:
```bash
python build/build_pgo.py full  # Doesn't collect profile data
```

**new_workflow**:
```bash
python build/build_pgo_unified.py --clang  # Actually works
```

**benefits**:
- Profile data actually collected
- 5-15% real performance gain
- Cross-platform support
- Simpler workflow

### From Manual Clang PGO

**old_workflow** (manual):
```bash
# Phase 1
export CPPFLAGS="-fprofile-generate=profiles"
python build/build.py

# Phase 2
python benchmarks/bench_phase0.py

# Phase 3
llvm-profdata merge -output=merged.profdata profiles/*.profraw

# Phase 4
export CPPFLAGS="-fprofile-use=merged.profdata"
python build/build.py
```

**new_workflow** (automated):
```bash
python build/build_pgo_unified.py --clang
```

**benefits** - One command, automatic error handling, better UX

---

## References

**Clang PGO Documentation:** https://clang.llvm.org/docs/UsersManual.html#profile-guided-optimization

**llvm-profdata Manual:** https://llvm.org/docs/CommandGuide/llvm-profdata.html

**LLVM Downloads:** https://releases.llvm.org/download.html

**PGO Best Practices:** https://llvm.org/docs/HowToBuildWithPGO.html

---

**Version:** 2.0 · **Date:** 2025-11-23 · **Status:** Production-Ready · **Recommended:** Clang PGO
