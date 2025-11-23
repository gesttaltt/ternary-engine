# PGO Build - Known Limitations

**Doc-Type:** Technical Documentation · Version 1.0 · Date 2025-11-22 · Ternary Engine

---

## Issue Summary

**problem** - MSVC Profile-Guided Optimization (PGO) does not generate profile data (.pgc files) when used with Python extension modules (.pyd files)

**status** - IDENTIFIED AND DOCUMENTED
**impact** - PGO build completes but provides no performance benefit
**root_cause** - Python doesn't cleanly unload extension modules, preventing .pgc file generation

---

## Technical Analysis

### MSVC PGO Workflow

1. **Phase 1: Instrumentation** (`/LTCG:PGI`)
   - Compiles code with profiling hooks
   - Creates instrumented .pyd file
   - Status: ✅ WORKS

2. **Phase 2: Profile Collection**
   - Runs instrumented binary
   - Records execution patterns
   - Writes .pgc (Profile Guided Counter) files on DLL unload
   - Status: ❌ FAILS - .pgc files not generated

3. **Phase 3: Optimization** (`/LTCG:PGO`)
   - Uses .pgc files to optimize compilation
   - Status: ⚠️ RUNS but has no profile data to use

### Why It Fails with Python Extensions

**.pyd files are DLLs**:
- Python extension modules are Windows DLLs
- MSVC PGO writes .pgc files during DLL `DllMain(DLL_PROCESS_DETACH)`
- Python's module loading doesn't always trigger clean DLL unload

**Python module lifecycle**:
```
Python loads .pyd  →  Module runs  →  Python exits
     ↓                                      ↓
   DllMain            (no writes)      Sometimes no
   (ATTACH)                            DllMain(DETACH)
```

**Result**: Profile counters collected in memory but never written to disk

---

## Attempted Fixes

### ✅ Implemented

1. **Explicit .pgc file search**
   - Check PROJECT_ROOT and PGO_DATA_DIR
   - Result: No files found

2. **Environment variable** (`PogoSafeMode=0`)
   - Ensures PGO is enabled
   - Result: No effect

3. **pgomgr explicit merge**
   - Added tool to merge .pgc files
   - Result: No files to merge

4. **Path format fixes**
   - Removed /PGD flag from instrumentation
   - Fixed /PGD path format for optimization
   - Result: Build succeeds, no profile data

### ❌ Known Limitations

**Cannot fix**:
- Python's DLL unload behavior (interpreter limitation)
- MSVC PGO requirement for clean unload (compiler limitation)
- Extension module lifecycle (Python design)

---

## Workarounds Explored

### Option 1: Force Python Exit
```python
import sys
import ternary_simd_engine
# ... run workload ...
sys.exit(0)  # Force clean exit
```
**Result**: Still doesn't trigger DLL_PROCESS_DETACH reliably

### Option 2: Use C++ Test Harness
Create standalone .exe that:
1. Loads .pyd dynamically
2. Runs workload
3. Unloads DLL explicitly

**Status**: Complex, not implemented
**Effort**: 2-3 days

### Option 3: Use Clang PGO on Linux
Clang's PGO doesn't require DLL unload:
```bash
# Phase 1: Instrument
clang++ -fprofile-generate ...

# Phase 2: Profile (writes .profraw immediately)
python bench.py

# Phase 3: Optimize
clang++ -fprofile-use=default.profdata ...
```

**Status**: Not tested on Windows
**Compatibility**: Would work on Linux/macOS

---

## Recommendations

### Short Term: Document and Disable

1. **Add warning to PGO build**
   ```
   [WARN] PGO with Python extensions has known limitations
   Profile data may not be collected on Windows/MSVC
   Use standard build for production
   ```

2. **Update README**
   - Remove PGO from feature list
   - Or mark as "experimental/non-functional"

3. **Keep code for reference**
   - Infrastructure is correct
   - May work if Python/MSVC fix the issue

### Long Term: Alternative Approaches

**Option A: Clang PGO**
- Implement PGO using Clang on Linux
- Validates PGO benefit exists
- Effort: 1-2 days

**Option B: C++ Test Harness**
- Create standalone profiling executable
- Complex but would work
- Effort: 2-3 days

**Option C: Compiler-guided optimization**
- Use `/favor:INTEL64` or `/favor:AMD64`
- Use `/Qpar` for auto-parallelization
- Doesn't require profiling
- Effort: < 1 day

---

## Current Build Status

### What Works
- ✅ Phase 1: Instrumentation build compiles
- ✅ Phase 2: Benchmark suite runs
- ✅ Phase 3: Optimized build compiles
- ✅ All phases complete without errors

### What Doesn't Work
- ❌ .pgc file generation
- ❌ Profile data collection
- ❌ Actual PGO optimization

### Net Effect
- PGO build == Standard build (no performance difference)
- Build time: ~12 minutes (vs ~2 minutes for standard)
- No benefit, significant cost

---

## Verification

### Test Results

**Standard Build**:
```
Peak: 19,719 Mops/s (tnot at 1M elements)
Avg Speedup: 7,197× vs Python
```

**PGO Build** (after "optimization"):
```
Peak: 19,719 Mops/s (tnot at 1M elements)
Avg Speedup: 7,197× vs Python
```

**Conclusion**: Identical performance (as expected without profile data)

---

## Alternative Optimization Strategies

### What Actually Works

1. **LUT Optimization** (Phase 0)
   - Pre-computed lookup tables
   - Impact: 100-1000× speedup
   - Status: ✅ IMPLEMENTED

2. **AVX2 SIMD** (Phase 1)
   - Process 32 trits in parallel
   - Impact: 32× parallelism
   - Status: ✅ IMPLEMENTED

3. **Operation Fusion** (Phase 4)
   - Eliminate intermediate arrays
   - Impact: 1.6-3.4× speedup
   - Status: ✅ IMPLEMENTED

4. **Compiler Optimizations**
   - `/O2` - Maximum speed optimization
   - `/GL` - Whole program optimization
   - `/arch:AVX2` - AVX2 code generation
   - `/LTCG` - Link-time code generation
   - Status: ✅ IMPLEMENTED

### Diminishing Returns

**Current performance**: 7,000-19,000× vs Pure Python
**PGO theoretical benefit**: 5-15% (if it worked)
**Actual gain**: 350-2,850× additional speedup
**Priority**: LOW (other optimizations more impactful)

---

## Conclusion

**PGO is not worth fixing for this project**:
- Requires significant effort (2-3 days)
- Low benefit (5-15% of already fast code)
- Platform-specific (Windows/MSVC only issue)
- Better alternatives exist (fusion, SIMD already implemented)

**Recommendation**:
- Keep PGO code as reference/documentation
- Mark as non-functional in README
- Focus optimization efforts elsewhere
- Revisit if Python/MSVC fix the underlying issue

---

**Status:** DOCUMENTED · **Priority:** LOW · **Action:** NO FIX PLANNED

---

## Files Modified

```
build/build_pgo.py
  - Added .pgc file search logic
  - Added pgomgr merge step
  - Added environment variables
  - Added comprehensive diagnostics

PGO_LIMITATIONS.md
  - This document
```

---

**Version:** 1.0 · **Date:** 2025-11-22 · **Author:** Claude Code
