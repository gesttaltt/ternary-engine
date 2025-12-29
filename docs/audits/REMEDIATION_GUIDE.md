# Remediation Guide: Underutilized Components

**Doc-Type:** Action Plan · Version 1.0 · Generated 2025-12-09

This guide provides step-by-step remediation actions for all issues identified in the underutilized components audit.

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Immediate Actions (Today)](#immediate-actions-today)
3. [Short-Term Actions (This Week)](#short-term-actions-this-week)
4. [Medium-Term Actions (This Month)](#medium-term-actions-this-month)
5. [Long-Term Actions (This Quarter)](#long-term-actions-this-quarter)
6. [Validation Checklist](#validation-checklist)

---

## Quick Reference

### Impact vs Effort Matrix

```
                    LOW EFFORT                    HIGH EFFORT
              ┌─────────────────────────────────────────────────┐
              │                                                 │
   HIGH       │  ★ Enable Dual-Shuffle    │  256-Byte LUT      │
   IMPACT     │  ★ Run PGO Build          │  AVX-512 Backend   │
              │  ★ Delete Deprecated      │  GPU Acceleration  │
              │                           │                     │
              ├─────────────────────────────────────────────────┤
              │                                                 │
   LOW        │  Remove nul file          │  ARM NEON Backend  │
   IMPACT     │  Document config flags    │  Perfetto Profiler │
              │  Add missing tests        │                     │
              │                                                 │
              └─────────────────────────────────────────────────┘

★ = Recommended Priority
```

### Command Quick Reference

```bash
# Enable dual-shuffle (5 min)
# Edit src/core/simd/backend_avx2_v2_optimized.cpp:61
# Uncomment: init_dual_shuffle_luts();

# Run PGO build (30 min)
python build/build_pgo.py full

# Delete deprecated benchmarks (1 min)
rm -rf benchmarks/deprecated/

# Clean nul artifact (1 min)
rm nul

# Run full test suite
python tests/run_tests.py

# Run benchmarks to validate
python benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py
```

---

## Immediate Actions (Today)

### Action 1: Delete Artifact File

**Time:** 1 minute
**Impact:** Clean repository
**Risk:** None

```bash
# From project root
rm nul

# Verify
ls -la | grep nul
# Should return nothing
```

### Action 2: Delete Deprecated Benchmarks

**Time:** 1 minute
**Impact:** Remove 108 KB of dead code
**Risk:** None (superseded by current benchmarks)

```bash
# From project root
rm -rf benchmarks/deprecated/

# Verify
ls benchmarks/
# Should not contain 'deprecated' directory
```

### Action 3: Enable Dual-Shuffle XOR Optimization

**Time:** 5 minutes
**Impact:** Potential 1.5× performance improvement
**Risk:** Low (code is tested, just disabled)

#### Step 1: Edit the file

```cpp
// File: src/core/simd/backend_avx2_v2_optimized.cpp
// Line: 61

// BEFORE:
void avx2_v2_init() {
    init_canonical_luts();
    // init_dual_shuffle_luts();  // TODO: Enable for additional performance
    g_avx2_v2_initialized = true;
}

// AFTER:
void avx2_v2_init() {
    init_canonical_luts();
    init_dual_shuffle_luts();  // Enabled: dual-shuffle optimization
    g_avx2_v2_initialized = true;
}
```

#### Step 2: Rebuild

```bash
python build/build.py
```

#### Step 3: Validate

```bash
# Run tests
python tests/run_tests.py

# Run benchmarks
python benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py
```

#### Step 4: Compare Performance

```python
# Before enabling (record baseline):
# tadd @ 1M: XX.X Gops/s

# After enabling (record new):
# tadd @ 1M: XX.X Gops/s

# Calculate improvement:
# improvement = (new - baseline) / baseline * 100
```

### Action 4: Document Results

After enabling dual-shuffle, update:

```markdown
<!-- In README.md, update performance section -->
## Performance
- Peak throughput: XX.X Gops/s (improved from 35.0 Gops/s via dual-shuffle)
- Dual-shuffle optimization: Enabled
```

---

## Short-Term Actions (This Week)

### Action 5: Run PGO Build

**Time:** 30 minutes
**Impact:** 5-15% performance improvement
**Risk:** Low

```bash
# Run complete PGO workflow
python build/build_pgo.py full

# This will:
# 1. Build with instrumentation
# 2. Run profiling workload
# 3. Build optimized version
```

#### Expected Output

```
Phase 1: INSTRUMENT
  Building with instrumentation... [OK]

Phase 2: PROFILE
  Running profiling workload... [OK]
  Profile data collected: 2.5 MB

Phase 3: OPTIMIZE
  Building with profile data... [OK]
  Output: build/artifacts/pgo/latest/ternary_simd_engine.pyd

PGO build complete!
```

#### Validation

```bash
# Compare standard vs PGO build
python benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py

# Record results for both:
# Standard build: XX.X Gops/s
# PGO build: XX.X Gops/s
```

### Action 6: Add Backend Switching Tests

**Time:** 2 hours
**Impact:** Validate plugin API works
**Risk:** None

Create `tests/python/test_backend_comprehensive.py`:

```python
"""
test_backend_comprehensive.py - Backend plugin system tests

Tests backend registration, switching, and dispatch.
"""
import numpy as np
import pytest

try:
    import ternary_backend
    HAS_BACKEND = True
except ImportError:
    HAS_BACKEND = False


@pytest.mark.skipif(not HAS_BACKEND, reason="ternary_backend not built")
class TestBackendComprehensive:
    """Comprehensive backend tests."""

    @pytest.fixture(autouse=True)
    def setup_backend(self):
        """Initialize backend before each test."""
        ternary_backend.init()
        yield
        ternary_backend.shutdown()

    def test_list_backends_not_empty(self):
        """Verify backends are registered."""
        backends = ternary_backend.list_backends()
        assert len(backends) >= 1

    def test_all_backends_have_required_fields(self):
        """Verify backend info structure."""
        for backend in ternary_backend.list_backends():
            assert hasattr(backend, 'name')
            assert hasattr(backend, 'version')
            assert len(backend.name) > 0

    def test_set_and_get_backend(self):
        """Verify backend switching works."""
        backends = ternary_backend.list_backends()

        for backend in backends:
            ternary_backend.set_backend(backend.name)
            active = ternary_backend.get_active()
            assert active.name == backend.name

    def test_operations_consistent_across_backends(self):
        """Verify all backends produce same results."""
        np.random.seed(42)
        a = np.random.randint(0, 3, size=1000, dtype=np.uint8)
        b = np.random.randint(0, 3, size=1000, dtype=np.uint8)

        results = {}
        for backend in ternary_backend.list_backends():
            ternary_backend.set_backend(backend.name)
            results[backend.name] = ternary_backend.tadd(a, b).copy()

        # All results should be identical
        names = list(results.keys())
        for i in range(1, len(names)):
            np.testing.assert_array_equal(
                results[names[0]], results[names[i]],
                err_msg=f"{names[0]} != {names[i]}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Action 7: Add CPU Detection Tests

**Time:** 1 hour
**Impact:** Validate CPU detection works
**Risk:** None

Create `tests/python/test_cpu_detection.py`:

```python
"""
test_cpu_detection.py - CPU capability detection tests
"""
import pytest
import ternary_simd_engine as engine


class TestCPUDetection:
    """CPU capability detection tests."""

    def test_has_avx2_returns_bool(self):
        """Verify has_avx2 returns boolean."""
        result = engine.has_avx2()
        assert isinstance(result, bool)

    def test_has_avx2_is_true(self):
        """Module requires AVX2, so this must be True."""
        assert engine.has_avx2() == True

    def test_detect_simd_level_returns_int(self):
        """Verify SIMD level is integer."""
        level = engine.detect_simd_level()
        assert isinstance(level, int)

    def test_detect_simd_level_range(self):
        """SIMD level should be 0-4."""
        level = engine.detect_simd_level()
        assert 0 <= level <= 4

    def test_simd_level_string_not_empty(self):
        """SIMD level string should be non-empty."""
        level_str = engine.simd_level_string()
        assert isinstance(level_str, str)
        assert len(level_str) > 0

    def test_simd_level_string_valid(self):
        """SIMD level string should be recognized."""
        valid_strings = ["None", "SSE2", "SSE4.1", "AVX2", "AVX-512", "Unknown"]
        level_str = engine.simd_level_string()
        assert level_str in valid_strings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Action 8: Document Configuration Flags

**Time:** 30 minutes
**Impact:** Enable production tuning
**Risk:** None

Add to `build/README.md`:

```markdown
## Build Configuration Flags

### Compile-Time Flags

| Flag | Purpose | Impact |
|------|---------|--------|
| `TERNARY_NO_SANITIZE` | Skip input validation | 3-5% faster |
| `TERNARY_FORCE_SCALAR` | Use scalar instead of SIMD | Debugging only |
| `NDEBUG` | Disable assertions | Standard release |

### Runtime Thresholds

| Constant | Default | Purpose |
|----------|---------|---------|
| `TERNARY_OMP_THRESHOLD` | 32K × cores | OpenMP parallelization threshold |
| `TERNARY_STREAM_THRESHOLD` | 1M | Streaming store threshold |
| `TERNARY_PREFETCH_DIST` | 512 | Prefetch distance (bytes) |

### Example: Production Build

```bash
# Maximum performance for validated inputs
python build/build.py --extra-flags "/DTERNARY_NO_SANITIZE"
```
```

---

## Medium-Term Actions (This Month)

### Action 9: Remove/Implement tand and tor

**Time:** 2 hours
**Impact:** Clean API
**Risk:** Low

#### Option A: Remove from API (Recommended)

```cpp
// Edit: src/core/simd/backend_plugin_api.h

struct TernaryBackend {
    const char* name;
    const char* version;
    uint32_t capabilities;

    // Operations
    void (*tadd)(const uint8_t*, const uint8_t*, uint8_t*, size_t);
    void (*tmul)(const uint8_t*, const uint8_t*, uint8_t*, size_t);
    void (*tmin)(const uint8_t*, const uint8_t*, uint8_t*, size_t);
    void (*tmax)(const uint8_t*, const uint8_t*, uint8_t*, size_t);
    void (*tnot)(const uint8_t*, uint8_t*, size_t);

    // Fused operations
    void (*fused_tnot_tadd)(const uint8_t*, const uint8_t*, uint8_t*, size_t);
    void (*fused_tnot_tmul)(const uint8_t*, const uint8_t*, uint8_t*, size_t);
    void (*fused_tnot_tmin)(const uint8_t*, const uint8_t*, uint8_t*, size_t);
    void (*fused_tnot_tmax)(const uint8_t*, const uint8_t*, uint8_t*, size_t);

    // REMOVED: void (*tand)(...);
    // REMOVED: void (*tor)(...);
};
```

Then update all backend implementations to remove the NULL entries.

### Action 10: Integrate Profiler into Debug Builds

**Time:** 4 hours
**Impact:** Enable VTune analysis
**Risk:** Low

#### Step 1: Add profiler macros to hot paths

```cpp
// File: src/core/simd/backend_avx2_v2_optimized.cpp

#include "../profiling/ternary_profiler.h"

#ifdef TERNARY_ENABLE_VTUNE
TERNARY_PROFILE_DOMAIN(g_ternary, "TernaryEngine");
TERNARY_PROFILE_TASK_NAME(g_tadd, "tadd_avx2");
TERNARY_PROFILE_TASK_NAME(g_tmul, "tmul_avx2");
TERNARY_PROFILE_TASK_NAME(g_tnot, "tnot_avx2");
#endif

void tadd_avx2_v2(const uint8_t* a, const uint8_t* b,
                  uint8_t* r, size_t n) {
    TERNARY_PROFILE_TASK_BEGIN(g_ternary, g_tadd);

    // ... existing implementation ...

    TERNARY_PROFILE_TASK_END(g_ternary);
}
```

#### Step 2: Add profiling build target

```python
# Add to build/build.py

def build_with_profiling():
    """Build with VTune profiling annotations."""
    extra_flags = ["/DTERNARY_ENABLE_VTUNE"]

    # Windows: Link against VTune ITT library
    vtune_dir = os.environ.get("VTUNE_PROFILER_DIR",
        "C:/Program Files (x86)/Intel/oneAPI/vtune/latest")
    extra_link = [f"/LIBPATH:{vtune_dir}/lib64", "libittnotify.lib"]

    # Build with extra flags
    # ...
```

### Action 11: Consolidate Build Scripts

**Time:** 8 hours
**Impact:** Simpler maintenance
**Risk:** Medium

Create unified build system:

```python
# build/build_unified.py

"""
Unified build system for Ternary Engine.

Usage:
    python build/build_unified.py [options]

Options:
    --target {standard,dense243,tritnet,all}
    --pgo           Enable Profile-Guided Optimization
    --profile       Enable VTune profiling
    --debug         Debug build
    --production    Maximum optimization
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="standard",
                        choices=["standard", "dense243", "tritnet", "all"])
    parser.add_argument("--pgo", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--production", action="store_true")

    args = parser.parse_args()

    # Build based on options
    if args.target == "standard" or args.target == "all":
        build_standard(args)

    if args.target == "dense243" or args.target == "all":
        build_dense243(args)

    if args.target == "tritnet" or args.target == "all":
        build_tritnet(args)

if __name__ == "__main__":
    main()
```

---

## Long-Term Actions (This Quarter)

### Action 12: Complete TritNet Phase 2A

**Time:** 16+ hours
**Impact:** Core innovation validation
**Risk:** Medium

See `docs/research/tritnet/TRITNET_ROADMAP.md` for detailed plan.

Key steps:
1. Fix cross-entropy loss implementation
2. Train tnot model to 100% accuracy
3. Validate learned weights
4. Document GO/NO-GO decision

### Action 13: Implement AVX-512 Backend

**Time:** 24+ hours
**Impact:** Future performance
**Risk:** Low (optional feature)

Requires:
1. Create `backend_avx512_impl.cpp`
2. Implement all operations using AVX-512
3. Add runtime detection
4. Benchmark vs AVX2

### Action 14: GPU Acceleration (TritNet)

**Time:** 40+ hours
**Impact:** ML viability
**Risk:** High (new territory)

Requires:
1. CUDA implementation of TritNet GEMM
2. Batch inference optimization
3. Memory management
4. Integration with PyTorch

---

## Validation Checklist

### After Immediate Actions

- [ ] `nul` file deleted
- [ ] `benchmarks/deprecated/` deleted
- [ ] Dual-shuffle enabled
- [ ] All tests still pass (65/65)
- [ ] Performance improved or unchanged

### After Short-Term Actions

- [ ] PGO build completes successfully
- [ ] PGO shows measurable improvement (>5%)
- [ ] Backend switching tests pass
- [ ] CPU detection tests pass
- [ ] Configuration flags documented

### After Medium-Term Actions

- [ ] tand/tor removed from API
- [ ] Profiler integrated into debug builds
- [ ] Build system consolidated
- [ ] Test coverage increased to 85%

### After Long-Term Actions

- [ ] TritNet Phase 2A complete
- [ ] GO/NO-GO decision documented
- [ ] AVX-512 backend available (optional)
- [ ] GPU acceleration explored

---

## Tracking Progress

### Create Issue Tracker

```markdown
# GitHub Issues to Create

## High Priority
- [ ] Enable dual-shuffle XOR optimization #XX
- [ ] Run and validate PGO builds #XX
- [ ] Add comprehensive backend tests #XX

## Medium Priority
- [ ] Remove deprecated benchmarks #XX
- [ ] Clean tand/tor from API #XX
- [ ] Integrate profiler annotations #XX

## Low Priority
- [ ] Consolidate build scripts #XX
- [ ] Remove unused CPU detection #XX
- [ ] Complete TritNet training #XX
```

### Progress Dashboard

```markdown
## Remediation Progress

| Category | Total | Complete | Remaining |
|----------|-------|----------|-----------|
| Immediate | 4 | 0 | 4 |
| Short-Term | 4 | 0 | 4 |
| Medium-Term | 3 | 0 | 3 |
| Long-Term | 3 | 0 | 3 |
| **Total** | **14** | **0** | **14** |

Last Updated: 2025-12-09
```

---

**Document Version:** 1.0
**Last Updated:** 2025-12-09
**Author:** Claude Code Audit
