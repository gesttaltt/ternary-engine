# Underutilized Components Analysis

**Doc-Type:** Technical Audit · Version 1.0 · Generated 2025-12-09

This report identifies code, infrastructure, and features in the Ternary Engine repository that are not being used to their full potential.

---

## Detailed Documentation

This executive summary is supported by detailed audit documents:

| Document | Description |
|----------|-------------|
| [Disabled Optimizations](../docs/audits/DISABLED_OPTIMIZATIONS.md) | Deep dive into 5 disabled performance optimizations |
| [Unused Infrastructure](../docs/audits/UNUSED_INFRASTRUCTURE.md) | Analysis of 7 underutilized systems |
| [Dead Code Inventory](../docs/audits/DEAD_CODE_INVENTORY.md) | Complete inventory of ~3,400 lines of dead code |
| [Test Coverage Gaps](../docs/audits/TEST_COVERAGE_GAPS.md) | Analysis of missing test coverage |
| [Remediation Guide](../docs/audits/REMEDIATION_GUIDE.md) | Step-by-step action plan with commands |
| [Audit Index](../docs/audits/README.md) | Overview and quick reference |

---

## Executive Summary

| Category | Count | Potential Impact |
|----------|-------|------------------|
| **Disabled Optimizations** | 3 major | 15-45% performance unclaimed |
| **Unused Infrastructure** | 7 systems | Wasted development effort |
| **Partial Implementations** | 6 features | Blocking production readiness |
| **Dead Code** | 2 operations + 10 files | Code debt |
| **Test Coverage Gaps** | 4 areas | Reliability risk |

**Estimated Performance Left on Table:** 20-40% improvement available

---

## 1. Disabled Optimizations (High Impact)

### 1.1 Dual-Shuffle XOR Optimization (DISABLED)

**Location:** [src/core/simd/opt_dual_shuffle_xor.h](src/core/simd/opt_dual_shuffle_xor.h)

**Status:** 400+ lines of production-ready code, commented out

**Evidence:**
```cpp
// backend_avx2_v2_optimized.cpp:61
// init_dual_shuffle_luts();  // TODO: Enable for additional performance
```

**Claimed Performance:**
- AMD Zen2/3/4: 1.5-1.7× speedup
- Intel Alder Lake: 1.2-1.5× speedup
- Expected: 35-45 Gops/s (up from 28-35 Gops/s)

**Why Disabled:** Unknown - code appears complete with unit tests defined

**Recommendation:** Enable and benchmark; if claims hold, integrate into default path

---

### 1.2 256-Byte LUT Expansion (INCOMPLETE)

**Location:** [src/core/simd/opt_lut_256byte_expanded.h:268](src/core/simd/opt_lut_256byte_expanded.h)

**Status:** Architecture designed, placeholder implementation

**Evidence:**
```cpp
// TODO: Full implementation requires splitting 256-byte LUT into chunks
// and using multiple shuffles with index calculation
```

**Claimed Performance:** 10-20% improvement

**Why Incomplete:** Marked "Phase 6" with no timeline

**Recommendation:** Either implement or remove from roadmap claims

---

### 1.3 Profile-Guided Optimization (NEVER RUN)

**Location:** [build/build_pgo.py](build/build_pgo.py) (400+ lines)

**Status:** Complete 3-phase PGO system, never executed in CI

**Evidence:**
- No PGO artifacts in `build/artifacts/pgo/`
- No CI job references PGO builds
- Documentation says "5-15% improvement available"

**Why Unused:** OpenMP CI crashes documented (now fixed but not re-enabled)

**Recommendation:** Add PGO to CI pipeline for release builds

---

## 2. Unused Infrastructure

### 2.1 Profiler Framework (INTEGRATED BUT UNUSED)

**Location:** [src/core/profiling/ternary_profiler.h](src/core/profiling/ternary_profiler.h) (294 lines)

**Status:** Three backends defined, zero usage

| Backend | Status | Usage |
|---------|--------|-------|
| VTune (ITT API) | Fully integrated | **Never called** |
| NVTX (GPU) | Framework ready | **Never called** |
| Perfetto | Stub only | **Never called** |

**Evidence:**
```bash
# These flags are documented but never applied:
-DTERNARY_ENABLE_VTUNE
-DTERNARY_ENABLE_NVTX
-DTERNARY_ENABLE_PERFETTO
```

**Recommendation:** Add profiling build target, document profiling workflow

---

### 2.2 Backend Plugin API (PARTIALLY USED)

**Location:** [src/engine/bindings_backend_api.cpp](src/engine/bindings_backend_api.cpp)

**Status:** Full backend dispatch system built, limited exposure

| Function | Exposed | Tested | Used in Production |
|----------|---------|--------|-------------------|
| `init()` | Yes | No | No |
| `list_backends()` | Yes | No | No |
| `set_backend()` | Yes | No | No |
| `get_active()` | Yes | No | No |
| `dispatch_*()` | Yes | No | No |

**Evidence:**
- `ternary_backend` module built but not documented
- No imports in benchmarks or tests
- No integration tests for backend switching

**Recommendation:** Either document and test, or remove complexity

---

### 2.3 CPU Capability Detection (OVER-ENGINEERED)

**Location:** [src/core/simd/cpu_simd_capability.h](src/core/simd/cpu_simd_capability.h)

**Defined but never called:**
```cpp
has_avx512bw()     // No AVX-512 backend exists
has_neon()         // No ARM backend exists
has_sve()          // No SVE backend exists
detect_best_simd() // Only called once at init
simd_level_name()  // Exposed to Python, never tested
```

**Recommendation:** Remove unused detection or implement corresponding backends

---

### 2.4 Optimization Config Flags (NEVER APPLIED)

**Location:** [src/core/config/optimization_config.h](src/core/config/optimization_config.h)

| Flag | Purpose | Usage |
|------|---------|-------|
| `TERNARY_NO_SANITIZE` | 3-5% gain for validated code | Never built with |
| `TERNARY_OMP_THRESHOLD` | Runtime OpenMP tuning | Never overridden |
| `TERNARY_STREAM_THRESHOLD` | Cache bypass tuning | Hardcoded default |
| `TERNARY_PREFETCH_DIST` | Prefetch distance | Never tuned |

**Recommendation:** Document production build flags, add to build.py options

---

### 2.5 Build Script Variants (RARELY USED)

**Location:** [build/](build/)

| Script | Purpose | Last Used |
|--------|---------|-----------|
| `build.py` | Standard build | Active |
| `build_dense243.py` | Dense243 | Active |
| `build_pgo.py` | PGO builds | Never |
| `build_pgo_unified.py` | Cross-platform PGO | Never |
| `build_backend.py` | Backend-specific | Unknown |
| `build_reference.py` | Reference impl | Test harness only |
| `build_test_packing.py` | Packing tests | Standalone |
| `build_tritnet_gemm.py` | TritNet GEMM | Experimental |

**Recommendation:** Consolidate into single build system with options

---

## 3. Partial Implementations (TODOs)

### 3.1 AVX2 Packing Operations

**Location:** [src/core/packing/](src/core/packing/)

```cpp
// octet_pack.h:204
static inline void octet_pack_avx2(...)
    // TODO: Implement AVX2 version in Phase 6

// octet_pack.h:216
static inline void octet_unpack_avx2(...)
    // TODO: Implement AVX2 version in Phase 6

// sixtet_pack.h:239
static inline void sixtet_pack_avx2(...)
    // TODO: Implement AVX2 version in Phase 6

// sixtet_pack.h:251
static inline void sixtet_unpack_avx2(...)
    // TODO: Implement AVX2 version in Phase 6
```

**Status:** All fall back to scalar implementations

**Recommendation:** Implement or remove AVX2 function signatures

---

### 3.2 TritNet Cross-Entropy Loss

**Location:** [models/tritnet/src/train_tritnet.py:237](models/tritnet/src/train_tritnet.py)

```python
# TODO: Cross-entropy requires model output reshape to [batch, 5, 3]
# For now, using MSE. Cross-entropy is future work.
print("  WARNING: Cross-entropy not yet implemented, using MSE instead")
```

**Impact:** Training may be suboptimal for classification task

**Recommendation:** Implement proper cross-entropy loss for trit classification

---

### 3.3 TritNet GEMM Edge Cases

**Location:** [models/tritnet/gemm/](models/tritnet/gemm/)

```cpp
// tritnet_gemm_avx2.cpp:155
// TODO: Implement scalar fallback or use masked AVX2

// tritnet_gemm_naive.cpp:88
// TODO: Handle K not multiple of 5 with padding

// TODO: Actually use this for OpenMP parallelization
```

**Status:** Blocking production use of GEMM

---

## 4. Dead/Unused Code

### 4.1 Unimplemented Operations (tand, tor)

**Location:** All backend implementations

```cpp
// backend_avx2_v1_baseline.cpp:139-140
.tand = NULL,
.tor = NULL

// backend_avx2_v2_optimized.cpp:641-642
.tand = NULL,
.tor = NULL

// backend_scalar_impl.cpp:126-127
.tand = NULL,
.tor = NULL
```

**Status:** API slots defined, never implemented

**Recommendation:** Either implement or remove from API

---

### 4.2 Deprecated Benchmarks (10 files, 108 KB)

**Location:** [benchmarks/deprecated/](benchmarks/deprecated/)

| File | Size | Purpose |
|------|------|---------|
| `bench_backend_fusion.py` | 10 KB | Superseded |
| `bench_backends.py` | 12 KB | Superseded |
| `bench_backends_improved.py` | 20 KB | Superseded |
| `bench_fusion_phase41.py` | 10 KB | Multiple versions |
| `bench_fusion_poc.py` | 15 KB | Proof of concept |
| `bench_fusion_rigorous.py` | 10 KB | Duplicate |
| `bench_fusion_simple.py` | 4 KB | Duplicate |
| `bench_fusion_validation.py` | 8 KB | Duplicate |
| `bench_with_load_context.py` | 17 KB | Experimental |
| `README.md` | 1 KB | - |

**Recommendation:** Archive to separate branch or delete

---

## 5. Test Coverage Gaps

### 5.1 Backend Integration Tests

**Available tests:** [tests/python/test_backend_integration.py](tests/python/test_backend_integration.py)

**Missing tests:**
- Backend switching verification
- Performance comparison between backends
- Fallback behavior when preferred backend unavailable

---

### 5.2 Python Bindings Not Tested

| Function | Exposed | Tested |
|----------|---------|--------|
| `detect_simd_level()` | Yes | No |
| `simd_level_string()` | Yes | No |
| `set_backend()` | Yes | No |
| `get_active()` | Yes | No |
| `get_capabilities_string()` | Yes | No |

---

### 5.3 Dual-Shuffle Validation

**Location:** [tests/python/test_dual_shuffle_validation.py](tests/python/test_dual_shuffle_validation.py)

**Status:** Test file exists but optimization is disabled, so tests are meaningless

---

### 5.4 TritNet GEMM Integration

**Location:** [tests/python/test_tritnet_gemm_integration.py](tests/python/test_tritnet_gemm_integration.py)

**Status:** Integration test exists but module rarely built

---

## 6. Documentation vs Reality

### 6.1 Features Documented as Complete but Partial

| Feature | Documentation Status | Actual Status |
|---------|---------------------|---------------|
| Phase 6 Optimizations | "Planned" in roadmap | Design only, no implementation |
| GPU/TPU Acceleration | In TritNet roadmap | Framework only |
| Multi-platform | "Production" claimed | Windows x64 only |
| Competitive Benchmark | "6 phases" | Only Phase 1-2 complete |

---

### 6.2 Performance Claims Require Validation

| Claim | Source | Validation |
|-------|--------|------------|
| 45.3 Gops/s peak | README.md | Validated 2025-11-28 |
| 8,234× vs Python | README.md | Validated |
| Dual-shuffle 1.5-1.7× | opt_dual_shuffle_xor.h | **Never benchmarked** |
| PGO 5-15% | build_pgo.py | **Never run** |
| 256B LUT 10-20% | opt_lut_256byte_expanded.h | **Not implemented** |

---

## 7. Recommendations by Priority

### High Priority (Performance Impact)

1. **Enable Dual-Shuffle XOR** - 400 lines ready, 1.5× speedup claimed
2. **Run PGO Builds** - 5-15% improvement, infrastructure complete
3. **Complete 256B LUT** - 10-20% claimed, or remove from claims

### Medium Priority (Code Quality)

4. **Test Backend Switching** - API exists, no validation
5. **Implement tand/tor or Remove** - Dead API slots
6. **Delete Deprecated Benchmarks** - 108 KB of confusion
7. **Consolidate Build Scripts** - 8 variants, 3 used

### Low Priority (Technical Debt)

8. **Remove Unused CPU Detection** - AVX-512/NEON/SVE unused
9. **Document Production Build Flags** - TERNARY_NO_SANITIZE, etc.
10. **Integrate Profiler** - 294 lines never called

---

## 8. Quick Wins (< 1 Hour Each)

| Task | File | Impact |
|------|------|--------|
| Uncomment dual-shuffle init | backend_avx2_v2_optimized.cpp:61 | Potential 1.5× speedup |
| Run PGO build once | `python build/build_pgo.py full` | 5-15% improvement |
| Delete deprecated/ folder | benchmarks/deprecated/ | Clean codebase |
| Add backend integration test | tests/python/ | Validate plugin API |
| Document build flags | build/README.md | Enable production tuning |

---

## Appendix: Files Requiring Attention

### Most Underutilized Files

1. `src/core/simd/opt_dual_shuffle_xor.h` - 400 lines disabled
2. `src/core/profiling/ternary_profiler.h` - 294 lines unused
3. `build/build_pgo.py` - 400 lines never executed
4. `src/core/simd/opt_lut_256byte_expanded.h` - Placeholder only
5. `benchmarks/deprecated/` - 10 abandoned files

### Files with Highest TODO Count

1. `src/core/packing/sixtet_pack.h` - 2 TODOs (AVX2)
2. `src/core/packing/octet_pack.h` - 2 TODOs (AVX2)
3. `models/tritnet/src/train_tritnet.py` - 1 TODO (cross-entropy)
4. `src/core/simd/opt_lut_256byte_expanded.h` - 1 TODO (implementation)
5. `src/core/simd/opt_dual_shuffle_xor.h` - 1 TODO (microbenchmark)

---

**Generated:** 2025-12-09
**Analyzer:** Claude Code
**Repository:** ternary-engine
