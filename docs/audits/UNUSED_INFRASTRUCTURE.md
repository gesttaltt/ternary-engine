# Unused Infrastructure Audit

**Doc-Type:** Technical Audit · Version 1.0 · Generated 2025-12-09

This document provides a detailed analysis of infrastructure components that have been implemented but are not being utilized in the Ternary Engine codebase.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Profiler Framework](#1-profiler-framework)
3. [Backend Plugin API](#2-backend-plugin-api)
4. [CPU Capability Detection](#3-cpu-capability-detection)
5. [Optimization Configuration Flags](#4-optimization-configuration-flags)
6. [Build System Variants](#5-build-system-variants)
7. [TritNet Infrastructure](#6-tritnet-infrastructure)
8. [Remediation Recommendations](#remediation-recommendations)

---

## Executive Summary

| Infrastructure | Lines of Code | Development Cost | Current Usage | Recommendation |
|----------------|---------------|------------------|---------------|----------------|
| Profiler Framework | 294 | High | **0%** | Integrate or Remove |
| Backend Plugin API | 400+ | High | **20%** | Complete Integration |
| CPU Detection | 144 | Medium | **10%** | Remove Unused |
| Config Flags | ~50 | Low | **0%** | Document & Use |
| Build Variants | 1200+ | High | **25%** | Consolidate |
| TritNet GEMM | 800+ | High | **10%** | Complete or Archive |

**Total Unused Code:** ~2,800 lines representing significant development investment

---

## 1. Profiler Framework

### Overview

A complete cross-platform profiler integration framework supporting Intel VTune, NVIDIA NVTX, and Google Perfetto has been implemented but is never used.

### Location

**Primary File:** `src/core/profiling/ternary_profiler.h` (294 lines)

### Implementation Details

#### Supported Backends

```cpp
// Three profiler backends defined:

#if defined(TERNARY_ENABLE_VTUNE)
    // Intel VTune ITT API integration
    #include <ittnotify.h>
    #define TERNARY_PROFILE_DOMAIN(name, str) \
        __itt_domain* name = __itt_domain_create(str)
    #define TERNARY_PROFILE_TASK_BEGIN(domain, name) \
        __itt_task_begin(domain, __itt_null, __itt_null, name)
    #define TERNARY_PROFILE_TASK_END(domain) \
        __itt_task_end(domain)

#elif defined(TERNARY_ENABLE_NVTX)
    // NVIDIA NVTX integration (for GPU profiling)
    #include <nvToolsExt.h>
    #define TERNARY_PROFILE_RANGE_PUSH(name) nvtxRangePush(name)
    #define TERNARY_PROFILE_RANGE_POP() nvtxRangePop()

#elif defined(TERNARY_ENABLE_PERFETTO)
    // Google Perfetto (Chrome tracing)
    // Stub only - not implemented

#else
    // Default: No-op macros (zero overhead)
    #define TERNARY_PROFILE_DOMAIN(name, str)
    #define TERNARY_PROFILE_TASK_BEGIN(domain, name)
    #define TERNARY_PROFILE_TASK_END(domain)
#endif
```

#### Intended Usage Pattern

```cpp
// Example from header documentation:

#include "ternary_profiler.h"

// Define profiling domain and task names (once per file)
TERNARY_PROFILE_DOMAIN(g_domain, "TernaryCore");
TERNARY_PROFILE_TASK_NAME(g_simd_loop, "SIMD_Loop");
TERNARY_PROFILE_TASK_NAME(g_omp_region, "OpenMP_Region");

void process_large_array(const uint8_t* a, const uint8_t* b,
                         uint8_t* r, size_t n) {
    TERNARY_PROFILE_TASK_BEGIN(g_domain, g_simd_loop);

    #pragma omp parallel for
    for (size_t i = 0; i < n; i += 32) {
        // SIMD operations
    }

    TERNARY_PROFILE_TASK_END(g_domain);
}
```

### Current Usage

**Zero (0) uses in the entire codebase:**

```bash
# Search for profiler macro usage
$ grep -r "TERNARY_PROFILE" src/
# No results

# Search for include of profiler header
$ grep -r "ternary_profiler.h" src/
# No results
```

### Why It's Unused

1. **Compile Flags Never Applied:** The `TERNARY_ENABLE_VTUNE` flag is documented but never passed to any build
2. **No Profiling Workflow:** No documentation on how to run profiled builds
3. **External Tool Required:** Requires VTune/NVTX which may not be installed

### Enabling Profiler Integration

#### Step 1: Add Profiler Annotations to Hot Paths

```cpp
// In src/core/simd/backend_avx2_v2_optimized.cpp

#include "../profiling/ternary_profiler.h"

TERNARY_PROFILE_DOMAIN(g_ternary_domain, "TernaryEngine");
TERNARY_PROFILE_TASK_NAME(g_tadd_task, "tadd_avx2");
TERNARY_PROFILE_TASK_NAME(g_tmul_task, "tmul_avx2");

void tadd_avx2_v2(const uint8_t* a, const uint8_t* b,
                  uint8_t* r, size_t n) {
    TERNARY_PROFILE_TASK_BEGIN(g_ternary_domain, g_tadd_task);

    // ... existing implementation ...

    TERNARY_PROFILE_TASK_END(g_ternary_domain);
}
```

#### Step 2: Create Profiling Build Target

```python
# Add to build/build.py

def build_with_profiling():
    """Build with VTune profiling annotations enabled."""
    extra_flags = ["/DTERNARY_ENABLE_VTUNE"]
    extra_libs = ["libittnotify.lib"]
    # ... build with extra flags ...
```

#### Step 3: Document Profiling Workflow

```markdown
## Profiling with Intel VTune

1. Install Intel oneAPI Base Toolkit
2. Build with profiling: `python build/build.py --profile`
3. Run VTune: `vtune -collect hotspots python benchmark.py`
4. View results: `vtune-gui vtune_results/`
```

### Recommendation

**Priority: MEDIUM**

**Options:**

1. **Integrate:** Add profiler macros to hot paths, create profiling build target
2. **Remove:** Delete if profiling will never be used
3. **Defer:** Keep as-is with documentation that it's available but not integrated

**Estimated Effort:** 2-4 hours to integrate into hot paths

---

## 2. Backend Plugin API

### Overview

A sophisticated backend plugin system allowing runtime selection of SIMD implementations has been built but is only partially exposed and never tested.

### Location

- **Header:** `src/core/simd/backend_plugin_api.h`
- **Registry:** `src/core/simd/backend_registry_dispatch.cpp`
- **Python Bindings:** `src/engine/bindings_backend_api.cpp`
- **Backends:**
  - `src/core/simd/backend_scalar_impl.cpp`
  - `src/core/simd/backend_avx2_v1_baseline.cpp`
  - `src/core/simd/backend_avx2_v2_optimized.cpp`

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Backend Plugin API                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Scalar     │  │  AVX2 v1     │  │  AVX2 v2     │      │
│  │  (Fallback)  │  │ (Baseline)   │  │ (Optimized)  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │               │
│         └─────────────────┼─────────────────┘               │
│                           │                                 │
│                    ┌──────▼───────┐                         │
│                    │   Registry   │                         │
│                    │   Dispatch   │                         │
│                    └──────┬───────┘                         │
│                           │                                 │
│                    ┌──────▼───────┐                         │
│                    │   Python     │                         │
│                    │   Bindings   │                         │
│                    └──────────────┘                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### API Surface

```cpp
// backend_plugin_api.h

struct TernaryBackend {
    const char* name;
    const char* version;
    uint32_t capabilities;

    // Operation function pointers
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

    // Unimplemented placeholders
    void (*tand)(const uint8_t*, const uint8_t*, uint8_t*, size_t);  // NULL
    void (*tor)(const uint8_t*, const uint8_t*, uint8_t*, size_t);   // NULL
};

// Registry functions
void register_backend(const TernaryBackend* backend);
void set_active_backend(const char* name);
const TernaryBackend* get_active_backend();
const TernaryBackend** list_backends(size_t* count);
```

### Python Bindings

```python
# Available but undocumented module: ternary_backend

import ternary_backend

# Initialize backend system
ternary_backend.init()

# List available backends
backends = ternary_backend.list_backends()
for b in backends:
    print(f"{b.name} v{b.version}: {b.capabilities}")

# Set active backend
ternary_backend.set_backend("avx2_v2_optimized")

# Get current backend
active = ternary_backend.get_active()
print(f"Active: {active.name}")

# Operations dispatch through active backend
result = ternary_backend.tadd(a, b)
```

### Current Usage

| Component | Implemented | Exposed to Python | Tested | Used in Production |
|-----------|-------------|-------------------|--------|-------------------|
| Backend struct | ✓ | ✓ | ✗ | ✗ |
| Registry | ✓ | ✓ | ✗ | ✗ |
| Dispatch | ✓ | ✓ | ✗ | ✗ |
| list_backends() | ✓ | ✓ | ✗ | ✗ |
| set_backend() | ✓ | ✓ | ✗ | ✗ |
| get_active() | ✓ | ✓ | ✗ | ✗ |

### Missing Test Coverage

```python
# tests/python/test_backend_integration.py exists but is minimal

# Missing tests:
# 1. Backend switching verification
def test_backend_switching():
    """Verify set_backend actually changes dispatch target."""
    import ternary_backend

    ternary_backend.init()

    # Test with each backend
    for backend in ternary_backend.list_backends():
        ternary_backend.set_backend(backend.name)
        active = ternary_backend.get_active()
        assert active.name == backend.name

        # Verify operations still work
        result = ternary_backend.tadd(a, b)
        np.testing.assert_array_equal(result, expected)

# 2. Fallback behavior
def test_fallback_when_unavailable():
    """Verify graceful fallback when requested backend unavailable."""
    pass

# 3. Performance comparison
def test_backend_performance_ordering():
    """Verify optimized backends are faster than baseline."""
    pass
```

### Recommendation

**Priority: MEDIUM**

The backend system represents significant development investment. Options:

1. **Complete Integration:**
   - Add comprehensive tests
   - Document in README
   - Expose in main `ternary_simd_engine` module
   - Use for automatic backend selection

2. **Simplify:**
   - Remove plugin API complexity
   - Hardcode best backend selection at compile time
   - Reduce maintenance burden

3. **Current Path:**
   - Keep infrastructure
   - Add tests to prevent regression
   - Document for future use

**Estimated Effort:** 4-8 hours for full integration

---

## 3. CPU Capability Detection

### Overview

Comprehensive CPU feature detection is implemented but most detected features have no corresponding backend.

### Location

**File:** `src/core/simd/cpu_simd_capability.h` (144 lines)

### Implemented Detection

```cpp
// Functions defined in cpu_simd_capability.h

// Actually used
bool has_avx2();           // Used to validate AVX2 backend
SIMDLevel detect_best_simd();  // Called once at module init

// Defined but no backend exists
bool has_avx512f();        // No AVX-512 backend
bool has_avx512bw();       // No AVX-512 backend
bool has_avx512vl();       // No AVX-512 backend
bool has_neon();           // No ARM NEON backend
bool has_sve();            // No ARM SVE backend
bool has_sse41();          // No SSE4.1-specific backend

// Exposed to Python but never tested
const char* simd_level_name(SIMDLevel level);
```

### Usage Analysis

```cpp
// The only actual usage in the codebase:

// bindings_core_ops.cpp - Module initialization
PYBIND11_MODULE(ternary_simd_engine, m) {
    if (!has_avx2()) {
        throw std::runtime_error("AVX2 required but not available");
    }
    // ... rest of initialization
}

// bindings_core_ops.cpp - Exposed to Python
m.def("has_avx2", &has_avx2, "Check if CPU supports AVX2");
m.def("detect_simd_level", []() {
    return static_cast<int>(detect_best_simd());
});
m.def("simd_level_string", []() {
    return std::string(simd_level_name(detect_best_simd()));
});
```

### Unused Capabilities

| Function | Backend Exists | Reason Unused |
|----------|---------------|---------------|
| `has_avx512f()` | No | AVX-512 not implemented |
| `has_avx512bw()` | No | AVX-512 not implemented |
| `has_avx512vl()` | No | AVX-512 not implemented |
| `has_neon()` | No | ARM not supported |
| `has_sve()` | No | ARM SVE not supported |
| `has_sse41()` | No | SSE4.1 fallback not implemented |

### Recommendation

**Priority: LOW**

Options:

1. **Remove Unused:** Delete detection for unsupported ISAs
2. **Keep for Future:** Maintain as placeholder for future backends
3. **Implement Fallbacks:** Add SSE4.1 fallback for older CPUs

**Minimal Change:** Remove ARM detection (NEON/SVE) since this is a Windows x64-only project

---

## 4. Optimization Configuration Flags

### Overview

Multiple compile-time and runtime configuration options are defined but never used in production builds.

### Location

**File:** `src/core/config/optimization_config.h`

### Defined Flags

```cpp
// optimization_config.h

// Compile-time flags (never passed in builds)
#ifndef TERNARY_NO_SANITIZE
    // When defined: Skip input validation for 3-5% performance gain
    // Usage: -DTERNARY_NO_SANITIZE
#endif

#ifndef TERNARY_FORCE_SCALAR
    // When defined: Use scalar instead of SIMD (debugging)
    // Usage: -DTERNARY_FORCE_SCALAR
#endif

// Runtime thresholds (hardcoded defaults)
#ifndef TERNARY_OMP_THRESHOLD
    #define TERNARY_OMP_THRESHOLD (32768 * get_num_cores())
#endif

#ifndef TERNARY_STREAM_THRESHOLD
    #define TERNARY_STREAM_THRESHOLD 1048576  // 1M elements
#endif

#ifndef TERNARY_PREFETCH_DIST
    #define TERNARY_PREFETCH_DIST 512  // bytes ahead
#endif
```

### Current Build Commands

```python
# build/build.py - Standard build

extra_compile_args = [
    "/O2",      # Optimization level
    "/GL",      # Whole program optimization
    "/arch:AVX2",
    "/std:c++17",
    "/openmp",
    "/DNDEBUG",
    # Note: No TERNARY_* flags passed
]
```

### Recommended Production Flags

```python
# For production builds with validated inputs:
extra_compile_args = [
    "/O2",
    "/GL",
    "/arch:AVX2",
    "/std:c++17",
    "/openmp",
    "/DNDEBUG",
    "/DTERNARY_NO_SANITIZE",  # 3-5% gain for validated pipelines
]
```

### Recommendation

**Priority: LOW**

1. Document flags in build/README.md
2. Add `--production` flag to build.py that enables optimized settings
3. Benchmark with/without `TERNARY_NO_SANITIZE`

---

## 5. Build System Variants

### Overview

Eight different build scripts exist, but only 2-3 are regularly used.

### Inventory

| Script | Purpose | Lines | Last Used | Status |
|--------|---------|-------|-----------|--------|
| `build.py` | Standard build | ~200 | Active | ✓ Used |
| `build_all.py` | Build all modules | ~100 | Active | ✓ Used |
| `build_dense243.py` | Dense243 module | ~150 | Active | ✓ Used |
| `build_pgo.py` | MSVC PGO | ~400 | Never | ✗ Unused |
| `build_pgo_unified.py` | Clang PGO | ~300 | Never | ✗ Unused |
| `build_backend.py` | Backend builds | ~150 | Unknown | ? Unknown |
| `build_reference.py` | Reference impl | ~100 | Rarely | ⚠ Test only |
| `build_test_packing.py` | Packing tests | ~80 | Rarely | ⚠ Standalone |
| `build_tritnet_gemm.py` | TritNet GEMM | ~150 | Rarely | ⚠ Experimental |

### Recommendation

**Priority: LOW**

Consider consolidating into unified build system:

```python
# Proposed: build/build.py --target <target> --options

# Examples:
python build/build.py                      # Standard build
python build/build.py --target all         # All modules
python build/build.py --target dense243    # Dense243 module
python build/build.py --pgo                # With PGO
python build/build.py --profile            # With profiling
python build/build.py --debug              # Debug build
```

---

## 6. TritNet Infrastructure

### Overview

Significant infrastructure for TritNet (neural network-based ternary arithmetic) exists but is incomplete.

### Components

| Component | Location | Status |
|-----------|----------|--------|
| Truth Tables | `models/datasets/tritnet/` | ✓ Complete |
| Model Architecture | `models/tritnet/src/tritnet_model.py` | ✓ Complete |
| Training Script | `models/tritnet/src/train_tritnet.py` | ⚠ Partial (TODO) |
| GEMM Naive | `models/tritnet/gemm/tritnet_gemm_naive.cpp` | ⚠ Partial |
| GEMM AVX2 | `models/tritnet/gemm/tritnet_gemm_avx2.cpp` | ⚠ Partial |
| Python Bindings | `src/engine/bindings_tritnet_gemm.cpp` | ✓ Complete |

### Incomplete Items

```python
# train_tritnet.py:237
# TODO: Cross-entropy requires model output reshape to [batch, 5, 3]
# For now, using MSE. Cross-entropy is future work.
print("  WARNING: Cross-entropy not yet implemented, using MSE instead")
```

```cpp
// tritnet_gemm_avx2.cpp:155
// TODO: Implement scalar fallback or use masked AVX2

// tritnet_gemm_naive.cpp:88
// TODO: Handle K not multiple of 5 with padding
```

### Recommendation

**Priority: MEDIUM**

TritNet is a core innovation of the project. Options:

1. **Complete Phase 2A:** Fix TODOs in training, validate 100% accuracy
2. **Archive Experimental:** Move to experimental/ with clear status
3. **Document Limitations:** Update docs with current capabilities

---

## Remediation Recommendations

### Immediate Actions

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| 1 | Add profiler macros to hot paths | Enable VTune analysis | 2 hours |
| 2 | Test backend switching | Validate plugin API | 2 hours |
| 3 | Document config flags | Enable production tuning | 1 hour |

### Short-Term Actions

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| 4 | Consolidate build scripts | Reduce maintenance | 4 hours |
| 5 | Remove unused CPU detection | Clean code | 1 hour |
| 6 | Complete TritNet training | Core innovation | 8 hours |

### Long-Term Actions

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| 7 | Implement AVX-512 backend | Future performance | 16 hours |
| 8 | ARM NEON backend | Cross-platform | 24 hours |
| 9 | GPU/TPU acceleration | TritNet viability | 40+ hours |

---

**Document Version:** 1.0
**Last Updated:** 2025-12-09
**Author:** Claude Code Audit
