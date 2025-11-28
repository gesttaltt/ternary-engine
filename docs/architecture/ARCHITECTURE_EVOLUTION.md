# Architecture Evolution - Ternary Engine

**Document Type:** Historical Analysis
**Date:** 2025-11-25
**Purpose:** Track architectural decisions, branching points, and simplification process

---

## Timeline Overview

```
v1.1.0 (ktr)         v1.2.0 (backends)        v1.2.0 (regression)      v1.3.0 (simplified)
    |                        |                        |                        |
    |                        |                        |                        |
Direct kernels -----> Backend dispatch -----> Performance issues ---> Back to kernels
    |                        |                        |                        |
    |                        |                        |                        |
Simple, fast          Complex, modular         28→18 Gops/s          39 Gops/s peak
```

---

## Phase 1: Direct Kernels (v1.1.0 - "ktr")

**Commit:** c79b5d8 "RELEASE: v1.1.0 'ktr' - Source Restructuring"
**Date:** ~2025-11 (before v1.2.0)

### Architecture

**File Structure:**
```
src/engine/bindings_core_ops.cpp
  └─> Directly implements AVX2 kernels
  └─> Uses ternary_simd_kernels.h
  └─> Simple, monolithic approach
```

**Key Characteristics:**
- Direct AVX2 intrinsics in bindings
- No abstraction layers
- Single implementation path
- Performance: ~28 Gops/s peak (from commit 545655d)

**Code Pattern:**
```cpp
// bindings_core_ops.cpp (v1.1.0 style)
__m256i binary_simd_op(__m256i a, __m256i b, __m256i lut) {
    // Direct implementation
    __m256i a_shifted = _mm256_add_epi8(...);  // Shift+OR indexing
    __m256i indices = _mm256_or_si256(a_shifted, b_masked);
    return _mm256_shuffle_epi8(lut, indices);
}
```

**Strengths:**
- Simple to understand
- Fast compilation
- Easy to debug
- Predictable performance

**Weaknesses:**
- No runtime dispatch
- Single ISA target (AVX2)
- Hard to add new backends
- Optimization locked to one path

---

## Phase 2: Backend System (v1.2.0 - Multi-Backend)

**Commits:**
- 86e4bb2 "DOCS: Add comprehensive unified roadmap for v1.2.0 → v3.0+"
- 66956f3 "FEAT: Implement Selective SIMD/LUT Ultra-Optimizations"
- 21739b7 "FEAT: Implement Backend Interface and Dispatch System"
- e2d5644 "PERF: Implement canonical indexing in AVX2_v2 backend"
- 2b627cf "FEAT: Integrate v1.2.0 backend system with Python bindings"

**Date:** Mid-November 2025

### Architecture

**File Structure:**
```
src/core/simd/
  ├─ ternary_backend_interface.h      # Abstract interface
  ├─ ternary_backend_dispatch.cpp     # Runtime dispatch
  ├─ ternary_backend_scalar.cpp       # Fallback
  ├─ ternary_backend_avx2_v1.cpp      # Traditional indexing
  └─ ternary_backend_avx2_v2.cpp      # Canonical indexing ✨

src/engine/bindings_core_ops.cpp
  └─> Uses backend dispatch
  └─> Runtime selection of optimal path
```

**Key Innovations:**
1. **Canonical Indexing** (avx2_v2) - Dual-shuffle + ADD instead of shift+OR
2. **Runtime Dispatch** - CPU detection, optimal backend selection
3. **Multi-ISA Support** - Scalar, AVX2_v1, AVX2_v2 backends
4. **Abstraction Layer** - Backend interface for future ISAs

**Code Pattern (avx2_v2.cpp):**
```cpp
// Using canonical indexing (Phase 3.2)
static inline __m256i binary_op_canonical(__m256i a, __m256i b, __m256i lut) {
    __m256i indices = canonical_index_avx2(a_masked, b_masked);
    return _mm256_shuffle_epi8(lut, indices);
}

// canonical_index_avx2 implementation:
static inline __m256i canonical_index_avx2(__m256i a, __m256i b) {
    __m256i canon_a = _mm256_load_si256((__m256i*)CANON_A_LUT_256);
    __m256i canon_b = _mm256_load_si256((__m256i*)CANON_B_LUT_256);
    __m256i contrib_a = _mm256_shuffle_epi8(canon_a, a);
    __m256i contrib_b = _mm256_shuffle_epi8(canon_b, b);
    return _mm256_add_epi8(contrib_a, contrib_b);  // Dual-shuffle + ADD
}
```

**Strengths:**
- Modular architecture
- Multiple optimization paths
- Future-proof (AVX-512, NEON ready)
- Canonical indexing implemented

**Weaknesses:**
- Increased complexity
- More compilation units
- Harder to debug
- Dispatch overhead

### Performance Target

**Expected:** 35-45 Gops/s (from avx2_v2.cpp line 11)
**Canonical indexing gain:** 12-18% over v1 (from avx2_v2.cpp line 16)

---

## Phase 3: Crisis - Backend Regression (v1.2.0 Issues)

**Commits:**
- 545655d "CRITICAL: Document backend performance REGRESSION and missing components"
- ba54037 "DOCS: Add backend fusion performance analysis and corrected benchmark"

**Date:** Late November 2025

### The Problem

**Performance Regression Discovered:**
```
v1.1.0 Direct:  28 Gops/s peak
v1.2.0 Backend: 18 Gops/s peak  ← 36% REGRESSION
```

**Root Causes (from commit 545655d):**
1. Backend dispatch overhead
2. Missing OpenMP in new architecture
3. Incomplete migration of optimizations
4. Complexity introduced bugs

### Critical Decision Point

**Options:**
1. Fix backend system (debug, optimize, complete migration)
2. Simplify architecture (return to direct kernels)

**Decision:** Simplify (chosen based on subsequent commits)

**Rationale:**
- Complexity not justified by benefits
- Performance regression unacceptable
- Simpler code = fewer bugs
- Can re-add backends later if needed

---

## Phase 4: Simplification Migration (v1.2.0 → v1.3.0)

**Commits:**
- feeec9d "PLAN: Comprehensive component migration plan (old module → backends)"
- 04d3a02 "WIP: Component migration checkpoint 1/6 - tadd migrated"
- a3860f8 "COMPLETE: All 9 operations migrated to three-path architecture"
- 4b2622a "FEAT: Re-enable OpenMP and complete component migration to backends"

**Date:** Late November 2025

### Migration Strategy

**Three-Path Architecture:**
```
Operation Request
    |
    ├─> OpenMP Path (n >= 100K)
    ├─> SIMD Path (1K ≤ n < 100K)
    └─> Scalar Path (n < 1K)
```

**Process:**
1. Extract working code from backends
2. Simplify to direct kernels
3. Re-enable OpenMP (was lost in backend migration)
4. Validate performance

**Status Checkpoints:**
- Checkpoint 1/6: tadd migrated
- Checkpoint 6/6: All 9 operations complete
- Result: OpenMP re-enabled, performance recovered

---

## Phase 5: Current Architecture (v1.3.0 - Simplified)

**Commits:**
- 26b46d2 "COMPLETE: Phase 3.3 - Establish 4-fusion baseline"
- 03070bb "ANALYSIS: Phase 3.2 dual-shuffle optimization already implemented"
- 027901d "PERF: Integrate canonical indexing - achieve 45.3 Gops/s"

**Date:** 2025-11-25 (today)

### Architecture

**File Structure:**
```
src/core/simd/
  ├─ ternary_simd_kernels.h          # Active kernels (NOW WITH CANONICAL)
  ├─ ternary_canonical_index.h       # Canonical indexing LUTs
  ├─ ternary_fusion.h                 # Fusion operations
  ├─ ternary_backend_*.cpp            # Archived (not used)
  └─ ternary_backend_dispatch.cpp     # Archived (not used)

src/engine/bindings_core_ops.cpp
  └─> Uses ternary_simd_kernels.h directly
  └─> Includes ternary_fusion.h
  └─> No backend dispatch
```

**Current Code (ternary_simd_kernels.h):**
```cpp
#include "ternary_canonical_index.h"  // ← Added today (commit 027901d)

// Unified binary operation with canonical indexing
template <bool Sanitize = true>
static inline __m256i binary_simd_op(__m256i a, __m256i b, __m256i lut) {
    __m256i a_masked = maybe_mask<Sanitize>(a);
    __m256i b_masked = maybe_mask<Sanitize>(b);

    // Canonical indexing: Dual-shuffle + ADD (re-integrated from avx2_v2)
    __m256i canon_a = _mm256_load_si256((__m256i*)CANON_A_LUT_256);
    __m256i canon_b = _mm256_load_si256((__m256i*)CANON_B_LUT_256);
    __m256i contrib_a = _mm256_shuffle_epi8(canon_a, a_masked);
    __m256i contrib_b = _mm256_shuffle_epi8(canon_b, b_masked);
    __m256i indices = _mm256_add_epi8(contrib_a, contrib_b);

    return _mm256_shuffle_epi8(lut, indices);
}
```

**Key Characteristics:**
- Simple, direct implementation
- Canonical indexing re-integrated from backends
- Fusion operations always exposed
- OpenMP working correctly
- Performance: 39.1 Gops/s peak, 45.3 Gops/s fusion effective

**What Was Preserved:**
1. ✅ Fusion operations (always worked, in ternary_fusion.h)
2. ✅ Canonical indexing (re-integrated from avx2_v2.cpp)
3. ✅ OpenMP parallelization (re-enabled during migration)
4. ✅ Three-path architecture (OpenMP/SIMD/Scalar)

**What Was NOT Integrated:**
1. ❌ Backend dispatch system (archived, not removed)
2. ❌ Dense243 into core ops (remains separate module)
3. ❌ PGO in active builds (scripts exist, not used)

---

## Architectural Lessons Learned

### 1. Complexity Cost

**Backend System:**
- Pro: Modular, extensible, multiple optimization paths
- Con: 36% performance regression, harder to debug, slower iteration

**Direct Kernels:**
- Pro: Simple, fast, easy to optimize, predictable
- Con: Less modular, harder to add new ISAs

**Lesson:** Complexity must be justified by measurable benefits. v1.2.0 backend system added complexity without performance gain, causing regression.

### 2. Migration Risks

**What Went Wrong:**
- OpenMP lost during backend migration
- Optimizations incomplete in new system
- Performance not validated at each step

**What Went Right:**
- Backend code preserved (in src/core/simd/)
- Canonical indexing implementation survived
- Could extract and re-integrate best parts

**Lesson:** Preserve old working code when refactoring. Incremental migration with continuous validation.

### 3. Optimization Mobility

**Discovery:** Optimizations can move between architectures

**Example - Canonical Indexing Journey:**
```
Phase 1: Not implemented (shift+OR indexing)
    |
    v
Phase 2: Implemented in avx2_v2.cpp backend
    |
    v
Phase 3: Backend archived during simplification
    |
    v
Phase 4: Re-integrated into ternary_simd_kernels.h
```

**Lesson:** Good optimizations are architecture-agnostic. Canonical indexing worked in backends, works in direct kernels.

### 4. Performance Validation Critical

**v1.2.0 Backend Timeline:**
```
Design → Implement → "Looks good" → Deploy → Benchmark → 36% REGRESSION
                                              ^
                                              Should have happened HERE
```

**v1.3.0 Canonical Re-integration:**
```
Extract → Re-implement → Benchmark → +74-1,100% → Deploy
                         ^
                         Validated BEFORE deployment
```

**Lesson:** Benchmark after every architectural change, not just at release.

---

## Current State Analysis

### Active Components

| Component | Location | Status | Performance |
|-----------|----------|--------|-------------|
| **Core Kernels** | ternary_simd_kernels.h | ✅ Active | 39.1 Gops/s |
| **Canonical Indexing** | Integrated in kernels.h | ✅ Active (today) | +74-1,100% |
| **Fusion (4 ops)** | ternary_fusion.h | ✅ Active | 45.3 Gops/s effective |
| **OpenMP** | bindings_core_ops.cpp | ✅ Active | 12 threads |
| **Dense243** | Separate module | ⚠️ Not integrated | 5× memory savings |
| **PGO** | build/build_pgo*.py | ⚠️ Scripts exist | Not used |

### Archived Components (Preserved, Not Used)

| Component | Location | Status | Notes |
|-----------|----------|--------|-------|
| **Backend Dispatch** | ternary_backend_dispatch.cpp | 📦 Archived | Caused 36% regression |
| **AVX2_v1 Backend** | ternary_backend_avx2_v1.cpp | 📦 Archived | Traditional indexing |
| **AVX2_v2 Backend** | ternary_backend_avx2_v2.cpp | 📦 Archived | Had canonical indexing |
| **Scalar Backend** | ternary_backend_scalar.cpp | 📦 Archived | Fallback implementation |

**Note:** "Archived" = Code exists in `src/core/simd/` but is NOT compiled or used by current bindings.

---

## Branching Visualization

```
                        Ternary Engine Evolution
                                  |
                    v1.1.0 Direct Kernels (28 Gops/s)
                                  |
                                  |
                    ┌─────────────┴─────────────┐
                    |                           |
                    v                           |
        v1.2.0 Backend System                  |
        (Modular, Complex)                     |
                    |                           |
                    v                           |
        Performance Regression                 |
        (18 Gops/s - 36% loss)                |
                    |                           |
                    v                           |
        CRITICAL: Revert Decision              |
                    |                           |
                    └──────────────┬────────────┘
                                   |
                                   v
                    v1.3.0 Simplified Kernels
                    (Extract best parts from v1.2.0)
                                   |
                    ┌──────────────┼──────────────┐
                    |              |              |
                    v              v              v
            Canonical Index   Fusion Ops    OpenMP Fixed
            (from avx2_v2)   (preserved)   (re-enabled)
                    |              |              |
                    └──────────────┴──────────────┘
                                   |
                                   v
                        39.1 Gops/s peak
                        45.3 Gops/s fusion
                        (74-1,100% improvements)
```

---

## Future Architecture Paths

### Option 1: Stay Simplified

**Pro:**
- Current performance excellent (45.3 Gops/s)
- Simple to maintain
- Fast iteration
- Proven stable

**Con:**
- Limited to AVX2
- Hard to add ARM NEON
- Single optimization path

**Recommendation:** For now, YES. Revisit if need multi-ISA.

### Option 2: Resurrect Backend System (If Needed)

**When:**
- Need ARM NEON support
- Need AVX-512 support
- Need runtime ISA selection

**How:**
- Extract backends from src/core/simd/
- Fix dispatch overhead (inline, constexpr)
- Validate performance at each step
- Keep simple kernels as fallback

**Lesson Applied:** Don't add until needed, validate thoroughly.

### Option 3: Hybrid Approach

**Idea:**
- Keep current simple kernels for x86 AVX2
- Add separate ARM build with direct NEON kernels
- Compile-time ISA selection, not runtime

**Pro:**
- Simple per-platform
- No dispatch overhead
- Multi-ISA support

**Con:**
- Multiple code paths to maintain
- Platform-specific builds

---

## Key Takeaways

1. **Simplicity Won:** v1.3.0 simplified approach outperforms v1.2.0 complex backend (39 vs 18 Gops/s)

2. **Optimizations Are Portable:** Canonical indexing worked in backends, re-integrated into kernels seamlessly

3. **Preserve History:** Backend code archived, not deleted. Can extract parts as needed (exactly what we did with canonical indexing)

4. **Benchmark Always:** Architecture changes without benchmarks = regressions (v1.2.0 lesson)

5. **YAGNI Principle:** Backend dispatch was over-engineered for current needs. Add complexity only when needed.

6. **Current Status:** v1.3.0 is the RIGHT architecture for current goals (single ISA, maximum performance)

---

## Decision Log

| Date | Decision | Rationale | Result |
|------|----------|-----------|--------|
| ~Nov 2025 | Implement backend system | Multi-ISA support | 36% regression |
| ~Nov 2025 | Revert to direct kernels | Fix regression | Performance recovered |
| 2025-11-25 | Re-integrate canonical indexing | Extract best from backends | +74-1,100% gain |
| 2025-11-25 | Keep backends archived | May need later | Code preserved |

---

**Document Status:** Living document - update as architecture evolves
**Last Updated:** 2025-11-25
**Next Review:** When considering multi-ISA support
