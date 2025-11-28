# Architecture Branching Analysis - Ternary Engine

**Type:** Root Cause Analysis
**Date:** 2025-11-25
**Focus:** Why backend system was built, why it regressed, why we reverted

---

## The Branching Event

**Timeline:**
```
v1.1.0 (ktr) → v1.2.0 (backend) → v1.2.0 (crisis) → v1.3.0 (simplified)
  28 Gops/s        Design phase       18 Gops/s        39.1 Gops/s
```

**Question:** Why did we branch into backend architecture if it caused regression?

**Answer:** Backend architecture was RIGHT idea, WRONG timing.

---

## Why Backend Architecture Was Built

### Motivation (from git commits 86e4bb2, 21739b7)

**Problem Statement:**
- v1.1.0 hardcoded to AVX2
- Cannot support ARM NEON without code duplication
- Cannot support AVX-512 without rewrite
- Wanted runtime ISA selection

**Design Goals:**
1. Multi-ISA support (AVX2, AVX-512, ARM NEON, WebAssembly SIMD)
2. Runtime dispatch based on CPU detection
3. Modular backends for different optimization strategies
4. Future-proof architecture

**Implementation Plan:**
```
ternary_backend_interface.h    # Abstract interface
  └─> ternary_backend_dispatch.cpp  # Runtime selection
       ├─> ternary_backend_scalar.cpp      # Fallback
       ├─> ternary_backend_avx2_v1.cpp     # Traditional
       └─> ternary_backend_avx2_v2.cpp     # Canonical indexing
```

**Expected Benefits:**
- 35-45 Gops/s with canonical indexing (avx2_v2)
- Easy to add new ISAs
- Clean separation of concerns
- Multiple optimization paths

**What Actually Happened:**
- Performance: 18 Gops/s (36% REGRESSION)
- Complexity increased
- Dispatch overhead not anticipated
- OpenMP lost in migration

---

## Root Cause of Regression

### Performance Analysis (from commit 545655d)

**Expected:** v1.1.0 (28 Gops/s) → v1.2.0 (35-45 Gops/s)
**Actual:** v1.1.0 (28 Gops/s) → v1.2.0 (18 Gops/s)

**Root Causes Identified:**

1. **Dispatch Overhead**
   - Runtime function pointer selection
   - Indirect calls prevent inlining
   - CPU branch prediction misses
   - Impact: ~10-15% overhead

2. **Missing OpenMP**
   - Backend migration didn't preserve OpenMP
   - Lost parallelization on large arrays
   - Impact: ~40-50% loss on 1M+ elements

3. **Incomplete Migration**
   - Some optimizations not fully ported
   - Three-path architecture (OpenMP/SIMD/Scalar) broken
   - Impact: ~15-20% loss

4. **Increased Complexity**
   - More indirection layers
   - Harder to optimize
   - Compiler less effective
   - Impact: ~5-10% loss

**Combined Effect:** 36% regression

---

## Why We Reverted (Simplification Decision)

### Decision Matrix

| Factor | Backend System | Direct Kernels | Winner |
|--------|---------------|----------------|--------|
| **Performance** | 18 Gops/s | 28 Gops/s | Direct |
| **Complexity** | High (5 files) | Low (1 file) | Direct |
| **Maintainability** | Hard | Easy | Direct |
| **Multi-ISA Support** | Yes | No | Backend |
| **Compilation Time** | Slower | Faster | Direct |
| **Debug-ability** | Harder | Easier | Direct |

**Score:** Direct kernels win 5-1

**Critical Factor:** We don't need multi-ISA support RIGHT NOW. It's YAGNI (You Ain't Gonna Need It).

**Decision:** Revert to simplified direct kernels, preserve backend code for future.

---

## The Simplification Process

### Migration Strategy (commits feeec9d, 04d3a02, a3860f8, 4b2622a)

**Phase 1: Extract Working Parts**
```
Backend System
  ├─> Canonical indexing (avx2_v2.cpp) → Extract
  ├─> OpenMP patterns → Extract
  ├─> Fusion operations → Preserve
  └─> Dispatch system → Archive
```

**Phase 2: Rebuild in Direct Kernels**
```
bindings_core_ops.cpp
  └─> Use ternary_simd_kernels.h
      ├─> Add OpenMP back (three-path)
      ├─> Keep fusion working
      └─> (Canonical indexing deferred to v1.3.0)
```

**Phase 3: Validate**
```
Benchmark → 28 Gops/s recovered → SUCCESS
```

**Phase 4: Archive Backends**
```
src/core/simd/
  ├─ ternary_backend_*.cpp  # Not deleted, just not used
  └─ Files preserved for future extraction
```

---

## What We Learned From Backends

### Good Ideas Extracted

1. **Canonical Indexing** - Re-integrated today (commit 027901d)
   - From: avx2_v2.cpp
   - To: ternary_simd_kernels.h
   - Result: +74-1,100% improvements

2. **Three-Path Architecture** - Preserved
   - OpenMP path (n >= 100K)
   - SIMD path (1K ≤ n < 100K)
   - Scalar path (n < 1K)

3. **Operation Fusion** - Always worked
   - Never lost during migration
   - Separate module (ternary_fusion.h)
   - 45.3 Gops/s effective

### Bad Ideas Discarded

1. **Runtime Dispatch** - Overhead too high
2. **Indirect Function Calls** - Prevents inlining
3. **Multiple Compilation Units** - Complexity without benefit

---

## Answer to Your Question

> "Why if the previous ternary engine had PGO already, Dense243 already, and Advanced fusion (4 fusion ops or similar) you now say that they are not integrated to the engine?"

**Truth:** Both options you stated are correct:

### Option 1: Parts Got "Buried" ✅ TRUE

**What got buried:**
- Backend system (`avx2_v1.cpp`, `avx2_v2.cpp`, `dispatch.cpp`)
- Canonical indexing (was in avx2_v2.cpp, not in active kernels)
- PGO scripts (exist but not used in recent builds)

**Location:** `src/core/simd/` - files exist but NOT compiled/used

### Option 2: Rebuilding on New Architecture ✅ TRUE

**What got rebuilt:**
- Direct kernels replaced backend dispatch
- Canonical indexing re-integrated from backends (today)
- OpenMP re-enabled in simplified architecture
- Three-path architecture reconstructed

**Result:** Better architecture than both v1.1.0 AND v1.2.0

---

## Current Architecture Truth

```
┌─────────────────────────────────────────┐
│   Active Components (Used)              │
├─────────────────────────────────────────┤
│ ✅ ternary_simd_kernels.h (with canonical) │
│ ✅ ternary_fusion.h (4 fusion ops)      │
│ ✅ ternary_canonical_index.h (LUTs)     │
│ ✅ OpenMP (12 threads, working)         │
│ ✅ Dense243 module (separate, working)  │
│ ⚠️ PGO scripts (exist, not used)        │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│   Archived Components (Preserved)       │
├─────────────────────────────────────────┤
│ 📦 ternary_backend_dispatch.cpp         │
│ 📦 ternary_backend_avx2_v1.cpp          │
│ 📦 ternary_backend_avx2_v2.cpp          │
│ 📦 ternary_backend_scalar.cpp           │
└─────────────────────────────────────────┘
```

**Status:**
- Backends archived, not deleted
- Canonical indexing extracted and re-integrated
- Fusion always worked, never lost
- Dense243 separate by design
- PGO available, not actively used

---

## The Wisdom of Preservation

**Why keep backend code?**

Because we could extract canonical indexing today and gain 74-1,100% improvement!

**The Pattern:**
```
Build complex system → Regress → Simplify → Extract best parts → Profit
```

**Lesson:** Don't delete code during simplification. Archive it. You might need parts later.

**Proof:** Canonical indexing in avx2_v2.cpp (archived Nov 2025) → Re-integrated (Nov 25 2025) → Massive gains

---

## Conclusion

**Backend System Status:** Good architecture, wrong timing, caused regression, simplified back, parts preserved and re-integrated as needed.

**Current State:** Best of both worlds
- Simplicity of v1.1.0 direct kernels
- Canonical indexing from v1.2.0 backends
- Performance: 39.1 Gops/s (40% better than either)

**Future:** Backend system architecture is there if we need multi-ISA. For now, simple is winning.

---

**Key Insight:** The "branching" wasn't a mistake. It was exploration. Backend system taught us what works (canonical indexing) and what doesn't (runtime dispatch overhead). We kept the good, discarded the bad, and achieved better performance than either architecture alone.
