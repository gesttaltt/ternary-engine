# Incremental Roadmap - Ternary Engine v3.0 (Safe & Revertible)

**Type:** Incremental Development Plan
**Date:** 2025-11-25
**Status:** Planning Phase
**Philosophy:** Add, don't modify. Validate, then integrate.

---

## Core Principle: Non-Breaking Incremental Development

**NEVER modify production code directly. ADD new code alongside it.**

### Git Branching Strategy

```
main (v1.3.0 - 45.3 Gops/s STABLE)
  │
  ├── phase-1-invariant-measurement (Week 1-2)
  │     └── merge → main (if validated)
  │
  ├── phase-2-hybrid-selector (Week 3-4)
  │     └── merge → main (if validated)
  │
  ├── phase-3-advanced-optimizations (Week 5-6)
  │     └── merge → main (if validated)
  │
  └── v3.0-release (if all phases successful)
```

**Rollback Plan:** If any phase fails, simply abandon the branch. Main remains stable at 45.3 Gops/s.

---

## 🚨 MANDATORY BENCHMARKING POLICY

### Benchmark After EVERY Feature (No Exceptions)

**CRITICAL RULE:** No code changes are merged without benchmarking validation.

```bash
# MANDATORY workflow after ANY code change:
1. Make code changes
2. Build module
3. Run benchmarks
4. Validate against baseline (auto-fail if regression > 5%)
5. Generate comparison report
6. Only proceed if validation passes
```

### Automated Validation

**Tool:** `benchmarks/utils/benchmark_validator.py`

**Usage:**
```bash
# After every feature addition
python build/build.py  # or build_experimental.py
python benchmarks/bench_phase0.py
python benchmarks/utils/benchmark_validator.py \\
    --baseline benchmarks/results/baseline_v1.3.0.json \\
    --current benchmarks/results/bench_results_LATEST.json \\
    --threshold 0.05  # 5% regression threshold

# Exit code 0 = PASS (proceed), 1 = FAIL (investigate)
```

### Baseline Performance (v1.3.0) - DO NOT REGRESS

**Element-wise Operations @ 1M elements:**
- tadd: **36,111 Mops/s** (±5% acceptable)
- tmul: **35,540 Mops/s** (±5% acceptable)
- tmin: **31,569 Mops/s** (±5% acceptable)
- tmax: **29,471 Mops/s** (±5% acceptable)
- tnot: **39,056 Mops/s** (±5% acceptable)

**Fusion Operations @ 1M elements:**
- fused_tnot_tadd: **45,300 Mops/s** effective (±5% acceptable)

**Files:**
- Baseline: `benchmarks/results/baseline_v1.3.0.json`
- Validator: `benchmarks/utils/benchmark_validator.py`
- Reports: `reports/benchmark_validation/`

### Regression Thresholds

**Auto-fail conditions:**
- Any operation regresses >5% from baseline
- Build fails or crashes
- Test suite fails (correctness)

**Require investigation:**
- Variance >5% between runs (measurement noise)
- Performance improves but correctness fails
- New warnings or errors in build

### Continuous Benchmarking Workflow

**After EVERY commit:**
```bash
# 1. Correctness tests (MUST PASS)
python tests/test_phase0.py

# 2. Quick performance check
python benchmarks/bench_phase0.py --quick

# 3. Validate against baseline
python benchmarks/utils/benchmark_validator.py --auto

# 4. If PASS: continue. If FAIL: investigate immediately.
```

**Before EVERY merge to main:**
```bash
# 1. Full test suite
python run_tests.py --full

# 2. Comprehensive benchmarks
python benchmarks/bench_phase0.py  # Full size range
python benchmarks/bench_fusion.py  # Fusion operations

# 3. Validation with report generation
python benchmarks/utils/benchmark_validator.py \\
    --baseline benchmarks/results/baseline_v1.3.0.json \\
    --current benchmarks/results/bench_results_LATEST.json \\
    --output reports/benchmark_validation/validation_pre_merge.md

# 4. Review validation report
# 5. Only merge if ALL validations pass
```

### Performance Tracking

**Maintain history:**
```
benchmarks/results/
  baseline_v1.3.0.json          # FROZEN baseline
  bench_results_20251125_*.json # Historical results
  bench_results_LATEST.json     # Current (symlink)

reports/benchmark_validation/
  validation_20251125_*.md      # Historical validation reports
  validation_LATEST.md          # Current (symlink)
```

**Git commit message must include:**
```
FEAT: Add new optimization X

Performance Impact:
  - tadd@1M: 36.1 → 38.2 Gops/s (+5.8%)
  - Validation: PASSED (0% regression)
  - Report: reports/benchmark_validation/validation_20251126_143022.md
```

---

## Current Stable Baseline (v1.3.0)

### ✅ Production Files (DO NOT MODIFY)

```
src/core/simd/ternary_simd_kernels.h       # 45.3 Gops/s validated kernel
src/core/simd/ternary_canonical_index.h    # Canonical indexing LUTs
src/core/simd/ternary_fusion.h             # 4 fusion operations
src/engine/bindings_core_ops.cpp           # Production Python bindings
build/build.py                              # Production build script
```

**Performance:** 45.3 Gops/s effective, 39.1 Gops/s peak
**Status:** FROZEN - These files are off-limits for modifications

### 📦 Available Infrastructure (Not Active)

```
src/core/profiling/ternary_profiler.h      # VTune integration (feature flag ready)
src/engine/bindings_dense243.cpp           # Dense243 module (separate)
src/engine/bindings_backend_api.cpp        # Archived backend (preserved)
build/build_pgo_unified.py                 # PGO build (not used)
```

**Pattern to Follow:** Feature flags like `TERNARY_ENABLE_VTUNE`

---

## Phase 1: Invariant Measurement Suite (Week 1-2)

**Goal:** Understand performance characteristics WITHOUT modifying production code

**Git Branch:** `phase-1-invariant-measurement`

### What We Add (New Files Only)

```
benchmarks/
  bench_invariants.py                     # NEW - Main measurement suite
  utils/
    hardware_metrics.py                   # NEW - CPU counters (perf/VTune)
    geometric_metrics.py                  # NEW - Entropy, correlation, fractals
  analysis/
    cluster_analysis.py                   # NEW - Pattern discovery
  datasets/
    synthetic/
      low_entropy_1M.npy                  # NEW - Synthetic datasets
      medium_entropy_1M.npy               # NEW
      high_entropy_1M.npy                 # NEW
      fractal_patterns_1M.npy             # NEW

reports/
  invariant_analysis_YYYY-MM-DD.md        # NEW - Findings report
```

### What We DO NOT Modify

- ❌ src/core/simd/ternary_simd_kernels.h
- ❌ src/engine/bindings_core_ops.cpp
- ❌ build/build.py
- ❌ Any production code

### Integration Strategy

**Profiler Integration (Optional - Feature Flag):**
```cpp
// Add to ternary_simd_kernels.h if profiling enabled
#ifdef TERNARY_ENABLE_PROFILING
#include "../profiling/ternary_profiler.h"
#define PROFILE_SIMD_OP(name) TERNARY_PROFILE_TASK_BEGIN(g_domain, name)
#else
#define PROFILE_SIMD_OP(name) /* no-op */
#endif
```

**Build Command:**
```bash
# Production build (unchanged)
python build/build.py

# Profiling build (optional)
python build/build.py --enable-profiling
```

### Validation Criteria

- [ ] Benchmark suite runs without errors
- [ ] Identifies ≥3 distinct performance regions with p < 0.05
- [ ] **MANDATORY: No regression in production performance (45.3 Gops/s maintained)**
- [ ] Reproducible results across multiple runs

### 🚨 MANDATORY: Benchmark After Phase 1

**Before merging Phase 1 to main:**
```bash
# 1. Build production module (unchanged)
python build/build.py

# 2. Run comprehensive benchmarks
python benchmarks/bench_phase0.py

# 3. Validate NO regression
python benchmarks/utils/benchmark_validator.py \\
    --baseline benchmarks/results/baseline_v1.3.0.json \\
    --current benchmarks/results/bench_results_$(date +%Y%m%d_%H%M%S).json \\
    --output reports/benchmark_validation/phase1_validation.md

# 4. MUST show 0% regression on all operations
# 5. Phase 1 only ADDS measurement code, does NOT modify kernels
```

**Expected Result:**
- All operations: **0.0% delta** (no changes to production code)
- Validation: **PASS** (exact match to baseline)
- Measurement infrastructure: Adds overhead ONLY when enabled via feature flag

### Deliverables

1. **bench_invariants.py** - Comprehensive measurement suite
2. **Synthetic datasets** - Low/medium/high entropy test cases
3. **Analysis report** - Statistical findings with clustering
4. **Zero impact on production** - Main branch unchanged

### Merge Decision

**Merge to main IF:**
- All benchmarks pass
- Production performance unchanged
- Documentation complete
- No breaking changes

**Abandon branch IF:**
- Cannot measure invariants reliably
- Overhead too high (>5% impact)
- Results not reproducible

---

## Phase 2: Hybrid Selector (Week 3-4)

**Goal:** Add adaptive path selection WITHOUT breaking production kernel

**Git Branch:** `phase-2-hybrid-selector` (branched from main AFTER Phase 1 merge)

### What We Add (New Files Only)

```
src/core/simd/
  ternary_hybrid_kernels.h                # NEW - Hybrid implementation
  ternary_path_selector.h                 # NEW - Runtime selection logic
  ternary_geometric_path.h                # NEW - Optimized for low entropy
  ternary_cold_path.h                     # NEW - Optimized for high entropy

src/engine/
  bindings_experimental.cpp               # NEW - Experimental bindings

build/
  build_experimental.py                   # NEW - Experimental build script

tests/
  test_hybrid_selector.py                 # NEW - Hybrid validation tests

benchmarks/
  bench_hybrid.py                         # NEW - Hybrid performance tests
```

### What We DO NOT Modify

- ❌ src/core/simd/ternary_simd_kernels.h (production kernel)
- ❌ src/engine/bindings_core_ops.cpp (production bindings)
- ❌ build/build.py (production build)

### Architecture Design

**Separate Module Approach:**
```python
# Production module (unchanged)
import ternary_simd_engine as tse
result = tse.tadd(a, b)  # 45.3 Gops/s validated

# Experimental module (new)
import ternary_experimental as tex
result = tex.tadd_hybrid(a, b)  # With adaptive path selection
```

**C++ Implementation (New File):**
```cpp
// src/core/simd/ternary_hybrid_kernels.h (NEW FILE)
#ifndef TERNARY_HYBRID_KERNELS_H
#define TERNARY_HYBRID_KERNELS_H

#include "ternary_simd_kernels.h"       // Production baseline
#include "ternary_path_selector.h"      // NEW
#include "ternary_geometric_path.h"     // NEW
#include "ternary_cold_path.h"          // NEW

namespace ternary {
namespace experimental {

// Hybrid kernel with runtime path selection
template <bool Sanitize = true>
static inline __m256i tadd_hybrid(__m256i a, __m256i b) {
    PathSelector selector;

    if (selector.use_geometric_path(a, b)) {
        return geometric::tadd_simd<Sanitize>(a, b);
    } else {
        // Fall back to production kernel
        return ::tadd_simd<Sanitize>(a, b);
    }
}

} // namespace experimental
} // namespace ternary

#endif
```

**Path Implementations:**
```cpp
// src/core/simd/ternary_geometric_path.h (NEW FILE)
// Optimized for low entropy, high correlation inputs
namespace ternary {
namespace experimental {
namespace geometric {

template <bool Sanitize = true>
static inline __m256i tadd_simd(__m256i a, __m256i b) {
    // Enhanced canonical indexing with prefetching
    // ... implementation ...
}

} // namespace geometric
} // namespace experimental
} // namespace ternary


// src/core/simd/ternary_cold_path.h (NEW FILE)
// Optimized for high entropy, random inputs (copy of production kernel)
namespace ternary {
namespace experimental {
namespace cold {

template <bool Sanitize = true>
static inline __m256i tadd_simd(__m256i a, __m256i b) {
    // Identical to production kernel for now
    return ::tadd_simd<Sanitize>(a, b);
}

} // namespace cold
} // namespace experimental
} // namespace ternary
```

### Build System Integration

**Production Build (Unchanged):**
```bash
python build/build.py
# Output: ternary_simd_engine.pyd (45.3 Gops/s validated)
```

**Experimental Build (New):**
```bash
python build/build_experimental.py
# Output: ternary_experimental.pyd (hybrid architecture)
```

**Both Modules Can Coexist:**
```python
import ternary_simd_engine as prod  # Stable production
import ternary_experimental as exp  # New hybrid

# Compare performance
prod_result = prod.tadd(a, b)
exp_result = exp.tadd_hybrid(a, b)
assert np.array_equal(prod_result, exp_result)  # Must match!
```

### Validation Criteria

- [ ] Experimental module builds without errors
- [ ] Hybrid kernel matches production correctness (100% pass rate)
- [ ] **MANDATORY: Performance ≥ production baseline (no regression)**
- [ ] Path selection overhead < 5%
- [ ] **MANDATORY: Production module unaffected (still 45.3 Gops/s)**

### 🚨 MANDATORY: Benchmark After Phase 2

**Dual validation - Both modules MUST be benchmarked:**

```bash
# 1. Build BOTH modules
python build/build.py                    # Production
python build/build_experimental.py        # Experimental

# 2. Benchmark production (MUST match baseline)
python benchmarks/bench_phase0.py
python benchmarks/utils/benchmark_validator.py \\
    --baseline benchmarks/results/baseline_v1.3.0.json \\
    --current benchmarks/results/bench_results_PRODUCTION.json \\
    --output reports/benchmark_validation/phase2_production_validation.md

# 3. Benchmark experimental (MUST match OR exceed baseline)
python benchmarks/bench_experimental.py  # NEW benchmark for experimental module
python benchmarks/utils/benchmark_validator.py \\
    --baseline benchmarks/results/baseline_v1.3.0.json \\
    --current benchmarks/results/bench_results_EXPERIMENTAL.json \\
    --threshold 0.0  # Allow 0% regression (must equal or exceed)
    --output reports/benchmark_validation/phase2_experimental_validation.md

# 4. Compare experimental vs production
python benchmarks/utils/compare_modules.py \\
    --baseline-module ternary_simd_engine \\
    --experimental-module ternary_experimental \\
    --output reports/benchmark_validation/phase2_comparison.md
```

**Expected Results:**

**Production Module:**
- All operations: **0.0% delta** (unchanged)
- Validation: **PASS** (exact match to v1.3.0)

**Experimental Module:**
- All operations: **≥0% delta** (equal or better)
- Target: **+10%** improvement (50 Gops/s effective)
- Minimum acceptable: **0%** (no regression)
- Path selection overhead: **<5%** on high-entropy inputs

**Decision Matrix:**

| Experimental Performance | Action |
|--------------------------|--------|
| ≥ +10% (50+ Gops/s) | ✅ **Merge to main** - Replace production |
| +5% to +10% | ✅ **Merge to main** - Worthwhile improvement |
| 0% to +5% | ⚠️ **Keep experimental** - Not enough gain to justify complexity |
| < 0% (regression) | ❌ **Abandon branch** - Hybrid slower than production |

### Deliverables

1. **ternary_experimental module** - Separate Python module
2. **Hybrid kernel implementation** - With path selection
3. **Comprehensive tests** - Correctness and performance
4. **Zero impact on production** - Main module unchanged

### Merge Decision

**Merge to main IF:**
- Hybrid matches or exceeds production performance
- No correctness issues
- Overhead acceptable (<5%)
- All tests pass

**Keep as experimental IF:**
- Performance gains marginal (<10%)
- Path selection overhead too high
- Not ready for production

**Abandon branch IF:**
- Hybrid slower than production
- Correctness issues
- Cannot validate reliably

---

## Phase 3: Advanced Optimizations (Week 5-6)

**Goal:** Add PGO, Dense243 integration, advanced fusion WITHOUT breaking validated code

**Git Branch:** `phase-3-advanced-optimizations` (branched from main AFTER Phase 2 merge)

### Sub-Phase 3.1: PGO Integration

**What We Add:**
```
build/
  build_pgo_production.py                 # NEW - PGO for production kernel
  profiles/
    geometric_workload.json               # NEW - Profiling data
    random_workload.json                  # NEW
```

**What We DO NOT Modify:**
- ❌ Source code (PGO only changes compilation, not code)

**Build Process:**
```bash
# Step 1: Instrument build
python build/build_pgo_production.py --phase instrument

# Step 2: Generate profile data
python benchmarks/bench_phase0.py --pgo-profile

# Step 3: Optimized build
python build/build_pgo_production.py --phase optimize

# Output: ternary_simd_engine.pyd (PGO-optimized)
```

**Rollback:** Keep non-PGO build artifacts. If PGO causes issues, revert to standard build.

**Validation:**
- [ ] PGO build matches correctness of standard build
- [ ] **MANDATORY: Performance improvement 5-15% (expected)**
- [ ] **MANDATORY: No regressions on any operation**

### 🚨 MANDATORY: Benchmark After Sub-Phase 3.1 (PGO)

```bash
# 1. Build PGO-optimized module
python build/build_pgo_production.py --full

# 2. Benchmark PGO build
python benchmarks/bench_phase0.py
python benchmarks/bench_fusion.py

# 3. Validate improvement
python benchmarks/utils/benchmark_validator.py \\
    --baseline benchmarks/results/baseline_v1.3.0.json \\
    --current benchmarks/results/bench_results_PGO.json \\
    --threshold -0.05  # Negative = require improvement
    --output reports/benchmark_validation/phase3.1_pgo_validation.md

# 4. MUST show 5-15% improvement
```

**Expected Results:**
- All operations: **+5% to +15%** improvement
- Target: **50-52 Gops/s** effective throughput
- Minimum acceptable: **+3%** (48 Gops/s)

**Decision:**
- If <+3%: Revert to non-PGO build (not worth compilation complexity)
- If +3% to +5%: Consider PGO optional
- If +5% to +15%: **Adopt PGO** as standard build
- If >+15%: **Unexpected but great!** Investigate why

---

### Sub-Phase 3.2: Dense243 Integration

**What We Add (New Files):**
```
src/engine/lib/dense243_integration/
  dense243_wrapper.h                      # NEW - Integration layer
  dense243_pack_simd.h                    # NEW - SIMD-accelerated pack/unpack

src/engine/
  bindings_core_ops_dense243.cpp          # NEW - Dense243-enabled bindings

build/
  build_dense243_integrated.py            # NEW - Build with Dense243
```

**What We DO NOT Modify:**
- ❌ src/core/simd/ternary_simd_kernels.h (production kernel)
- ❌ src/engine/bindings_core_ops.cpp (standard bindings)
- ❌ Existing Dense243 module (already validated separately)

**Integration Strategy:**
```python
# Production module (unchanged)
import ternary_simd_engine as tse

# Dense243-integrated module (new)
import ternary_simd_dense243 as tsd

# Automatic format selection
a_dense = tsd.TernaryArray([1, 0, -1, 1, 0] * 1000000)  # Uses Dense243 internally
b_dense = tsd.TernaryArray([0, 1, -1, 0, 1] * 1000000)
result = tsd.tadd(a_dense, b_dense)  # Packed arithmetic
```

**Architecture:**
```cpp
// NEW: src/engine/lib/dense243_integration/dense243_wrapper.h
namespace ternary {
namespace dense243 {

class TernaryArrayDense243 {
    std::vector<uint8_t> packed_data;  // 5 trits per byte

    // SIMD-accelerated pack/unpack
    void pack_from_2bit(__m256i* src, size_t n);
    void unpack_to_2bit(__m256i* dst, size_t n);

    // Arithmetic on packed data
    void add(const TernaryArrayDense243& other);
};

} // namespace dense243
} // namespace ternary
```

**Validation:**
- [ ] Dense243 pack/unpack correctness
- [ ] Memory reduction: 5× vs standard 2-bit
- [ ] **MANDATORY: Performance on large arrays (10M+): +20-30% expected**
- [ ] **MANDATORY: Production module unaffected**

### 🚨 MANDATORY: Benchmark After Sub-Phase 3.2 (Dense243)

```bash
# 1. Build Dense243-integrated module
python build/build_dense243_integrated.py

# 2. Benchmark BOTH modules on large arrays (10M elements)
python benchmarks/bench_dense243.py --sizes 10000000  # NEW benchmark

# 3. Validate Dense243 performance
python benchmarks/utils/benchmark_validator.py \\
    --baseline benchmarks/results/baseline_v1.3.0.json \\
    --current benchmarks/results/bench_results_DENSE243.json \\
    --threshold 0.0  # Dense243 must not regress on small arrays
    --output reports/benchmark_validation/phase3.2_dense243_validation.md

# 4. Measure memory efficiency
python benchmarks/utils/measure_memory.py \\
    --baseline-module ternary_simd_engine \\
    --dense243-module ternary_simd_dense243 \\
    --output reports/benchmark_validation/phase3.2_memory_analysis.md

# 5. Validate production unchanged
python benchmarks/bench_phase0.py
python benchmarks/utils/benchmark_validator.py \\
    --baseline benchmarks/results/baseline_v1.3.0.json \\
    --current benchmarks/results/bench_results_PRODUCTION_CHECK.json \\
    --output reports/benchmark_validation/phase3.2_production_unchanged.md
```

**Expected Results:**

**Small arrays (1M):**
- Dense243: **-10% to +0%** (pack/unpack overhead acceptable)
- Minimum: No worse than -15%

**Large arrays (10M):**
- Dense243: **+20% to +30%** (memory bandwidth savings)
- Target: 56+ Gops/s sustained
- Minimum: +15%

**Memory:**
- Storage: **5× reduction** (validated)
- Bandwidth: **5× less** traffic to RAM

**Production:**
- All operations: **0.0% delta** (unchanged)

**Decision:**
- If Dense243 performance < baseline on 10M: Investigate pack/unpack overhead
- If memory reduction < 4×: Implementation issue, debug
- If production affected: **Critical - rollback immediately**

---

### Sub-Phase 3.3: Advanced Fusion Patterns

**What We Add (New Files):**
```
src/core/simd/
  ternary_fusion_advanced.h               # NEW - 3-op fusion chains

src/engine/
  bindings_fusion_advanced.cpp            # NEW - Advanced fusion bindings

tests/
  test_fusion_advanced.py                 # NEW - 3-op fusion tests

benchmarks/
  bench_fusion_advanced.py                # NEW - Advanced fusion benchmarks
```

**What We DO NOT Modify:**
- ❌ src/core/simd/ternary_fusion.h (validated 2-op fusion)
- ❌ src/engine/bindings_core_ops.cpp

**Architecture:**
```cpp
// NEW: src/core/simd/ternary_fusion_advanced.h
#ifndef TERNARY_FUSION_ADVANCED_H
#define TERNARY_FUSION_ADVANCED_H

#include "ternary_simd_kernels.h"
#include "ternary_fusion.h"  // Existing 2-op fusion

namespace ternary {
namespace fusion {
namespace advanced {

// 3-operation fusion: tadd(tmul(a, b), c)
template <bool Sanitize = true>
static inline __m256i fused_mul_add(__m256i a, __m256i b, __m256i c) {
    __m256i temp = tmul_simd<false>(a, b);  // Internal: no sanitize
    return tadd_simd<Sanitize>(temp, c);    // Final: sanitize if requested
}

// 4-operation fusion: tnot(tadd(tmul(a, b), c))
template <bool Sanitize = true>
static inline __m256i fused_not_mul_add(__m256i a, __m256i b, __m256i c) {
    __m256i temp = fused_mul_add<false>(a, b, c);
    return tnot_simd<Sanitize>(temp);
}

} // namespace advanced
} // namespace fusion
} // namespace ternary

#endif
```

**Python API (New Module):**
```python
import ternary_fusion_advanced as tfa

# 3-op fusion
result = tfa.fused_mul_add(a, b, c)  # tadd(tmul(a, b), c)

# 4-op fusion
result = tfa.fused_not_mul_add(a, b, c)  # tnot(tadd(tmul(a, b), c))
```

**Validation:**
- [ ] Correctness: matches sequential operations
- [ ] **MANDATORY: Effective throughput: 2-3× expected (90+ Gops/s)**
- [ ] **MANDATORY: Production 2-op fusion unaffected**

### 🚨 MANDATORY: Benchmark After Sub-Phase 3.3 (Advanced Fusion)

```bash
# 1. Build advanced fusion module
python build/build_fusion_advanced.py

# 2. Benchmark 3-op and 4-op fusion
python benchmarks/bench_fusion_advanced.py  # NEW benchmark for 3-op fusion

# 3. Validate effective throughput
python benchmarks/utils/benchmark_validator.py \\
    --baseline benchmarks/results/baseline_v1.3.0.json \\
    --current benchmarks/results/bench_results_FUSION_ADVANCED.json \\
    --threshold -0.50  # Negative = require massive improvement (2-3×)
    --output reports/benchmark_validation/phase3.3_fusion_validation.md

# 4. Validate existing 2-op fusion unchanged
python benchmarks/bench_fusion.py  # Existing 2-op benchmark
python benchmarks/utils/benchmark_validator.py \\
    --baseline benchmarks/results/baseline_v1.3.0.json \\
    --current benchmarks/results/bench_results_2OP_FUSION_CHECK.json \\
    --output reports/benchmark_validation/phase3.3_2op_unchanged.md

# 5. Effective throughput calculation
python benchmarks/utils/calculate_effective_throughput.py \\
    --fusion-type 3op \\
    --operations 3000000  # 3M ops (3× 1M)
    --time-us <measured> \\
    --output reports/benchmark_validation/phase3.3_effective_throughput.md
```

**Expected Results:**

**3-op Fusion (fused_mul_add):**
- Speedup vs sequential: **2-3×**
- Effective throughput: **90+ Gops/s** (3M ops in ~33 µs)
- Minimum acceptable: **2×** (80 Gops/s)

**4-op Fusion (fused_not_mul_add):**
- Speedup vs sequential: **3-4×**
- Effective throughput: **100+ Gops/s** (4M ops in ~40 µs)
- Minimum acceptable: **2.5×** (90 Gops/s)

**Existing 2-op Fusion:**
- All operations: **0.0% delta** (unchanged)
- fused_tnot_tadd still **15.93×** @ 1M
- fused_tnot_tadd still **45.3 Gops/s** effective

**Decision:**
- If 3-op fusion < 2×: Not worth complexity, abandon
- If 3-op fusion 2-3×: **Merge to main**
- If 3-op fusion > 3×: **Exceptional!** Investigate and document
- If 2-op fusion regresses: **Critical - rollback immediately**

---

## Rollback Strategy for Each Phase

### Phase 1 Rollback (Invariant Measurement)
**If issues discovered:**
```bash
git checkout main  # Discard phase-1 branch
# Production remains at 45.3 Gops/s, zero impact
```

### Phase 2 Rollback (Hybrid Selector)
**If hybrid underperforms:**
```bash
git checkout main  # Discard phase-2 branch
# Keep experimental module as research artifact
# Production module unchanged
```

**Partial integration:**
```bash
# If hybrid works for some operations but not others
git cherry-pick <working-commits>  # Select only validated parts
```

### Phase 3 Rollback (Advanced Optimizations)

**PGO Rollback:**
```bash
python build/build.py  # Standard build, no PGO
# Zero code changes, just compilation flags
```

**Dense243 Rollback:**
```bash
# Don't install ternary_simd_dense243 module
# Keep standard ternary_simd_engine
```

**Advanced Fusion Rollback:**
```bash
# Don't install ternary_fusion_advanced module
# Keep existing 2-op fusion
```

---

## Performance Targets (Incremental)

### After Phase 1: Invariant Measurement
- **Performance:** 45.3 Gops/s (unchanged)
- **Knowledge Gain:** Understanding of performance regions
- **Risk:** Zero (no code changes)

### After Phase 2: Hybrid Selector
- **Performance Target:** 50+ Gops/s (10% gain)
- **Fallback:** Keep production at 45.3 Gops/s
- **Risk:** Low (separate module)

### After Phase 3.1: PGO
- **Performance Target:** 50-52 Gops/s (5-15% gain)
- **Fallback:** Revert to non-PGO build
- **Risk:** Low (no code changes)

### After Phase 3.2: Dense243
- **Performance Target:** 56+ Gops/s on large arrays
- **Fallback:** Use standard encoding
- **Risk:** Low (separate module)

### After Phase 3.3: Advanced Fusion
- **Performance Target:** 90+ Gops/s effective throughput
- **Fallback:** Use existing 2-op fusion
- **Risk:** Low (separate module)

### v3.0 Final (If All Phases Succeed)
- **Element-wise peak:** 52+ Gops/s (vs 39.1 current)
- **Effective throughput:** 90+ Gops/s (vs 45.3 current)
- **Memory efficiency:** 5× with Dense243 (optional)

---

## Testing Strategy

### Continuous Validation

**After EVERY commit:**
```bash
# 1. Production correctness (must pass)
python tests/test_phase0.py

# 2. Production performance (must maintain)
python benchmarks/bench_phase0.py --quick
# Expected: 45.3 Gops/s ± 3%

# 3. Experimental tests (if applicable)
python tests/test_experimental.py
```

**Before EVERY merge to main:**
```bash
# Full validation suite
python run_tests.py --full
python benchmarks/bench_phase0.py --comprehensive
python benchmarks/bench_fusion.py

# Regression detection
python scripts/compare_benchmarks.py \
  benchmarks/results/baseline_v1.3.0.json \
  benchmarks/results/current.json
```

### Regression Thresholds

**Auto-fail merge if:**
- Any correctness test fails (0% tolerance)
- Production performance < 43 Gops/s (5% regression threshold)
- Memory usage increases >10% without Dense243

**Require investigation if:**
- Performance variance >5% between runs
- New warnings or errors in build log
- Test coverage drops <95%

---

## Feature Flag System

### Compile-Time Flags (Following ternary_profiler.h Pattern)

```cpp
// src/core/config/ternary_features.h (NEW)
#ifndef TERNARY_FEATURES_H
#define TERNARY_FEATURES_H

// Phase 1: Profiling integration
#ifdef TERNARY_ENABLE_PROFILING
#include "../profiling/ternary_profiler.h"
#define PROFILE_SIMD_OP(name) TERNARY_PROFILE_TASK_BEGIN(g_domain, name)
#else
#define PROFILE_SIMD_OP(name) /* no-op */
#endif

// Phase 2: Hybrid selector
#ifdef TERNARY_ENABLE_HYBRID
#include "simd/ternary_hybrid_kernels.h"
#define USE_HYBRID_KERNEL 1
#else
#define USE_HYBRID_KERNEL 0
#endif

// Phase 3.2: Dense243 integration
#ifdef TERNARY_ENABLE_DENSE243
#include "packing/dense243_integration.h"
#define USE_DENSE243 1
#else
#define USE_DENSE243 0
#endif

#endif
```

**Build Commands:**
```bash
# Production (default)
python build/build.py
# Flags: None

# With profiling
python build/build.py --enable-profiling
# Flags: -DTERNARY_ENABLE_PROFILING

# With hybrid selector
python build/build_experimental.py --enable-hybrid
# Flags: -DTERNARY_ENABLE_HYBRID

# With Dense243
python build/build.py --enable-dense243
# Flags: -DTERNARY_ENABLE_DENSE243

# All features
python build/build_full.py --all-features
# Flags: -DTERNARY_ENABLE_PROFILING -DTERNARY_ENABLE_HYBRID -DTERNARY_ENABLE_DENSE243
```

---

## Documentation Requirements

### Each Phase Must Include:

1. **README_PHASE_N.md** - Phase overview and goals
2. **CHANGELOG_PHASE_N.md** - What changed, what's new
3. **VALIDATION_PHASE_N.md** - Test results and benchmarks
4. **ROLLBACK_PHASE_N.md** - How to revert if needed

### Example (Phase 2):
```
docs/phases/
  phase-2-hybrid-selector/
    README.md                   # Overview and architecture
    CHANGELOG.md                # Files added/modified
    VALIDATION.md               # Test results
    ROLLBACK.md                 # Reversion procedure
    benchmarks/
      baseline_phase2.json      # Performance before
      results_phase2.json       # Performance after
```

---

## Success Metrics

### Phase 1: Invariant Measurement
- [ ] ≥3 distinct performance regions identified
- [ ] Statistical significance p < 0.05
- [ ] Zero impact on production (45.3 Gops/s maintained)
- [ ] Reproducible across 10+ runs

### Phase 2: Hybrid Selector
- [ ] Experimental module builds successfully
- [ ] Hybrid matches production correctness (100%)
- [ ] Performance ≥ 50 Gops/s (10% gain over production)
- [ ] Path selection overhead < 5%
- [ ] Production module unaffected

### Phase 3.1: PGO
- [ ] PGO build matches standard build correctness
- [ ] Performance gain 5-15% (50-52 Gops/s)
- [ ] No regressions on any operation

### Phase 3.2: Dense243
- [ ] Memory reduction: 5× validated
- [ ] Performance on 10M+ elements: +20-30%
- [ ] Pack/unpack overhead acceptable
- [ ] Production module unaffected

### Phase 3.3: Advanced Fusion
- [ ] 3-op fusion correctness validated
- [ ] Effective throughput ≥ 90 Gops/s
- [ ] Existing 2-op fusion unaffected

### v3.0 Release Criteria
- [ ] All phases validated independently
- [ ] No regressions vs v1.3.0 baseline
- [ ] Comprehensive documentation
- [ ] Clean git history with revertible commits

---

## Execution Timeline (Conservative)

### Week 1: Phase 1.1 - Infrastructure
- Day 1-2: Set up git branch, directory structure
- Day 3-4: Implement hardware metrics (perf/VTune integration)
- Day 5: Validate metrics collection, no production impact

**Checkpoint:** Merge to main if metrics work, zero regression

---

### Week 2: Phase 1.2 - Analysis
- Day 1-2: Implement geometric metrics (entropy, correlation)
- Day 3-4: Generate synthetic datasets, run measurements
- Day 5: Statistical analysis, write report

**Checkpoint:** Merge to main if analysis complete, findings documented

---

### Week 3: Phase 2.1 - Hybrid Infrastructure
- Day 1-2: Create experimental module structure
- Day 3-4: Implement path selector and geometric path
- Day 5: Build experimental module, validate it compiles

**Checkpoint:** Can build both modules, production unaffected

---

### Week 4: Phase 2.2 - Hybrid Validation
- Day 1-2: Comprehensive correctness tests
- Day 3-4: Performance benchmarking
- Day 5: Decision: Merge or keep experimental

**Checkpoint:** Merge to main if performance ≥ 50 Gops/s

---

### Week 5: Phase 3.1 & 3.2 - PGO + Dense243
- Day 1-2: PGO integration and validation
- Day 3-4: Dense243 integration layer
- Day 5: Benchmark both optimizations

**Checkpoint:** Merge PGO if validated, Dense243 as optional module

---

### Week 6: Phase 3.3 - Advanced Fusion
- Day 1-2: Implement 3-op fusion patterns
- Day 3-4: Validation and benchmarking
- Day 5: Final v3.0 validation and release

**Checkpoint:** v3.0 release if all criteria met

---

## Risk Mitigation

### Risk 1: Phase Takes Longer Than Expected
**Mitigation:** Each phase is independent. Can ship v2.1, v2.2, etc. incrementally
**Fallback:** Keep production at v1.3.0 until phase ready

### Risk 2: Hybrid Selector Doesn't Help
**Mitigation:** Keep as experimental module, don't merge to main
**Fallback:** Production remains at 45.3 Gops/s

### Risk 3: PGO Causes Regressions
**Mitigation:** Separate PGO build from standard build
**Fallback:** Revert to non-PGO compilation

### Risk 4: Dense243 Pack/Unpack Too Slow
**Mitigation:** Keep Dense243 as optional separate module
**Fallback:** Use standard 2-bit encoding

### Risk 5: Advanced Fusion Adds Complexity
**Mitigation:** Keep advanced fusion as separate module
**Fallback:** Use existing 2-op fusion

---

## Conclusion

**Philosophy:** Add, don't replace. Validate, then integrate.

**Current Stable Baseline:** v1.3.0 at 45.3 Gops/s (FROZEN)

**Development Strategy:**
- New files, never modify production code
- Git branches for each phase
- Feature flags for experimental features
- Separate modules that coexist with production
- Merge only after validation

**Rollback Strategy:**
- Abandon branch if phase fails
- Production code never broken
- Each phase independently revertible

**Timeline:** 6 weeks, but flexible
- Can ship partial results (v2.1, v2.2, etc.)
- Production remains stable throughout
- v3.0 release only if all phases succeed

**Expected Outcome:**
- Element-wise: 52+ Gops/s (conservative PGO estimate)
- Effective: 90+ Gops/s (with advanced fusion)
- Memory: 5× reduction (with Dense243 option)
- Risk: Minimal (production code protected)

---

**Status:** Planning complete, ready for Phase 1 execution
**Next Action:** Create `phase-1-invariant-measurement` git branch
**Decision Point:** After each phase, validate before proceeding

---

**Document Version:** 2.0 (Incremental & Revertible)
**Last Updated:** 2025-11-25
**Supersedes:** hybrid_architecture_roadmap_v3.0.md (replaced with safer approach)
