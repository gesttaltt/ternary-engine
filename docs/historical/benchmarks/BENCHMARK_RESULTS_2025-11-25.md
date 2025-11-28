# Benchmark Results - 2025-11-25

**Date:** 2025-11-25
**Platform:** Windows x64, AMD64 Family 23 Model 96, 12 cores
**Modules Tested:** ternary_simd_engine, ternary_dense243_module, ternary_tritnet_gemm

---

## Executive Summary

Fresh build and benchmark of all three modules after Phase 3.2/3.3 completion. Standard operations perform excellently (20.7 Gops/s peak), but **CRITICAL ISSUE FOUND:** Matrix multiplication (GEMM) is 50-100× slower than target performance.

**Key Findings:**
- ✅ **Standard Operations:** 20.7 Gops/s peak (tadd @ 1M elements) - **EXCELLENT**
- ❌ **Matrix Multiplication:** 0.24-0.39 Gops/s - **CRITICAL PERFORMANCE GAP**
- ✅ **Correctness:** All tests passing (standard ops + GEMM)
- ⚠️ **GEMM Status:** Implementation correct but needs urgent optimization

---

## Standard Operations Performance (ternary_simd_engine)

### Peak Throughput

| Operation | Peak Throughput | Array Size | ns/element |
|-----------|----------------|------------|------------|
| **tadd** | **20,756 Mops/s** | 1M | 0.048 |
| **tmul** | **20,325 Mops/s** | 1M | 0.049 |
| **tmin** | 13,778 Mops/s | 100K | 0.073 |
| **tmax** | 13,200 Mops/s | 100K | 0.076 |
| **tnot** | 15,726 Mops/s | 100K | 0.064 |

### Scaling Behavior

**Small Arrays (32 elements):**
- 20-30 Mops/s
- Function call overhead dominates

**Medium Arrays (1,000 elements):**
- 600-700 Mops/s
- L2 cache-resident

**Large Arrays (100,000 elements):**
- 13,200-15,700 Mops/s
- Peak SIMD throughput

**Very Large Arrays (1,000,000 elements):**
- 20,300-20,700 Mops/s (tadd/tmul)
- 2,600-7,000 Mops/s (tmin/tmax/tnot)
- OpenMP parallelization effective for tadd/tmul

### Speedup vs Pure Python

| Operation | Average Speedup |
|-----------|----------------|
| tadd | 2,058× |
| tmul | 1,784× |
| tmin | 2,132× |
| tmax | 2,058× |
| tnot | 1,070× |

**Status:** ✅ **EXCELLENT PERFORMANCE** - Meets all targets

---

## Matrix Multiplication Performance (ternary_tritnet_gemm)

### Benchmark Results

| Matrix Size | TritNet GEMM | NumPy BLAS | Ratio | Time (TritNet) |
|-------------|--------------|------------|-------|----------------|
| 8×16×20 | **0.39 Gops/s** | 1.42 Gops/s | 3.6× slower | 0.01 ms |
| 128×128×130 | **0.30 Gops/s** | 15.11 Gops/s | 50.6× slower | 7.13 ms |
| 512×512×510 | **0.23 Gops/s** | 91.06 Gops/s | 387.5× slower | 569 ms |
| 2048×2048×2050 | **0.24 Gops/s** | 189.60 Gops/s | 784.0× slower | 35.6 sec |

### Performance Analysis

**Best Performance:** 0.39 Gops/s (tiny matrices)
**Target Performance:** 20-30 Gops/s (large matrices)
**Achieved Performance:** 0.24 Gops/s (large matrices)
**Gap:** **83-125× below target**

**Average Ratio:** 306× slower than NumPy BLAS

**Status:** ❌ **CRITICAL PERFORMANCE GAP** - Urgent optimization needed

### Correctness Validation

| Test Case | Status | Max Error |
|-----------|--------|-----------|
| 8×16×20 | ✅ PASS | 9.54e-07 |
| 64×64×65 | ✅ PASS | 5.72e-06 |

All correctness tests passing - implementation is mathematically correct.

### Integration Tests

| Test | Status |
|------|--------|
| Module Availability | ✅ PASS |
| Forward Pass Correctness | ✅ PASS (max error: 0.00e+00) |
| Gradient Flow (PyTorch) | ✅ PASS |
| Performance vs PyTorch | ✅ PASS (1.17× faster on 1×20×16) |

**Note:** PyTorch integration works correctly, but underlying GEMM is slow for all backends.

---

## Critical Issues Identified

### Issue 1: GEMM Performance Gap (RESOLVED - Root Cause Identified)

**Severity:** CRITICAL
**Impact:** Makes Phase 4 (Matrix Multiplication) incomplete
**Status:** ✅ Root cause analysis complete (see `reports/reasons.md`)

**Details:**
- Current: 0.24 Gops/s on large matrices
- Target: 20-30 Gops/s
- Gap: 83-125× slower than target
- NumPy BLAS is 306× faster on average

**Root Cause (CONFIRMED via isolated component benchmarks):**
1. **Primary (56× impact):** Missing AVX2 SIMD vectorization - GEMM operates at scalar baseline
2. **Secondary (2× impact):** Missing OpenMP parallelization - single-threaded on 12-core CPU
3. **Tertiary (3× impact):** Missing cache blocking - performance degrades with matrix size

**Statistical Evidence:**
- GEMM is compute-bound, NOT memory-bound (4,311× below bandwidth limit)
- Dense243 overhead only 2.5-11% (NOT the bottleneck)
- Gap consistent (57-96×) across all sizes indicating vectorization issue

**Detailed Analysis:** See `reports/reasons.md` for comprehensive statistical analysis with:
- 5 isolated component benchmarks
- Cross-correlation with 3 data sources
- Causality isolation (7 hypotheses tested)
- Hierarchical bottleneck ranking with expected gains
- Optimization roadmap: SIMD → OpenMP → Cache blocking → 20-40 Gops/s

**Recommendation:** Systematic optimization in separate project before kernel integration

### Issue 2: Performance Regression at 1M Elements

**Severity:** MEDIUM
**Impact:** tmin/tmax/tnot drop from 13-16 Gops/s (100K) to 2.6-7 Gops/s (1M)

**Details:**
- tadd/tmul: Improve from 14 → 20 Gops/s (OpenMP effective)
- tmin/tmax/tnot: Drop from 13-16 → 2.6-7 Gops/s (OpenMP ineffective)

**Possible Causes:**
- Memory bandwidth saturation
- Cache thrashing at 1M elements (~1 MB)
- OpenMP overhead exceeding benefit for these ops

**Recommendation:** Profile with VTune to understand memory access patterns

---

## Build Artifacts

**All modules built successfully:**

1. **ternary_simd_engine.cp312-win_amd64.pyd** (162.5 KB)
   - Standard ternary operations (tadd, tmul, tmin, tmax, tnot)
   - AVX2 SIMD vectorization
   - OpenMP parallelization
   - Status: ✅ Production-ready

2. **ternary_dense243_module.cp312-win_amd64.pyd** (169 KB)
   - Dense243 encoding (5 trits/byte)
   - Pack/unpack operations
   - Status: ✅ Production-ready

3. **ternary_tritnet_gemm.cp312-win_amd64.pyd** (171.5 KB)
   - Ternary matrix multiplication
   - Naive + AVX2 implementations
   - Dense243 integration
   - Status: ⚠️ **Functionally correct, performance critical**

**Compiler:** MSVC 19.29, /O2 /GL /arch:AVX2 /std:c++17
**OpenMP:** Enabled (12 threads auto-configured)

---

## Recommendations

### Immediate Actions (Priority 1 - Before Phase 4 completion)

**1. Profile GEMM with VTune (CRITICAL)**
- Identify bottleneck (CPU-bound vs memory-bound)
- Measure cache miss rates
- Analyze SIMD utilization
- Check for serialization points

**2. Optimize Based on Profiling:**

If memory-bound:
- Implement cache blocking (tile matrices)
- Optimize memory access patterns (reduce TLB misses)
- Add prefetching for better memory pipeline

If compute-bound:
- Verify AVX2 usage in hot loops
- Ensure FMA instructions generated
- Check for auto-vectorization failures
- Unroll inner loops if needed

**3. Add OpenMP to GEMM:**
- Parallelize over M dimension (rows)
- Test with different tile sizes
- Measure scaling efficiency

**4. Benchmark-Driven Optimization:**
- Measure after each change
- Target: 5× improvement → 1-2 Gops/s (acceptable minimum)
- Stretch target: 20× improvement → 5 Gops/s (competitive with BitNet)

### Secondary Actions (Priority 2 - After Phase 4 complete)

**5. Investigate 1M Element Regression:**
- Profile tmin/tmax/tnot at 1M elements
- Understand why OpenMP doesn't help
- Consider disabling OpenMP for these ops at large sizes

**6. Document Performance Characteristics:**
- Update PHASE_4_MATRIX_MULTIPLICATION_STATUS.md with findings
- Add performance optimization guide
- Document bottlenecks and mitigation strategies

---

## Benchmark Commands Used

```bash
# Clean artifacts
python build/clean_all.py

# Build all modules
python build/build.py                  # ternary_simd_engine
python build/build_dense243.py         # ternary_dense243_module
python build/build_tritnet_gemm.py     # ternary_tritnet_gemm

# Run benchmarks
python benchmarks/bench_phase0.py --quick      # Standard operations
python benchmarks/bench_gemm.py --quick        # Matrix multiplication

# Run integration tests
python tests/python/test_tritnet_gemm_integration.py
```

---

## Files Generated

**Benchmark Results:**
- `benchmarks/results/bench_results_20251125_133226.json` - Standard ops results
- `benchmarks/results/bench_results_20251125_133226.csv` - Standard ops CSV
- `benchmarks/results/bench_gemm_results_20251125_134017.json` - GEMM results

**Documentation:**
- `docs/PHASE_3.2_DUAL_SHUFFLE_ANALYSIS.md` - Dual-shuffle optimization (complete)
- `docs/PHASE_3.3_FUSION_BASELINE.md` - Fusion baseline (complete)
- `docs/PHASE_4_MATRIX_MULTIPLICATION_STATUS.md` - GEMM status (needs update)
- `benchmarks/bench_gemm.py` - New GEMM benchmark script (created)

---

## Conclusions

### Phase 3.2 & 3.3 - ✅ SUCCESS
- Dual-shuffle optimization working (12-18% gain)
- 4-fusion baseline validated (7-35× speedup)
- Standard operations at peak performance (20.7 Gops/s)
- All 16/16 fusion tests passing

### Phase 4 - ⚠️ CRITICAL BLOCKER
- GEMM implementation exists and is correct
- **Critical performance gap:** 0.24 Gops/s vs 20-30 Gops/s target
- 83-125× below target, 306× slower than NumPy BLAS
- **BLOCKER FOR AI/ML VIABILITY**

### Next Steps
1. **VTune profiling** to identify GEMM bottleneck (highest priority)
2. Implement targeted optimizations based on profile data
3. Re-benchmark after each optimization
4. Target: Achieve at least 5 Gops/s (25× current performance)
5. Document findings and update Phase 4 status

---

**Status:** Phase 3 complete ✅, Phase 4 blocked on performance optimization ⚠️
**Updated:** 2025-11-25
**Next Action:** VTune profiling of GEMM implementation
