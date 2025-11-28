# Changelog

All notable changes to the Ternary Engine library are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Documentation Restructuring - 2025-11-28

**Major documentation reorganization for clarity and maintainability.**

**Reports Directory (`reports/`):**
- Created semantic directory structure: `architecture/`, `performance/`, `roadmaps/`, `research/`, `investigations/`, `process/`, `archive/`
- Moved 15+ reports to appropriate categories based on content type
- Added `reports/README.md` with index and status tracking (CURRENT, PLANNING, HISTORICAL, etc.)
- Archived outdated reports to `archive/outdated/` and `archive/sessions/`

**Docs Directory (`docs/`):**
- Created new structure: `planning/`, `research/tritnet/`, `research/bitnet/`, `historical/phases/`, `historical/audits/`, `historical/benchmarks/`, `specifications/`
- Moved 30+ documents to appropriate semantic categories
- Updated `docs/README.md` with new structure and current performance metrics (45.3 Gops/s)
- Updated `docs/FEATURES.md` with corrected file paths (simd_avx2_32trit_ops.h)

**Updated Performance Metrics:**
- Peak throughput (fused): 45.3 Gops/s (up from 37.2 Gops/s)
- Element-wise peak: 39.1 Gops/s (tnot @ 1M elements)
- Canonical indexing: 33% faster SIMD via dual-shuffle + ADD

---

### Critical Fix - 2025-11-28 - Canonical Indexing LUT Correctness

**Context:** SIMD kernel produced incorrect results for arrays ≥32 elements. Root cause: incomplete implementation of Phase 3.2 canonical indexing optimization.

**Root Cause Identified:** Two incompatible indexing schemes were mixed:
- Traditional: `(a << 2) | b` → indices 0,1,2,4,5,6,8,9,10 (gaps)
- Canonical: `a*3 + b` → indices 0,1,2,3,4,5,6,7,8 (compact)

SIMD kernel used canonical index calculation but LUTs were organized for traditional indexing, causing reads from wrong LUT positions.

**Fixed:**
- `src/core/algebra/ternary_lut_gen.h` - Added `make_canonical_binary_lut()` and `make_canonical_unary_lut()` generators
- `src/core/algebra/ternary_algebra.h` - Added canonical LUT variants (TADD_LUT_CANONICAL, etc.)
- `src/core/simd/simd_avx2_32trit_ops.h` - Updated SIMD kernels to use canonical LUTs

**Added:**
- `docs/CANONICAL_INDEXING_FIX.md` - Complete root cause analysis and resolution
- `benchmarks/bench_canonical_fix.py` - Validation benchmark with overhead breakdown
- `benchmarks/SKEPTICAL_METRICS.md` - Skeptical benchmarking framework
- `benchmarks/test_falsification.py` - Falsification test suite

**Performance Findings (Post-Fix):**
- SIMD kernel: 29.3× faster than NumPy (when data is native ternary)
- Conversion overhead: 97.6% of full pipeline time
- Full pipeline vs NumPy: NumPy wins (conversion negates kernel advantage)
- Crossover point: None found (conversion dominates at all sizes)

**Key Insight:** The value of ternary engine is NOT in beating NumPy at int8 operations. The value is in:
1. Native ternary data that never converts (29× advantage realized)
2. Memory compression (4× vs int8)
3. Sparse computation (zeros can be skipped - future work)

**Validation:**
- All correctness tests pass (was 0% for n≥32, now 100%)
- Benchmarks saved to `benchmarks/results/canonical_fix_*.json`

---

### Analysis - 2025-11-25 - GEMM Performance Root Cause Analysis

**Context:** GEMM v1.0.0 exists (from TritNet v1.0.0 based on BitNet b1.58) but is functionally complete yet unoptimized.

**Root Cause Identified:** Comprehensive statistical analysis identified missing SIMD vectorization (56× impact), OpenMP parallelization (2× impact), and cache blocking (3× impact) as primary bottlenecks. GEMM achieves 0.37 Gops/s vs 20-30 Gops/s target.

**Added:**
- `reports/reasons.md` - Comprehensive root cause analysis report
  - 5 isolated component benchmarks
  - Cross-correlation analysis of 3 data sources
  - Causality isolation (7 hypotheses tested)
  - Hierarchical bottleneck ranking with expected gains
  - Statistical evidence: GEMM is compute-bound, NOT memory-bound (4,311× below bandwidth limit)
  - Optimization roadmap: SIMD → OpenMP → Cache blocking → 20-40 Gops/s
- `benchmarks/bench_gemm_isolated.py` - Component isolation benchmark (5 test suites)
- `benchmarks/bench_gemm.py` - Full GEMM performance benchmark
- `benchmarks/results/bench_gemm_isolated_20251125_141722.json` - Isolated benchmark data
- `benchmarks/results/bench_gemm_results_20251125_134017.json` - Full benchmark data

**Updated:**
- `docs/PHASE_4_MATRIX_MULTIPLICATION_STATUS.md` - Added origin story, performance analysis, optimization roadmap
- `docs/BENCHMARK_RESULTS_2025-11-25.md` - Updated with root cause findings
- `README.md` - Updated matrix multiplication status section

**Key Findings:**
- GEMM v1.0.0 origins: Built by TritNet v1.0.0 based on BitNet b1.58 (Microsoft's production ternary model)
- Critical gap: TritNet v1.0.0 has NOT applied Ternary Engine optimization learnings
- Performance: 0.24-0.39 Gops/s actual vs 20-30 Gops/s target (56-125× below)
- Correctness: ✅ All tests passing, mathematically validated
- Root cause: Missing AVX2 SIMD (primary), OpenMP (secondary), cache blocking (tertiary)
- Dense243 overhead: Only 2.5-11% (NOT the bottleneck)

**Decision:** Do NOT merge GEMM to main Ternary Kernel yet. User creating separate project for detailed optimization exploration.

---

## [1.3.0] - 2025-11-25 - "fusion" - Operation Fusion & Dual-Shuffle Optimization

### Major Achievement: Phase 3.2 & 3.3 Complete - Structurally Sound Optimization Foundation

This release completes Phase 3.2 (dual-shuffle optimization with canonical indexing) and Phase 3.3 (operation fusion baseline), establishing a structurally sound foundation for future neural network-based fusion expansion.

### Added

**Phase 3.3: Operation Fusion Baseline**:
- 4 Binary→Unary fusion patterns (structurally complete):
  - `fused_tnot_tadd` - tnot(tadd(a, b)) - 8.61× average speedup
  - `fused_tnot_tmul` - tnot(tmul(a, b)) - 7.12× average speedup
  - `fused_tnot_tmin` - tnot(tmin(a, b)) - 8.29× average speedup
  - `fused_tnot_tmax` - tnot(tmax(a, b)) - 2.72× average speedup
- Three-path architecture: OpenMP + SIMD + scalar fallback
- Comprehensive validation: 16/16 tests passing
- `docs/PHASE_3.3_FUSION_BASELINE.md` - Complete fusion strategy documentation

**Phase 3.2: Dual-Shuffle Optimization**:
- Canonical indexing with ADD-combining already implemented and working
- Dual-shuffle approach: Two parallel shuffles + ADD (not XOR)
- Performance: 12-18% improvement over traditional shift/OR indexing
- File: `src/core/simd/ternary_canonical_index.h`
- `docs/PHASE_3.2_DUAL_SHUFFLE_ANALYSIS.md` - Comprehensive analysis document
- `tests/python/test_dual_shuffle_validation.py` - XOR decomposability validation

**Future Work Documentation**:
- Neural network-based fusion expansion strategy
- Unused bit encoding optimization (0b11 pattern for carry/sign)
- Temporal harness through hardware clock cycles
- Integration plan for continuous manifold mapping (3^9 operations)

### Performance Results

**Fusion Operations (Large Arrays - 1M elements):**

| Operation | Small Arrays | Large Arrays | Peak Speedup |
|-----------|-------------|--------------|--------------|
| fused_tnot_tadd | 1.7-2.0× | 28.8× | 32.48× |
| fused_tnot_tmul | 1.8-2.4× | 22.1× | 28.05× |
| fused_tnot_tmin | 1.8-2.3× | 26.8× | 35.34× |
| fused_tnot_tmax | 1.7-2.4× | 4.4× | 14.57× |

**Key Metrics:**
- Average fusion speedup: 7-28× (array size dependent)
- Best case: 35.34× speedup (fused_tnot_tmin @ 1M elements)
- Small arrays: 1.7-2.4× (eliminated intermediate allocation)
- Large arrays: 22-35× (memory traffic reduction + OpenMP)

**Canonical Indexing Performance:**
- 12-18% improvement over shift/OR indexing
- Two parallel shuffles execute independently
- ADD combining on different execution port (no contention)
- Works for ALL operations (tadd, tmul, tmin, tmax)

### Changed

**Optimization Strategy:**
- Established 4-fusion baseline as structurally sound foundation
- Deferred 24+ pattern combinatorial expansion to neural network approach
- Documented XOR variant as not viable with current encoding
- Added future work on temporal carry encoding using unused bits

**Documentation Updates:**
- Phase 3.2 marked complete (dual-shuffle ADD working)
- Phase 3.3 marked complete (4-fusion baseline validated)
- Added neural network integration architecture
- Documented unused bit encoding for future research

### Technical Details

**Why 4 Fusions is Structurally Sound:**
1. Complete coverage of Binary→Unary pattern
2. Proven 7-35× speedup validated
3. Production-ready with full test coverage
4. Maintainable without combinatorial explosion

**Why Manual Expansion is Infeasible:**
- Binary→Binary: 16 patterns (4×4)
- 3-op chains: 64 patterns (4×4×4)
- 4-op chains: 256 patterns (grows exponentially)
- Manual implementation: ~50 lines per pattern × N²
- Test explosion: O(N²) growth

**Future Neural Network Approach:**
- User has separate project mapping 3^9 = 19,683 ternary operations
- Continuous differentiable manifold representation
- Learned fusion generation (replaces manual implementation)
- Enables arbitrary N-operation chains
- Integration after discrete foundation complete

**Dual-Shuffle Variants:**
- ADD-combining: ✅ Working, 12-18% gain, all operations
- XOR-combining: ❌ Not viable (saturating ops break XOR decomposability)
- Future encoding: Use 0b11 for carry/sign with temporal harness

### Validation

**Phase 3.3 Fusion:**
- 16/16 correctness tests passing
- All 9 input combinations validated ({-1,0,+1} × {-1,0,+1})
- Performance benchmarks across 4 array sizes (1K, 10K, 100K, 1M)
- Statistical rigor: 50 iterations, outlier removal, median-based

**Phase 3.2 Canonical Indexing:**
- Already working in production since Phase 3.1
- Used by all binary operations via `binary_op_canonical()`
- Dual-shuffle + ADD approach validated
- XOR variant tested and documented as not viable

**Platform:**
- Windows x64 with MSVC
- AVX2 SIMD (32 parallel trits)
- OpenMP parallelization (for arrays ≥100K elements)

### Breaking Changes

None - All changes are internal optimizations. External API unchanged.

### Known Issues

- XOR-based dual-shuffle not viable with current 2-bit encoding
- Binary→Binary fusion patterns (16) deferred to neural network approach
- Unary→Binary fusion patterns (4) low priority (commutative with Binary→Unary)

### Roadmap

**Completed:**
- ✅ Phase 3.2: Dual-shuffle optimization (ADD-based canonical indexing)
- ✅ Phase 3.3: Fusion operations baseline (4 patterns validated)

**Next:**
- Phase 4: Matrix multiplication (TritNet GEMM integration)
- Phase 5: Platform expansion (Linux/macOS validation)
- Phase 6: Production hardening (CI/CD, benchmarking)
- Phase 7: Neural network fusion integration (future)

**Codename:** fusion
**Platform:** Windows x64, MSVC, AVX2
**Validation Date:** 2025-11-25

---

## [1.2.0] - 2025-11-25 - "loadaware" - Load-Aware Benchmarking & Peak Performance

### Major Achievement: 37.2 Gops/s Validated with 95% Confidence

This release introduces load-aware benchmarking methodology for reproducible results and validates peak performance of 37.2 billion operations per second.

### Added

**Load-Aware Benchmarking System**:
- `benchmarks/utils/system_load_monitor.py` - System load detection and classification
  - Monitors CPU usage (total and per-core)
  - Tracks memory utilization
  - Detects high-load processes by category (browsers, Docker, antivirus, etc.)
  - Calculates load score (0-100) with classification (LOW/MEDIUM/HIGH/VERY_HIGH)
  - Generates reproducibility recommendations
- `benchmarks/bench_with_load_context.py` - Load-aware benchmark wrapper
  - Pre/post benchmark system state capture
  - Automatic reproducibility assessment
  - Confidence rating based on system load
  - JSON results with full load context

**Documentation**:
- `docs/BENCHMARK_FINDINGS_2025-11-25.md` - Comprehensive benchmark analysis
  - Peak performance validation (37.2 Gops/s)
  - Variance analysis by array size
  - Reproducibility guidelines
  - Load factor impact analysis

### Performance Results (95% Confidence)

**Peak Throughput - Backend AVX2_v2:**

| Category | Operation | Throughput | Array Size |
|----------|-----------|------------|------------|
| **Fusion** | fused_tnot_tadd | **37,244 Mops/s** | 1M |
| | fused_tnot_tmin | **37,244 Mops/s** | 1M |
| | fused_tnot_tmax | 36,101 Mops/s | 1M |
| | fused_tnot_tmul | 32,206 Mops/s | 1M |
| **Regular** | tnot | **29,412 Mops/s** | 100K |
| | tadd | 21,482 Mops/s | 1M |
| | tmul | 21,277 Mops/s | 100K |

**Key Metrics:**
- Peak: 37.2 Gops/s (fusion @ 1M elements)
- Fusion speedup: 1.73× vs separate operations
- Average speedup: ~8,000× vs pure Python
- Reproducibility: 95% confidence with low system load

### Changed

**README.md Updates:**
- Performance badges updated to 37,244 Mops/s peak
- Speedup badge updated to ~8,000× average
- New performance table with fusion results
- Added link to benchmark findings document
- Version bumped to 1.2.0

### Technical Details

**Load Classification Thresholds:**
- LOW (0-20): 95% confidence, < 10% variance
- MEDIUM (20-40): 70-80% confidence, 10-20% variance
- HIGH (40-60): 50-60% confidence, 20-30% variance
- VERY_HIGH (60+): < 50% confidence, > 30% variance

**Monitored Process Categories:**
- Browsers (Chrome, Firefox, Edge, Opera, Brave)
- Docker/WSL (Docker Desktop, containerd, vmmemWSL)
- Development (VS Code, Visual Studio, PyCharm)
- Cloud sync (Google Drive, OneDrive, Dropbox)
- Security (Windows Defender, antivirus)
- Communication (Slack, Discord, Teams, Zoom)

### Validation

- All tests passing (5/5 test suites)
- Three benchmark runs with decreasing system load
- Final run: LOW load (score 16-19), 95% confidence
- Results saved to `benchmarks/results/load_aware/`

**Codename:** loadaware
**Platform:** Windows x64, MSVC, AVX2
**Validation Date:** 2025-11-25

---

## [1.1.0] - 2025-11-24 - "ktr" - Source Restructuring & Performance Validation

### 🎯 Major Changes: Unified src/ Structure & Comprehensive Benchmarking

This release completes the source code reorganization into a unified `src/` directory structure and provides comprehensive performance validation with realistic benchmarks.

### Added

**Source Code Restructuring**:
- Unified `src/` directory structure (from `ternary_core/` and `ternary_engine/`)
  - `src/core/` - Production kernel (algebra, SIMD, FFI, profiling)
  - `src/engine/` - Python bindings and library code
- Reduced nesting depth from 4 to 3 levels
- Eliminated fragile `../../../` includes
- Cleaner build include paths (single `src/` directory)

**Performance Validation**:
- Comprehensive benchmark suite execution
- Performance range documentation (28.6-35.0 Gops/s)
- Detailed investigation of performance characteristics
- Competitive benchmarks vs NumPy INT8

**Documentation**:
- `local-reports/2025-11-24/BENCHMARK_REPORT.md` - Complete benchmark analysis
- `local-reports/2025-11-24/PERFORMANCE_INVESTIGATION.md` - 35 Gops/s validation
- Updated README.md with performance range explanation
- Fixed benchmark paths after restructuring

### Changed

**Build System**:
- Updated all build scripts to use `src/` includes
- Fixed paths in `run_all_benchmarks.py`
- Updated module imports in benchmarks

**Version**:
- Bumped version from 1.0.0 to 1.1.0
- Updated build scripts: `build.py`, `build_pgo.py`

**Performance Metrics**:
- Updated badges: 28,585 Mops/s sustained, 35,042 Mops/s peak
- Average speedup: 6,976× vs Python
- Validation date: 2025-11-24

### Fixed

- Benchmark script paths after src/ reorganization
- Dense243 module import name (ternary_dense243_module)
- Test file imports to use correct paths

### Performance Results

**Peak Throughput (1M elements, Nov 24, 2025):**
- tnot: 28,584.90 Mops/s
- tmin: 22,814.17 Mops/s
- tmul: 21,793.47 Mops/s
- tmax: 18,455.60 Mops/s
- tadd: 13,733.83 Mops/s

**Fusion Operations:**
- Range: 1.59× - 21.65× speedup
- Average: 2.97× speedup
- All documented claims validated ✓

**Competitive vs NumPy:**
- Addition: 3.34× average speedup
- Multiplication: 7.52× average speedup
- Memory efficiency: 4× better than INT8

### Investigation Findings

Verified that 35,042 Mops/s peak (Nov 23) and 28,585 Mops/s sustained (Nov 24) are both valid:
- ✅ No optimizations lost during src/ refactoring
- ✅ All C++ code byte-for-byte identical (only include paths changed)
- ✅ Performance variance due to system load, CPU boost, thermal state
- ✅ Both measurements represent actual system performance

### Breaking Changes

None - External API unchanged, module names unchanged

### Validation

- All 3 modules built successfully (ternary_simd_engine, ternary_dense243_module, ternary_tritnet_gemm)
- All tests passing on Windows x64
- Comprehensive benchmarks completed
- Git history verified (no code lost)

**Codename:** ktr
**Platform:** Windows x64, MSVC, AVX2
**Validation Date:** 2025-11-24

## [1.0.0] - 2025-10-29 - Clean Architecture & Deployment-Ready Kernel

### 🎯 Major Milestone: Production-Ready Kernel with Clean Separation

This release establishes a clear architectural boundary between the validated kernel (`ternary_core/`) and experimental optimizations (`ternary_engine/`).

### Added

**New Architecture**:
- `ternary_core/` - Production-ready kernel directory structure
  - `ternary_core/algebra/` - Core ternary operations (ternary_algebra.h, ternary_lut_gen.h)
  - `ternary_core/simd/` - SIMD kernels (ternary_simd_kernels.h, ternary_cpu_detect.h, ternary_fusion.h)
  - `ternary_core/ffi/` - C FFI layer (ternary_c_api.h)
  - `ternary_core/core_api.h` - Unified entry point
- `ternary_engine/experimental/` - Experimental optimizations
  - `ternary_engine/experimental/dense243/` - Dense243 encoding (broken, needs redesign)
  - `ternary_engine/experimental/fusion/` - Full fusion suite (pending validation)

**Critical Fixes**:
- **Alignment validation** for streaming stores (`_mm256_stream_si256`)
  - Added `is_aligned_32()` check before using non-temporal stores
  - Prevents segfaults on unaligned NumPy arrays
- **Hardware concurrency clamping** to [1, 64]
  - Prevents crash when `std::thread::hardware_concurrency()` returns 0 (some VMs)
- **Runtime ISA dispatch** with graceful fallback
  - Module checks `has_avx2()` at initialization
  - Throws clear error on unsupported CPUs instead of illegal instruction

**Documentation**:
- `local-reports/savefile.md` - Complete kernel vs engine separation analysis
- Updated `docs/ISSUE_OPENMP_CRASHES.md` - Root cause identified and resolved
- Updated `docs/README.md` - Architecture update notice
- Updated README.md - New structure, deployment status, v1.0.0 roadmap

### Fixed

**Critical Bug Fixes (OPT-001-CRASH)**:
1. **Streaming store alignment violation** (ternary_simd_engine.cpp:294, 362)
   - Root cause of OpenMP crashes on CI runners
   - Now validates 32-byte alignment before using `_mm256_stream_si256`
   - Falls back to `_mm256_storeu_si256` if unaligned
2. **Zero hardware concurrency** (ternary_simd_engine.cpp:102)
   - `std::thread::hardware_concurrency()` can return 0
   - Multiplying threshold by 0 forced all arrays into OpenMP path
   - Now clamped to [1, 64] for safe operation
3. **Missing ISA dispatch** (ternary_simd_engine.cpp:434)
   - Module hard-coded AVX2 with no runtime detection
   - Now checks CPU capabilities at module init
   - Graceful error message on unsupported hardware

### Changed

**Architecture Reorganization**:
- **Removed duplicates** - Deleted 10 root-level files now in ternary_core/ternary_engine/
  - ternary_algebra.h, ternary_lut_gen.h, ternary_simd_kernels.h, etc.
- **Updated include paths** - All source files use new ternary_core/ paths
- **Updated build scripts** - build.py and build_fusion.py include new directories
- **Main engine** (ternary_simd_engine.cpp) references ternary_core/ hierarchy

**Deployment Status**:
- ✅ Production-ready: ternary_core/ (validated, 100% test coverage)
- ⚠️ Experimental: ternary_engine/ (pending validation)

### Performance

**No Regressions**:
- Build: 154.5 KB module (same as before)
- Tests: 60/60 Phase 0 tests pass
- Speedup: 1.5-1.8× fusion PoC validated

### Breaking Changes

**None** - This is a pure refactoring with bug fixes. All APIs remain compatible.

### Migration Guide

**For users:** No changes required - module API is identical

**For developers:**
- Include paths changed: `#include "ternary_core/algebra/ternary_algebra.h"`
- Root-level headers removed (now in ternary_core/)
- Experimental code isolated in ternary_engine/

### Known Issues

- Dense243 encoding broken (needs redesign)
- OpenMP tests disabled pending CI validation
- Full fusion suite (Phase 4.1) pending benchmarks

### Commits

- `28df626` - Architectural clarity report (savefile.md)
- `eee9179` - Critical fixes (alignment + ISA dispatch)
- `58730fe` - Architectural restructuring (ternary_core/ternary_engine/)
- `c35589e` - Cleanup of duplicate files

---

## [0.3.0] - 2025-10-13 - Phase 3: Production Refinements

### Added

**New Infrastructure Files**:
- `ternary_cpu_detect.h` (206 lines) - Runtime CPU feature detection (x86-64, ARM64)
- `ternary_c_api.h` (253 lines) - Cross-language C FFI layer (Rust/Zig/C#/Go integration)
- `ternary_profiler.h` (253 lines) - Optional profiler annotations (VTune ITT, NVIDIA NVTX)
- `benchmarks/bench_kernels.cpp` (264 lines) - Pure C++ microbenchmarks (no Python overhead)
- `avx512-future-support/ternary_simd_config.h` (260 lines) - Multi-ISA abstraction layer (future use)
- `avx512-future-support/README.md` - Future AVX-512/ARM NEON integration guide

**Documentation**:
- `docs/headers.md` - C++ header design best practices
- `local-reports/phase3-implementation-summary.md` - Comprehensive Phase 3 summary
- Updated `docs/optimization-roadmap.md` with Phase 3 plans

### Changed

**Core Engine Optimizations** (`ternary_simd_engine.cpp`):
- **OPT-PHASE3-01**: Adaptive OMP threshold - Dynamic threshold based on `std::thread::hardware_concurrency()` (5-10% gain on multi-core)
- **OPT-PHASE3-03**: Prefetch distance tuning - Configurable `PREFETCH_DIST = 512` bytes (2-5% throughput improvement)
- **OPT-PHASE3-04**: Optional compile-time sanitization - `TERNARY_NO_SANITIZE` macro support (3-5% gain in validated pipelines)

### Performance Impact

| Optimization | Expected Gain | Conditions |
|-------------|--------------|------------|
| Adaptive OMP threshold | 5-10% | Multi-core systems (8+ cores) |
| Prefetch distance tuning | 2-5% | Memory-bound workloads |
| Sanitization switch | 3-5% | Validated data pipelines |
| AVX-512 support (future) | 2× | AVX-512BW capable CPUs |
| **Total (additive)** | **10-20%** | Optimal conditions |

### Fixed
- Clarified future AVX-512 infrastructure with explicit "FUTURE USE" comments
- Isolated unused abstraction layer to separate directory

### Infrastructure
- Enhanced cross-language ecosystem (Rust, Zig, C#, Go via C API)
- Profiling framework infrastructure (VTune ITT, NVTX) - roadmap feature
- Multi-platform SIMD readiness (ARM NEON, future ARM SVE)
- CI-friendly feature detection and testing

### Backward Compatibility
✅ **100% backward compatible** - All optimizations opt-in via compile flags or runtime detection

**Commits**:
- `675893d` - Isolate unused AVX-512 abstraction layer to separate directory
- `d3c4ef1` - Add clarifying comments for abstraction layer usage
- `271778a` - Implement Phase 3 optimizations #2, #10 (SIMD config, profiler)
- `2eafb5e` - Implement Phase 3 optimizations #5, #6, #7 (CPU detect, C API, benchmarks)
- `42e5213` - Implement Phase 3 optimizations #1, #3, #4 (Adaptive OMP, prefetch, sanitization)
- `3d4e30a` - Update optimization roadmap with Phase 3 plans
- `50fd9e3` - Update documentation: Add Layer 0 and header design principles

---

## [0.2.0] - 2025-10-12 - Phase 2: Complexity Compression

### Added
- `ternary_errors.h` (119 lines) - Centralized domain-specific exception handling
- `docs/error-handling.md` - Error handling documentation
- `docs/headers.md` - Header design philosophy (YAGNI principle)
- Comprehensive source code documentation in `docs/`

### Changed

**Core Architecture Simplification**:
- **Template-based unification**: Single `process_binary_array<Sanitize>()` template replaces multiple paths
- **OPT-HASWELL-02**: Template-based optional masking for input sanitization
- Eliminated aligned/unaligned branching (modern CPUs: negligible difference)
- Removed manual loop unrolling (trust compiler auto-optimization)
- Centralized error handling via `ternary_errors.h`

**Code Reduction**:
- Collapsed 6 execution paths to 3 clean paths
- 73% code reduction (from ~1200 to ~330 lines in main engine)
- <5% performance loss for massive maintainability gain

**Documentation**:
- Added `docs/source-code-overview.md` - High-level code guide
- Added `docs/ternary-core-header.md` - `ternary_algebra.h` detailed docs
- Added `docs/ternary-core-simd.md` - `ternary_simd_engine.cpp` guide
- Added `docs/optimization-complexity-rationale.md` - Phase 2 design decisions

### Philosophy
**Phase Coherence**: Only add complexity if it provides >10% performance gain

### Performance
- **Maintained**: 100x faster than pure Python
- **Trade-off**: <5% loss for 73% code reduction
- **Verdict**: Production-ready with optimal complexity/performance balance

**Commits**:
- Multiple commits refactoring engine architecture
- Documentation updates and creation

---

## [0.1.0] - 2025-10-11 - Phase 1: Multi-Path Optimization

### Added
- OpenMP parallelization for large arrays (n ≥ 100K)
- Aligned load optimization for cache-aligned arrays
- Manual loop unrolling (4x unroll factor)
- Profile-Guided Optimization (PGO) build scripts

### Performance
- 65x speedup on large arrays (multi-core systems)
- Optimal cache utilization through alignment

### Issues
- High complexity (6+ execution paths)
- Code duplication across operation types
- Difficult to maintain and extend

---

## [0.0.5] - 2025-10-10 - Phase 0.5: SIMD LUT Shuffles

### Added
- SIMD implementation using `_mm256_shuffle_epi8` for parallel LUT lookups
- Unified semantic domain (no conversions between scalar and SIMD)
- AVX2 vectorization (process 32 trits per operation)

### Changed
- Replaced arithmetic SIMD with LUT-based SIMD
- Maintained 2-bit trit encoding throughout pipeline

### Performance
- 1.34x to 2.87x speedup vs arithmetic SIMD
- 10-100x speedup vs pure Python (depending on array size)

---

## [0.0.1] - 2025-10-09 - Phase 0: LUT Optimization

### Added
- **OPT-AUTO-LUT**: Constexpr compile-time LUT generation (`ternary_lut_gen.h`)
  - `make_binary_lut()` - Template for 16-entry binary operation LUTs
  - `make_unary_lut()` - Template for 4-entry unary operation LUTs
  - Algorithm-as-documentation approach
  - Single source of truth for mathematical rules
- Lookup table (LUT) based operations for all ternary operations
- Branch-free scalar operations
- 2-bit trit encoding (0b00 = -1, 0b01 = 0, 0b10 = +1)

### Changed
- Replaced conversion-based operations with direct LUT lookups
- Eliminated branches from hot paths

### Performance
- 3-10x theoretical speedup (1.07x measured vs optimized baseline)
- Sub-nanosecond operation latency (single L1 cache access)

### Philosophy
- **Algorithm-as-documentation**: LUTs generated from high-level logic
- **Single source of truth**: Mathematical rules defined once
- **Zero runtime cost**: Everything computed at compile time
- **Infinite maintainability ROI**: Changes to logic automatically propagate

---

## [0.0.0] - 2025-10-08 - Initial Release

### Added
- Basic ternary logic operations (tadd, tmul, tmin, tmax, tnot)
- Python bindings via pybind11
- NumPy array interface
- Conversion-based implementation (trit ↔ integer conversions)

### Operations
- `tadd(a, b)` - Saturated ternary addition
- `tmul(a, b)` - Ternary multiplication
- `tmin(a, b)` - Element-wise minimum
- `tmax(a, b)` - Element-wise maximum
- `tnot(a)` - Ternary negation

### Performance
- 10x faster than pure Python
- Bottleneck: Conversion overhead and branches

---

## Version Naming Scheme

- **0.x.0** - Major phases (Phase 0, 1, 2, 3, etc.)
- **0.x.y** - Minor updates and bug fixes within a phase
- **1.0.0** - First production-ready release (post-Phase 3)

## Unreleased Features (Future Roadmap)

### Phase 4: Specialization (Planned)
- Kernel fusion (fused multiply-add, chained operations)
- Domain-specific kernels (fractal iteration, modulo-3 arithmetic)
- GPU acceleration (CUDA implementation using NVTX profiling)

### Multi-Platform (Planned)
- ARM NEON support (128-bit vectors, 16 trits/op)
- ARM SVE support (scalable vector extension)
- RISC-V Vector extension
- WebAssembly SIMD (WASM SIMD128)

### Integration (Planned)
- Rust bindings via C API
- Zig bindings via C API
- Julia bindings
- Go bindings via cgo

---

## Performance History

| Version | Implementation | Throughput (10M elements) | Speedup vs 0.0.0 |
|---------|---------------|---------------------------|------------------|
| 0.0.0 | Conversion-based | ~1,000 ME/s | 1x |
| 0.0.1 | LUT scalar | ~2,000 ME/s | 2x |
| 0.0.5 | SIMD LUT | ~5,000 ME/s | 5x |
| 0.1.0 | OpenMP + optimizations | ~10,000 ME/s | 10x |
| 0.2.0 | Complexity compression | ~9,500 ME/s | 9.5x |
| **0.3.0** | **Production refinements** | **~10,500 ME/s** | **10.5x** |

*(ME/s = Million Elements per second)*

---

**Maintained by**: Jonathan Verdun (Ternary Engine Project)
**License**: Apache 2.0
**Last Updated**: 2025-11-28
