# Changelog

All notable changes to the Ternary Engine library are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
**Last Updated**: 2025-10-13
