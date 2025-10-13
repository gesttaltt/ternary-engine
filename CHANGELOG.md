# Changelog

All notable changes to the Ternary Core SIMD library are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive project documentation (CONTRIBUTING.md, CHANGELOG.md)
- README files for build/, tests/, and all major directories
- Reorganized docs/ into categorized subdirectories

### Changed
- Moved legacy build artifacts to local-reports/legacy-artifacts/
- Reorganized documentation into api-reference/, architecture/, build-system/, historical/

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
- Production-grade profiling workflows (VTune, NVTX)
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

**Maintained by**: Ternary Core Contributors
**License**: Apache 2.0
**Last Updated**: 2025-10-13
