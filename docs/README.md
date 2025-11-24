# Documentation Index

This directory contains comprehensive documentation for the ternary-engine library.

## 🆕 Architecture Update & Code Consolidation (v1.0.0 - 2025-11-22)

**The project structure has been reorganized, consolidated, and fully validated:**

✅ **`src/core/`** - Production-ready kernel (65/65 tests passing, 7,315× avg speedup)
- Core algebra and LUT generation (validated)
- SIMD kernels and CPU detection (validated)
- C FFI layer (cross-language ready)
- Operation fusion (validated: 1.6-15.5× speedup) - **Now integrated in main module**

✅ **`src/engine/`** - Main SIMD engine with fusion operations
- Unified module with all core operations
- Fusion operations available: `fused_tnot_tadd`, `fused_tnot_tmul`, `fused_tnot_tmin`, `fused_tnot_tmax`
- Build via `build/build.py`

⚠️ **Deprecated/Archived:**
- **ternary_fusion_engine** - Merged into main `ternary_simd_engine` module (see MIGRATION_NOTES.md)
- **dense243/** - Archived to `legacy/dense243_broken/` (documented as broken)
- **ternary_profiler.h** - Deleted (never integrated, pure overhead)

**Note:** Implementation is now in:
- `src/core/algebra/` - Core algebra and LUT generation
- `src/core/simd/` - SIMD kernels, CPU detection, and fusion operations
- `src/core/common/` - Common utilities and error handling
- `src/engine/` - Main SIMD engine with Python bindings
- `build/` - Build scripts
- `tests/` - Test suite
- `legacy/` - Archived code (broken or deprecated features)

See MIGRATION_NOTES.md for migration guide from ternary_fusion_engine to unified module.

## Documentation Organization

Documentation is organized into logical categories:

```
docs/
├── README.md (this file)           # Documentation index
│
├── api-reference/                  # API and source code documentation
│   ├── source-code-overview.md    # High-level code guide (START HERE)
│   ├── ternary-engine-header.md     # ternary_algebra.h detailed docs
│   ├── ternary-engine-simd.md       # ternary_simd_engine.cpp guide
│   ├── headers.md                  # Header design philosophy
│   └── error-handling.md           # Exception handling system
│
├── architecture/                   # Design and architecture
│   ├── architecture.md             # System architecture overview
│   ├── optimization-roadmap.md     # Historical optimization evolution
│   └── optimization-complexity-rationale.md  # Phase 2 design decisions
│
├── build-system/                   # Build system documentation
│   ├── README.md                   # Build system overview
│   ├── artifact-organization.md    # Artifact management
│   ├── setup-standard.md           # Standard build details
│   ├── setup-pgo.md                # PGO build details
│   └── setup-reference.md          # Reference build details
│
├── historical/                     # Historical documentation
│   └── BASELINE-PRE-AUTOLUT.md     # Pre-LUT optimization baseline
│
├── PGO_README.md                   # Profile-Guided Optimization guide
└── general-readme.md               # General project information
```

## API Reference

**Start here to understand the core implementation:**

- **[API Reference: Source Code Overview](./api-reference/source-code-overview.md)** - High-level guide
  - Architecture layers (4 core files)
  - Operation flow examples
  - Performance summary
  - Reading guide for different purposes

- **[API Reference: ternary_algebra.h](./api-reference/ternary-engine-header.md)** - Core algebra header (143 lines)
  - Trit encoding scheme (2-bit balanced ternary)
  - Constexpr-generated lookup tables (OPT-AUTO-LUT)
  - Scalar operations with force-inlining
  - Type definitions and utilities
  - Uses `ternary_lut_gen.h` for compile-time LUT generation

- **[API Reference: ternary_simd_engine.cpp](./api-reference/ternary-engine-simd.md)** - SIMD implementation (333 lines)
  - AVX2 acceleration techniques
  - Template-based unified processing with optional masking (OPT-HASWELL-02)
  - Execution path selection (OpenMP, SIMD, scalar)
  - OpenMP parallelization (OPT-PHASE3-01)
  - Python integration via pybind11
  - Centralized error handling via `ternary_errors.h`

- **[API Reference: Error Handling](./api-reference/error-handling.md)** - Exception handling system
  - Domain-specific exception types
  - YAGNI principle (minimal exception set)
  - Python exception mapping
  - Usage examples for C++ and Python

- **[API Reference: Headers Design](./api-reference/headers.md)** - Header design philosophy
  - When to use headers vs .cpp files
  - YAGNI principle application
  - Header-only vs compiled approach

## Architecture & Design

- **[Architecture Overview](./architecture/architecture.md)** - Overall system architecture
  - Layer-by-layer breakdown
  - Data flow and execution paths
  - Performance characteristics

- **[Optimization Roadmap](./architecture/optimization-roadmap.md)** - Historical optimization evolution
  - Evolution from Phase 0 to Phase 3
  - Lessons learned at each phase
  - Future roadmap (Phase 4+)

- **[Optimization Complexity Rationale](./architecture/optimization-complexity-rationale.md)** - Phase 2 design decisions
  - Why certain optimizations were removed
  - Code simplification philosophy
  - Complexity vs performance tradeoffs

## Build System

- **[Build System Overview](./build-system/README.md)** - Complete build documentation
  - Build scripts and usage
  - Compiler flags and options
  - Artifact management
  - Troubleshooting

- **[Artifact Organization](./build-system/artifact-organization.md)** - Build artifact management
- **[Standard Build](./build-system/setup-standard.md)** - Standard optimized build
- **[PGO Build](./build-system/setup-pgo.md)** - Profile-Guided Optimization
- **[Reference Build](./build-system/setup-reference.md)** - Baseline reference build

## Quick Start Guides

### I want to...

**Understand how the code works:**
1. [Source Code Overview](./api-reference/source-code-overview.md) ← START HERE
2. [ternary_algebra.h Documentation](./api-reference/ternary-engine-header.md)
3. [ternary_simd_engine.cpp Documentation](./api-reference/ternary-engine-simd.md)
4. [Error Handling Documentation](./api-reference/error-handling.md)

**Add a new ternary operation:**
1. Read [CONTRIBUTING.md § Adding New Operations](../CONTRIBUTING.md#adding-new-operations)
2. Read [ternary_algebra.h § Adding Operations](./api-reference/ternary-engine-header.md#future-considerations)
3. Use `make_binary_lut()` or `make_unary_lut()` from `ternary_lut_gen.h`

**Build the project:**
1. Quick: `python build/build.py`
2. Detailed: [Build System Overview](./build-system/README.md)
3. PGO: [PGO_README.md](./PGO_README.md)

**Optimize performance:**
1. Read [ternary_simd_engine.cpp § Performance Analysis](./api-reference/ternary-engine-simd.md#performance-analysis)
2. Read [PGO_README.md](./PGO_README.md)
3. Profile with `python benchmarks/bench_phase0.py`
4. See [Optimization Roadmap](./architecture/optimization-roadmap.md)

**Understand design decisions:**
1. [Optimization Complexity Rationale](./architecture/optimization-complexity-rationale.md)
2. [Source Code Overview § Key Design Principles](./api-reference/source-code-overview.md#key-design-principles)

**Port to a new architecture (ARM/NEON):**
1. Read [ternary_simd_engine.cpp § Platform-Specific Notes](./api-reference/ternary-engine-simd.md#platform-specific-notes)
2. Keep `ternary_algebra.h` and `ternary_lut_gen.h` unchanged (portable)
3. Reimplement SIMD layer with NEON intrinsics in `ternary_simd_engine.cpp`

**Contribute to the project:**
1. Read [CONTRIBUTING.md](../CONTRIBUTING.md)
2. Review [CHANGELOG.md](../CHANGELOG.md) for recent changes
3. Check [Optimization Roadmap](./architecture/optimization-roadmap.md) for future plans

## Core Concepts

### Balanced Ternary Encoding

Values: **-1, 0, +1** (three states)
Encoding: **0b00, 0b01, 0b10** (2 bits per trit)

See [ternary-engine-header.md § Encoding Scheme](./ternary-engine-header.md#encoding-scheme)

### LUT-Based Operations

All operations use lookup tables instead of arithmetic:
```c
return TADD_LUT[(a << 2) | b];  // Faster than arithmetic
```

See [ternary-engine-header.md § Lookup Tables](./ternary-engine-header.md#lookup-tables-luts)

### SIMD Acceleration

Process 32 trits in parallel using AVX2:
```cpp
__m256i result = _mm256_shuffle_epi8(lut, indices);  // 32 parallel lookups
```

See [ternary-engine-simd.md § SIMD Operations](./ternary-engine-simd.md#simd-operations)

### Three Execution Paths

1. **OpenMP Parallel** (n ≥ 100K): Multi-threaded SIMD
2. **Serial SIMD** (32 ≤ n < 100K): Single-threaded SIMD
3. **Scalar Tail** (n < 32): Fallback to scalar LUTs

See [ternary-engine-simd.md § Execution Paths](./ternary-engine-simd.md#execution-paths-detailed)

## Performance Summary (Validated 2025-10-29)

**Peak Throughput:**
- **tadd**: 13,047 Mops/s - 7,316× vs Python
- **tmul**: 14,058 Mops/s - 7,584× vs Python
- **tmin**: 13,447 Mops/s - 8,681× vs Python
- **tmax**: 13,341 Mops/s - 8,127× vs Python
- **tnot**: 18,518 Mops/s - 4,767× vs Python

**Average Speedup: 7,315×** (validated with statistical rigor)

**Operation Fusion (Phase 4.0):**
- fused_tnot_tadd: 1.6-15.5× speedup (validated)
- Conservative estimate: 1.94× minimum

*(Mops/s = Million operations per second)*

See [source-code-overview.md § Performance Summary](./source-code-overview.md#performance-summary)

## Contributing

When updating documentation:

1. **Keep consistency**: Update all related docs when changing implementation
2. **Add examples**: Code examples help clarify complex concepts
3. **Cross-reference**: Link between related sections
4. **Update this index**: Add new docs to the navigation sections above

## License

All documentation is released under Apache 2.0 (same as source code).

---

## Recent Updates

**2025-11-22 - Code Consolidation:**
- Merged fusion operations into main `ternary_simd_engine` module
- Archived broken dense243 implementation to `legacy/`
- Removed unused profiler infrastructure
- Reorganized build scripts to `build/`
- Updated all documentation paths

See `MIGRATION_NOTES.md` and `legacy/README.md` for migration details.

---

**Last Updated**: 2025-11-22 (code consolidation and documentation update)
**Maintained by**: Jonathan Verdun (Ternary Engine Project)
