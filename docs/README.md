# Documentation Index

This directory contains comprehensive documentation for the ternary-kernel-python-c library.

## Documentation Organization

Documentation is organized into logical categories:

```
docs/
├── README.md (this file)           # Documentation index
│
├── api-reference/                  # API and source code documentation
│   ├── source-code-overview.md    # High-level code guide (START HERE)
│   ├── ternary-core-header.md     # ternary_algebra.h detailed docs
│   ├── ternary-core-simd.md       # ternary_simd_engine.cpp guide
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

- **[API Reference: ternary_algebra.h](./api-reference/ternary-core-header.md)** - Core algebra header (143 lines)
  - Trit encoding scheme (2-bit balanced ternary)
  - Constexpr-generated lookup tables (OPT-AUTO-LUT)
  - Scalar operations with force-inlining
  - Type definitions and utilities
  - Uses `ternary_lut_gen.h` for compile-time LUT generation

- **[API Reference: ternary_simd_engine.cpp](./api-reference/ternary-core-simd.md)** - SIMD implementation (333 lines)
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
2. [ternary_algebra.h Documentation](./api-reference/ternary-core-header.md)
3. [ternary_simd_engine.cpp Documentation](./api-reference/ternary-core-simd.md)
4. [Error Handling Documentation](./api-reference/error-handling.md)

**Add a new ternary operation:**
1. Read [CONTRIBUTING.md § Adding New Operations](../CONTRIBUTING.md#adding-new-operations)
2. Read [ternary_algebra.h § Adding Operations](./api-reference/ternary-core-header.md#future-considerations)
3. Use `make_binary_lut()` or `make_unary_lut()` from `ternary_lut_gen.h`

**Build the project:**
1. Quick: `python build/scripts/setup.py`
2. Detailed: [Build System Overview](./build-system/README.md)
3. PGO: [PGO_README.md](./PGO_README.md)

**Optimize performance:**
1. Read [ternary_simd_engine.cpp § Performance Analysis](./api-reference/ternary-core-simd.md#performance-analysis)
2. Read [PGO_README.md](./PGO_README.md)
3. Profile with `python benchmarks/bench_phase0.py`
4. See [Optimization Roadmap](./architecture/optimization-roadmap.md)

**Understand design decisions:**
1. [Optimization Complexity Rationale](./architecture/optimization-complexity-rationale.md)
2. [Source Code Overview § Key Design Principles](./api-reference/source-code-overview.md#key-design-principles)

**Port to a new architecture (ARM/NEON):**
1. Read [ternary_simd_engine.cpp § Platform-Specific Notes](./api-reference/ternary-core-simd.md#platform-specific-notes)
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

See [ternary-core-header.md § Encoding Scheme](./ternary-core-header.md#encoding-scheme)

### LUT-Based Operations

All operations use lookup tables instead of arithmetic:
```c
return TADD_LUT[(a << 2) | b];  // Faster than arithmetic
```

See [ternary-core-header.md § Lookup Tables](./ternary-core-header.md#lookup-tables-luts)

### SIMD Acceleration

Process 32 trits in parallel using AVX2:
```cpp
__m256i result = _mm256_shuffle_epi8(lut, indices);  // 32 parallel lookups
```

See [ternary-core-simd.md § SIMD Operations](./ternary-core-simd.md#simd-operations)

### Three Execution Paths

1. **OpenMP Parallel** (n ≥ 100K): Multi-threaded SIMD
2. **Serial SIMD** (32 ≤ n < 100K): Single-threaded SIMD
3. **Scalar Tail** (n < 32): Fallback to scalar LUTs

See [ternary-core-simd.md § Execution Paths](./ternary-core-simd.md#execution-paths-detailed)

## Performance Summary

| Implementation | Throughput | Speedup vs Python |
|----------------|------------|-------------------|
| Python         | 100 ME/s   | 1x                |
| C++ naive      | 333 ME/s   | 3x                |
| C++ LUT        | 2,000 ME/s | 20x               |
| **C++ SIMD**   | **10,000 ME/s** | **100x**      |

*(ME/s = Million Elements per second)*

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

**Last Updated**: 2025-10-13
**Maintained by**: Ternary Core Contributors
