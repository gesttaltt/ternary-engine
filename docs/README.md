# Documentation Index

This directory contains comprehensive documentation for the ternary-kernel-python-c library.

## Source Code Documentation (NEW)

**Start here to understand the core implementation:**

- **[Source Code Overview](./source-code-overview.md)** - High-level guide to the pure source code
  - Architecture layers
  - Operation flow examples
  - Performance summary
  - Reading guide for different purposes

- **[ternary_core.h Documentation](./ternary-core-header.md)** - Core header file (125 lines)
  - Trit encoding scheme
  - Lookup table (LUT) design
  - Scalar operations
  - Type definitions and utilities

- **[ternary_core_simd_full.cpp Documentation](./ternary-core-simd.md)** - SIMD implementation (297 lines)
  - AVX2 acceleration techniques
  - Template-based unified processing
  - Execution path selection
  - OpenMP parallelization
  - Python integration via pybind11

## Additional Documentation

### Build and Optimization

- **[general-readme.md](./general-readme.md)** - General project information
- **[PGO_README.md](./PGO_README.md)** - Profile-Guided Optimization guide
- **[architecture.md](./architecture.md)** - Overall architecture overview

### Design Rationale

- **[optimization-complexity-rationale.md](./optimization-complexity-rationale.md)** - Phase 2 design decisions
  - Why certain optimizations were removed
  - Phase coherence philosophy
  - Complexity vs performance tradeoffs

- **[optimization-roadmap.md](./optimization-roadmap.md)** - Historical optimization journey
  - Evolution from Phase 0 to Phase 2
  - Lessons learned

## Quick Navigation

### I want to...

**Understand how the code works:**
1. [Source Code Overview](./source-code-overview.md)
2. [ternary_core.h Documentation](./ternary-core-header.md)
3. [ternary_core_simd_full.cpp Documentation](./ternary-core-simd.md)

**Add a new ternary operation:**
1. Read [ternary_core.h § Adding Operations](./ternary-core-header.md#future-considerations)
2. Read [ternary_core_simd_full.cpp § Operation Wrappers](./ternary-core-simd.md#operation-wrappers)

**Optimize performance:**
1. Read [ternary_core_simd.md § Performance Analysis](./ternary-core-simd.md#performance-analysis)
2. Read [PGO_README.md](./PGO_README.md)
3. Profile with `python benchmarks/bench_phase0.py`

**Understand design decisions:**
1. [optimization-complexity-rationale.md](./optimization-complexity-rationale.md)
2. [Source Code Overview § Key Design Principles](./source-code-overview.md#key-design-principles)

**Build the project:**
1. See [PGO_README.md § Building](./PGO_README.md) for standard build
2. Or run: `python build/scripts/setup.py build_ext --inplace`

**Port to a new architecture (ARM/NEON):**
1. Read [ternary_core_simd.md § Platform-Specific Notes](./ternary-core-simd.md#platform-specific-notes)
2. Keep `ternary_core.h` unchanged (portable)
3. Reimplement SIMD layer with NEON intrinsics

## File Organization

```
docs/
├── README.md (this file)
│
├── Source Code Documentation
│   ├── source-code-overview.md         ← START HERE
│   ├── ternary-core-header.md          ← ternary_core.h
│   └── ternary-core-simd.md            ← ternary_core_simd_full.cpp
│
├── Architecture & Design
│   ├── architecture.md
│   ├── optimization-complexity-rationale.md
│   └── optimization-roadmap.md
│
└── Build & Performance
    ├── general-readme.md
    └── PGO_README.md
```

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

**Last Updated**: 2025-10-12 (Phase 2 completion)
**Primary Authors**: Ternary Core Contributors
