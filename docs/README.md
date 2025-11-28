# Documentation Index

**Doc-Type:** Documentation Index · Version 2.0 · Updated 2025-11-28

Comprehensive documentation for the Ternary Engine - a production-grade balanced ternary arithmetic library with SIMD acceleration.

---

## Current Status (2025-11-28)

**Performance:** 45.3 Gops/s effective throughput (fused operations)
**Element-wise Peak:** 39.1 Gops/s (tnot @ 1M elements)
**Platform:** Windows x64 validated (65/65 tests passing)

---

## Documentation Organization

```
docs/
├── README.md                    # This index
├── FEATURES.md                  # Complete feature catalog
├── IP_AND_LICENSING_GUIDANCE.md # IP protection guidance
│
├── api-reference/               # API documentation
│   ├── source-code-overview.md  # High-level code guide (START HERE)
│   ├── ternary-core-header.md   # ternary_algebra.h docs
│   ├── ternary-core-simd.md     # SIMD implementation
│   ├── headers.md               # Header design philosophy
│   └── error-handling.md        # Exception system
│
├── architecture/                # Design & architecture
│   ├── architecture.md          # System architecture
│   ├── ARCHITECTURE_EVOLUTION.md # Historical evolution
│   ├── BRIDGE_LAYER_ARCHITECTURE.md # Bridge layer design
│   ├── CANONICAL_INDEXING_FIX.md # Critical bug fix docs
│   ├── optimization-roadmap.md  # Optimization history
│   └── optimization-complexity-rationale.md
│
├── specifications/              # Encoding specifications
│   ├── dense243_encoding_spec.md  # 5 trits/byte encoding
│   ├── triadsextet_encoding_spec.md # 6 trits/byte encoding
│   └── encoding_ecosystem_overview.md # All encodings
│
├── planning/                    # Development planning
│   ├── ROADMAP.md              # Project roadmap
│   ├── backend_api_design.md   # Backend API design
│   └── v1.2.0_implementation_plan_HISTORICAL.md
│
├── research/                    # Research & experimental
│   ├── tritnet/                # TritNet neural arithmetic
│   │   ├── TRITNET_VISION.md
│   │   ├── TRITNET_ROADMAP.md
│   │   └── TRITNET_GEMM_STATUS.md
│   └── bitnet/                 # BitNet analysis
│       ├── BITNET_ARCHITECTURE_ANALYSIS.md
│       └── BITNET_INTEGRATION_STRATEGY.md
│
├── build-system/               # Build documentation
│   ├── README.md
│   ├── setup-standard.md
│   ├── setup-pgo.md
│   └── artifact-organization.md
│
├── pgo/                        # Profile-Guided Optimization
│   ├── README.md
│   └── PGO_LIMITATIONS.md
│
├── features/                   # Feature documentation
│   ├── phase4-fusion-plan.md
│   └── optimization-leverage-report.md
│
├── historical/                 # Historical documentation
│   ├── phases/                 # Phase completion docs
│   ├── benchmarks/             # Historical benchmarks
│   ├── audits/                 # Code audits
│   └── (various dated reports)
│
└── profiling/                  # Profiling documentation
```

---

## Quick Navigation

### Getting Started

| Goal | Document |
|------|----------|
| Understand the code | [api-reference/source-code-overview.md](api-reference/source-code-overview.md) |
| See all features | [FEATURES.md](FEATURES.md) |
| Build the project | [build-system/README.md](build-system/README.md) |
| Run benchmarks | [../benchmarks/README.md](../benchmarks/README.md) |

### Architecture Deep-Dives

| Topic | Document |
|-------|----------|
| System architecture | [architecture/architecture.md](architecture/architecture.md) |
| Bridge layer (int8 ↔ ternary) | [architecture/BRIDGE_LAYER_ARCHITECTURE.md](architecture/BRIDGE_LAYER_ARCHITECTURE.md) |
| Canonical indexing fix | [architecture/CANONICAL_INDEXING_FIX.md](architecture/CANONICAL_INDEXING_FIX.md) |
| Architecture evolution | [architecture/ARCHITECTURE_EVOLUTION.md](architecture/ARCHITECTURE_EVOLUTION.md) |

### Encoding Specifications

| Encoding | Density | Document |
|----------|---------|----------|
| Dense243 (T5) | 95.3% (5 trits/byte) | [specifications/dense243_encoding_spec.md](specifications/dense243_encoding_spec.md) |
| TriadSextet | 42% (3 trits → base-27) | [specifications/triadsextet_encoding_spec.md](specifications/triadsextet_encoding_spec.md) |
| Ecosystem overview | - | [specifications/encoding_ecosystem_overview.md](specifications/encoding_ecosystem_overview.md) |

### Research Projects

| Project | Status | Document |
|---------|--------|----------|
| TritNet (Neural Arithmetic) | Phase 2A | [research/tritnet/TRITNET_VISION.md](research/tritnet/TRITNET_VISION.md) |
| BitNet Integration | Analysis | [research/bitnet/BITNET_ARCHITECTURE_ANALYSIS.md](research/bitnet/BITNET_ARCHITECTURE_ANALYSIS.md) |

### Planning & Roadmap

| Document | Description |
|----------|-------------|
| [planning/ROADMAP.md](planning/ROADMAP.md) | Project roadmap (v1.x → v3.0) |
| [planning/backend_api_design.md](planning/backend_api_design.md) | Backend API design |

---

## Core Implementation Files

The production kernel is located in `src/core/`:

| File | Purpose | Lines |
|------|---------|-------|
| `src/core/algebra/ternary_algebra.h` | LUT definitions, scalar ops | ~200 |
| `src/core/simd/simd_avx2_32trit_ops.h` | SIMD kernels (canonical indexing) | ~400 |
| `src/core/simd/fused_binary_unary_ops.h` | Fusion operations | ~200 |
| `src/engine/bindings_core_ops.cpp` | Python bindings | ~600 |

---

## Key Concepts

### Balanced Ternary Encoding

```
Values:   -1,  0, +1  (three states)
Encoding: 0b00, 0b01, 0b10  (2 bits per trit)
Invalid:  0b11 (sanitized to 0)
```

### Indexing Schemes

| Scheme | Formula | Use Case |
|--------|---------|----------|
| Traditional | `(a << 2) \| b` | Sixtet packing compatible |
| Canonical | `a*3 + b` | SIMD optimized (33% faster) |

### SIMD Architecture

- 32 trits processed per AVX2 instruction
- Dual-shuffle + ADD for canonical indexing
- Automatic scalar fallback for tail elements

---

## Performance Summary (Validated 2025-11-28)

| Metric | Value |
|--------|-------|
| Peak throughput (fused) | 45.3 Gops/s |
| Peak throughput (element-wise) | 39.1 Gops/s |
| Average speedup vs Python | 8,234x |
| Platform validated | Windows x64 |

---

## Contributing to Documentation

When updating documentation:

1. **Update this index** when adding new documents
2. **Cross-reference** between related documents
3. **Mark outdated docs** with _HISTORICAL or move to `historical/`
4. **Use consistent formatting** (Doc-Type header, version, date)

---

## Historical Documentation

Older documentation is preserved in `historical/` for reference:

- **historical/phases/** - Phase completion summaries
- **historical/benchmarks/** - Historical benchmark results
- **historical/audits/** - Code audits and technical debt catalogs
- Various dated reports from development history

---

**Last Updated:** 2025-11-28
**Maintained by:** Ternary Engine Team
