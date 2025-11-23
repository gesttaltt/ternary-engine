# Ternary Engine Codebase Overview

Quick reference guide for navigating the Ternary Engine codebase.

## Project Purpose

Production-grade balanced ternary arithmetic library achieving:
- **35,042 Mops/s** peak throughput (35 billion operations/second)
- **8,234× average speedup** vs pure Python
- **2-bit trit encoding** enabling 8× memory reduction vs FP16
- **TritNet innovation** replacing memory-bound LUTs with compute-bound neural networks

## Directory Structure

```
ternary-engine/
├── ternary_core/              # Production kernel (Windows x64 validated)
│   ├── algebra/               # Core ternary operations
│   │   ├── ternary_lut_gen.h      # Compile-time LUT generation (111 lines)
│   │   └── ternary_algebra.h      # Scalar ops + LUTs (143 lines)
│   ├── simd/                  # SIMD acceleration
│   │   ├── ternary_simd_kernels.h # AVX2 vectorization (103 lines)
│   │   ├── ternary_cpu_detect.h   # Runtime CPU detection (185 lines)
│   │   └── ternary_fusion.h       # Operation fusion (204 lines)
│   ├── ffi/
│   │   └── ternary_c_api.h        # C FFI layer (255 lines)
│   └── core_api.h             # Unified entry point
│
├── ternary_engine/            # Experimental features
│   └── experimental/
│       ├── dense243/          # High-density encoding (validated)
│       └── fusion/            # Operation fusion (Phase 4.0 validated)
│
├── scripts/                   # Build and development automation
│   ├── build/                 # Build scripts
│   │   ├── build.py               # Standard optimized build
│   │   ├── build_dense243.py      # Dense243 module
│   │   ├── build_pgo_unified.py   # Clang PGO
│   │   └── clean_all.py           # Cleanup utility
│   ├── tritnet/               # TritNet neural network training
│   │   ├── generate_truth_tables.py  # Dataset generation
│   │   ├── ternary_layers.py         # PyTorch ternary layers
│   │   ├── tritnet_model.py          # Model architectures
│   │   └── train_tritnet.py          # Training orchestration
│   └── orchestration/         # High-level workflows (future)
│
├── benchmarks/                # Performance benchmarking
│   ├── bench_phase0.py            # Core performance suite
│   ├── bench_competitive.py       # 6-phase competitive analysis
│   ├── bench_model_quantization.py  # Real model testing
│   └── utils/visualization.py     # Report generation
│
├── tests/                     # Test suite (65 tests)
│   ├── test_phase0.py             # Correctness (50 tests)
│   ├── test_omp.py                # OpenMP scaling (25 tests)
│   ├── test_errors.py             # Error handling
│   └── test_fusion.py             # Fusion validation
│
├── docs/                      # Documentation
│   ├── api-reference/             # API documentation
│   ├── architecture/              # Design docs
│   ├── build-system/              # Build system docs
│   └── pgo/                       # PGO guides
│
├── datasets/                  # Training datasets
│   └── tritnet/                   # Truth tables (236,439 samples, 78.33 MB)
│
├── models/                    # Trained models
│   └── tritnet/                   # TritNet models (.tritnet format)
│
├── opentimestamps/            # IP protection
│   └── timestamps/                # Blockchain timestamps (.ots files)
│
├── .claude/                   # Claude Code configuration
│   ├── CLAUDE.md                  # Project configuration
│   ├── commands/                  # Slash commands
│   ├── context/                   # Context documents
│   └── templates/                 # Code templates
│
└── Root level files
    ├── ternary_simd_engine.cpp    # Main engine (uses ternary_core/)
    ├── README.md                  # Project overview
    ├── TESTING.md                 # Testing guide
    └── CONTRIBUTING.md            # Development guidelines
```

## Key Files by Purpose

### Core Implementation (Production)
- **ternary_core/algebra/ternary_algebra.h** - Scalar operations, branch-free LUTs
- **ternary_core/simd/ternary_simd_kernels.h** - AVX2 SIMD (32 parallel operations)
- **ternary_simd_engine.cpp** - Python bindings via pybind11

### TritNet (Experimental)
- **scripts/tritnet/ternary_layers.py** - Ternary quantization layers
- **scripts/tritnet/tritnet_model.py** - Model architectures (TritNetUnary, TritNetBinary)
- **scripts/tritnet/train_tritnet.py** - Training pipeline

### Build System
- **build/build.py** - Standard optimized build
- **build/build_pgo_unified.py** - Profile-Guided Optimization (Clang)

### Testing
- **tests/test_phase0.py** - Correctness tests (50 cases)
- **benchmarks/bench_phase0.py** - Performance benchmarks

### Documentation
- **README.md** - Project overview, performance claims
- **docs/api-reference/** - Complete API documentation
- **.claude/CLAUDE.md** - Claude Code configuration

## Operations

**Supported ternary operations:**
- **tadd** - Saturated addition (clamps to [-1, +1])
- **tmul** - Standard multiplication
- **tmin** - Element-wise minimum
- **tmax** - Element-wise maximum
- **tnot** - Sign flip (0 unchanged)

## Technology Stack

**Languages:**
- C++17 (production kernel)
- Python 3.7+ (bindings, tools, training)

**Frameworks:**
- pybind11 (C++/Python integration)
- NumPy (array operations)
- PyTorch 2.0+ (TritNet training)

**Hardware:**
- x86-64 with AVX2 (Intel Haswell 2013+, AMD Excavator 2015+)
- 32-byte aligned memory for streaming stores
- Multi-core for OpenMP (when re-enabled)

## Development Status

**Production Ready (Windows x64):**
- ✅ Core algebra system
- ✅ SIMD kernels (AVX2)
- ✅ Runtime CPU detection
- ✅ Dense243 encoding
- ✅ Operation fusion Phase 4.0

**In Progress:**
- 🔧 TritNet Phase 2A (tnot 100% accuracy validation)
- 🔧 Phase 4.1 fusion (implementation complete, benchmarks pending)
- 🔧 Competitive benchmarking (2/5 criteria validated)

**Planned:**
- 📋 TritNet Phases 3-5 (C++ integration, GPU acceleration)
- 📋 Multi-platform validation (Linux/macOS)
- 📋 ARM NEON support
- 📋 OpenMP re-enablement

## Quick Commands

```bash
# Build
python build/build.py

# Test
python run_tests.py

# Benchmark
python benchmarks/bench_phase0.py

# TritNet training
python scripts/tritnet/train_tritnet.py --operation tnot

# Clean
python build/clean_all.py

# Create IP timestamp
python scripts/timestamp_snapshot.py --create
```

## Performance Highlights

**Peak throughput:** 35,042 Mops/s (tnot, 1M elements)
**Average speedup:** 8,234× vs pure Python
**Latency:** 0.029 ns/element (SIMD), 0.5 ns (scalar LUT)
**Fusion speedup:** 1.6× to 15.5× (validated Phase 4.0)

Validated 2025-11-23 on Windows x64, 12 cores

## Important Notes

- **Platform:** Only Windows x64 is production-validated
- **Arrays:** 1D arrays only (multi-dimensional planned)
- **CPU requirement:** AVX2 instruction set mandatory
- **OpenMP:** Disabled by default (root cause fixed, needs CI validation)
- **Encoding:** 0b00=-1, 0b01=0, 0b10=+1, 0b11=reserved/undefined

## Contact & Resources

**Repository:** https://github.com/gesttaltt/ternary-engine
**License:** Apache 2.0
**Author:** Jonathan Verdun
**Documentation:** docs/ directory
**Issues:** GitHub Issues
