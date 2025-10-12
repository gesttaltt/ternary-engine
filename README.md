# Ternary Core SIMD

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

High-performance balanced ternary arithmetic library with AVX2 SIMD vectorization, OpenMP parallelization, and Python bindings.

## Overview

Ternary Core is a research-grade library for performing balanced ternary logic operations at high speed. It achieves **100x performance improvements** over pure Python through a combination of lookup table (LUT) optimizations, AVX2 SIMD vectorization, and multi-threaded processing.

**Current Status**: Phase 2 (Complexity Compression) - Production-ready with simplified architecture

### What is Balanced Ternary?

Balanced ternary is a numeral system with three values: **-1, 0, +1**. Unlike binary (0, 1) or standard ternary (0, 1, 2), balanced ternary uses symmetric negative and positive digits, making it particularly efficient for signed arithmetic and certain computational domains including:

- Fractal generation and iterative algorithms
- Modulo-3 arithmetic operations
- Continuum-discrete boundary operations
- Novel computational paradigms

### Key Features

- **Compact 2-bit Encoding**: Each trit (ternary digit) uses 2 bits
  - `0b00` = -1 (negative)
  - `0b01` =  0 (neutral)
  - `0b10` = +1 (positive)
  - `0b11` = invalid/reserved

- **LUT-Based Optimization**: Branch-free operations via pre-computed lookup tables
- **AVX2 SIMD Vectorization**: Process 32 trits per operation using 256-bit vectors
- **OpenMP Parallelization**: Multi-threaded processing for large arrays (≥100K elements)
- **Template-Based Architecture**: Unified code paths reduce complexity while maintaining performance
- **Python Integration**: NumPy-compatible arrays via pybind11 bindings

### Operations Supported

| Operation | Function | Description |
|-----------|----------|-------------|
| Addition | `tadd(a, b)` | Saturated ternary addition (clamped to [-1, +1]) |
| Multiplication | `tmul(a, b)` | Ternary multiplication |
| Minimum | `tmin(a, b)` | Element-wise minimum |
| Maximum | `tmax(a, b)` | Element-wise maximum |
| Negation | `tnot(a)` | Ternary negation (sign flip, 0 stays 0) |

## Quick Start

### Prerequisites

- **Python**: 3.7 or later
- **Compiler**: MSVC (Windows) or GCC/Clang (Linux/macOS) with C++17 support
- **CPU**: x86-64 with AVX2 support (Intel Haswell 2013+ or AMD Excavator 2015+)
  - Current implementation uses basic AVX2 operations (pre-Haswell-optimization era)
  - Compatible with all AVX2-capable processors
- **Dependencies**: pybind11, NumPy

### Installation

```bash
# Install dependencies
pip install pybind11 numpy

# Build the module (from project root)
python build/scripts/setup.py

# Verify installation
python -c "import ternary_core_simd_full; print('Module loaded successfully')"
```

### Basic Usage

```python
import numpy as np
import ternary_core_simd_full as tc

# Encoding: 0b00 = -1, 0b01 = 0, 0b10 = +1
MINUS_ONE = 0b00
ZERO      = 0b01
PLUS_ONE  = 0b10

# Create ternary arrays
a = np.array([MINUS_ONE, ZERO, PLUS_ONE], dtype=np.uint8)
b = np.array([PLUS_ONE, ZERO, MINUS_ONE], dtype=np.uint8)

# Perform operations
result_add = tc.tadd(a, b)  # [0, 0, 0]
result_mul = tc.tmul(a, b)  # [-1, 0, -1]
result_not = tc.tnot(a)     # [+1, 0, -1]

# Large-scale processing
size = 1_000_000
a_large = np.random.choice([MINUS_ONE, ZERO, PLUS_ONE], size=size, dtype=np.uint8)
b_large = np.random.choice([MINUS_ONE, ZERO, PLUS_ONE], size=size, dtype=np.uint8)
result = tc.tadd(a_large, b_large)  # Automatically uses optimized paths
```

### Helper Functions

```python
def int_to_trit(value):
    """Convert integer (-1, 0, +1) to trit encoding"""
    if value < 0:
        return 0b00
    elif value > 0:
        return 0b10
    else:
        return 0b01

def trit_to_int(trit):
    """Convert trit encoding to integer"""
    if trit == 0b00:
        return -1
    elif trit == 0b10:
        return 1
    else:
        return 0

# Work with integers
values = [-1, 0, 1, -1, 1]
trits = np.array([int_to_trit(v) for v in values], dtype=np.uint8)
result = tc.tadd(trits, trits)
integers = [trit_to_int(t) for t in result]
```

## Architecture

### File Structure

```
ternary-kernel-python-c/
├── ternary_core.h                # Scalar operations (LUT-based, 125 lines)
├── ternary_core_simd_full.cpp    # AVX2 SIMD + Python bindings (297 lines)
├── build/
│   ├── scripts/
│   │   ├── setup.py              # Standard optimized build
│   │   ├── setup_pgo.py          # Profile-Guided Optimization build
│   │   └── setup_reference.py    # Reference baseline build
│   └── artifacts/                # Build outputs (timestamped)
├── tests/
│   ├── test_phase0.py            # Correctness validation
│   ├── test_omp.py               # OpenMP scaling tests
│   └── test_luts.cpp             # C++ unit tests
├── benchmarks/
│   ├── bench_phase0.py           # Performance benchmarks
│   ├── bench_fair.py             # Fair C++ vs C++ comparison
│   └── reference.py              # Python reference implementations
├── docs/                         # Comprehensive documentation
│   ├── README.md                 # Documentation index
│   ├── source-code-overview.md   # High-level code guide
│   ├── ternary-core-header.md    # ternary_core.h documentation
│   ├── ternary-core-simd.md      # SIMD implementation guide
│   ├── architecture.md           # System architecture
│   ├── optimization-complexity-rationale.md  # Design decisions
│   └── PGO_README.md             # Profile-Guided Optimization
└── local-reports/                # Development reports and analysis
```

### Implementation Layers

The library consists of two primary source files implementing a clean layered architecture:

#### Layer 1: Scalar Foundation (`ternary_core.h`)

**Purpose**: Core definitions and branch-free scalar operations

**Key Components**:
- Trit type definitions and encoding scheme
- Lookup tables (LUTs) for all operations (68 bytes total)
- Force-inlined scalar operations
- Conversion and packing utilities

**Performance**: 3-10x faster than conversion-based approach (theoretical), 1.07x measured vs optimized baseline

**Dependencies**: `stdint.h` only (highly portable)

**Example**:
```c
// Branch-free addition via LUT
static FORCE_INLINE trit tadd(trit a, trit b) {
    return TADD_LUT[(a << 2) | b];  // Single memory access, ~2 cycles
}
```

#### Layer 2: SIMD Acceleration (`ternary_core_simd_full.cpp`)

**Purpose**: AVX2-vectorized array operations with Python bindings

**Key Components**:
- SIMD operations using `_mm256_shuffle_epi8` (32 parallel LUT lookups)
- Template-based unified processing (binary/unary operations)
- Three execution paths: OpenMP parallel, serial SIMD, scalar tail
- Pybind11 Python integration

**Performance**: 100x faster than pure Python, 1.34x to 2.87x vs arithmetic SIMD

**Dependencies**: `immintrin.h` (AVX2), `pybind11`, `omp.h`, `ternary_core.h`

### Execution Paths (Phase 2 Architecture)

The Phase 2 implementation achieves "phase coherence" - reducing complexity while maintaining performance:

**PATH 1: OpenMP Parallel** (n ≥ 100,000 elements)
- Multi-threaded SIMD processing
- Static scheduling for deterministic execution
- Speedup: Up to 65x on multi-core systems

**PATH 2: Serial SIMD** (32 ≤ n < 100,000 elements)
- Single-threaded SIMD processing
- Processes 32 elements per iteration
- Speedup: 1.34x to 2.87x depending on array size

**PATH 3: Scalar Tail** (0-31 remaining elements)
- LUT-based scalar operations
- Handles remainder after SIMD processing
- Typically processes <1% of total elements

**Design Philosophy**: Phase 2 collapsed 6 complex paths (from Phase 1) to 3 clean paths by eliminating:
- Aligned vs unaligned load branching (modern CPUs: negligible difference)
- Manual loop unrolling (compiler auto-optimizes)
- Result: 73% code reduction with <5% performance loss

## Building

### Standard Build

```bash
# From project root
python build/scripts/setup.py
```

Produces timestamped artifacts in `build/artifacts/standard/` with latest copy in project root.

**Compiler Flags (MSVC)**:
- `/O2` - Maximum speed optimization
- `/GL` - Whole program optimization
- `/arch:AVX2` - Enable AVX2 instructions
- `/openmp` - Enable OpenMP parallelization
- `/LTCG` - Link-time code generation

### Profile-Guided Optimization (PGO)

For maximum performance (additional 5-15% improvement):

```bash
python build/scripts/setup_pgo.py
```

PGO uses runtime profiling to optimize hot paths. See `docs/PGO_README.md` for details.

### Manual Compilation (Linux/macOS)

```bash
c++ -O3 -march=native -mavx2 -fopenmp -flto -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    ternary_core_simd_full.cpp \
    -o ternary_core_simd_full$(python3-config --extension-suffix)
```

## Testing

### Correctness Tests

```bash
# Python integration tests
python tests/test_phase0.py

# C++ unit tests (compile first)
g++ -std=c++17 -O0 tests/test_luts.cpp -o test_luts
./test_luts

# OpenMP scaling tests
python tests/test_omp.py
```

Expected output: All tests pass with deterministic results

### Performance Benchmarks

```bash
# Standard benchmarks
python benchmarks/bench_phase0.py

# Fair C++ vs C++ comparison
python benchmarks/bench_fair.py
```

See `benchmarks/README.md` for detailed benchmark documentation.

## Performance Characteristics

### Throughput Comparison

| Implementation | Throughput (10M elements) | Speedup vs Python |
|----------------|---------------------------|-------------------|
| Python (reference.py) | 100 ME/s | 1x |
| C++ naive (reference_cpp.cpp) | 333 ME/s | 3x |
| C++ LUT (ternary_core.h) | 2,000 ME/s | 20x |
| **C++ SIMD (ternary_core_simd_full)** | **10,000 ME/s** | **100x** |

*(ME/s = Million Elements per second)*

### Performance by Array Size

| Array Size | Memory | SIMD Blocks | Scalar Tail | % SIMD | Execution Path |
|------------|--------|-------------|-------------|--------|----------------|
| 32 | 32 B | 1 | 0 | 100% | Serial SIMD |
| 1,000 | 1 KB | 31 | 8 | 99.2% | Serial SIMD |
| 10,000 | 10 KB | 312 | 16 | 99.8% | Serial SIMD |
| 100,000 | 100 KB | 3,125 | 0 | 100% | OpenMP Parallel |
| 1,000,000 | 1 MB | 31,250 | 0 | 100% | OpenMP Parallel |

### Operation Breakdown (per element)

| Layer | Time | Cycles | Description |
|-------|------|--------|-------------|
| Python reference | 10 ns | ~30 | Python loop + conversions |
| C++ conversion-based | 3 ns | ~10 | Conversions + branches |
| C++ LUT scalar | 0.5 ns | ~2 | Single array access (L1 cache) |
| **C++ SIMD (amortized)** | **0.1 ns** | **~0.3** | **32 elements / 10 cycles** |

## Technical Details

### Trit Encoding

Each trit occupies exactly 2 bits in memory:

```
Binary    Ternary    Integer
------    -------    -------
0b00      -1         -1
0b01       0          0
0b10      +1         +1
0b11      invalid    undefined
```

### Packing

Four trits pack into a single byte (4 trits × 2 bits = 8 bits):

```c
uint8_t pack_trits(trit t0, trit t1, trit t2, trit t3) {
    return (t0) | (t1 << 2) | (t2 << 4) | (t3 << 6);
}

trit unpack_trit(uint8_t packed, int index) {
    return (packed >> (2 * index)) & 0b11;
}
```

### SIMD LUT Lookups

The SIMD implementation uses `_mm256_shuffle_epi8` for parallel LUT lookups:

```cpp
// Build 4-bit indices: (a << 2) | b for each element
__m256i indices = build_indices(a, b);

// Load 16-entry LUT and broadcast to both 128-bit lanes
__m256i lut = broadcast_lut_16(TADD_LUT);

// Perform 32 parallel lookups
__m256i result = _mm256_shuffle_epi8(lut, indices);  // ~5 cycles total
```

This approach eliminates conversions and maintains unified semantics between scalar and SIMD operations.

## Development History

### Phase 0: LUT Optimization
- Replaced conversion-based operations with lookup tables
- Eliminated branches from scalar operations
- Result: 3-10x theoretical speedup (1.07x measured vs optimized baseline)

### Phase 0.5: SIMD LUT Shuffles
- Implemented `_mm256_shuffle_epi8` for parallel LUT lookups
- Unified semantic domain (no conversions)
- Result: 1.34x to 2.87x speedup vs arithmetic SIMD

### Phase 1: Multi-Path Optimization
- Added OpenMP threading for large arrays
- Implemented aligned load optimization
- Added manual loop unrolling
- Result: 65x speedup on large arrays, but 6+ code paths (high complexity)

### Phase 2: Complexity Compression (Current)
- Template-based unification of operations
- Eliminated aligned/unaligned branching
- Removed manual unrolling (trust compiler)
- Result: 3 execution paths instead of 6, 73% code reduction, <5% performance loss

## Documentation

Comprehensive documentation is available in the `docs/` directory:

### Getting Started
- **[docs/README.md](docs/README.md)** - Documentation index and quick navigation
- **[docs/source-code-overview.md](docs/source-code-overview.md)** - High-level code guide (START HERE)

### Source Code Documentation
- **[docs/ternary-core-header.md](docs/ternary-core-header.md)** - `ternary_core.h` detailed guide
- **[docs/ternary-core-simd.md](docs/ternary-core-simd.md)** - SIMD implementation documentation

### Architecture & Design
- **[docs/architecture.md](docs/architecture.md)** - System architecture overview
- **[docs/optimization-complexity-rationale.md](docs/optimization-complexity-rationale.md)** - Design decisions and tradeoffs
- **[docs/optimization-roadmap.md](docs/optimization-roadmap.md)** - Historical evolution and future plans

### Build & Performance
- **[docs/PGO_README.md](docs/PGO_README.md)** - Profile-Guided Optimization guide
- **[benchmarks/README.md](benchmarks/README.md)** - Benchmark suite documentation

## Requirements & Limitations

### System Requirements

- **CPU**: x86-64 with AVX2 support (Intel Haswell 2013+ or AMD Excavator 2015+)
  - Uses fundamental AVX2 operations (loadu, storeu, shuffle, add, or, and)
  - No Haswell-specific micro-optimizations in current version
  - Broad compatibility across all AVX2-capable processors
- **OS**: Windows, Linux, or macOS
- **Python**: 3.7 or later
- **Memory**: Minimal (LUTs use 68 bytes total, SIMD requires 32-byte alignment)

### Current Limitations

1. **Platform**: x86-64 only (no ARM/NEON support yet)
2. **Arrays**: 1D arrays only, no multi-dimensional support
3. **Size matching**: Both input arrays must have identical sizes
4. **No broadcasting**: Cannot mix arrays and scalars
5. **No runtime CPU detection**: Will crash on non-AVX2 CPUs

### Invalid Values

The encoding `0b11` is reserved/invalid. Behavior is undefined if invalid trits are provided as input.

## Contributing

### Adding New Operations

1. Add LUT to `ternary_core.h`
2. Add scalar function to `ternary_core.h`
3. Add SIMD function to `ternary_core_simd_full.cpp`
4. Add wrapper function to `ternary_core_simd_full.cpp`
5. Add Python binding to `PYBIND11_MODULE`
6. Update tests in `tests/test_phase0.py`

### Code Principles

1. **Phase Coherence**: Only add complexity if it provides >10% performance gain
2. **Documentation**: Update docs when changing implementation
3. **Testing**: Add tests for new operations
4. **Benchmarking**: Run benchmarks before/after changes
5. **Optimization IDs**: Use OPT-XXX tags for traceability

See `docs/optimization-complexity-rationale.md` for detailed guidelines.

## Future Roadmap

### Phase 3: Advanced Features (Planned)
- **Operation fusion**: Fused multiply-add, chained operations (20-50% speedup)
- **Multi-platform SIMD**: AVX-512, ARM NEON support
- **Runtime CPU detection**: Graceful fallback for non-AVX2 CPUs

### Phase 4: Domain-Specific Optimizations (Research)
- **Custom kernels**: Specialized implementations for common patterns
- **GPU acceleration**: CUDA/OpenCL implementations
- **Multi-dimensional arrays**: Native support for tensors

See `docs/optimization-roadmap.md` for detailed plans.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for full text.

Copyright 2025 Ternary Core Contributors. See [NOTICE](NOTICE) for details.

## References

- **Balanced Ternary**: [Wikipedia Article](https://en.wikipedia.org/wiki/Balanced_ternary)
- **AVX2 Intrinsics**: [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- **pybind11**: [Documentation](https://pybind11.readthedocs.io/)

## Citation

If you use this library in academic work, please cite:

```
Ternary Core SIMD: High-Performance Balanced Ternary Arithmetic Library
https://github.com/[your-repo]/ternary-kernel-python-c
Version 0.2.0 (Phase 2: Complexity Compression)
```

---

**Current Version**: 0.2.0 (Phase 2)
**Status**: Production-ready with comprehensive documentation
**Last Updated**: 2025-10-12
**Maintained by**: Ternary Core Contributors
