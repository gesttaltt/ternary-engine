# Ternary Core SIMD

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Production-grade balanced ternary arithmetic library with AVX2 SIMD vectorization, OpenMP parallelization, and Python bindings.

## Overview

Ternary Core implements high-performance balanced ternary logic operations using lookup table optimization, AVX2 SIMD vectorization (32 parallel operations), and OpenMP multi-threading. Achieves 100× throughput vs pure Python implementations.

**Balanced Ternary**: Three-valued logic system using {-1, 0, +1} with symmetric negative/positive representation. Applications include fractal generation, modulo-3 arithmetic, and specialized computational workflows.

### Features

- **2-bit trit encoding** - Compact representation (0b00=-1, 0b01=0, 0b10=+1)
- **Branch-free operations** - Pre-computed lookup tables eliminate conditional logic
- **AVX2 vectorization** - Process 32 trits per operation via `_mm256_shuffle_epi8`
- **OpenMP parallelization** - Automatic multi-threading for arrays ≥100K elements
- **NumPy integration** - Zero-copy array processing via pybind11

### Supported Operations

| Operation | Function | Description |
|-----------|----------|-------------|
| Addition | `tadd(a, b)` | Saturated addition (clamps to [-1, +1]) |
| Multiplication | `tmul(a, b)` | Standard multiplication |
| Minimum | `tmin(a, b)` | Element-wise minimum |
| Maximum | `tmax(a, b)` | Element-wise maximum |
| Negation | `tnot(a)` | Sign flip (0 unchanged) |

## Installation

### Requirements

- **Python** 3.7+
- **Compiler** C++17 (MSVC/GCC/Clang)
- **CPU** x86-64 with AVX2 (Intel Haswell 2013+, AMD Excavator 2015+)
- **Dependencies** pybind11, NumPy

### Build

```bash
pip install pybind11 numpy
python build.py
python -c "import ternary_simd_engine; print('Success')"
```

### Manual Compilation

**Linux/macOS:**
```bash
c++ -O3 -march=native -mavx2 -fopenmp -flto -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    ternary_simd_engine.cpp \
    -o ternary_simd_engine$(python3-config --extension-suffix)
```

**Windows (MSVC):**
```bash
cl /O2 /GL /arch:AVX2 /openmp /std:c++17 /EHsc /LD ^
   ternary_simd_engine.cpp /link /LTCG
```

## Usage

### Basic Example

```python
import numpy as np
import ternary_simd_engine as tc

# Encoding constants
MINUS_ONE = 0b00
ZERO      = 0b01
PLUS_ONE  = 0b10

# Create arrays
a = np.array([MINUS_ONE, ZERO, PLUS_ONE], dtype=np.uint8)
b = np.array([PLUS_ONE, ZERO, MINUS_ONE], dtype=np.uint8)

# Operations
result = tc.tadd(a, b)  # [0, 0, 0]
```

### Conversion Helpers

```python
def int_to_trit(value):
    return 0b00 if value < 0 else 0b10 if value > 0 else 0b01

def trit_to_int(trit):
    return -1 if trit == 0b00 else 1 if trit == 0b10 else 0

# Convert integer arrays
values = [-1, 0, 1, -1, 1]
trits = np.array([int_to_trit(v) for v in values], dtype=np.uint8)
result = tc.tadd(trits, trits)
```

## Performance

### Throughput

| Implementation | Throughput (10M elements) | Speedup |
|----------------|---------------------------|---------|
| Python | 100 ME/s | 1× |
| C++ (naive) | 333 ME/s | 3× |
| C++ (LUT) | 2,000 ME/s | 20× |
| **C++ (SIMD)** | **10,000 ME/s** | **100×** |

*(ME/s = Million Elements/second)*

### Execution Paths

| Array Size | Memory | Path | Speedup |
|------------|--------|------|---------|
| < 32 | < 32 B | Scalar | 20× |
| 32 - 100K | 32 B - 100 KB | Serial SIMD | 1.3-2.9× |
| ≥ 100K | ≥ 100 KB | OpenMP Parallel | Up to 65× |

### Latency (per element)

| Implementation | Time | CPU Cycles |
|----------------|------|------------|
| Python | 10 ns | ~30 |
| C++ LUT | 0.5 ns | ~2 |
| **C++ SIMD** | **0.1 ns** | **~0.3** |

## Architecture

### Core Files

```
ternary_lut_gen.h        # Compile-time LUT generation (111 lines)
ternary_algebra.h        # Scalar operations + LUTs (143 lines)
ternary_errors.h         # Exception handling (119 lines)
ternary_simd_engine.cpp  # Vectorized execution (333 lines)
```

Total implementation: **~700 lines of core code**

### Design

**Layer 0**: Constexpr LUT generation - Compile-time table construction
**Layer 1**: Scalar operations - Branch-free lookup table operations
**Layer 2**: SIMD vectorization - 32-wide parallel processing via AVX2
**Layer 3**: Python bindings - Zero-copy NumPy integration

See [docs/api-reference/](docs/api-reference/) for detailed architecture documentation.

## Testing

```bash
# Correctness
python tests/test_phase0.py

# Performance
python benchmarks/bench_phase0.py

# OpenMP scaling
python tests/test_omp.py
```

See [tests/README.md](tests/README.md) for full testing documentation.

## Documentation

- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development guidelines
- **[CHANGELOG.md](CHANGELOG.md)** - Version history
- **[docs/](docs/)** - Complete API reference and architecture docs
- **[build/README.md](build/README.md)** - Build system documentation
- **[tests/README.md](tests/README.md)** - Test suite documentation

## Limitations

- **Platform**: x86-64 only (ARM/NEON support planned)
- **Arrays**: 1D arrays only
- **CPU requirement**: AVX2 instruction set (2013+ Intel, 2015+ AMD)
- **Size matching**: Binary operations require identical array sizes
- **Invalid encoding**: 0b11 is reserved/undefined

## Advanced Features

### Profile-Guided Optimization

Additional 5-15% performance gain:

```bash
python build_pgo.py full
```

See [docs/PGO_README.md](docs/PGO_README.md) for details.

### Compile-Time Options

```cpp
// Disable input sanitization for validated data pipelines (3-5% gain)
#define TERNARY_NO_SANITIZE
```

## Roadmap

**Current**: v0.3.0 - Production optimizations (Phase 3)

**Planned**:
- Multi-platform SIMD (AVX-512, ARM NEON)
- Runtime CPU detection and fallback
- Operation fusion (fused multiply-add)
- Multi-dimensional array support
- Profiler integration (VTune ITT, NVTX for GPU, Perfetto)
  - Framework implemented in `ternary_profiler.h`
  - Awaiting integration into execution engine

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development workflow
- Coding standards
- Testing requirements
- Performance guidelines

## License

Apache License 2.0 - See [LICENSE](LICENSE)

Copyright 2025 Jonathan Verdun (Ternary Core Experimental Project)

## Citation

```bibtex
@software{ternary_core_simd,
  title={Ternary Core SIMD: High-Performance Balanced Ternary Arithmetic},
  author={Jonathan Verdun},
  year={2025},
  version={0.3.0},
  url={https://github.com/[your-repo]/ternary-kernel-python-c}
}
```

## References

- [Balanced Ternary (Wikipedia)](https://en.wikipedia.org/wiki/Balanced_ternary)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- [pybind11 Documentation](https://pybind11.readthedocs.io/)

---

**Version**: 0.3.0
**Status**: Production
**Updated**: 2025-10-13
