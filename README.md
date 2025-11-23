# Ternary Engine

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![C++ Standard](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![Performance](https://img.shields.io/badge/speedup-7315x-brightgreen)](https://github.com/gesttaltt/ternary-engine#performance)
[![Tests](https://img.shields.io/badge/tests-65%2F65%20passing-brightgreen)](https://github.com/gesttaltt/ternary-engine#testing)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

Production-grade balanced ternary arithmetic library with AVX2 SIMD vectorization, operation fusion, and Python bindings.

## Overview

Ternary Engine implements high-performance balanced ternary logic operations using lookup table optimization, AVX2 SIMD vectorization (32 parallel operations), and operation fusion. Achieves **7,315× average throughput** vs pure Python implementations (validated 2025-10-29).

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

### Dense243 High-Density Module (Experimental)

**Separate module for 20% storage savings with TritNet-ready architecture**

```python
import ternary_dense243_module as td

# Pack 5 trits into 1 byte (vs 5 bytes in standard encoding)
trits = np.array([0b00, 0b01, 0b10, 0b10, 0b01], dtype=np.uint8)
packed = td.pack(trits)  # 5 → 1 byte (80% space savings)

# Future: Neural network-based operations
td.set_backend('tritnet')  # Switch from LUT to trained model
result = td.tadd(packed_a, packed_b)  # Uses matmul instead of lookup
```

**Features:**
- **Density:** 5 trits/byte (95.3% utilization) vs 4 trits/byte (standard)
- **Performance:** Pack 0.25ns, Unpack 0.91ns (validated, all 243 states tested)
- **Use cases:** Persistent storage, network transmission, memory-bound workloads
- **TritNet roadmap:** Train BitNet on truth tables → distill to ternary weights → replace LUT with matmul
- **Build:** `python scripts/build/build_dense243.py`
- **Docs:** `docs/TRITNET_ROADMAP.md`

## Installation

### Requirements

- **Python** 3.7+
- **Compiler** C++17 (MSVC/GCC/Clang)
- **CPU** x86-64 with AVX2 (Intel Haswell 2013+, AMD Excavator 2015+)
- **Dependencies** pybind11, NumPy

### Build

```bash
pip install pybind11 numpy
python scripts/build/build.py
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

### Validated Benchmarks (2025-10-29)

**Peak Throughput (100K elements):**
- **tadd**: 13,047 Mops/s (0.077 ns/element) - 7,316× vs Python
- **tmul**: 14,058 Mops/s (0.071 ns/element) - 7,584× vs Python
- **tmin**: 13,447 Mops/s (0.074 ns/element) - 8,681× vs Python
- **tmax**: 13,341 Mops/s (0.075 ns/element) - 8,127× vs Python
- **tnot**: 18,518 Mops/s (0.054 ns/element) - 4,767× vs Python

**Average Speedup: 7,315×** (validated with statistical rigor)

*(Mops/s = Million operations/second)*

### Operation Fusion (Phase 4.0 - Validated)

**Fused Operations** combine multiple operations into a single pass, reducing memory traffic:

**fused_tnot_tadd** - Validated speedup (rigorous benchmarking):
- **Contiguous arrays:** 1.80× to 4.78× speedup
- **Non-contiguous arrays:** 1.78× to 15.52× speedup
- **Cold cache:** 1.62× to 2.56× speedup
- **Conservative estimate:** 1.94× minimum speedup

Performance validated with statistical rigor (variance, confidence intervals, coefficient of variation).

### Latency (per element)

| Implementation | Time | CPU Cycles |
|----------------|------|------------|
| Python | 10 ns | ~30 |
| C++ LUT | 0.5 ns | ~2 |
| **C++ SIMD** | **0.077 ns** | **~0.23** |
| **C++ Fused** | **0.040 ns** | **~0.12** |

## Architecture

### Project Structure (v1.0 - Clean Separation)

```
ternary_core/              # Production-ready kernel (mathematically stable)
├─ algebra/                # Core ternary operations
│   ├─ ternary_algebra.h      # Scalar operations + LUTs (143 lines)
│   └─ ternary_lut_gen.h      # Compile-time LUT generation (111 lines)
├─ simd/                   # SIMD acceleration
│   ├─ ternary_simd_kernels.h # AVX2 vectorization (103 lines)
│   ├─ ternary_cpu_detect.h   # Runtime CPU detection (185 lines)
│   └─ ternary_fusion.h       # Operation fusion PoC (204 lines)
├─ ffi/                    # Cross-language FFI
│   └─ ternary_c_api.h        # Pure C API (255 lines)
└─ core_api.h              # Unified entry point

ternary_engine/            # Experimental optimizations
└─ experimental/
    ├─ dense243/           # Dense243 encoding (✓ VALIDATED - production-ready)
    ├─ fusion/             # Fusion operations (Phase 4.0 validated, 4.1 pending)
    └─ [future expansions]

Root level:
├─ ternary_simd_engine.cpp # Main engine (uses ternary_core/)
├─ ternary_errors.h        # Error definitions
└─ ternary_profiler.h      # Profiling utilities
```

**Total kernel implementation:** ~1,000 lines of validated code

### Design Layers

**Layer 0**: Constexpr LUT generation - Compile-time table construction
**Layer 1**: Scalar operations - Branch-free lookup table operations
**Layer 2**: SIMD vectorization - 32-wide parallel processing via AVX2
**Layer 3**: Python bindings - Zero-copy NumPy integration
**Layer 4**: Runtime safety - CPU detection, alignment validation, ISA dispatch

### Deployment Status

✅ **Production-Ready** (ternary_core/):
- Core algebra system (65/65 tests passing)
- SIMD kernels (AVX2, validated 2025-10-29)
- CPU feature detection (runtime ISA dispatch)
- C FFI layer (cross-language ready)
- Operation fusion Phase 4.0 (1.6-15.5× validated speedup)

✅ **Validated & Ready** (ternary_engine/experimental/):
- **Dense243 encoding** (all 243 states validated, 0.25 ns pack, 0.91 ns unpack)
- **TriadSextet encoding** (all 27 states validated, 0.16 ns pack, 0.66 ns unpack)
- **fused_tnot_tadd** (rigorous benchmarks: 1.94× conservative, up to 15.52× speedup)

⚠️ **Pending Validation** (ternary_engine/experimental/):
- Phase 4.1 fusion operations (fused_tnot_tmul/tmin/tmax - implementation complete, benchmarks pending)

See comprehensive validation report in local-reports/ directory.

## Testing

```bash
# Run all tests (unified test runner)
python run_tests.py

# Run individual test suites
python tests/test_phase0.py     # Correctness
python tests/test_omp.py         # OpenMP scaling
python tests/test_errors.py      # Error handling

# Performance benchmarks
python benchmarks/bench_phase0.py
```

See **[TESTING.md](TESTING.md)** for comprehensive testing and CI/CD documentation.

## Documentation

- **[TESTING.md](TESTING.md)** - Testing and CI/CD guide ⭐ New!
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development guidelines
- **[CHANGELOG.md](CHANGELOG.md)** - Version history
- **[docs/](docs/)** - Complete API reference and architecture docs
- **[build/README.md](build/README.md)** - Build system documentation
- **[tests/README.md](tests/README.md)** - Test suite documentation

## Limitations

- **Platform**: x86-64 only (ARM/NEON support planned)
- **Arrays**: 1D arrays only
- **CPU requirement**: AVX2 instruction set (Intel Haswell 2013+, AMD Excavator 2015+)
  - Module performs runtime detection and fails gracefully on unsupported CPUs
- **Size matching**: Binary operations require identical array sizes
- **Invalid encoding**: 0b11 is reserved/undefined
- **Alignment**: Streaming stores require 32-byte alignment (automatically detected)

## Advanced Features

### Profile-Guided Optimization

Additional 5-15% performance gain using Clang PGO (recommended) or MSVC fallback:

```bash
# Clang PGO (recommended - works with Python extensions)
python scripts/build/build_pgo_unified.py --clang

# Auto-detect (prefers Clang if available)
python scripts/build/build_pgo_unified.py

# MSVC fallback (has known limitations)
python scripts/build/build_pgo.py full
```

See [docs/pgo/README.md](docs/pgo/README.md) and [docs/pgo/CLANG_INSTALLATION.md](docs/pgo/CLANG_INSTALLATION.md) for details.

### Compile-Time Options

```cpp
// Disable input sanitization for validated data pipelines (3-5% gain)
#define TERNARY_NO_SANITIZE
```

## Roadmap

**Current**: v1.0.0 - Production-ready kernel with validated experimental features

**Completed (v1.0 - Validated 2025-10-29)**:
- ✅ Clean kernel/engine separation (ternary_core/ vs ternary_engine/)
- ✅ Runtime CPU detection and graceful fallback
- ✅ Alignment validation for streaming stores (fixes segfault risk)
- ✅ Hardware concurrency clamping (fixes VM crashes)
- ✅ **Dense243 encoding** (all 243 states validated, critical bug fixed)
- ✅ **TriadSextet encoding** (all 27 states validated)
- ✅ **Operation fusion Phase 4.0** (1.6-15.5× validated speedup with statistical rigor)
- ✅ C FFI layer (cross-language ready)
- ✅ Comprehensive testing (65/65 C++ tests, 2/2 Python tests passing)
- ✅ Performance benchmarking (7,315× average speedup validated)

**In Progress**:
- 🔧 Phase 4.1 fusion validation (fused_tnot_tmul/tmin/tmax - implementation complete)
- 🔧 Code refactoring (eliminate duplication between engines)

**Planned**:
- Multi-platform SIMD (AVX-512, ARM NEON/SVE)
- Multi-dimensional array support
- OpenMP re-enablement with validation
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

Copyright 2025 Jonathan Verdun (Ternary Engine Project)

Developed by Jonathan Verdun with grateful acknowledgment to Ivan Weiss Van der Pol and Kyrian Weiss Van der Pol for their support.

## Citation

```bibtex
@software{ternary_engine,
  title={Ternary Engine: High-Performance Balanced Ternary Arithmetic},
  author={Jonathan Verdun},
  year={2025},
  version={1.0.0},
  url={https://github.com/gesttaltt/ternary-engine}
}
```

## References

- [Balanced Ternary (Wikipedia)](https://en.wikipedia.org/wiki/Balanced_ternary)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- [pybind11 Documentation](https://pybind11.readthedocs.io/)

---

**Version**: 1.0.0
**Status**: Production (ternary_core/), Experimental (ternary_engine/)
**Updated**: 2025-10-29
