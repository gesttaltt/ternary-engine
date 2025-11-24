# Backend API Documentation (v1.2.0)

**Doc-Type:** Backend System Reference · Version 1.2.0 · Updated 2025-11-24 · Author Ternary Engine Team

Documentation for the v1.2.0 backend system - a pluggable architecture for ternary operations with multiple optimized implementations.

---

## Overview

The v1.2.0 backend system provides a unified interface for accessing multiple implementations of ternary arithmetic operations. The system automatically selects the best available backend for your hardware and allows runtime backend switching for benchmarking and testing.

**Key Features:**
- **Runtime Backend Selection**: Choose between Scalar, AVX2_v1, and AVX2_v2 backends
- **Automatic Hardware Detection**: System selects the best available backend on initialization
- **Unified Dispatch API**: Single API for all ternary operations regardless of backend
- **Performance Transparency**: Query backend capabilities and compare performance
- **Cross-Backend Validation**: All backends produce identical results

---

## Available Backends

### Scalar (v1.2.0)
**Description**: Portable scalar reference implementation
**Performance**: ~2,000-2,700 Mops/s
**Availability**: Always available (no special CPU requirements)
**Use Case**: Baseline for correctness validation and portable deployment

### AVX2_v1 (v1.1.0)
**Description**: AVX2 baseline implementation from v1.1.0
**Performance**: ~10,000-44,000 Mops/s (5-16× faster than Scalar)
**Availability**: Requires AVX2 support (Intel Haswell 2013+, AMD Excavator 2015+)
**Use Case**: Proven production baseline for AVX2-capable systems

### AVX2_v2 (v1.2.0)
**Description**: AVX2 with v1.2.0 optimizations (traditional indexing)
**Performance**: ~9,000-45,000 Mops/s (5-17× faster than Scalar)
**Availability**: Requires AVX2 support
**Use Case**: Latest optimizations with ongoing development
**Note**: Currently using traditional indexing; canonical indexing optimization deferred until proper LUT reorganization

---

## Installation

Build the backend module:

```bash
python build/build_backend.py
```

This creates `ternary_backend.cp312-win_amd64.pyd` (or `.so` on Linux/macOS) in the project root.

---

## Basic Usage

### Initialization

```python
import ternary_backend as tb

# Initialize backend system (must be called first)
tb.init()
```

### Listing Available Backends

```python
# Get list of all backends
backends = tb.list_backends()

for backend in backends:
    print(f"{backend.name} (v{backend.version})")
    print(f"  Description: {backend.description}")
    print(f"  Active: {backend.is_active}")
    print(f"  Available: {backend.is_available}")
    print(f"  Capabilities: 0x{backend.capabilities:08X}")
    print(f"  Batch size: {backend.preferred_batch_size}")
    print()
```

**Output:**
```
Scalar (v1.2.0)
  Description: Portable scalar reference implementation
  Active: False
  Available: True
  Capabilities: 0x00000001
  Batch size: 1

AVX2_v1 (v1.1.0)
  Description: AVX2 baseline implementation (v1.1.0)
  Active: False
  Available: True
  Capabilities: 0x00000004
  Batch size: 32

AVX2_v2 (v1.2.0)
  Description: AVX2 with v1.2.0 optimizations (canonical indexing)
  Active: True
  Available: True
  Capabilities: 0x00000004
  Batch size: 32
```

### Querying Active Backend

```python
# Get name of currently active backend
active = tb.get_active()
print(f"Active backend: {active}")  # "AVX2_v2" (selected automatically)
```

### Selecting a Backend

```python
# Set specific backend
tb.set_backend("AVX2_v1")

# Verify selection
print(f"Active backend: {tb.get_active()}")  # "AVX2_v1"
```

### Dispatch Operations

All ternary operations are accessed through dispatch functions that route to the active backend:

```python
import numpy as np

# Create test data (2-bit trit encoding: 0=-1, 1=0, 2=+1)
a = np.array([0, 1, 2, 0, 1], dtype=np.uint8)
b = np.array([2, 1, 0, 1, 2], dtype=np.uint8)

# Unary operation
result = tb.tnot(a)  # Ternary NOT

# Binary operations
result = tb.tadd(a, b)  # Ternary addition
result = tb.tmul(a, b)  # Ternary multiplication
result = tb.tmax(a, b)  # Ternary maximum
result = tb.tmin(a, b)  # Ternary minimum
```

---

## Performance Benchmarking

### Quick Performance Comparison

```python
import time
import numpy as np
import ternary_backend as tb

tb.init()

# Test data (1M trits)
n = 1_000_000
a = np.random.randint(0, 3, n, dtype=np.uint8)
b = np.random.randint(0, 3, n, dtype=np.uint8)

backends = tb.list_backends()

for backend in backends:
    tb.set_backend(backend.name)

    # Warm-up
    _ = tb.tadd(a, b)

    # Benchmark
    start = time.perf_counter()
    for _ in range(10):
        _ = tb.tadd(a, b)
    elapsed = time.perf_counter() - start

    # Calculate throughput
    ops = n * 10
    mops_per_sec = (ops / elapsed) / 1e6

    print(f"{backend.name:12s}: {mops_per_sec:8.2f} Mops/s")
```

**Expected Output (AMD Ryzen 5800H):**
```
Scalar      :  2,723.50 Mops/s
AVX2_v1     :  9,238.81 Mops/s
AVX2_v2     :  8,727.15 Mops/s
```

### Comprehensive Benchmarks

For detailed performance analysis:

```bash
# Full benchmark suite (all sizes)
python benchmarks/bench_backends.py

# Quick test (4 sizes)
python benchmarks/bench_backends.py --quick

# Custom output directory
python benchmarks/bench_backends.py --output=my_results/
```

**Benchmark Results Include:**
- Throughput (Mops/s) for all operations and sizes
- Latency (ns/element)
- Speedup relative to Scalar baseline
- Statistical variance and coefficient of variation
- JSON results file with complete metadata

---

## Cross-Backend Validation

Validate that all backends produce identical results:

```python
import numpy as np
import ternary_backend as tb

tb.init()

# Test data
np.random.seed(42)
n = 100
a = np.random.randint(0, 3, n, dtype=np.uint8)
b = np.random.randint(0, 3, n, dtype=np.uint8)

# Collect results from each backend
backends = tb.list_backends()
results = {}

for backend in backends:
    tb.set_backend(backend.name)
    results[backend.name] = {
        'tnot': tb.tnot(a),
        'tadd': tb.tadd(a, b),
        'tmul': tb.tmul(a, b),
        'tmax': tb.tmax(a, b),
        'tmin': tb.tmin(a, b),
    }

# Compare all backends against Scalar
reference = results['Scalar']
for backend_name, backend_results in results.items():
    if backend_name == 'Scalar':
        continue

    for op_name, result in backend_results.items():
        if not np.array_equal(result, reference[op_name]):
            print(f"MISMATCH: {backend_name}.{op_name}")
        else:
            print(f"MATCH: {backend_name}.{op_name}")
```

**Expected Output:**
```
MATCH: AVX2_v1.tnot
MATCH: AVX2_v1.tadd
MATCH: AVX2_v1.tmul
MATCH: AVX2_v1.tmax
MATCH: AVX2_v1.tmin
MATCH: AVX2_v2.tnot
MATCH: AVX2_v2.tadd
MATCH: AVX2_v2.tmul
MATCH: AVX2_v2.tmax
MATCH: AVX2_v2.tmin
```

---

## Backend Capabilities

Each backend reports a capability bitfield:

```python
# Get capabilities string
backend = tb.list_backends()[0]
caps_str = tb.get_capabilities_string(backend.capabilities)
print(caps_str)
```

**Capability Flags:**
- `TERNARY_CAP_SCALAR (0x01)`: Scalar implementation
- `TERNARY_CAP_SIMD_128 (0x02)`: 128-bit SIMD (SSE)
- `TERNARY_CAP_SIMD_256 (0x04)`: 256-bit SIMD (AVX2)
- `TERNARY_CAP_SIMD_512 (0x08)`: 512-bit SIMD (AVX-512)
- `TERNARY_CAP_OPENMP (0x10)`: OpenMP parallelization
- `TERNARY_CAP_FUSION (0x20)`: Fused operations
- `TERNARY_CAP_CANONICAL (0x40)`: Canonical indexing
- `TERNARY_CAP_DUAL_SHUFFLE (0x80)`: Dual-shuffle optimization
- `TERNARY_CAP_LUT_256B (0x100)`: 256-byte LUT support

---

## Performance Characteristics

### Scalar Backend
- **Throughput**: 2,000-2,700 Mops/s
- **Latency**: 360-500 ns/element
- **Scaling**: Linear with array size
- **Memory**: Minimal cache impact
- **Best For**: Arrays < 100 elements, portability

### AVX2_v1 Backend
- **Throughput**: 10,000-44,000 Mops/s
- **Latency**: 23-100 ns/element
- **Scaling**: Excellent for 100K+ elements (11-16× speedup)
- **Memory**: Cache-friendly for arrays < 1M elements
- **Best For**: Production workloads, proven performance

### AVX2_v2 Backend
- **Throughput**: 9,000-45,000 Mops/s
- **Latency**: 22-110 ns/element
- **Scaling**: Similar to AVX2_v1
- **Memory**: Under investigation (some operations slower at 1M+ elements)
- **Best For**: Ongoing optimization, future improvements

**Performance Notes:**
- Small arrays (< 100): Scalar competitive due to overhead
- Medium arrays (100-100K): AVX2 shows 1.1-12× speedup
- Large arrays (100K-1M): AVX2 shows 5-17× speedup
- Very large arrays (10M+): Memory bandwidth becomes bottleneck

---

## Error Handling

All operations validate input:

```python
import numpy as np
import ternary_backend as tb

tb.init()

# Mismatched array sizes
a = np.array([0, 1, 2], dtype=np.uint8)
b = np.array([0, 1], dtype=np.uint8)

try:
    result = tb.tadd(a, b)
except ValueError as e:
    print(f"Error: {e}")  # "tadd: array size mismatch"
```

**Common Errors:**
- `ValueError`: Array size mismatch for binary operations
- `RuntimeError`: Backend not initialized (forgot `tb.init()`)
- `RuntimeError`: Backend not available on this CPU

---

## Integration with Existing Code

### Migrating from ternary_simd_engine

The backend API is designed to coexist with the existing `ternary_simd_engine` module:

```python
# Old code (ternary_simd_engine)
import ternary_simd_engine as tc
result = tc.tadd(a, b)

# New code (ternary_backend)
import ternary_backend as tb
tb.init()
tb.set_backend("AVX2_v2")
result = tb.tadd(a, b)
```

**Benefits of Backend API:**
- Runtime backend selection
- Automatic best-backend selection
- Cross-backend validation
- Performance comparison tools

---

## Testing

Run the integration test suite:

```bash
python tests/python/test_backend_integration.py
```

**Tests Include:**
- Backend initialization
- Backend listing and selection
- Dispatch operations correctness
- Cross-backend consistency
- Performance comparison

---

## Advanced Usage

### Automatic Best-Backend Selection

The system automatically selects the best backend on initialization based on:
1. SIMD width (512 > 256 > 128 > scalar)
2. Optimization features (dual-shuffle, canonical, LUT-256B, fusion)
3. CPU availability check

```python
import ternary_backend as tb

# Initialize (automatically selects best)
tb.init()

# Check what was selected
print(f"Auto-selected: {tb.get_active()}")  # Usually "AVX2_v2" on modern CPUs
```

### Backend Selection for Specific Workloads

```python
import ternary_backend as tb

tb.init()

# For small arrays (< 100 elements): Scalar may be faster
if array_size < 100:
    tb.set_backend("Scalar")

# For medium arrays (100-100K): AVX2 provides good speedup
elif array_size < 100_000:
    tb.set_backend("AVX2_v2")

# For large arrays (100K+): AVX2 provides maximum throughput
else:
    tb.set_backend("AVX2_v2")
```

### Persistent Backend Selection

```python
import os
import ternary_backend as tb

# Set environment variable for default backend
os.environ['TERNARY_BACKEND'] = 'AVX2_v1'

tb.init()
# Will use AVX2_v1 instead of auto-selected AVX2_v2
```

---

## Architecture Notes

### Backend Selection Algorithm

The `ternary_backend_select_best()` function scores backends:

```
Score = SIMD_WIDTH_SCORE + OPTIMIZATION_FEATURES_SCORE

SIMD_WIDTH_SCORE:
- AVX-512: 1000 points
- AVX2:     500 points
- SSE:      250 points
- Scalar:     0 points

OPTIMIZATION_FEATURES:
- Dual-shuffle:  100 points
- Canonical:      50 points
- LUT-256B:       50 points
- Fusion:         25 points
- OpenMP:         25 points
```

### Dispatch Overhead

Backend dispatch adds minimal overhead:
- Function pointer dereference: ~1-2 cycles
- Backend availability check: ~0 cycles (inlined)
- Total overhead: < 0.1% for arrays > 32 elements

### Memory Layout

All backends use 2-bit trit encoding:
- `0b00` = -1 (MINUS_ONE)
- `0b01` =  0 (ZERO)
- `0b10` = +1 (PLUS_ONE)
- `0b11` = Invalid (reserved)

**Storage**: 4 trits per byte in 2-bit encoding

---

## Troubleshooting

### Backend Not Available

```python
import ternary_backend as tb

tb.init()

# Check if specific backend available
backends = tb.list_backends()
avx2_available = any(b.name == "AVX2_v2" and b.is_available for b in backends)

if not avx2_available:
    print("AVX2 not available on this CPU")
    print("Falling back to Scalar")
    tb.set_backend("Scalar")
```

### Performance Lower Than Expected

**Possible Causes:**
1. **Thermal throttling**: Monitor CPU temperature
2. **Memory bandwidth**: Test with smaller arrays
3. **Background processes**: Close other applications
4. **Wrong backend**: Verify `tb.get_active()`
5. **Debug build**: Ensure release build (`/O2 /GL /LTCG`)

**Debug Commands:**
```python
# Check active backend
print(f"Active: {tb.get_active()}")

# Check backend capabilities
backends = tb.list_backends()
for b in backends:
    if b.is_active:
        print(f"Capabilities: 0x{b.capabilities:08X}")
        print(f"Batch size: {b.preferred_batch_size}")
```

---

## Future Developments

### Planned Features (v1.3.0+)

1. **Canonical Indexing**: Properly reorganized LUTs for `(a*3)+b` indexing
2. **Dual-Shuffle XOR**: Parallel shuffle + XOR combining
3. **LUT-256B**: 256-byte LUTs for improved cache utilization
4. **Operation Fusion**: `tadd_tmul`, `tmul_tadd` fused operations
5. **AVX-512 Backend**: 64-wide SIMD for latest Intel/AMD CPUs
6. **ARM NEON Backend**: Mobile/embedded deployment
7. **GPU Backend**: CUDA/ROCm for massive parallelism

### Contributing

To add a new backend:

1. Implement operations in `src/core/simd/ternary_backend_<name>.cpp`
2. Define backend struct with `TernaryBackendInfo` and function pointers
3. Register backend in `ternary_backend_init()`
4. Test with integration suite
5. Benchmark and document performance

See `src/core/simd/ternary_backend_scalar.cpp` for reference implementation.

---

## References

- **Backend Interface**: `src/core/simd/ternary_backend_interface.h`
- **Dispatch Implementation**: `src/core/simd/ternary_backend_dispatch.cpp`
- **Python Bindings**: `src/engine/bindings_backend_api.cpp`
- **Build Script**: `build/build_backend.py`
- **Integration Tests**: `tests/python/test_backend_integration.py`
- **Benchmark Script**: `benchmarks/bench_backends.py`

---

## Changelog

| Date       | Version | Description                              |
|:-----------|:--------|:-----------------------------------------|
| 2025-11-24 | v1.2.0  | Initial backend system with Scalar, AVX2_v1, AVX2_v2 |

---

**Version:** 1.2.0 · **Updated:** 2025-11-24 · **Status:** Production-ready (Windows x64 validated)
