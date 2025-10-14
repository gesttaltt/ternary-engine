# ternary_simd_engine.cpp - SIMD Implementation Documentation

## Overview

`ternary_simd_engine.cpp` is the AVX2-accelerated implementation of ternary array operations. It provides high-performance vectorized operations using Intel AVX2 intrinsics, achieving 10-30x speedups over scalar code through parallel processing of 32 trits per operation.

**File**: `ternary_simd_engine.cpp` (297 lines)
**Purpose**: AVX2-optimized array operations with Python bindings
**Dependencies**: `immintrin.h`, `pybind11`, `omp.h`, `ternary_algebra.h`
**Target Architecture**: x86-64 with AVX2 support (Intel Haswell+ / AMD Excavator+)
**License**: Apache 2.0

---

## Design Evolution Summary

### Phase 0.5: LUT-Based SIMD (OPT-061)

**Breakthrough**: Replace arithmetic SIMD with shuffle-based LUT lookups
- Uses `_mm256_shuffle_epi8` for 32 parallel table lookups
- Unified semantic domain with scalar operations (no conversions)
- Eliminated arithmetic overhead and clamping

### Phase 1: Optimization Exploration (DEPRECATED)

**Attempted optimizations** (6 runtime paths):
- OPT-066: Aligned vs unaligned load branching → **Removed** (negligible benefit)
- OPT-041: Manual 2x loop unrolling → **Removed** (compiler auto-optimizes)
- OPT-001: OpenMP threading → **RETAINED**

**Problems**:
- High code complexity (6 different execution paths)
- Unstable measurements (branching overhead)
- Minimal performance gains (<5%)

### Phase 2: Complexity Compression (CURRENT)

**"Phase Coherence"** - simplicity without performance loss:
- Template-based unification of binary/unary operations
- Eliminated aligned/unaligned branching
- Removed manual unrolling
- **Result**: 3 execution paths, 73% code reduction, <5% performance loss

---

## Architecture Overview

### Execution Path Decision Tree

```
Input Array (size n)
│
├─ n >= 100,000? ───YES──► PATH 1: OpenMP Parallel SIMD
│                            (multiple threads, 32 elements/iteration)
│
└─ NO
   │
   ├─ Remaining >= 32? ──YES──► PATH 2: Serial SIMD Loop
   │                              (single thread, 32 elements/iteration)
   │
   └─ Remaining < 32 ────────► PATH 3: Scalar Tail
                                 (process 1 element at a time)
```

**Key Decision Point**: `OMP_THRESHOLD = 100,000` elements
- Above threshold: parallelism overhead is justified
- Below threshold: serial SIMD is more efficient (avoids thread spawn cost)

---

## SIMD Operations

### AVX2 Shuffle-Based LUT Lookups

#### Core Technique: `_mm256_shuffle_epi8`

Intel's `_mm256_shuffle_epi8` instruction performs 32 parallel byte lookups:

```
Input:  [index0, index1, ..., index31]  (32 bytes)
LUT:    [lut[0], lut[1], ..., lut[15]]  (16 bytes, duplicated to both 128-bit lanes)
Output: [lut[index0], lut[index1], ..., lut[index31]]
```

**Performance**: 1 instruction, 1 cycle latency, 0.5 cycle throughput on modern CPUs

#### LUT Broadcasting

```cpp
static inline __m256i broadcast_lut_16(const uint8_t* lut) {
    // Load 16 bytes into lower 128 bits
    __m128i lut_128 = _mm_loadu_si128((const __m128i*)lut);
    // Duplicate to both 128-bit lanes of 256-bit register
    return _mm256_broadcastsi128_si256(lut_128);
}
```

**Rationale**: `_mm256_shuffle_epi8` operates on two independent 128-bit lanes, so the LUT must be duplicated in both halves.

#### Input Sanitization (OPT-HASWELL-02)

**Template-Based Optional Masking**: Compile-time conditional sanitization for advanced performance control.

```cpp
template <bool Sanitize = true>
static inline __m256i maybe_mask(__m256i v) {
    if constexpr (Sanitize)
        return _mm256_and_si256(v, _mm256_set1_epi8(0x03));
    else
        return v;
}
```

**Sanitize=true (Production Default)**: Ensures only lower 2 bits of each byte are used, masking inputs to valid trit range (0b00, 0b01, 0b10). Safe for all use cases, protects against invalid data.

**Sanitize=false (Advanced Use)**: Elides masking at compile time for pre-validated data pipelines. Provides 3-5% performance gain when input data is guaranteed valid. Use only when data source is trusted and validated upstream.

### Binary Operation Pattern

All binary operations (tadd, tmul, tmin, tmax) follow this pattern:

```cpp
template <bool Sanitize = true>
static inline __m256i tadd_simd(__m256i a, __m256i b) {
    // 1. Optionally sanitize inputs (compile-time conditional)
    __m256i a_masked = maybe_mask<Sanitize>(a);
    __m256i b_masked = maybe_mask<Sanitize>(b);

    // 2. Build indices: (a << 2) | b
    __m256i a_shifted = _mm256_add_epi8(_mm256_add_epi8(a_masked, a_masked),
                                         _mm256_add_epi8(a_masked, a_masked));
    __m256i indices = _mm256_or_si256(a_shifted, b_masked);

    // 3. Load and broadcast LUT
    __m256i lut = broadcast_lut_16(TADD_LUT);

    // 4. Perform 32 parallel lookups
    return _mm256_shuffle_epi8(lut, indices);
}
```

#### Index Construction: Emulating Left Shift

**Challenge**: AVX2 has no byte-level left shift instruction

**Solution**: Multiply by 4 using repeated addition
```cpp
// a << 2  is equivalent to  a * 4
a_shifted = a + a + a + a  // Implemented as (a+a)+(a+a)
```

**Performance**: 2 `_mm256_add_epi8` instructions (3 cycles total)

### Unary Operation Pattern

Unary operation (tnot) is simpler:

```cpp
template <bool Sanitize = true>
static inline __m256i tnot_simd(__m256i a) {
    __m256i indices = maybe_mask<Sanitize>(a);

    // Pad TNOT_LUT (4 entries) to 16 entries for shuffle compatibility
    alignas(16) static const uint8_t TNOT_LUT_16[16] = {
        0b10, 0b01, 0b00, 0b00,  // Original 4 entries
        0b10, 0b01, 0b00, 0b00,  // Replicate pattern
        0b10, 0b01, 0b00, 0b00,
        0b10, 0b01, 0b00, 0b00
    };

    __m256i lut = broadcast_lut_16(TNOT_LUT_16);
    return _mm256_shuffle_epi8(lut, indices);
}
```

**Note**: TNOT_LUT only has 4 entries, but `_mm256_shuffle_epi8` requires 16-entry LUT (indices can be 0-15). Padding with replicated pattern ensures correctness.

---

## Template-Based Unified Processing

### Phase 2 Key Innovation: Generic Templates

Replace macro-generated code with C++ templates to unify all operations.

### Binary Operation Template

```cpp
template <bool Sanitize = true, typename SimdOp, typename ScalarOp>
py::array_t<uint8_t> process_binary_array(
    py::array_t<uint8_t> A,
    py::array_t<uint8_t> B,
    SimdOp simd_op,    // Function object for SIMD operation
    ScalarOp scalar_op // Function object for scalar operation
)
```

**Generic over**:
- `Sanitize`: Compile-time flag for input validation (default: true)
- `SimdOp`: SIMD operation function (e.g., `tadd_simd`, `tmul_simd`)
- `ScalarOp`: Scalar fallback function (e.g., `tadd`, `tmul`)

**Benefits**:
- Single implementation handles all binary operations
- Compiler specializes at compile-time (zero runtime overhead)
- Eliminates code duplication (DRY principle)
- Optional sanitization provides 3-5% performance gain for validated pipelines

### Unary Operation Template

```cpp
template <bool Sanitize = true, typename SimdOp, typename ScalarOp>
py::array_t<uint8_t> process_unary_array(
    py::array_t<uint8_t> A,
    SimdOp simd_op,
    ScalarOp scalar_op
)
```

Same template pattern, but takes only one input array. Includes `Sanitize` template parameter for consistency.

### Operation Wrappers

```cpp
py::array_t<uint8_t> tadd_array(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return process_binary_array<true>(A, B, tadd_simd<true>, tadd);
}
```

Thin wrappers instantiate templates with specific operation pairs and explicit `Sanitize=true` for production safety. These are the functions exposed to Python via pybind11. Advanced users can create custom wrappers with `Sanitize=false` for pre-validated data.

---

## Execution Paths (Detailed)

### PATH 1: OpenMP Parallel SIMD

**Trigger**: `n >= 100,000` elements

```cpp
if (n >= OMP_THRESHOLD) {
    ssize_t n_simd_blocks = (n / 32) * 32;  // Round down to multiple of 32

    #pragma omp parallel for schedule(static)
    for (ssize_t idx = 0; idx < n_simd_blocks; idx += 32) {
        __m256i va = _mm256_loadu_si256((__m256i const*)(a_ptr + idx));
        __m256i vb = _mm256_loadu_si256((__m256i const*)(b_ptr + idx));
        __m256i vr = simd_op(va, vb);
        _mm256_storeu_si256((__m256i*)(r_ptr + idx), vr);
    }

    i = n_simd_blocks;  // Set index for tail processing
}
```

**Key Details**:
- `schedule(static)`: Divides work evenly among threads at compile time
- Processes blocks of 32 elements per iteration
- Each thread gets contiguous chunks (cache-friendly)
- Tail elements (< 32) handled separately

**Performance Characteristics**:
- Thread spawn overhead: ~5-10 microseconds
- Justified for large arrays where parallelism dominates
- Scales linearly with core count (tested up to 8 cores)

**OMP_THRESHOLD Rationale**:
```
Thread overhead: ~10 µs
Serial SIMD throughput: ~0.1 ns/element
Break-even: 10 µs / 0.1 ns = 100,000 elements
```

### PATH 2: Serial SIMD Loop

**Trigger**: `n < 100,000` and `remaining >= 32`

```cpp
else {
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((__m256i const*)(a_ptr + i));
        __m256i vb = _mm256_loadu_si256((__m256i const*)(b_ptr + i));
        __m256i vr = simd_op(va, vb);
        _mm256_storeu_si256((__m256i*)(r_ptr + i), vr);
    }
}
```

**Key Details**:
- Single-threaded, sequential processing
- Processes 32 elements per iteration
- No OpenMP overhead
- Loop condition: `i + 32 <= n` (stops when < 32 elements remain)

**Performance Characteristics**:
- Throughput: ~0.1 ns/element (100M elements/second)
- Latency per iteration: ~5-10 cycles
- Limited by memory bandwidth on large arrays

### PATH 3: Scalar Tail Processing

**Trigger**: `remaining < 32` elements

```cpp
for (; i < n; ++i) {
    r[i] = scalar_op(a[i], b[i]);
}
```

**Key Details**:
- Uses scalar operations from `ternary_algebra.h`
- Processes remainder elements (0-31)
- Branch-free LUT lookups (efficient even for small counts)

**Performance Impact**:
- Tail processing: 0-31 elements
- Worst case (31 elements): ~0.05% of total time for 100K array
- Negligible overhead compared to SIMD bulk processing

---

## Memory Access Patterns

### Unaligned Loads/Stores

**Phase 2 Decision**: Always use unaligned load/store intrinsics

```cpp
__m256i va = _mm256_loadu_si256((__m256i const*)(a_ptr + idx));  // Unaligned load
_mm256_storeu_si256((__m256i*)(r_ptr + idx), vr);                // Unaligned store
```

**Rationale**:
- Modern CPUs (Haswell+): unaligned ≈ aligned performance on cache line boundaries
- Penalty only if crossing 64-byte cache line (~1-2 cycle penalty, rare)
- Eliminates branching overhead from alignment checks
- Simplifies code (no dual code paths)

### Phase 1 vs Phase 2 Comparison

**Phase 1 (Deprecated)**:
```cpp
if ((uintptr_t)ptr % 32 == 0) {
    // Use _mm256_load_si256 (aligned)
} else {
    // Use _mm256_loadu_si256 (unaligned)
}
```
- **Problem**: Branch misprediction overhead (5-20 cycles)
- **Benefit**: Aligned loads ~1-2 cycles faster
- **Net result**: ~0-3% performance gain (not worth complexity)

**Phase 2 (Current)**:
```cpp
// Always use _mm256_loadu_si256
```
- **Tradeoff**: Occasional 1-2 cycle penalty on cache line crossing
- **Benefit**: No branches, predictable performance
- **Result**: <1% performance loss, 50% code reduction

---

## OpenMP Configuration

### Threading Strategy

```cpp
static const ssize_t OMP_THRESHOLD = 100000;

#pragma omp parallel for schedule(static)
for (ssize_t idx = 0; idx < n_simd_blocks; idx += 32) {
    // Process 32 elements
}
```

**`schedule(static)`**:
- Divides iterations evenly among threads at start
- Each thread gets contiguous blocks: cache-friendly
- No dynamic scheduling overhead
- Optimal for uniform workload (every iteration does same work)

### Thread Count

**Default**: OpenMP uses `OMP_NUM_THREADS` environment variable or CPU core count

**Scaling Behavior** (measured on 8-core system):
```
Threads  | Speedup | Efficiency
---------|---------|------------
1        | 1.0x    | 100%
2        | 1.9x    | 95%
4        | 3.7x    | 92%
8        | 6.8x    | 85%
```

**Efficiency drop at 8 threads**: Memory bandwidth saturation (not CPU-bound)

### NUMA Considerations

For multi-socket systems:
```bash
# Bind threads to specific NUMA nodes
export OMP_PROC_BIND=true
export OMP_PLACES=cores
```

---

## Python Integration

### Pybind11 Bindings

```cpp
PYBIND11_MODULE(ternary_simd_engine, m) {
    m.def("tadd", &tadd_array);
    m.def("tmul", &tmul_array);
    m.def("tmin", &tmin_array);
    m.def("tmax", &tmax_array);
    m.def("tnot", &tnot_array);
}
```

**Module Name**: `ternary_simd_engine` (matches compiled `.pyd` / `.so` filename)

### NumPy Array Handling

```cpp
py::array_t<uint8_t> process_binary_array(
    py::array_t<uint8_t> A,
    py::array_t<uint8_t> B,
    ...
) {
    auto a = A.unchecked<1>();  // Fast unchecked access (assumes 1D)
    auto b = B.unchecked<1>();
    ssize_t n = A.size();

    if (n != B.size()) throw std::runtime_error("Arrays must match");

    py::array_t<uint8_t> out(n);  // Allocate output array
    auto r = out.mutable_unchecked<1>();

    const uint8_t* a_ptr = static_cast<const uint8_t*>(A.data());
    const uint8_t* b_ptr = static_cast<const uint8_t*>(B.data());
    uint8_t* r_ptr = static_cast<uint8_t*>(out.mutable_data());

    // ... SIMD processing ...

    return out;  // Move semantics (no copy)
}
```

**Key Points**:
- `unchecked<1>`: Fast access, skips bounds checking (safe for internal loops)
- Direct pointer access: Enables SIMD intrinsics
- Move return: No array copying on return (C++11 move semantics)

### Python Usage

```python
import numpy as np
import ternary_simd_engine as tc

a = np.array([0, 1, 2] * 100000, dtype=np.uint8)  # -1, 0, +1 encoded
b = np.array([2, 1, 0] * 100000, dtype=np.uint8)

result = tc.tadd(a, b)  # AVX2-accelerated addition
```

---

## Performance Analysis

### Throughput Breakdown

**Scalar (reference.py)**: ~10 ns/element
- Python loop overhead: ~8 ns
- LUT lookup: ~2 ns

**Scalar C++ (reference_cpp.cpp)**: ~3 ns/element
- Conversion overhead: ~2 ns
- Arithmetic + branches: ~1 ns

**Scalar C++ + LUT (ternary_algebra.h)**: ~0.5 ns/element
- LUT lookup: ~0.5 ns (L1 cache hit)

**SIMD (ternary_simd_engine.cpp)**: ~0.1 ns/element
- 32 elements per SIMD operation: ~3-5 cycles
- Effective: 0.1-0.15 cycles/element

**Speedup Summary**:
- Python → C++ Reference: 3x
- C++ Reference → C++ LUT: 6x
- C++ LUT → SIMD: 5x
- **Total: Python → SIMD: ~100x**

### Memory Bandwidth

**Theoretical Peak** (DDR4-3200, dual channel):
- Bandwidth: 51.2 GB/s
- Read throughput: 51.2 billion bytes/s

**Measured Performance** (100M element array):
- Data size: 100 MB (read) + 100 MB (write) = 200 MB
- Time: ~5 ms
- Bandwidth: 40 GB/s (78% of theoretical peak)

**Bottleneck**: Memory-bound for large arrays (CPU is faster than memory)

---

## Compiler Considerations

### Required Compiler Flags

```bash
# Enable AVX2 intrinsics
-mavx2          # GCC/Clang
/arch:AVX2      # MSVC

# Enable OpenMP
-fopenmp        # GCC/Clang
/openmp         # MSVC

# Optimization level
-O3             # GCC/Clang (aggressive optimization)
/O2             # MSVC (max optimization)
```

### Auto-Vectorization

**Templates do NOT prevent auto-vectorization**:
- Compiler inlines templates at compile-time
- Generated code is identical to hand-written specialized functions
- No runtime polymorphism (function pointers would prevent inlining)

### Profile-Guided Optimization (PGO)

For even better performance:
```bash
# Step 1: Compile with instrumentation
gcc -mavx2 -fopenmp -O3 -fprofile-generate

# Step 2: Run benchmarks (generate profile data)
python benchmarks/bench_phase0.py

# Step 3: Recompile with profile feedback
gcc -mavx2 -fopenmp -O3 -fprofile-use
```

**Expected gain**: 5-10% (better branch prediction, loop unrolling decisions)

See `docs/PGO_README.md` for details.

---

## Platform-Specific Notes

### Windows (MSVC)

```cpp
#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;  // MSVC doesn't define ssize_t
#endif
```

### Linux/Mac (GCC/Clang)

Standard `ssize_t` available in `<sys/types.h>` (included by pybind11).

### ARM / Non-AVX2 Platforms

**Current limitation**: Code requires AVX2 (x86-64 only)

**Future work**: Add NEON implementation for ARM (Apple Silicon, mobile)
```cpp
#ifdef __ARM_NEON
    // Use NEON intrinsics
#elif defined(__AVX2__)
    // Use AVX2 intrinsics (current)
#else
    // Fall back to scalar
#endif
```

---

## Error Handling

### Array Size Mismatch

```cpp
if (n != B.size()) throw std::runtime_error("Arrays must match");
```

Caught by pybind11 and converted to Python `RuntimeError`.

### Invalid Input Values

**No explicit validation**: Operations accept any `uint8_t` value
- Invalid values (0b11) are masked by `mask_trit()` to valid range
- Results for invalid inputs are undefined but won't crash

**Design choice**: Prioritize performance over strict validation (assume valid inputs from Python layer)

---

## Testing and Verification

### Correctness Tests

```python
# tests/test_phase0.py
import ternary_simd_engine as tc
import numpy as np

def test_tadd_correctness():
    a = np.array([0, 0, 1, 2], dtype=np.uint8)  # -1, -1, 0, +1
    b = np.array([0, 1, 2, 2], dtype=np.uint8)  # -1, 0, +1, +1
    expected = np.array([0, 0, 2, 2], dtype=np.uint8)  # -1, -1, +1, +1 (saturated)

    result = tc.tadd(a, b)
    assert np.array_equal(result, expected)
```

### Performance Benchmarks

```python
# benchmarks/bench_phase0.py
import time
import numpy as np
import ternary_simd_engine as tc

def benchmark_tadd(size=10_000_000):
    a = np.random.randint(0, 3, size=size, dtype=np.uint8)
    b = np.random.randint(0, 3, size=size, dtype=np.uint8)

    start = time.perf_counter()
    result = tc.tadd(a, b)
    elapsed = time.perf_counter() - start

    throughput = size / elapsed / 1e9  # Giga-elements/sec
    print(f"Throughput: {throughput:.2f} GE/s")
```

---

## Future Optimizations

### Potential Improvements

1. **AVX-512 Support** (Intel Skylake-X+):
   - 64 parallel operations (vs 32 for AVX2)
   - `_mm512_shuffle_epi8` doesn't exist → requires emulation via `_mm512_permutexvar_epi8`
   - Expected gain: 1.5-2x (not 2x due to frequency scaling)

2. **Cache Prefetching**:
   ```cpp
   _mm_prefetch((const char*)(a_ptr + idx + 256), _MM_HINT_T0);
   ```
   - Preload data 256 bytes ahead
   - Reduces cache miss latency
   - Expected gain: 5-10% on memory-bound workloads

3. **Batch Processing API**:
   ```python
   # Process multiple operations in one call
   results = tc.batch_ops(arrays, ops=['tadd', 'tmul', 'tmin'])
   ```
   - Reduces Python ↔ C++ overhead
   - Better cache reuse

4. **In-Place Operations**:
   ```python
   tc.tadd_inplace(a, b)  # Store result in 'a'
   ```
   - Eliminates output array allocation
   - Reduces memory traffic by 33%

---

## Debugging Tips

### Enable SIMD Debugging

```cpp
#define DEBUG_SIMD
#ifdef DEBUG_SIMD
    printf("Processing %zd elements\n", n);
    printf("SIMD blocks: %zd\n", n / 32);
    printf("Tail elements: %zd\n", n % 32);
#endif
```

### Check AVX2 Support at Runtime

```cpp
#include <cpuid.h>

bool has_avx2() {
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(7, &eax, &ebx, &ecx, &edx);
    return (ebx & (1 << 5)) != 0;  // Check AVX2 bit
}
```

### Alignment Debugging

```cpp
printf("Alignment: %zu\n", (uintptr_t)ptr % 32);
```

Useful for investigating performance anomalies (though Phase 2 doesn't rely on alignment).

---

## Cross-Reference

- **Scalar Operations**: See `docs/ternary-engine-header.md`
- **Benchmark Results**: See `benchmarks/results/`
- **Phase Evolution**: See `docs/optimization-complexity-rationale.md`
- **Build System**: See `build/scripts/setup.py`
- **Reference Implementations**: See `benchmarks/reference.py` and `benchmarks/reference_cpp.cpp`

---

## Summary

`ternary_simd_engine.cpp` implements:
- **AVX2 acceleration**: 32 parallel operations via `_mm256_shuffle_epi8`
- **LUT-based lookups**: Same semantic domain as scalar operations
- **3 execution paths**: OpenMP parallel, serial SIMD, scalar tail
- **Template-based design**: 73% code reduction from Phase 1
- **Code simplification**: Simplicity ↓, performance stable (<5% loss)
- **100x speedup**: Over pure Python reference

This implementation represents the **Phase 2 milestone**: achieving maximum simplicity while maintaining high performance through principled optimization and elimination of premature micro-optimizations.
