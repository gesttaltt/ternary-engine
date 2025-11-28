# Ternary Engine Feature List

**Doc-Type:** Technical Reference · Version 1.0 · Updated 2025-11-27

Comprehensive catalog of all features, optimizations, and abstractions in the Ternary Engine project.

---

## Table of Contents

1. [Core Operations](#1-core-operations)
2. [Encoding Formats](#2-encoding-formats)
3. [SIMD Implementations](#3-simd-implementations)
4. [LUT System](#4-lut-system)
5. [Fusion Operations](#5-fusion-operations)
6. [Low-Level Optimizations](#6-low-level-optimizations)
7. [Backend Architecture](#7-backend-architecture)
8. [Build System](#8-build-system)
9. [Benchmarking Suite](#9-benchmarking-suite)
10. [TritNet Neural Operations](#10-tritnet-neural-operations)
11. [Python Bindings](#11-python-bindings)
12. [Platform Support](#12-platform-support)

---

## 1. Core Operations

### 1.1 Arithmetic Operations

| Operation | Symbol | Formula | Description |
|-----------|--------|---------|-------------|
| `tadd` | + | clamp(a + b, -1, +1) | Saturated ternary addition |
| `tmul` | × | a × b | Ternary multiplication |
| `tnot` | - | -a | Ternary negation (sign flip) |
| `tmin` | min | min(a, b) | Ternary minimum |
| `tmax` | max | max(a, b) | Ternary maximum |

**Files:**
- `src/core/algebra/ternary_algebra.h` - Scalar operations
- `src/core/simd/simd_avx2_32trit_ops.h` - SIMD operations

### 1.2 Trit Encoding

**2-bit encoding (standard):**
- `0b00` = -1 (MINUS_ONE)
- `0b01` = 0 (ZERO)
- `0b10` = +1 (PLUS_ONE)
- `0b11` = invalid (sanitized to 0)

**File:** `src/core/algebra/ternary_algebra.h:23`

### 1.3 Packing Functions

| Function | Description |
|----------|-------------|
| `pack_trits(t0,t1,t2,t3)` | Pack 4 trits into 1 byte |
| `unpack_trit(pack,idx)` | Extract trit at index from packed byte |

---

## 2. Encoding Formats

### 2.1 Standard 2-bit Encoding

- **Density:** 4 trits/byte (25% utilization, 16/256 states)
- **Status:** Production-ready
- **Use case:** All core operations

### 2.2 Dense243 (T5) Encoding

- **Density:** 5 trits/byte (95.3% utilization, 243/256 states)
- **Formula:** `b = Σ(tᵢ + 1) × 3ⁱ` for i ∈ [0,4]
- **Memory reduction:** 20% vs standard encoding
- **Status:** Implementation complete, module build pending

**Files:**
- `src/engine/dense243/ternary_dense243.h` - Core encoding
- `src/engine/dense243/ternary_dense243_simd.h` - SIMD extraction
- `docs/specifications/dense243_encoding_spec.md` - Specification

### 2.3 TriadSextet Encoding

- **Density:** 6 trits/byte (theoretical maximum)
- **Status:** Experimental
- **File:** `src/engine/dense243/ternary_triadsextet.h`

### 2.4 Octet/Sixtet Packing

- **Purpose:** Alternative packing formats
- **Files:**
  - `src/core/packing/octet_pack.h`
  - `src/core/packing/sixtet_pack.h`

---

## 3. SIMD Implementations

### 3.1 AVX2 v1 (Legacy)

- **File:** `src/core/simd/ternary_backend_avx2_v1.cpp`
- **Features:** Basic AVX2 with traditional indexing
- **Index formula:** `idx = (a << 2) | b`
- **Status:** Superseded by v2

### 3.2 AVX2 v2 (Production)

- **File:** `src/core/simd/ternary_backend_avx2_v2.cpp`
- **Version:** 1.3.0
- **Width:** 32 trits/operation (256-bit registers)

**Features:**
| Feature | Description | Improvement |
|---------|-------------|-------------|
| Canonical indexing | `idx = (a*3)+b` via dual-shuffle | 12-18% |
| OpenMP parallelization | Multi-threaded for large arrays | Variable |
| Prefetch optimization | Hide memory latency | 5-10% |
| Streaming stores | Reduce cache pollution | 10-20% on large arrays |
| Fusion operations | Fused binary+unary ops | 1.5-4× |

**Capabilities:**
```cpp
TERNARY_CAP_SIMD_256 | TERNARY_CAP_CANONICAL | TERNARY_CAP_FUSION
```

### 3.3 Scalar Backend (Fallback)

- **File:** `src/core/simd/ternary_backend_scalar.cpp`
- **Purpose:** Non-SIMD fallback for compatibility
- **Status:** Active

### 3.4 SIMD Kernels (Standalone)

- **File:** `src/core/simd/ternary_simd_kernels.h`
- **Purpose:** Pure SIMD kernels without pybind11 dependency
- **Use case:** Benchmarks, standalone C++ applications

---

## 4. LUT System

### 4.1 Compile-Time LUT Generation

**File:** `src/core/algebra/ternary_lut_gen.h`

**Generators:**
| Function | Output Size | Description |
|----------|-------------|-------------|
| `make_binary_lut(op)` | 16 entries | Binary operations (a,b) → result |
| `make_unary_lut(op)` | 4 entries | Unary operations a → result |
| `make_unary_lut_padded(op)` | 16 entries | SIMD-compatible unary LUT |

**Benefits:**
- Single source of truth (algorithm = documentation)
- Zero runtime overhead (constexpr)
- Compile-time verification via static_assert

### 4.2 Standard LUTs

| LUT | Size | Index Formula | Description |
|-----|------|---------------|-------------|
| `TADD_LUT` | 16 | (a<<2)\|b | Saturated addition |
| `TMUL_LUT` | 16 | (a<<2)\|b | Multiplication |
| `TMIN_LUT` | 16 | (a<<2)\|b | Minimum |
| `TMAX_LUT` | 16 | (a<<2)\|b | Maximum |
| `TNOT_LUT` | 4 | a | Negation (scalar) |
| `TNOT_LUT_SIMD` | 16 | a | Negation (SIMD-padded) |

### 4.3 Canonical LUTs

**File:** `src/core/algebra/ternary_canonical_lut.h`

| LUT | Description |
|-----|-------------|
| `TADD_CANONICAL_LUT` | Addition with canonical indexing |
| `TMUL_CANONICAL_LUT` | Multiplication with canonical indexing |
| `TMIN_CANONICAL_LUT` | Minimum with canonical indexing |
| `TMAX_CANONICAL_LUT` | Maximum with canonical indexing |
| `TNOT_CANONICAL_LUT` | Negation with canonical indexing |

### 4.4 Dense243 Extraction LUTs

**File:** `src/engine/dense243/ternary_dense243.h`

| LUT | Description |
|-----|-------------|
| `DENSE243_EXTRACT_T0_LUT` | Extract trit 0 from packed byte |
| `DENSE243_EXTRACT_T1_LUT` | Extract trit 1 from packed byte |
| `DENSE243_EXTRACT_T2_LUT` | Extract trit 2 from packed byte |
| `DENSE243_EXTRACT_T3_LUT` | Extract trit 3 from packed byte |
| `DENSE243_EXTRACT_T4_LUT` | Extract trit 4 from packed byte |

### 4.5 LUT Broadcasting

**Function:** `broadcast_lut_16()`

Broadcasts 16-byte LUT to both lanes of 256-bit AVX2 register:
```cpp
__m128i lut_128 = _mm_loadu_si128((const __m128i*)lut);
return _mm256_broadcastsi128_si256(lut_128);
```

**File:** `src/core/simd/ternary_simd_kernels.h:30`

---

## 5. Fusion Operations

### 5.1 Phase 4.1 Fused Operations

**File:** `src/core/simd/ternary_fusion.h`

| Operation | Pattern | Avg Speedup | Status |
|-----------|---------|-------------|--------|
| `fused_tnot_tadd` | tnot(tadd(a,b)) | 1.76× | Validated |
| `fused_tnot_tmul` | tnot(tmul(a,b)) | 1.71× | Validated |
| `fused_tnot_tmin` | tnot(tmin(a,b)) | 4.06× | Validated |
| `fused_tnot_tmax` | tnot(tmax(a,b)) | 3.68× | Validated |

**Memory Traffic Reduction:**
- Unfused: 5N bytes (read A, read B, write temp, read temp, write result)
- Fused: 3N bytes (read A, read B, write result)
- Reduction: 40%

### 5.2 Implementation Pattern

```cpp
template <bool Sanitize = true>
static inline __m256i fused_tnot_tadd_simd(__m256i a, __m256i b) {
    __m256i temp = tadd_simd<Sanitize>(a, b);
    return tnot_simd<Sanitize>(temp);  // temp stays in register
}
```

---

## 6. Low-Level Optimizations

### 6.1 Canonical Indexing

**File:** `src/core/simd/ternary_canonical_index.h`

**Traditional:** `idx = (a << 2) | b` (shift + OR, dependent chain)
**Canonical:** `idx = (a * 3) + b` via dual-shuffle + ADD

**Implementation:**
```cpp
__m256i contrib_a = _mm256_shuffle_epi8(CANON_A_LUT, a);  // a * 3
__m256i contrib_b = _mm256_shuffle_epi8(CANON_B_LUT, b);  // b
__m256i indices = _mm256_add_epi8(contrib_a, contrib_b);  // combine
```

**Improvement:** 12-18%

### 6.2 Dual-Shuffle XOR

**File:** `src/core/simd/ternary_dual_shuffle.h`

**Concept:** Decompose operation into XOR-composable components
```cpp
result = LUT_A(a) XOR LUT_B(b)
```

**Benefits:**
- Two shuffles run in parallel (different data dependencies)
- XOR runs on Port 0 while shuffles run on Port 5 (Intel) / Port 3 (AMD)

**Status:** Experimental (LUTs generated, integration pending)

### 6.3 Optimization Thresholds

**File:** `src/core/config/optimization_config.h`

| Constant | Default | Description |
|----------|---------|-------------|
| `OMP_THRESHOLD` | 32K × cores | OpenMP parallelization threshold |
| `STREAM_THRESHOLD` | 1,000,000 | Non-temporal store threshold |
| `PREFETCH_DIST` | 512 | Prefetch stride in bytes |

### 6.4 Memory Optimizations

| Optimization | Trigger | Benefit |
|--------------|---------|---------|
| Prefetch | Always | Hide memory latency |
| Streaming stores | n ≥ 1M elements | Reduce cache pollution |
| 32-byte alignment check | Before streaming | Ensure safe stores |
| Memory fence | After streaming | Ensure consistency |

### 6.5 Sanitization Control

**Compile flag:** `-DTERNARY_NO_SANITIZE`

**Effect:** Disable input masking for validated data pipelines
**Benefit:** 3-5% performance gain

```cpp
template <bool Sanitize = true>
static inline __m256i maybe_mask(__m256i v) {
    if constexpr (Sanitize)
        return _mm256_and_si256(v, _mm256_set1_epi8(0x03));
    else
        return v;
}
```

### 6.6 Force Inline

**Macro:** `FORCE_INLINE`
```cpp
#ifdef _MSC_VER
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE __attribute__((always_inline)) inline
#endif
```

---

## 7. Backend Architecture

### 7.1 Backend Interface

**File:** `src/core/simd/ternary_backend_interface.h`

**Structure:**
```cpp
typedef struct TernaryBackend {
    TernaryBackendInfo info;

    // Core operations
    void (*tnot)(uint8_t* dst, const uint8_t* src, size_t n);
    void (*tadd)(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n);
    void (*tmul)(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n);
    void (*tmax)(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n);
    void (*tmin)(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n);

    // Fusion operations
    void (*fused_tnot_tadd)(...);
    void (*fused_tnot_tmul)(...);
    void (*fused_tnot_tmin)(...);
    void (*fused_tnot_tmax)(...);
} TernaryBackend;
```

### 7.2 Backend Registration

**File:** `src/core/simd/ternary_backend_dispatch.cpp`

**Available backends:**
| Backend | Priority | Capabilities |
|---------|----------|--------------|
| AVX2_v2 | High | SIMD_256, Canonical, Fusion |
| AVX2_v1 | Medium | SIMD_256 |
| Scalar | Low | Baseline |

### 7.3 CPU Detection

**File:** `src/core/simd/ternary_cpu_detect.h`

**Functions:**
| Function | Description |
|----------|-------------|
| `has_avx2()` | Check AVX2 support |
| `has_avx512()` | Check AVX-512 support |

---

## 8. Build System

### 8.1 Build Scripts

| Script | Purpose |
|--------|---------|
| `build/build.py` | Standard optimized build |
| `build/build_dense243.py` | Dense243 module build |
| `build/build_pgo_unified.py` | PGO build (Clang) |
| `build/build_pgo.py` | PGO build (MSVC) |
| `build/clean_all.py` | Comprehensive cleanup |

### 8.2 Compiler Flags

**MSVC:**
```
/O2 /GL /arch:AVX2 /std:c++17 /LTCG /openmp
```

**GCC/Clang:**
```
-O3 -march=native -mavx2 -flto -std=c++17 -fopenmp
```

### 8.3 Profile-Guided Optimization

**Documentation:** `docs/pgo/README.md`

**Process:**
1. Instrumented build
2. Training run with representative workload
3. Optimized build using profile data

**Expected gain:** 5-15%

---

## 9. Benchmarking Suite

### 9.1 Active Benchmarks

| Script | Target | Description |
|--------|--------|-------------|
| `bench_simd_core_ops.py` | `ternary_simd_engine` | Core operations throughput |
| `bench_simd_fusion_ops.py` | `ternary_simd_engine` | Fusion speedup measurement |
| `bench_competitive.py` | Multiple | 6-phase competitive analysis |
| `bench_dense243.py` | `ternary_dense243_module` | Dense243 performance |
| `bench_input_characteristics.py` | `ternary_simd_engine` | Performance vs data patterns |
| `bench_regression_detect.py` | JSON files | Regression detection |
| `bench_model_quantization.py` | PyTorch models | LLM quantization |
| `bench_power_efficiency.py` | `ternary_simd_engine` | Energy efficiency |

### 9.2 Macro Benchmarks

| Script | Workload |
|--------|----------|
| `macro/bench_layer_forward.py` | Neural layer forward pass |
| `macro/bench_image_pipeline.py` | Image processing pipeline |

### 9.3 Validated Performance

| Metric | Value | Platform |
|--------|-------|----------|
| Peak throughput | 35,042 Mops/s | Windows x64 |
| Avg vs Python | 8,234× | Windows x64 |
| Fusion speedup | 1.5-4× | Windows x64 |

---

## 10. TritNet Neural Operations

### 10.1 Vision

Replace memory-bound LUT operations with compute-bound neural networks for GPU/TPU acceleration.

**Documentation:** `docs/TRITNET_VISION.md`

### 10.2 Architecture

| Model | Input | Hidden | Output | Purpose |
|-------|-------|--------|--------|---------|
| TritNetUnary | 5 trits | 8 neurons | 5 trits | Unary ops (tnot) |
| TritNetBinary | 10 trits | 16 neurons | 5 trits | Binary ops (tadd, tmul, etc.) |

### 10.3 Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | Complete | Truth table generation (243/59,049 samples) |
| 2A | In Progress | Proof-of-concept (tnot to 100%) |
| 2B | Pending | Scale to all operations |
| 3 | Pending | C++ integration |
| 4 | Pending | GPU acceleration |
| 5 | Pending | Learned generalization |

**Files:**
- `models/tritnet/src/` - Training code
- `datasets/tritnet/` - Truth table datasets

### 10.4 GEMM Operations

**File:** `src/engine/bindings_tritnet_gemm.cpp`

**Status:** Prototype (module not built)

---

## 11. Python Bindings

### 11.1 Modules

| Module | File | Description |
|--------|------|-------------|
| `ternary_simd_engine` | `bindings_core_ops.cpp` | Core SIMD operations |
| `ternary_dense243_module` | `bindings_dense243.cpp` | Dense243 encoding |
| `ternary_tritnet_gemm` | `bindings_tritnet_gemm.cpp` | TritNet GEMM (prototype) |

### 11.2 API (ternary_simd_engine)

```python
import ternary_simd_engine as tse

# Core operations (return numpy arrays)
result = tse.tadd(a, b)
result = tse.tmul(a, b)
result = tse.tmin(a, b)
result = tse.tmax(a, b)
result = tse.tnot(a)

# Fusion operations
result = tse.fused_tnot_tadd(a, b)
result = tse.fused_tnot_tmul(a, b)
result = tse.fused_tnot_tmin(a, b)
result = tse.fused_tnot_tmax(a, b)
```

---

## 12. Platform Support

### 12.1 Production Ready

| Platform | Status | Tests |
|----------|--------|-------|
| Windows x64 (MSVC) | Validated | 65/65 passing |

### 12.2 Experimental

| Platform | Status | Notes |
|----------|--------|-------|
| Linux (GCC) | Untested | Build scripts provided |
| macOS (Clang) | Untested | Build scripts provided |

### 12.3 CPU Requirements

**Minimum:** x86-64 with AVX2
- Intel Haswell (2013+)
- AMD Excavator (2015+)

**Runtime detection:** Graceful fallback to scalar if AVX2 unavailable

### 12.4 Future Targets

| Platform | Status |
|----------|--------|
| ARM NEON/SVE | Planned |
| AVX-512 | Planned |
| WebAssembly SIMD | Planned |

---

## Quick Reference: File Locations

```
src/
├── core/
│   ├── algebra/
│   │   ├── ternary_algebra.h      # Scalar operations + LUTs
│   │   ├── ternary_lut_gen.h      # Compile-time LUT generation
│   │   └── ternary_canonical_lut.h # Canonical LUTs
│   ├── simd/
│   │   ├── ternary_simd_kernels.h      # Standalone SIMD kernels
│   │   ├── ternary_backend_avx2_v2.cpp # Production AVX2 backend
│   │   ├── ternary_fusion.h            # Fusion operations
│   │   ├── ternary_canonical_index.h   # Canonical indexing
│   │   ├── ternary_dual_shuffle.h      # Dual-shuffle XOR (experimental)
│   │   └── ternary_cpu_detect.h        # CPU feature detection
│   ├── config/
│   │   └── optimization_config.h  # Tuning constants
│   └── packing/
│       ├── octet_pack.h           # 8-trit packing
│       └── sixtet_pack.h          # 6-trit packing
└── engine/
    ├── bindings_core_ops.cpp      # Python module: ternary_simd_engine
    ├── bindings_dense243.cpp      # Python module: ternary_dense243_module
    └── dense243/
        ├── ternary_dense243.h     # Dense243 encoding
        └── ternary_dense243_simd.h # SIMD extraction
```

---

## Related Documentation

- [Architecture Overview](architecture/architecture.md)
- [Optimization Roadmap](architecture/optimization-roadmap.md)
- [Dense243 Specification](t5-dense243-spec.md)
- [TritNet Vision](TRITNET_VISION.md)
- [Build System Guide](build-system/README.md)
- [API Reference](api-reference/headers.md)

---

**Version:** 1.0 · **Updated:** 2025-11-27 · **Author:** Ternary Engine Team
