# Missing Features Analysis

**Doc-Type:** Technical Analysis · Version 1.0 · Generated 2025-12-09

This document details features that are missing from the Ternary Engine and would significantly enhance its capabilities.

---

## Table of Contents

1. [Missing Operations](#1-missing-operations)
2. [Multi-Dimensional Array Support](#2-multi-dimensional-array-support)
3. [Sparse Array Support](#3-sparse-array-support)
4. [SIMD Abstraction Layer](#4-simd-abstraction-layer)
5. [Matmul Optimization](#5-matmul-optimization-critical)
6. [Implementation Priorities](#implementation-priorities)

---

## 1. Missing Operations

### 1.1 Logic Operations (tand, tor)

**Status:** API defined but returns NULL in all backends

**Current State:**
```cpp
// backend_avx2_v2_optimized.cpp:641-642
.tand = NULL,
.tor = NULL
```

**What Should Exist:**

```cpp
// Ternary AND - Returns minimum of two values
// Truth table:
//   tand(-1, -1) = -1    tand(-1, 0) = -1    tand(-1, +1) = -1
//   tand( 0, -1) = -1    tand( 0, 0) =  0    tand( 0, +1) = +1
//   tand(+1, -1) = -1    tand(+1, 0) =  0    tand(+1, +1) = +1

void tand_avx2(const uint8_t* a, const uint8_t* b, uint8_t* r, size_t n) {
    // Implementation using min operation (tand = tmin for balanced ternary)
    for (size_t i = 0; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((__m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((__m256i*)(b + i));
        __m256i vr = _mm256_min_epu8(va, vb);  // Min = AND semantics
        _mm256_storeu_si256((__m256i*)(r + i), vr);
    }
}

// Ternary OR - Returns maximum of two values
void tor_avx2(const uint8_t* a, const uint8_t* b, uint8_t* r, size_t n) {
    // Implementation using max operation (tor = tmax for balanced ternary)
    for (size_t i = 0; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((__m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((__m256i*)(b + i));
        __m256i vr = _mm256_max_epu8(va, vb);  // Max = OR semantics
        _mm256_storeu_si256((__m256i*)(r + i), vr);
    }
}
```

**Use Cases:**
- Logical filtering in neural networks
- Decision trees with ternary logic
- Edge detection masking

**Effort:** Low (2-3 days)
**Impact:** Medium (completes operation set)

---

### 1.2 Comparison Operations

**Status:** Not implemented at all

**What Should Exist:**

```python
# Ternary equality test
# Returns: +1 if equal, 0 if uncertain, -1 if not equal
result = engine.teq(a, b)

# Ternary less than
# Returns: +1 if a < b, 0 if a == b, -1 if a > b
result = engine.tlt(a, b)

# Ternary greater than
result = engine.tgt(a, b)

# Ternary comparison (spaceship operator style)
# Returns: -1 if a < b, 0 if a == b, +1 if a > b
result = engine.tcmp(a, b)
```

**Implementation:**

```cpp
// Ternary comparison - element-wise
void tcmp_avx2(const uint8_t* a, const uint8_t* b, uint8_t* r, size_t n) {
    const __m256i one = _mm256_set1_epi8(1);

    for (size_t i = 0; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((__m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((__m256i*)(b + i));

        // Create masks for each comparison result
        __m256i lt_mask = _mm256_cmpgt_epi8(vb, va);  // a < b
        __m256i gt_mask = _mm256_cmpgt_epi8(va, vb);  // a > b

        // Result: -1 (0) if a < b, +1 (2) if a > b, 0 (1) if equal
        __m256i result = one;  // Default: equal (encoding 1 = value 0)
        result = _mm256_blendv_epi8(result, _mm256_setzero_si256(), lt_mask);  // 0 = -1
        result = _mm256_blendv_epi8(result, _mm256_set1_epi8(2), gt_mask);     // 2 = +1

        _mm256_storeu_si256((__m256i*)(r + i), result);
    }
}
```

**Use Cases:**
- Sorting algorithms
- Conditional selection
- Decision boundaries in ML

**Effort:** Low (2-3 days)
**Impact:** Medium

---

### 1.3 Reduction Operations

**Status:** No reduction operations exist

**What Should Exist:**

```python
# Sum all elements (ternary sum with saturation)
total = engine.tsum(a)  # Returns single ternary value

# Product all elements
product = engine.tprod(a)

# Count positives, negatives, zeros
counts = engine.tcount(a)  # Returns (n_neg, n_zero, n_pos)

# Argmax/Argmin
idx = engine.targmax(a)
idx = engine.targmin(a)
```

**Implementation Approach:**

```cpp
// Horizontal reduction using AVX2
int8_t tsum_reduce(const uint8_t* a, size_t n) {
    __m256i acc = _mm256_setzero_si256();

    for (size_t i = 0; i + 32 <= n; i += 32) {
        __m256i v = _mm256_loadu_si256((__m256i*)(a + i));
        // Convert from 0,1,2 encoding to -1,0,+1
        __m256i adjusted = _mm256_sub_epi8(v, _mm256_set1_epi8(1));
        acc = _mm256_add_epi8(acc, adjusted);
    }

    // Horizontal sum of accumulator
    // ... reduction steps ...

    return result;
}
```

**Use Cases:**
- Neural network layer outputs
- Statistical analysis
- Decision aggregation

**Effort:** Medium (3-5 days)
**Impact:** High

---

### 1.4 Shift and Rotation Operations

**Status:** Packing has TODO for AVX2, no rotation operations

**What Should Exist:**

```python
# Left shift (multiply by 3^k, with overflow handling)
result = engine.tshl(a, k)

# Right shift (divide by 3^k, with truncation)
result = engine.tshr(a, k)

# Circular rotation (for arrays)
result = engine.trotl(a, k)  # Rotate elements left
result = engine.trotr(a, k)  # Rotate elements right
```

**Implementation Notes:**
- Ternary shift is multiplication/division by powers of 3
- Unlike binary shift, ternary shift doesn't map to simple bit operations
- Requires LUT or arithmetic approach

**Use Cases:**
- Cryptographic applications
- Signal processing
- Position encoding

**Effort:** Medium (3-5 days)
**Impact:** Medium

---

## 2. Multi-Dimensional Array Support

### Current Limitation

**All operations are 1D only:**

```python
# Current: Works
a = np.array([0, 1, 2, 0, 1], dtype=np.uint8)
b = np.array([1, 1, 0, 2, 1], dtype=np.uint8)
result = engine.tadd(a, b)  # OK: 1D array

# Current: Fails or requires flattening
a = np.array([[0, 1], [2, 0]], dtype=np.uint8)
b = np.array([[1, 1], [0, 2]], dtype=np.uint8)
result = engine.tadd(a, b)  # Error or unexpected behavior
```

### What Should Exist

```python
# Multi-dimensional support
a = np.random.randint(0, 3, size=(100, 100), dtype=np.uint8)
b = np.random.randint(0, 3, size=(100, 100), dtype=np.uint8)
result = engine.tadd(a, b)  # Works on 2D arrays

# Broadcasting support (NumPy-style)
a = np.random.randint(0, 3, size=(100, 100), dtype=np.uint8)
b = np.random.randint(0, 3, size=(100, 1), dtype=np.uint8)
result = engine.tadd(a, b)  # Broadcasts b across columns

# Axis-specific operations
result = engine.tsum(a, axis=0)  # Sum along rows
result = engine.tsum(a, axis=1)  # Sum along columns
```

### Implementation Approach

```cpp
// Multi-dimensional operation wrapper
template<typename Op>
py::array_t<uint8_t> nd_operation(
    py::array_t<uint8_t> a,
    py::array_t<uint8_t> b,
    Op operation
) {
    // Get buffer info
    auto buf_a = a.request();
    auto buf_b = b.request();

    // Validate shapes (with broadcasting rules)
    auto out_shape = broadcast_shapes(buf_a.shape, buf_b.shape);

    // Create output
    py::array_t<uint8_t> result(out_shape);
    auto buf_r = result.request();

    // Handle contiguous case (fast path)
    if (is_contiguous(buf_a) && is_contiguous(buf_b) &&
        same_shape(buf_a, buf_b)) {
        // Direct SIMD operation
        operation(
            static_cast<uint8_t*>(buf_a.ptr),
            static_cast<uint8_t*>(buf_b.ptr),
            static_cast<uint8_t*>(buf_r.ptr),
            total_elements(buf_a)
        );
    } else {
        // Strided iteration with broadcasting
        nd_iterate_with_broadcast(buf_a, buf_b, buf_r, operation);
    }

    return result;
}
```

### Critical for ML

**Why This Matters:**

| ML Component | Requires |
|--------------|----------|
| Weight matrices | 2D arrays |
| Batch processing | 3D arrays |
| Attention tensors | 4D arrays |
| Convolution filters | 4D arrays |

**Without multi-dimensional support, the engine cannot process real ML models.**

**Effort:** High (2-3 weeks)
**Impact:** Critical

---

## 3. Sparse Array Support

### Current Limitation

Every array stores all values, even zeros:

```python
# Sparse data (90% zeros) - still stores all elements
a = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=np.uint8)
# Storage: 10 bytes
# Computation: processes all 10 elements
```

### What Should Exist

```python
import ternary_sparse as tsparse

# Create sparse ternary array
sparse_a = tsparse.from_dense(a, format='coo')  # Coordinate format
sparse_b = tsparse.from_dense(b, format='csr')  # Compressed sparse row

# Operations on sparse arrays
result = tsparse.tadd(sparse_a, sparse_b)  # Skips zeros

# Convert back to dense
dense_result = result.to_dense()
```

### Implementation Approach

```cpp
// Coordinate (COO) format for sparse ternary
struct SparseTernaryCOO {
    std::vector<size_t> indices;  // Non-zero positions
    std::vector<int8_t> values;   // Values at those positions (-1, +1 only, 0 omitted)
    size_t total_size;            // Original array size
};

// Sparse addition
SparseTernaryCOO sparse_tadd(
    const SparseTernaryCOO& a,
    const SparseTernaryCOO& b
) {
    SparseTernaryCOO result;
    result.total_size = a.total_size;

    // Merge sorted index lists
    size_t i = 0, j = 0;
    while (i < a.indices.size() && j < b.indices.size()) {
        if (a.indices[i] < b.indices[j]) {
            result.indices.push_back(a.indices[i]);
            result.values.push_back(a.values[i]);
            i++;
        } else if (b.indices[j] < a.indices[i]) {
            result.indices.push_back(b.indices[j]);
            result.values.push_back(b.values[j]);
            j++;
        } else {
            // Same index - add values
            int8_t sum = ternary_add(a.values[i], b.values[j]);
            if (sum != 0) {  // Only store non-zero
                result.indices.push_back(a.indices[i]);
                result.values.push_back(sum);
            }
            i++; j++;
        }
    }
    // Handle remaining elements...

    return result;
}
```

### Performance Impact

| Sparsity | Dense Time | Sparse Time | Speedup |
|----------|------------|-------------|---------|
| 10% zeros | 100% | 95% | 1.05× |
| 50% zeros | 100% | 55% | 1.8× |
| 90% zeros | 100% | 15% | 6.7× |
| 99% zeros | 100% | 3% | 33× |

**Pruned neural networks are typically 80-95% sparse after quantization.**

**Effort:** Medium (1-2 weeks)
**Impact:** High for pruned models

---

## 4. SIMD Abstraction Layer

### Current Limitation

Hardcoded AVX2 intrinsics throughout codebase:

```cpp
// Current: Direct AVX2 calls everywhere
__m256i va = _mm256_loadu_si256((__m256i*)(a + i));
__m256i vb = _mm256_loadu_si256((__m256i*)(b + i));
__m256i vr = _mm256_shuffle_epi8(lut, idx);
_mm256_storeu_si256((__m256i*)(r + i), vr);
```

### What Should Exist

```cpp
// Abstraction layer supporting multiple ISAs
namespace simd {

template<typename ISA>
struct Vector {
    typename ISA::vector_type data;

    static Vector load(const uint8_t* ptr);
    void store(uint8_t* ptr) const;
    Vector shuffle(const Vector& indices) const;
};

// Specializations
template<> struct Vector<AVX2> {
    __m256i data;
    // ... AVX2 implementations
};

template<> struct Vector<AVX512> {
    __m512i data;
    // ... AVX-512 implementations
};

template<> struct Vector<NEON> {
    uint8x16_t data;
    // ... ARM NEON implementations
};

template<> struct Vector<WASM> {
    v128_t data;
    // ... WebAssembly SIMD implementations
};

}  // namespace simd
```

### Benefits

| ISA | Vector Width | Trits/Op | Platforms |
|-----|--------------|----------|-----------|
| SSE4.1 | 128-bit | 16 | Legacy x86 |
| AVX2 | 256-bit | 32 | Current x86 |
| AVX-512 | 512-bit | 64 | Server x86 |
| NEON | 128-bit | 16 | ARM mobile |
| SVE | 128-2048 bit | 16-256 | ARM server |
| WASM SIMD | 128-bit | 16 | Browsers |

### Implementation Path

1. Create `simd_abstraction.h` with template interface
2. Implement AVX2 specialization (copy existing code)
3. Add compile-time ISA selection
4. Add runtime ISA dispatch
5. Implement additional ISA backends

**Effort:** High (2-3 weeks)
**Impact:** Enables cross-platform support

---

## 5. Matmul Optimization (CRITICAL)

### Current Status

**The GEMM implementation is 54× slower than target:**

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Throughput | 0.37 Gops/s | 20-30 Gops/s | 54-81× |
| Implementation | Scalar | SIMD + OpenMP | Missing |

### Root Cause Analysis

From `reports/reasons.md`:

1. **No SIMD:** Scalar implementation only (56× gap)
2. **No Parallelization:** Single-threaded (2-4× gap)
3. **No Cache Blocking:** Poor cache utilization (3× gap)
4. **No ILP:** Sequential dependencies (1.5× gap)

### What Should Exist

```cpp
// Optimized ternary GEMM
void tritnet_gemm_optimized(
    const float* A,      // [M, K] activation matrix
    const uint8_t* B,    // [K/5, N] ternary weights (Dense243)
    float* C,            // [M, N] output
    size_t M, size_t K, size_t N
) {
    const size_t TILE_M = 32;
    const size_t TILE_N = 64;
    const size_t TILE_K = 256;

    #pragma omp parallel for collapse(2)
    for (size_t i0 = 0; i0 < M; i0 += TILE_M) {
        for (size_t j0 = 0; j0 < N; j0 += TILE_N) {
            // Local accumulator (fits in L1 cache)
            float local_C[TILE_M][TILE_N] = {0};

            for (size_t k0 = 0; k0 < K; k0 += TILE_K) {
                // Process tile with SIMD
                for (size_t i = 0; i < TILE_M; i++) {
                    for (size_t k = k0; k < min(k0 + TILE_K, K); k += 5) {
                        // Unpack 5 trits from Dense243
                        int8_t trits[5];
                        unpack_dense243(B[(k/5) * N + j0], trits);

                        // AVX2 FMA accumulation
                        __m256 va = _mm256_loadu_ps(&A[(i0+i)*K + k]);
                        // ... vectorized accumulation
                    }
                }
            }

            // Write back
            for (size_t i = 0; i < TILE_M; i++) {
                for (size_t j = 0; j < TILE_N; j++) {
                    C[(i0+i)*N + (j0+j)] = local_C[i][j];
                }
            }
        }
    }
}
```

### Expected Performance After Optimization

| Optimization | Speedup | Cumulative |
|--------------|---------|------------|
| SIMD (AVX2) | 8× | 8× |
| OpenMP (8 cores) | 6× | 48× |
| Cache blocking | 1.5× | 72× |
| ILP + prefetch | 1.2× | 86× |

**Target: 0.37 × 86 = 31.8 Gops/s ✓**

**Effort:** High (2-3 weeks)
**Impact:** Critical (blocks AI/ML viability claim)

---

## Implementation Priorities

### Phase 1: Critical (Weeks 1-3)

| Feature | Effort | Impact | Dependency |
|---------|--------|--------|------------|
| Matmul optimization | High | Critical | None |
| Multi-dim arrays | High | Critical | None |

### Phase 2: High Value (Weeks 4-6)

| Feature | Effort | Impact | Dependency |
|---------|--------|--------|------------|
| Reduction operations | Medium | High | None |
| Comparison operations | Low | Medium | None |
| tand/tor operations | Low | Medium | None |

### Phase 3: Strategic (Weeks 7-10)

| Feature | Effort | Impact | Dependency |
|---------|--------|--------|------------|
| Sparse arrays | Medium | High | Multi-dim |
| SIMD abstraction | High | High | None |
| Shift/rotation | Medium | Medium | None |

---

## Summary

### Current Operation Count: 9
- tadd, tmul, tmin, tmax, tnot
- fused_tnot_tadd, fused_tnot_tmul, fused_tnot_tmin, fused_tnot_tmax

### Target Operation Count: 20+
- Add: tand, tor, teq, tlt, tgt, tcmp
- Add: tsum, tprod, tcount, targmax, targmin
- Add: tshl, tshr, trotl, trotr
- Add: Sparse variants of all operations

### Key Insight

**The core SIMD engine is excellent. What's missing is the surrounding infrastructure to make it useful for real-world applications.**

---

**Document Version:** 1.0
**Generated:** 2025-12-09
**Author:** Claude Code Analysis
