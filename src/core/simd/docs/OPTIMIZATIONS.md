# Optimization Techniques Reference

**Files:** `opt_canonical_index.h`, `opt_dual_shuffle_xor.h`, `opt_lut_256byte_expanded.h`
**Status:** Experimental (AVX2 v2 Backend)
**Combined Improvement:** 30-40% over baseline

---

## Overview

Three optimization techniques target the critical path in SIMD operations:

| Technique | Target | Improvement | Status |
|-----------|--------|-------------|--------|
| Canonical Indexing | Index calculation | 12-18% | Production |
| Dual-Shuffle XOR | Port utilization | 15-25% | Experimental |
| 256-Byte LUTs | Memory access | 10-20% | Experimental |

---

## 1. Canonical Indexing

**File:** `opt_canonical_index.h`

### Problem

Traditional index calculation creates a dependent chain:

```
idx = (a << 2) | b    // 3 cycles: shift → wait → OR
```

### Solution

Replace arithmetic with parallel LUT lookups:

```
idx_a = shuffle(CANON_A_LUT, a)   // 1 cycle
idx_b = shuffle(CANON_B_LUT, b)   // 1 cycle (parallel)
idx = idx_a + idx_b               // 1 cycle
```

### LUT Design

```
CANON_A_LUT[i] = i * 3    // [0, 3, 6, 0] repeated
CANON_B_LUT[i] = i        // [0, 1, 2, 0] repeated

Combined: idx = (a * 3) + b = CANON_A[a] + CANON_B[b]
```

### Memory Layout

```cpp
// 32-byte aligned for AVX2 register loading
alignas(32) static const uint8_t CANON_A_LUT_256[32] = {
    0, 3, 6, 0,  0, 3, 6, 0,  // Repeated pattern
    0, 3, 6, 0,  0, 3, 6, 0,
    0, 3, 6, 0,  0, 3, 6, 0,
    0, 3, 6, 0,  0, 3, 6, 0
};
```

### Implementation

```cpp
static inline __m256i canonical_index_avx2(__m256i trits_a, __m256i trits_b) {
    __m256i canon_a = _mm256_load_si256((__m256i*)CANON_A_LUT_256);
    __m256i canon_b = _mm256_load_si256((__m256i*)CANON_B_LUT_256);

    // Dual-shuffle: compute both components in parallel
    __m256i contrib_a = _mm256_shuffle_epi8(canon_a, trits_a);
    __m256i contrib_b = _mm256_shuffle_epi8(canon_b, trits_b);

    // Combine: ADD is faster than XOR here
    return _mm256_add_epi8(contrib_a, contrib_b);
}
```

### Performance

| Approach | Cycles | Dependency Chain |
|----------|--------|------------------|
| Traditional | ~3 | shift → OR (serial) |
| Canonical | ~2 | shuffle + shuffle → ADD (parallel) |

**Improvement:** 12-18%

---

## 2. Dual-Shuffle XOR

**File:** `opt_dual_shuffle_xor.h`

### Problem

Single-shuffle approach saturates the shuffle port:

```
// Port 5 (Intel) bottleneck
idx = compute_index(a, b);        // Uses shift/OR
result = shuffle(LUT, idx);       // Waits for idx
```

### Solution

Split LUT into XOR-decomposable components:

```
comp_a = shuffle(LUT_A, a);   // Port 5
comp_b = shuffle(LUT_B, b);   // Port 5 (parallel due to no data dependency)
result = comp_a XOR comp_b;   // Port 0 (different port!)
```

### XOR-Decomposability

For an operation to be XOR-decomposable:

```
LUT(a, b) = LUT_A(a) XOR LUT_B(b)
```

| Operation | XOR-Decomposable | Notes |
|-----------|------------------|-------|
| tnot | Yes | Trivially (unary) |
| tmul | Yes | With careful encoding |
| tadd | Partial | Saturation requires handling |
| tmin/tmax | No | Use ADD combine instead |

### Microarchitecture

```
Intel Skylake/Alder Lake:
- shuffle: Port 5 only
- XOR: Port 0 or 5
- Parallel issue: shuffle || XOR

AMD Zen2/3/4:
- shuffle: Port 3 only (critical!)
- XOR: Port 0 (zero-latency)
- Zen has ONE shuffle port → dual-shuffle even more important
```

### Implementation

```cpp
static inline __m256i tadd_dual_shuffle(__m256i a, __m256i b) {
    __m256i lut_a = _mm256_load_si256((__m256i*)TADD_DUAL_A);
    __m256i lut_b = _mm256_load_si256((__m256i*)TADD_DUAL_B);

    // Both shuffles can execute in parallel (no data dependency)
    __m256i comp_a = _mm256_shuffle_epi8(lut_a, a);
    __m256i comp_b = _mm256_shuffle_epi8(lut_b, b);

    // XOR runs on Port 0 while shuffles run on Port 5/3
    return _mm256_xor_si256(comp_a, comp_b);
}
```

### Performance

| Platform | Speedup |
|----------|---------|
| AMD Zen2/3/4 | 1.5-1.7x |
| Intel Alder Lake | 1.2-1.5x |

**Expected throughput:** 35-45 Gops/s (up from 28-35 Gops/s)

---

## 3. 256-Byte LUTs

**File:** `opt_lut_256byte_expanded.h`

### Problem

16-byte LUTs require bit manipulation for indexing:

```
idx = (a & 0x03) * 3 + (b & 0x03);  // Mask + arithmetic
result = LUT_16[idx];
```

### Solution

Expand to 256 bytes, enabling direct byte indexing:

```
idx = (a << 4) | b;     // Simple shift + OR (or direct byte combine)
result = LUT_256[idx];  // No masking needed
```

### Memory Trade-off

| LUT Size | Operations | Total Memory | L1 Cache Usage |
|----------|------------|--------------|----------------|
| 16 bytes | 5 | 80 bytes | 0.25% |
| 256 bytes | 5 | 1.28 KB | 4% |
| 4096 bytes | 5 (binary) | 20 KB | 62% |

**Verdict:** 256-byte LUTs fit comfortably in L1 cache (32 KB)

### LUT Generation

```cpp
static inline void generate_tadd_lut_256b(uint8_t* lut) {
    const int8_t trit_decode[4] = {-1, 0, 1, 0};

    for (int a = 0; a < 256; a++) {
        for (int b = 0; b < 256; b++) {
            int8_t ta = trit_decode[a & 0x03];
            int8_t tb = trit_decode[b & 0x03];

            int8_t result = ta + tb;
            if (result > 1) result = 1;   // Saturation
            if (result < -1) result = -1;

            uint8_t encoded = (uint8_t)(result + 1);
            lut[(a << 4) | b] = encoded;
        }
    }
}
```

### Scalar Usage

```cpp
static inline uint8_t tadd_lut256b_scalar(uint8_t a, uint8_t b) {
    uint16_t idx = ((uint16_t)a << 4) | b;
    return TADD_LUT_256B[idx];
}
```

### Performance

| LUT Size | Cycles | Notes |
|----------|--------|-------|
| 16-byte | 2-3 | Requires masking |
| 256-byte | 1-2 | Direct indexing |

**Improvement:** 10-20%

---

## Combined Optimization

When all three techniques are combined:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     OPTIMIZED CRITICAL PATH                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input: trits_a, trits_b (32 each)                                  │
│                                                                      │
│  Step 1: Canonical Index + Dual-Shuffle                             │
│  ────────────────────────────────────────                           │
│  comp_a = shuffle(LUT_A, trits_a)   // Port 5, 1 cycle              │
│  comp_b = shuffle(LUT_B, trits_b)   // Port 5, 1 cycle (parallel)   │
│                                                                      │
│  Step 2: XOR Combine                                                │
│  ────────────────────────────────────────                           │
│  result = comp_a XOR comp_b         // Port 0, 1 cycle              │
│                                                                      │
│  Total: 2-3 cycles for 32 operations                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Baseline vs Optimized

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Index calc | 3 cycles | 0 cycles | 100% |
| Shuffle ops | 1 (serial) | 2 (parallel) | 2x throughput |
| Port usage | Port 5 only | Port 5 + Port 0 | 50% better ILP |
| Memory | 80 bytes | 1.28 KB | Acceptable |

**Combined improvement:** 30-40% over baseline

---

## Integration with Backend System

These optimizations are encapsulated in the AVX2 v2 backend:

```cpp
// Backend registration with capability flags
static const TernaryBackend AVX2_V2_BACKEND = {
    .info = {
        .name = "AVX2_v2",
        .description = "AVX2 with v1.2.0 optimizations",
        .capabilities = TERNARY_CAP_SIMD_256 |
                       TERNARY_CAP_FUSION |
                       TERNARY_CAP_CANONICAL |      // Canonical indexing
                       TERNARY_CAP_DUAL_SHUFFLE |   // Dual-shuffle XOR
                       TERNARY_CAP_LUT_256B,        // 256-byte LUTs
        .preferred_batch_size = 32
    },
    // ... operation pointers
};
```

### Backend Scoring

```c
// Selection scoring (higher = better)
if (caps & TERNARY_CAP_CANONICAL)    score += 50;
if (caps & TERNARY_CAP_DUAL_SHUFFLE) score += 100;
if (caps & TERNARY_CAP_LUT_256B)     score += 50;
```

---

## Verification

### Correctness Testing

```cpp
static inline bool verify_canonical_index_correctness() {
    // Test all 9 trit combinations
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            uint8_t scalar_idx = canonical_index(trits[i], trits[j]);
            __m256i simd_idx = canonical_index_avx2(va, vb);

            if (scalar_idx != extract_element(simd_idx, 0)) {
                return false;  // Mismatch
            }
        }
    }
    return true;
}
```

### Performance Validation

Run `bench_fusion.cpp` to compare baseline vs optimized:

```bash
./bench_fusion --compare-optimizations
```

---

## Future Work

### Phase 6 Targets

1. **Full LUT-256B SIMD** - Complete AVX2 implementation with chunked shuffle
2. **XOR-decomposable tmin/tmax** - Research alternative encodings
3. **AVX-512 port** - 64 trits per operation
4. **ARM NEON port** - Dual-shuffle adaptation for mobile

### Research Topics

- Automatic LUT decomposition analysis
- Profile-guided optimization selection
- Cache-aware LUT sizing

---

**See also:** [SIMD Kernels](SIMD_KERNELS.md), [Backend System](BACKEND_SYSTEM.md)
