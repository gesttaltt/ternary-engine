# SIMD Kernels Reference

**File:** `simd_avx2_32trit_ops.h`
**Status:** Production-ready
**Dependencies:** `ternary_algebra.h`, `opt_canonical_index.h`

---

## Overview

The SIMD kernels process 32 trits simultaneously using AVX2 256-bit vectors. Each trit is stored as 1 byte (2-bit encoding), enabling byte-level SIMD operations via `_mm256_shuffle_epi8`.

---

## Data Format

### Trit Encoding

```
Value   Binary   Byte
─────   ──────   ────
 -1     0b00     0x00
  0     0b01     0x01
 +1     0b10     0x02
invalid 0b11     0x03 (sanitized to 0)
```

### Vector Layout

```
__m256i register (32 bytes):
┌────┬────┬────┬────┬────┬────┬────┬────┬───┬────┬────┬────┬────┬────┬────┬────┐
│ t0 │ t1 │ t2 │ t3 │ t4 │ t5 │ t6 │ t7 │...│t24 │t25 │t26 │t27 │t28 │t29 │t30 │t31 │
└────┴────┴────┴────┴────┴────┴────┴────┴───┴────┴────┴────┴────┴────┴────┴────┘
  Lane 0 (bytes 0-15)                        Lane 1 (bytes 16-31)
```

---

## Core Algorithm

### Binary Operations (tadd, tmul, tmin, tmax)

The implementation uses **canonical indexing** with **dual-shuffle + ADD**:

```cpp
template <bool Sanitize = true>
static inline __m256i binary_simd_op(__m256i a, __m256i b, __m256i lut) {
    // 1. Optional input sanitization (mask to 2 bits)
    __m256i a_masked = maybe_mask<Sanitize>(a);
    __m256i b_masked = maybe_mask<Sanitize>(b);

    // 2. Load canonical LUTs (compile-time constants)
    __m256i canon_a = _mm256_load_si256((__m256i*)CANON_A_LUT_256);
    __m256i canon_b = _mm256_load_si256((__m256i*)CANON_B_LUT_256);

    // 3. Dual-shuffle: Two parallel lookups (no data dependency)
    __m256i contrib_a = _mm256_shuffle_epi8(canon_a, a_masked);
    __m256i contrib_b = _mm256_shuffle_epi8(canon_b, b_masked);

    // 4. Combine indices with ADD (faster than OR)
    __m256i indices = _mm256_add_epi8(contrib_a, contrib_b);

    // 5. Final operation lookup
    return _mm256_shuffle_epi8(lut, indices);
}
```

### Index Calculation

Traditional approach (3 cycles):
```
idx = (a << 2) | b    // Dependent shift + OR
```

Canonical approach (2 cycles):
```
idx = CANON_A[a] + CANON_B[b]    // Parallel shuffles + ADD
```

**Improvement:** 12-18% faster

---

## Function Reference

### tadd_simd

```cpp
template <bool Sanitize = true>
static inline __m256i tadd_simd(__m256i a, __m256i b);
```

**Operation:** Saturated ternary addition
- `(-1) + (-1) = -1` (saturates)
- `(-1) + 0 = -1`
- `0 + 0 = 0`
- `0 + (+1) = +1`
- `(+1) + (+1) = +1` (saturates)

**Template parameter:**
- `Sanitize=true`: Masks inputs to valid 2-bit range (safe)
- `Sanitize=false`: Assumes pre-sanitized inputs (faster)

---

### tmul_simd

```cpp
template <bool Sanitize = true>
static inline __m256i tmul_simd(__m256i a, __m256i b);
```

**Operation:** Ternary multiplication
- `x * 0 = 0` (absorbing element)
- `x * (+1) = x` (identity)
- `x * (-1) = -x` (negation)

**Properties:** Commutative, associative, distributive

---

### tmin_simd / tmax_simd

```cpp
template <bool Sanitize = true>
static inline __m256i tmin_simd(__m256i a, __m256i b);
static inline __m256i tmax_simd(__m256i a, __m256i b);
```

**Operation:** Ternary minimum/maximum

**Properties:** Commutative, associative, idempotent

---

### tnot_simd

```cpp
template <bool Sanitize = true>
static inline __m256i tnot_simd(__m256i a);
```

**Operation:** Ternary negation (sign flip)
- `tnot(-1) = +1`
- `tnot(0) = 0` (fixed point)
- `tnot(+1) = -1`

**Properties:** Involution (`tnot(tnot(x)) = x`)

---

## Pre-Broadcasted LUT Cache

LUTs are pre-loaded and broadcasted once at startup to avoid repeated memory access:

```cpp
struct BroadcastedLUTs {
    __m256i tadd;
    __m256i tmul;
    __m256i tmin;
    __m256i tmax;
    __m256i tnot;

    BroadcastedLUTs()
        : tadd(broadcast_lut_16(TADD_LUT.data()))
        , tmul(broadcast_lut_16(TMUL_LUT.data()))
        // ...
    {}
};

static const BroadcastedLUTs g_luts;  // Global singleton
```

**Memory:** 5 × 32 bytes = 160 bytes (fits in L1 cache)

---

## Performance Characteristics

### Throughput

| Size | SIMD (ns/elem) | Scalar (ns/elem) | Speedup |
|------|----------------|------------------|---------|
| 1K | 0.31 | 10.5 | 33.9× |
| 10K | 0.29 | 10.2 | 35.2× |
| 100K | 0.28 | 10.1 | 36.1× |
| 1M | 0.28 | 10.0 | 35.7× |

### Instruction Mix (per 32 trits)

| Instruction | Count | Latency | Port |
|-------------|-------|---------|------|
| `_mm256_loadu_si256` | 2 | 1 | 2/3 |
| `_mm256_and_si256` | 2 | 1 | 0/1/5 |
| `_mm256_shuffle_epi8` | 4 | 1 | 5 |
| `_mm256_add_epi8` | 1 | 1 | 0/1/5 |
| `_mm256_storeu_si256` | 1 | 1 | 2/3/7 |
| **Total** | **10** | | |

---

## Sanitization Mode

### When to use `Sanitize=true` (default)

- External inputs (user data, file I/O)
- Untrusted data sources
- After deserialization

### When to use `Sanitize=false`

- Internal operations (output of previous SIMD op)
- Known-good data (compile-time constants)
- Performance-critical inner loops

**Cost:** Sanitization adds 2 `_mm256_and_si256` instructions (~1% overhead)

---

## Integration Notes

### Include Path

```cpp
#include "core/simd/simd_avx2_32trit_ops.h"
```

### Compiler Flags

```bash
# GCC/Clang
-O3 -march=native -mavx2 -std=c++17

# MSVC
/O2 /arch:AVX2 /std:c++17
```

### Alignment

Input/output arrays should be 32-byte aligned for optimal performance:

```cpp
alignas(32) uint8_t a[1024];
alignas(32) uint8_t b[1024];
alignas(32) uint8_t result[1024];
```

---

**See also:** [Fusion Operations](FUSION.md), [Optimizations](OPTIMIZATIONS.md)
