# Fusion Operations Reference

**File:** `fused_binary_unary_ops.h`
**Status:** Validated (2025-10-29)
**Phase:** 4.0 (PoC) + 4.1 (Full Suite)

---

## Overview

Fusion eliminates intermediate arrays by combining sequential operations into a single pass. This reduces memory traffic by 40% (5N → 3N bytes) and keeps intermediate values in registers.

---

## Validation Summary

### Phase 4.1 Results (2025-10-29)

| Operation | Min Speedup | Max Speedup | Average | CV Range |
|-----------|-------------|-------------|---------|----------|
| `fused_tnot_tadd` | 1.62× | 1.95× | 1.76× | 3-27% |
| `fused_tnot_tmul` | 1.53× | 1.86× | 1.71× | 10-33% |
| `fused_tnot_tmin` | 1.61× | 11.26× | 4.06× | 15-88% |
| `fused_tnot_tmax` | 1.65× | 9.50× | 3.68× | 18-84% |

**Conservative claim:** 1.53× minimum speedup (any operation, any size)
**Typical speedup:** 2.80× average across all scenarios

---

## Memory Traffic Analysis

### Unfused (Two Separate Operations)

```
Operation: tnot(tadd(a, b))

Pass 1: tadd(A, B) → temp
  - Read A:     N bytes
  - Read B:     N bytes
  - Write temp: N bytes
  Subtotal: 3N bytes

Pass 2: tnot(temp) → result
  - Read temp:    N bytes
  - Write result: N bytes
  Subtotal: 2N bytes

Total: 5N bytes
```

### Fused (Single Pass)

```
Operation: fused_tnot_tadd(A, B) → result

  - Read A:       N bytes
  - Read B:       N bytes
  - Write result: N bytes

Total: 3N bytes
```

**Reduction:** 40% less memory traffic

---

## Implementation

### SIMD Fused Operations

```cpp
template <bool Sanitize = true>
static inline __m256i fused_tnot_tadd_simd(__m256i a, __m256i b) {
    __m256i temp = tadd_simd<Sanitize>(a, b);
    return tnot_simd<Sanitize>(temp);  // temp stays in register
}
```

Key insight: `temp` never leaves the CPU register file.

### Scalar Fused Operations

```cpp
static inline uint8_t fused_tnot_tadd_scalar(uint8_t a, uint8_t b) {
    return tnot(tadd(a, b));  // Single function call, no intermediate storage
}
```

---

## Function Reference

### fused_tnot_tadd_simd

```cpp
template <bool Sanitize = true>
static inline __m256i fused_tnot_tadd_simd(__m256i a, __m256i b);
```

**Semantics:** `tnot(tadd(a, b))`
**Performance:** 1.62-1.95× speedup
**Stability:** Most consistent (CV 3-27%)

---

### fused_tnot_tmul_simd

```cpp
template <bool Sanitize = true>
static inline __m256i fused_tnot_tmul_simd(__m256i a, __m256i b);
```

**Semantics:** `tnot(tmul(a, b))`
**Performance:** 1.53-1.86× speedup
**Stability:** Good (CV 10-33%)

---

### fused_tnot_tmin_simd

```cpp
template <bool Sanitize = true>
static inline __m256i fused_tnot_tmin_simd(__m256i a, __m256i b);
```

**Semantics:** `tnot(tmin(a, b))`
**Performance:** 1.61-11.26× speedup
**Stability:** High variance on large arrays
**Note:** Best peak performance for non-contiguous access

---

### fused_tnot_tmax_simd

```cpp
template <bool Sanitize = true>
static inline __m256i fused_tnot_tmax_simd(__m256i a, __m256i b);
```

**Semantics:** `tnot(tmax(a, b))`
**Performance:** 1.65-9.50× speedup
**Stability:** High variance on large arrays

---

## Instruction Count Analysis

### Unfused (7 instructions, 5 memory ops)

```
Operation 1 (tadd):
  loadu   A[i]          ; 1 instruction
  loadu   B[i]          ; 1 instruction
  shuffle (compute)     ; 1 instruction
  storeu  temp[i]       ; 1 instruction
                        ; Subtotal: 4 instructions

Operation 2 (tnot):
  loadu   temp[i]       ; 1 instruction
  shuffle (compute)     ; 1 instruction
  storeu  result[i]     ; 1 instruction
                        ; Subtotal: 3 instructions

Total: 7 instructions, 5 memory ops (2+1+1+1)
```

### Fused (5 instructions, 3 memory ops)

```
fused_tnot_tadd:
  loadu   A[i]          ; 1 instruction
  loadu   B[i]          ; 1 instruction
  shuffle (tadd)        ; 1 instruction
  shuffle (tnot)        ; 1 instruction (temp in register)
  storeu  result[i]     ; 1 instruction

Total: 5 instructions, 3 memory ops (2+1)
```

**Reduction:** 29% fewer instructions, 40% fewer memory ops

---

## Performance vs Array Size

| Array Size | Bound Type | Expected Speedup |
|------------|------------|------------------|
| < 100K | Compute | 1.1-1.3× |
| 100K - 1M | L3 cache | 1.5-1.8× |
| > 1M | DRAM | 1.8-2.5× |

Large arrays benefit most because memory bandwidth becomes the bottleneck.

---

## Usage Example

```cpp
#include "core/simd/fused_binary_unary_ops.h"

void apply_fused_operation(
    const uint8_t* a,
    const uint8_t* b,
    uint8_t* result,
    size_t n
) {
    size_t i = 0;

    // SIMD path with fusion
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i vr = fused_tnot_tadd_simd<true>(va, vb);
        _mm256_storeu_si256((__m256i*)(result + i), vr);
    }

    // Scalar tail
    for (; i < n; ++i) {
        result[i] = fused_tnot_tadd_scalar(a[i], b[i]);
    }
}
```

---

## Caveats and Limitations

### Micro vs Macro Speedup

- **Micro-kernel speedup:** 1.5-11× (isolated operations)
- **End-to-end application:** Typically 10-25% lower
- Reason: Real workloads have other bottlenecks (I/O, control flow)

### Variance

- High variance on large arrays (CV up to 88%)
- Performance depends on memory layout, cache state
- Use conservative estimates for planning (1.5× typical)

### When Fusion Helps Most

- Memory-bound workloads (large arrays)
- Sequential access patterns
- Operations that share intermediate results

### When Fusion Doesn't Help

- Compute-bound workloads (small arrays)
- Random access patterns
- Operations without shared intermediates

---

## Future Work (Phase 4.2+)

### Planned Fusion Patterns

| Pattern | Status | Expected Speedup |
|---------|--------|------------------|
| `tadd(tadd(a,b),c)` | Planned | 1.3-1.5× |
| `tmul(tnot(a),b)` | Planned | 1.4-1.8× |
| `tmin(tmax(a,b),c)` | Planned | 1.2-1.4× |
| 3+ operation chains | Research | TBD |

### Automatic Fusion Detection

Future compiler-like analysis to automatically fuse operations in expression trees.

---

**See also:** [SIMD Kernels](SIMD_KERNELS.md), [Optimizations](OPTIMIZATIONS.md)
