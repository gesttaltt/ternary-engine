# Canonical Indexing Fix: Root Cause Analysis and Resolution

**Doc-Type:** Technical Analysis · Version 1.0 · 2025-11-28

---

## Executive Summary

A critical correctness bug was discovered in the SIMD kernel path: the canonical indexing optimization was incompletely implemented, causing incorrect results for arrays ≥32 elements. This document explains the root cause, the fix, and the performance implications.

**Key Finding:** After fixing, the SIMD kernel is 29x faster than NumPy, but format conversion overhead (97.6% of pipeline time) negates this advantage in typical usage.

---

## The Bug

### Symptoms

```
n < 32:   CORRECT (scalar path)
n >= 32:  INCORRECT (SIMD path) - all results shifted by -1
```

### Root Cause

Two incompatible indexing schemes were mixed:

| Scheme | Formula | Index Range | Used By |
|:-------|:--------|:------------|:--------|
| Traditional | `(a << 2) \| b` | 0,1,2,4,5,6,8,9,10 (gaps) | Scalar ops, original LUTs |
| Canonical | `a*3 + b` | 0,1,2,3,4,5,6,7,8 (compact) | SIMD dual-shuffle |

**The Problem:** SIMD kernel used canonical indexing formula, but LUTs were organized for traditional indexing.

**Example:**
```
tadd(+1, 0) where +1=2, 0=1 in encoding

Traditional: idx = (2 << 2) | 1 = 9  →  LUT[9] = correct result
Canonical:   idx = 2*3 + 1 = 7      →  LUT[7] = WRONG VALUE (padding/garbage)
```

### Why It Happened

The canonical indexing optimization (Phase 3.2) was implemented in two parts:
1. ✅ Index calculation mechanism (dual-shuffle + ADD)
2. ❌ LUT reorganization for new index scheme (MISSING)

The optimization promised 12-18% speedup by replacing dependent arithmetic (shift+OR) with parallel operations (two shuffles + ADD). The mechanism was correct, but the LUTs still expected traditional indices.

---

## The Fix

### Solution: Create Canonical LUTs

Added new compile-time LUT generator in `ternary_lut_gen.h`:

```cpp
template <typename Func>
constexpr std::array<uint8_t, 16> make_canonical_binary_lut(Func op) {
    std::array<uint8_t, 16> lut{};

    // Generate 9 valid entries (3x3 combinations)
    for (size_t a = 0; a < 3; ++a) {
        for (size_t b = 0; b < 3; ++b) {
            size_t index = a * 3 + b;  // Canonical index: 0-8
            lut[index] = op(a, b);
        }
    }

    // Pad entries 9-15 for AVX2 shuffle safety
    for (size_t i = 9; i < 16; ++i) {
        lut[i] = lut[0];
    }

    return lut;
}
```

Created canonical versions of all operation LUTs in `ternary_algebra.h`:
- `TADD_LUT_CANONICAL`
- `TMUL_LUT_CANONICAL`
- `TMIN_LUT_CANONICAL`
- `TMAX_LUT_CANONICAL`
- `TNOT_LUT_CANONICAL`

Updated SIMD kernel to use canonical LUTs:

```cpp
BroadcastedLUTs()
    : tadd(broadcast_lut_16(TADD_LUT_CANONICAL.data()))
    , tmul(broadcast_lut_16(TMUL_LUT_CANONICAL.data()))
    // ...
{}
```

### Files Modified

1. `src/core/algebra/ternary_lut_gen.h` - Added canonical LUT generators
2. `src/core/algebra/ternary_algebra.h` - Added canonical LUT definitions
3. `src/core/simd/simd_avx2_32trit_ops.h` - Updated to use canonical LUTs

---

## Performance Analysis

### Post-Fix Benchmark Results (2025-11-28)

```
PHASE 2: Full Pipeline (int8 → ternary → operation → int8)

      Size |   Ternary |     NumPy |  Speedup
-----------+-----------+-----------+---------
        64 |   6.8 M/s |   9.9 M/s |   0.69x
     1,024 |  80.6 M/s | 116.6 M/s |   0.69x
    16,384 | 294.6 M/s | 313.1 M/s |   0.94x
 1,048,576 | 183.9 M/s | 222.8 M/s |   0.83x

Winner: NumPy (at all sizes)
```

```
PHASE 3: Overhead Breakdown (size=100,000)

  Full pipeline:       0.291 ms (100.0%)
  Input conversion:    0.186 ms ( 64.0%)
  SIMD kernel:         0.007 ms (  2.5%)
  Output conversion:   0.098 ms ( 33.6%)

  Kernel-only speedup: 29.3x vs NumPy
```

### Key Insights

1. **The SIMD kernel is 29x faster than NumPy** when operating on native ternary format.

2. **Conversion overhead is 97.6%** of total pipeline time, completely negating the kernel speedup.

3. **NumPy wins in typical usage** because it operates directly on int8 without format conversion.

4. **No crossover point exists** in the tested range - format conversion overhead always dominates.

---

## Architectural Implications

### When Ternary Engine Wins

The engine provides value when:
1. **Data is born ternary** - No input conversion needed
2. **Data dies ternary** - No output conversion needed
3. **Multiple operations** - Amortize conversion over many ops
4. **Memory-constrained** - 4x compression vs int8 matters more than speed

### When NumPy Wins

NumPy is better when:
1. Data arrives as int8/float and must return as int8/float
2. Single operation per array
3. Memory is not constrained
4. Integration with other NumPy code

### Strategic Direction

This analysis validates the "sparse computation" direction:

```
Current (conversion-heavy):
  int8 → uint8 → SIMD kernel → uint8 → int8
         ↑                          ↑
      64% overhead              34% overhead

Ideal (native ternary):
  ternary_tensor → SIMD kernel → ternary_tensor
                      ↑
                  2.5% of time, 29x faster than NumPy
```

The path forward is not to compete with NumPy on int8 operations, but to:
1. Create native ternary data structures
2. Keep data in ternary format across operation chains
3. Only convert at system boundaries (input/output)

---

## Verification

### Correctness Tests

All array sizes now pass (was failing for n≥32 before fix):

```
Size      32: OK
Size      64: OK
Size     100: OK
Size   1,000: OK
Size  10,000: OK
Size 100,000: OK
```

### Regression Prevention

Added to test suite:
- `benchmarks/bench_canonical_fix.py` - Correctness + performance validation
- `benchmarks/test_falsification.py` - Ongoing falsification tests

---

## Lessons Learned

1. **Complete the optimization** - Implementing half an optimization is worse than no optimization (it breaks correctness while adding complexity).

2. **Test at SIMD boundaries** - The bug only manifested at n=32 because that's when SIMD path activates. Tests should explicitly cover these boundaries.

3. **Skeptical benchmarking matters** - The falsification framework caught this bug that unit tests missed.

4. **Know your bottleneck** - 29x kernel speedup means nothing when conversion is 97.6% of the work.

---

## References

- Benchmark results: `benchmarks/results/canonical_fix_*.json`
- Canonical indexing design: `docs/CANONICAL_INDEXING_ANALYSIS.md`
- SIMD kernel implementation: `src/core/simd/simd_avx2_32trit_ops.h`
- LUT generation: `src/core/algebra/ternary_lut_gen.h`

---

**Status:** RESOLVED
**Impact:** Critical correctness bug fixed, performance characteristics now accurately measured
**Next Steps:** Explore native ternary data structures to eliminate conversion overhead
