# Canonical Indexing Analysis and Reorganization Plan

**Doc-Type:** Technical Analysis · Version 1.0 · Updated 2025-11-24 · Author Ternary Engine Team

Analysis of LUT reorganization required for canonical indexing optimization in v1.3.0.

---

## Problem Statement

The current LUT organization uses **traditional indexing** `idx = (a<<2)|b` which is incompatible with **canonical indexing** `idx = (a*3)+b`. This prevents the canonical indexing optimization from being used in AVX2_v2 backend.

### Current System (Traditional Indexing)

**Index Calculation:** `idx = (a << 2) | b`

**Trit Encoding:** 2-bit format
- `0b00` = -1 (MINUS_ONE)
- `0b01` = 0  (ZERO)
- `0b10` = +1 (PLUS_ONE)
- `0b11` = Invalid (unused)

**Index Mapping for Valid Trits:**
```
Input (a,b)    2-bit values    Traditional idx
───────────────────────────────────────────────
(-1, -1)       (00, 00)        (00<<2)|00 = 0
(-1,  0)       (00, 01)        (00<<2)|01 = 1
(-1, +1)       (00, 10)        (00<<2)|10 = 2
( 0, -1)       (01, 00)        (01<<2)|00 = 4
( 0,  0)       (01, 01)        (01<<2)|01 = 5
( 0, +1)       (01, 10)        (01<<2)|10 = 6
(+1, -1)       (10, 00)        (10<<2)|00 = 8
(+1,  0)       (10, 01)        (10<<2)|01 = 9
(+1, +1)       (10, 10)        (10<<2)|10 = 10
```

**Characteristics:**
- Produces indices: 0, 1, 2, 4, 5, 6, 8, 9, 10 (9 valid entries)
- Unused indices: 3, 7, 11, 12, 13, 14, 15 (7 invalid entries)
- Total LUT size: 16 entries (for AVX2 _mm256_shuffle_epi8 compatibility)

---

## Canonical Indexing System

**Index Calculation:** `idx = (a * 3) + b`

**Normalized Trit Values:** 0-based encoding
- 0 = -1 (MINUS_ONE) → from 0b00
- 1 = 0  (ZERO)      → from 0b01
- 2 = +1 (PLUS_ONE)  → from 0b10

**Index Mapping for Valid Trits:**
```
Input (a,b)    Normalized      Canonical idx
───────────────────────────────────────────────
(-1, -1)       (0, 0)          (0*3)+0 = 0
(-1,  0)       (0, 1)          (0*3)+1 = 1
(-1, +1)       (0, 2)          (0*3)+2 = 2
( 0, -1)       (1, 0)          (1*3)+0 = 3
( 0,  0)       (1, 1)          (1*3)+1 = 4
( 0, +1)       (1, 2)          (1*3)+2 = 5
(+1, -1)       (2, 0)          (2*3)+0 = 6
(+1,  0)       (2, 1)          (2*3)+1 = 7
(+1, +1)       (2, 2)          (2*3)+2 = 8
```

**Characteristics:**
- Produces contiguous indices: 0, 1, 2, 3, 4, 5, 6, 7, 8 (9 valid entries)
- No gaps in index space
- Natural 3×3 matrix organization
- More cache-friendly (contiguous memory access)

---

## LUT Reorganization Mapping

To convert from traditional to canonical indexing, we need to remap LUT entries:

**Mapping Table:**

| Input (a,b) | Traditional idx | Canonical idx | Action |
|-------------|-----------------|---------------|--------|
| (-1, -1)    | 0               | 0             | Copy LUT[0] → CANON[0] |
| (-1,  0)    | 1               | 1             | Copy LUT[1] → CANON[1] |
| (-1, +1)    | 2               | 2             | Copy LUT[2] → CANON[2] |
| ( 0, -1)    | 4               | 3             | Copy LUT[4] → CANON[3] |
| ( 0,  0)    | 5               | 4             | Copy LUT[5] → CANON[4] |
| ( 0, +1)    | 6               | 5             | Copy LUT[6] → CANON[5] |
| (+1, -1)    | 8               | 6             | Copy LUT[8] → CANON[6] |
| (+1,  0)    | 9               | 7             | Copy LUT[9] → CANON[7] |
| (+1, +1)    | 10              | 8             | Copy LUT[10] → CANON[8] |

**Transformation Function:**
```cpp
// Convert traditional LUT to canonical LUT
void reorganize_to_canonical(const uint8_t* traditional_lut, uint8_t* canonical_lut) {
    canonical_lut[0] = traditional_lut[0];   // (-1,-1)
    canonical_lut[1] = traditional_lut[1];   // (-1, 0)
    canonical_lut[2] = traditional_lut[2];   // (-1,+1)
    canonical_lut[3] = traditional_lut[4];   // ( 0,-1)
    canonical_lut[4] = traditional_lut[5];   // ( 0, 0)
    canonical_lut[5] = traditional_lut[6];   // ( 0,+1)
    canonical_lut[6] = traditional_lut[8];   // (+1,-1)
    canonical_lut[7] = traditional_lut[9];   // (+1, 0)
    canonical_lut[8] = traditional_lut[10];  // (+1,+1)
}
```

---

## Example: TADD Operation

### Traditional LUT (Current)
```
Index:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
Value: 00 01 10  X 01 01 10  X 10 10 10  X  X  X  X  X

Where X = unused/invalid entries
```

**Lookup for tadd(-1, +1):**
- a = 0b00 (-1), b = 0b10 (+1)
- idx = (0b00 << 2) | 0b10 = 2
- result = LUT[2] = 0b10 (+1) ✓ Correct: -1 + 1 = 0, saturated to +1

### Canonical LUT (Target)
```
Index:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
Value: 00 01 10 01 01 10 10 10 10  X  X  X  X  X  X  X

Where X = padding for AVX2 compatibility (16 bytes)
```

**Lookup for tadd(-1, +1):**
- a = 0b00 (-1), b = 0b10 (+1)
- Normalize: a_norm = 0, b_norm = 2
- idx = (0 * 3) + 2 = 2
- result = CANON_LUT[2] = 0b10 (+1) ✓ Correct: -1 + 1 = 0, saturated to +1

---

## Benefits of Canonical Indexing

### 1. Eliminates Arithmetic Operations
**Traditional:**
```assembly
; Traditional indexing
shl     a, 2        ; Shift left by 2
or      a, b        ; Combine with b
shuffle result, a   ; Lookup
; 3 operations, dependent chain
```

**Canonical with Dual-Shuffle:**
```assembly
; Canonical indexing
shuffle contrib_a, a    ; Parallel on Port 5
shuffle contrib_b, b    ; Parallel on Port 5
add     result, contrib_a, contrib_b  ; Parallel on Port 0
; 3 operations, 2 can execute in parallel
```

### 2. Better Microarchitecture Utilization
- Traditional: Shift → OR → Shuffle (dependent chain, serial execution)
- Canonical: Shuffle A || Shuffle B → Add (2 parallel, 1 dependent)

**Expected ILP Gain:**
- Intel (Haswell+): Shuffle on Port 5, Add on Port 0/1/5 → parallel execution
- AMD (Zen2+): Shuffle on Port 3, Add on Port 0/1/2/3 → parallel execution

### 3. More Cache-Friendly
- Contiguous index space (0-8) vs sparse space (0,1,2,4,5,6,8,9,10)
- Better spatial locality
- Fewer cache line crossings

### 4. Enables Further Optimizations
- Dual-shuffle XOR (parallel shuffle + combine)
- Better compiler optimization (contiguous loops)
- Potential for 9-entry compact LUTs (future)

---

## Performance Expectations

### Baseline (Traditional Indexing in AVX2_v2)
- Current performance: ~9,000-45,000 Mops/s
- Average speedup over Scalar: 4.8×

### With Canonical Indexing
**Conservative Estimate:** +12-15% throughput improvement
- Expected: ~10,000-52,000 Mops/s
- Average speedup over Scalar: 5.4×

**Optimistic Estimate:** +15-18% throughput improvement
- Expected: ~10,000-54,000 Mops/s
- Average speedup over Scalar: 5.6×

**Basis for Estimates:**
- Eliminates 1 dependent arithmetic operation (shift/OR)
- Enables better instruction-level parallelism
- Similar to measured gains in other SIMD optimizations

### Validation Criteria
- ✅ Cross-backend correctness maintained (Scalar == AVX2_v1 == AVX2_v2_canonical)
- ✅ Measurable speedup: >10% improvement over traditional
- ✅ Statistical rigor: CV < 20%, confidence intervals reported
- ✅ All 13/13 integration tests pass

---

## Implementation Plan

### Step 1: Create Canonical LUT Generator
**File:** `src/core/algebra/ternary_canonical_lut.h`

**Tasks:**
1. Create `make_canonical_binary_lut()` template function
2. Generate canonical LUTs for TADD, TMUL, TMIN, TMAX
3. Pad to 16 entries for AVX2 compatibility
4. Add compile-time validation

**Time:** 1-2 hours

### Step 2: Update AVX2_v2 Backend
**File:** `src/core/simd/ternary_backend_avx2_v2.cpp`

**Tasks:**
1. Replace traditional LUT loading with canonical LUTs
2. Update `binary_op_traditional()` to `binary_op_canonical()`
3. Use `canonical_index_avx2()` from `ternary_canonical_index.h`
4. Update initialization to load canonical LUTs

**Time:** 1 hour

### Step 3: Validate Correctness
**Tests:** `tests/python/test_backend_integration.py`

**Tasks:**
1. Run full integration test suite
2. Verify 13/13 tests pass
3. Validate cross-backend correctness
4. Check for any edge cases

**Time:** 30 minutes

### Step 4: Benchmark Performance
**Script:** `benchmarks/bench_backends.py`

**Tasks:**
1. Run full benchmark suite
2. Compare canonical vs traditional indexing
3. Calculate actual speedup
4. Generate statistical analysis (CV, confidence intervals)
5. Document results

**Time:** 30 minutes

### Step 5: Documentation and Commit
**Files:** Update relevant documentation

**Tasks:**
1. Update V1.2.0_STATUS.md
2. Update BACKEND_API.md
3. Add canonical indexing notes
4. Commit with comprehensive message

**Time:** 30 minutes

**Total Estimated Time:** 3.5-4 hours

---

## Verification Checklist

### Correctness
- [ ] All 13/13 integration tests pass
- [ ] Cross-backend correctness: Scalar == AVX2_v1 == AVX2_v2_canonical
- [ ] No segfaults or memory errors
- [ ] Edge cases handled correctly (all 9 combinations)

### Performance
- [ ] Canonical faster than traditional by >10%
- [ ] No performance regression for any operation
- [ ] Speedup consistent across array sizes
- [ ] Statistical validation (CV < 20%)

### Code Quality
- [ ] Compile-time LUT generation maintained
- [ ] Code follows project style
- [ ] Comments explain canonical indexing
- [ ] No magic numbers (use symbolic constants)

### Documentation
- [ ] Algorithm explained in comments
- [ ] Performance characteristics documented
- [ ] Validation date and platform recorded
- [ ] Commit message comprehensive

---

## Rollback Plan

If canonical indexing fails validation:

1. **Revert Changes:**
   ```bash
   git revert HEAD
   ```

2. **Restore Traditional Indexing:**
   - Keep `binary_op_traditional()` in AVX2_v2
   - Document why canonical was reverted
   - Add to v1.3.1 roadmap

3. **Investigate Root Cause:**
   - Analyze which test failed
   - Check LUT generation correctness
   - Verify index calculation
   - Compare against theoretical expectations

4. **Document Findings:**
   - Add to known issues
   - Update performance expectations
   - Plan fix for next iteration

---

## References

- **Current LUT System:** `src/core/algebra/ternary_algebra.h`
- **LUT Generation:** `src/core/algebra/ternary_lut_gen.h`
- **Canonical Indexing:** `src/core/simd/ternary_canonical_index.h`
- **AVX2_v2 Backend:** `src/core/simd/ternary_backend_avx2_v2.cpp`
- **Integration Tests:** `tests/python/test_backend_integration.py`
- **Benchmarks:** `benchmarks/bench_backends.py`

---

**Version:** 1.0 · **Status:** Ready for Implementation · **Date:** 2025-11-24
