# Bridge Layer Architecture: Algebraic Foundations

**Doc-Type:** Architectural Design · Version 1.0 · 2025-11-28

---

## Executive Summary

The Bridge Layer is the critical interface between the Python world (int8 semantic values) and the Ternary Engine world (2-bit encoded operations). This document establishes the mathematical foundations using abstract algebra and graph theory to derive optimal computational paths.

**Key Insight:** The bridge is not merely a conversion layer—it's a **functor** between two categories that preserves algebraic structure while minimizing computational overhead.

---

## 1. Algebraic Foundations

### 1.1 The Balanced Ternary Algebraic Structure

Let **T** = {-1, 0, +1} be the balanced ternary set.

#### Ring-like Properties (T, ⊕, ⊗)

**Addition (saturated):**
```
⊕ : T × T → T
a ⊕ b = clamp(a + b, -1, +1)

    ⊕ │ -1   0  +1
   ───┼────────────
   -1 │ -1  -1   0
    0 │ -1   0  +1
   +1 │  0  +1  +1
```

Properties:
- Commutative: a ⊕ b = b ⊕ a ✓
- Associative: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c) ✓
- Identity: 0 is identity (a ⊕ 0 = a) ✓
- **NOT a group**: No inverse for non-zero elements (1 ⊕ x ≠ 0 for any x)
- **Structure**: Bounded join-semilattice with saturation

**Multiplication:**
```
⊗ : T × T → T
a ⊗ b = a × b (standard multiplication)

    ⊗ │ -1   0  +1
   ───┼────────────
   -1 │ +1   0  -1
    0 │  0   0   0
   +1 │ -1   0  +1
```

Properties:
- Commutative: a ⊗ b = b ⊗ a ✓
- Associative: (a ⊗ b) ⊗ c = a ⊗ (b ⊗ c) ✓
- Identity: +1 is identity ✓
- Zero: 0 is absorbing (a ⊗ 0 = 0) ✓
- **Monoid**: (T, ⊗, +1)
- **NOT a group**: 0 has no inverse

**Negation:**
```
¬ : T → T
¬a = -a

¬(-1) = +1
¬(0)  = 0
¬(+1) = -1
```

Properties:
- Involution: ¬(¬a) = a ✓
- Isomorphism: ¬ is an automorphism of (T, ⊗)

#### Classification

(T, ⊕, ⊗) is a **bounded commutative semiring with involution**:
- (T, ⊕) is a commutative monoid with bounds
- (T, ⊗) is a commutative monoid
- Distributivity: a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c) ✓
- Involution ¬ reverses ⊗ sign

---

### 1.2 Representation Theory

We define two concrete representations of **T**:

#### Representation R₁: Semantic (int8)

```
R₁ = ({-1, 0, +1} ⊂ ℤ₈, +̂, ×̂, -̂)

Encoding:
  -1 → 0xFF (binary: 11111111)
   0 → 0x00 (binary: 00000000)
  +1 → 0x01 (binary: 00000001)
```

#### Representation R₂: Computational (uint8 2-bit)

```
R₂ = ({0, 1, 2} ⊂ ℕ₈, LUT_⊕, LUT_⊗, LUT_¬)

Encoding:
  -1 → 0b00 = 0
   0 → 0b01 = 1
  +1 → 0b10 = 2
```

#### The Isomorphism φ

The bridge isomorphism φ: R₁ → R₂ is defined by:

```
φ(x) = x + 1  (mod 256, interpreted as uint8)

φ(-1) = 0xFF + 1 = 0x00 = 0 ✓
φ(0)  = 0x00 + 1 = 0x01 = 1 ✓
φ(+1) = 0x01 + 1 = 0x02 = 2 ✓
```

The inverse φ⁻¹: R₂ → R₁ is:

```
φ⁻¹(y) = y - 1  (mod 256, interpreted as int8)

φ⁻¹(0) = 0 - 1 = 0xFF = -1 ✓
φ⁻¹(1) = 1 - 1 = 0x00 = 0  ✓
φ⁻¹(2) = 2 - 1 = 0x01 = +1 ✓
```

**Critical Property (Homomorphism):**
```
For any operation ⊙ ∈ {⊕, ⊗, ¬}:
  φ(a ⊙ b) = φ(a) ⊙' φ(b)

Where ⊙' is the LUT-based implementation in R₂.
```

This means operations in R₁ and R₂ are **structure-preserving**.

---

### 1.3 Category-Theoretic Framework

Define the category **Tern** of ternary representations:

**Objects:**
- `Int8`: Semantic representation (Python world)
- `Uint8_2bit`: Computational representation (SIMD world)
- `Dense243`: Storage representation (5 trits/byte)
- `Packed4`: Future packed representation (4 trits/byte)

**Morphisms:**
- `φ: Int8 → Uint8_2bit` (cost: 1 SIMD instruction per 32 elements)
- `φ⁻¹: Uint8_2bit → Int8` (cost: 1 SIMD instruction per 32 elements)
- `pack243: Uint8_2bit → Dense243` (cost: ~10 instructions per 5 elements)
- `unpack243: Dense243 → Uint8_2bit` (cost: ~5 LUT lookups per byte)

**Functors:**
The bridge layer implements a **functor F: Tern → Tern** that:
1. Maps objects to themselves (identity on objects)
2. Maps morphisms to fused compositions

**Natural Transformations:**
Optimization strategies are natural transformations between functors that preserve semantics while reducing cost.

---

## 2. Graph-Theoretic Optimization

### 2.1 Computation DAG Model

A ternary computation is modeled as a directed acyclic graph (DAG):

```
G = (V, E, w)

V = Nodes (arrays in specific formats)
E = Edges (operations/conversions)
w: E → ℝ⁺ (edge weights = computational cost)
```

**Example: tadd(a, b) with naive approach:**
```
        ┌──────────────────────────────────────┐
        │          Computation DAG             │
        └──────────────────────────────────────┘

     a_int8                    b_int8
        │                         │
        │ w=1                     │ w=1
        ▼                         ▼
     a_uint8 ─────────┬───────── b_uint8
                      │
                      │ w=3 (kernel)
                      ▼
                  r_uint8
                      │
                      │ w=1
                      ▼
                  r_int8

Total cost: 1 + 1 + 3 + 1 = 6 units
Memory ops: 6 (2 input converts + 2 kernel loads + 1 kernel store + 1 output convert)
```

**Example: tadd(a, b) with fused approach:**
```
     a_int8 ─────────┬───────── b_int8
                     │
                     │ w=5 (fused)
                     ▼
                 r_int8

Total cost: 5 units (all in registers)
Memory ops: 3 (2 loads + 1 store)
```

### 2.2 Cost Model

Define the cost function C: E → ℝ⁺:

```
C(edge) = α·(compute_cycles) + β·(memory_bytes) + γ·(allocation_overhead)

Where:
  α = 1.0   (compute weight)
  β = 10.0  (memory weight - memory is expensive)
  γ = 100.0 (allocation weight - Python/NumPy overhead is massive)
```

**Cost Table (per 32 elements):**

| Operation | Cycles | Memory | Alloc | Total Cost |
|:----------|:-------|:-------|:------|:-----------|
| SIMD add_epi8 | 1 | 0 | 0 | 1 |
| SIMD shuffle | 1 | 0 | 0 | 1 |
| SIMD load | 1 | 32 | 0 | 321 |
| SIMD store | 1 | 32 | 0 | 321 |
| NumPy arr+1 | 10 | 64 | 1 | 750 |
| NumPy astype | 10 | 32 | 1 | 430 |

**Key Insight:** NumPy operations are 100-750x more expensive than register operations due to allocation overhead.

### 2.3 Optimization as Shortest Path

Given a computation specification, find the minimum-cost path through format conversions.

**Algorithm: Dijkstra on Format DAG**

```
1. Build format graph with all possible paths
2. Assign costs using cost model
3. Find shortest path from input format to output format
4. Fuse adjacent operations on same format
```

**Example: tnot(tadd(a, tmul(b, c)))**

Naive cost:
```
b_int8 → b_uint8 → ┐
                   ├─[tmul]→ t1_uint8 → ┐
c_int8 → c_uint8 → ┘                    ├─[tadd]→ t2_uint8 → [tnot]→ r_uint8 → r_int8
a_int8 → a_uint8 ──────────────────────→┘

Conversions: 4 in + 1 out = 5 × 750 = 3750
Operations: 3 × 5 = 15
Total: 3765 units
```

Fused cost:
```
a_int8 ─┐
b_int8 ─┼─[fused_tnot_tadd_tmul_int8]→ r_int8
c_int8 ─┘

Conversions: 0 (all in registers)
Operations: 1 fused = 20 (internal: 3 converts + 3 ops)
Total: 20 units
```

**Speedup: 188x** from graph optimization alone.

---

## 3. Canonical Index Theory

### 3.1 Index Space Analysis

For binary operations f(a, b), we need to map (a, b) → index for LUT lookup.

**Traditional Indexing:**
```
idx = (a << 2) | b

For a,b ∈ {0,1,2}:
  (0,0)→0  (0,1)→1  (0,2)→2
  (1,0)→4  (1,1)→5  (1,2)→6
  (2,0)→8  (2,1)→9  (2,2)→10

Index range: {0,1,2,4,5,6,8,9,10} (gaps at 3,7,11-15)
```

**Canonical Indexing:**
```
idx = a*3 + b

For a,b ∈ {0,1,2}:
  (0,0)→0  (0,1)→1  (0,2)→2
  (1,0)→3  (1,1)→4  (1,2)→5
  (2,0)→6  (2,1)→7  (2,2)→8

Index range: {0,1,2,3,4,5,6,7,8} (compact)
```

### 3.2 Direct Int8 Indexing

For int8 inputs a,b ∈ {-1,0,+1}:

```
idx = (a+1)*3 + (b+1) = 3a + b + 4

For a,b ∈ {-1,0,+1}:
  (-1,-1)→0  (-1,0)→1  (-1,+1)→2
  (0,-1)→3   (0,0)→4   (0,+1)→5
  (+1,-1)→6  (+1,0)→7  (+1,+1)→8

Same index range as canonical!
```

**Implementation:**
```cpp
// Instead of:
a' = a + 1;
b' = b + 1;
idx = shuffle(CANON_A, a') + shuffle(CANON_B, b');

// We could use:
// Direct formula: idx = 3a + b + 4
// But SIMD multiplication by 3 is expensive...

// Better: Create INT8_CANON_A_LUT that maps:
//   0xFF (-1) → 0
//   0x00 (0)  → 3
//   0x01 (+1) → 6

// Then: idx = shuffle(INT8_CANON_A, a & 0x03) + shuffle(INT8_CANON_B, b & 0x03) + bias
```

Wait, this doesn't work directly because -1 = 0xFF has lower nibble = 0xF = 15.

**Solution: Masked Index Tables**

```cpp
// For int8 input, mask to get usable index:
// a & 0x03 gives: -1 (0xFF) → 3, 0 (0x00) → 0, +1 (0x01) → 1

// Create LUT for this mapping:
INT8_REMAP_LUT[4] = {
    [0] = 1,  // 0x00 (0) → 1 (middle trit)
    [1] = 2,  // 0x01 (+1) → 2 (plus trit)
    [2] = ?,  // 0x02 (invalid)
    [3] = 0   // 0x03 (-1 & 0x03 = 3) → 0 (minus trit)
}
```

Hmm, this is getting complex. The simplest approach remains:
1. Add 1 to convert int8 → uint8 (single instruction)
2. Use existing canonical LUTs
3. Subtract 1 to convert result back (single instruction)

The overhead is 2 cycles out of ~5 total cycles = 40% of kernel time, but kernel time is <3% of total time. The real savings come from eliminating Python/NumPy overhead.

---

## 4. Fused Bridge Operations

### 4.1 Operation Taxonomy

**Level 0: Atomic Operations**
- φ: int8 → uint8 (add 1)
- φ⁻¹: uint8 → int8 (sub 1)
- tadd_kernel, tmul_kernel, etc.

**Level 1: Fused Binary (existing)**
- tnot(tadd(a,b)), tnot(tmul(a,b)), etc.

**Level 2: Fused Bridge (NEW)**
- tadd_int8: φ⁻¹(tadd_kernel(φ(a), φ(b)))
- tmul_int8: φ⁻¹(tmul_kernel(φ(a), φ(b)))
- etc.

**Level 3: Fully Fused (NEW)**
- tnot_tadd_int8: φ⁻¹(tnot(tadd_kernel(φ(a), φ(b))))
- chain_int8: φ⁻¹(op_n(...(op_1(φ(a), φ(b)))...))

### 4.2 SIMD Implementation Pattern

```cpp
template <bool Sanitize = true>
static inline __m256i bridge_binary_op(
    __m256i a_int8,
    __m256i b_int8,
    __m256i lut
) {
    const __m256i one = _mm256_set1_epi8(1);

    // φ: int8 → uint8 (in registers, no memory)
    __m256i a = _mm256_add_epi8(a_int8, one);
    __m256i b = _mm256_add_epi8(b_int8, one);

    // Kernel operation (canonical indexing)
    __m256i result = binary_simd_op<Sanitize>(a, b, lut);

    // φ⁻¹: uint8 → int8 (in registers, no memory)
    return _mm256_sub_epi8(result, one);
}
```

### 4.3 Cost Analysis

**Per 32 elements:**
```
Operation          Cycles   Memory Traffic
─────────────────────────────────────────
Load a_int8        1        32 bytes read
Load b_int8        1        32 bytes read
Add 1 to a         1        0 (register)
Add 1 to b         1        0 (register)
Load CANON_A       0        0 (cached in register)
Load CANON_B       0        0 (cached in register)
Shuffle a          1        0 (register)
Shuffle b          1        0 (register)
Add indices        1        0 (register)
Load LUT           0        0 (cached in register)
Shuffle result     1        0 (register)
Sub 1              1        0 (register)
Store result       1        32 bytes write
─────────────────────────────────────────
TOTAL              10       96 bytes
```

**Comparison:**
```
Approach           Cycles   Memory    Allocations
─────────────────────────────────────────────────
NumPy naive        ~100     ~320      5
Fused C++ bridge   10       96        0
─────────────────────────────────────────────────
Speedup            10x      3.3x      ∞
```

---

## 5. Extended Bridge: Multi-Format Support

### 5.1 Format Lattice

```
                    Int8 (semantic)
                      │
                      │ φ (cost: 1)
                      ▼
                  Uint8_2bit (compute)
                   /      \
    pack243 (10) /          \ pack4 (future)
               /              \
          Dense243            Packed4
        (storage)           (compression)
```

### 5.2 Lazy Evaluation Strategy

Instead of eagerly converting, track format and convert lazily:

```python
class TernaryTensor:
    def __init__(self, data, format='int8'):
        self._data = data
        self._format = format

    def _ensure_format(self, target):
        if self._format != target:
            self._data = convert(self._data, self._format, target)
            self._format = target

    def tadd(self, other):
        # Compute in optimal format
        self._ensure_format('uint8_2bit')
        other._ensure_format('uint8_2bit')
        result = te.tadd(self._data, other._data)
        return TernaryTensor(result, 'uint8_2bit')

    def to_numpy(self):
        self._ensure_format('int8')
        return self._data
```

### 5.3 Computation Graph Optimization

For complex expressions, build a graph and optimize:

```python
# User writes:
result = tnot(tadd(a, tmul(b, c)))

# System builds graph:
#   a ─────────────────┐
#   b ─┬─[tmul]─┐      │
#   c ─┘        └─[tadd]─┬─[tnot]─► result
#               ────────┘

# Optimizer rewrites to:
#   a ─┐
#   b ─┼─[fused_tnot_tadd_tmul]─► result
#   c ─┘

# With format awareness:
#   a_int8 ─┐
#   b_int8 ─┼─[fused_tnot_tadd_tmul_int8]─► result_int8
#   c_int8 ─┘
```

---

## 6. Implementation Roadmap

### Phase 1: Core Bridge (Immediate)
- [ ] `fused_bridge_ops.h`: Fused int8 operations in C++
- [ ] Extend `bindings_core_ops.cpp` with `tadd_int8`, etc.
- [ ] Benchmark: Expect 10-50x speedup over current pipeline

### Phase 2: Backend Integration
- [ ] Add int8 variants to `backend_plugin_api.h`
- [ ] Implement for scalar, AVX2 backends
- [ ] Runtime dispatch based on input format

### Phase 3: Lazy Tensor Wrapper
- [ ] Python `TernaryTensor` class with format tracking
- [ ] Deferred execution model
- [ ] Automatic format optimization

### Phase 4: Graph Compiler
- [ ] Expression DAG construction
- [ ] Cost-based optimization pass
- [ ] Code generation for fused operations

---

## 7. Mathematical Guarantees

### 7.1 Correctness Theorem

**Theorem:** For any ternary operation ⊙ and inputs a, b ∈ T:
```
fused_bridge(a, b) = naive_pipeline(a, b)
```

**Proof:**
```
fused_bridge(a, b)
  = φ⁻¹(kernel(φ(a), φ(b)))           [by definition of fused]
  = φ⁻¹(φ(a ⊙ b))                      [by homomorphism property]
  = a ⊙ b                              [by φ⁻¹ ∘ φ = id]
  = naive_pipeline(a, b)               [by definition of naive]
∎
```

### 7.2 Performance Theorem

**Theorem:** The fused bridge is asymptotically optimal for single operations.

**Proof:**
Lower bound: Any computation must:
1. Read inputs: Ω(n) memory traffic
2. Write output: Ω(n) memory traffic
3. Compute: Ω(n) operations

Fused bridge achieves:
1. Read inputs: O(n) - exactly n bytes per input
2. Write output: O(n) - exactly n bytes
3. Compute: O(n) - constant operations per element

Therefore: fused_bridge ∈ Θ(n) which is optimal.
∎

---

## 8. Conclusion

The Bridge Layer is a **functorial mapping** between representation categories that:

1. **Preserves algebraic structure** through the isomorphism φ
2. **Minimizes computational cost** through operation fusion
3. **Enables graph-level optimization** through lazy evaluation
4. **Provides mathematical guarantees** of correctness and optimality

The key insight is that the +1/-1 conversions, while trivial in isolation, become expensive when executed through Python/NumPy due to allocation overhead. By fusing them with the SIMD kernel in C++, we eliminate this overhead entirely.

**Expected Performance Improvement:**
- Current: 97% time in conversion, 3% in kernel
- With bridge: 0% in conversion (fused), 100% in kernel
- Net speedup: **~30x for full pipeline**

---

## References

1. Mac Lane, S. "Categories for the Working Mathematician" - Functor theory
2. Cormen et al. "Introduction to Algorithms" - Graph optimization
3. Intel Intrinsics Guide - SIMD operations cost model
4. NumPy internals - Allocation overhead analysis

---

**Status:** DESIGN COMPLETE
**Next:** Implementation of Phase 1 (Core Bridge)
