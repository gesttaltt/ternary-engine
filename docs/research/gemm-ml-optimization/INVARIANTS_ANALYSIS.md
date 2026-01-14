# Invariants of ML-Discovered Matrix Multiplication Algorithms

**Doc-Type:** Deep Analysis · Version 1.0 · Updated 2025-12-29

Examining the mathematical invariants and deeper substratum underlying AlphaTensor's discoveries.

---

## The Fundamental Insight

**Why do fewer multiplications work?**

Matrix multiplication is **bilinear** - linear in both arguments. This algebraic property means:

```
C = A × B  can be rewritten as  C = Σᵢ (linear_combo_of_A)ᵢ × (linear_combo_of_B)ᵢ × (output_pattern)ᵢ
```

The trick: Form clever **linear combinations BEFORE multiplying**, then linearly recombine the products.

---

## Tensor Representation

### The 3D Tensor View

Any bilinear map can be represented as a 3D tensor. For n×n matrix multiplication:

```
T ∈ ℝ^(n² × n² × n²)

where T[i,j,k] encodes: "Does input element i times input element j contribute to output element k?"
```

### Rank = Multiplications

**Critical theorem**: The tensor rank R(T) equals the minimum number of multiplications needed.

| Matrix Size | Naive Multiplications | Best Known Rank | Algorithm |
|-------------|----------------------|-----------------|-----------|
| 2×2 | 8 | 7 | Strassen (1969) |
| 3×3 | 27 | 23 | Laderman (1976) |
| 4×4 | 64 | 47 | **AlphaTensor (2022)** |
| 5×5 | 125 | 96 | AlphaTensor (2022) |

---

## The Invariants

### 1. Rank Preservation Under Symmetry

Two decompositions are **equivalent** if one can be obtained from the other by:

**Sign Flipping**: For λ₁, λ₂, λ₃ ∈ {−1, +1} where λ₁λ₂λ₃ = 1:
```
(λ₁u, λ₂v, λ₃w) ≡ (u, v, w)
```
This preserves the rank-one tensor since (λ₁u ⊗ λ₂v ⊗ λ₃w) = (λ₁λ₂λ₃)(u ⊗ v ⊗ w) = (u ⊗ v ⊗ w)

**Order Invariance**: Factor sequence doesn't matter (summation is commutative)
```
Σᵢ (uᵢ ⊗ vᵢ ⊗ wᵢ) = Σ_π(i) (u_π(i) ⊗ v_π(i) ⊗ w_π(i))
```

**Basis Change**: Invertible transformations preserve rank
```
T' = (A ⊗ B ⊗ C) · T  has same rank as T
```

### 2. Invariants That Distinguish Non-Equivalent Algorithms

AlphaTensor found **14,236 non-equivalent** factorizations for 4×4 matrices. They differ in:

1. **Matrix ranks of components**: The rank distribution across factor matrices
2. **Sparsity patterns**: Which entries are zero
3. **Coefficient sets**: Which values appear (e.g., {-2,-1,0,1,2} vs {-1,0,1})
4. **Structural symmetries**: Cyclic, reflection, or no symmetry

---

## The Deeper Substratum

### Why Strassen Works (Algebraically)

The Karatsuba trick for polynomials reveals the pattern:

```python
# Naive: 4 multiplications
(a₀ + a₁x)(b₀ + b₁x) = a₀b₀ + (a₀b₁ + a₁b₀)x + a₁b₁x²

# Karatsuba: 3 multiplications
m₀ = a₀b₀
m₁ = a₁b₁
m₂ = (a₀ + a₁)(b₀ + b₁)  # = a₀b₀ + a₀b₁ + a₁b₀ + a₁b₁

Result = m₀ + (m₂ - m₀ - m₁)x + m₁x²
```

**The insight**: We "borrow" terms from multiple products and cancel them out with additions.

Strassen applies this same principle to 2×2 block matrices.

### The Geometric View

**Secant varieties of Segre varieties** stratify tensor space by rank:

```
σ₁(Seg) ⊂ σ₂(Seg) ⊂ ... ⊂ σᵣ(Seg) ⊂ ...

where σᵣ = tensors of rank ≤ r
```

Finding optimal algorithms = finding which σᵣ contains the matrix multiplication tensor.

**Border rank** provides even tighter bounds (approximate decompositions).

---

## What AlphaTensor Actually Found

### Structural Discoveries

1. **Discrete Fourier Transform Rediscovery**: For cyclic convolution, AlphaTensor independently rediscovered DFT
   - Shows fundamental mathematical structures emerge from optimization

2. **Scaling Patterns for Structured Matrices**: For skew-symmetric n×n matrices:
   ```
   Multiplications ≈ (n-1)(n+2)/2 ≈ n²/2
   ```
   - Half the naive count, pattern extends to arbitrary n

3. **Hardware-Aware Trade-offs**: Different decompositions optimal for GPU vs TPU
   - Same tensor rank, different practical performance
   - Memory access patterns matter more than operation count

### The 14,000+ Algorithms

All 14,236 non-equivalent 4×4 algorithms share:
- Same tensor rank (47)
- Same correctness (verified)
- Different sparsity, coefficient, symmetry patterns

This suggests a vast **equivalence class landscape** - the space of optimal algorithms is not a point but a manifold.

---

## Implications for Ternary Computing

### Ternary Matrix Multiplication Tensor

For ternary matmul with inputs/weights in {-1, 0, +1}:

```
T_ternary ∈ {-1, 0, +1}^(n² × n² × n²)

Constraint: All coefficients in {-1, 0, +1}
```

**Key question**: Does restricting to ternary coefficients change the optimal rank?

### Potential Advantages

1. **Simpler Arithmetic**:
   - Ternary × Ternary → Ternary (9 cases, vs infinite for real)
   - Addition/subtraction only (no actual multiplication needed)

2. **Sparser Decompositions**:
   - Zero coefficients eliminate terms entirely
   - May enable lower effective rank

3. **Finite Search Space**:
   - With ternary coefficients, AlphaTensor-style search is discrete
   - May be more tractable than real-valued case

### The Ternary Bilinear Structure

```python
# Ternary multiplication is a bilinear map:
tmul: {-1,0,+1} × {-1,0,+1} → {-1,0,+1}

# Truth table (9 entries):
#     -1   0  +1
# -1 [+1,  0, -1]
#  0 [ 0,  0,  0]
# +1 [-1,  0, +1]

# This is just sign(a) × sign(b) with saturation
```

The tensor for ternary n×n matmul has special structure:
- **Extreme sparsity potential**: Many zero products
- **Sign symmetry**: Negation patterns are preserved
- **Absorption**: Zeros propagate (0 × anything = 0)

---

## Research Directions

### 1. Ternary-Specific AlphaTensor

Train RL agent with ternary coefficient constraint:
- Search space: decompositions with factors in {-1, 0, +1}
- May find algorithms impossible with real coefficients

### 2. Exploit Zero Structure

For ternary weights with ~30% zeros (typical in quantized NNs):
- Sparse tensor decomposition
- Skip multiplications with zero factors
- Asymptotically faster for sparse inputs

### 3. Hardware Co-Design

Design decompositions for:
- SIMD: Maximize parallelism per instruction
- Cache: Minimize memory traffic patterns
- Energy: Prefer additions over multiplications

---

## Key References

1. [AlphaTensor Nature Paper](https://www.nature.com/articles/s41586-022-05172-4)
2. [Geometric Rank of Tensors](https://arxiv.org/abs/2002.09472)
3. [Complexity of Matrix Multiplication](https://conferences.mpi-inf.mpg.de/adfocs-17/material/FLG_H1.pdf)
4. [Strassen's Algorithm Algebraic Analysis](https://link.springer.com/article/10.1007/s11565-019-00318-1)
5. [Geometry and Complexity of Matrix Multiplication (Landsberg)](https://www.ams.org/journals/bull/2008-45-02/S0273-0979-08-01176-2/S0273-0979-08-01176-2.pdf)

---

## Summary: The Deep Pattern

**The invariant that enables faster matrix multiplication**:

> Bilinearity allows pre-computation sharing. Linear combinations of inputs can be formed before multiplication, and the products can be linearly recombined. The tensor rank—not naive element counting—determines the true minimum.

**For ternary computing**:

> The discrete {-1, 0, +1} structure may admit even sparser decompositions. The constraint becomes an advantage: finite search space, natural sparsity, and sign-only arithmetic.

**The 14,000+ non-equivalent algorithms prove**:

> The space of optimal solutions is rich, not unique. Hardware constraints (cache, SIMD, energy) select among mathematically equivalent options. ML can explore this space; humans cannot.

---

**Version**: 1.0 · **Updated**: 2025-12-29 · **Scope**: Mathematical Foundations
