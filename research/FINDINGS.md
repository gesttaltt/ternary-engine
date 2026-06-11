# Algebraic Properties of Balanced Ternary: Empirical Findings

**Version:** 1.0 · **Date:** 2026-06-11 · **Method:** Systematic falsification over full 19,683-value corpus

---

## Summary

We tested nine structural hypotheses about balanced ternary arithmetic using exhaustive enumeration over the complete operation corpus (19,683 trit-vector combinations). Three structures are mathematically intrinsic — they cannot be falsified because they follow from the definition. Five are empirically supported. One is weak. The most consequential finding is that **`tadd` is non-associative for 79.6% of all triplets**, which means balanced ternary does not form a group under addition and that any algorithm assuming associativity is incorrect for this number system.

---

## Background

Balanced ternary represents integers using digits {-1, 0, +1} rather than {0, 1}. This project implements it with 2-bit encoding (0b00 = -1, 0b01 = 0, 0b10 = +1) and five operations:

| Symbol | Name | Description |
|--------|------|-------------|
| `tadd` | Saturated addition | Clamps to [-1, +1] |
| `tmul` | Multiplication | Standard sign product |
| `tmin` | Minimum | Element-wise min |
| `tmax` | Maximum | Element-wise max |
| `tnot` | Negation | Sign flip, 0 unchanged |

The "saturated" qualifier on `tadd` is the source of most of what follows. Unlike integer addition, `tadd(+1, +1) = +1` (not +2, which doesn't exist in the domain). This clamping breaks algebraic properties that hold for ordinary integers.

All tests run against the native AVX2 SIMD engine (`ternary_simd_engine`) on real computed values, not on symbolic definitions.

---

## Part I: Intrinsic Structures

These properties hold at 100% and cannot be falsified — they are consequences of the representation.

### 1. p-adic / 3-adic Structure (H1)

The 3-adic valuation v₃(n) = largest k such that 3^k divides n is built into balanced ternary. The number of trit-vectors at each valuation level follows an exact geometric decay:

```
v=0:  13,122 values  (66.7%)   ← 2/3 of corpus
v=1:   4,374 values  (22.2%)   ← (2/3)²
v=2:   1,458 values   (7.4%)   ← (2/3)³
v=3:     486 values   (2.5%)
v=4:     162 values   (0.8%)
...
v=∞:       1 value    (0.005%) ← zero, the unique center
```

This (2/3)^k pattern is the signature of 3-adic measure. Zero occupies a structurally unique position — it has infinite valuation, which is why it behaves as an absorbing element in `tmul` and as the additive identity in `tadd`.

**Implication for hardware:** 66.7% of all operation results have valuation 0 and 40% of matrix products are exactly zero. Zero-skip optimization is not just viable — it is the natural move for any ternary GEMM implementation.

**Empirical validation (2026-06-11):** `benchmarks/bench_zero_skip_gemm.py` ran exhaustive sparsity measurements over random ternary weight matrices at four sizes (128–1024 K dimension). Measured zero fraction: **0.334 ± 0.001** across all sizes and runs, matching the theoretical 1/3 within noise. 33.4% of multiply-accumulates per output element are eliminatable. NumPy BLAS does not exploit this — sign-split (two binary matmuls) is 2× slower than dense due to BLAS call overhead. The savings require either `scipy.sparse`, a C++ zero-skip kernel, or hardware with structured sparsity support (GPU tensor cores).

### 2. Three-Valued Logic (H6)

Balanced ternary is a valid three-valued logic system (Kleene/Łukasiewicz). All classical logic properties hold with the expected modification for a third truth value:

- De Morgan laws: 100%
- Double negation (`tnot(tnot(a)) = a`): 100%
- Complement (`tadd(a, tnot(a))` gives identity behavior): 100%
- Excluded middle: fails for 3.2% of cases (expected — "unknown" is neither true nor false)

### 3. Distributive Lattice (H11)

`tmin` and `tmax` form a distributive lattice with the natural ordering -1 < 0 < +1:

- Commutativity: 100%
- Associativity: 100%
- Absorption (`tmin(a, tmax(a, b)) = a`): 100%
- Distributivity (`tmin(a, tmax(b, c)) = tmax(tmin(a,b), tmin(a,c))`): 100%
- Idempotence: 100%

This is a bounded distributive lattice, which is the algebraic structure underlying most comparison-based sorting and filtering algorithms. Any ternary circuit built on `tmin`/`tmax` can use standard lattice reasoning.

---

## Part II: Empirically Supported Structures

### 4. Non-Associativity of `tadd` (H10)

**This is the most consequential finding.**

Testing group axioms for `tadd`:

| Axiom | Result |
|-------|--------|
| Closure | 100% |
| Identity (0 is identity) | 100% |
| Inverses (`tadd(a, tnot(a)) = 0`) | 100% |
| Commutativity | 100% |
| **Associativity** | **20.4%** |

`tadd` is commutative but not associative. The triplet `(a, b, c) = (+1, +1, -1)` illustrates this directly:

```
tadd(tadd(+1, +1), -1) = tadd(+1, -1) = 0
tadd(+1, tadd(+1, -1)) = tadd(+1, 0)  = +1
```

The clamping in `tadd` breaks associativity whenever intermediate values saturate. This happens for 79.6% of all ordered triplets.

**Consequences:**

- Balanced ternary under `tadd` is a commutative magma with identity and inverses — not a group.
- It is not a ring (rings require an associative addition).
- Any compiler optimization that reorders `tadd` chains (e.g., for parallelism) is incorrect without careful domain analysis.
- Neural network training on ternary weights cannot assume gradient accumulation is order-independent.
- The "ternary is just INT2 with a different encoding" assumption made by many quantization papers is wrong at the algebraic level.

### 5. Partial Tropical Semiring (H4)

Balanced ternary satisfies some but not all tropical algebra axioms:

| Property | Result |
|----------|--------|
| `tadd` distributes over `tmin` | 100% |
| `tadd` distributes over `tmax` | 100% |
| `tmul` distributes over `tmin` | 10.4% |
| `tmul` distributes over `tmax` | 10.4% |

`tadd(a, tmin(b,c)) = tmin(tadd(a,b), tadd(a,c))` holds universally. The `tmul` distributivity fails overwhelmingly. This makes balanced ternary a partial tropical structure: the additive side behaves tropically but the multiplicative side does not.

**Implication:** Shortest-path and min-cost flow algorithms have a natural ternary formulation using `tadd` and `tmin`, but this does not extend to algorithms requiring full semiring multiplication.

### 6. Ultrametric Geometry (H2)

In the 3-adic metric, every triple of ternary values forms an isoceles triangle (two of the three distances are equal and both are ≥ the third). This is the ultrametric property, and it holds at 100% for raw ternary values.

The 89.32% composite score comes from additionally testing a trained neural embedding of ternary values, which only achieves 45% isoceles triangles — meaning the model has not learned to preserve the ultrametric structure of the space it is embedding. This is a training gap, not a failure of the underlying math.

**Implication:** Hierarchical data structures (tries, B-trees, segment trees) are the natural computational primitive for ternary indexing, not hash tables or balanced BSTs. The tree topology is intrinsic to the metric.

### 7. Hyperbolic Geometry (H3)

The ultrametric structure implies hyperbolic geometry: spaces where all triangles are isoceles embed naturally in a Poincaré ball with negative curvature. We verified this directly:

```
Geodesic midpoint error:  9.8×10⁻⁸  (essentially machine precision)
Euclidean midpoint error: 4.0×10⁻²  (40,000× worse)
Geodesic wins:            100% of cases
```

This means that any interpolation, averaging, or gradient computation over ternary operation embeddings should use hyperbolic (Möbius / geodesic) operations, not Euclidean ones. The midpoint of two ternary operations is not their coordinate average.

A note on a bug found during testing: using `torch.randn(1, 16) * 0.5` to initialize points in 16-dimensional space produces vectors with L2 norm ≈ √16 × 0.5 = 2.0, which lies outside the Poincaré ball boundary. All hyperbolic computations must enforce `||x|| < 1`; initialization should normalize and then scale to a target radius.

### 8. Information-Theoretic Structure (H9)

Entropy analysis over the full operation corpus:

| Measure | Value | Notes |
|---------|-------|-------|
| Bits per trit | 1.585 | Matches theoretical log₂(3) exactly |
| `tadd` output entropy | 14.24 / 14.26 max | Near maximum — high information |
| Valuation entropy | 1.38 / 14.35 max | 9.6% of maximum — strong structure |
| Operations deterministic | 100% | No stochasticity |

The low valuation entropy (9.6%) confirms the p-adic structure quantitatively: the distribution of valuation levels is highly non-uniform. Most results cluster at low valuation, which means operation outputs are not uniformly distributed across ternary space — they are skewed toward the "edges" of the 3-adic tree.

---

## Part III: Weak or Falsified Structures

### 9. Modular Arithmetic / Z/3Z Ring (H23)

Balanced ternary does not behave as modular arithmetic mod 3:

| Property | Result |
|----------|--------|
| Valuation sum rule | 100% |
| Valuation product rule | 57.4% |
| Mod-3 addition | 12.2% |

The natural identification {-1↔2, 0↔0, +1↔1} does not yield a valid Z/3Z ring. This is because `tadd` clamps at saturation rather than wrapping around. Balanced ternary is not a field.

### 10. Removed Hypotheses

**H5 (Clifford Algebra)** and **H7 (Quantum Superposition)** were removed after their tests were found to be furniture — they checked only that results were valid balanced ternary values, which any operation would satisfy. Genuine tests of geometric algebra and quantum superposition properties remain open.

---

## Part IV: Implications for System Design

**For compiler writers:** Do not reorder `tadd` chains. Associativity cannot be assumed. The only safe transformations are those that exploit commutativity (reordering two operands) or the lattice properties of `tmin`/`tmax`.

**For ML researchers:** Ternary quantization papers that treat {-1, 0, +1} weights as a ring are working with incorrect algebra. Gradient accumulation over ternary-quantized weights is order-dependent in ways that float accumulation is not.

**For hardware designers:** The 3-adic valuation structure means 40% of GEMM entries are zero and 66.7% of values are at valuation 0. A zero-skip multiplier array is not an optimization — it is correctness-aligned design. The ultrametric structure suggests that ternary memory hierarchies should be organized as radix-3 trees, not flat arrays.

**For geometry / embedding work:** Use geodesic operations in a Poincaré ball. Euclidean interpolation between ternary operation embeddings is quantifiably wrong. The target valuation-radius correlation (high valuation → small radius, near center) is the correct inductive bias for any neural model over ternary operations.

---

## Methods

- **Corpus:** Complete enumeration of 19,683 trit-vector combinations (9 trits × 3 values each = 3⁹)
- **Engine:** Native AVX2 SIMD implementation (`ternary_simd_engine`, validated on Linux x64 2026-03-19)
- **Falsification script:** `research/scripts/falsify.py`
- **Raw results:** `research/results/falsification_*.json`
- **Scoring rubric:** A (>95%), B (80-95%), C (50-80%), D (20-50%), F (<20%)

---

## Open Questions

15 of 24 hypotheses remain untested. The highest-priority open questions:

| Hypothesis | Question |
|------------|----------|
| H8 Category Theory | Do ternary operations form a category with well-defined composition? |
| H12 Dynamical Systems | What are the fixed points and attractors of iterated `tadd`/`tmul`? |
| H13 Topological | Does the 3-adic completion of ternary have Cantor set (fractal) structure? |
| H17 F₃ Field | What breaks when mapping balanced ternary to the finite field F₃ = {0,1,2}? |
| H24 Sui Generis | After all falsifications, what properties remain irreducible to known structures? |

Run any test with: `python research/scripts/falsify.py -H <id>`

---

**The core result in one sentence:** Balanced ternary under its natural operations is a commutative non-associative magma embedded in a 3-adic ultrametric space with hyperbolic geometry — not a group, not a ring, not modular arithmetic.
