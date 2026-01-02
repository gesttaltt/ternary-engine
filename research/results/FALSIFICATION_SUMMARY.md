# Ternary Semantic Falsification - Results Summary

**Doc-Type:** Research Results · Version 1.2 · Updated 2025-01-02

---

## Philosophy

We do not search for solutions. We LISTEN to errors.
By falsifying hypotheses, we discover truth through negative space.

---

## Test Results (2025-01-02)

| Hypothesis | Score | Grade | Status | Key Finding |
|------------|-------|-------|--------|-------------|
| H1 p-adic/3-adic | 100% | A | INTRINSIC | Built into ternary representation |
| H2 Ultrametric Tree | 89.32% | B | SUPPORTED | Raw=100%, model=45% isoceles |
| H3 Hyperbolic/Poincaré | 99.80% | A | SUPPORTED | Math correct, VRC=0.035 |
| H4 Tropical Algebra | 87.20% | B | SUPPORTED | tadd distributes, tmul doesn't |
| H6 Three-Valued Logic | 100% | A | INTRINSIC | De Morgan laws hold perfectly |
| H9 Information Theory | 90.91% | B | SUPPORTED | Entropy confirms p-adic structure |
| H10 Group Theory | 84.08% | B | SUPPORTED | **tadd non-associative (20%)** |
| H11 Lattice/Order | 100% | A | INTRINSIC | tmin/tmax form distributive lattice |
| H23 Modular Arithmetic | 56.53% | C | WEAK | Products fail, mod-3 addition fails |

---

## Critical Discoveries

### 1. Intrinsic Structures (100% - Cannot Be Falsified)

**H1 (p-adic)**, **H6 (Three-Valued Logic)**, and **H11 (Lattice)** are mathematically intrinsic to balanced ternary:

- **p-adic structure**: The 3-adic valuation v₃(n) = max k where 3^k divides n is built into ternary representation
- **Three-valued logic**: De Morgan laws, double negation all hold (excluded middle fails as expected)
- **Lattice structure**: tmin/tmax form a distributive lattice with absorption, idempotence, distributivity

These are not hypotheses to test - they are mathematical facts about the ternary number system.

### 2. Ultrametric Property (H2)

**Key insight**: Raw ternary values ARE perfectly ultrametric (100% isoceles triangles in p-adic metric).

The 89.32% score comes from testing the TRAINED MODEL embeddings, which only achieve 45% isoceles. This reveals:

- **The math is correct** - ternary values intrinsically satisfy ultrametric inequality
- **The model needs training** - current embeddings don't preserve ultrametric structure
- **Training gap identified** - loss function was missing radial alignment (now fixed)

### 3. Hyperbolic Geometry (H3)

**Bug found and fixed**: The test was using improperly scaled points in 16D that landed on the Poincare ball boundary, causing numerical instability.

After fix:
- Geodesic midpoint: error = 9.8e-08 (essentially perfect)
- Euclidean midpoint: error = 0.04 (worse as expected)
- Geodesic wins: 100% of cases

**Model gap**: Valuation-radius correlation (VRC) is 0.035 instead of target -0.8. The model hasn't learned to position high-valuation values near the center.

### 4. Tropical Algebra (H4) - PARTIAL

Ternary forms a **partial tropical semiring**:
- **tadd distributes over tmin/tmax**: 100% (both directions)
- **tmul does NOT distribute**: 10.4% over tmin, 10.4% over tmax
- **Grade B (87.20%)** - strong but partial tropical structure

This means `tadd(a, tmin(b,c)) = tmin(tadd(a,b), tadd(a,c))` but NOT for tmul.

### 5. Three-Valued Logic (H6) - INTRINSIC

All three-valued logic properties hold perfectly:
- **De Morgan laws**: 100%
- **Double negation**: 100%
- **Complement**: 100%
- **Excluded middle fails**: Only 3.2% (expected for ternary with three values)

### 6. Information Theory (H9)

Entropy analysis confirms p-adic structure:
- **Operation entropies**: add near max (14.24/14.26), mul/min/max lower
- **Valuation entropy**: 1.38 (only 9.6% of max) - strong p-adic structure
- **Operations deterministic**: 100%
- **Bits per trit**: 1.585 (theoretical log₂(3))

### 7. Group Theory (H10) - NON-ASSOCIATIVE

**Major discovery**: Balanced ternary with tadd is NOT a group!
- **Closure**: 100%
- **Identity**: 100%
- **Inverses**: 100%
- **Commutativity**: 100%
- **Associativity**: Only **20.4%**!

`tadd(tadd(a,b), c) ≠ tadd(a, tadd(b,c))` for 79.6% of triplets.

This is a fundamental property that affects all algorithms assuming associativity.

### 8. Modular Arithmetic (H23) - WEAK

- **Valuation sum**: Works (100%)
- **Valuation product**: Fails (57.4%)
- **Mod-3 addition**: Fails (12.2%)

Ternary values do NOT form a proper Z/3Z ring.

---

## Valuation Distribution (Corpus)

```
v=0: 13,122 values (66.7%)  ← 2/3 of all values
v=1:  4,374 values (22.2%)  ← (2/3)²
v=2:  1,458 values (7.4%)   ← (2/3)³
v=3:    486 values (2.5%)
v=4:    162 values (0.8%)
v=5:     54 values (0.3%)
v=6:     18 values (0.09%)
v=7:      6 values (0.03%)
v=8:      2 values (0.01%)
v=∞:      1 value  (0.005%) - zero
```

This (2/3)^k pattern is the signature of 3-adic structure.

---

## Fixes Applied

### 1. H3 Test Fix (falsify.py)
```python
# BEFORE (wrong - points land on boundary in 16D)
x = torch.randn(1, 16) * 0.5  # norm ~ sqrt(16)*0.5 = 2.0 -> boundary!

# AFTER (correct - proper scaling)
x_dir = torch.randn(1, dim)
x = x_dir / x_dir.norm() * target_radius * torch.rand(1).item()
```

### 2. Loss Function Fix (hyperbolic_ops.py)
Added missing radial alignment and ultrametric losses to `HyperbolicOperationLoss.forward()`:
- `radial_alignment_loss` (weight 0.5) - enforces valuation-radius correlation
- `attractor_ultrametric_loss` (weight 0.3) - enforces isoceles property

### 3. tnot Operation (falsify.py)
Created `tnot_batch()` helper since `simd_batch_operation` doesn't support 'not':
```python
def tnot_batch(indices):
    results = np.zeros_like(indices)
    for i, idx in enumerate(indices):
        trits = index_to_trits(idx, 9)
        neg_trits = -trits  # Negate each trit
        results[i] = trits_to_index(neg_trits)
    return results
```

---

## Component Wiring

All tests use REAL components:

| Component | Module | Status |
|-----------|--------|--------|
| SIMD Engine | `ternary_simd_engine` | OK |
| Hyperbolic VAE | `hyperbolic_ops.py` | OK |
| Ultrametric Energy | `ebm/ultrametric_energy.py` | OK |
| Operation LUTs | `data.py` | OK (2M samples) |
| Corpus | 19,683 values | OK |
| Trained Model | `best_model.pt` | Needs retraining |

---

## Next Steps

### Implement Remaining Tests (15 of 24)
- **Tier 2**: H5 (Clifford), H7 (Quantum), H8 (Category)
- **Tier 3**: H12 (Dynamical), H13 (Topological), H14 (Neural)
- **Tier 4**: H15-H22, H24

### Train Model with Fixed Loss
- Resume training with radial_alignment_loss and attractor_ultrametric_loss
- Target: VRC < -0.5 for H3, >80% isoceles for H2

### Test Multiple Models
- v5_11_3 (arithmetic centering)
- homeostasis (radial hierarchy)
- codon_encoder (biological hierarchy)

---

## Files

- `research/scripts/falsify.py` - Main falsification framework
- `research/TERNARY_SEMANTIC_HYPOTHESES.md` - All 24 hypothesis definitions
- `research/results/falsification_*.json` - Raw test results
- `research/results/SESSION_STATE.md` - Session resume document

---

**Conclusion**: 9 of 24 hypotheses tested. Three are intrinsic (H1, H6, H11), five are supported (H2, H3, H4, H9, H10), one is weak (H23). Major discovery: tadd is non-associative (20% only).
