# Ternary Semantic Falsification - Results Summary

**Doc-Type:** Research Results · Version 1.1 · Updated 2025-01-02

---

## Philosophy

We do not search for solutions. We LISTEN to errors.
By falsifying hypotheses, we discover truth through negative space.

---

## Test Results (2025-01-02)

| Hypothesis | Score | Grade | Status | Key Finding |
|------------|-------|-------|--------|-------------|
| H1 p-adic/3-adic | 100% | A | SUPPORTED | **Intrinsic** to ternary values |
| H2 Ultrametric Tree | 88.52% | B | SUPPORTED | Raw values 100% isoceles, model 44% |
| H3 Hyperbolic/Poincare | 99.80% | A | SUPPORTED | Math correct, model VRC needs training |
| H4 Tropical Algebra | 54.35% | D | WEAK | tadd distributes over tmin/tmax (100%), tmul does NOT (10%) |
| H6 Three-Valued Logic | 100% | A | SUPPORTED | **Intrinsic** - De Morgan, double negation all hold |
| H9 Information Theory | 90.91% | B | SUPPORTED | Valuation entropy 9.6% of max, confirms p-adic |
| H10 Group Theory | 84.16% | B | SUPPORTED | **tadd non-associative (20%)** - major finding |
| H11 Lattice/Order | 100% | A | SUPPORTED | **Intrinsic** to tmin/tmax |
| H23 Modular Arithmetic | 56.53% | C | WEAK | Works for valuation sums, fails for products |

---

## Critical Discoveries

### 1. Intrinsic Structures (100% - Cannot Be Falsified)

**H1 (p-adic)** and **H11 (Lattice)** are mathematically intrinsic to balanced ternary:

- **p-adic structure**: The 3-adic valuation v3(n) = max k where 3^k divides n is built into ternary representation
- **Lattice structure**: tmin/tmax form a distributive lattice with all properties (absorption, idempotence, distributivity) holding perfectly

These are not hypotheses to test - they are mathematical facts about the ternary number system.

### 2. Ultrametric Property (H2)

**Key insight**: Raw ternary values ARE perfectly ultrametric (100% isoceles triangles in p-adic metric).

The 88.88% score comes from testing the TRAINED MODEL embeddings, which only achieve 44% isoceles. This reveals:

- **The math is correct** - ternary values intrinsically satisfy ultrametric inequality
- **The model needs training** - current embeddings don't preserve ultrametric structure
- **Training gap identified** - loss function was missing radial alignment (now fixed)

### 3. Hyperbolic Geometry (H3)

**Bug found and fixed**: The test was using improperly scaled points in 16D that landed on the Poincare ball boundary, causing numerical instability.

After fix:
- Geodesic midpoint: error = 9.5e-08 (essentially perfect)
- Euclidean midpoint: error = 0.04 (worse as expected)
- Geodesic wins: 100% of cases

**Model gap**: Valuation-radius correlation (VRC) is 0.035 instead of target -0.8. The model hasn't learned to position high-valuation values near the center.

### 4. Tropical Algebra (H4) - PARTIAL

Ternary forms a **partial tropical semiring**:
- **tadd distributes over tmin/tmax**: 100% (both directions)
- **tmul does NOT distribute**: 10.4% over tmin, 10.4% over tmax
- **Grade D (54.35%)** - only partial tropical structure

This means `tadd(a, tmin(b,c)) = tmin(tadd(a,b), tadd(a,c))` but NOT for tmul.

### 5. Three-Valued Logic (H6) - INTRINSIC

All three-valued logic properties hold perfectly:
- **De Morgan laws**: 100%
- **Double negation**: 100%
- **Complement**: 100%
- **Excluded middle fails**: Expected (ternary has three values)

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
- **Associativity**: Only 20.4%!

`tadd(tadd(a,b), c) ≠ tadd(a, tadd(b,c))` for 79.6% of triplets.

This is a fundamental property that was previously assumed.

### 8. Modular Arithmetic (H23) - WEAK

- **Valuation sum**: Works (100%)
- **Valuation product**: Fails (57.4%)
- **Mod-3 addition**: Fails (12.2%)

Ternary values do NOT form a proper Z/3Z ring.

---

## Valuation Distribution (Corpus)

```
v=0: 13,122 values (66.7%)
v=1:  4,374 values (22.2%)
v=2:  1,458 values (7.4%)
v=3:    486 values (2.5%)
v=4:    162 values (0.8%)
v=5:     54 values (0.3%)
v=6:     18 values (0.09%)
v=7:      6 values (0.03%)
v=8:      2 values (0.01%)
v=9:      1 value  (0.005%)
```

This 2/3^k pattern is the signature of 3-adic structure.

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

---

## Next Steps

1. **Train hyperbolic model** with fixed loss function (user will do manually)
2. **Implement more hypothesis tests** (H4-H10, H12-H24)
3. **Focus on H2 improvement** after training shows VRC < -0.5
4. **Document emergent patterns** from falsification results

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

## Files Modified

- `research/scripts/falsify.py` - Fixed H3 test scaling
- `models/3-vae-gemm-v1/hyperbolic_ops.py` - Added radial/ultrametric losses
- `research/results/falsification_*.json` - Test results

---

**Conclusion**: The falsification framework is now properly wired and produces meaningful results. H1 and H11 are proven intrinsic. H2 and H3 require model training to fully validate.
