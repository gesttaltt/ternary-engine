# Ternary Semantic Falsification Research

VZ|**Doc-Type:** Research Overview · Version 1.1 · Updated 2026-03-19

---

## Philosophy

**We do not search for solutions. We LISTEN to errors.**

By systematically falsifying hypotheses, we discover the solution space through its negative image. Each falsified hypothesis removes territory from the map of possibility. What remains after all eliminations IS the truth.

---

## Directory Structure

```
research/
├── README.md                    # This file - overview and quick start
├── TERNARY_SEMANTIC_HYPOTHESES.md  # All 24 hypotheses definitions
├── configs/
│   └── schema.yaml              # YAML schema for hypothesis definitions
├── scripts/
│   └── falsify.py               # Main falsification framework
└── results/
    ├── FALSIFICATION_SUMMARY.md # Human-readable summary of all findings
    ├── SESSION_STATE.md         # State for resuming work
    └── falsification_*.json     # Raw test results (timestamped)
```

---

## Quick Start

### Run All Implemented Tests
```bash
python research/scripts/falsify.py --all
```

### Run Specific Hypothesis
```bash
python research/scripts/falsify.py -H H1   # p-adic
python research/scripts/falsify.py -H H9   # Information theory
```

### Run by Tier
```bash
python research/scripts/falsify.py --tier tier1_easy
python research/scripts/falsify.py --tier tier2_medium
```

---

ZZ|## Current Status (2026-03-19)

QX|### Implemented Tests (10 of 24)

WV|| ID | Hypothesis | Score | Grade | Status |
KT||----|------------|-------|-------|--------|
VS|| H1 | p-adic / 3-adic | 100% | A | **INTRINSIC** |
BX|| H2 | Ultrametric Tree | 100% | A | Supported |
KP|| H3 | Hyperbolic/Poincaré | 100% | A | Supported |
JB|| H4 | Tropical Algebra | 87.2% | B | Supported |
KQ|| H6 | Three-Valued Logic | 100% | A | **INTRINSIC** |
BV|| H8 | Category-Theoretic | 100% | A | Supported |
VZ|| H9 | Information Theory | 90.9% | B | Supported |
ZN|| H10 | Group Theory | 84.1% | B | Non-associative! |
TZ|| H11 | Lattice/Order | 100% | A | **INTRINSIC** |
YM|| H23 | Modular/Saturated | 88.9% | B | Supported |
KB|

WS|### Not Yet Implemented (13)
HQ|
KX|H12 (Dynamical), H13 (Topological),
QK|H14 (Neural), H15 (Spin-1), H16 (Game), H17 (F₃ Algebraic), H18 (Homological),
XX|H19 (Modal), H20 (Stochastic), H21 (Cellular Automata), H22 (Signed Graph), H24 (Sui Generis)
SZ|

---

## Key Discoveries

XB|### 1. Intrinsic Structures (Cannot Be Falsified)
QK|- **H1 (p-adic)**: 3-adic valuation is built into ternary representation
HY|- **H2 (Ultrametric)**: All triangles satisfy isoceles property
QY|- **H3 (Hyperbolic)**: Geodesic midpoint equidistance holds
BS|- **H6 (Three-Valued Logic)**: De Morgan laws, double negation hold perfectly
HR|- **H8 (Category)**: Associativity, identity, functoriality hold
MX|- **H11 (Lattice)**: tmin/tmax form a distributive lattice
WR|

### 2. Non-Associativity of tadd (H10)
**Major finding**: `tadd(tadd(a,b), c) ≠ tadd(a, tadd(b,c))` for **79.6%** of triplets!
Balanced ternary with tadd is NOT a group.

### 3. Partial Tropical Structure (H4)
- tadd distributes over tmin/tmax: **100%**
- tmul does NOT distribute: only **10.4%**

YX|### 4. Model Training Gaps (H4, H9)
JH|Raw ternary values are perfectly ultrametric/hyperbolic, but trained model embeddings
KP|don't preserve this structure yet.
ZT|
Raw ternary values are perfectly ultrametric/hyperbolic, but trained model embeddings
don't preserve this structure yet. VRC = 0.035 (target: -0.8).

---

## Component Wiring

All tests use REAL project components:

| Component | Module | Status |
|-----------|--------|--------|
| SIMD Engine | `ternary_simd_engine` | OK |
| Hyperbolic VAE | `models/3-vae-gemm-v1/hyperbolic_ops.py` | OK |
| Ultrametric Energy | `models/gemm_discovery/ebm/ultrametric_energy.py` | OK |
| Operation LUTs | `models/3-vae-gemm-v1/data.py` | OK (2M samples) |
| Corpus | 19,683 ternary values | OK |
| Trained Model | `best_model.pt` | Needs retraining |

---

## Scoring Rubric

| Grade | Score Range | Meaning |
|-------|-------------|---------|
| A | >95% | Strongly supported |
| B | 80-95% | Supported |
| C | 50-80% | Weak evidence |
| D | 20-50% | Mostly falsified |
| F | <20% | Falsified |

---

## Adding New Hypothesis Tests

1. Add test method to `FalsificationRunner` class in `falsify.py`:
```python
def test_H<N>_<name>(self) -> FalsificationResult:
    """Test H<N>: <Hypothesis Name>"""
    # Implementation...
```

2. Add to mapping in `run_hypothesis()`:
```python
mapping = {
    'H<N>': self.test_H<N>_<name>,
    ...
}
```

3. Add to `run_all_implemented()` list:
```python
implemented = ['H1', ..., 'H<N>']
```

---

## Related Files

- **Main config**: `.claude/CLAUDE.md` (see "Trained Models" section)
- **Hypotheses**: `research/TERNARY_SEMANTIC_HYPOTHESES.md`
- **24 hypothesis catalog**: `docs/TERNARY_SEMANTIC_HYPOTHESES.md`

---

## References

- [Ternary Semantic Hypotheses](./TERNARY_SEMANTIC_HYPOTHESES.md)
- [Falsification Summary](./results/FALSIFICATION_SUMMARY.md)
- [Session State](./results/SESSION_STATE.md)
