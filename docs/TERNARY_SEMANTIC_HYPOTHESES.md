# Ternary Fundamental Semantics: Hypothesis Space

**Doc-Type:** Research Hypotheses · Version 1.0 · Created 2025-12-30

---

## Purpose

Comprehensive catalog of **all plausible fundamental semantics** for balanced ternary arithmetic {-1, 0, +1}. Each hypothesis proposes a different ontological foundation. We will prototype and falsify each systematically.

**No preference ordering.** Each deserves equal initial consideration.

---

## Hypothesis Catalog

### H1: p-adic / 3-adic Semantics

**Claim:** Ternary operations exist in 3-adic number space where distance is defined by divisibility by 3.

**Core structure:**
- v₃(n) = largest k where 3^k divides n
- Distance: d(a,b) = 3^(-v₃(a-b))
- Closer = more divisible by 3

**Falsification test:**
```python
# If ternary operations preserve 3-adic distance structure
def test_p_adic():
    for a, b, c in samples:
        # Triangle inequality should be ULTRAMETRIC
        assert d(a,c) <= max(d(a,b), d(b,c))  # Strong form
```

**Predictions:**
- Operations cluster by valuation level
- High-valuation results are rare (exponentially decreasing)
- Zero (v=∞) is the unique center

---

### H2: Ultrametric Tree Semantics

**Claim:** Ternary values form a tree where distance = height of lowest common ancestor.

**Core structure:**
- All triangles are isoceles
- d(a,c) ≤ max(d(a,b), d(b,c))
- Operations = tree traversals

**Falsification test:**
```python
def test_ultrametric():
    for a, b, c in all_triples:
        dists = sorted([d(a,b), d(b,c), d(a,c)])
        assert dists[1] == dists[2]  # Two largest must be equal
```

**Predictions:**
- Hierarchical clustering of operations
- No "intermediate" distances - only discrete levels
- Nearest neighbor is always at same or higher level

---

### H3: Hyperbolic / Poincaré Ball Semantics

**Claim:** Ternary space has constant negative curvature (hyperbolic geometry).

**Core structure:**
- Points live in unit ball ||x|| < 1
- Geodesics are circular arcs
- Exponential volume growth with radius

**Falsification test:**
```python
def test_hyperbolic():
    # Geodesic midpoint should be equidistant
    mid = geodesic_midpoint(a, b)
    assert abs(d_hyp(a, mid) - d_hyp(b, mid)) < epsilon
```

**Predictions:**
- Tree-like structures embed naturally
- Boundary represents "infinity"
- Operations follow geodesic flows

---

### H4: Tropical Algebra Semantics

**Claim:** Ternary operations are best understood via tropical (min-plus or max-plus) semirings.

**Core structure:**
- ⊕ = min (or max)
- ⊗ = + (addition becomes multiplication)
- No subtraction (semiring, not ring)

**Falsification test:**
```python
def test_tropical():
    # Tropical matmul: C[i,j] = min_k(A[i,k] + B[k,j])
    C_tropical = tropical_matmul(A, B)
    C_standard = standard_matmul(A, B)
    # Check correlation or structural similarity
```

**Predictions:**
- Shortest path algorithms natural
- Idempotent operations (a ⊕ a = a)
- Piecewise linear behavior

---

### H5: Clifford Algebra / Geometric Algebra Semantics

**Claim:** Ternary values are elements of a Clifford algebra with geometric interpretation.

**Core structure:**
- Basis vectors e₁, e₂, e₃ with e_i² = ±1 or 0
- Products encode rotations, reflections
- Ternary = scalar + bivector components

**Falsification test:**
```python
def test_clifford():
    # Ternary operations should preserve geometric structure
    # Check if operations correspond to rotations/reflections
    for op in [tadd, tmul]:
        result = op(a, b)
        # Verify geometric invariants
```

**Predictions:**
- Rotation-like composition
- Duality between operations
- Natural extension to higher dimensions

---

### H6: Three-Valued Logic Semantics (Łukasiewicz / Kleene)

**Claim:** Ternary is fundamentally a three-valued logic system.

**Core structure:**
- Values: True (+1), Unknown (0), False (-1)
- Łukasiewicz: a → b = min(1, 1-a+b)
- Kleene: Strong/weak conjunction variants

**Falsification test:**
```python
def test_three_valued_logic():
    # Check if operations satisfy logic axioms
    # Modus ponens, excluded middle violations, etc.
    assert implies(a, implies(b, a)) == TRUE  # Axiom schema
```

**Predictions:**
- Natural handling of uncertainty
- Specific tautologies hold/fail
- Definable modal operators

---

### H7: Quantum-Inspired Superposition Semantics

**Claim:** Ternary 0 represents superposition of +1 and -1 (quantum-like).

**Core structure:**
- |+⟩, |-⟩ basis states
- |0⟩ = (|+⟩ + |-⟩)/√2 (superposition)
- Operations = unitary-like transformations

**Falsification test:**
```python
def test_quantum_like():
    # Check interference patterns
    # 0 ⊗ 0 should show interference, not classical mixing
    result = tmul(0, 0)  # What does "superposition × superposition" give?
```

**Predictions:**
- Interference effects in compound operations
- Non-classical correlations
- Measurement-like collapse

---

### H8: Category-Theoretic / Topos Semantics

**Claim:** Ternary operations form a category with morphisms preserving structure.

**Core structure:**
- Objects = ternary values or structures
- Morphisms = operations
- Composition = operation chaining

**Falsification test:**
```python
def test_categorical():
    # Check functoriality
    # F(g ∘ f) = F(g) ∘ F(f)
    assert compose(op1, op2)(a) == op1(op2(a))
    # Check natural transformations
```

**Predictions:**
- Universal properties
- Adjunctions between operation types
- Limits/colimits in ternary structures

---

### H9: Information-Theoretic Semantics

**Claim:** Ternary fundamentals are best expressed via entropy and information.

**Core structure:**
- H(X) = -Σ p(x) log₃ p(x)
- Mutual information I(X;Y)
- Channel capacity in base-3

**Falsification test:**
```python
def test_information():
    # Operations should maximize/preserve information
    H_input = entropy(a, b)
    H_output = entropy(op(a, b))
    # Check information preservation/loss patterns
```

**Predictions:**
- Specific entropy bounds on operations
- Optimal coding for operation results
- Rate-distortion behavior

---

### H10: Group-Theoretic Semantics (Z/3Z, S₃)

**Claim:** Ternary operations are group actions on Z/3Z or related groups.

**Core structure:**
- Z/3Z = {0, 1, 2} with addition mod 3
- S₃ = symmetric group on 3 elements
- Representation theory

**Falsification test:**
```python
def test_group():
    # Check group axioms
    assert op(a, identity) == a  # Identity
    assert op(a, inverse(a)) == identity  # Inverse
    assert op(a, op(b, c)) == op(op(a, b), c)  # Associativity
```

**Predictions:**
- Specific subgroup structure
- Conjugacy class behavior
- Character theory applications

---

### H11: Lattice / Order-Theoretic Semantics

**Claim:** Ternary values form a lattice with meet (∧) and join (∨) operations.

**Core structure:**
- Partial order: -1 ≤ 0 ≤ +1 (or different ordering)
- Meet = tmin, Join = tmax
- Distributivity properties

**Falsification test:**
```python
def test_lattice():
    # Distributivity
    assert meet(a, join(b, c)) == join(meet(a, b), meet(a, c))
    # Absorption
    assert meet(a, join(a, b)) == a
```

**Predictions:**
- Specific lattice type (distributive, modular, Boolean extension)
- Fixed points of operations
- Galois connections

---

### H12: Dynamical Systems / Attractor Semantics

**Claim:** Ternary operations are iterated maps with fixed points and attractors.

**Core structure:**
- f: T³ → T (ternary map)
- Fixed points: f(x) = x
- Basins of attraction
- Periodic orbits

**Falsification test:**
```python
def test_dynamical():
    # Iterate operations and check convergence
    x = random_ternary()
    for _ in range(1000):
        x = op(x, x)  # Self-application
    # Should converge to fixed point or cycle
```

**Predictions:**
- Specific attractor structure
- Lyapunov exponents
- Bifurcation behavior

---

### H13: Topological / Cantor Set Semantics

**Claim:** Ternary numbers have Cantor set structure with fractal properties.

**Core structure:**
- Base-3 representation
- Self-similarity at all scales
- Fractal dimension

**Falsification test:**
```python
def test_topological():
    # Check self-similarity
    # Operations should preserve fractal structure
    dim = box_counting_dimension(ternary_set)
    assert abs(dim - log(2)/log(3)) < epsilon  # Cantor set dimension
```

**Predictions:**
- Nowhere dense sets
- Perfect sets (no isolated points)
- Specific Hausdorff dimension

---

### H14: Neural / Biological Semantics

**Claim:** Ternary mirrors neural signaling: excitation (+1), inhibition (-1), resting (0).

**Core structure:**
- Neurons fire (+1), inhibit (-1), or rest (0)
- Operations = synaptic integration
- Threshold dynamics

**Falsification test:**
```python
def test_neural():
    # Check if operations match neural integration rules
    # Excitation + Inhibition = varies based on strength
    # Should match biological observations
```

**Predictions:**
- Winner-take-all behavior
- Lateral inhibition patterns
- Homeostatic balance

---

### H15: Physical Spin-1 Semantics

**Claim:** Ternary corresponds to spin-1 quantum systems (m = -1, 0, +1).

**Core structure:**
- Sz eigenvalues: -ℏ, 0, +ℏ
- SU(2) → SO(3) representation
- Clebsch-Gordan coefficients

**Falsification test:**
```python
def test_spin1():
    # Check angular momentum addition rules
    # 1 ⊗ 1 = 0 ⊕ 1 ⊕ 2
    # Ternary operations should follow CG coefficients
```

**Predictions:**
- Specific coupling rules
- Selection rules
- Wigner-Eckart theorem behavior

---

### H16: Game-Theoretic Semantics

**Claim:** Ternary represents outcomes in three-player or three-strategy games.

**Core structure:**
- Win (+1), Draw (0), Lose (-1)
- Nash equilibria
- Evolutionary stable strategies

**Falsification test:**
```python
def test_game():
    # Operations should correspond to game composition
    # Sequential games, simultaneous games
    # Check Nash equilibrium properties
```

**Predictions:**
- Specific equilibrium structure
- Dominated strategy elimination
- Correlated equilibria

---

### H17: Algebraic Geometry / F₃ Semantics

**Claim:** Ternary operations are best understood over the field F₃ = {0, 1, 2}.

**Core structure:**
- Finite field with 3 elements
- Varieties over F₃
- Frobenius endomorphism

**Falsification test:**
```python
def test_F3():
    # Check field axioms (mapping -1→2, 0→0, +1→1)
    # Multiplicative group is cyclic of order 2
    assert (a * b) % 3 == expected  # Field multiplication
```

**Predictions:**
- Quadratic residue structure
- Elliptic curves over F₃
- Weil conjectures

---

### H18: Homological / Chain Complex Semantics

**Claim:** Ternary operations form chain complexes with boundary operators.

**Core structure:**
- ∂² = 0 (boundary of boundary is zero)
- Homology groups
- Exact sequences

**Falsification test:**
```python
def test_homological():
    # Define boundary operator on ternary chains
    # Check ∂∂ = 0
    assert boundary(boundary(chain)) == zero_chain
```

**Predictions:**
- Non-trivial homology groups
- Betti numbers
- Euler characteristic

---

### H19: Modal Logic Semantics

**Claim:** Ternary represents modal operators: necessary (+1), possible (0), impossible (-1).

**Core structure:**
- □ (necessity), ◇ (possibility)
- Kripke frames
- Accessibility relations

**Falsification test:**
```python
def test_modal():
    # Check modal axioms
    # □(A → B) → (□A → □B)  (K axiom)
    # □A → A (T axiom, if reflexive)
```

**Predictions:**
- Specific modal logic system (K, T, S4, S5)
- Frame conditions
- Bisimulation behavior

---

### H20: Stochastic / Probabilistic Semantics

**Claim:** Ternary values encode probability transitions.

**Core structure:**
- +1 = certain positive
- -1 = certain negative
- 0 = maximum entropy / uncertainty

**Falsification test:**
```python
def test_probabilistic():
    # Operations should be stochastic matrices
    # Check Markov chain properties
    # Stationary distributions
```

**Predictions:**
- Specific stationary distributions
- Mixing times
- Ergodic behavior

---

### H21: Cellular Automata Semantics

**Claim:** Ternary operations are local rules in cellular automata.

**Core structure:**
- 3^k neighborhood states
- Update rule: f: {-1,0,+1}^k → {-1,0,+1}
- Wolfram classification

**Falsification test:**
```python
def test_CA():
    # Run ternary CA and classify behavior
    # Class I (fixed), II (periodic), III (chaotic), IV (complex)
    behavior = classify_CA(rule)
```

**Predictions:**
- Specific universality properties
- Edge of chaos behavior
- Glider/spaceship structures

---

### H22: Signed Graph / Network Semantics

**Claim:** Ternary represents signed relationships in networks.

**Core structure:**
- Edge weights: positive (+1), negative (-1), absent (0)
- Structural balance theory
- Signed Laplacian

**Falsification test:**
```python
def test_signed_graph():
    # Check structural balance
    # All triangles should be balanced (odd number of - edges)
    for triangle in triangles:
        assert product(triangle.edges) > 0  # Balanced
```

**Predictions:**
- Community structure
- Frustration patterns
- Spectral properties

---

### H23: Residue / Modular Semantics

**Claim:** Ternary is fundamentally modular arithmetic with signed representation.

**Core structure:**
- {-1, 0, +1} ≅ Z/3Z via -1↔2
- Chinese Remainder Theorem
- Quadratic reciprocity

**Falsification test:**
```python
def test_modular():
    # Operations should satisfy modular identities
    assert (a + b) % 3 == (a % 3 + b % 3) % 3
    # Check Legendre symbols
```

**Predictions:**
- CRT decomposition
- Prime factorization behavior
- Multiplicative structure

---

### H24: Ternary-Specific / Sui Generis Semantics

**Claim:** Ternary has unique semantics not reducible to any known structure.

**Core structure:**
- Intrinsic ternary axioms
- Not embeddable in binary, quaternary, etc.
- Unique algebraic closure

**Falsification test:**
```python
def test_sui_generis():
    # Check for properties that don't fit any known structure
    # If all other hypotheses are falsified, this remains
```

**Predictions:**
- Novel algebraic identities
- Irreducible complexity
- Emergent properties at ternary level only

---

## Falsification Protocol

For each hypothesis:

1. **Define measurable predictions** - What does this semantics predict?
2. **Implement test** - Code the falsification test
3. **Run on corpus** - All 19,683 ternary operations
4. **Score fit** - How well does data match predictions?
5. **Record anomalies** - What doesn't fit?

### Scoring Rubric

| Score | Meaning |
|-------|---------|
| A | >95% predictions hold |
| B | 80-95% predictions hold |
| C | 50-80% predictions hold |
| D | 20-50% predictions hold |
| F | <20% predictions hold (FALSIFIED) |

---

## Quick Implementation Priority

Start with most testable (easiest to falsify):

1. **H11: Lattice** - Simple algebraic tests
2. **H10: Group** - Direct axiom checking
3. **H6: Three-valued logic** - Truth table analysis
4. **H1: p-adic** - Valuation computation
5. **H2: Ultrametric** - Triangle inequality
6. **H4: Tropical** - Semiring axioms

Then progress to harder:

7. **H3: Hyperbolic** - Requires embedding
8. **H12: Dynamical** - Requires iteration
9. **H9: Information** - Requires statistics
10. **H7: Quantum** - Requires careful setup

---

## Notes

- Multiple hypotheses may be simultaneously true (complementary views)
- Some may be special cases of others (hierarchical)
- The "true" semantics may be a combination
- Falsification ≠ uselessness (may still be useful approximation)

---

**Status:** Hypothesis catalog complete. Ready for systematic falsification.
