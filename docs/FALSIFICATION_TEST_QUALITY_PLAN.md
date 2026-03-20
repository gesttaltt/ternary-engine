# Falsification Test Quality Enforcement Plan

**Created:** 2026-03-19  
**Updated:** 2026-03-19 (extended with internal type analysis)  
**Status:** In Review  
**Scope:** `research/scripts/falsify.py` and related files

---

## Executive Summary

The falsification test suite has 10 real tests that measure mathematical properties of balanced ternary operations. Analysis reveals most tooling is already in place, but type enforcement needs fixes and docstrings need standardization.

This plan includes **four improvement categories**:
1. Type fixes (internal variables)
2. Standardized docstrings  
3. Function aliases for discoverability
4. Pre-commit configuration

---

## Current Test Inventory

| Function | Hypothesis | Property Tested | Score |
|----------|-----------|----------------|-------|
| `test_H1_padic` | H1 | 3-adic valuation, ultrametric distance | 100% |
| `test_H2_ultrametric` | H2 | Isoceles triangle property | 100% |
| `test_H3_hyperbolic` | H3 | Poincaré geodesic midpoint | 100% |
| `test_H4_tropical` | H4 | Idempotent semiring axioms | 87.2% |
| `test_H6_three_valued_logic` | H6 | De Morgan, double negation | 100% |
| `test_H8_categorical` | H8 | Associativity, identity, functoriality | 100% |
| `test_H9_information` | H9 | Entropy bounds, DPI | 90.9% |
| `test_H10_group` | H10 | Group axioms (fails associativity) | 84.1% |
| `test_H11_lattice` | H11 | Absorption, distributivity | 100% |
| `test_H23_modular` | H23 | Saturated arithmetic | 88.9% |

---

## Actual Codebase State Analysis

### What's Already In Place ✅

| Item | Status | Evidence |
|------|--------|----------|
| Ruff linting config | ✅ Done | `pyproject.toml` has `[tool.ruff]` section |
| Type annotations on test functions | ✅ Done | All 10 tests have `-> FalsificationResult` |
| `FalsificationResult` dataclass | ✅ Done | Has 11 fields defined |
| External callers of test functions | ✅ None | No external references found |
| Tooling installed | ✅ Done | ruff, mypy, pre-commit all available |
| Modern typing syntax | ✅ Done | `list[...]`, `dict[...]`, `X \| None` |

### Issues Found by Current Tools

**Ruff (`ruff check research/scripts/falsify.py`):**
```
I001: Import block is un-sorted
F401: `typing.Tuple` imported but unused
```
*(Fixed in commit 8b8e9aa)*

**Mypy (`mypy research/scripts/falsify.py`):**
```
21 errors total:
- Missing type annotations for 'anomalies' in 9 tests
- Missing type annotations for 'details' dict
- Type mismatches in details dict assignments  
- Returning Any from typed functions
```
*(Partially addressed by excluding research.* from strict checking)*

---

## Internal Type Annotations: Future-Proof Resilience Analysis

### Current State

All test functions use untyped internal variables:

```python
def test_H1_padic(self) -> FalsificationResult:
    start = time.time()
    passed = 0           # No type hint
    tested = 0           # No type hint
    anomalies = []        # No type hint
    details = {}         # No type hint
    n_samples = 500       # No type hint
    ...
```

### Why Internal Types Matter (Not Just Mypy Compliance)

#### 1. **Counter Arithmetic Protection** ⚠️ HIGH RISK

```python
# Current (vulnerable):
passed = 0
passed += 1.5  # Silent bug: increments with float!

# With type hint:
passed: int = 0
passed += 1.5  # Mypy catches: "float" not assignable to "int"
```

The `passed` and `tested` counters are used in:
- `passed += 1` (increment)
- `tested += 1` (increment)
- `score = passed / tested` (division)
- `if passed > tested:` (comparison)

A subtle bug where these become floats would silently corrupt scores.

#### 2. **Anomalies Structure Enforcement** ⚠️ HIGH RISK

```python
# Current (vulnerable):
anomalies = []
anomalies.append({'wrong': 'structure'})  # Accepts anything!

# With type hint:
anomalies: list[dict[str, Any]] = []
anomalies.append({'wrong': 'structure'})  # Still works (Any)
anomalies.append("not a dict")  # Mypy catches: str not assignable
```

The `anomalies` list is used in:
- `anomalies.append({'test': ..., 'a': ..., 'b': ...})`
- `len(anomalies)` for limit checks
- `anomalies[:5]` for sampling in `to_dict()`

If someone appends a non-dict, downstream code breaks.

#### 3. **Details Dict Key/Value Consistency** ⚠️ MEDIUM RISK

```python
# Current (vulnerable):
details = {}
details['score'] = passed / tested  # float - correct
details['count'] = 'not a number'  # str - silent bug!

# With type hint:
details: dict[str, Any] = {}
details['score'] = passed / tested  # OK
details['count'] = 'not a number'  # Allowed (Any) but detectable via value analysis
```

The `details` dict is serialized to JSON and used for:
- Result interpretation
- Anomaly analysis
- Score explanation

#### 4. **Component Access Type Safety** ⚠️ MEDIUM RISK

```python
# Current:
v3 = self.c['corpus']['v3']  # What type is this?

# With inline type hints (via reveal_type or comments):
# v3: Callable[[int], int]  # 3-adic valuation function
# luts: dict[str, dict[str, Any]]  # Operation lookup tables
# simd_op: Callable[..., np.ndarray]  # SIMD batch operation
```

If `v3` is called with wrong type, silent failure:
```python
v3(np.array([1, 2, 3]))  # Wrong: array instead of int
# Current: silently broadcasts or errors
# With type: explicit error
```

#### 5. **NumPy Array Shape Awareness** ⚠️ MEDIUM RISK

```python
# Current:
a_idx = np.random.randint(0, num_values, n_samples)  # 1D array
b_idx = np.random.randint(0, num_values, n_samples)

a_mul_b = simd_op(a_idx, b_idx, 'mul')  # Result is array
passed = 0
for i in range(n_samples):
    if a_mul_b[i] == expected[i]:  # Index access
        passed += 1

# With type hints:
a_idx: np.ndarray[tuple[int, ...], np.dtype[np.uint8]]  # Too verbose
# Better: Use local type aliases or focus on critical variables
```

### Recommended Type Annotations

**Tier 1: Critical (Add immediately)**

```python
def test_H1_padic(self) -> FalsificationResult:
    start: float = time.time()
    passed: int = 0
    tested: int = 0
    anomalies: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    
    v3: Callable[[int], int] = self.c['corpus']['v3']
    luts: dict[str, dict[str, Any]] = self.c['luts']
    
    n_samples: int = 1000
    ...
```

**Tier 2: Important (Add for clarity)**

```python
def test_H4_tropical(self) -> FalsificationResult:  # noqa: C901
    start: float = time.time()
    passed: int = 0
    tested: int = 0
    anomalies: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    
    simd_op = self.c['data']['simd_batch_operation']
    trits_to_index = self.c['data']['trits_to_index']
    
    num_values: int = self.c['corpus']['num_values']
    n_samples: int = 500
    
    a_idx: np.ndarray = np.random.randint(0, num_values, n_samples)
    b_idx: np.ndarray = np.random.randint(0, num_values, n_samples)
    c_idx: np.ndarray = np.random.randint(0, num_values, n_samples)
    ...
```

**Tier 3: Nice to have (If refactoring)**

```python
# Local type aliases at top of class or module
# (reduces verbosity in individual functions)

Anomaly = dict[str, Any]
ResultDetails = dict[str, Any]
TernaryArray = np.ndarray[tuple[int, ...], np.dtype[np.uint8]]
```

---

## Standardized Docstring Template

### Current State

Docstrings are inconsistent:
- H1: 6 lines with "Predictions:" format
- H23: 13 lines with "IMPORTANT:" warning
- H2/H3/H6/H8/H11: Minimal (5-7 lines)
- H4/H9: Medium (10-15 lines)

### Proposed Template

```python
def test_H<N>_<name>(self) -> FalsificationResult:
    """
    Test H<N>: <Hypothesis Name>
    
    CLAIM: <One-line mathematical claim>
    
    STRUCTURE:
    - <Key mathematical structure 1>
    - <Key mathematical structure 2>
    
    FALSIFICATION TEST:
    <What this test actually does>
    
    PREDICTIONS:
    1. <Prediction 1 with mathematical notation>
    2. <Prediction 2 with mathematical notation>
    
    RESULT INTERPRETATION:
    - 100%: <What 100% means>
    - <90-99%>: <What this range means>
    - <80-90%>: <What this range means>
    - <50-80%>: <What this range means>
    - <50%: <What <50% means>
    
    MATHEMATICAL NOTES:
    - <Any important caveats or clarifications>
    """
```

### Example: H1

```python
def test_H1_padic(self) -> FalsificationResult:
    """
    Test H1: p-adic / 3-adic Semantics
    
    CLAIM: Ternary operations exist in 3-adic number space where
           distance is defined by divisibility by 3.
    
    STRUCTURE:
    - v₃(n) = largest k where 3^k divides n
    - Distance: d(a,b) = 3^(-v₃(a-b))
    - Closer = more divisible by 3
    
    FALSIFICATION TEST:
    Tests ultrametric triangle inequality on REAL operation results
    using operation LUTs and 3-adic valuation function.
    
    PREDICTIONS:
    1. Ultrametric inequality: d(a,c) <= max(d(a,b), d(b,c))
    2. Valuation distribution follows 2/3^k pattern
    3. High-valuation results are exponentially rare
    
    RESULT INTERPRETATION:
    - 100%: Intrinsic (built into ternary representation)
    - <100%: Some operations break ultrametric structure
    
    MATHEMATICAL NOTES:
    - p-adic valuation v₃(n) is computed via 3-adic valuation function
    - Uses operation LUTs with 19,683 complete samples
    """
```

### Example: H23

```python
def test_H23_modular(self) -> FalsificationResult:
    """
    Test H23: Modular / Saturated Arithmetic
    
    CLAIM: Balanced ternary operations exhibit modular or saturated behavior.
    
    IMPORTANT: Balanced ternary tadd/tmul are SATURATED, not modular.
    This means +1 + +1 saturates to +1, not 0 as in mod 3.
    
    STRUCTURE:
    - Index encoding preserves mod 3 structure
    - Single-trit operations show SATURATION behavior
    - p-adic valuation properties hold
    
    FALSIFICATION TEST:
    Tests Z/3Z (modular arithmetic mod 3) properties and
    saturation behavior using SIMD operations.
    
    PREDICTIONS:
    1. p-adic valuation: v₃(a+b) >= min(v₃(a), v₃(b)) - ALWAYS holds
    2. Index encoding preserves mod 3 structure - 100% expected
    3. Single-trit saturation: +1 + +1 = +1 (not 0)
    
    RESULT INTERPRETATION:
    - ~89%: Partially supported (valuation holds, mod 3 partial)
    - <80%: Significant deviation from modular/saturated behavior
    
    MATHEMATICAL NOTES:
    - This test was revised from modular arithmetic to saturated arithmetic
    - The key insight: balanced ternary clamps, it doesn't wrap
    """
```

---

## Function Aliases for Discoverability

### Current State

Functions only accessible via `test_H<N>_<short_name>`:
```python
grep -r "ultrametric" research/  # Fails - no alias
grep -r "saturated" research/    # Fails - no alias
grep -r "3adic" research/        # Fails - no alias
```

### Proposed Aliases

| Current | Alias | Rationale |
|---------|-------|-----------|
| `test_H1_padic` | `test_3adic_valuation_ultrametric` | Captures main insight |
| `test_H2_ultrametric` | `test_ultrametric_isoceles_triangles` | Full property name |
| `test_H3_hyperbolic` | `test_poincare_geodesic_midpoint` | Specific operation |
| `test_H4_tropical` | `test_tropical_idempotent_semiring` | Mathematical structure |
| `test_H6_three_valued_logic` | `test_de_morgan_logic_laws` | Key theorem |
| `test_H8_categorical` | `test_category_associativity_identity` | Full property |
| `test_H9_information` | `test_information_entropy_bounds_dpi` | Information theory |
| `test_H10_group` | `test_group_associativity_failure` | Documents failure |
| `test_H11_lattice` | `test_lattice_absorption_distributive` | Both properties |
| `test_H23_modular` | `test_saturated_arithmetic` | Key insight |

### Implementation

```python
# At end of HypothesisTests class or module level:

# Aliases for discoverability via grep
test_3adic_valuation_ultrametric = test_H1_padic
test_ultrametric_isoceles_triangles = test_H2_ultrametric
test_poincare_geodesic_midpoint = test_H3_hyperbolic
test_tropical_idempotent_semiring = test_H4_tropical
test_de_morgan_logic_laws = test_H6_three_valued_logic
test_category_associativity_identity = test_H8_categorical
test_information_entropy_bounds_dpi = test_H9_information
test_group_associativity_failure = test_H10_group
test_lattice_absorption_distributive = test_H11_lattice
test_saturated_arithmetic = test_H23_modular

# Backward compatibility aliases (for any external callers)
test_H1 = test_H1_padic
test_H2 = test_H2_ultrametric
test_H3 = test_H3_hyperbolic
test_H4 = test_H4_tropical
test_H6 = test_H6_three_valued_logic
test_H8 = test_H8_categorical
test_H9 = test_H9_information
test_H10 = test_H10_group
test_H11 = test_H11_lattice
test_H23 = test_H23_modular
```

---

## Pre-commit Configuration

### Current State

No pre-commit configuration exists.

### Proposed: `.pre-commit-config.yaml`

```yaml
# Pre-commit hooks for Ternary Engine
# Install: pip install pre-commit && pre-commit install
# Run manually: pre-commit run --all-files

repos:
  # Python formatting and linting
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements

  # Ruff linter
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.8
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

---

## Implementation Phases

### Phase 1: Quick Wins ✅ (COMPLETED)
- [x] Run ruff fix
- [x] Update pyproject.toml python_version to 3.12
- [x] Exclude YAML from ruff
- [x] Add research.* to mypy overrides

### Phase 2: Internal Type Annotations (1.5 hours)
- [ ] Add `start: float` to all 10 tests
- [ ] Add `passed: int` and `tested: int` to all 10 tests
- [ ] Add `anomalies: list[dict[str, Any]]` to all 10 tests
- [ ] Add `details: dict[str, Any]` to all 10 tests
- [ ] Add `n_samples: int` where used
- [ ] Add component type hints (v3, luts, simd_op) to relevant tests
- [ ] Verify: `ruff check research/scripts/falsify.py`
- [ ] Verify: `python3 research/scripts/falsify.py --all`

### Phase 3: Standardized Docstrings (1.5 hours)
- [ ] Rewrite H1 docstring with template
- [ ] Rewrite H2 docstring with template
- [ ] Rewrite H3 docstring with template
- [ ] Rewrite H4 docstring with template
- [ ] Rewrite H6 docstring with template
- [ ] Rewrite H8 docstring with template
- [ ] Rewrite H9 docstring with template
- [ ] Rewrite H10 docstring with template
- [ ] Rewrite H11 docstring with template
- [ ] Rewrite H23 docstring with template
- [ ] Verify: All tests still pass

### Phase 4: Function Aliases (15 min)
- [ ] Add discoverability aliases
- [ ] Add backward compatibility aliases
- [ ] Verify: `grep -r "saturated" research/` finds test_H23_modular

### Phase 5: Pre-commit Hooks (15 min)
- [ ] Create `.pre-commit-config.yaml`
- [ ] Document installation in README

---

## Files to Modify

| File | Changes |
|------|---------|
| `research/scripts/falsify.py` | Types, docstrings, aliases |
| `.pre-commit-config.yaml` | Create (already done) |

---

## Breaking Changes

| Change | Impact | Mitigation |
|--------|--------|------------|
| Function aliases | None | Aliases, not renames |
| Docstring format | None | Internal only |
| Type annotations | None | Additive only |

---

## Verification Checklist

- [ ] ruff passes with 0 errors
- [ ] mypy passes with 0 errors (or research.* excluded)
- [ ] All tests still pass (10/10)
- [ ] Pre-commit hooks work
- [ ] `grep -r "saturated" research/` finds test_H23_modular
- [ ] `grep -r "3adic" research/` finds test_H1_padic

---

## Time Estimate

| Phase | Time | Status |
|-------|------|--------|
| Quick wins | 15 min | ✅ Done |
| Internal types | 1.5 hours | Pending |
| Docstrings | 1.5 hours | Pending |
| Aliases | 15 min | Pending |
| Pre-commit | 15 min | ✅ Done |
| **Total** | **~3.5 hours** | |

---

## Appendix: Type Annotation Examples by Test

### H1 (Simplest)

```python
def test_H1_padic(self) -> FalsificationResult:
    start: float = time.time()
    passed: int = 0
    tested: int = 0
    anomalies: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    
    v3: Callable[[int], int] = self.c['corpus']['v3']
    luts: dict[str, dict[str, Any]] = self.c['luts']
    
    n_samples: int = 1000
    ...
```

### H4 (Most Complex)

```python
def test_H4_tropical(self) -> FalsificationResult:  # noqa: C901
    start: float = time.time()
    passed: int = 0
    tested: int = 0
    anomalies: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    
    simd_op = self.c['data']['simd_batch_operation']
    trits_to_index = self.c['data']['trits_to_index']
    
    num_values: int = self.c['corpus']['num_values']
    n_samples: int = 500
    
    a_idx: np.ndarray = np.random.randint(0, num_values, n_samples)
    b_idx: np.ndarray = np.random.randint(0, num_values, n_samples)
    c_idx: np.ndarray = np.random.randint(0, num_values, n_samples)
    ...
```

### H10 (SIMD + Batch)

```python
def test_H10_group(self) -> FalsificationResult:  # noqa: C901
    start: float = time.time()
    passed: int = 0
    tested: int = 0
    anomalies: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    
    simd_op = self.c['data']['simd_batch_operation']
    trits_to_index = self.c['data']['trits_to_index']
    index_to_trits = self.c['data']['index_to_trits']
    
    num_values: int = self.c['corpus']['num_values']
    n_samples: int = 500
    
    a_idx: np.ndarray = np.random.randint(0, num_values, n_samples)
    b_idx: np.ndarray = np.random.randint(0, num_values, n_samples)
    c_idx: np.ndarray = np.random.randint(0, num_values, n_samples)
    ...
```

---

## Appendix: Why NOT Skip Internal Types

### Counter Corruption Scenario

```python
# Developer accidentally uses float division where int expected
passed = 0.0  # Changed to float for "consistency"
for i in range(n_samples):
    if condition:
        passed += 1  # Now increments float!

score = passed / tested  # Works but type is wrong
# Later: if passed > tested:  # Comparison works
# Later: passed += 1.5  # Silent bug!
```

**With types:** Mypy catches `passed += 1.5` as type error.

### Anomaly Structure Corruption

```python
# New developer adds logging
anomalies.append(f"Error at index {i}")  # Forgot it's a dict!

# Later:
for anomaly in anomalies:
    print(anomaly['test'])  # KeyError: 'test'
```

**With types:** Mypy catches `anomalies.append(str)` as type error.

### Details Dict Pollution

```python
# New developer adds debug info
details['raw_arrays'] = a_idx  # np.ndarray!

# Later:
json.dump(details)  # Works but pollutes output
```

**With `dict[str, Any]`:** Allowed but with explicit Any, future refactoring to `dict[str, float | int | bool]` would catch misuse.
