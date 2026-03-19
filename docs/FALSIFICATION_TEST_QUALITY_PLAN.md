# Falsification Test Quality Enforcement Plan

**Created:** 2026-03-19  
**Status:** In Review  
**Scope:** `research/scripts/falsify.py` and related files

---

## Executive Summary

The falsification test suite has 10 real tests that measure mathematical properties of balanced ternary operations. Analysis reveals most tooling is already in place, but type enforcement needs fixes and docstrings need standardization.

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

### Issues Found by Current Tools

**Ruff (`ruff check research/scripts/falsify.py`):**
```
I001: Import block is un-sorted
F401: `typing.Tuple` imported but unused
```

**Mypy (`mypy research/scripts/falsify.py`):**
```
21 errors total:
- Missing type annotations for 'anomalies' in 9 tests
- Missing type annotations for 'details' dict
- Type mismatches in details dict assignments  
- Returning Any from typed functions
```

---

## Issues Identified

### 1. Type Errors (Priority: HIGH)

**Problem:** 21 mypy errors

```python
# Line 336: Need type annotation for "anomalies"
anomalies = []  # Should be: anomalies: List[Dict[str, Any]] = []

# Line 383: Type mismatch
details['valuation_distribution'] = {}  # Type confusion
```

### 2. Import Issues (Priority: LOW)

- I001: Imports not sorted
- F401: Unused `Tuple` import

### 3. Docstring Inconsistency (Priority: MEDIUM)

Current docstrings: 6-17 lines, inconsistent format

### 4. Vague Naming (Priority: MEDIUM)

`test_H1_padic` reveals nothing about what property is tested

---

## Implementation Plan

### Phase 1: Quick Wins (15 min)

1. **Run ruff fix:** `ruff check research/scripts/falsify.py --fix`
2. **Update pyproject.toml:** Change `python_version = "3.7"` to `"3.9"`

### Phase 2: Type Fixes (30 min)

For each test function, add:
```python
anomalies: List[Dict[str, Any]] = []
details: Dict[str, Any] = {}
```

### Phase 3: Pre-commit (15 min)

Create `.pre-commit-config.yaml`

### Phase 4: Documentation (45 min)

Update docstrings to template format.

### Phase 5: Naming (20 min)

Rename functions with aliases for backward compat.

---

## Proposed Renames

| Current | Proposed | Alias |
|---------|----------|-------|
| `test_H1_padic` | `test_3adic_valuation_ultrametric` | `test_H1_padic` |
| `test_H2_ultrametric` | `test_ultrametric_isoceles_triangles` | `test_H2_ultrametric` |
| `test_H3_hyperbolic` | `test_poincare_geodesic_midpoint` | `test_H3_hyperbolic` |
| `test_H4_tropical` | `test_tropical_idempotent_semiring` | `test_H4_tropical` |
| `test_H6_three_valued_logic` | `test_de_morgan_logic_laws` | `test_H6_three_valued_logic` |
| `test_H8_categorical` | `test_category_associativity_identity` | `test_H8_categorical` |
| `test_H9_information` | `test_information_entropy_bounds` | `test_H9_information` |
| `test_H10_group` | `test_group_associativity_failure` | `test_H10_group` |
| `test_H11_lattice` | `test_lattice_absorption_distributive` | `test_H11_lattice` |
| `test_H23_modular` | `test_saturated_arithmetic` | `test_H23_modular` |

---

## Files to Modify

| File | Changes |
|------|---------|
| `pyproject.toml` | Fix python_version to 3.9 |
| `research/scripts/falsify.py` | Types, docstrings, names |
| `.pre-commit-config.yaml` | Create new file |

---

## Breaking Changes

| Change | Impact | Mitigation |
|--------|---------|------------|
| Function renames | Low | Aliases for backward compat |
| Docstring format | None | Internal only |

---

## Verification Checklist

- [ ] ruff passes with 0 errors
- [ ] mypy passes with 0 errors
- [ ] All tests still pass (10/10)
- [ ] Pre-commit hooks work

---

## Time Estimate

| Phase | Time |
|-------|------|
| Quick wins | 15 min |
| Type fixes | 30 min |
| Pre-commit | 15 min |
| Docstrings | 45 min |
| Naming | 20 min |
| **Total** | **~2.5 hours** |
