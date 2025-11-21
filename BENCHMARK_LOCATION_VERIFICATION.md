# Benchmark Location Verification

## Current Structure ✅ CORRECT

```
project_root/  (C:\Users\Gestalt\Desktop\ternary\repos\ternary-engine\)
│
├── ternary_simd_engine.pyd        ← Compiled modules (after build)
├── ternary_fusion_engine.pyd      ← Compiled modules (after build)
│
├── build.py                        ← Build script
├── build_fusion.py                 ← Fusion build script
│
└── benchmarks/                     ← Benchmarks directory
    ├── bench_phase0.py            ← Main benchmark (uses .parent.parent)
    ├── bench_compare.py
    ├── benchmark_framework.py
    ├── run_all_benchmarks.py
    │
    ├── micro/                      ← Micro-benchmarks subdirectory
    │   ├── bench_fusion_phase41.py    (uses .parent.parent.parent)
    │   ├── bench_fusion_simple.py     (uses .parent.parent.parent)
    │   ├── bench_fusion_poc.py        (uses .parent.parent.parent)
    │   └── bench_fusion_rigorous.py   (uses .parent.parent.parent)
    │
    └── macro/                      ← Macro-benchmarks subdirectory
        ├── bench_image_pipeline.py    (uses .parent.parent.parent)
        └── bench_neural_layer.py      (uses .parent.parent.parent)
```

---

## How Benchmarks Are Invoked

### ✅ CORRECT: All benchmarks run FROM project root

```bash
# From project root
python benchmarks/bench_phase0.py
python benchmarks/micro/bench_fusion_phase41.py
python benchmarks/macro/bench_image_pipeline.py
```

**Evidence from codebase:**
- GitHub workflows: `python benchmarks/bench_phase0.py`
- Documentation (54 references): `python benchmarks/bench_phase0.py`
- README.md: `python benchmarks/bench_phase0.py`
- All examples assume running from project root

---

## Path Resolution Logic

### bench_phase0.py (benchmarks directory)

**File location:** `project_root/benchmarks/bench_phase0.py`

```python
PROJECT_ROOT = Path(__file__).parent.parent
# Path(__file__) = project_root/benchmarks/bench_phase0.py
# .parent        = project_root/benchmarks/
# .parent.parent = project_root/  ✅ CORRECT
```

### Micro-benchmarks (benchmarks/micro/ subdirectory)

**File location:** `project_root/benchmarks/micro/bench_fusion_phase41.py`

```python
PROJECT_ROOT = Path(__file__).parent.parent.parent
# Path(__file__)        = project_root/benchmarks/micro/bench_fusion_phase41.py
# .parent               = project_root/benchmarks/micro/
# .parent.parent        = project_root/benchmarks/
# .parent.parent.parent = project_root/  ✅ CORRECT (FIXED)
```

### Macro-benchmarks (benchmarks/macro/ subdirectory)

**File location:** `project_root/benchmarks/macro/bench_image_pipeline.py`

```python
PROJECT_ROOT = Path(__file__).parent.parent.parent
# Path(__file__)        = project_root/benchmarks/macro/bench_image_pipeline.py
# .parent               = project_root/benchmarks/macro/
# .parent.parent        = project_root/benchmarks/
# .parent.parent.parent = project_root/  ✅ CORRECT (ALREADY CORRECT)
```

---

## Why Benchmarks Stay in benchmarks/ Directory

### 1. **Organization**: Keeps benchmark code separate from core code
   - Core: `ternary_simd_engine.cpp`, `build.py`
   - Tests: `tests/`
   - Benchmarks: `benchmarks/`

### 2. **Consistency**: All documentation and workflows expect this structure
   - 54 references to `python benchmarks/bench_phase0.py`
   - GitHub Actions workflows reference `benchmarks/`
   - PGO scripts reference `benchmarks/`

### 3. **Module imports work correctly** when project root in sys.path:
   ```python
   # After adding project_root to sys.path:
   import ternary_simd_engine           # ✅ Found at project_root/
   from benchmarks import benchmark_framework  # ✅ Found at project_root/benchmarks/
   ```

---

## What Was Wrong (Before Fix)

### Micro-benchmarks had INCORRECT path

**Before fix:**
```python
# benchmarks/micro/bench_fusion_phase41.py
sys.path.insert(0, str(Path(__file__).parent.parent))  # ❌ WRONG
# Resolved to: project_root/benchmarks/
```

**Problem:**
- Added `benchmarks/` to sys.path
- But modules are in `project_root/`, not `benchmarks/`
- Result: `ImportError: No module named 'ternary_simd_engine'`

**After fix:**
```python
# benchmarks/micro/bench_fusion_phase41.py
PROJECT_ROOT = Path(__file__).parent.parent.parent  # ✅ CORRECT
sys.path.insert(0, str(PROJECT_ROOT))
# Resolved to: project_root/
```

---

## Verification

### Test 1: File Locations
```bash
$ find . -name "bench_*.py" -type f
./benchmarks/bench_compare.py
./benchmarks/bench_phase0.py
./benchmarks/macro/bench_image_pipeline.py
./benchmarks/macro/bench_neural_layer.py
./benchmarks/micro/bench_fusion_phase41.py
./benchmarks/micro/bench_fusion_poc.py
./benchmarks/micro/bench_fusion_rigorous.py
./benchmarks/micro/bench_fusion_simple.py
```
✅ All benchmarks are in `benchmarks/` directory tree

### Test 2: Path Resolution
```bash
$ python test_path_fixes.py
✓  benchmarks/bench_phase0.py                         (.parent × 2) → CORRECT
✓  benchmarks/micro/bench_fusion_phase41.py           (.parent × 3) → CORRECT
✓  benchmarks/micro/bench_fusion_simple.py            (.parent × 3) → CORRECT
✓  benchmarks/micro/bench_fusion_poc.py               (.parent × 3) → CORRECT
✓  benchmarks/micro/bench_fusion_rigorous.py          (.parent × 3) → CORRECT
✓  benchmarks/macro/bench_image_pipeline.py           (.parent × 3) → CORRECT
✓  benchmarks/macro/bench_neural_layer.py             (.parent × 3) → CORRECT
```
✅ All paths correctly resolve to project root

### Test 3: Invocation Pattern
```bash
# From documentation (54 references):
$ python benchmarks/bench_phase0.py

# From GitHub workflow:
python benchmarks/bench_phase0.py --output=benchmarks/results/current

# From PGO script:
python benchmarks/bench_phase0.py
```
✅ All invocations assume running from project root

---

## Conclusion

### ✅ BENCHMARKS SHOULD STAY IN benchmarks/ DIRECTORY

**Reasons:**
1. **Established pattern**: 54 references in codebase expect `benchmarks/`
2. **Organizational clarity**: Separates benchmarks from core code
3. **GitHub workflows**: All CI/CD expects `benchmarks/` structure
4. **Path fixes are correct**: Using `.parent.parent.parent` for subdirectories

### ✅ OUR FIXES WERE CORRECT

The micro-benchmark path fixes changed:
- From: `.parent.parent` (resolves to `benchmarks/`) ❌
- To: `.parent.parent.parent` (resolves to `project_root/`) ✅

This matches the pattern used by macro-benchmarks, which were already correct.

---

## Summary Table

| File | Location | Needs | Our Fix | Status |
|------|----------|-------|---------|--------|
| bench_phase0.py | benchmarks/ | .parent × 2 | No change | ✅ Already correct |
| bench_fusion_phase41.py | benchmarks/micro/ | .parent × 3 | ✅ Fixed | ✅ Correct |
| bench_fusion_simple.py | benchmarks/micro/ | .parent × 3 | ✅ Fixed | ✅ Correct |
| bench_fusion_poc.py | benchmarks/micro/ | .parent × 3 | ✅ Fixed | ✅ Correct |
| bench_fusion_rigorous.py | benchmarks/micro/ | .parent × 3 | ✅ Fixed | ✅ Correct |
| bench_image_pipeline.py | benchmarks/macro/ | .parent × 3 | No change | ✅ Already correct |
| bench_neural_layer.py | benchmarks/macro/ | .parent × 3 | No change | ✅ Already correct |

**All benchmarks remain in their proper `benchmarks/` subdirectories.**
**All path fixes correctly resolve to project root.**
**All invocations run from project root.**
