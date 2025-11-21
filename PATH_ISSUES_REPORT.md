# Path Configuration Issues Report

Generated: 2025-11-21

## Executive Summary

Found **6 critical path configuration issues** that will prevent benchmarking scripts from running correctly:
- 4 files in `benchmarks/micro/` have incorrect Python path setup
- 1 GitHub workflow has inconsistent baseline path references
- 1 potential runtime issue with module imports

---

## Issue 1: Micro-Benchmarks - Incorrect Project Root Path

**Severity**: CRITICAL
**Impact**: Scripts will fail with ImportError for ternary_simd_engine and ternary_fusion_engine

### Affected Files

1. `benchmarks/micro/bench_fusion_phase41.py` (lines 21-22)
2. `benchmarks/micro/bench_fusion_simple.py` (line 15)
3. `benchmarks/micro/bench_fusion_poc.py` (line 45)
4. `benchmarks/micro/bench_fusion_rigorous.py` (line 19)

### Current (INCORRECT) Code

```python
# Add project root and benchmarks to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
```

### Problem

For a file at `project_root/benchmarks/micro/bench_fusion_phase41.py`:
- `Path(__file__).parent.parent` resolves to `project_root/benchmarks/` (NOT project root!)
- `Path(__file__).parent` resolves to `project_root/benchmarks/micro/`

Then the scripts try to import:
- `import ternary_simd_engine` (located in `project_root/`)
- `import ternary_fusion_engine` (located in `project_root/`)

**These modules are NOT in `benchmarks/`, so imports will FAIL.**

### Correct Code

```python
# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
```

### Path Resolution Details

```
File location:  project_root/benchmarks/micro/bench_fusion_phase41.py
├─ Path(__file__)              = project_root/benchmarks/micro/bench_fusion_phase41.py
├─ .parent                     = project_root/benchmarks/micro/
├─ .parent.parent              = project_root/benchmarks/  ← WRONG!
└─ .parent.parent.parent       = project_root/            ← CORRECT!

Modules needed:
├─ ternary_simd_engine.pyd     = project_root/ternary_simd_engine.pyd
└─ ternary_fusion_engine.pyd   = project_root/ternary_fusion_engine.pyd
```

---

## Issue 2: bench_fusion_phase41.py - Redundant Path Additions

**Severity**: MINOR
**Impact**: Unnecessary path pollution, but not breaking

### File

`benchmarks/micro/bench_fusion_phase41.py` (lines 21-22)

### Current Code

```python
sys.path.insert(0, str(Path(__file__).parent.parent))  # benchmarks/
sys.path.insert(0, str(Path(__file__).parent))         # benchmarks/micro/
```

### Issue

After fixing Issue 1, only ONE sys.path addition is needed (project root). The script imports:
- `benchmark_framework` from `project_root/benchmarks/benchmark_framework.py`

If project root is in sys.path, Python can find `benchmarks/benchmark_framework.py` automatically via standard module resolution.

### Recommended Fix

```python
# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# benchmark_framework will be found as benchmarks.benchmark_framework
# or can be imported if benchmarks/__init__.py exists
```

---

## Issue 3: GitHub Workflow - Baseline Path Inconsistency

**Severity**: HIGH
**Impact**: Baseline comparison will never run in CI/CD

### File

`.github/workflows/benchmarks.yml` (lines 76, 101-108)

### Problem

**Line 76** - Benchmark results are written to:
```yaml
python benchmarks/bench_phase0.py --output=benchmarks/results/current
```

**Lines 101-108** - Comparison looks for baseline at:
```yaml
if [ -f "benchmarks/results/baseline/bench_results_*.json" ]; then
  python benchmarks/bench_compare.py \
    benchmarks/results/baseline/bench_results_*.json \
    benchmarks/results/current/bench_results_*.json
```

**Issue**: The baseline directory `benchmarks/results/baseline/` is NEVER created or populated by the workflow. The comparison will always skip with "No baseline found for comparison".

### Recommended Fix (Option 1: Store Previous Build Artifacts)

```yaml
- name: Download baseline from previous build
  if: github.event_name == 'pull_request'
  continue-on-error: true
  uses: dawidd6/action-download-artifact@v2
  with:
    workflow: benchmarks.yml
    branch: main
    name: benchmark-results
    path: benchmarks/results/baseline/

- name: Compare with baseline
  if: github.event_name == 'pull_request'
  continue-on-error: true
  run: |
    if [ -f "benchmarks/results/baseline/bench_results_"*.json ]; then
      python benchmarks/bench_compare.py \
        benchmarks/results/baseline/bench_results_*.json \
        benchmarks/results/current/bench_results_*.json \
        --threshold=5.0
    else
      echo "No baseline found - this is the first benchmark run"
    fi
```

### Recommended Fix (Option 2: Compare Against Main Branch Committed Results)

Store baseline results in git repository:
```yaml
- name: Checkout main branch baseline
  if: github.event_name == 'pull_request'
  run: |
    git fetch origin main
    git checkout origin/main -- benchmarks/results/baseline/ || echo "No baseline in main"

- name: Run current benchmark
  run: |
    python benchmarks/bench_phase0.py --output=benchmarks/results/current
```

---

## Issue 4: Macro-Benchmarks - Correct but Inconsistent Pattern

**Severity**: LOW
**Impact**: None (works correctly), but inconsistent with other benchmarks

### Files

- `benchmarks/macro/bench_image_pipeline.py` (line 17)
- `benchmarks/macro/bench_neural_layer.py` (line 16)

### Current Code (CORRECT)

```python
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

### Observation

The macro-benchmarks use the CORRECT path (`.parent.parent.parent` for project root), but micro-benchmarks use the WRONG path. This inconsistency suggests the micro-benchmark files were copy-pasted incorrectly.

### Recommendation

Update micro-benchmarks to match the correct pattern used by macro-benchmarks.

---

## Verification Commands

After fixing the issues, verify with these commands:

```bash
# Test micro-benchmarks can import modules
cd project_root/
python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path('benchmarks/micro').parent.parent.parent)); import ternary_simd_engine; print('✓ Import successful')"

# Run corrected benchmark
python benchmarks/micro/bench_fusion_phase41.py

# Verify macro-benchmarks still work
python benchmarks/macro/bench_image_pipeline.py
```

---

## Summary of Required Changes

### Micro-Benchmarks (4 files)

**bench_fusion_phase41.py** (lines 21-22):
```python
# BEFORE
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# AFTER
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
```

**bench_fusion_simple.py** (line 15):
```python
# BEFORE
sys.path.insert(0, str(Path(__file__).parent.parent))

# AFTER
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
```

**bench_fusion_poc.py** (line 45):
```python
# BEFORE
sys.path.insert(0, str(Path(__file__).parent.parent))

# AFTER
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
```

**bench_fusion_rigorous.py** (line 19):
```python
# BEFORE
sys.path.insert(0, str(Path(__file__).parent.parent))

# AFTER
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
```

### GitHub Workflow

**.github/workflows/benchmarks.yml** (lines 88-108):

Add artifact download step and fix comparison logic (see Issue 3 for full code).

---

## Testing Checklist

- [ ] Fix all 4 micro-benchmark files
- [ ] Run `python benchmarks/micro/bench_fusion_phase41.py` to verify imports work
- [ ] Run `python benchmarks/micro/bench_fusion_simple.py` to verify imports work
- [ ] Run `python benchmarks/micro/bench_fusion_poc.py` to verify imports work
- [ ] Run `python benchmarks/micro/bench_fusion_rigorous.py` to verify imports work
- [ ] Fix GitHub workflow baseline comparison
- [ ] Verify macro-benchmarks still work (should be unchanged)
- [ ] Test full benchmark suite: `python benchmarks/run_all_benchmarks.py --quick`
- [ ] Create PR to verify CI/CD workflow runs correctly

---

## Root Cause Analysis

The path errors in micro-benchmarks likely originated from:

1. **Copy-paste from bench_phase0.py** which correctly uses `.parent.parent` because it's in `benchmarks/`, not `benchmarks/micro/`
2. **Missing verification** - scripts were never run to test imports
3. **No CI/CD integration** - micro-benchmarks aren't in the GitHub workflow, so failures weren't caught

### Prevention Strategies

1. Add all benchmark scripts to CI/CD pipeline
2. Create a shared `benchmarks/__init__.py` with path setup helper
3. Use pytest fixtures for consistent path resolution
4. Add pre-commit hook to validate import paths

---

## Impact Assessment

| Issue | Severity | Files Affected | User Impact |
|-------|----------|----------------|-------------|
| Micro-benchmark paths | CRITICAL | 4 | Scripts fail immediately with ImportError |
| Workflow baseline | HIGH | 1 | Baseline comparison never runs |
| Path inconsistency | LOW | 6 | Confusion, maintenance burden |

**Estimated Fix Time**: 15-30 minutes
**Testing Time**: 10 minutes
**Total**: 25-40 minutes

---

## Additional Notes

### Module Location Map

```
project_root/
├── ternary_simd_engine.pyd         ← Built by build.py
├── ternary_fusion_engine.pyd       ← Built by build_fusion.py
├── benchmarks/
│   ├── benchmark_framework.py      ← Shared utilities
│   ├── bench_phase0.py            ← Correct: .parent.parent
│   ├── micro/
│   │   ├── bench_fusion_*.py      ← WRONG: .parent.parent (should be .parent.parent.parent)
│   └── macro/
│       ├── bench_*.py             ← CORRECT: .parent.parent.parent
```

### Import Resolution

When `sys.path` contains `project_root/`:
- ✓ `import ternary_simd_engine` → finds `project_root/ternary_simd_engine.pyd`
- ✓ `from benchmarks.benchmark_framework import ...` → finds `project_root/benchmarks/benchmark_framework.py`

When `sys.path` contains `project_root/benchmarks/`:
- ✗ `import ternary_simd_engine` → NOT FOUND (module is in parent directory)
- ✓ `from benchmark_framework import ...` → finds `benchmarks/benchmark_framework.py`

---

**End of Report**
