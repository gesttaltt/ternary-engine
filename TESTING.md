# Testing and CI/CD Guide

## Overview

Ternary Engine uses a comprehensive testing and continuous integration system to ensure code quality, correctness, and performance across multiple platforms and Python versions.

## Quick Start

### Running Tests Locally

```bash
# Run all tests
python run_tests.py

# Run specific test suite
python run_tests.py --suite=phase0
python run_tests.py --suite=omp
python run_tests.py --suite=errors

# Verbose output
python run_tests.py --verbose

# Without colors (for CI)
python run_tests.py --no-color
```

### Running Individual Tests

```bash
# Correctness tests
python tests/test_phase0.py

# OpenMP parallelization tests
python tests/test_omp.py

# Error handling tests
python tests/test_errors.py
```

### Running Benchmarks

```bash
# Full benchmark suite
python benchmarks/bench_phase0.py

# Quick benchmark (fewer sizes)
python benchmarks/bench_phase0.py --quick

# Master orchestrator (build + benchmark + compare)
python benchmarks/run_all_benchmarks.py
```

## Test Suites

### Phase 0 Correctness Tests (`test_phase0.py`)

**Purpose**: Validates mathematical correctness of all ternary operations

**Coverage**:
- All 5 operations (tadd, tmul, tmin, tmax, tnot)
- Truth table validation
- Edge cases (all -1, all 0, all +1)
- 39 total test cases

**Expected Runtime**: ~10 seconds

**Example Output**:
```
==================================================
  Phase 0 LUT Optimization Test Suite (Python)
==================================================

=== Testing tadd ===
  ✓ All 9 test cases passed

[...]

🎉 ALL TESTS PASSED! 🎉
```

### OpenMP Parallelization Tests (`test_omp.py`)

**Purpose**: Validates OpenMP threading and performance scaling

**Coverage**:
- Small arrays (< 100K elements) - serial SIMD
- Large arrays (≥ 100K elements) - OpenMP parallel
- Very large arrays (10M elements) - peak parallelization
- Correctness verification

**Expected Runtime**: ~5 seconds

**Example Output**:
```
=== Test 1: Small array (50K elements) ===
Time: 0.164 ms
Throughput: 305.3 M trits/sec
✓ Small array test passed

[...]

✓ All OPT-001 tests passed!
```

### Error Handling Tests (`test_errors.py`)

**Purpose**: Validates error conditions and edge cases

**Coverage**:
- Array size mismatch errors
- Empty array handling
- Single element arrays
- SIMD boundary cases (31, 32, 33, 63, 64, 65 elements)
- Large arrays (100M elements)
- Invalid trit values (0b11)
- Wrong data types
- Unary operation errors

**Expected Runtime**: ~30 seconds (includes 100M element test)

**Example Output**:
```
======================================================================
  Error Handling and Edge Case Test Suite
======================================================================

=== Test: Array Size Mismatch ===
✓ Correctly raised exception: RuntimeError

[...]

🎉 ALL ERROR HANDLING TESTS PASSED! 🎉
```

## Unified Test Runner

The `run_tests.py` script provides a unified interface for running all test suites with color-coded output and comprehensive reporting.

### Features

- ✅ Pre-flight checks (module build verification)
- ✅ Color-coded output (green=pass, red=fail, yellow=warning)
- ✅ Detailed error reporting
- ✅ Summary statistics
- ✅ Individual suite selection
- ✅ CI-friendly mode (--no-color)
- ✅ Timeout protection (60s per suite)

### Return Codes

- `0` - All tests passed
- `1` - One or more tests failed

### Usage Examples

```bash
# Standard run
python run_tests.py

# Quick mode (future implementation)
python run_tests.py --quick

# Specific suite
python run_tests.py --suite=errors

# Verbose output with full test details
python run_tests.py --verbose

# CI mode (no colors, machine-readable)
python run_tests.py --no-color
```

## Continuous Integration

### GitHub Actions Workflows

The project uses GitHub Actions for automated testing and benchmarking.

#### CI Workflow (`.github/workflows/ci.yml`)

**Triggers**:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

**Jobs**:

1. **Test on Windows** (`test-windows`)
   - Matrix: Python 3.8, 3.9, 3.10, 3.11, 3.12
   - Builds module with MSVC
   - Runs all test suites
   - Uploads artifacts (.pyd files)

2. **Test on Linux** (`test-linux`)
   - Matrix: Python 3.8, 3.9, 3.10, 3.11, 3.12
   - Checks AVX2 CPU support
   - Builds module with GCC
   - Runs all test suites
   - Uploads artifacts (.so files)

3. **Test on macOS** (`test-macos`)
   - Matrix: Python 3.8, 3.9, 3.10, 3.11, 3.12
   - Checks AVX2 CPU support
   - Builds module with Clang
   - Runs all test suites
   - Uploads artifacts (.so files)

4. **Code Quality** (`code-quality`)
   - Checks project structure
   - Verifies critical files exist
   - Reports code statistics

**Total Test Matrix**: 15 configurations (3 platforms × 5 Python versions)

#### Benchmark Workflow (`.github/workflows/benchmarks.yml`)

**Triggers**:
- Push to `main` branch
- Pull requests to `main`
- Manual workflow dispatch

**Jobs**:

1. **Benchmark (Linux)** (`benchmark`)
   - Runs quick benchmark suite
   - Compares with baseline (PR only)
   - Uploads results as artifacts
   - Generates performance summary

2. **Benchmark (Windows)** (`benchmark-windows`)
   - Windows-specific benchmarking
   - Uploads results separately

3. **PGO Benchmark** (`pgo-benchmark`)
   - Runs only on main branch pushes
   - Full PGO build + benchmark
   - Measures maximum performance

**Timeout**: 30-45 minutes per job

### Viewing CI Results

1. **GitHub Actions Tab**: https://github.com/gesttaltt/ternary-engine/actions
2. **Pull Request Checks**: Automatically shown on PRs
3. **Artifacts**: Download from workflow runs (valid for 90 days)

### CI Failure Debugging

```bash
# Download artifacts locally
gh run download <run-id>

# View workflow logs
gh run view <run-id> --log

# Re-run failed jobs
gh run rerun <run-id> --failed
```

## Adding New Tests

### 1. Create Test File

```python
# tests/test_feature.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import ternary_simd_engine as tc

def test_feature():
    """Test description"""
    # Test implementation
    assert condition, "Failure message"
    print("✓ Test passed")
    return True

def main():
    results = []
    results.append(test_feature())

    if all(results):
        print("✓ ALL TESTS PASSED!")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    exit(main())
```

### 2. Register in Test Runner

Edit `run_tests.py`:

```python
test_suites = {
    # ... existing suites ...
    'feature': {
        'name': 'Feature Tests',
        'script': TESTS_DIR / 'test_feature.py',
        'required': True
    }
}
```

### 3. Verify Locally

```bash
python run_tests.py --suite=feature
python run_tests.py  # Run all including new test
```

## Performance Regression Detection

### Manual Comparison

```bash
# Establish baseline
python benchmarks/bench_phase0.py --output=benchmarks/results/baseline

# After changes
python benchmarks/bench_phase0.py --output=benchmarks/results/current

# Compare
python benchmarks/bench_compare.py \
    benchmarks/results/baseline/bench_results_*.json \
    benchmarks/results/current/bench_results_*.json \
    --threshold=5.0
```

### Automated (CI)

The benchmark workflow automatically compares PR performance against the main branch baseline (when available).

## Best Practices

### Before Committing

```bash
# 1. Run all tests
python run_tests.py

# 2. Run quick benchmark
python benchmarks/bench_phase0.py --quick

# 3. Check for regressions
python benchmarks/bench_compare.py baseline.json current.json
```

### Writing Tests

1. **Test one thing per test function**
2. **Use descriptive names** (test_array_size_mismatch)
3. **Provide clear failure messages**
4. **Include both positive and negative cases**
5. **Test edge cases** (empty, single element, boundary conditions)

### Performance Testing

1. **Use consistent hardware** for baseline comparisons
2. **Warmup iterations** to stabilize CPU state
3. **Multiple runs** for statistical significance
4. **Document test environment** (CPU, RAM, OS)

## Troubleshooting

### Test Failures

**Module not found**:
```bash
# Rebuild module
python build.py

# Verify import
python -c "import ternary_simd_engine; print('OK')"
```

**Segmentation fault**:
- Usually indicates AVX2 incompatibility
- Check CPU support: `grep avx2 /proc/cpuinfo` (Linux)
- Try reference build without AVX2

**Timeout**:
- Increase timeout in run_tests.py (default 60s)
- Use `--quick` mode for faster testing

### CI Failures

**Platform-specific failures**:
- Check artifact uploads for detailed logs
- Test locally on same platform if possible
- Review platform-specific build flags

**Flaky tests**:
- Add retry logic or increase tolerances
- Check for timing-dependent behavior
- Verify thread safety

## Coverage Goals

Current test coverage:

- ✅ **Correctness**: 100% (all operations validated)
- ✅ **Error handling**: 100% (size mismatch, empty arrays, invalid input)
- ✅ **Edge cases**: 95% (SIMD boundaries, large arrays)
- ✅ **Platform**: 100% (Windows, Linux, macOS)
- ✅ **Python versions**: 100% (3.8-3.12)

Future coverage targets:

- ⏳ **Stress testing**: Multi-threading edge cases
- ⏳ **Fuzzing**: Random input validation
- ⏳ **Property-based testing**: Hypothesis integration
- ⏳ **Integration tests**: Real-world usage scenarios

## Related Documentation

- **[tests/README.md](tests/README.md)** - Test suite details
- **[benchmarks/README.md](benchmarks/README.md)** - Benchmarking guide
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
- **[build/README.md](build/README.md)** - Build system docs

---

**Last Updated**: 2025-10-14
**Maintained by**: Jonathan Verdun (Ternary Engine Project)
