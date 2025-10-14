# Test Suite

Comprehensive test suite for the Ternary Engine library, covering correctness, performance, and scaling behavior.

## Structure

```
tests/
├── test_phase0.py     # Core correctness tests
├── test_omp.py        # OpenMP scaling tests
└── test_luts.cpp      # C++ unit tests (standalone)
```

## Quick Start

### Python Tests

From the project root:

```bash
# Core correctness tests
python tests/test_phase0.py

# OpenMP scaling tests
python tests/test_omp.py
```

### C++ Unit Tests

Compile and run standalone:

```bash
# Linux/macOS
g++ -std=c++17 -O0 tests/test_luts.cpp -o test_luts
./test_luts

# Windows (MSVC)
cl /std:c++17 /EHsc tests/test_luts.cpp
test_luts.exe
```

## Test Coverage

### test_phase0.py - Core Correctness

**Purpose**: Validates mathematical correctness of all ternary operations across different array sizes and input patterns.

**Coverage**:
- ✅ Binary operations (tadd, tmul, tmin, tmax)
- ✅ Unary operations (tnot)
- ✅ Edge cases (all -1, all 0, all +1)
- ✅ Mixed patterns (alternating, random)
- ✅ Size variations (32, 1K, 10K, 100K, 1M elements)
- ✅ SIMD alignment boundaries (31, 32, 33 elements)

**Expected Output**:
```
Running test_phase0.py...
Testing TADD...        ✓ PASS
Testing TMUL...        ✓ PASS
Testing TMIN...        ✓ PASS
Testing TMAX...        ✓ PASS
Testing TNOT...        ✓ PASS
All tests passed!
```

**Assertion Count**: ~50 test cases

### test_omp.py - OpenMP Scaling

**Purpose**: Validates OpenMP parallelization correctness and measures scaling behavior across different thread counts.

**Coverage**:
- ✅ Thread count variations (1, 2, 4, 8, 16 threads)
- ✅ Array sizes at OMP threshold (100K elements)
- ✅ Deterministic results (same input → same output)
- ✅ Speedup measurements (sequential vs parallel)
- ✅ Cache effects and NUMA behavior

**Expected Output**:
```
Running test_omp.py...
Threads: 1   | Time: 10.5ms | Speedup: 1.00×
Threads: 2   | Time:  5.8ms | Speedup: 1.81×
Threads: 4   | Time:  3.2ms | Speedup: 3.28×
Threads: 8   | Time:  1.9ms | Speedup: 5.53×
Threads: 16  | Time:  1.2ms | Speedup: 8.75×
All tests passed!
```

**Notes**:
- Requires AVX2-capable CPU
- Best results on CPUs with 8+ cores
- Speedup limited by memory bandwidth (~8-10× max)

### test_luts.cpp - C++ Unit Tests

**Purpose**: Low-level validation of LUT generation and scalar operations without Python overhead.

**Coverage**:
- ✅ Constexpr LUT generation (`ternary_lut_gen.h`)
- ✅ Scalar operations (`ternary_algebra.h`)
- ✅ Trit encoding/decoding
- ✅ Packing and unpacking
- ✅ Force-inline effectiveness

**Expected Output**:
```
Running C++ unit tests...
[PASS] LUT generation
[PASS] Scalar tadd
[PASS] Scalar tmul
[PASS] Scalar tmin
[PASS] Scalar tmax
[PASS] Scalar tnot
All 6 tests passed!
```

**Compilation Flags**: Use `-O0` to disable optimizations for testing

## Running Specific Tests

### Single Operation Test

```python
# In test_phase0.py, comment out other operations
import ternary_simd_engine as tc
import numpy as np

# Test only TADD
a = np.array([0b00, 0b01, 0b10], dtype=np.uint8)
b = np.array([0b00, 0b01, 0b10], dtype=np.uint8)
result = tc.tadd(a, b)
expected = np.array([0b00, 0b01, 0b10], dtype=np.uint8)
assert np.array_equal(result, expected), "TADD failed"
print("✓ TADD passed")
```

### Thread Count Override

```python
# In test_omp.py
import os
os.environ['OMP_NUM_THREADS'] = '4'  # Force 4 threads
import ternary_simd_engine as tc
# ... run tests
```

## Testing Best Practices

### Before Committing

Always run the full test suite:

```bash
# Quick validation (< 1 minute)
python tests/test_phase0.py

# Full suite (< 5 minutes)
python tests/test_phase0.py && python tests/test_omp.py
```

### After Optimization Changes

1. Run tests with and without optimization flags
2. Compare performance with baseline
3. Check for deterministic results (no randomness)

### Debugging Test Failures

```python
# Add verbose output
import numpy as np
np.set_printoptions(formatter={'int': lambda x: f'0b{x:02b}'})

result = tc.tadd(a, b)
print(f"Input A:    {a}")
print(f"Input B:    {b}")
print(f"Result:     {result}")
print(f"Expected:   {expected}")
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install pybind11 numpy
      - name: Build library
        run: python build.py
      - name: Run correctness tests
        run: python tests/test_phase0.py
      - name: Run OpenMP tests
        run: python tests/test_omp.py
      - name: Run C++ unit tests
        run: |
          g++ -std=c++17 tests/test_luts.cpp -o test_luts
          ./test_luts
```

## Performance Regression Testing

### Baseline Comparison

```bash
# Build reference implementation
python build_reference.py

# Run benchmarks
python benchmarks/bench_fair.py > baseline.txt

# Make changes, rebuild
python build.py

# Compare
python benchmarks/bench_fair.py > optimized.txt
diff baseline.txt optimized.txt
```

## Known Limitations

### Platform-Specific

- **AVX2 Required**: Tests will crash on CPUs without AVX2 support
- **Windows Path Length**: Some paths may exceed MAX_PATH on Windows
- **OpenMP Availability**: Some compilers may not support OpenMP

### Test Data

- **Deterministic**: All tests use fixed seeds for reproducibility
- **Limited Coverage**: Tests focus on correctness, not exhaustive fuzzing
- **No GPU Tests**: CUDA/OpenCL not yet supported

## Adding New Tests

### For New Operations

1. Add scalar test to `test_luts.cpp`:
```cpp
// Test new operation
trit result = new_op(0b00, 0b10);
assert(result == expected_value);
```

2. Add Python test to `test_phase0.py`:
```python
def test_new_op():
    a = np.array([0b00, 0b01, 0b10], dtype=np.uint8)
    b = np.array([0b00, 0b01, 0b10], dtype=np.uint8)
    result = tc.new_op(a, b)
    expected = compute_expected(a, b)
    assert np.array_equal(result, expected)
```

### For Edge Cases

```python
# Test boundary conditions
def test_edge_cases():
    # Empty array
    a = np.array([], dtype=np.uint8)
    result = tc.tadd(a, a)
    assert len(result) == 0

    # Single element
    a = np.array([0b01], dtype=np.uint8)
    result = tc.tadd(a, a)
    assert result[0] == 0b01

    # SIMD boundary (32 elements)
    a = np.ones(32, dtype=np.uint8) * 0b01
    result = tc.tadd(a, a)
    assert np.all(result == 0b01)
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'ternary_simd_engine'"

**Solution**: Build the library first:
```bash
python build.py
```

### "Illegal instruction" or "SIGILL"

**Cause**: CPU does not support AVX2

**Solution**: Check CPU compatibility:
```bash
grep avx2 /proc/cpuinfo  # Linux
sysctl machdep.cpu.features | grep AVX2  # macOS
```

### Tests Pass but Benchmarks Show Regression

**Cause**: Tests only validate correctness, not performance

**Solution**: Run full benchmark suite:
```bash
python benchmarks/bench_phase0.py
```

## Related Documentation

- **[../benchmarks/README.md](../benchmarks/README.md)** - Performance benchmarking
- **[../docs/api-reference/source-code-overview.md](../docs/api-reference/source-code-overview.md)** - Implementation details
- **[../docs/api-reference/ternary-core-simd.md](../docs/api-reference/ternary-core-simd.md)** - SIMD implementation

## Test Metrics

| Test File | LOC | Test Cases | Runtime | Coverage |
|-----------|-----|------------|---------|----------|
| test_phase0.py | ~200 | ~50 | ~10s | 95% |
| test_omp.py | ~150 | ~25 | ~30s | 80% (OpenMP) |
| test_luts.cpp | ~100 | ~20 | <1s | 90% (scalar) |

## Future Testing

### Phase 3 Testing (Planned)

- Multi-platform testing (ARM NEON, AVX-512)
- Fuzz testing with random inputs
- Property-based testing (Hypothesis)
- GPU correctness validation

### Continuous Performance Monitoring

- Automated benchmark runs on PR
- Performance regression alerts
- Historical performance tracking

---

**Last Updated**: 2025-10-13
**Maintained by**: Jonathan Verdun (Ternary Engine Project)
