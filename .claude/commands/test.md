Run the comprehensive Ternary Engine test suite.

Run all tests using the unified test runner:
```bash
python run_tests.py
```

Or run individual test suites:

**Correctness tests** (50 test cases):
```bash
python tests/test_phase0.py
```

**OpenMP scaling validation** (25 test cases):
```bash
python tests/test_omp.py
```

**Error handling tests**:
```bash
python tests/test_errors.py
```

**Fusion operation validation**:
```bash
python tests/test_fusion.py
```

Expected result: All 65 tests should pass on Windows x64. Linux/macOS results are experimental.

Note: Performance benchmarks are separate - use /benchmark command for those.
