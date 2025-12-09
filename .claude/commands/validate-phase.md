# Phase Validation Command

Validate a development phase for production readiness with comprehensive testing and documentation review.

## Task

1. **Run Test Suite**
   Execute: `python tests/run_tests.py`
   - Record pass/fail counts
   - Note any test failures with details

2. **Run Performance Benchmarks**
   Execute: `python benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py`
   - Capture throughput metrics
   - Compare against baseline (45.3 Gops/s fusion, 39.1 Gops/s element-wise)

3. **Check Documentation Status**
   Verify these files are up-to-date:
   - README.md performance claims match benchmarks
   - CHANGELOG.md has recent entries
   - docs/ has relevant API documentation

4. **Verify No Regressions**
   - Test count should be ≥65
   - Performance should be within 5% of baseline
   - No new test failures

5. **Generate Validation Report**

## Output Format

```markdown
## Phase Validation Report
**Phase:** [Phase name/number]
**Date:** [current date]
**Platform:** Windows x64

### Test Results
- **Total Tests:** X
- **Passed:** X
- **Failed:** X
- **Status:** PASS/FAIL

### Performance Results
| Operation | Throughput | vs Baseline | Status |
|-----------|------------|-------------|--------|
| ...       | X Gops/s   | +/-X%       | OK/WARN|

### Documentation Status
- [ ] README.md updated
- [ ] CHANGELOG.md updated
- [ ] API docs current

### Overall Assessment
**GO/NO-GO:** [GO or NO-GO]

**Reason:** [Brief explanation]

### Next Steps
[Recommended actions]
```
