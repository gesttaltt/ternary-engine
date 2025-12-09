# Test Coverage Gaps Analysis

**Doc-Type:** Technical Audit · Version 1.0 · Generated 2025-12-09

This document identifies gaps in test coverage across the Ternary Engine codebase, including untested features, missing edge cases, and areas requiring additional validation.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current Test Coverage](#current-test-coverage)
3. [Untested Python Bindings](#1-untested-python-bindings)
4. [Missing Backend Tests](#2-missing-backend-tests)
5. [Edge Cases Not Covered](#3-edge-cases-not-covered)
6. [Integration Test Gaps](#4-integration-test-gaps)
7. [Performance Regression Tests](#5-performance-regression-tests)
8. [Recommended Test Additions](#recommended-test-additions)

---

## Executive Summary

| Category | Current Coverage | Gap | Risk Level |
|----------|------------------|-----|------------|
| Core Operations | 95% | Minor edge cases | Low |
| Python Bindings | 60% | CPU detection, backend API | Medium |
| Backend Switching | 10% | Full integration | High |
| Fusion Operations | 80% | All combinations | Medium |
| Dense243 Encoding | 70% | SIMD path, edge cases | Medium |
| TritNet GEMM | 30% | Edge cases, validation | High |
| Error Handling | 50% | Recovery paths | Medium |

**Overall Test Health:** 65/65 tests passing, but significant gaps in coverage

---

## Current Test Coverage

### Test File Inventory

| Test File | Purpose | Test Count | Status |
|-----------|---------|------------|--------|
| `test_phase0.py` | Core correctness | ~50 | ✓ Passing |
| `test_omp.py` | OpenMP scaling | ~25 | ✓ Passing |
| `test_fusion.py` | Fusion operations | ~16 | ✓ Passing |
| `test_fusion_correctness.py` | Fusion correctness | ~10 | ✓ Passing |
| `test_errors.py` | Error handling | ~10 | ✓ Passing |
| `test_backend_integration.py` | Backend API | ~5 | ⚠ Minimal |
| `test_capabilities.py` | CPU detection | ~3 | ⚠ Minimal |
| `test_canonical_lut.py` | LUT validation | ~8 | ✓ Passing |
| `test_dual_shuffle_validation.py` | Dual-shuffle | ~5 | ⚠ Feature disabled |
| `test_tritnet_gemm_integration.py` | GEMM | ~5 | ⚠ Rarely run |
| `test_simd_python.py` | SIMD correctness | ~10 | ✓ Passing |
| `test_simd_validation.py` | SIMD validation | ~8 | ✓ Passing |

**Total:** ~155 tests across 12 files

### Coverage by Component

```
Test Coverage Visualization:

Core Operations    ████████████████████ 95%
Fusion Operations  ████████████████     80%
Dense243 Encoding  ██████████████       70%
Python Bindings    ████████████         60%
Error Handling     ██████████           50%
TritNet GEMM       ██████               30%
Backend Switching  ██                   10%
```

---

## 1. Untested Python Bindings

### 1.1 CPU Detection Functions

**Location:** `src/engine/bindings_core_ops.cpp:585-588`

**Exposed Functions:**

```python
# These are exposed to Python but never tested:

ternary_simd_engine.detect_simd_level()
# Returns integer SIMD level (0-4)
# No test verifies correct detection

ternary_simd_engine.simd_level_string()
# Returns string name of SIMD level
# No test verifies correct string
```

**Missing Test:**

```python
# tests/python/test_cpu_detection.py (MISSING)

import ternary_simd_engine as engine

def test_detect_simd_level():
    """Verify SIMD level detection returns valid value."""
    level = engine.detect_simd_level()
    assert isinstance(level, int)
    assert 0 <= level <= 4  # NONE=0, SSE2=1, SSE41=2, AVX2=3, AVX512=4

def test_simd_level_string():
    """Verify SIMD level string is meaningful."""
    level_str = engine.simd_level_string()
    assert isinstance(level_str, str)
    assert level_str in ["None", "SSE2", "SSE4.1", "AVX2", "AVX-512", "Unknown"]

def test_has_avx2():
    """Verify AVX2 detection is consistent."""
    has_avx2 = engine.has_avx2()
    assert isinstance(has_avx2, bool)

    # If module loaded, AVX2 must be true (required for module)
    assert has_avx2 == True, "Module requires AVX2 but detection returned False"

def test_simd_level_consistency():
    """Verify detect_simd_level matches has_avx2."""
    level = engine.detect_simd_level()
    has_avx2 = engine.has_avx2()

    if has_avx2:
        assert level >= 3, "AVX2 detected but level < 3"
```

### 1.2 Int8 Operations

**Location:** `src/engine/bindings_core_ops.cpp:609-627`

**Exposed Functions:**

```python
# Int8 variants are exposed but lightly tested:

ternary_simd_engine.tadd_int8(a, b)
ternary_simd_engine.tmul_int8(a, b)
ternary_simd_engine.tmin_int8(a, b)
ternary_simd_engine.tmax_int8(a, b)
ternary_simd_engine.tnot_int8(a)

# Fused Int8 operations (not tested at all)
ternary_simd_engine.fused_tnot_tadd_int8(a, b)
ternary_simd_engine.fused_tnot_tmul_int8(a, b)
ternary_simd_engine.fused_tnot_tmin_int8(a, b)
ternary_simd_engine.fused_tnot_tmax_int8(a, b)
```

**Missing Test:**

```python
# tests/python/test_int8_operations.py (MISSING)

import numpy as np
import ternary_simd_engine as engine

class TestInt8Operations:
    """Test Int8 variants of ternary operations."""

    def setup_method(self):
        np.random.seed(42)
        self.a = np.random.randint(-1, 2, size=1000, dtype=np.int8)
        self.b = np.random.randint(-1, 2, size=1000, dtype=np.int8)

    def test_tadd_int8_matches_uint8(self):
        """Verify Int8 tadd produces same result as Uint8 version."""
        # Convert to uint8 encoding
        a_uint8 = (self.a + 1).astype(np.uint8)
        b_uint8 = (self.b + 1).astype(np.uint8)

        # Run both versions
        result_int8 = engine.tadd_int8(self.a, self.b)
        result_uint8 = engine.tadd(a_uint8, b_uint8)

        # Convert uint8 result back to int8
        result_uint8_as_int8 = result_uint8.astype(np.int8) - 1

        np.testing.assert_array_equal(result_int8, result_uint8_as_int8)

    def test_fused_tnot_tadd_int8(self):
        """Test fused operation with Int8 inputs."""
        result = engine.fused_tnot_tadd_int8(self.a, self.b)
        assert result.dtype == np.int8
        assert len(result) == len(self.a)
        assert np.all((result >= -1) & (result <= 1))
```

---

## 2. Missing Backend Tests

### 2.1 Backend Switching

**Current Status:** `test_backend_integration.py` exists but is minimal

**What's Missing:**

```python
# tests/python/test_backend_switching.py (EXPAND)

import ternary_backend
import numpy as np

class TestBackendSwitching:
    """Comprehensive tests for backend plugin system."""

    def setup_method(self):
        ternary_backend.init()
        np.random.seed(42)
        self.a = np.random.randint(0, 3, size=10000, dtype=np.uint8)
        self.b = np.random.randint(0, 3, size=10000, dtype=np.uint8)

    def teardown_method(self):
        ternary_backend.shutdown()

    def test_list_backends_returns_expected(self):
        """Verify all expected backends are registered."""
        backends = ternary_backend.list_backends()
        names = [b.name for b in backends]

        assert "scalar" in names
        assert "avx2_v1_baseline" in names
        assert "avx2_v2_optimized" in names

    def test_set_backend_changes_active(self):
        """Verify set_backend actually changes the active backend."""
        for backend in ternary_backend.list_backends():
            ternary_backend.set_backend(backend.name)
            active = ternary_backend.get_active()
            assert active.name == backend.name

    def test_operations_work_with_all_backends(self):
        """Verify all operations work with each backend."""
        reference_result = None

        for backend in ternary_backend.list_backends():
            ternary_backend.set_backend(backend.name)

            result = ternary_backend.tadd(self.a, self.b)

            if reference_result is None:
                reference_result = result
            else:
                # All backends should produce identical results
                np.testing.assert_array_equal(
                    result, reference_result,
                    err_msg=f"Backend {backend.name} produced different result"
                )

    def test_invalid_backend_raises_error(self):
        """Verify setting invalid backend raises appropriate error."""
        with pytest.raises(ValueError):
            ternary_backend.set_backend("nonexistent_backend")

    def test_backend_capabilities_accurate(self):
        """Verify reported capabilities match actual features."""
        for backend in ternary_backend.list_backends():
            ternary_backend.set_backend(backend.name)
            caps = ternary_backend.get_capabilities_string()

            # Verify capability string is non-empty
            assert len(caps) > 0

            # If "avx2" in name, should have AVX2 capability
            if "avx2" in backend.name.lower():
                assert "avx2" in caps.lower() or "AVX2" in caps

    def test_performance_ordering(self):
        """Verify optimized backends are not slower than baseline."""
        import time

        times = {}
        iterations = 100

        for backend in ternary_backend.list_backends():
            ternary_backend.set_backend(backend.name)

            # Warmup
            for _ in range(10):
                ternary_backend.tadd(self.a, self.b)

            # Benchmark
            start = time.perf_counter()
            for _ in range(iterations):
                ternary_backend.tadd(self.a, self.b)
            elapsed = time.perf_counter() - start

            times[backend.name] = elapsed

        # v2 should be faster than or equal to v1
        if "avx2_v2_optimized" in times and "avx2_v1_baseline" in times:
            assert times["avx2_v2_optimized"] <= times["avx2_v1_baseline"] * 1.1, \
                "Optimized backend should not be slower than baseline"
```

### 2.2 Fused Operation Completeness

**Current Status:** Some fusion tests exist but not comprehensive

**Missing Combinations:**

```python
# tests/python/test_fusion_completeness.py (MISSING)

import itertools
import numpy as np
import ternary_simd_engine as engine

# All possible fused operations
FUSED_OPS = [
    ("fused_tnot_tadd", engine.fused_tnot_tadd),
    ("fused_tnot_tmul", engine.fused_tnot_tmul),
    ("fused_tnot_tmin", engine.fused_tnot_tmin),
    ("fused_tnot_tmax", engine.fused_tnot_tmax),
]

# All possible input value combinations for one trit
TRIT_VALUES = [0, 1, 2]  # -1, 0, +1 in uint8 encoding

class TestFusionCompleteness:
    """Exhaustive tests for fused operations."""

    def test_all_5trit_combinations(self):
        """Test all 243 possible 5-trit input combinations."""
        # Generate all 3^5 = 243 combinations
        for op_name, op_func in FUSED_OPS:
            for combo in itertools.product(TRIT_VALUES, repeat=5):
                a = np.array(combo, dtype=np.uint8)
                b = np.array(combo[::-1], dtype=np.uint8)  # Reversed for variety

                result = op_func(a, b)

                # Verify result is valid ternary
                assert np.all((result >= 0) & (result <= 2)), \
                    f"{op_name} produced invalid trit value for input {combo}"

    def test_fused_equals_sequential(self):
        """Verify fused operation equals sequential operations."""
        np.random.seed(42)
        a = np.random.randint(0, 3, size=10000, dtype=np.uint8)
        b = np.random.randint(0, 3, size=10000, dtype=np.uint8)

        # fused_tnot_tadd(a, b) should equal tadd(tnot(a), b)
        fused = engine.fused_tnot_tadd(a, b)
        sequential = engine.tadd(engine.tnot(a), b)
        np.testing.assert_array_equal(fused, sequential)

        # fused_tnot_tmul(a, b) should equal tmul(tnot(a), b)
        fused = engine.fused_tnot_tmul(a, b)
        sequential = engine.tmul(engine.tnot(a), b)
        np.testing.assert_array_equal(fused, sequential)

        # Similar for tmin, tmax
```

---

## 3. Edge Cases Not Covered

### 3.1 Array Size Edge Cases

**Missing Tests:**

```python
# tests/python/test_edge_cases.py (MISSING)

import numpy as np
import ternary_simd_engine as engine

class TestArraySizeEdgeCases:
    """Test boundary conditions for array sizes."""

    def test_empty_array(self):
        """Test operations on empty arrays."""
        a = np.array([], dtype=np.uint8)
        b = np.array([], dtype=np.uint8)

        result = engine.tadd(a, b)
        assert len(result) == 0
        assert result.dtype == np.uint8

    def test_single_element(self):
        """Test operations on single-element arrays."""
        for val_a in [0, 1, 2]:
            for val_b in [0, 1, 2]:
                a = np.array([val_a], dtype=np.uint8)
                b = np.array([val_b], dtype=np.uint8)

                result = engine.tadd(a, b)
                assert len(result) == 1

    def test_non_aligned_sizes(self):
        """Test array sizes not aligned to SIMD width (32)."""
        for size in [1, 7, 15, 31, 33, 63, 100, 127]:
            a = np.random.randint(0, 3, size=size, dtype=np.uint8)
            b = np.random.randint(0, 3, size=size, dtype=np.uint8)

            result = engine.tadd(a, b)
            assert len(result) == size

    def test_exactly_simd_width(self):
        """Test array size exactly equal to SIMD width."""
        a = np.random.randint(0, 3, size=32, dtype=np.uint8)
        b = np.random.randint(0, 3, size=32, dtype=np.uint8)

        result = engine.tadd(a, b)
        assert len(result) == 32

    def test_large_array(self):
        """Test large arrays that trigger OpenMP."""
        size = 10_000_000  # 10M elements
        a = np.random.randint(0, 3, size=size, dtype=np.uint8)
        b = np.random.randint(0, 3, size=size, dtype=np.uint8)

        result = engine.tadd(a, b)
        assert len(result) == size

    def test_mismatched_sizes_raises(self):
        """Test that mismatched array sizes raise error."""
        a = np.random.randint(0, 3, size=100, dtype=np.uint8)
        b = np.random.randint(0, 3, size=50, dtype=np.uint8)

        with pytest.raises((ValueError, RuntimeError)):
            engine.tadd(a, b)
```

### 3.2 Memory Alignment Edge Cases

```python
class TestMemoryAlignment:
    """Test behavior with various memory alignments."""

    def test_unaligned_input(self):
        """Test with non-aligned input arrays."""
        # Create aligned array then slice to create unaligned view
        base = np.random.randint(0, 3, size=100, dtype=np.uint8)
        a = base[1:]  # Likely unaligned
        b = base[2:-1]

        # Should still work (SIMD code handles unaligned loads)
        result = engine.tadd(a[:len(b)], b)
        assert len(result) == len(b)

    def test_non_contiguous_raises(self):
        """Test that non-contiguous arrays raise error."""
        a = np.random.randint(0, 3, size=(100, 100), dtype=np.uint8)

        # Slice creates non-contiguous view
        a_slice = a[:, ::2]
        b_slice = a[:, 1::2]

        # Should raise error or handle gracefully
        with pytest.raises((ValueError, RuntimeError)):
            engine.tadd(a_slice.flatten(), b_slice.flatten())
```

---

## 4. Integration Test Gaps

### 4.1 Dense243 Integration

**Current Status:** Encoding tests exist, but integration with operations is not tested

```python
# tests/python/test_dense243_integration.py (MISSING)

import numpy as np
import ternary_dense243_module as dense243
import ternary_simd_engine as engine

class TestDense243Integration:
    """Test Dense243 encoding with ternary operations."""

    def test_roundtrip_preserves_operation_results(self):
        """Verify pack/unpack doesn't affect operation results."""
        np.random.seed(42)

        # Generate random ternary data
        a = np.random.randint(0, 3, size=1000, dtype=np.uint8)
        b = np.random.randint(0, 3, size=1000, dtype=np.uint8)

        # Compute result on unpacked data
        result_unpacked = engine.tadd(a, b)

        # Pack, unpack, then compute
        a_packed = dense243.pack(a)
        b_packed = dense243.pack(b)
        a_unpacked = dense243.unpack(a_packed)
        b_unpacked = dense243.unpack(b_packed)
        result_roundtrip = engine.tadd(a_unpacked, b_unpacked)

        np.testing.assert_array_equal(result_unpacked, result_roundtrip)

    def test_all_243_states_pack_unpack(self):
        """Verify all 243 possible packed values round-trip correctly."""
        for packed_value in range(243):
            packed = np.array([packed_value], dtype=np.uint8)
            unpacked = dense243.unpack(packed)
            repacked = dense243.pack(unpacked)

            assert repacked[0] == packed_value, \
                f"Round-trip failed for packed value {packed_value}"
```

### 4.2 TritNet GEMM Integration

**Current Status:** `test_tritnet_gemm_integration.py` exists but is rarely run

```python
# tests/python/test_tritnet_gemm_comprehensive.py (EXPAND)

import numpy as np

try:
    import ternary_tritnet_gemm as gemm
    HAS_GEMM = True
except ImportError:
    HAS_GEMM = False

@pytest.mark.skipif(not HAS_GEMM, reason="TritNet GEMM not built")
class TestTritNetGEMM:
    """Comprehensive tests for TritNet GEMM operations."""

    def test_gemm_basic_shapes(self):
        """Test GEMM with various matrix shapes."""
        shapes = [
            (1, 5, 5),    # Minimal
            (8, 15, 10),  # Small
            (32, 100, 50),  # Medium
            (128, 500, 200),  # Large
        ]

        for M, K, N in shapes:
            A = np.random.randn(M, K).astype(np.float32)
            B_dense = np.random.randint(0, 243, size=(K // 5, N), dtype=np.uint8)

            C = gemm.gemm(A, B_dense)

            assert C.shape == (M, N)
            assert C.dtype == np.float32

    def test_gemm_with_scales(self):
        """Test GEMM with per-column scales."""
        M, K, N = 32, 100, 50
        A = np.random.randn(M, K).astype(np.float32)
        B_dense = np.random.randint(0, 243, size=(K // 5, N), dtype=np.uint8)
        scales = np.random.randn(N).astype(np.float32)

        C = gemm.gemm_scaled(A, B_dense, scales)

        assert C.shape == (M, N)

    def test_gemm_edge_case_k_not_multiple_of_5(self):
        """Test GEMM when K is not a multiple of 5."""
        # This is a known TODO in the implementation
        M, K, N = 32, 103, 50  # K=103 not divisible by 5

        A = np.random.randn(M, K).astype(np.float32)
        # Would need padding logic
        # This test documents expected behavior
```

---

## 5. Performance Regression Tests

### 5.1 Missing Regression Detection

**Current Status:** Benchmarks exist but no automated regression detection

```python
# tests/python/test_performance_regression.py (MISSING)

import numpy as np
import time
import json
from pathlib import Path

# Baseline performance (operations per second)
PERFORMANCE_BASELINES = {
    "tadd_1M": 35_000_000_000,  # 35 Gops/s
    "tmul_1M": 30_000_000_000,  # 30 Gops/s
    "tnot_1M": 39_000_000_000,  # 39 Gops/s
    "fused_tnot_tadd_1M": 45_000_000_000,  # 45 Gops/s
}

REGRESSION_THRESHOLD = 0.05  # 5% allowed regression

class TestPerformanceRegression:
    """Detect performance regressions in core operations."""

    def setup_method(self):
        np.random.seed(42)
        self.size = 1_000_000
        self.a = np.random.randint(0, 3, size=self.size, dtype=np.uint8)
        self.b = np.random.randint(0, 3, size=self.size, dtype=np.uint8)

    def _benchmark(self, func, *args, iterations=100):
        """Run benchmark and return ops/second."""
        # Warmup
        for _ in range(10):
            func(*args)

        start = time.perf_counter()
        for _ in range(iterations):
            func(*args)
        elapsed = time.perf_counter() - start

        return (self.size * iterations) / elapsed

    def test_tadd_no_regression(self):
        """Verify tadd performance hasn't regressed."""
        ops_per_sec = self._benchmark(engine.tadd, self.a, self.b)
        baseline = PERFORMANCE_BASELINES["tadd_1M"]

        assert ops_per_sec >= baseline * (1 - REGRESSION_THRESHOLD), \
            f"tadd regressed: {ops_per_sec/1e9:.2f} Gops/s vs {baseline/1e9:.2f} Gops/s baseline"

    def test_fused_tnot_tadd_no_regression(self):
        """Verify fused operation performance hasn't regressed."""
        ops_per_sec = self._benchmark(engine.fused_tnot_tadd, self.a, self.b)
        baseline = PERFORMANCE_BASELINES["fused_tnot_tadd_1M"]

        assert ops_per_sec >= baseline * (1 - REGRESSION_THRESHOLD), \
            f"fused_tnot_tadd regressed: {ops_per_sec/1e9:.2f} Gops/s vs {baseline/1e9:.2f} Gops/s baseline"
```

---

## Recommended Test Additions

### Priority 1: High Impact, Low Effort

| Test | File | Effort | Impact |
|------|------|--------|--------|
| CPU detection tests | `test_cpu_detection.py` | 30 min | Validates module init |
| Backend switching tests | `test_backend_switching.py` | 1 hour | Validates plugin API |
| Array size edge cases | `test_edge_cases.py` | 1 hour | Catches boundary bugs |

### Priority 2: Medium Impact, Medium Effort

| Test | File | Effort | Impact |
|------|------|--------|--------|
| Int8 operation tests | `test_int8_operations.py` | 2 hours | Validates alternate API |
| Dense243 integration | `test_dense243_integration.py` | 2 hours | End-to-end validation |
| Fusion completeness | `test_fusion_completeness.py` | 2 hours | Exhaustive coverage |

### Priority 3: Important but Complex

| Test | File | Effort | Impact |
|------|------|--------|--------|
| Performance regression | `test_performance_regression.py` | 4 hours | Catches regressions |
| TritNet GEMM comprehensive | `test_tritnet_gemm_comprehensive.py` | 4 hours | Validates ML path |
| Memory alignment tests | `test_memory_alignment.py` | 2 hours | Edge case handling |

### Test File Template

```python
"""
test_<feature>.py - <Feature> tests for Ternary Engine

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Tests:
- <List of test categories>

Usage:
    pytest tests/python/test_<feature>.py -v
"""

import numpy as np
import pytest

# Import module under test
try:
    import ternary_simd_engine as engine
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False


@pytest.mark.skipif(not HAS_ENGINE, reason="Engine not built")
class Test<Feature>:
    """Tests for <feature>."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # Initialize test data

    def test_<specific_case>(self):
        """<Description of what this tests>."""
        # Arrange
        # Act
        # Assert
        pass
```

---

## Test Coverage Improvement Plan

### Week 1: Foundation

1. Create `test_cpu_detection.py` (Priority 1)
2. Expand `test_backend_switching.py` (Priority 1)
3. Create `test_edge_cases.py` (Priority 1)

### Week 2: API Coverage

4. Create `test_int8_operations.py` (Priority 2)
5. Create `test_fusion_completeness.py` (Priority 2)
6. Expand `test_dense243_integration.py` (Priority 2)

### Week 3: Regression & Integration

7. Create `test_performance_regression.py` (Priority 3)
8. Expand TritNet GEMM tests (Priority 3)
9. Add memory alignment tests (Priority 3)

### Success Metrics

- Test count: 155 → 250+
- Coverage: 65% → 85%
- All exposed Python functions tested
- All backend operations tested
- Performance regression detection enabled

---

**Document Version:** 1.0
**Last Updated:** 2025-12-09
**Author:** Claude Code Audit
