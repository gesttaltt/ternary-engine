"""
test_backend_integration.py - Backend System Integration Tests

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Tests for v1.2.0 backend system integration:
- Backend initialization
- Backend discovery and selection
- Dispatch operations
- Cross-backend correctness
- Performance comparison (informational)

Usage:
    python tests/python/test_backend_integration.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# Test counters
tests_run = 0
tests_passed = 0
tests_failed = 0

def test(name):
    """Test decorator"""
    global tests_run
    tests_run += 1
    print(f"\n[{tests_run}] Testing: {name}...")
    return name

def assert_true(condition, message=""):
    """Assert helper"""
    global tests_passed, tests_failed
    if condition:
        print(f"  ✓ PASS")
        tests_passed += 1
    else:
        print(f"  ✗ FAIL: {message}")
        tests_failed += 1

def assert_equal(a, b, message=""):
    """Assert equal helper"""
    global tests_passed, tests_failed
    if a == b:
        print(f"  ✓ PASS")
        tests_passed += 1
    else:
        print(f"  ✗ FAIL: {a} != {b}. {message}")
        tests_failed += 1

def arrays_equal(a, b):
    """Check if two NumPy arrays are equal"""
    return np.array_equal(a, b)

# ============================================================================
# Backend System Tests
# ============================================================================

def test_backend_import():
    """Test backend module import"""
    test("Backend module import")
    try:
        import ternary_backend
        assert_true(True)
        return ternary_backend
    except ImportError as e:
        assert_true(False, f"Failed to import ternary_backend: {e}")
        return None

def test_backend_init(tb):
    """Test backend initialization"""
    test("Backend initialization")
    try:
        success = tb.init()
        assert_true(success, "Initialization failed")
    except Exception as e:
        assert_true(False, f"Exception during init: {e}")

def test_list_backends(tb):
    """Test backend listing"""
    test("List available backends")
    try:
        backends = tb.list_backends()
        print(f"  Found {len(backends)} backend(s):")
        for backend in backends:
            print(f"    - {backend.name} (v{backend.version})")
            print(f"      {backend.description}")
            print(f"      Active: {backend.is_active}")
        assert_true(len(backends) >= 1, "Should have at least Scalar backend")
    except Exception as e:
        assert_true(False, f"Exception: {e}")

def test_get_active_backend(tb):
    """Test get active backend"""
    test("Get active backend")
    try:
        active = tb.get_active()
        print(f"  Active backend: {active}")
        assert_true(active != "None", "Should have an active backend")
    except Exception as e:
        assert_true(False, f"Exception: {e}")

def test_backend_selection(tb):
    """Test backend selection"""
    test("Backend selection")
    try:
        # Try to set Scalar backend (always available)
        success = tb.set_backend("Scalar")
        assert_true(success, "Failed to set Scalar backend")

        active = tb.get_active()
        assert_equal(active, "Scalar", "Active backend should be Scalar")
    except Exception as e:
        assert_true(False, f"Exception: {e}")

# ============================================================================
# Dispatch Operations Tests
# ============================================================================

def test_dispatch_tnot(tb):
    """Test ternary NOT dispatch"""
    test("Dispatch tnot")
    try:
        # Create test data (0, 1, 2 encoding for -1, 0, +1)
        a = np.array([0, 1, 2, 0, 1, 2], dtype=np.uint8)

        result = tb.tnot(a)

        # tnot inverts: 0→2, 1→1, 2→0
        expected = np.array([2, 1, 0, 2, 1, 0], dtype=np.uint8)

        assert_true(arrays_equal(result, expected),
                   f"Result mismatch: {result} != {expected}")
    except Exception as e:
        assert_true(False, f"Exception: {e}")

def test_dispatch_tadd(tb):
    """Test ternary ADD dispatch"""
    test("Dispatch tadd")
    try:
        a = np.array([0, 1, 2, 0, 1], dtype=np.uint8)
        b = np.array([0, 1, 2, 1, 2], dtype=np.uint8)

        result = tb.tadd(a, b)

        # Basic sanity: result should be in valid range
        assert_true(np.all(result <= 2) and np.all(result >= 0),
                   f"Result out of range: {result}")
    except Exception as e:
        assert_true(False, f"Exception: {e}")

def test_dispatch_tmul(tb):
    """Test ternary MUL dispatch"""
    test("Dispatch tmul")
    try:
        a = np.array([0, 1, 2, 0, 1], dtype=np.uint8)
        b = np.array([0, 1, 2, 1, 2], dtype=np.uint8)

        result = tb.tmul(a, b)

        assert_true(np.all(result <= 2) and np.all(result >= 0),
                   f"Result out of range: {result}")
    except Exception as e:
        assert_true(False, f"Exception: {e}")

def test_dispatch_tmax(tb):
    """Test ternary MAX dispatch"""
    test("Dispatch tmax")
    try:
        a = np.array([0, 1, 2, 0, 1], dtype=np.uint8)
        b = np.array([2, 1, 0, 1, 2], dtype=np.uint8)

        result = tb.tmax(a, b)

        # tmax should return the greater of each pair
        expected = np.array([2, 1, 2, 1, 2], dtype=np.uint8)

        assert_true(arrays_equal(result, expected),
                   f"Result mismatch: {result} != {expected}")
    except Exception as e:
        assert_true(False, f"Exception: {e}")

def test_dispatch_tmin(tb):
    """Test ternary MIN dispatch"""
    test("Dispatch tmin")
    try:
        a = np.array([0, 1, 2, 0, 1], dtype=np.uint8)
        b = np.array([2, 1, 0, 1, 2], dtype=np.uint8)

        result = tb.tmin(a, b)

        # tmin should return the lesser of each pair
        expected = np.array([0, 1, 0, 0, 1], dtype=np.uint8)

        assert_true(arrays_equal(result, expected),
                   f"Result mismatch: {result} != {expected}")
    except Exception as e:
        assert_true(False, f"Exception: {e}")

# ============================================================================
# Fusion Operations Tests (Phase 4.1)
# ============================================================================

def test_dispatch_fused_tnot_tadd(tb):
    """Test fused tnot(tadd(a, b))"""
    test("Dispatch fused_tnot_tadd")
    try:
        a = np.array([0, 1, 2, 0, 1], dtype=np.uint8)
        b = np.array([0, 1, 2, 1, 2], dtype=np.uint8)

        # Calculate expected result: tnot(tadd(a, b))
        # tadd: [0+0=0, 1+1=2, 2+2=1, 0+1=1, 1+2=2] (saturated)
        # tnot: [2, 0, 1, 1, 0]
        
        # Let's use the scalar backend to verify truth if we trust it, 
        # or calculate manually. Manual is safer.
        # a: -1, 0, +1, -1, 0
        # b: -1, 0, +1,  0, +1
        # sum: -2->-1, 0, +2->+1, -1, +1
        # tnot(sum): +1, 0, -1, +1, -1
        # encoded: 2, 1, 0, 2, 0
        
        expected = np.array([2, 1, 0, 2, 0], dtype=np.uint8)
        
        result = tb.fused_tnot_tadd(a, b)

        assert_true(arrays_equal(result, expected),
                   f"Result mismatch: {result} != {expected}")
    except Exception as e:
        assert_true(False, f"Exception: {e}")

def test_dispatch_fused_tnot_tmul(tb):
    """Test fused tnot(tmul(a, b))"""
    test("Dispatch fused_tnot_tmul")
    try:
        a = np.array([0, 1, 2, 0, 1], dtype=np.uint8)
        b = np.array([0, 1, 2, 1, 2], dtype=np.uint8)
        
        # a: -1, 0, +1, -1, 0
        # b: -1, 0, +1,  0, +1
        # mul: +1, 0, +1, 0, 0
        # tnot: -1, 0, -1, 0, 0
        # encoded: 0, 1, 0, 1, 1
        
        expected = np.array([0, 1, 0, 1, 1], dtype=np.uint8)

        result = tb.fused_tnot_tmul(a, b)

        assert_true(arrays_equal(result, expected),
                   f"Result mismatch: {result} != {expected}")
    except Exception as e:
        assert_true(False, f"Exception: {e}")

def test_dispatch_fused_tnot_tmin(tb):
    """Test fused tnot(tmin(a, b))"""
    test("Dispatch fused_tnot_tmin")
    try:
        a = np.array([0, 1, 2, 0, 1], dtype=np.uint8)
        b = np.array([2, 1, 0, 1, 2], dtype=np.uint8)
        
        # a: -1, 0, +1, -1, 0
        # b: +1, 0, -1,  0, +1
        # min: -1, 0, -1, -1, 0
        # tnot: +1, 0, +1, +1, 0
        # encoded: 2, 1, 2, 2, 1
        
        expected = np.array([2, 1, 2, 2, 1], dtype=np.uint8)

        result = tb.fused_tnot_tmin(a, b)

        assert_true(arrays_equal(result, expected),
                   f"Result mismatch: {result} != {expected}")
    except Exception as e:
        assert_true(False, f"Exception: {e}")

def test_dispatch_fused_tnot_tmax(tb):
    """Test fused tnot(tmax(a, b))"""
    test("Dispatch fused_tnot_tmax")
    try:
        a = np.array([0, 1, 2, 0, 1], dtype=np.uint8)
        b = np.array([2, 1, 0, 1, 2], dtype=np.uint8)
        
        # a: -1, 0, +1, -1, 0
        # b: +1, 0, -1,  0, +1
        # max: +1, 0, +1, 0, +1
        # tnot: -1, 0, -1, 0, -1
        # encoded: 0, 1, 0, 1, 0
        
        expected = np.array([0, 1, 0, 1, 0], dtype=np.uint8)

        result = tb.fused_tnot_tmax(a, b)

        assert_true(arrays_equal(result, expected),
                   f"Result mismatch: {result} != {expected}")
    except Exception as e:
        assert_true(False, f"Exception: {e}")

# ============================================================================
# Cross-Backend Correctness Tests
# ============================================================================

def test_cross_backend_correctness(tb):
    """Test correctness across different backends"""
    test("Cross-backend correctness")

    try:
        # Get all backends
        backends = tb.list_backends()
        backend_names = [b.name for b in backends]

        print(f"  Testing backends: {backend_names}")

        if len(backends) < 2:
            print("  SKIP: Need at least 2 backends for comparison")
            assert_true(True)  # Don't fail
            return

        # Generate test data (100 random trits)
        np.random.seed(42)
        n = 100
        a = np.random.randint(0, 3, n, dtype=np.uint8)
        b = np.random.randint(0, 3, n, dtype=np.uint8)

        # Collect results from each backend
        results = {}
        for backend_name in backend_names:
            tb.set_backend(backend_name)
            results[backend_name] = {
                'tnot': tb.tnot(a),
                'tadd': tb.tadd(a, b),
                'tmul': tb.tmul(a, b),
                'tmax': tb.tmax(a, b),
                'tmin': tb.tmin(a, b),
                'fused_tnot_tadd': tb.fused_tnot_tadd(a, b),
                'fused_tnot_tmul': tb.fused_tnot_tmul(a, b),
                'fused_tnot_tmin': tb.fused_tnot_tmin(a, b),
                'fused_tnot_tmax': tb.fused_tnot_tmax(a, b),
            }

        # Compare all backends against the first one (reference)
        reference_name = backend_names[0]
        reference_results = results[reference_name]

        all_match = True
        ops_to_test = ['tnot', 'tadd', 'tmul', 'tmax', 'tmin', 
                       'fused_tnot_tadd', 'fused_tnot_tmul', 'fused_tnot_tmin', 'fused_tnot_tmax']
        
        for backend_name in backend_names[1:]:
            for op in ops_to_test:
                if not arrays_equal(reference_results[op], results[backend_name][op]):
                    print(f"  ✗ Mismatch: {reference_name}.{op} != {backend_name}.{op}")
                    all_match = False
                else:
                    print(f"  ✓ Match: {reference_name}.{op} == {backend_name}.{op}")

        assert_true(all_match, "Not all backends produce same results")

    except Exception as e:
        assert_true(False, f"Exception: {e}")

# ============================================================================
# Performance Comparison (Informational)
# ============================================================================

def test_performance_comparison(tb):
    """Performance comparison (informational only)"""
    test("Performance comparison (informational)")

    try:
        import time

        backends = tb.list_backends()
        backend_names = [b.name for b in backends]

        print(f"  Comparing performance across backends:")

        # Test data (1M trits)
        n = 1_000_000
        a = np.random.randint(0, 3, n, dtype=np.uint8)
        b = np.random.randint(0, 3, n, dtype=np.uint8)

        for backend_name in backend_names:
            tb.set_backend(backend_name)

            # Warm-up
            _ = tb.tadd(a, b)

            # Benchmark
            start = time.perf_counter()
            for _ in range(10):
                _ = tb.tadd(a, b)
            elapsed = time.perf_counter() - start

            # Calculate throughput
            ops = n * 10  # 10 iterations
            mops_per_sec = (ops / elapsed) / 1e6

            print(f"    {backend_name:12s}: {mops_per_sec:8.2f} Mops/s")

        print("  Note: For detailed benchmarks, run benchmarks/bench_backends.py")
        assert_true(True)  # Informational test

    except Exception as e:
        print(f"  Warning: Performance test failed: {e}")
        assert_true(True)  # Don't fail on performance test

# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    print("="*70)
    print("Ternary Backend Integration Tests (v1.2.0)")
    print("="*70)

    # Import backend module
    tb = test_backend_import()
    if tb is None:
        print("\n❌ FATAL: Cannot import ternary_backend module")
        print("   Run: python build/build_backend.py")
        return 1

    # Backend system tests
    print("\n--- Backend System Tests ---")
    test_backend_init(tb)
    test_list_backends(tb)
    test_get_active_backend(tb)
    test_backend_selection(tb)

    # Dispatch operations
    print("\n--- Dispatch Operations ---")
    test_dispatch_tnot(tb)
    test_dispatch_tadd(tb)
    test_dispatch_tmul(tb)
    test_dispatch_tmax(tb)
    test_dispatch_tmin(tb)
    
    # Fusion operations
    print("\n--- Fusion Operations ---")
    test_dispatch_fused_tnot_tadd(tb)
    test_dispatch_fused_tnot_tmul(tb)
    test_dispatch_fused_tnot_tmin(tb)
    test_dispatch_fused_tnot_tmax(tb)

    # Cross-backend correctness
    print("\n--- Cross-Backend Correctness ---")
    test_cross_backend_correctness(tb)

    # Performance comparison
    print("\n--- Performance Comparison ---")
    test_performance_comparison(tb)

    # Summary
    print("\n" + "="*70)
    print(f"Test Results: {tests_passed}/{tests_run} passed, {tests_failed} failed")
    print("="*70)

    if tests_failed == 0:
        print("✅ ALL TESTS PASSED")
        return 0
    else:
        print(f"❌ {tests_failed} TEST(S) FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
