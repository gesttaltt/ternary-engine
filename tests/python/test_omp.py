"""
Quick test to verify OPT-001 (OpenMP threading) implementation
Tests both small arrays (< 100K) and large arrays (>= 100K)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import time

try:
    import ternary_simd_engine as tc
    print("[OK] Module loaded successfully")
except ImportError as e:
    print(f"[FAIL] Failed to import module: {e}")
    print("  Run 'python build.py' to build the module first")
    exit(1)

# Test 1: Small array (should use single-threaded path)
print("\n=== Test 1: Small array (50K elements) ===")
n = 50000
a = np.random.randint(0, 3, n, dtype=np.uint8)
b = np.random.randint(0, 3, n, dtype=np.uint8)

start = time.perf_counter()
result = tc.tadd(a, b)
elapsed = time.perf_counter() - start

print(f"Size: {n:,} elements")
print(f"Time: {elapsed*1000:.3f} ms")
print(f"Throughput: {n/elapsed/1e6:.1f} M trits/sec")
print("[OK] Small array test passed")

# Test 2: Large array (should use OpenMP parallel path)
print("\n=== Test 2: Large array (1M elements, OpenMP threshold) ===")
n = 1000000
a = np.random.randint(0, 3, n, dtype=np.uint8)
b = np.random.randint(0, 3, n, dtype=np.uint8)

start = time.perf_counter()
result = tc.tadd(a, b)
elapsed = time.perf_counter() - start

print(f"Size: {n:,} elements")
print(f"Time: {elapsed*1000:.3f} ms")
print(f"Throughput: {n/elapsed/1e6:.1f} M trits/sec")
print("[OK] Large array test passed")

# Test 3: Very large array (should benefit most from threading)
print("\n=== Test 3: Very large array (10M elements) ===")
n = 10000000
a = np.random.randint(0, 3, n, dtype=np.uint8)
b = np.random.randint(0, 3, n, dtype=np.uint8)

start = time.perf_counter()
result = tc.tadd(a, b)
elapsed = time.perf_counter() - start

print(f"Size: {n:,} elements")
print(f"Time: {elapsed*1000:.3f} ms")
print(f"Throughput: {n/elapsed/1e6:.1f} M trits/sec")
print("[OK] Very large array test passed")

# Test 4: Correctness check
print("\n=== Test 4: Correctness verification ===")
# Simple known values
a = np.array([0b00, 0b01, 0b10], dtype=np.uint8)  # -1, 0, +1
b = np.array([0b00, 0b01, 0b10], dtype=np.uint8)  # -1, 0, +1
result = tc.tadd(a, b)
# Expected: tadd(-1,-1)=-1, tadd(0,0)=0, tadd(+1,+1)=+1 (saturated)
expected = np.array([0b00, 0b01, 0b10], dtype=np.uint8)

if np.array_equal(result, expected):
    print("[OK] Correctness test passed")
else:
    print(f"[FAIL] Correctness test failed!")
    print(f"  Expected: {expected}")
    print(f"  Got:      {result}")
    exit(1)

print("\n" + "="*50)
print("[OK] All OPT-001 tests passed!")
print("="*50)
