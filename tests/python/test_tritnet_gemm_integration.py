#!/usr/bin/env python3
"""
test_tritnet_gemm_integration.py - Test TritNet GEMM Integration

Validates that the optimized C++ GEMM is correctly integrated into
the TernaryLinear layer and produces correct results.

USAGE:
    python tests/test_tritnet_gemm_integration.py

REQUIREMENTS:
    - ternary_tritnet_gemm module must be built
    - Run: python build/build_tritnet_gemm.py

TESTS:
    1. Module availability check
    2. Forward pass correctness (GEMM vs PyTorch)
    3. Gradient flow validation
    4. Performance comparison
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to path (for modules in root)
# File is in tests/python/, so go up 3 levels: test -> python/ -> tests/ -> project root
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))  # For ternary_tritnet_gemm module
sys.path.insert(0, str(ROOT_DIR / "models" / "tritnet" / "src"))  # For ternary_layers

from ternary_layers import TernaryLinear, GEMM_AVAILABLE


def test_module_availability():
    """Test 1: Verify ternary_tritnet_gemm module is available."""
    print("=" * 70)
    print("TEST 1: Module Availability")
    print("=" * 70)

    if GEMM_AVAILABLE:
        print("✅ ternary_tritnet_gemm module is available")
        import ternary_tritnet_gemm as gemm
        print(f"   Functions: {', '.join([f for f in dir(gemm) if not f.startswith('_')])}")
        return True
    else:
        print("❌ ternary_tritnet_gemm module NOT available")
        print("   Build with: python build/build_tritnet_gemm.py")
        return False


def test_forward_pass_correctness():
    """Test 2: Verify GEMM produces same results as PyTorch."""
    print("\n" + "=" * 70)
    print("TEST 2: Forward Pass Correctness")
    print("=" * 70)

    if not GEMM_AVAILABLE:
        print("⚠️  Skipping (module not available)")
        return False

    # Create two identical layers
    in_features, out_features = 20, 16
    layer_pytorch = TernaryLinear(in_features, out_features, use_ternary_gemm=False, quantize_weights=True)
    layer_gemm = TernaryLinear(in_features, out_features, use_ternary_gemm=True, quantize_weights=True)

    # Copy weights so they're identical
    with torch.no_grad():
        layer_gemm.weight.copy_(layer_pytorch.weight)

    # Test input
    batch_size = 8
    x = torch.randn(batch_size, in_features)

    # Forward pass
    y_pytorch = layer_pytorch(x)
    y_gemm = layer_gemm(x)

    # Compare outputs
    max_error = torch.abs(y_pytorch - y_gemm).max().item()
    mean_error = torch.abs(y_pytorch - y_gemm).mean().item()

    print(f"Input shape: [{batch_size}, {in_features}]")
    print(f"Output shape: [{batch_size}, {out_features}]")
    print(f"Max absolute error: {max_error:.2e}")
    print(f"Mean absolute error: {mean_error:.2e}")

    if max_error < 1e-5:
        print("✅ GEMM output matches PyTorch (within 1e-5)")
        return True
    else:
        print("❌ GEMM output differs from PyTorch!")
        print(f"   PyTorch output (first 5): {y_pytorch[0, :5]}")
        print(f"   GEMM output (first 5):    {y_gemm[0, :5]}")
        return False


def test_gradient_flow():
    """Test 3: Verify gradients still work with GEMM (via PyTorch fallback during backward)."""
    print("\n" + "=" * 70)
    print("TEST 3: Gradient Flow")
    print("=" * 70)

    if not GEMM_AVAILABLE:
        print("⚠️  Skipping (module not available)")
        return False

    # Create layer with GEMM
    layer = TernaryLinear(10, 16, use_ternary_gemm=True, quantize_weights=True)

    # Forward + backward
    x = torch.randn(4, 10, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()

    # Check gradients exist
    has_weight_grad = layer.weight.grad is not None
    has_input_grad = x.grad is not None

    print(f"Weight gradient exists: {has_weight_grad}")
    print(f"Input gradient exists: {has_input_grad}")

    if has_weight_grad and has_input_grad:
        print("✅ Gradients flow correctly")
        print(f"   Weight grad norm: {layer.weight.grad.norm().item():.4f}")
        print(f"   Input grad norm: {x.grad.norm().item():.4f}")
        return True
    else:
        print("❌ Gradient flow broken!")
        return False


def test_performance_comparison():
    """Test 4: Compare performance of GEMM vs PyTorch."""
    print("\n" + "=" * 70)
    print("TEST 4: Performance Comparison")
    print("=" * 70)

    if not GEMM_AVAILABLE:
        print("⚠️  Skipping (module not available)")
        return False

    import time

    # Test configuration (small like TritNet)
    in_features, out_features = 20, 16
    batch_size = 1
    num_runs = 1000

    # Create layers
    layer_pytorch = TernaryLinear(in_features, out_features, use_ternary_gemm=False, quantize_weights=True)
    layer_gemm = TernaryLinear(in_features, out_features, use_ternary_gemm=True, quantize_weights=True)

    # Copy weights
    with torch.no_grad():
        layer_gemm.weight.copy_(layer_pytorch.weight)

    # Test input
    x = torch.randn(batch_size, in_features)

    # Warm-up
    for _ in range(10):
        _ = layer_pytorch(x)
        _ = layer_gemm(x)

    # Benchmark PyTorch
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = layer_pytorch(x)
    pytorch_time = (time.perf_counter() - start) * 1000  # Convert to ms

    # Benchmark GEMM
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = layer_gemm(x)
    gemm_time = (time.perf_counter() - start) * 1000  # Convert to ms

    speedup = pytorch_time / gemm_time

    print(f"Configuration: {batch_size}×{in_features}×{out_features}")
    print(f"Number of runs: {num_runs}")
    print(f"\nPyTorch time: {pytorch_time:.2f} ms ({pytorch_time/num_runs:.4f} ms/run)")
    print(f"GEMM time:    {gemm_time:.2f} ms ({gemm_time/num_runs:.4f} ms/run)")
    print(f"\nSpeedup: {speedup:.2f}×")

    if speedup > 1.0:
        print(f"✅ GEMM is {speedup:.2f}× faster than PyTorch")
    elif speedup > 0.5:
        print(f"⚠️  GEMM is {speedup:.2f}× (slower due to overhead on small matrices)")
    else:
        print(f"❌ GEMM is significantly slower ({speedup:.2f}×)")

    return True


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "TritNet GEMM Integration Tests" + " " * 22 + "║")
    print("╚" + "═" * 68 + "╝")

    results = []

    # Run tests
    results.append(("Module Availability", test_module_availability()))
    results.append(("Forward Pass Correctness", test_forward_pass_correctness()))
    results.append(("Gradient Flow", test_gradient_flow()))
    results.append(("Performance Comparison", test_performance_comparison()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} | {test_name}")

    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All tests PASSED!")
        return 0
    else:
        print(f"❌ {total - passed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
