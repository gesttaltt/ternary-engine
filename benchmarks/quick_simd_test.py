"""Quick SIMD kernel benchmark for ternary NN operations."""
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import ternary_simd_engine as tc

MINUS_ONE, ZERO, PLUS_ONE = 0b00, 0b01, 0b10

def main():
    print("=" * 60)
    print("TERNARY SIMD vs NUMPY: Neural Network Weight Operations")
    print("=" * 60)

    rng = np.random.default_rng(42)

    for size in [100_000, 1_000_000, 10_000_000]:
        print(f"\n--- {size:,} weights ---")

        # Simulate ternary weights/gradients
        a_int = rng.choice([-1, 0, 1], size=size, p=[0.45, 0.1, 0.45]).astype(np.int8)
        b_int = rng.choice([-1, 0, 1], size=size, p=[0.45, 0.1, 0.45]).astype(np.int8)

        a_trit = np.where(a_int == -1, MINUS_ONE, np.where(a_int == 1, PLUS_ONE, ZERO)).astype(np.uint8)
        b_trit = np.where(b_int == -1, MINUS_ONE, np.where(b_int == 1, PLUS_ONE, ZERO)).astype(np.uint8)

        # Warmup
        for _ in range(3):
            _ = a_int * b_int
            _ = tc.tmul(a_trit, b_trit)

        n_iters = max(5, 10_000_000 // size)

        # NumPy
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = a_int * b_int
        np_time = (time.perf_counter() - start) / n_iters

        # Ternary SIMD
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = tc.tmul(a_trit, b_trit)
        tc_time = (time.perf_counter() - start) / n_iters

        np_mops = size / np_time / 1e6
        tc_mops = size / tc_time / 1e6
        speedup = np_time / tc_time

        print(f"  NumPy:   {np_time*1e6:>8.1f} µs  ({np_mops:>8.1f} Mops/s)")
        print(f"  Ternary: {tc_time*1e6:>8.1f} µs  ({tc_mops:>8.1f} Mops/s)")
        print(f"  Speedup: {speedup:>8.1f}×")

    print("\n" + "=" * 60)
    print("GRADIENT SIGN ACCUMULATION (tadd - saturating addition)")
    print("=" * 60)

    n_params = 10_000_000
    n_batches = 10

    grads_int = [rng.choice([-1, 0, 1], size=n_params, p=[0.4, 0.2, 0.4]).astype(np.int8) for _ in range(n_batches)]
    grads_trit = [np.where(g == -1, MINUS_ONE, np.where(g == 1, PLUS_ONE, ZERO)).astype(np.uint8) for g in grads_int]

    # NumPy accumulation
    start = time.perf_counter()
    acc = np.zeros(n_params, dtype=np.int8)
    for g in grads_int:
        acc = np.clip(acc + g, -1, 1).astype(np.int8)
    np_time = time.perf_counter() - start

    # Ternary accumulation
    start = time.perf_counter()
    acc_t = np.full(n_params, ZERO, dtype=np.uint8)
    for g in grads_trit:
        acc_t = tc.tadd(acc_t, g)
    tc_time = time.perf_counter() - start

    total_ops = n_params * n_batches
    print(f"\n{n_params:,} params × {n_batches} batches = {total_ops:,} ops")
    print(f"  NumPy:   {np_time*1000:>6.1f} ms  ({total_ops/np_time/1e6:>8.1f} Mops/s)")
    print(f"  Ternary: {tc_time*1000:>6.1f} ms  ({total_ops/tc_time/1e6:>8.1f} Mops/s)")
    print(f"  Speedup: {np_time/tc_time:>6.1f}×")

if __name__ == "__main__":
    main()
