# Python Benchmarks (Interpreter Overhead Included)

**Status:** Functional but timing measurements include Python overhead

---

## Important Caveat

These benchmarks **work correctly** for functional validation but their **timing measurements are approximate** due to:

1. **Python interpreter overhead** - Function call dispatch, object allocation
2. **pybind11 marshalling** - NumPy array conversion at FFI boundary
3. **GIL contention** - Global Interpreter Lock serializes some operations
4. **Memory management** - Python's reference counting and garbage collection
5. **time.perf_counter() resolution** - Coarser than std::chrono::steady_clock

### What This Means

| Metric | Reliability | Notes |
|--------|-------------|-------|
| Correctness | High | Results are mathematically correct |
| Relative rankings | Medium | A > B comparisons usually valid |
| Absolute throughput | Low | Mops/s values include 5-30% overhead |
| Latency | Low | Per-op timing distorted by FFI costs |

---

## When to Use These Benchmarks

**Use for:**
- Functional validation (correctness testing)
- Relative comparisons (is optimization A faster than B?)
- Integration testing (end-to-end workflows)
- API regression detection (did we break something?)

**Do NOT use for:**
- Absolute throughput claims (use `cpp-native-kernels/` instead)
- Low-latency validation
- Marketing performance numbers
- Hardware capacity planning

---

## Files

| File | Purpose |
|------|---------|
| `bench_simd_core_ops.py` | Core SIMD operations benchmark |
| `bench_simd_fusion_ops.py` | Fusion operations benchmark |
| `bench_competitive.py` | 6-phase competitive analysis vs NumPy |
| `bench_dense243.py` | Dense243 encoding benchmark |
| `bench_model_quantization.py` | Real model quantization (Phase 5) |
| `bench_power_efficiency.py` | Power consumption analysis |
| `bench_regression_detect.py` | Regression detection suite |
| `bench_input_characteristics.py` | Input pattern analysis |
| `benchmark_framework.py` | Shared framework utilities |
| `build_competitive.py` | Build helper for competitive suite |
| `download_models.py` | Model download utility |
| `run_all_benchmarks.py` | Orchestrator for full suite |
| `run_competitive.bat/.sh` | Shell wrappers |

---

## Validation Against Native Benchmarks

To validate these results, compare with `cpp-native-kernels/`:

```bash
# Run Python benchmark
python bench_simd_core_ops.py

# Compare with native benchmark
cd ../cpp-native-kernels
g++ -O3 -march=native -mavx2 -std=c++17 bench_kernels.cpp -o bench_kernels
./bench_kernels --csv
```

**Expected difference:** Python results should be 5-30% slower than native.

If Python shows **higher** throughput than native, something is wrong (likely measurement error).

---

## Known Overhead Sources

### 1. pybind11 Array Conversion (~5-15%)
```python
# Each call crosses FFI boundary
result = ternary_simd_engine.tadd(a, b)  # NumPy -> C++ -> NumPy
```

### 2. Python Function Call (~1-5%)
```python
# Python dispatch overhead per call
for _ in range(1000):
    result = op(a, b)  # Each call has interpreter overhead
```

### 3. Memory Allocation (~5-10%)
```python
# NumPy allocates output array each call
result = np.empty_like(a)  # Allocation not in C++ benchmark
```

### 4. GIL Release/Acquire (~1-3%)
```cpp
// pybind11 releases GIL, then reacquires
py::gil_scoped_release release;
// ... SIMD work ...
py::gil_scoped_acquire acquire;
```

---

## Recommended Improvements

To make these benchmarks more accurate:

1. **Batch operations** - Amortize FFI overhead over larger arrays
2. **Warmup runs** - Discard first N iterations (JIT, cache warming)
3. **Statistical reporting** - Report min/median/p99, not just mean
4. **Subtract baseline** - Measure empty loop overhead and subtract
5. **Cross-validate** - Compare with native benchmarks

---

## Historical Context

These benchmarks were the original validation suite. The `cpp-native-kernels/` folder was created to provide ground-truth measurements for:

- Production throughput claims (35,042 Mops/s)
- Optimization validation
- Hardware capacity planning

Both benchmark suites should be maintained - Python for integration testing, native for performance claims.

---

**Last reviewed:** 2025-11-27
**Status:** Functional, timing approximate
