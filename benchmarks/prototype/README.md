# Prototype Benchmarks

**Status:** Pending - Requires `ternary_tritnet_gemm` module

These scripts benchmark TritNet GEMM (matrix multiplication) functionality that is still in development.

## Files

| File | Purpose |
|------|---------|
| `bench_gemm.py` | Full TritNet GEMM benchmark suite |
| `bench_gemm_isolated.py` | Isolated component benchmarking for GEMM analysis |

## Dependencies

Requires:
- `ternary_tritnet_gemm` - TritNet GEMM C++ module (not yet built)
- `ternary_dense243_module` - Dense243 packing (optional, provides faster packing)

## Build Instructions

When ready to use:
```bash
python build/build_tritnet_gemm.py
python build/build_dense243.py  # Optional
```

## What These Benchmark

### bench_gemm.py
- Naive vs AVX2-optimized GEMM
- Comparison with NumPy BLAS
- Scaling across matrix sizes (8x16 to 2048x2048)
- Correctness validation

### bench_gemm_isolated.py
- Baseline operations (LUT access)
- Dense243 pack/unpack overhead
- Memory access patterns
- GEMM component breakdown
- Theoretical performance limits

## Target Performance

- Goal: 20-30 Gops/s on large matrices
- Memory bandwidth limited for small matrices
- Should approach NumPy BLAS within 2-5x for large matrices
