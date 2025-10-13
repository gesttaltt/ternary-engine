# AVX-512 Future Support Infrastructure

**Status**: FUTURE USE - Not yet integrated into main codebase

## Purpose

This directory contains infrastructure for future multi-ISA support (AVX-512BW, ARM NEON, ARM SVE). Currently **unused** by the main engine (`ternary_simd_engine.cpp`), which still uses direct AVX2 intrinsics.

## Contents

### `ternary_simd_config.h`
Multi-ISA abstraction layer providing portable SIMD macros:
- **AVX-512BW**: 512-bit vectors (64 trits/op)
- **AVX2**: 256-bit vectors (32 trits/op) - Default
- **ARM NEON**: 128-bit vectors (16 trits/op)
- **Scalar**: Fallback for non-SIMD platforms

**Portable Operations**:
- `TERNARY_VEC` - Vector type
- `TERNARY_LOAD` / `TERNARY_STORE` - Memory operations
- `TERNARY_SHUFFLE` - LUT lookups
- `TERNARY_AND` / `TERNARY_OR` / `TERNARY_ADD` - Arithmetic
- `TERNARY_SET1` - Broadcast
- `TERNARY_PREFETCH` - Cache hints

## Current Status

### ✅ Where It's Used
- **Nowhere** - Completely unused by production code
- Preserved as documentation and reference implementation

### ❌ Where It's NOT Used
- `ternary_simd_engine.cpp` - Uses direct AVX2 intrinsics
- `ternary_c_api.h` - Uses direct AVX2 intrinsics
- `benchmarks/bench_kernels.cpp` - Uses direct AVX2 intrinsics

## Phase 4 Integration Plan

### Step 1: Refactor Main Engine
Replace direct AVX2 intrinsics in `ternary_simd_engine.cpp`:
```cpp
// Before (current):
__m256i va = _mm256_loadu_si256(...);

// After (Phase 4):
TERNARY_VEC va = TERNARY_LOAD(...);
```

### Step 2: Add Runtime Dispatch
Use `ternary_cpu_detect.h` to select optimal ISA at runtime:
```cpp
if (has_avx512bw()) {
    // Compile with -mavx512bw
} else if (has_avx2()) {
    // Compile with -mavx2 (current)
}
```

### Step 3: Benchmark & Validate
- Test on AVX-512 capable hardware (Skylake-X, Ice Lake, Sapphire Rapids)
- Measure 2× expected throughput gain
- Validate correctness across all ISAs

### Step 4: ARM Support
- Test on Apple M1/M2, AWS Graviton
- Validate ARM NEON implementation (128-bit, 16 trits/op)
- Future: ARM SVE support (scalable vectors)

## Why Kept?

1. **Documentation**: Demonstrates portable SIMD patterns
2. **Future-Ready**: Saves months of work when AVX-512/ARM needed
3. **Reference**: Shows correct abstraction layer design
4. **Gradual Migration**: Enables step-by-step refactoring

## Compilation Examples

```bash
# AVX2 (current default)
g++ -O3 -march=native -mavx2 -std=c++17 ...

# AVX-512BW (future)
g++ -O3 -march=skylake-avx512 -mavx512bw -std=c++17 ...

# ARM NEON (future)
aarch64-g++ -O3 -march=native -std=c++17 ...
```

## Related Files

- `../ternary_cpu_detect.h` - Runtime ISA detection
- `../ternary_simd_engine.cpp` - Main engine (to be refactored)
- `../ternary_c_api.h` - C FFI layer (also uses direct AVX2)
- `../benchmarks/bench_kernels.cpp` - Benchmarks (also uses direct AVX2)

---

**Last Updated**: 2025-10-13
**Status**: Isolated, documented, ready for Phase 4 integration
