# Disabled Optimizations Audit

**Doc-Type:** Technical Audit · Version 1.0 · Generated 2025-12-09

This document provides a detailed analysis of performance optimizations that have been implemented but are currently disabled or incomplete in the Ternary Engine codebase.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Dual-Shuffle XOR Optimization](#1-dual-shuffle-xor-optimization)
3. [256-Byte LUT Expansion](#2-256-byte-lut-expansion)
4. [Profile-Guided Optimization (PGO)](#3-profile-guided-optimization-pgo)
5. [AVX2 Packing Operations](#4-avx2-packing-operations)
6. [Streaming Store Optimization](#5-streaming-store-optimization)
7. [Remediation Priority Matrix](#remediation-priority-matrix)

---

## Executive Summary

| Optimization | Claimed Improvement | Lines of Code | Status | Effort to Enable |
|--------------|---------------------|---------------|--------|------------------|
| Dual-Shuffle XOR | 1.5-1.7× (AMD), 1.2-1.5× (Intel) | 400+ | Commented out | 5 minutes |
| 256-Byte LUT | 10-20% | 300+ | Placeholder | 4-8 hours |
| PGO Builds | 5-15% | 400+ | Never run | 30 minutes |
| AVX2 Packing | Unknown | ~100 | Stubbed | 2-4 hours |
| Streaming Stores | ~10% for large arrays | Implemented | Unvalidated | 1 hour |

**Total Potential Improvement:** 30-60% performance gain (cumulative, not additive)

---

## 1. Dual-Shuffle XOR Optimization

### Overview

The Dual-Shuffle XOR optimization is a microarchitectural optimization that exploits instruction-level parallelism on modern x86 CPUs. It replaces the traditional single-shuffle lookup with two parallel shuffles combined via XOR.

### Location

- **Header:** `src/core/simd/opt_dual_shuffle_xor.h`
- **Integration Point:** `src/core/simd/backend_avx2_v2_optimized.cpp:61`
- **Test File:** `tests/python/test_dual_shuffle_validation.py`

### Technical Details

#### Traditional Approach (Current)

```cpp
// Single shuffle with dependent index calculation
// Bottleneck: Shuffle port saturation (1 shuffle/cycle)

// 1. Compute index (dependent arithmetic - blocks pipeline)
idx = (a << 2) | b;  // Requires a to be ready, then shift, then OR

// 2. Lookup (waits for index computation)
result = _mm256_shuffle_epi8(LUT, idx);
```

**Pipeline Analysis:**
- Index calculation: 2-3 cycles (shift + OR)
- Shuffle: 1 cycle latency, but limited to Port 5 (Intel) / Port 3 (AMD)
- Total: 3-4 cycles per operation, shuffle port is bottleneck

#### Dual-Shuffle XOR Approach (Disabled)

```cpp
// Two parallel shuffles with XOR combination
// Exploits different execution ports

// 1. Parallel shuffles (no data dependency between them)
lo = _mm256_shuffle_epi8(LUT_A, a);  // Port 5 (Intel) / Port 3 (AMD)
hi = _mm256_shuffle_epi8(LUT_B, b);  // Can execute in parallel

// 2. Combine via XOR (runs on ALU port, different from shuffle)
result = _mm256_xor_si256(lo, hi);   // Port 0 (Intel/AMD)
```

**Pipeline Analysis:**
- Two shuffles: Can issue in same cycle on different ports
- XOR: 1 cycle, different execution port
- Total: 2 cycles per operation (vs 3-4 cycles)

### Mathematical Foundation

The optimization works because ternary operations can be decomposed as:

```
LUT(a, b) = LUT_A(a) XOR LUT_B(b)
```

This property holds when the operation forms a group structure where XOR acts as the group operation in the byte-encoded domain.

**Operations that are XOR-decomposable:**
- `tnot`: Trivially decomposable (unary operation)
- `tadd`: Partially decomposable with saturation handling
- `tmul`: Decomposable with careful encoding

**Operations requiring different approach:**
- `tmax`, `tmin`: Not pure XOR, but can use ADD instead

### Implementation Status

The header file contains:

```cpp
// opt_dual_shuffle_xor.h - 400+ lines

// LUT generation functions
static inline void generate_tnot_dual_luts(uint8_t* lut_a, uint8_t* lut_b);
static inline void generate_tadd_dual_luts(uint8_t* lut_a, uint8_t* lut_b);
static inline void generate_tmul_dual_luts(uint8_t* lut_a, uint8_t* lut_b);

// Dual-shuffle operation implementations
static inline __m256i tnot_dual_shuffle(__m256i a, __m256i lut_a);
static inline __m256i tadd_dual_shuffle(__m256i a, __m256i b,
                                        __m256i lut_a, __m256i lut_b);
static inline __m256i tmul_dual_shuffle(__m256i a, __m256i b,
                                        __m256i lut_a, __m256i lut_b);

// Initialization function
void init_dual_shuffle_luts();
```

### Why It's Disabled

In `backend_avx2_v2_optimized.cpp:61`:

```cpp
void avx2_v2_init() {
    // Initialize canonical LUTs
    init_canonical_luts();

    // init_dual_shuffle_luts();  // TODO: Enable for additional performance

    g_avx2_v2_initialized = true;
}
```

**Probable Reasons:**
1. Requires additional memory for dual LUTs (2× LUT storage)
2. May have been deferred pending benchmarking
3. Correctness validation may not have been completed
4. "TODO" suggests it was planned but never prioritized

### Performance Claims

From the header file comments:

```cpp
/*
 * Performance:
 * - AMD Zen2/3/4: 1.5-1.7× speedup
 * - Intel Alder Lake: 1.2-1.5× speedup
 * - Expected sustained: 35-45 Gops/s (up from 28-35 Gops/s)
 *
 * Microarchitecture:
 * - Intel: shuffle (Port 5) || XOR (Port 0) → parallel execution
 * - AMD: shuffle (Port 3) || XOR (Port 0) → parallel execution
 */
```

### Enabling the Optimization

#### Step 1: Uncomment Initialization

```cpp
// In backend_avx2_v2_optimized.cpp:61
void avx2_v2_init() {
    init_canonical_luts();
    init_dual_shuffle_luts();  // Enable dual-shuffle LUTs
    g_avx2_v2_initialized = true;
}
```

#### Step 2: Switch Operation Implementations

The backend needs to call dual-shuffle versions instead of standard shuffle:

```cpp
// Replace standard shuffle calls with dual-shuffle versions
// This may require modifying the operation dispatch tables
```

#### Step 3: Benchmark

```bash
# Run before/after comparison
python benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py

# Run validation tests
python tests/python/test_dual_shuffle_validation.py
```

### Risks and Considerations

1. **Memory Overhead:** Requires 2× LUT storage (minor impact)
2. **Correctness:** Must validate all operations produce identical results
3. **Edge Cases:** Verify behavior for all trit value combinations
4. **Platform Variance:** Different gains on Intel vs AMD vs older CPUs

### Recommendation

**Priority: HIGH**

This is 400+ lines of production-ready code with clear performance claims. The implementation appears complete with test infrastructure. Enabling requires:

1. Uncomment one line of code
2. Run validation tests
3. Benchmark on target hardware
4. If gains confirmed, enable by default

**Estimated Effort:** 30 minutes to validate, 5 minutes to enable

---

## 2. 256-Byte LUT Expansion

### Overview

This optimization expands lookup tables from 16 bytes (shuffle limit) to 256 bytes, covering all possible 2-trit input combinations without index manipulation.

### Location

- **Header:** `src/core/simd/opt_lut_256byte_expanded.h`
- **Lines:** ~300 (mostly documentation and stubs)

### Technical Details

#### Current Limitation

AVX2 `_mm256_shuffle_epi8` can only index into 16 bytes per lane:

```cpp
// Current: Index must be 0-15 per lane
// For 2 trits (4 values each), we need 4×4 = 16 entries ✓
// But this requires index calculation: idx = (a << 2) | b
```

#### Proposed Solution

```cpp
// 256-byte LUT: Direct indexing without calculation
// Each byte combination maps directly to result
// No shift/OR required

// Problem: shuffle only reads 16 bytes
// Solution: Split into 16 chunks, use multiple shuffles
```

### Current Status

From the header:

```cpp
// Line 268
// TODO: Full implementation requires splitting 256-byte LUT into chunks
// and using multiple shuffles with index calculation
```

**Status:** Design documented, implementation incomplete

### Implementation Approach

```cpp
/**
 * 256-byte LUT split into 16 chunks of 16 bytes each
 *
 * For input byte 'idx' (0-255):
 *   chunk = idx >> 4;     // Which 16-byte segment (0-15)
 *   offset = idx & 0x0F;  // Offset within segment (0-15)
 *
 * Implementation requires:
 * 1. Broadcast chunk selector
 * 2. Load all 16 chunks
 * 3. Select correct chunk via comparison + blend
 * 4. Shuffle within selected chunk
 */
```

### Why It's Incomplete

The optimization requires either:

1. **Multiple Shuffles:** 16 shuffles per operation (likely slower)
2. **AVX-512 VPERMI2B:** Single instruction, but requires AVX-512 (not available on all CPUs)
3. **Table Lookup with PSHUFB:** Complex multi-step approach

The implementation was likely deferred because:
- Complexity outweighs potential benefit
- AVX-512 version would be simpler but limits compatibility
- Current canonical indexing already optimized

### Claimed Performance

```cpp
// From header comments:
// Expected: 10-20% improvement over canonical indexing
// Reality: Unvalidated, may be optimistic
```

### Recommendation

**Priority: LOW**

The complexity is high and the benefit is uncertain. Consider:

1. **Option A:** Implement AVX-512 version only (simpler, limited hardware)
2. **Option B:** Remove from roadmap and claims
3. **Option C:** Defer indefinitely with clear documentation

**Estimated Effort:** 4-8 hours for multi-shuffle approach, 2 hours for AVX-512 only

---

## 3. Profile-Guided Optimization (PGO)

### Overview

PGO is a compiler optimization technique that uses runtime profiling data to make better optimization decisions. The Ternary Engine has a complete 3-phase PGO build system that has never been executed.

### Location

- **Build Script:** `build/build_pgo.py` (400+ lines)
- **Unified Script:** `build/build_pgo_unified.py` (Clang support)
- **Artifacts:** `build/artifacts/pgo/` (empty)

### Technical Details

#### How PGO Works

```
Phase 1: Instrumentation Build
┌─────────────────────────────────────────────────┐
│ Compiler adds profiling instrumentation to code │
│ - Function call counts                          │
│ - Branch taken/not-taken frequencies            │
│ - Loop iteration counts                         │
└─────────────────────────────────────────────────┘
                      ↓
Phase 2: Profile Collection
┌─────────────────────────────────────────────────┐
│ Run representative workloads                    │
│ - Benchmarks with typical data sizes            │
│ - Common operation patterns                     │
│ - Real-world usage scenarios                    │
└─────────────────────────────────────────────────┘
                      ↓
Phase 3: Optimized Build
┌─────────────────────────────────────────────────┐
│ Compiler uses profile data for optimization     │
│ - Hot path optimization                         │
│ - Better branch prediction hints                │
│ - Optimal function inlining decisions           │
│ - Cache-friendly code layout                    │
└─────────────────────────────────────────────────┘
```

### Implementation Status

The build script is complete:

```python
# build/build_pgo.py

def phase1_instrument():
    """Build with instrumentation to collect profiling data"""
    # Compiles with /GL /LTCG:PGInstrument (MSVC)
    # or -fprofile-generate (GCC/Clang)
    pass

def phase2_profile():
    """Run profiling workload to collect data"""
    # Executes benchmarks to generate .pgc/.profraw files
    pass

def phase3_optimize():
    """Build with profile data for final optimization"""
    # Compiles with /LTCG:PGOptimize (MSVC)
    # or -fprofile-use (GCC/Clang)
    pass

def full_pgo():
    """Run all three phases automatically"""
    phase1_instrument()
    phase2_profile()
    phase3_optimize()
```

### Why It's Never Run

From documentation:

```python
# Note: OpenMP disabled (documented CI crashes)
# PGO builds were likely deferred due to CI instability
```

Additionally:
- No CI job references PGO
- No PGO artifacts exist in the repository
- README doesn't mention PGO builds

### Expected Performance Gains

```python
# From build_pgo.py comments:
"""
Expected improvements from PGO:
- Hot path optimization: 5-10% improvement
- Branch prediction: 2-5% improvement
- Function inlining: 3-5% improvement
- Overall: 5-15% cumulative improvement
"""
```

### Running PGO Manually

```powershell
# From project root
cd c:\Users\Alejandro\Documents\Ivan\Work\ternary-engine

# Option 1: Run all phases automatically
python build/build_pgo.py full

# Option 2: Run phases individually
python build/build_pgo.py instrument  # Phase 1
python build/build_pgo.py profile     # Phase 2
python build/build_pgo.py optimize    # Phase 3

# Option 3: Clean PGO artifacts
python build/build_pgo.py clean
```

### Expected Output

```
Phase 1: INSTRUMENT
  Building with instrumentation...
  Output: build/artifacts/pgo/instrumented/YYYYMMDD_HHMMSS/

Phase 2: PROFILE
  Running profiling workload...
  Collecting: ternary_simd_engine.pgc (MSVC) or default.profraw (Clang)

Phase 3: OPTIMIZE
  Building with profile data...
  Output: build/artifacts/pgo/optimized/YYYYMMDD_HHMMSS/
  Latest: build/artifacts/pgo/latest/
```

### Integration with CI

To enable PGO in CI:

```yaml
# .github/workflows/pgo-build.yml
name: PGO Build

on:
  release:
    types: [published]
  workflow_dispatch:

jobs:
  pgo-build:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install numpy pybind11

      - name: Run PGO Build
        run: python build/build_pgo.py full

      - name: Upload PGO Artifact
        uses: actions/upload-artifact@v4
        with:
          name: ternary-simd-engine-pgo
          path: build/artifacts/pgo/latest/
```

### Recommendation

**Priority: HIGH**

PGO is a proven optimization technique with measurable benefits. The infrastructure is complete:

1. Run `python build/build_pgo.py full` locally
2. Benchmark against standard build
3. If gains confirmed (>5%), add to release process
4. Document PGO build process in README

**Estimated Effort:** 30 minutes initial run, 1 hour to integrate into CI

---

## 4. AVX2 Packing Operations

### Overview

The packing operations (octet_pack, sixtet_pack) have AVX2 function signatures defined but fall back to scalar implementations.

### Location

- **Octet Pack:** `src/core/packing/octet_pack.h:203-216`
- **Sixtet Pack:** `src/core/packing/sixtet_pack.h:239-251`

### Current Status

```cpp
// octet_pack.h:203-206
static inline void octet_pack_avx2(
    const uint8_t* trits, size_t count, uint8_t* packed
) {
    // TODO: Implement AVX2 version in Phase 6
    octet_pack_scalar(trits, count, packed);  // Falls back to scalar
}

static inline void octet_unpack_avx2(
    const uint8_t* packed, size_t count, uint8_t* trits
) {
    // TODO: Implement AVX2 version in Phase 6
    octet_unpack_scalar(packed, count, trits);  // Falls back to scalar
}
```

### Potential AVX2 Implementation

```cpp
// Octet packing: 8 trits → 1 byte (base-3 encoding)
// 3^8 = 6561 states, fits in 13 bits (uses 16-bit intermediate)

static inline void octet_pack_avx2_impl(
    const uint8_t* trits, size_t count, uint8_t* packed
) {
    const __m256i powers = _mm256_setr_epi16(
        1, 3, 9, 27, 81, 243, 729, 2187,   // 3^0 to 3^7
        1, 3, 9, 27, 81, 243, 729, 2187
    );

    for (size_t i = 0; i + 16 <= count; i += 16) {
        // Load 16 trits (2 octets worth)
        __m256i t = _mm256_loadu_si256((__m256i*)(trits + i));

        // Convert to 16-bit and multiply by powers
        __m256i t16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(t, 0));
        __m256i weighted = _mm256_mullo_epi16(t16, powers);

        // Horizontal sum for each octet
        // ... (requires multiple reduction steps)
    }

    // Handle remainder with scalar
    // ...
}
```

### Complexity Analysis

The AVX2 implementation is non-trivial because:

1. **Non-power-of-2 Base:** Base-3 doesn't map cleanly to binary operations
2. **Horizontal Sum:** Requires multiple shuffle/add operations
3. **Variable-Length:** Packing 8 trits to ~13 bits requires careful handling

### Recommendation

**Priority: LOW**

The scalar implementation is likely adequate for most use cases:

1. Packing is typically done once for storage, not in hot paths
2. The complexity of AVX2 implementation may not yield significant gains
3. Consider benchmarking scalar performance first

**Options:**
- **Implement:** 2-4 hours of development
- **Remove TODOs:** Update to "scalar implementation sufficient"
- **Defer:** Keep TODO but update Phase 6 timeline

---

## 5. Streaming Store Optimization

### Overview

Streaming stores bypass the cache for large arrays, reducing cache pollution. This is implemented but never benchmarked to validate the threshold.

### Location

- **Implementation:** `src/core/simd/backend_avx2_v2_optimized.cpp:113-134`
- **Threshold:** `TERNARY_STREAM_THRESHOLD` (default: 1M elements)

### Current Implementation

```cpp
// backend_avx2_v2_optimized.cpp

#ifndef TERNARY_STREAM_THRESHOLD
#define TERNARY_STREAM_THRESHOLD 1048576  // 1M elements
#endif

void tadd_avx2_v2(const uint8_t* a, const uint8_t* b, uint8_t* r, size_t n) {
    if (n >= TERNARY_STREAM_THRESHOLD) {
        // Use streaming stores for large arrays
        for (size_t i = 0; i + 32 <= n; i += 32) {
            __m256i va = _mm256_loadu_si256((__m256i*)(a + i));
            __m256i vb = _mm256_loadu_si256((__m256i*)(b + i));
            __m256i vr = /* compute result */;
            _mm256_stream_si256((__m256i*)(r + i), vr);  // Streaming store
        }
        _mm_sfence();  // Ensure stores complete
    } else {
        // Use regular stores for smaller arrays
        for (size_t i = 0; i + 32 <= n; i += 32) {
            // ... regular _mm256_storeu_si256 ...
        }
    }
}
```

### Validation Needed

The 1M threshold is a reasonable default but should be validated:

```python
# Benchmark script to validate streaming store threshold
import ternary_simd_engine as engine
import numpy as np
import time

sizes = [100_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]

for size in sizes:
    a = np.random.randint(-1, 2, size=size, dtype=np.int8)
    b = np.random.randint(-1, 2, size=size, dtype=np.int8)

    # Warm-up
    for _ in range(3):
        engine.tadd(a, b)

    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        engine.tadd(a, b)
    elapsed = time.perf_counter() - start

    throughput = (size * 100) / elapsed / 1e9
    print(f"Size: {size:>10,} | Throughput: {throughput:.2f} Gops/s")
```

### Recommendation

**Priority: MEDIUM**

1. Run benchmark to validate 1M threshold
2. Test on different CPU architectures (different cache sizes)
3. Document optimal threshold in configuration

---

## Remediation Priority Matrix

| Optimization | Impact | Effort | Risk | Priority |
|--------------|--------|--------|------|----------|
| Dual-Shuffle XOR | High (1.5×) | Low (30 min) | Low | **P0** |
| PGO Builds | Medium (5-15%) | Low (30 min) | Low | **P0** |
| Streaming Validation | Low (verify) | Low (1 hr) | Low | **P1** |
| AVX2 Packing | Low | Medium (4 hr) | Medium | **P2** |
| 256-Byte LUT | Medium (10-20%) | High (8 hr) | High | **P3** |

### Immediate Actions (This Week)

1. **Enable Dual-Shuffle XOR**
   ```cpp
   // backend_avx2_v2_optimized.cpp:61
   init_dual_shuffle_luts();  // Uncomment this line
   ```

2. **Run PGO Build**
   ```bash
   python build/build_pgo.py full
   ```

3. **Benchmark Both**
   ```bash
   python benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py
   ```

### Short-Term Actions (This Month)

4. Validate streaming store threshold
5. Decide on AVX2 packing (implement or remove TODO)
6. Update documentation with validated performance numbers

### Long-Term Actions (This Quarter)

7. Consider 256-byte LUT for AVX-512 only
8. Integrate PGO into release workflow
9. Profile-guided threshold tuning

---

**Document Version:** 1.0
**Last Updated:** 2025-12-09
**Author:** Claude Code Audit
