# Baseline Snapshot: Pre-AutoLUT Generation

**Date**: 2025-10-12
**Commit**: 9eb708f (Document OPT-HASWELL-02 template-based optional masking in SIMD documentation)
**Purpose**: Establish baseline state before transitioning from manual to constexpr-generated lookup tables

---

## Current Architecture State

### Phase Status
- **Phase 2 (Complexity Compression)**: Complete
  - 3 execution paths (OpenMP/Serial-SIMD/Tail)
  - 73% code reduction from Phase 1
  - Template-based unification achieved

### Recent Optimizations Applied
- **OPT-HASWELL-02**: Template-based optional masking (3-5% gain)
  - `Sanitize=true` (default): Safe production mode with input validation
  - `Sanitize=false` (advanced): Pre-validated pipeline optimization
- **OPT-HASWELL-01**: Shift replacement NOT applied (AVX2 lacks byte-level shifts)
  - Triple-add pattern retained as optimal for AVX2

---

## Manual LUT Implementation (Current)

### Defined Lookup Tables

All LUTs are manually hand-written static const arrays in `ternary_algebra.h`:

#### 1. TADD_LUT (16 entries)
**Operation**: Saturated ternary addition
**Index Format**: `(a << 2) | b`
**Size**: 16 bytes
**Location**: `ternary_algebra.h:40-49`

```cpp
static const uint8_t TADD_LUT[16] = {
    0b00, 0b00, 0b01, 0b00,  // a=-1
    0b00, 0b01, 0b10, 0b00,  // a=0
    0b01, 0b10, 0b10, 0b00,  // a=+1
    0b00, 0b00, 0b00, 0b00   // a=invalid
};
```

#### 2. TMUL_LUT (16 entries)
**Operation**: Ternary multiplication
**Index Format**: `(a << 2) | b`
**Size**: 16 bytes
**Location**: `ternary_algebra.h:52-61`

```cpp
static const uint8_t TMUL_LUT[16] = {
    0b10, 0b01, 0b00, 0b00,  // a=-1
    0b01, 0b01, 0b01, 0b00,  // a=0
    0b00, 0b01, 0b10, 0b00,  // a=+1
    0b00, 0b00, 0b00, 0b00   // a=invalid
};
```

#### 3. TMIN_LUT (16 entries)
**Operation**: Ternary minimum
**Index Format**: `(a << 2) | b`
**Size**: 16 bytes
**Location**: `ternary_algebra.h:64-73`

```cpp
static const uint8_t TMIN_LUT[16] = {
    0b00, 0b00, 0b00, 0b00,  // a=-1
    0b00, 0b01, 0b01, 0b00,  // a=0
    0b00, 0b01, 0b10, 0b00,  // a=+1
    0b00, 0b00, 0b00, 0b00   // a=invalid
};
```

#### 4. TMAX_LUT (16 entries)
**Operation**: Ternary maximum
**Index Format**: `(a << 2) | b`
**Size**: 16 bytes
**Location**: `ternary_algebra.h:76-85`

```cpp
static const uint8_t TMAX_LUT[16] = {
    0b00, 0b01, 0b10, 0b00,  // a=-1
    0b01, 0b01, 0b10, 0b00,  // a=0
    0b10, 0b10, 0b10, 0b00,  // a=+1
    0b00, 0b00, 0b00, 0b00   // a=invalid
};
```

#### 5. TNOT_LUT (4 entries)
**Operation**: Ternary negation
**Index Format**: `a & 0x03`
**Size**: 4 bytes
**Location**: `ternary_algebra.h:88-93`

```cpp
static const uint8_t TNOT_LUT[4] = {
    0b10,  // tnot(-1) = +1
    0b01,  // tnot(0) = 0
    0b00,  // tnot(+1) = -1
    0b00   // tnot(invalid) = undefined
};
```

### Total LUT Memory Footprint
- Binary operations: 4 × 16 = 64 bytes
- Unary operation: 1 × 4 = 4 bytes
- **Total**: 68 bytes

---

## File Structure Snapshot

### Source Files
```
ternary-engine/
├── ternary_algebra.h              # Mathematical rules & manual LUTs (125 lines)
├── ternary_simd_engine.cpp        # Vectorized execution layer (314 lines)
├── build/
│   └── scripts/
│       ├── setup.py               # Standard build
│       ├── setup_pgo.py           # PGO build
│       └── setup_reference.py     # Reference build
├── tests/
│   ├── test_phase0.py
│   ├── test_omp.py
│   └── test_luts.cpp
├── benchmarks/
│   ├── bench_phase0.py
│   ├── bench_fair.py
│   └── reference.py
├── docs/
│   ├── ternary-engine-header.md     # Documents ternary_algebra.h
│   ├── ternary-engine-simd.md       # Documents ternary_simd_engine.cpp
│   └── [other docs...]
└── local-reports/
    ├── optimization.md            # Optimization roadmap
    ├── haswell.md                 # Haswell optimization analysis
    └── plan.md                    # Implementation plan
```

---

## Current Performance Characteristics

### Throughput (Measured)
- **Python reference**: ~100 ME/s (Million Elements/second)
- **C++ LUT scalar**: ~2,000 ME/s (20x vs Python)
- **C++ SIMD**: ~10,000 ME/s (100x vs Python)

### Execution Paths
1. **OpenMP Parallel**: n ≥ 100,000 elements
2. **Serial SIMD**: 32 ≤ n < 100,000 elements
3. **Scalar Tail**: 0-31 remaining elements

### Compatibility
- **Platform**: x86-64 with AVX2 support
- **Minimum CPU**: Intel Haswell (2013+), AMD Excavator (2015+)
- **Tested on**: AMD Ryzen 5 4500 (AVX2-only)

---

## Rationale for Auto-LUT Generation

### Current Pain Points

1. **Manual Maintenance**:
   - All 5 LUTs hand-written and hand-verified
   - Prone to human error (typos, incorrect values)
   - Difficult to audit correctness

2. **Inflexibility**:
   - Adding new operations requires manual LUT construction
   - No systematic way to verify algebraic properties
   - High barrier to experimentation

3. **Documentation Coupling**:
   - LUT values and their derivation are disconnected
   - Comments explain logic but don't enforce correctness

### Proposed Solution: Constexpr Generation

**Goal**: Replace manual LUTs with compile-time generated tables

**Approach**:
```cpp
// Before (manual):
static const uint8_t TADD_LUT[16] = {0b00, 0b00, 0b01, ...};

// After (constexpr):
constexpr auto TADD_LUT = make_binary_lut([](int a, int b) {
    int sum = trit_to_int(a) + trit_to_int(b);
    return int_to_trit(clamp(sum, -1, 1));  // Saturated addition
});
```

**Benefits**:
- **Correctness**: Algorithm is the documentation
- **Maintainability**: Single source of truth
- **Flexibility**: New operations via lambda expressions
- **Zero Runtime Cost**: Computed at compile time
- **Auditability**: Algebraic rules visible in code

### Implementation Plan

1. Create `ternary_lut_gen.h` with constexpr generators
2. Update `ternary_algebra.h` to use generated LUTs
3. Verify binary-identical output via tests
4. Measure compile-time impact
5. Document transition in commit messages

---

## Verification Checklist

Before proceeding with auto-LUT generation:

- [x] All tests passing (`tests/test_phase0.py`, `tests/test_luts.cpp`)
- [x] Benchmarks establish baseline performance
- [x] Documentation current and accurate
- [x] Git history clean with descriptive commits
- [x] Build system stable across platforms
- [x] No known bugs or correctness issues

---

## Commit History Context

Recent commits leading to this baseline:

```
9eb708f - Document OPT-HASWELL-02 template-based optional masking in SIMD documentation
5e6f363 - Implement template-based optional masking optimization (OPT-HASWELL-02)
1dba5bd - Refactor: Rename source files for clarity and intent
3ccaa99 - Pre-Haswell optimization snapshot: Document baseline compatibility
```

---

## Next Steps

1. **Create `ternary_lut_gen.h`**:
   - Define `make_binary_lut<N>()` template
   - Define `make_unary_lut<N>()` template
   - Implement constexpr evaluation framework

2. **Update `ternary_algebra.h`**:
   - Replace manual LUTs with constexpr auto declarations
   - Keep algebraic operations unchanged
   - Maintain API compatibility

3. **Verify Correctness**:
   - Run full test suite
   - Compare generated LUTs against manual versions
   - Ensure binary-identical output

4. **Measure Impact**:
   - Compile-time overhead
   - Binary size comparison
   - Runtime performance (should be identical)

5. **Document Transition**:
   - Update docs to reference constexpr generation
   - Add examples of creating new operations
   - Archive this baseline document

---

**Status**: Production-ready Tier-1 prototype, ready for transition to constexpr LUT generation.

**Recommendation**: Proceed with auto-LUT implementation. This baseline provides clean rollback point and reference for validation.
