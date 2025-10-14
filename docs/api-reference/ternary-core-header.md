# ternary_algebra.h - Core Algebra Header Documentation

## Overview

`ternary_algebra.h` is the foundational header file that defines the balanced ternary logic system and provides optimized scalar operations with compile-time generated lookup tables. It serves as the semantic core for all ternary operations in the library, establishing the encoding scheme, constexpr-generated LUTs, and basic operations that are used by both scalar and SIMD implementations.

**File**: `ternary_algebra.h` (108 lines)
**Purpose**: Core definitions, constexpr LUT generation, LUT-based scalar operations, encoding/conversion utilities
**Dependencies**: `<stdint.h>`, `ternary_lut_gen.h`
**License**: Apache 2.0

---

## Balanced Ternary System

### Encoding Scheme

The library uses a 2-bit encoding for ternary values (trits):

```
Value  | Binary | Hex
-------|--------|-----
  -1   | 0b00   | 0x00
   0   | 0b01   | 0x01
  +1   | 0b10   | 0x02
Invalid| 0b11   | 0x03
```

**Design Rationale**:
- Each trit occupies 2 bits (can store 4 states, 3 used + 1 reserved)
- Natural ordering: `-1 < 0 < +1` corresponds to `0b00 < 0b01 < 0b10`
- Efficient LUT indexing via bit manipulation
- Compatible with byte-aligned operations (4 trits pack into 1 byte)

### Type Definition

```c
typedef uint8_t trit;
```

Uses `uint8_t` for:
- Natural alignment with byte operations
- Direct compatibility with NumPy uint8 arrays
- Efficient SIMD processing (32 trits per 256-bit AVX2 register)

---

## Lookup Tables (LUTs)

The core optimization strategy combines two approaches:
- **OPT-086**: Replace arithmetic operations with direct table lookups
- **OPT-AUTO-LUT**: Generate LUTs at compile time via constexpr (reduced maintenance overhead)

All LUTs are generated using `constexpr` functions from `ternary_lut_gen.h`, ensuring:
- Algorithm is the documentation (centralized definition)
- Zero runtime cost (evaluated at compile time)
- Auditability (algebraic rules visible in code)
- Flexibility (new operations via lambda expressions)

### Constexpr LUT Generation Framework

**File**: `ternary_lut_gen.h`

**Key Functions**:
```cpp
template <typename Func>
constexpr std::array<uint8_t, 16> make_binary_lut(Func op);

template <typename Func>
constexpr std::array<uint8_t, 4> make_unary_lut(Func op);
```

**Example Generation**:
```cpp
constexpr auto TADD_LUT = make_binary_lut([](uint8_t a, uint8_t b) -> uint8_t {
    int sa = trit_to_int_constexpr(a);
    int sb = trit_to_int_constexpr(b);
    int sum = sa + sb;
    return int_to_trit_constexpr(clamp_ternary(sum));  // Saturated addition
});
```

### Binary Operation LUTs

Binary operations use 16-entry tables indexed by `(a << 2) | b`, generated via `make_binary_lut()`:

#### TADD_LUT - Saturating Ternary Addition

**Algorithm**: `clamp(a + b, -1, +1)`

```cpp
constexpr auto TADD_LUT = make_binary_lut([](uint8_t a, uint8_t b) -> uint8_t {
    int sa = trit_to_int_constexpr(a);
    int sb = trit_to_int_constexpr(b);
    int sum = sa + sb;
    return int_to_trit_constexpr(clamp_ternary(sum));
});
```

**Generated Values** (for reference):
```
Index: 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
Value: 00 00 01 00 00 01 10 00 01 10 10 00 00 00 00 00
```

**Saturation Behavior**:
- `-1 + (-1) = -1` (not -2, saturates at lower bound)
- `+1 + (+1) = +1` (not +2, saturates at upper bound)
- Preserves ternary domain constraints

#### TMUL_LUT - Ternary Multiplication

**Algorithm**: `a * b`

```cpp
constexpr auto TMUL_LUT = make_binary_lut([](uint8_t a, uint8_t b) -> uint8_t {
    int sa = trit_to_int_constexpr(a);
    int sb = trit_to_int_constexpr(b);
    int product = sa * sb;
    return int_to_trit_constexpr(product);
});
```

**Generated Values** (for reference):
```
Index: 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
Value: 10 01 00 00 01 01 01 00 00 01 10 00 00 00 00 00
       a=-1: (-1)*(-1)=+1, (-1)*0=0, (-1)*(+1)=-1
       a=0:  0*(-1)=0, 0*0=0, 0*(+1)=0 (absorbing)
       a=+1: (+1)*(-1)=-1, (+1)*0=0, (+1)*(+1)=+1
```

**Properties**:
- Sign-based multiplication: `sign(a*b) = sign(a) * sign(b)`
- Zero absorption: `0 * x = 0` for all x
- Commutative: `a * b = b * a`

#### TMIN_LUT - Ternary Minimum

**Algorithm**: `min(a, b)` where `-1 < 0 < +1`

```cpp
constexpr auto TMIN_LUT = make_binary_lut([](uint8_t a, uint8_t b) -> uint8_t {
    int sa = trit_to_int_constexpr(a);
    int sb = trit_to_int_constexpr(b);
    int minimum = (sa < sb) ? sa : sb;
    return int_to_trit_constexpr(minimum);
});
```

**Generated Values** (for reference):
```
Index: 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
Value: 00 00 00 00 00 01 01 00 00 01 10 00 00 00 00 00
       a=-1: min(-1, x) = -1 for all x
       a=0:  min(0, -1)=-1, min(0,0)=0, min(0,+1)=0
       a=+1: min(+1,-1)=-1, min(+1,0)=0, min(+1,+1)=+1
```

**Order**: `-1 < 0 < +1`

#### TMAX_LUT - Ternary Maximum

**Algorithm**: `max(a, b)` where `-1 < 0 < +1`

```cpp
constexpr auto TMAX_LUT = make_binary_lut([](uint8_t a, uint8_t b) -> uint8_t {
    int sa = trit_to_int_constexpr(a);
    int sb = trit_to_int_constexpr(b);
    int maximum = (sa > sb) ? sa : sb;
    return int_to_trit_constexpr(maximum);
});
```

**Generated Values** (for reference):
```
Index: 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
Value: 00 01 10 00 01 01 10 00 10 10 10 00 00 00 00 00
       a=-1: max(-1,-1)=-1, max(-1,0)=0, max(-1,+1)=+1
       a=0:  max(0,-1)=0, max(0,0)=0, max(0,+1)=+1
       a=+1: max(+1, x) = +1 for all x
```

### Unary Operation LUT

#### TNOT_LUT - Ternary Negation

**Algorithm**: `-a`

```cpp
constexpr auto TNOT_LUT = make_unary_lut([](uint8_t a) -> uint8_t {
    int sa = trit_to_int_constexpr(a);
    int negated = -sa;
    return int_to_trit_constexpr(negated);
});
```

**Generated Values** (for reference):
```
Index: 0  1  2  3
Value: 10 01 00 00
       tnot(-1) = +1
       tnot(0)  = 0
       tnot(+1) = -1
       tnot(invalid) = undefined
```

Indexed directly by `a & 0b11` (only 4 entries needed).

**Properties**:
- Arithmetic negation: `tnot(a) = -a`
- Involutive: `tnot(tnot(a)) = a`

---

## Scalar Operations

All operations are force-inlined (OPT-051) for zero-overhead abstraction.

### Platform-Specific Inlining

```c
#ifdef _MSC_VER
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE __attribute__((always_inline)) inline
#endif
```

Ensures aggressive inlining on both MSVC (Windows) and GCC/Clang (Linux/Mac).

### Operation Implementations

#### Binary Operations

```c
static FORCE_INLINE trit tadd(trit a, trit b) {
    return TADD_LUT[(a << 2) | b];
}

static FORCE_INLINE trit tmul(trit a, trit b) {
    return TMUL_LUT[(a << 2) | b];
}

static FORCE_INLINE trit tmin(trit a, trit b) {
    return TMIN_LUT[(a << 2) | b];
}

static FORCE_INLINE trit tmax(trit a, trit b) {
    return TMAX_LUT[(a << 2) | b];
}
```

**Index Calculation**: `(a << 2) | b`
- Shifts `a` left by 2 bits: places it in bits [3:2]
- ORs with `b`: places it in bits [1:0]
- Result: 4-bit index in range [0, 15]

**Performance**:
- Single array access (L1 cache hit: ~1-4 cycles)
- No branches, no arithmetic overflow
- Compiler inlines to direct memory load

#### Unary Operation

```c
static FORCE_INLINE trit tnot(trit a) {
    return TNOT_LUT[a & 0b11];
}
```

**Index Calculation**: `a & 0b11`
- Masks to lower 2 bits (sanitizes invalid input)
- Direct 4-entry table access

---

## Conversion Utilities

### Integer ↔ Trit Conversion

```c
static inline trit int_to_trit(int v) {
    return (v < 0) ? 0b00 : (v > 0) ? 0b10 : 0b01;
}

static inline int trit_to_int(trit t) {
    return (t == 0b00) ? -1 : (t == 0b10) ? 1 : 0;
}
```

**Usage**: Primarily for testing and external interfaces. Internal operations stay in trit domain to avoid conversion overhead.

### Trit Packing

```c
static inline uint8_t pack_trits(trit t0, trit t1, trit t2, trit t3) {
    return (t0) | (t1 << 2) | (t2 << 4) | (t3 << 6);
}

static inline trit unpack_trit(uint8_t pack, int idx) {
    return (pack >> (2 * idx)) & 0b11;
}
```

**Purpose**: Pack 4 trits into a single byte for dense storage.

**Layout**:
```
Byte: [t3:t3][t2:t2][t1:t1][t0:t0]
      bits 7:6  5:4  3:2  1:0
```

**Note**: Currently unused in main implementation (operates on unpacked arrays), but available for future optimizations.

---

## Design Evolution

### Pre-Phase 0: Conversion-Based Approach

**Old approach** (see `benchmarks/reference_cpp.cpp`):
```c
trit old_tadd(trit a, trit b) {
    int ai = trit_to_int(a);  // Conversion overhead
    int bi = trit_to_int(b);  // Conversion overhead
    int sum = ai + bi;
    if (sum > 1) sum = 1;      // Saturation branch
    if (sum < -1) sum = -1;    // Saturation branch
    return int_to_trit(sum);   // Conversion overhead
}
```

**Problems**:
- Multiple conversions per operation (6-10 instructions)
- Unpredictable branches (pipeline stalls)
- Integer arithmetic overhead

### Phase 0: LUT Optimization (OPT-086)

**Current approach**:
```c
static FORCE_INLINE trit tadd(trit a, trit b) {
    return TADD_LUT[(a << 2) | b];  // ~1-4 cycles total
}
```

**Benefits**:
- Single memory load (~1-4 cycles on L1 hit)
- Branch-free execution
- No arithmetic operations
- Same semantic domain (trit→trit, no conversions)

**Measured Improvement**: 3-10x faster than conversion-based approach

---

## Integration with SIMD Implementation

The header provides the **scalar fallback** and **semantic reference** for SIMD operations in `ternary_simd_engine.cpp`:

1. **Scalar Tail Processing**: SIMD loops process 32 elements at a time; remaining elements use these scalar functions
2. **LUT Reuse**: SIMD operations use `_mm256_shuffle_epi8` with the same constexpr-generated LUT tables
3. **Correctness Verification**: SIMD results are validated against scalar reference

### Example Integration

```cpp
// ternary_simd_engine.cpp
#include "ternary_algebra.h"

// SIMD uses broadcast LUTs (constexpr-generated at compile time)
__m256i lut = broadcast_lut_16(TADD_LUT.data());  // From ternary_algebra.h
__m256i result = _mm256_shuffle_epi8(lut, indices);

// Scalar tail uses scalar functions
for (; i < n; ++i) {
    r[i] = tadd(a[i], b[i]);  // From ternary_algebra.h
}
```

---

## Performance Characteristics

### LUT Access Performance

**L1 Cache Hit** (typical case):
- Latency: 1-4 cycles
- Throughput: 2-3 loads/cycle on modern CPUs

**Cache Efficiency**:
- Total LUT size: 16 + 16 + 16 + 16 + 4 = 68 bytes
- Fits in single cache line (64 bytes) with padding
- Always cache-resident (static const, read-only)

### Force Inline Impact

**Without force inline**:
- Function call overhead: ~5-10 cycles
- Register spills/fills
- Prevents further optimizations

**With force inline** (OPT-051):
- Zero call overhead
- Enables constant propagation
- Better register allocation

**Measured Impact**: 5-15% performance improvement for small operations

---

## Compiler Optimization Notes

### Constant Folding

When operands are compile-time constants:
```c
trit result = tadd(0b01, 0b10);  // Compiler knows: a=1, b=2
// Can fold to: trit result = TADD_LUT[6]; → result = 0b10;
```

**Constexpr LUT Benefits**:
- LUTs are evaluated at compile time (zero runtime cost)
- Compiler can inline LUT values for constant operands
- No static initialization overhead

### Auto-Vectorization Limitations

**Scalar loops don't auto-vectorize well**:
```c
for (int i = 0; i < n; i++) {
    out[i] = tadd(a[i], b[i]);
}
```

Compilers struggle because:
- Array indexing patterns are complex for ternary encoding
- LUT dependencies prevent vectorization analysis

**Solution**: Explicit SIMD implementation in `ternary_simd_engine.cpp`

---

## Thread Safety

**All operations are thread-safe**:
- LUTs are `static const` (read-only)
- Functions are stateless
- No global mutable state

Safe for:
- OpenMP parallel loops
- Multi-threaded applications
- Concurrent reads from multiple threads

---

## Future Considerations

### Potential Enhancements

1. **Extended Operations**:
   - Ternary comparison: `tcmp(a, b)` → {-1, 0, +1}
   - Ternary absolute value: `tabs(a)` → {0, +1}
   - Ternary sign: `tsign(a)` → {-1, 0, +1}

2. **Packed Representations**:
   - Enable 4-trits-per-byte operations for memory-intensive workloads
   - Requires new SIMD packing/unpacking strategies

3. **Extended Precision**:
   - Multi-byte trit representations for overflow detection
   - Carry propagation for non-saturating addition

### Stability Guarantee

The current encoding (`-1=0b00, 0=0b01, +1=0b10`) and LUT values are **stable** and should not change without major version bump, as they define the core semantics of the library.

---

## Cross-Reference

- **SIMD Implementation**: See `docs/ternary-engine-simd.md`
- **Optimization History**: See `docs/optimization-complexity-rationale.md`
- **Architecture Overview**: See `docs/architecture.md`
- **Reference Implementations**: See `benchmarks/reference.py` and `benchmarks/reference_cpp.cpp`

---

## Summary

`ternary_algebra.h` provides:
- Efficient 2-bit encoding for balanced ternary (-1, 0, +1)
- Constexpr-generated LUT-based scalar operations (3-10x faster than arithmetic, OPT-AUTO-LUT)
- Compile-time generation approach (reduced maintenance overhead)
- Platform-agnostic force-inline macros
- Thread-safe, cache-friendly design
- Foundation for SIMD acceleration

This header represents the **Phase 0 optimization milestone**: replacing conversion-based operations with direct lookup tables, establishing the semantic core that all higher-level implementations build upon. The constexpr LUT generation framework ensures LUTs are generated at compile time from algebraic rules, eliminating manual maintenance while achieving zero runtime cost.
