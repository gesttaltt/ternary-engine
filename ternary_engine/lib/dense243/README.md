# Dense243 Encoding - TritNet-Ready Module

**Status:** ✅ Restored and Production-Ready (All 10/10 tests passing)
**Density:** 5 trits/byte (95.3% state utilization)
**Purpose:** High-density storage with future TritNet neural network integration

---

## Overview

Dense243 encoding packs 5 balanced trits into a single byte using base-243 representation, achieving 95.3% density (243/256 states) compared to standard 2-bit encoding (4 trits/byte, 25% density).

### Performance Validated (2025-10-29)

- **Pack:** 0.25 ns/operation (4 billion ops/sec)
- **Unpack:** 0.91 ns/operation (1.1 billion ops/sec)
- **Tests:** All 243 valid states verified ✅
- **Critical fix:** Variable name bug corrected (o3→o4 multiplication chain)

---

## TritNet Vision

### The Roadmap

**Phase 1 (CURRENT):** LUT-based dense243 operations
- ✅ Pack/unpack between 2-bit and dense243 formats
- ✅ Direct operations on dense243-encoded data
- ✅ Python module with clean API

**Phase 2 (IN PROGRESS):** Train TritNet on truth tables
```
1. Generate complete dense243 truth tables for all operations
2. Train tiny BitNet (bitnet.cpp) on exact arithmetic
3. Distill resulting model to ternary weights {-1, 0, +1}
4. Export as TritNet model compatible with this module
```

**Phase 3 (FUTURE):** Replace LUT with matmul
```python
import ternary_dense243_module as td

# Switch backend from LUT to TritNet
td.set_backend('tritnet')  # Loads trained TritNet weights

# Operations now use tiny NN inference instead of LUT
result = td.tadd(a, b)  # Calls matmul instead of lookup
```

### Why TritNet?

**Advantages:**
- **Learnable operations:** Discover optimal arithmetic beyond hand-coded LUTs
- **Compression:** Single weight matrix vs 243-entry tables
- **Generalization:** Potential to handle fuzzy/approximate ternary logic
- **Hardware-friendly:** Matmul accelerators (TPU, GPU) instead of memory lookups

**Architecture:**
```
Input: 5 trits (dense243 byte)
  ↓
Unpack to 5 ternary values {-1, 0, +1}
  ↓
TritNet Layer 1: [5 → 8] ternary weights
  ↓
TritNet Layer 2: [8 → 5] ternary weights
  ↓
Pack to dense243 byte
Output: Result byte
```

**Training:**
1. Generate all 243² pairs for binary ops (tadd, tmul, tmin, tmax)
2. Generate all 243 values for unary ops (tnot)
3. Train BitNet b1.58 model to 100% accuracy
4. Distill to pure ternary weights using BitNet pipeline
5. Export as `.tritnet` model file

---

## Files

### Core Implementation
- **ternary_dense243.h** - Scalar pack/unpack functions
- **ternary_dense243_simd.h** - SIMD extraction kernels (5× 256-entry LUTs)
- **ternary_triadsextet.h** - Bonus: 3 trits/6-bit encoding (42% density)

### Python Module
- **ternary_dense243_module.cpp** - Python bindings with TritNet hooks
- **build script:** `scripts/build/build_dense243.py`

### Tests
- **tests/test_dense243.cpp** - Comprehensive C++ test suite (10/10 passing)

### Documentation
- **docs/t5-dense243-spec.md** - Complete technical specification
- **This file** - TritNet roadmap and usage

---

## Build & Usage

### Build the Module

```bash
python scripts/build/build_dense243.py
```

### Python Usage

```python
import numpy as np
import ternary_dense243_module as td
import ternary_simd_engine as tc  # Standard 2-bit module

# Convert 2-bit trits to dense243 (20% space savings)
trits_2bit = np.array([0b00, 0b01, 0b10, 0b00, 0b10], dtype=np.uint8)
packed = td.pack(trits_2bit)  # 5 trits → 1 byte
print(f"Packed: {packed[0]} (vs 5 bytes in 2-bit)")

# Unpack back to 2-bit
unpacked = td.unpack(packed, num_trits=5)
assert np.array_equal(unpacked, trits_2bit)

# Direct operations on dense243 format (slower, but saves memory traffic)
a_dense = td.pack(np.array([0b10, 0b01, 0b00, 0b10, 0b01], dtype=np.uint8))
b_dense = td.pack(np.array([0b01, 0b10, 0b10, 0b00, 0b01], dtype=np.uint8))
result_dense = td.tadd(a_dense, b_dense)  # Operates on dense243 format

# Hybrid workflow: Store in dense243, compute in 2-bit
def save_dataset(data_2bit, filename):
    packed = td.pack(data_2bit)
    np.save(filename, packed)  # 20% smaller files

def load_and_compute(filename):
    packed = np.load(filename)
    data_2bit = td.unpack(packed)
    result = tc.tadd(data_2bit, other)  # Use fast 2-bit engine
    return td.pack(result)  # Store back in dense243
```

---

## TritNet Integration Checklist

- [ ] Generate truth table datasets (243² pairs per operation)
- [ ] Train BitNet model on exact arithmetic
- [ ] Validate 100% accuracy on all operations
- [ ] Distill to ternary weights using BitNet pipeline
- [ ] Export model as `.tritnet` format
- [ ] Implement TritNet inference backend in C++
- [ ] Add `set_backend('tritnet')` functionality
- [ ] Benchmark TritNet vs LUT performance
- [ ] Write TritNet integration guide

---

## Use Cases

### ✅ Good for Dense243
- **Persistent storage:** Save 20% disk space on large ternary datasets
- **Network transmission:** 20% bandwidth reduction
- **Archive/cold storage:** Maximize density for infrequently accessed data
- **Hybrid workflows:** Store dense243, transcode to 2-bit for compute

### ❌ Not ideal for Dense243
- **Hot compute paths:** 16× overhead vs 2-bit operations
- **Small arrays:** <100K elements (overhead dominates)
- **Cache-resident workloads:** Benefits masked by L3 cache

---

## References

- **BitNet:** https://github.com/microsoft/BitNet
- **bitnet.cpp:** https://github.com/microsoft/BitBLAS
- **Dense243 spec:** `docs/t5-dense243-spec.md`
- **Test results:** Commit 55fabfd (2025-10-29)

---

**Status:** Ready for TritNet training pipeline
**Next Step:** Generate truth table datasets and train initial TritNet model
