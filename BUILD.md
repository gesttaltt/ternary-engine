# Build Instructions - Ternary Core

## Quick Start (Linux/macOS)

```bash
c++ -O3 -march=native -mavx2 \
    -flto \
    -ffast-math \
    -funroll-loops \
    -finline-functions \
    -fomit-frame-pointer \
    -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    ternary_core_simd_full.cpp \
    -o ternary_core_simd_full$(python3-config --extension-suffix)
```

## Quick Start (Windows/MSVC)

```batch
cl /O2 /GL /arch:AVX2 /std:c++17 /LD /EHsc ^
   /I"C:\path\to\pybind11\include" ^
   /I"C:\path\to\python\include" ^
   ternary_core_simd_full.cpp ^
   /link /LTCG ^
   /LIBPATH:"C:\path\to\python\libs" ^
   /OUT:ternary_core_simd_full.pyd
```

---

## Compiler Flag Reference (OPT-111-114)

### GCC/Clang Flags

| Flag | Purpose | Impact |
|------|---------|--------|
| `-O3` | Maximum optimization level | 20-40% speedup |
| `-march=native` | Use all available CPU instructions | 10-30% speedup |
| `-mavx2` | Enable AVX2 SIMD (required) | Essential |
| `-flto` | Link-time optimization | 5-15% speedup |
| `-ffast-math` | Aggressive math optimizations | Safe for integer ops |
| `-funroll-loops` | Automatic loop unrolling | 5-10% speedup |
| `-finline-functions` | Aggressive function inlining | Reduces call overhead |
| `-fomit-frame-pointer` | Free up one register | Minor speedup |

### MSVC Flags

| Flag | Purpose | GCC Equivalent |
|------|---------|----------------|
| `/O2` | Maximize speed | `-O3` |
| `/GL` | Whole program optimization | `-flto` |
| `/arch:AVX2` | Enable AVX2 | `-mavx2` |
| `/LTCG` | Link-time code generation | `-flto` (link stage) |

---

## Platform-Specific Builds

### Linux (GCC)

```bash
# Debug build
g++ -g -O0 -march=native -mavx2 \
    -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    ternary_core_simd_full.cpp \
    -o ternary_core_simd_full.so

# Release build (optimized)
g++ -O3 -march=native -mavx2 -DNDEBUG \
    -flto -ffast-math -funroll-loops -finline-functions -fomit-frame-pointer \
    -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    ternary_core_simd_full.cpp \
    -o ternary_core_simd_full.so
```

### macOS (Clang)

```bash
# Release build
clang++ -O3 -march=native -mavx2 \
    -flto -ffast-math -funroll-loops -finline-functions -fomit-frame-pointer \
    -shared -std=c++17 -undefined dynamic_lookup \
    $(python3 -m pybind11 --includes) \
    ternary_core_simd_full.cpp \
    -o ternary_core_simd_full$(python3-config --extension-suffix)
```

### Windows (MSVC)

**Prerequisites**:
1. Install Visual Studio 2019 or later
2. Install Python 3.7+
3. Install pybind11: `pip install pybind11`

**Build steps**:
```batch
REM Open "x64 Native Tools Command Prompt for VS 2019"

REM Set paths
set PYTHON_ROOT=C:\Python39
set PYBIND11_ROOT=C:\path\to\pybind11

REM Release build
cl /O2 /GL /arch:AVX2 /std:c++17 /LD /EHsc /DNDEBUG ^
   /I"%PYBIND11_ROOT%\include" ^
   /I"%PYTHON_ROOT%\include" ^
   ternary_core_simd_full.cpp ^
   /link /LTCG ^
   /LIBPATH:"%PYTHON_ROOT%\libs" ^
   /OUT:ternary_core_simd_full.pyd
```

---

## Build Verification

### Test the compiled module:

```python
import numpy as np
import ternary_core_simd_full as tc

# Test basic operations
A = np.array([0b00, 0b01, 0b10], dtype=np.uint8)  # [-1, 0, +1]
B = np.array([0b10, 0b01, 0b00], dtype=np.uint8)  # [+1, 0, -1]

print("tadd:", tc.tadd(A, B))  # Expected: [0b01, 0b01, 0b01] = [0, 0, 0]
print("tmul:", tc.tmul(A, B))  # Expected: [0b00, 0b01, 0b00] = [-1, 0, -1]
print("tmin:", tc.tmin(A, B))  # Expected: [0b00, 0b01, 0b00] = [-1, 0, -1]
print("tmax:", tc.tmax(A, B))  # Expected: [0b10, 0b01, 0b10] = [+1, 0, +1]
print("tnot:", tc.tnot(A))     # Expected: [0b10, 0b01, 0b00] = [+1, 0, -1]
```

Expected output:
```
tadd: [1 1 1]
tmul: [0 1 0]
tmin: [0 1 0]
tmax: [2 1 2]
tnot: [2 1 0]
```

---

## Performance Benchmarking

### Simple benchmark:

```python
import numpy as np
import ternary_core_simd_full as tc
import time

# Create large arrays
n = 10_000_000
A = np.random.choice([0b00, 0b01, 0b10], n, dtype=np.uint8)
B = np.random.choice([0b00, 0b01, 0b10], n, dtype=np.uint8)

# Benchmark tadd
start = time.perf_counter()
C = tc.tadd(A, B)
elapsed = time.perf_counter() - start

trits_per_sec = n / elapsed
print(f"Throughput: {trits_per_sec / 1e6:.1f} M trits/s")
print(f"Time: {elapsed * 1000:.2f} ms for {n / 1e6:.1f}M trits")
```

### Expected performance (Phase 0 optimizations):
- **Pre-optimization**: ~30-50 M trits/s
- **Post-optimization**: ~40-75 M trits/s (30-50% improvement)
- **Scalar path**: 3-10× faster

---

## Troubleshooting

### Error: "Illegal instruction"
**Cause**: CPU doesn't support AVX2
**Solution**: Use a fallback build or upgrade CPU

### Error: "undefined symbol: _Py_..."
**Cause**: Python libraries not linked correctly
**Solution**: Add Python lib path: `-L$(python3-config --prefix)/lib -lpython3.x`

### Error: "pybind11/pybind11.h: No such file"
**Cause**: pybind11 not installed
**Solution**: `pip install pybind11`

### Warning: "unrecognized command line option"
**Cause**: Older compiler version
**Solution**: Remove unsupported flags one by one

---

## Advanced Options

### Profile-Guided Optimization (PGO)

**Step 1**: Build with profiling instrumentation
```bash
g++ -O3 -march=native -mavx2 -fprofile-generate \
    -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    ternary_core_simd_full.cpp \
    -o ternary_core_simd_full.so
```

**Step 2**: Run representative workload
```python
import numpy as np
import ternary_core_simd_full as tc

# Run typical operations
for _ in range(100):
    A = np.random.choice([0b00, 0b01, 0b10], 1000000, dtype=np.uint8)
    B = np.random.choice([0b00, 0b01, 0b10], 1000000, dtype=np.uint8)
    tc.tadd(A, B)
    tc.tmul(A, B)
    # ... more operations
```

**Step 3**: Build with profile data
```bash
g++ -O3 -march=native -mavx2 -fprofile-use \
    -flto -ffast-math -funroll-loops \
    -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    ternary_core_simd_full.cpp \
    -o ternary_core_simd_full.so
```

**Expected gain**: Additional 5-10% over standard optimizations

---

## Build System (Future)

A proper build system (CMake/setup.py) will be added in a future update to automate this process. For now, use the manual compilation commands above.

---

**Document Version**: 1.0
**Last Updated**: 2025-10-11
**Optimization Phase**: Phase 0 (Quick Wins)
