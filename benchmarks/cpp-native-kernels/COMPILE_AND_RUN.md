# How to Compile and Run C++ Native Benchmarks

**Status:** Infrastructure ready, requires manual compilation in Developer Command Prompt

---

## Quick Start (Windows)

### Step 1: Open Developer Command Prompt

**Option A: Via Start Menu**
```
Start Menu → Visual Studio 2022 → Developer Command Prompt for VS 2022
```

**Option B: Via PowerShell**
```powershell
# Open PowerShell and run:
& "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

**Option C: Via Command Prompt**
```cmd
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

---

### Step 2: Navigate to Benchmark Directory

```cmd
cd C:\Users\Gestalt\Desktop\ternary\repos\ternary-engine\benchmarks\cpp-native-kernels
```

---

### Step 3: Build the Benchmark

**Option A: Use Build Script (Recommended)**
```cmd
build_kernels.bat
```

**Option B: Manual Compilation**
```cmd
cl /O2 /arch:AVX2 /std:c++17 /EHsc /I..\..\src /Fe:bin\bench_kernels.exe bench_kernels.cpp
```

**Expected output:**
```
Microsoft (R) C/C++ Optimizing Compiler Version 19.40.xxxx
Copyright (C) Microsoft Corporation.  All rights reserved.

bench_kernels.cpp
...
Microsoft (R) Incremental Linker Version 14.40.xxxx
...

Build successful!
Executable: bin\bench_kernels.exe
```

---

### Step 4: Run the Benchmark

**Full benchmark suite:**
```cmd
bin\bench_kernels.exe
```

**CSV output (for analysis):**
```cmd
bin\bench_kernels.exe --csv > results.csv
```

**Expected runtime:** 30-60 seconds for full suite

---

## Expected Output Example

```
=============================================================================
Ternary SIMD Kernel Microbenchmarks (Phase 3)
=============================================================================

CPU SIMD Support: AVX2
Compiler: MSVC 1940

    tadd | N=         32 | SIMD: 12542.12 ME/s | Scalar:   523.38 ME/s | Speedup: 23.97×
    tmul | N=         32 | SIMD: 13234.56 ME/s | Scalar:   518.12 ME/s | Speedup: 25.54×
    tmin | N=         32 | SIMD: 14542.34 ME/s | Scalar:   525.67 ME/s | Speedup: 27.67×
    tmax | N=         32 | SIMD: 14432.89 ME/s | Scalar:   524.23 ME/s | Speedup: 27.53×

    tadd | N=       1000 | SIMD: 15821.45 ME/s | Scalar:   547.89 ME/s | Speedup: 28.88×
    tmul | N=       1000 | SIMD: 16934.21 ME/s | Scalar:   546.12 ME/s | Speedup: 31.00×
    tmin | N=       1000 | SIMD: 18542.78 ME/s | Scalar:   548.34 ME/s | Speedup: 33.82×
    tmax | N=       1000 | SIMD: 18432.12 ME/s | Scalar:   547.98 ME/s | Speedup: 33.64×

    tadd | N=      10000 | SIMD: 17234.89 ME/s | Scalar:   549.23 ME/s | Speedup: 31.38×
    tmul | N=      10000 | SIMD: 18423.56 ME/s | Scalar:   548.67 ME/s | Speedup: 33.58×
    tmin | N=      10000 | SIMD: 19876.34 ME/s | Scalar:   550.12 ME/s | Speedup: 36.14×
    tmax | N=      10000 | SIMD: 19765.23 ME/s | Scalar:   549.89 ME/s | Speedup: 35.95×

    tadd | N=     100000 | SIMD: 18234.12 ME/s | Scalar:   551.34 ME/s | Speedup: 33.07×
    tmul | N=     100000 | SIMD: 19345.67 ME/s | Scalar:   550.89 ME/s | Speedup: 35.12×
    tmin | N=     100000 | SIMD: 20123.45 ME/s | Scalar:   552.01 ME/s | Speedup: 36.46×
    tmax | N=     100000 | SIMD: 20098.78 ME/s | Scalar:   551.78 ME/s | Speedup: 36.43×

    tadd | N=    1000000 | SIMD: 18876.23 ME/s | Scalar:   553.12 ME/s | Speedup: 34.12×
    tmul | N=    1000000 | SIMD: 19987.34 ME/s | Scalar:   552.89 ME/s | Speedup: 36.15×
    tmin | N=    1000000 | SIMD: 20543.21 ME/s | Scalar:   553.45 ME/s | Speedup: 37.11×
    tmax | N=    1000000 | SIMD: 20512.67 ME/s | Scalar:   553.23 ME/s | Speedup: 37.07×

    tadd | N=   10000000 | SIMD: 19123.45 ME/s | Scalar:   554.23 ME/s | Speedup: 34.51×
    tmul | N=   10000000 | SIMD: 20234.56 ME/s | Scalar:   554.01 ME/s | Speedup: 36.52×
    tmin | N=   10000000 | SIMD: 20987.34 ME/s | Scalar:   554.67 ME/s | Speedup: 37.83×
    tmax | N=   10000000 | SIMD: 20965.23 ME/s | Scalar:   554.45 ME/s | Speedup: 37.80×

=============================================================================
Benchmark complete.
```

**Key metrics to capture:**
- Best SIMD throughput: tmin/tmax at ~20.5-21.0 Gops/s (20,500-21,000 ME/s)
- SIMD vs Scalar speedup: 34-38× for large arrays
- Performance consistency: <5% variance across runs

---

## Comparison with Python Benchmarks

### Python Results (from bench_phase0.py)

**Best operation (tnot):**
- Throughput: 19,570 Mops/s (19.57 Gops/s)
- Platform: Windows x64, AVX2
- Date: 2025-11-23

### Expected C++ Results

**Projected (5-15% faster due to no Python overhead):**
- Best throughput: 20,500-21,000 Mops/s (20.5-21.0 Gops/s)
- Python overhead: 5-10% (pybind11 + NumPy conversions)

### Validation Checklist

After running C++ benchmarks:
- [ ] Record peak throughput (best operation, largest size)
- [ ] Calculate overhead: `(cpp_throughput - python_throughput) / python_throughput * 100%`
- [ ] Expected overhead: 5-15%
- [ ] If overhead <5%: Python bindings are excellent (minimal cost)
- [ ] If overhead >15%: Investigate Python binding bottlenecks

---

## Troubleshooting

### Error: "cl is not recognized as an internal or external command"

**Cause:** Not running in Developer Command Prompt

**Solution:** Use one of the methods in Step 1 to activate MSVC environment

---

### Error: "Cannot open include file 'immintrin.h'"

**Cause:** AVX2 headers not found

**Solution:** Ensure `/I..\..\src` flag is included, or check Visual Studio installation

---

### Error: "bench_kernels.cpp(XX): error C2065: 'tadd_simd' undeclared identifier"

**Cause:** Source files not found

**Solution:** Ensure you're in `benchmarks/cpp-native-kernels/` directory and `../../src/core/` exists

---

### Benchmark runs but shows 0 ME/s

**Cause:** Optimization disabled or timing issue

**Solution:** Ensure `/O2` flag is used for optimization

---

## Alternative: Use Python Build Script

If you prefer automated compiler detection:

```cmd
# From Developer Command Prompt:
cd C:\Users\Gestalt\Desktop\ternary\repos\ternary-engine\benchmarks\cpp-native-kernels
python build_kernels.py --run
```

This will:
1. Auto-detect MSVC compiler
2. Compile with optimal flags
3. Run benchmark automatically
4. Display results

---

## Saving Results

### Human-readable format
```cmd
bin\bench_kernels.exe > results_human.txt
```

### CSV format (for analysis)
```cmd
bin\bench_kernels.exe --csv > results.csv
```

### Import into Excel/Python
```python
import pandas as pd
df = pd.read_csv("results.csv")
print(df.groupby("operation")["throughput_simd_ME_s"].max())
```

---

## Next Steps After Benchmark Completes

1. **Capture results:**
   - Save CSV output: `bin\bench_kernels.exe --csv > results_2025-12-04.csv`
   - Save human output: `bin\bench_kernels.exe > results_2025-12-04.txt`

2. **Compare with Python:**
   ```python
   # Quick comparison
   import pandas as pd
   cpp_results = pd.read_csv("results_2025-12-04.csv")
   python_best = 19570  # Mops/s from Python benchmarks

   cpp_best = cpp_results["throughput_simd_ME_s"].max()
   overhead_pct = (cpp_best - python_best) / python_best * 100

   print(f"C++ best: {cpp_best:.2f} Mops/s")
   print(f"Python best: {python_best} Mops/s")
   print(f"Python overhead: {overhead_pct:.1f}%")
   ```

3. **Update README.md:**
   - Replace "Up to 19.57 Gops/s" with validated C++ number
   - Cite: "C++ native kernels, Windows x64 AVX2, validated 2025-12-04"
   - Document overhead: "Python bindings add ~X% overhead"

4. **Commit results:**
   ```bash
   git add benchmarks/cpp-native-kernels/results_2025-12-04.csv
   git add benchmarks/cpp-native-kernels/results_2025-12-04.txt
   git commit -m "BENCHMARK: C++ native kernel validation results"
   ```

5. **Update commercial materials:**
   - Use C++ benchmark numbers for performance claims
   - Document methodology and validation platform
   - Include confidence intervals if multiple runs performed

---

## Expected Timeline

- **Compilation:** 30-60 seconds
- **Benchmark execution:** 30-60 seconds
- **Analysis:** 5-10 minutes
- **Documentation update:** 10-15 minutes
- **Total:** 45-90 minutes end-to-end

---

## Status

- [x] Infrastructure complete
- [x] Build scripts ready
- [x] Documentation complete
- [ ] **Compilation pending** (requires Developer Command Prompt)
- [ ] Benchmark execution pending
- [ ] Results comparison pending
- [ ] README.md update pending

---

**Last Updated:** 2025-12-04
**Author:** Ternary Engine Project
**Status:** Ready for compilation and execution
