# NavierLib v1.2 — Evaluation Package

**High-Performance Deterministic Calculation Engine for .NET**

NavierLib accelerates critical energy sector calculations by 30-50× compared to standard C# double-precision arithmetic, while maintaining 100% deterministic, bit-exact results for regulatory compliance.

---

## What is NavierLib?

NavierLib is a native x64 library optimized for Windows/.NET environments performing high-volume deterministic calculations common in utility and energy sectors:

- **Gas volume conversions** (Nm³ ↔ kWh) with physical corrections
- **Time-series aggregation** (15-minute profiling for regulatory reporting)
- **Unit conversions** with temperature, pressure, and compressibility corrections

**Key Features:**
- 30-50× faster than C# baseline for typical workloads
- 85% energy consumption reduction
- 100% deterministic, bit-exact results
- Zero external dependencies (static runtime)
- Requires Intel AVX2 (Haswell 2013+) or AMD (Excavator 2015+)

---

## Performance Highlights

| Operation | C# Baseline | NavierLib | Speedup |
|-----------|------------|-----------|---------|
| Gas volume conversion (10M records) | 9.41s | 0.267s | **35.2×** |
| 15-minute aggregation (1M records) | 3.2s | 0.11s | **29.1×** |
| Physical corrections (1M records) | 1.8s | 0.05s | **36.0×** |

**Energy Efficiency:**
- Measured on Intel Xeon Gold 6254 @ 3.10GHz
- 88% reduction in CPU time → proportional energy savings
- Validated with Intel RAPL counters

---

## Quick Integration Example

```csharp
using NavierLib;

// Check CPU support (one-time at startup)
if (!Api.IsSupported())
    throw new NotSupportedException("AVierLib requires AVX2");

// Convert 10,000 gas volumes (Nm³ → kWh)
int count = 10_000;
double[] input = new double[count * 4];  // [Volume, Temp, Pressure, Z] per record
double[] output = new double[count];

// Fill input with your data...
// input[i*4+0] = volume_nm3;
// input[i*4+1] = temperature_celsius;
// input[i*4+2] = pressure_bar;
// input[i*4+3] = compressibility_factor;

// Perform conversion (AVX2-accelerated)
Api.ConvertGasVolumeBatch(input, output, count);

// output now contains kWh values
```

---

## System Requirements

**Minimum:**
- Windows Server 2016 / Windows 10 x64
- Intel Haswell (2013+) or AMD Excavator (2015+) with AVX2
- .NET Framework 4.7.2 or .NET 6+
- 8 GB RAM

**Recommended:**
- Windows Server 2019/2022
- Intel Skylake/Ice Lake or AMD Zen2/Zen3/Zen4
- .NET 6 or .NET 8
- 16+ GB RAM

**Not Supported:**
- ARM64 (Apple M1/M2, ARM servers)
- x86 32-bit
- CPUs without AVX2 (pre-2013 Intel, pre-2015 AMD)

---

## Evaluation License

This evaluation package is licensed for:
- **Internal testing only** (30 days from receipt)
- **Non-production use** (development/QA environments)
- **Single organization** (not for redistribution)

For production licensing, contact: licensing@navierlib.com

---

## Package Contents

```
NavierLib_v1.2_Evaluation/
├── navierlib.dll          # Native x64 library (AVX2 optimized)
├── NavierLib.cs           # C# P/Invoke wrapper
├── BenchmarkConsole/
│   ├── NavierBenchConsole.exe
│   ├── testdata.bin       # 10M record synthetic dataset
│   └── run.bat            # One-click benchmark execution
├── README.commercial.md   # This file
└── benchmark_results.md   # Pre-validated performance data
```

---

## Running the Benchmark

**Option 1: One-click execution**
```
cd BenchmarkConsole
run.bat
```

**Option 2: Manual execution**
```
cd BenchmarkConsole
NavierBenchConsole.exe
```

Expected output:
```
NavierLib v1.2 — Evaluation Build
======================================================================

Platform: Windows 10 (Build 19045)
Runtime:  .NET 6.0.25
CPU:      16 cores

✓ AVX2 support detected
✓ CPU Features: AVX2: Yes, FMA: Yes

✓ Test dataset: 10,000,000 gas volume records loaded

======================================================================
BENCHMARK: Gas Volume Conversion (Nm³ → kWh)
======================================================================

RESULTS:
----------------------------------------------------------------------
NavierLib (AVX2):       0.267s
C# Baseline (double):   9.410s
Speedup:                35.2×
Energy Reduction:       ~97%
----------------------------------------------------------------------

✓ Correctness verified (sample: 1000 records)
  Max error: 0.0000000000E+00 (bit-exact match)
```

---

## Integration Notes

**DLL Placement:**
- Place `navierlib.dll` in same directory as your executable, OR
- Add to system PATH, OR
- Specify full path in DllImport attribute

**Thread Safety:**
- All functions are thread-safe
- Can be called concurrently from multiple threads
- Internal error handling uses thread-local storage

**Error Handling:**
```csharp
// Low-level (direct P/Invoke)
NavierLib.Native.nv_convert_gas_volume_batch(input, output, count);
string error = NavierLib.Native.GetLastError();
if (error != null)
    throw new Exception($"NavierLib error: {error}");

// High-level (managed API with automatic error handling)
try {
    NavierLib.Api.ConvertGasVolumeBatch(input, output, count);
} catch (NotSupportedException ex) {
    // Handle AVX2 not available
} catch (InvalidOperationException ex) {
    // Handle NavierLib internal error
}
```

---

## Technical Support

**Evaluation Period Support:**
- Email: eval-support@navierlib.com
- Response time: 1-2 business days

**Documentation:**
- API Reference: See NavierLib.cs inline documentation
- Integration Guide: Available upon request

**Known Limitations:**
- Windows x64 only (Linux/macOS not supported)
- AVX2 required (will return error on older CPUs)
- Maximum array size: 2^31 records (INT64_MAX)

---

## FAQ

**Q: Can I use this in production?**
A: This evaluation build is for testing only. Contact us for production licensing.

**Q: What if my CPU doesn't support AVX2?**
A: NavierLib will detect this at runtime and return an error. AVX2 is required for the optimizations. Most server CPUs from 2014+ have AVX2.

**Q: Is the data deterministic across runs?**
A: Yes. NavierLib produces bit-exact results for identical inputs, guaranteed across multiple runs and machines (given same CPU architecture).

**Q: Can I redistribute the DLL?**
A: No. Evaluation license is for internal use only. Contact us for redistribution rights.

**Q: How does this compare to Intel MKL or other libraries?**
A: NavierLib is specialized for energy sector calculations (gas volume conversions, aggregations). General-purpose math libraries like MKL don't provide these domain-specific operations.

---

**NavierLib Technologies** | licensing@navierlib.com | navierlib.com
Evaluation Build v1.2 | December 2025
