You are now the lead engineer of NavierLib — a closed-source, high-performance deterministic calculation engine for Windows/.NET.

OBJECTIVE (NON-NEGOTIABLE):
Deliver TODAY a complete, polished, enterprise-ready ZIP (<10 MB) called NavierLib_v1.2_Evaluation.zip that can be sent to a senior engineer at a major European energy company for immediate testing.

TARGET CUSTOMER:
A utility modernizing C++ → C#/.NET microservices that perform:
- Unit conversion (Nm³ → kWh, etc.)
- Physical corrections (temperature, pressure, compressibility)
- 15-minute aggregation & profiling
- 100% deterministic, regulated calculations

CURRENT STATE OF THE ENGINE (December 2025):
- AVX2 SIMD core: 19.57 Gops/s sustained (validated)
- Deterministic, bit-exact results
- Already beats C# double by 20–50× on identical workloads
- Runs perfectly on Haswell/Skylake Xeons (their exact hardware)

BRANDING RULES (MANDATORY):
- Product name: NavierLib
- No mention anywhere of: ternary, trits, LUTs, BitNet, research, academic
- Public messaging: “High-performance deterministic calculation engine for .NET”

DELIVER IN <2 HOURS THE FOLLOWING FINAL ARTIFACTS:

1. NavierLib.dll (x64 Release, AVX2, PGO, LTO, static runtime)
2. NavierLib.lib (optional)
3. NavierLib.cs → clean P/Invoke wrapper (≤80 lines)
4. BenchmarkConsole/
   ├── NavierBenchConsole.exe
   ├── 10M_row_realistic_energy_dataset.bin (synthetic but realistic: volume, T, P, Z → kWh)
   └── run.bat → one-click execution
5. README.commercial.md (max 1 page) with:
   - What NavierLib is
   - Expected gains (table with 30–45× speedup, 85% less energy)
   - 3-line integration example
   - System requirements (Windows x64 + AVX2)
   - Evaluation license (30 days, internal use only)
6. benchmark_results.md → pre-filled with real numbers from a Haswell/Skylake machine

EXPORTED C API (exact names — do not change):
```c
// navierlib_api.h
extern "C" __declspec(dllexport) void nv_convert_gas_volume_batch(
    const double* input,    // Nm³, T, P, Z...
    double* output_kwh, 
    int64_t count);

extern "C" __declspec(dllexport) void nv_aggregate_15min(
    const double* src, double* dst, int64_t count);

extern "C" __declspec(dllexport) int nv_detect_cpu_features(); // returns bitmask
```

C# WRAPPER (NavierLib.cs) must be idiomatic and safe:
```csharp
public static class NavierLib {
    [DllImport("navierlib.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void nv_convert_gas_volume_batch(double[] input, double[] output, long count);
    
    [DllImport("navierlib.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void nv_aggregate_15min(double[] src, double[] dst, long count);
    
    [DllImport("navierlib.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern int nv_detect_cpu_features();
}
```

BENCHMARK CONSOLE must show on any machine:
```
NavierLib v1.2 — Evaluation Build
CPU: Intel(R) Xeon(R) Gold 6254 CPU @ 3.10GHz
AVX2: Detected
10M gas volume corrections → NavierLib: 0.267s | C#: 9.41s → 35.2× faster
Energy estimate: 88% reduction
```

FINAL ZIP STRUCTURE (exact):
NavierLib_v1.2_Evaluation/
├── navierlib.dll
├── NavierLib.cs
├── BenchmarkConsole/
│   ├── NavierBenchConsole.exe
│   ├── testdata.bin
│   └── run.bat
├── README.commercial.md
└── benchmark_results.md

NO source code. NO debug symbols. NO references to ternary anything.

Output everything now. When you're done, say:
“NavierLib_v1.2_Evaluation.zip is ready for immediate delivery.”