# NavierLib Build Instructions

Complete guide to building NavierLib v1.2 evaluation package from source.

---

## Prerequisites

**Required Software:**
- Visual Studio 2019 or 2022 (with C++ Desktop Development workload)
- .NET SDK 6.0 or later
- Python 3.7+ (for build scripts)
- Windows 10/11 or Windows Server 2016+

**Hardware:**
- Intel CPU with AVX2 (Haswell 2013+) or AMD (Excavator 2015+)
- 8 GB RAM minimum
- 1 GB free disk space

---

## Quick Build (All Components)

**1. Open Visual Studio Developer Command Prompt:**
```
Start Menu → Visual Studio 2022 → x64 Native Tools Command Prompt for VS 2022
```

**2. Navigate to repository:**
```
cd C:\Users\Gestalt\Desktop\ternary\repos\ternary-engine
```

**3. Checkout dll-navierlib branch:**
```
git checkout dll-navierlib
```

**4. Build NavierLib DLL:**
```
python build\build_navierlib.py
```

Expected output:
```
Nav

ierLib Build System
Target: x64 Release DLL with AVX2 + LTCG

Compiling sources...
  navierlib_impl.cpp... OK

Linking DLL...
✓ NavierLib.dll built successfully
✓ Output: C:\Users\Gestalt\Desktop\ternary\repos\ternary-engine\dist\NavierLib
  navierlib.dll: 45.2 KB
```

**5. Build C# Benchmark Console:**
```
cd dist\BenchmarkConsole
csc /out:NavierBenchConsole.exe /reference:..\NavierLib.cs NavierBenchConsole.cs
cd ..\..
```

**6. Test the build:**
```
cd dist\BenchmarkConsole
run.bat
```

---

## Step-by-Step Build

### Step 1: Build NavierLib.dll

**Method A: Automated Build (Recommended)**
```batch
REM From repository root, in VS Developer Command Prompt
python build\build_navierlib.py
```

**Method B: Manual Build**
```batch
REM Create build directory
mkdir build\navierlib_build

REM Compile
cl.exe /c /O2 /Oi /GL /arch:AVX2 /fp:fast /GS- /Gy /MT /EHsc /std:c++17 ^
    /DNAVIERLIB_EXPORTS /DNDEBUG /W3 ^
    /Fobuild\navierlib_build\navierlib_impl.obj ^
    src\navierlib\navierlib_impl.cpp

REM Link
link.exe /DLL /LTCG /OPT:REF /OPT:ICF /MACHINE:X64 /SUBSYSTEM:WINDOWS ^
    /OUT:dist\NavierLib\navierlib.dll ^
    build\navierlib_build\navierlib_impl.obj

REM Copy header
copy src\navierlib\navierlib_api.h dist\NavierLib\
```

**Verify DLL:**
```batch
dumpbin /headers dist\NavierLib\navierlib.dll | findstr "machine"
REM Should show: x64

dumpbin /exports dist\NavierLib\navierlib.dll
REM Should list: nv_convert_gas_volume_batch, nv_aggregate_15min, etc.
```

### Step 2: Build C# Benchmark Console

**Method A: Visual Studio**
1. Create new Console App (.NET 6.0)
2. Add NavierBenchConsole.cs to project
3. Add NavierLib.cs to project
4. Build → Release → Any CPU
5. Copy output to dist/BenchmarkConsole/

**Method B: Command Line (from dist/BenchmarkConsole)**
```batch
REM Compile benchmark console
csc /out:NavierBenchConsole.exe /platform:x64 /optimize+ ^
    /reference:System.dll /reference:System.Runtime.InteropServices.dll ^
    NavierBenchConsole.cs ..\NavierLib.cs

REM Verify
NavierBenchConsole.exe
```

### Step 3: Package for Distribution

```batch
REM From repository root
python build\package_navierlib.py
```

This creates: `NavierLib_v1.2_Evaluation.zip` with:
```
NavierLib_v1.2_Evaluation/
├── navierlib.dll
├── NavierLib.cs
├── BenchmarkConsole/
│   ├── NavierBenchConsole.exe
│   ├── testdata.bin (auto-generated on first run)
│   └── run.bat
├── README.commercial.md
└── benchmark_results.md
```

---

## Troubleshooting

### Error: "MSVC compiler not found"

**Solution:** You must run from Visual Studio Developer Command Prompt, not regular Command Prompt.

Open: `Start Menu → Visual Studio 2022 → x64 Native Tools Command Prompt for VS 2022`

### Error: "AVX2 not supported on this CPU"

**Solution:** Your CPU does not support AVX2. NavierLib requires:
- Intel Haswell (2013) or newer
- AMD Excavator (2015) or newer

Check CPU:
```batch
wmic cpu get name
```

### Error: "Cannot find navierlib.dll"

**Solution:** Ensure DLL is in correct location:
```batch
REM DLL must be in same folder as NavierBenchConsole.exe
copy dist\NavierLib\navierlib.dll dist\BenchmarkConsole\
```

Or add to PATH:
```batch
set PATH=%PATH%;C:\path\to\dist\NavierLib
```

### Error: "FileNotFoundException: Could not load file or assembly"

**Solution:** Install Visual C++ Redistributable 2019/2022:
- Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
- Run installer
- Restart application

### Build Warnings

Safe to ignore:
- `warning LNK4075: ignoring '/EDITANDCONTINUE'` (expected in Release mode)
- `warning C4244: conversion from 'double' to 'float'` (intentional)

Require attention:
- `error LNK2001: unresolved external symbol` → Missing library or incorrect link flags
- `error C2065: undeclared identifier` → Check include paths

---

## Advanced Build Options

### Profile-Guided Optimization (PGO)

PGO can provide additional 5-15% performance gain:

```batch
REM Phase 1: Instrumentation build
python build\build_navierlib.py --pgo-instrument

REM Phase 2: Training run (collect profile data)
cd dist\BenchmarkConsole
NavierBenchConsole.exe
cd ..\..

REM Phase 3: Optimized build with profile
python build\build_navierlib.py --pgo-optimize
```

### Debug Build

For development/debugging:

```batch
REM Add /Zi /DEBUG flags
cl.exe /c /Zi /Od /arch:AVX2 /MTd /EHsc /std:c++17 ^
    /DNAVIERLIB_EXPORTS /D_DEBUG ^
    /Fobuild\navierlib_build\navierlib_impl.obj ^
    src\navierlib\navierlib_impl.cpp

link.exe /DLL /DEBUG /MACHINE:X64 ^
    /OUT:dist\NavierLib\navierlib_debug.dll ^
    build\navierlib_build\navierlib_impl.obj
```

### Static Analysis

Run code analysis:

```batch
cl.exe /analyze /c ... (compile flags)
```

---

## Verification

### Functional Test

```batch
cd dist\BenchmarkConsole
NavierBenchConsole.exe
```

Expected output:
- AVX2 support detected: ✓
- 10M records processed: ✓
- Speedup: 30-50×
- Correctness verified: ✓

### Performance Validation

Minimum acceptable performance (Intel Xeon Gold 6254 baseline):
- Gas volume conversion (10M): < 500 ms
- 15-min aggregation (900K): < 200 ms
- Speedup vs C#: > 20×

If performance is significantly lower:
1. Check CPU frequency (ensure not thermal throttling)
2. Disable background processes
3. Check if running in VM (reduces SIMD performance)
4. Verify AVX2 enabled: `dumpbin /disasm navierlib.dll | findstr "vpaddd"`

---

## Build Artifacts

**Release Artifacts (dist/NavierLib/):**
- navierlib.dll (40-60 KB) - Main library
- navierlib.lib (optional) - Import library for C++ linking
- navierlib_api.h - C API header

**Benchmark Artifacts (dist/BenchmarkConsole/):**
- NavierBenchConsole.exe (8-15 KB)
- NavierLib.cs (managed wrapper)
- testdata.bin (305 MB, auto-generated)

**Total Package Size:** < 10 MB (excluding testdata.bin)

---

## Continuous Integration

**GitHub Actions Example:**

```yaml
name: Build NavierLib

on: [push]

jobs:
  build:
    runs-on: windows-latest

    steps:
    - uses: actions/checkout@v3

    - name: Setup MSBuild
      uses: microsoft/setup-msbuild@v1

    - name: Build DLL
      run: python build\build_navierlib.py

    - name: Build Benchmark
      run: |
        cd dist\BenchmarkConsole
        csc /out:NavierBenchConsole.exe NavierBenchConsole.cs ..\NavierLib.cs

    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: NavierLib-v1.2
        path: dist/
```

---

## Support

**Build Issues:**
- GitHub Issues: https://github.com/your-repo/ternary-engine/issues
- Email: build-support@navierlib.com

**Documentation:**
- API Reference: See navierlib_api.h inline docs
- Commercial README: dist/README.commercial.md
- Benchmark Results: dist/benchmark_results.md

---

**Last Updated:** December 4, 2025
**Build System Version:** 1.2.0
