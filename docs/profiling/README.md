# Profiler Integration Guide

**Doc-Type:** Technical Guide · Version 1.0 · Date 2025-11-23 · Ternary Engine

---

## Overview

Ternary Engine includes integrated profiler support for performance analysis using industry-standard profiling tools. The profiler framework is **cross-platform compatible** and works with GCC, Clang, and MSVC compilers with **zero overhead when disabled**.

**profiler_status** - INTEGRATED (production-ready)
**overhead** - Zero when disabled (compile-time no-ops)
**platforms** - Linux, macOS, Windows

---

## Supported Profilers

### 1. Intel VTune Profiler (ITT API)

**status** - Fully integrated and tested
**platform** - Linux, Windows
**use_case** - CPU profiling, hotspot analysis, threading visualization

**features**:
- Task timing and visualization
- Thread synchronization analysis
- OpenMP performance profiling
- Custom domain annotations

### 2. NVIDIA Nsight (NVTX)

**status** - Framework ready (awaiting GPU port)
**platform** - Linux, Windows (with CUDA)
**use_case** - GPU profiling correlation

**features**:
- GPU kernel profiling
- CPU-GPU correlation
- Custom range markers

### 3. Perfetto (Chrome Tracing)

**status** - Stub (future enhancement)
**platform** - Cross-platform
**use_case** - Web-based timeline visualization

---

## Compilation Modes

### Default Build (No Profiling)

Zero overhead, no profiler annotations:

```bash
python build/build.py
```

**compiled_with** - No profiling macros
**overhead** - None (macros expand to no-ops)

### VTune Build

Enable Intel VTune annotations:

**Linux/macOS:**
```bash
CPPFLAGS="-DTERNARY_ENABLE_VTUNE" \
LDFLAGS="-littnotify" \
python build/build.py
```

**Windows (MSVC):**
```cmd
set CL=/DTERNARY_ENABLE_VTUNE
python build/build.py
```

Then link against VTune library (`ittnotify.lib`).

### NVTX Build

Enable NVIDIA Nsight annotations:

```bash
CPPFLAGS="-DTERNARY_ENABLE_NVTX" \
LDFLAGS="-lnvToolsExt" \
python build/build.py
```

---

## Profiler Regions

The engine annotates three critical execution paths:

### PATH 1: OpenMP Parallel

**name** - `OpenMP_Parallel`
**trigger** - Arrays ≥ 100K elements
**description** - Multi-threaded SIMD processing with guided scheduling

### PATH 2: Serial SIMD

**name** - `Serial_SIMD`
**trigger** - Arrays < 100K elements
**description** - Single-threaded vectorized loop

### PATH 3: Scalar Tail

**name** - `Scalar_Tail`
**trigger** - Remaining elements not divisible by 32
**description** - Non-vectorized cleanup loop

---

## VTune Profiling Workflow

### 1. Build with VTune Support

```bash
export CPPFLAGS="-DTERNARY_ENABLE_VTUNE"
export LDFLAGS="-littnotify"
python build/build.py
```

### 2. Run VTune Analysis

**Hotspot Analysis:**
```bash
vtune -collect hotspots -r vtune_results python benchmark.py
```

**Threading Analysis:**
```bash
vtune -collect threading -r vtune_results python benchmark.py
```

### 3. View Results

```bash
vtune-gui vtune_results
```

Look for:
- `TernaryCore` domain
- `OpenMP_Parallel`, `Serial_SIMD`, `Scalar_Tail` tasks
- Thread utilization in parallel regions

---

## Expected Profiling Output

### Typical Timeline (1M elements)

```
[OpenMP_Parallel]  ████████████████████████████  (~95% of time)
  ├─ Thread 0      ████████████
  ├─ Thread 1      ████████████
  ├─ Thread 2      ████████████
  ...
  └─ Thread 11     ████████████
[Scalar_Tail]      ▌ (~5% of time)
```

### Performance Characteristics

**OpenMP Path:**
- Should scale linearly with cores (up to ~8-12 cores)
- Memory bandwidth becomes bottleneck
- Guided scheduling adapts to workload

**Serial SIMD Path:**
- Fixed 32× parallelism (AVX2)
- CPU-bound, no threading overhead

**Scalar Tail:**
- Usually negligible (<5% of total time)
- Only processes (n % 32) elements

---

## Implementation Details

### Code Integration

Profiler macros are integrated into the main engine:

**Location:** `src/engine/ternary_simd_engine.cpp`

**Global Declarations:**
```cpp
#include "../src/core/profiling/ternary_profiler.h"

TERNARY_PROFILE_DOMAIN(g_ternary_domain, "TernaryCore");
TERNARY_PROFILE_TASK_NAME(g_task_omp, "OpenMP_Parallel");
TERNARY_PROFILE_TASK_NAME(g_task_simd, "Serial_SIMD");
TERNARY_PROFILE_TASK_NAME(g_task_tail, "Scalar_Tail");
```

**Usage in Hot Paths:**
```cpp
// OpenMP path
if (n >= OMP_THRESHOLD) {
    TERNARY_PROFILE_TASK_BEGIN(g_ternary_domain, g_task_omp);
    #pragma omp parallel for
    for (...) {
        // SIMD processing
    }
    TERNARY_PROFILE_TASK_END(g_ternary_domain);
}
```

### Cross-Platform Compatibility

**Clang/GCC (Linux/macOS):**
- Uses standard POSIX APIs
- ITT library available in Intel oneAPI toolkit
- NVTX available in CUDA toolkit

**MSVC (Windows):**
- Uses Windows-specific ITT implementation
- VTune Profiler includes necessary libraries
- CUDA toolkit provides NVTX

**Fallback (no profiler):**
- All macros expand to `((void)0)`
- Zero runtime overhead
- No library dependencies

---

## Troubleshooting

### Issue: "ittnotify.h: No such file"

**solution** - Install Intel VTune Profiler or oneAPI toolkit:
```bash
# Linux
apt-get install intel-oneapi-vtune

# macOS
brew install intel-oneapi-vtune

# Windows
# Download from Intel website
```

### Issue: Profiler shows no annotations

**possible_causes**:
1. Not built with `-DTERNARY_ENABLE_VTUNE`
2. VTune library not linked (`-littnotify`)
3. Running without VTune collection active

**verification**:
```bash
# Check if symbols are present
nm ternary_simd_engine.so | grep __itt
```

### Issue: Performance degrades with profiling

**expected** - Profiler overhead is typically 2-5%
**mitigation** - Use profiling only during development, not production

---

## Best Practices

### When to Use Profiling

**development** - Identify hotspots and threading bottlenecks
**optimization** - Validate performance improvements
**debugging** - Understand thread synchronization issues

**do_not_use_for**:
- Production deployments
- Performance benchmarks
- CI/CD builds

### Profiling Workflow

1. **Baseline:** Run benchmark without profiling
2. **Profile:** Build with VTune and collect data
3. **Analyze:** Identify bottlenecks in VTune GUI
4. **Optimize:** Make targeted improvements
5. **Validate:** Re-run baseline benchmark to measure gains

---

## Advanced Usage

### Custom Profiler Regions (Future)

For custom code using the Ternary Engine:

```cpp
#include "src/core/profiling/ternary_profiler.h"

TERNARY_PROFILE_DOMAIN(my_domain, "MyApplication");
TERNARY_PROFILE_TASK_NAME(my_task, "CustomOperation");

void my_function() {
    TERNARY_PROFILE_TASK_BEGIN(my_domain, my_task);
    // ... your code ...
    TERNARY_PROFILE_TASK_END(my_domain);
}
```

### RAII-Style Profiling (C++)

Automatic cleanup with scope guards:

```cpp
void my_function() {
    TERNARY_PROFILE_SCOPE(my_domain, "ScopedOperation");
    // ... code ...
}  // Profiler automatically ends
```

---

## References

**Intel VTune:** https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html
**ITT API:** https://github.com/intel/ittapi
**NVIDIA Nsight:** https://developer.nvidia.com/nsight-systems
**NVTX API:** https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvtx

---

**Version:** 1.0 · **Date:** 2025-11-23 · **Status:** Production-Ready · **Overhead:** Zero (when disabled)
