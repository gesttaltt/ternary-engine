# CPU Detection Reference

**File:** `cpu_simd_capability.h`
**Status:** Production-ready
**Supported Platforms:** x86-64 (Intel/AMD), ARM64 (NEON/SVE)

---

## Overview

Runtime CPU feature detection enables dynamic backend selection and graceful degradation. The system queries CPU capabilities at runtime using platform-specific instructions (CPUID on x86, compile-time macros on ARM).

---

## Architecture Support

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CPU DETECTION FLOW                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Program     │───▶│  detect_     │───▶│  SIMDLevel enum      │  │
│  │  Startup     │    │  best_simd() │    │  (AVX512/AVX2/...)   │  │
│  └──────────────┘    └──────────────┘    └──────────┬───────────┘  │
│                                                      │              │
│                                                      ▼              │
│                      ┌───────────────────────────────────────────┐  │
│                      │            BACKEND SELECTION              │  │
│                      │                                           │  │
│                      │  AVX-512BW → avx512_backend              │  │
│                      │  AVX2      → avx2_v1_backend (or v2)     │  │
│                      │  NEON      → neon_backend                │  │
│                      │  SVE       → sve_backend                 │  │
│                      │  NONE      → scalar_backend              │  │
│                      │                                           │  │
│                      └───────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Platform Detection

### Compile-Time Macros

```cpp
// x86-64 (Intel/AMD)
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define TERNARY_X86
#endif

// ARM64
#if defined(__aarch64__) || defined(_M_ARM64)
    #define TERNARY_ARM64
#endif
```

---

## SIMD Levels

### Enumeration

```cpp
enum class SIMDLevel {
    NONE,       // Scalar only (fallback)
    AVX2,       // 256-bit vectors (32 trits/op)
    AVX512BW,   // 512-bit vectors (64 trits/op)
    NEON,       // ARM 128-bit vectors (16 trits/op)
    SVE         // ARM scalable vectors (variable width)
};
```

### Capabilities

| Level | Vector Width | Trits/Op | Min CPU |
|-------|-------------|----------|---------|
| NONE | 8 bits | 1 | Any |
| AVX2 | 256 bits | 32 | Haswell (2013) |
| AVX512BW | 512 bits | 64 | Skylake-X (2017) |
| NEON | 128 bits | 16 | ARMv7+ |
| SVE | Variable | 16-128+ | ARMv8.2+ |

---

## x86-64 Detection

### CPUID Interface

```cpp
inline void cpuid(uint32_t leaf, uint32_t subleaf,
                  uint32_t* eax, uint32_t* ebx,
                  uint32_t* ecx, uint32_t* edx) {
#ifdef _MSC_VER
    int regs[4];
    __cpuidex(regs, (int)leaf, (int)subleaf);
    *eax = regs[0];
    *ebx = regs[1];
    *ecx = regs[2];
    *edx = regs[3];
#else
    __cpuid_count(leaf, subleaf, *eax, *ebx, *ecx, *edx);
#endif
}
```

### Feature Bit Definitions

```cpp
// CPUID leaf 7, subleaf 0, EBX register
#define CPUID_BIT_AVX2      (1 << 5)   // Bit 5
#define CPUID_BIT_AVX512F   (1 << 16)  // Bit 16
#define CPUID_BIT_AVX512BW  (1 << 30)  // Bit 30
```

### Detection Functions

```cpp
// AVX2 (Intel Haswell 2013+, AMD Excavator 2015+)
inline bool has_avx2() {
    uint32_t eax, ebx, ecx, edx;

    // Check maximum CPUID leaf
    cpuid(0, 0, &eax, &ebx, &ecx, &edx);
    if (eax < 7) return false;

    // Query extended features
    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    return (ebx & CPUID_BIT_AVX2) != 0;
}

// AVX-512 Foundation
inline bool has_avx512f() {
    uint32_t eax, ebx, ecx, edx;
    cpuid(0, 0, &eax, &ebx, &ecx, &edx);
    if (eax < 7) return false;

    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    return (ebx & CPUID_BIT_AVX512F) != 0;
}

// AVX-512 Byte/Word (required for 8-bit operations)
inline bool has_avx512bw() {
    uint32_t eax, ebx, ecx, edx;
    cpuid(0, 0, &eax, &ebx, &ecx, &edx);
    if (eax < 7) return false;

    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    return (ebx & CPUID_BIT_AVX512BW) != 0;
}
```

---

## ARM64 Detection

ARM features are detected via compile-time macros:

```cpp
// ARM NEON (128-bit SIMD)
inline bool has_neon() {
#ifdef __ARM_NEON
    return true;
#else
    return false;
#endif
}

// ARM SVE (Scalable Vector Extension)
inline bool has_sve() {
#ifdef __ARM_FEATURE_SVE
    return true;
#else
    return false;
#endif
}
```

**Note:** Runtime detection for ARM is possible via `/proc/cpuinfo` or HWCAP on Linux, but compile-time detection is more portable.

---

## Best SIMD Selection

```cpp
inline SIMDLevel detect_best_simd() {
#ifdef TERNARY_X86
    if (has_avx512bw()) return SIMDLevel::AVX512BW;
    if (has_avx2()) return SIMDLevel::AVX2;
    return SIMDLevel::NONE;
#elif defined(TERNARY_ARM64)
    if (has_sve()) return SIMDLevel::SVE;
    if (has_neon()) return SIMDLevel::NEON;
    return SIMDLevel::NONE;
#else
    return SIMDLevel::NONE;
#endif
}
```

---

## Human-Readable Output

```cpp
inline const char* simd_level_name(SIMDLevel level) {
    switch (level) {
        case SIMDLevel::AVX512BW: return "AVX-512BW";
        case SIMDLevel::AVX2:     return "AVX2";
        case SIMDLevel::NEON:     return "ARM NEON";
        case SIMDLevel::SVE:      return "ARM SVE";
        case SIMDLevel::NONE:     return "Scalar (no SIMD)";
        default:                  return "Unknown";
    }
}
```

---

## Usage Example

```cpp
#include "core/simd/cpu_simd_capability.h"

int main() {
    // Detect best available SIMD
    SIMDLevel level = detect_best_simd();

    printf("SIMD Level: %s\n", simd_level_name(level));

    // Individual feature checks
    if (has_avx2()) {
        printf("AVX2: Available\n");
    }

    if (has_avx512bw()) {
        printf("AVX-512BW: Available\n");
    }

    // Select backend based on detection
    switch (level) {
        case SIMDLevel::AVX512BW:
            // Use AVX-512 backend
            break;
        case SIMDLevel::AVX2:
            // Use AVX2 backend (production default)
            break;
        default:
            // Fall back to scalar
            break;
    }

    return 0;
}
```

---

## Integration with Backend System

The backend dispatch uses CPU detection for automatic selection:

```cpp
bool ternary_backend_init() {
    // Detect CPU capabilities
    SIMDLevel level = detect_best_simd();

    // Register available backends
    ternary_register_scalar_backend();

    if (level >= SIMDLevel::AVX2) {
        ternary_register_avx2_v1_backend();
        ternary_register_avx2_v2_backend();
    }

    if (level >= SIMDLevel::AVX512BW) {
        ternary_register_avx512_backend();
    }

    // Auto-select best available
    return ternary_backend_select_best();
}
```

---

## Error Handling

### Graceful Degradation

```cpp
void process_trits(const uint8_t* a, const uint8_t* b, uint8_t* result, size_t n) {
    SIMDLevel level = detect_best_simd();

    switch (level) {
        case SIMDLevel::AVX512BW:
            process_avx512(a, b, result, n);
            break;
        case SIMDLevel::AVX2:
            process_avx2(a, b, result, n);
            break;
        default:
            // Always works, just slower
            process_scalar(a, b, result, n);
            break;
    }
}
```

### Runtime Checks

```cpp
void init_simd_operations() {
    if (!has_avx2()) {
        fprintf(stderr, "Warning: AVX2 not available, using scalar fallback\n");
        fprintf(stderr, "Expected throughput: ~100 ME/s (vs ~3500 ME/s with AVX2)\n");
    }
}
```

---

## Platform Notes

### Windows (MSVC)

```cpp
#ifdef _MSC_VER
    #include <intrin.h>
    // Uses __cpuidex() intrinsic
#endif
```

### Linux/macOS (GCC/Clang)

```cpp
#ifndef _MSC_VER
    #include <cpuid.h>
    // Uses __cpuid_count() builtin
#endif
```

### Unsupported Platforms

```cpp
#if !defined(TERNARY_X86) && !defined(TERNARY_ARM64)
// Fallback stubs return false for all features
inline bool has_avx2() { return false; }
inline bool has_avx512f() { return false; }
inline bool has_avx512bw() { return false; }
inline bool has_sve() { return false; }
inline bool has_neon() { return false; }
#endif
```

---

## CPU Compatibility Matrix

### x86-64

| Feature | Intel | AMD |
|---------|-------|-----|
| AVX2 | Haswell (2013) | Excavator (2015) |
| AVX-512F | Skylake-X (2017) | Zen 4 (2022) |
| AVX-512BW | Skylake-X (2017) | Zen 4 (2022) |

### ARM64

| Feature | Apple | Qualcomm | AWS |
|---------|-------|----------|-----|
| NEON | M1+ | Snapdragon 8xx | Graviton 1+ |
| SVE | - | - | Graviton 3+ |

---

## Performance Impact

| Level | Trits/Op | Relative Throughput |
|-------|----------|---------------------|
| Scalar | 1 | 1x (~100 ME/s) |
| AVX2 | 32 | 35x (~3,500 ME/s) |
| AVX-512BW | 64 | 70x (~7,000 ME/s)* |

*AVX-512 throughput is theoretical; actual gains depend on thermal throttling.

---

## Future Work

### Planned Features

1. **Runtime ARM detection** - Linux HWCAP parsing
2. **Thermal-aware selection** - Avoid AVX-512 when throttling
3. **Hybrid CPU support** - P-core vs E-core on Intel Alder Lake
4. **RISC-V Vector** - RVV extension detection

---

**See also:** [Backend System](BACKEND_SYSTEM.md), [SIMD Kernels](SIMD_KERNELS.md)
