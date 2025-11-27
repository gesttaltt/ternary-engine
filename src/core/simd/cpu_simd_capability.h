// cpu_simd_capability.h — Runtime CPU SIMD capability detection (AVX2/AVX-512/NEON/SVE)
//
// Copyright (c) 2025 Jonathan Verdun (Ternary Engine Project)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// =============================================================================
// DESIGN RATIONALE
// =============================================================================
//
// OPT-PHASE3-05: Runtime CPU feature detection enables:
// - Dynamic fallback in hybrid builds (AVX-512 → AVX2 → scalar)
// - Graceful degradation on older CPUs
// - Build-time ISA selection for optimal performance
// - Future-proof architecture for multi-platform SIMD
//
// Uses platform-specific CPUID instructions to query CPU capabilities.
// Compatible with x86-64 (Intel/AMD) and ARM (feature detection via macros).
//
// =============================================================================

#ifndef CPU_SIMD_CAPABILITY_H
#define CPU_SIMD_CAPABILITY_H

#include <stdint.h>

// --- Platform-specific includes ---
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define TERNARY_X86
    #ifdef _MSC_VER
        #include <intrin.h>
    #else
        #include <cpuid.h>
    #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define TERNARY_ARM64
#endif

// --- CPUID bit definitions (x86-64) ---
#ifdef TERNARY_X86
    #define CPUID_BIT_AVX2      (1 << 5)   // EBX, leaf 7, subleaf 0
    #define CPUID_BIT_AVX512F   (1 << 16)  // EBX, leaf 7, subleaf 0
    #define CPUID_BIT_AVX512BW  (1 << 30)  // EBX, leaf 7, subleaf 0
#endif

// =============================================================================
// Runtime Feature Detection Functions
// =============================================================================

#ifdef TERNARY_X86

// --- x86-64 CPUID helper (cross-platform) ---
inline void cpuid(uint32_t leaf, uint32_t subleaf, uint32_t* eax, uint32_t* ebx, uint32_t* ecx, uint32_t* edx) {
#ifdef _MSC_VER
    int regs[4];
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    *eax = regs[0];
    *ebx = regs[1];
    *ecx = regs[2];
    *edx = regs[3];
#else
    __cpuid_count(leaf, subleaf, *eax, *ebx, *ecx, *edx);
#endif
}

// --- AVX2 detection (Intel Haswell 2013+, AMD Excavator 2015+) ---
inline bool has_avx2() {
    uint32_t eax, ebx, ecx, edx;

    // Check maximum CPUID leaf
    cpuid(0, 0, &eax, &ebx, &ecx, &edx);
    if (eax < 7) return false;

    // Query extended features (leaf 7, subleaf 0)
    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    return (ebx & CPUID_BIT_AVX2) != 0;
}

// --- AVX-512 Foundation detection (Intel Skylake-X 2017+) ---
inline bool has_avx512f() {
    uint32_t eax, ebx, ecx, edx;

    // Check maximum CPUID leaf
    cpuid(0, 0, &eax, &ebx, &ecx, &edx);
    if (eax < 7) return false;

    // Query extended features (leaf 7, subleaf 0)
    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    return (ebx & CPUID_BIT_AVX512F) != 0;
}

// --- AVX-512 Byte/Word instructions (Intel Skylake-X 2017+) ---
// Required for 8-bit/16-bit operations (ternary SIMD uses 8-bit)
inline bool has_avx512bw() {
    uint32_t eax, ebx, ecx, edx;

    // Check maximum CPUID leaf
    cpuid(0, 0, &eax, &ebx, &ecx, &edx);
    if (eax < 7) return false;

    // Query extended features (leaf 7, subleaf 0)
    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    return (ebx & CPUID_BIT_AVX512BW) != 0;
}

#endif // TERNARY_X86

// --- ARM SVE detection (ARM v8.2+) ---
#ifdef TERNARY_ARM64
inline bool has_sve() {
    #ifdef __ARM_FEATURE_SVE
        return true;  // Compile-time SVE support
    #else
        return false; // SVE not available
    #endif
}

inline bool has_neon() {
    #ifdef __ARM_NEON
        return true;  // Compile-time NEON support
    #else
        return false; // NEON not available
    #endif
}
#endif // TERNARY_ARM64

// --- Fallback stubs for unsupported platforms ---
#if !defined(TERNARY_X86) && !defined(TERNARY_ARM64)
inline bool has_avx2() { return false; }
inline bool has_avx512f() { return false; }
inline bool has_avx512bw() { return false; }
inline bool has_sve() { return false; }
inline bool has_neon() { return false; }
#endif

// =============================================================================
// Convenience Functions
// =============================================================================

// --- Best available SIMD instruction set ---
enum class SIMDLevel {
    NONE,       // Scalar only
    AVX2,       // 256-bit vectors (32 trits/op)
    AVX512BW,   // 512-bit vectors (64 trits/op)
    NEON,       // ARM 128-bit vectors (16 trits/op)
    SVE         // ARM scalable vectors (variable width)
};

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

// --- Human-readable SIMD level name ---
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

#endif // TERNARY_CPU_DETECT_H
