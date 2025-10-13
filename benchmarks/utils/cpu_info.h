// cpu_info.h — CPU information utilities for benchmarking
//
// Copyright (c) 2025 Jonathan Verdun (Ternary Core Experimental Project)
// Licensed under the Apache License, Version 2.0

#pragma once

#include <string>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace cpu_info {

/**
 * Get the number of hardware threads available
 */
inline int hardware_threads() {
    return std::thread::hardware_concurrency();
}

/**
 * Get CPU vendor string (e.g., "GenuineIntel", "AuthenticAMD")
 */
inline std::string vendor() {
    int cpu_info[4] = {0};
    char vendor_str[13] = {0};

#ifdef _WIN32
    __cpuid(cpu_info, 0);
#else
    __cpuid(0, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
#endif

    *reinterpret_cast<int*>(vendor_str) = cpu_info[1];
    *reinterpret_cast<int*>(vendor_str + 4) = cpu_info[3];
    *reinterpret_cast<int*>(vendor_str + 8) = cpu_info[2];

    return std::string(vendor_str);
}

/**
 * Check if AVX2 is supported
 */
inline bool has_avx2() {
    int cpu_info[4] = {0};

#ifdef _WIN32
    __cpuidex(cpu_info, 7, 0);
#else
    __cpuid_count(7, 0, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
#endif

    return (cpu_info[1] & (1 << 5)) != 0;  // EBX bit 5
}

/**
 * Check if AVX-512 is supported
 */
inline bool has_avx512() {
    int cpu_info[4] = {0};

#ifdef _WIN32
    __cpuidex(cpu_info, 7, 0);
#else
    __cpuid_count(7, 0, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
#endif

    return (cpu_info[1] & (1 << 16)) != 0;  // EBX bit 16 (AVX-512F)
}

/**
 * Get a human-readable CPU info string
 */
inline std::string summary() {
    std::string result = vendor();
    result += " | " + std::to_string(hardware_threads()) + " threads";
    if (has_avx512()) result += " | AVX-512";
    else if (has_avx2()) result += " | AVX2";
    return result;
}

} // namespace cpu_info
