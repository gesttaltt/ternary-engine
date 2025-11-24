// optimization_config.h — Shared Optimization Constants
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
// PURPOSE
// =============================================================================
//
// Centralized optimization constants shared across all SIMD engines:
// - ternary_simd_engine.cpp (main engine)
// - ternary_simd_engine_fusion.cpp (fusion operations)
// - Future SIMD modules
//
// This header eliminates code duplication and provides a single source of truth
// for tuning parameters.
//
// =============================================================================

#ifndef TERNARY_OPTIMIZATION_CONFIG_H
#define TERNARY_OPTIMIZATION_CONFIG_H

#include <thread>
#include <algorithm>

// =============================================================================
// PLATFORM COMPATIBILITY
// =============================================================================

// MSVC compatibility: ssize_t is not standard C++
#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

// =============================================================================
// OPTIMIZATION THRESHOLDS
// =============================================================================

// OPT-001: OpenMP threshold for large array parallelization
// Arrays >= threshold will use multi-threaded processing
// OPT-PHASE3-01: Adaptive threshold scales with CPU core count
// Formula: 32K elements per thread ensures good load balancing across CPU tiers
// FIX: Clamp hardware_concurrency() to [1, 64] (can return 0 on some VMs)
static const ssize_t OMP_THRESHOLD = 32768 * std::max(1u, std::min(64u, std::thread::hardware_concurrency()));

// OPT-STREAM: Streaming store threshold (arrays exceeding L3 cache size)
// Typical L3: 8-32 MB; use streaming stores for arrays > 1M elements (~1 MB)
// Non-temporal stores reduce cache pollution for memory-bound workloads
static const ssize_t STREAM_THRESHOLD = 1000000;

// OPT-PHASE3-03: Prefetch distance tuning
// Prefetch stride for hiding memory latency (can be tuned per CPU family)
// 512 bytes = 16 × 32-byte cache lines, optimal for Zen 2/4 and Raptor Lake
// Adjust to 256 for older CPUs or 1024 for server-class processors
constexpr int PREFETCH_DIST = 512;

// =============================================================================
// COMPILE-TIME FEATURE FLAGS
// =============================================================================

// OPT-PHASE3-04: Optional compile-time sanitization switch
// Define TERNARY_NO_SANITIZE at compile time to disable input sanitization
// for validated data pipelines (3-5% performance gain)
// Example: c++ -DTERNARY_NO_SANITIZE -O3 ...
#ifdef TERNARY_NO_SANITIZE
constexpr bool SANITIZE = false;
#else
constexpr bool SANITIZE = true;
#endif

// =============================================================================
// ALIGNMENT HELPERS
// =============================================================================

// Check if pointer is 32-byte aligned for AVX2 streaming stores
// Streaming stores (_mm256_stream_si256) require 32-byte alignment
// NumPy arrays do not guarantee this, so runtime check is mandatory
inline bool is_aligned_32(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) % 32) == 0;
}

// Check if pointer is 64-byte aligned for AVX-512 operations
inline bool is_aligned_64(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) % 64) == 0;
}

// =============================================================================
// TUNING NOTES
// =============================================================================
//
// These constants are tuned for:
// - Intel Haswell/Skylake/Raptor Lake (AVX2)
// - AMD Zen 2/3/4 (AVX2)
// - Typical consumer/workstation L3 cache sizes (8-32 MB)
//
// For different hardware profiles, consider:
//
// **High-core-count servers (32+ cores):**
//   - OMP_THRESHOLD: 16384 (half current value for finer parallelism)
//   - PREFETCH_DIST: 1024 (larger prefetch for high-bandwidth memory)
//
// **Low-power embedded (Atom, mobile):**
//   - OMP_THRESHOLD: 65536 (double current value, reduce threading overhead)
//   - PREFETCH_DIST: 256 (smaller prefetch for limited cache)
//   - STREAM_THRESHOLD: 500000 (smaller L3 caches)
//
// **AVX-512 systems:**
//   - OMP_THRESHOLD: 16384 (64 trits/op means less work per iteration)
//   - PREFETCH_DIST: 1024 (higher bandwidth utilization)
//
// To override defaults, define these before including this header:
// #define TERNARY_OMP_THRESHOLD 16384
// #define TERNARY_STREAM_THRESHOLD 2000000
// #define TERNARY_PREFETCH_DIST 1024
//
// =============================================================================

#endif // TERNARY_OPTIMIZATION_CONFIG_H
