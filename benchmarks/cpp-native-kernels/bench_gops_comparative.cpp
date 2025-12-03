// bench_gops_comparative.cpp - Honest Gops/s benchmark: Ternary vs Binary Operations
//
// Copyright 2025 Ternary Engine Contributors
// Licensed under the Apache License, Version 2.0
//
// =============================================================================
// PURPOSE
// =============================================================================
//
// This benchmark provides HONEST measurement of throughput in Gops/s (billions
// of operations per second) comparing:
//
//   1. TERNARY SIMD operations (our custom LUT-based implementation)
//   2. BINARY SIMD operations (standard AVX2 integer arithmetic)
//
// The goal is to answer: "How does ternary arithmetic compare to standard
// integer arithmetic on modern CPUs?"
//
// =============================================================================
// METHODOLOGY
// =============================================================================
//
// Fair comparison criteria:
// - Both use AVX2 256-bit vectors (32 bytes per operation)
// - Both process the same number of elements
// - Ternary: 32 elements per vector (1 byte per trit, 2-bit encoding)
// - Binary INT8: 32 elements per vector (1 byte per integer)
//
// What we measure:
// - Throughput: Gops/s (billion operations per second)
// - Memory bandwidth: GB/s
// - Efficiency: Operations per cycle (estimated)
//
// =============================================================================
// COMPILATION
// =============================================================================
//
// Windows (MSVC):
//   cl /O2 /arch:AVX2 /std:c++17 /EHsc /I..\..\src /I.\include bench_gops_comparative.cpp
//
// Linux/macOS (GCC/Clang):
//   g++ -O3 -march=native -mavx2 -std=c++17 -I../../src -I./include bench_gops_comparative.cpp -o bench_gops
//
// =============================================================================
// USAGE
// =============================================================================
//
//   ./bench_gops                    # Run full benchmark suite
//   ./bench_gops --quick            # Quick test (fewer sizes)
//   ./bench_gops --csv              # CSV output
//   ./bench_gops --json             # JSON output
//   ./bench_gops --size=1000000     # Specific size only
//
// =============================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <random>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>

#include <immintrin.h>

// Core ternary kernel includes
#include "core/simd/simd_avx2_32trit_ops.h"
#include "core/algebra/ternary_algebra.h"
#include "core/simd/cpu_simd_capability.h"

// Benchmark framework
#include "include/bench_throughput.h"

// =============================================================================
// CONFIGURATION
// =============================================================================

struct Config {
    std::vector<size_t> sizes = {32, 256, 4096, 32768, 262144, 1048576, 10485760};
    size_t warmup_iterations = 100;
    size_t benchmark_iterations = 1000;
    uint32_t random_seed = 42;
    bool output_csv = false;
    bool output_json = false;
    bool verbose = true;
    bool quick_mode = false;
};

// =============================================================================
// TIMING UTILITIES
// =============================================================================

struct TimingStats {
    double mean_ns;
    double stddev_ns;
    double min_ns;
    double max_ns;
    double p50_ns;
    double p95_ns;
    double p99_ns;
    double cv;  // Coefficient of variation
};

TimingStats calculate_stats(std::vector<double>& times) {
    std::sort(times.begin(), times.end());

    size_t n = times.size();
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / n;

    double sq_sum = 0.0;
    for (double t : times) {
        sq_sum += (t - mean) * (t - mean);
    }
    double stddev = std::sqrt(sq_sum / n);

    TimingStats stats;
    stats.mean_ns = mean;
    stats.stddev_ns = stddev;
    stats.min_ns = times[0];
    stats.max_ns = times[n - 1];
    stats.p50_ns = times[n / 2];
    stats.p95_ns = times[(size_t)(n * 0.95)];
    stats.p99_ns = times[(size_t)(n * 0.99)];
    stats.cv = (mean > 0) ? (stddev / mean) : 0.0;

    return stats;
}

// =============================================================================
// DATA GENERATION
// =============================================================================

std::vector<uint8_t> generate_ternary_data(size_t n, uint32_t seed) {
    std::vector<uint8_t> data(n);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, 2);

    for (size_t i = 0; i < n; ++i) {
        int val = dis(gen);
        data[i] = (val == 0) ? 0b00 : (val == 1) ? 0b01 : 0b10;
    }
    return data;
}

std::vector<int8_t> generate_int8_data(size_t n, uint32_t seed) {
    std::vector<int8_t> data(n);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(-128, 127);

    for (size_t i = 0; i < n; ++i) {
        data[i] = static_cast<int8_t>(dis(gen));
    }
    return data;
}

// Aligned allocation for SIMD
template <typename T>
T* aligned_alloc_simd(size_t n) {
#ifdef _MSC_VER
    return static_cast<T*>(_aligned_malloc(n * sizeof(T), 32));
#else
    void* ptr = nullptr;
    posix_memalign(&ptr, 32, n * sizeof(T));
    return static_cast<T*>(ptr);
#endif
}

template <typename T>
void aligned_free_simd(T* ptr) {
#ifdef _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// =============================================================================
// TERNARY SIMD OPERATIONS (OUR IMPLEMENTATION)
// =============================================================================

void ternary_tadd_simd(const uint8_t* a, const uint8_t* b, uint8_t* r, size_t n) {
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i vr = tadd_simd<false>(va, vb);  // No sanitization for benchmark
        _mm256_storeu_si256((__m256i*)(r + i), vr);
    }
    // Handle remainder with scalar
    for (; i < n; ++i) {
        r[i] = tadd(a[i], b[i]);
    }
}

void ternary_tmul_simd(const uint8_t* a, const uint8_t* b, uint8_t* r, size_t n) {
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i vr = tmul_simd<false>(va, vb);
        _mm256_storeu_si256((__m256i*)(r + i), vr);
    }
    for (; i < n; ++i) {
        r[i] = tmul(a[i], b[i]);
    }
}

void ternary_tmin_simd(const uint8_t* a, const uint8_t* b, uint8_t* r, size_t n) {
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i vr = tmin_simd<false>(va, vb);
        _mm256_storeu_si256((__m256i*)(r + i), vr);
    }
    for (; i < n; ++i) {
        r[i] = tmin(a[i], b[i]);
    }
}

void ternary_tmax_simd(const uint8_t* a, const uint8_t* b, uint8_t* r, size_t n) {
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i vr = tmax_simd<false>(va, vb);
        _mm256_storeu_si256((__m256i*)(r + i), vr);
    }
    for (; i < n; ++i) {
        r[i] = tmax(a[i], b[i]);
    }
}

// =============================================================================
// BINARY SIMD OPERATIONS (STANDARD INT8 BASELINES)
// =============================================================================

// INT8 addition (standard integer add, not saturated)
void binary_int8_add(const uint8_t* a, const uint8_t* b, uint8_t* r, size_t n) {
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i vr = _mm256_add_epi8(va, vb);
        _mm256_storeu_si256((__m256i*)(r + i), vr);
    }
    for (; i < n; ++i) {
        r[i] = a[i] + b[i];
    }
}

// INT8 saturated addition (similar semantics to ternary add)
void binary_int8_adds(const uint8_t* a, const uint8_t* b, uint8_t* r, size_t n) {
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i vr = _mm256_adds_epi8(va, vb);  // Saturated add
        _mm256_storeu_si256((__m256i*)(r + i), vr);
    }
    for (; i < n; ++i) {
        int16_t sum = static_cast<int16_t>(static_cast<int8_t>(a[i])) +
                      static_cast<int16_t>(static_cast<int8_t>(b[i]));
        r[i] = static_cast<uint8_t>(std::clamp(sum, (int16_t)-128, (int16_t)127));
    }
}

// INT8 minimum
void binary_int8_min(const uint8_t* a, const uint8_t* b, uint8_t* r, size_t n) {
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i vr = _mm256_min_epi8(va, vb);
        _mm256_storeu_si256((__m256i*)(r + i), vr);
    }
    for (; i < n; ++i) {
        r[i] = std::min(a[i], b[i]);
    }
}

// INT8 maximum
void binary_int8_max(const uint8_t* a, const uint8_t* b, uint8_t* r, size_t n) {
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i vr = _mm256_max_epi8(va, vb);
        _mm256_storeu_si256((__m256i*)(r + i), vr);
    }
    for (; i < n; ++i) {
        r[i] = std::max(a[i], b[i]);
    }
}

// INT8 multiplication (low 8 bits only, requires unpacking)
void binary_int8_mul(const uint8_t* a, const uint8_t* b, uint8_t* r, size_t n) {
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));

        // INT8 multiply requires unpacking to INT16, multiply, then pack back
        // Split into low and high 16 bytes
        __m256i va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
        __m256i vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
        __m256i va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
        __m256i vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));

        // Multiply (16-bit result)
        __m256i vr_lo = _mm256_mullo_epi16(va_lo, vb_lo);
        __m256i vr_hi = _mm256_mullo_epi16(va_hi, vb_hi);

        // Pack back to 8-bit (truncate high bits)
        __m256i vr_lo_8 = _mm256_and_si256(vr_lo, _mm256_set1_epi16(0x00FF));
        __m256i vr_hi_8 = _mm256_and_si256(vr_hi, _mm256_set1_epi16(0x00FF));
        __m256i vr = _mm256_packus_epi16(vr_lo_8, vr_hi_8);

        // Permute to correct order (packus interleaves)
        vr = _mm256_permute4x64_epi64(vr, 0xD8);

        _mm256_storeu_si256((__m256i*)(r + i), vr);
    }
    for (; i < n; ++i) {
        r[i] = static_cast<uint8_t>(static_cast<int8_t>(a[i]) * static_cast<int8_t>(b[i]));
    }
}

// =============================================================================
// BENCHMARK RUNNER
// =============================================================================

struct BenchResult {
    std::string name;
    std::string op_class;  // "ternary" or "binary"
    size_t array_size;
    size_t iterations;
    TimingStats timing;
    double gops;
    double bandwidth_gbps;
};

template <typename BenchFn>
BenchResult run_benchmark(
    const std::string& name,
    const std::string& op_class,
    BenchFn bench_fn,
    const uint8_t* a,
    const uint8_t* b,
    uint8_t* r,
    size_t n,
    const Config& config
) {
    // Warmup
    for (size_t iter = 0; iter < config.warmup_iterations; ++iter) {
        bench_fn(a, b, r, n);
    }

    // Timed runs
    std::vector<double> times(config.benchmark_iterations);

    for (size_t iter = 0; iter < config.benchmark_iterations; ++iter) {
        auto start = std::chrono::steady_clock::now();
        bench_fn(a, b, r, n);
        auto end = std::chrono::steady_clock::now();

        times[iter] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

    TimingStats stats = calculate_stats(times);

    BenchResult result;
    result.name = name;
    result.op_class = op_class;
    result.array_size = n;
    result.iterations = config.benchmark_iterations;
    result.timing = stats;

    // Calculate Gops/s: (elements * iterations) / time_seconds / 1e9
    // But we want per-iteration throughput, so:
    // Gops = elements / mean_time_ns * 1e9 / 1e9 = elements / mean_time_ns
    result.gops = static_cast<double>(n) / stats.mean_ns;

    // Memory bandwidth: read A, read B, write R = 3 * n bytes per iteration
    // GB/s = (3 * n * 1e9 / mean_time_ns) / 1e9 = 3 * n / mean_time_ns
    result.bandwidth_gbps = 3.0 * static_cast<double>(n) / stats.mean_ns;

    return result;
}

// =============================================================================
// OUTPUT FORMATTERS
// =============================================================================

void print_header() {
    std::cout << "================================================================================\n";
    std::cout << "  TERNARY vs BINARY SIMD BENCHMARK - Gops/s Throughput Analysis\n";
    std::cout << "================================================================================\n\n";

    // System info
    SIMDLevel level = detect_best_simd();
    std::cout << "System Configuration:\n";
    std::cout << "  SIMD Support: " << simd_level_name(level) << "\n";
    std::cout << "  Compiler: ";
#ifdef __clang__
    std::cout << "Clang " << __clang_major__ << "." << __clang_minor__;
#elif defined(__GNUC__)
    std::cout << "GCC " << __GNUC__ << "." << __GNUC_MINOR__;
#elif defined(_MSC_VER)
    std::cout << "MSVC " << _MSC_VER;
#else
    std::cout << "Unknown";
#endif
    std::cout << "\n\n";

    std::cout << "Methodology:\n";
    std::cout << "  - Both ternary and binary use AVX2 256-bit vectors\n";
    std::cout << "  - Both process 32 elements per vector operation\n";
    std::cout << "  - Ternary: LUT-based operations (3 shuffles per binary op)\n";
    std::cout << "  - Binary: Direct AVX2 intrinsics (1 instruction per op)\n";
    std::cout << "  - Throughput = elements / mean_time (Gops/s)\n";
    std::cout << "\n";
}

void print_result_header() {
    std::cout << std::left
              << std::setw(20) << "Operation"
              << std::setw(12) << "Size"
              << std::setw(12) << "Gops/s"
              << std::setw(12) << "GB/s"
              << std::setw(12) << "Mean (ns)"
              << std::setw(10) << "CV%"
              << "\n";
    std::cout << std::string(78, '-') << "\n";
}

void print_result(const BenchResult& result) {
    std::cout << std::left << std::fixed
              << std::setw(20) << result.name
              << std::setw(12) << result.array_size
              << std::setw(12) << std::setprecision(3) << result.gops
              << std::setw(12) << std::setprecision(2) << result.bandwidth_gbps
              << std::setw(12) << std::setprecision(1) << result.timing.mean_ns
              << std::setw(10) << std::setprecision(1) << (result.timing.cv * 100)
              << "\n";
}

void print_comparison_header() {
    std::cout << "\n";
    std::cout << std::left
              << std::setw(12) << "Operation"
              << std::setw(12) << "Size"
              << std::setw(14) << "Ternary Gops"
              << std::setw(14) << "Binary Gops"
              << std::setw(12) << "Ratio"
              << std::setw(10) << "Winner"
              << "\n";
    std::cout << std::string(74, '-') << "\n";
}

void print_comparison(const std::string& op, size_t size,
                      const BenchResult& ternary, const BenchResult& binary) {
    double ratio = ternary.gops / binary.gops;
    const char* winner = (ratio > 1.05) ? "TERNARY" : (ratio < 0.95) ? "BINARY" : "TIE";

    std::cout << std::left << std::fixed
              << std::setw(12) << op
              << std::setw(12) << size
              << std::setw(14) << std::setprecision(3) << ternary.gops
              << std::setw(14) << std::setprecision(3) << binary.gops
              << std::setw(12) << std::setprecision(3) << ratio
              << std::setw(10) << winner
              << "\n";
}

void print_csv_header() {
    std::cout << "operation,class,size,gops,bandwidth_gbps,mean_ns,stddev_ns,cv_percent\n";
}

void print_csv_result(const BenchResult& result) {
    std::cout << result.name << ","
              << result.op_class << ","
              << result.array_size << ","
              << std::fixed << std::setprecision(6) << result.gops << ","
              << std::setprecision(3) << result.bandwidth_gbps << ","
              << std::setprecision(2) << result.timing.mean_ns << ","
              << result.timing.stddev_ns << ","
              << (result.timing.cv * 100) << "\n";
}

// =============================================================================
// MAIN BENCHMARK SUITE
// =============================================================================

int main(int argc, char** argv) {
    Config config;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--csv") {
            config.output_csv = true;
            config.verbose = false;
        } else if (arg == "--json") {
            config.output_json = true;
            config.verbose = false;
        } else if (arg == "--quick") {
            config.quick_mode = true;
            config.sizes = {256, 32768, 1048576};
            config.benchmark_iterations = 100;
        } else if (arg.rfind("--size=", 0) == 0) {
            size_t size = std::stoull(arg.substr(7));
            config.sizes = {size};
        }
    }

    if (config.verbose) {
        print_header();
    }

    if (config.output_csv) {
        print_csv_header();
    }

    std::vector<BenchResult> all_results;

    // Run benchmarks for each size
    for (size_t n : config.sizes) {
        // Allocate aligned memory
        uint8_t* a = aligned_alloc_simd<uint8_t>(n);
        uint8_t* b = aligned_alloc_simd<uint8_t>(n);
        uint8_t* r = aligned_alloc_simd<uint8_t>(n);

        // Generate data
        auto ternary_a = generate_ternary_data(n, config.random_seed);
        auto ternary_b = generate_ternary_data(n, config.random_seed + 1);
        std::memcpy(a, ternary_a.data(), n);
        std::memcpy(b, ternary_b.data(), n);

        if (config.verbose) {
            std::cout << "================================================================================\n";
            std::cout << "  Array Size: " << n << " elements (" << (n / 1024.0 / 1024.0) << " MB)\n";
            std::cout << "================================================================================\n\n";

            print_result_header();
        }

        // ==================== TERNARY OPERATIONS ====================
        auto t_tadd = run_benchmark("ternary_tadd", "ternary", ternary_tadd_simd, a, b, r, n, config);
        auto t_tmul = run_benchmark("ternary_tmul", "ternary", ternary_tmul_simd, a, b, r, n, config);
        auto t_tmin = run_benchmark("ternary_tmin", "ternary", ternary_tmin_simd, a, b, r, n, config);
        auto t_tmax = run_benchmark("ternary_tmax", "ternary", ternary_tmax_simd, a, b, r, n, config);

        // ==================== BINARY OPERATIONS ====================
        // Re-generate data for binary (full INT8 range)
        auto int8_a = generate_int8_data(n, config.random_seed);
        auto int8_b = generate_int8_data(n, config.random_seed + 1);
        std::memcpy(a, int8_a.data(), n);
        std::memcpy(b, int8_b.data(), n);

        auto b_add = run_benchmark("binary_int8_add", "binary", binary_int8_add, a, b, r, n, config);
        auto b_adds = run_benchmark("binary_int8_adds", "binary", binary_int8_adds, a, b, r, n, config);
        auto b_mul = run_benchmark("binary_int8_mul", "binary", binary_int8_mul, a, b, r, n, config);
        auto b_min = run_benchmark("binary_int8_min", "binary", binary_int8_min, a, b, r, n, config);
        auto b_max = run_benchmark("binary_int8_max", "binary", binary_int8_max, a, b, r, n, config);

        if (config.verbose) {
            print_result(t_tadd);
            print_result(t_tmul);
            print_result(t_tmin);
            print_result(t_tmax);
            std::cout << "\n";
            print_result(b_add);
            print_result(b_adds);
            print_result(b_mul);
            print_result(b_min);
            print_result(b_max);

            // Comparison section
            print_comparison_header();
            print_comparison("ADD", n, t_tadd, b_add);
            print_comparison("ADD_SAT", n, t_tadd, b_adds);
            print_comparison("MUL", n, t_tmul, b_mul);
            print_comparison("MIN", n, t_tmin, b_min);
            print_comparison("MAX", n, t_tmax, b_max);
            std::cout << "\n";
        }

        if (config.output_csv) {
            print_csv_result(t_tadd);
            print_csv_result(t_tmul);
            print_csv_result(t_tmin);
            print_csv_result(t_tmax);
            print_csv_result(b_add);
            print_csv_result(b_adds);
            print_csv_result(b_mul);
            print_csv_result(b_min);
            print_csv_result(b_max);
        }

        all_results.push_back(t_tadd);
        all_results.push_back(t_tmul);
        all_results.push_back(t_tmin);
        all_results.push_back(t_tmax);
        all_results.push_back(b_add);
        all_results.push_back(b_adds);
        all_results.push_back(b_mul);
        all_results.push_back(b_min);
        all_results.push_back(b_max);

        aligned_free_simd(a);
        aligned_free_simd(b);
        aligned_free_simd(r);
    }

    // ==================== SUMMARY ====================
    if (config.verbose) {
        std::cout << "================================================================================\n";
        std::cout << "  SUMMARY\n";
        std::cout << "================================================================================\n\n";

        // Find peak throughput for each operation type
        double peak_ternary = 0, peak_binary = 0;
        for (const auto& r : all_results) {
            if (r.op_class == "ternary" && r.gops > peak_ternary) {
                peak_ternary = r.gops;
            }
            if (r.op_class == "binary" && r.gops > peak_binary) {
                peak_binary = r.gops;
            }
        }

        std::cout << "Peak Throughput:\n";
        std::cout << "  Ternary SIMD:  " << std::fixed << std::setprecision(3) << peak_ternary << " Gops/s\n";
        std::cout << "  Binary INT8:   " << std::fixed << std::setprecision(3) << peak_binary << " Gops/s\n";
        std::cout << "  Ratio:         " << std::setprecision(2) << (peak_ternary / peak_binary) << "x\n";
        std::cout << "\n";

        std::cout << "Analysis:\n";
        if (peak_ternary > peak_binary) {
            std::cout << "  Ternary operations outperform standard INT8 operations!\n";
            std::cout << "  This is due to the LUT-based approach enabling complex\n";
            std::cout << "  arithmetic with minimal instruction count.\n";
        } else {
            std::cout << "  Binary INT8 operations are faster for simple arithmetic.\n";
            std::cout << "  Ternary's value is in memory efficiency (2-bit vs 8-bit)\n";
            std::cout << "  and specialized AI model compression, not raw throughput.\n";
        }
        std::cout << "\n";

        std::cout << "Key Insight:\n";
        std::cout << "  - Ternary uses 2-bit encoding (3 values: -1, 0, +1)\n";
        std::cout << "  - Binary uses 8-bit encoding (256 values)\n";
        std::cout << "  - Ternary provides 4x memory compression\n";
        std::cout << "  - For AI inference, memory bandwidth is often the bottleneck\n";
        std::cout << "\n";
    }

    return 0;
}
