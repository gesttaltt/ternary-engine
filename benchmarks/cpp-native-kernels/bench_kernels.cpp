// bench_kernels.cpp — Kernel-level microbenchmarks (pure C++, no Python overhead)
//
// Copyright (c) 2025 Jonathan Verdun (Ternary Core Experimental Project)
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
// OPT-PHASE3-07: Advanced microbenchmarking enables:
// - Direct C++ vs C++ performance comparison (bypasses Python/NumPy overhead)
// - High-resolution timing via std::chrono::steady_clock
// - CSV/JSON output for automated analysis and CI integration
// - Comparison with xsimd, Eigen, and other SIMD libraries
//
// COMPILATION:
//   # From benchmarks/cpp-native-kernels/ directory:
//   g++ -O3 -march=native -mavx2 -std=c++17 -I../../src bench_kernels.cpp -o bench_kernels
//   clang++ -O3 -march=native -mavx2 -std=c++17 -I../../src bench_kernels.cpp -o bench_kernels
//
//   # Windows (MSVC):
//   cl /O2 /arch:AVX2 /std:c++17 /EHsc /I..\..\src bench_kernels.cpp
//
// USAGE:
//   ./bench_kernels                    # Run all benchmarks
//   ./bench_kernels --csv              # Output CSV format
//   ./bench_kernels --json             # Output JSON format
//   ./bench_kernels --sizes=1000,10000 # Benchmark specific sizes
//
// TARGET: src/core/ - Production SIMD kernels (validated, stable)
//
// =============================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <random>
#include <string>
#include <cmath>

// Core kernel includes (src/core/)
#include "core/simd/simd_avx2_32trit_ops.h"
#include "core/algebra/ternary_algebra.h"
#include "core/simd/cpu_simd_capability.h"

// =============================================================================
// Benchmark Configuration
// =============================================================================

struct BenchConfig {
    std::vector<size_t> sizes = {32, 1000, 10000, 100000, 1000000, 10000000};
    size_t iterations = 1000;     // Number of iterations per benchmark
    size_t warmup_iters = 100;    // Warmup iterations (not measured)
    bool output_csv = false;
    bool output_json = false;
};

// =============================================================================
// Utilities
// =============================================================================

// --- Random trit generation ---
std::vector<uint8_t> generate_random_trits(size_t n, uint32_t seed = 42) {
    std::vector<uint8_t> data(n);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, 2);

    for (size_t i = 0; i < n; ++i) {
        int val = dis(gen);
        data[i] = (val == 0) ? 0b00 : (val == 1) ? 0b01 : 0b10;
    }

    return data;
}

// --- Throughput calculation (Million Elements per second) ---
double calculate_throughput(size_t n, size_t iterations, double elapsed_ns) {
    return (n * iterations) / (elapsed_ns / 1e9) / 1e6;  // ME/s
}

// =============================================================================
// Benchmark: SIMD Binary Operations
// =============================================================================

template <typename SimdOp>
double bench_simd_binary(const char* op_name, SimdOp simd_op,
                          const std::vector<uint8_t>& A,
                          const std::vector<uint8_t>& B,
                          size_t iterations) {
    size_t n = A.size();
    std::vector<uint8_t> R(n);

    // Warmup (not measured)
    for (size_t iter = 0; iter < 100; ++iter) {
        for (size_t i = 0; i + 32 <= n; i += 32) {
            __m256i va = _mm256_loadu_si256((const __m256i*)(A.data() + i));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(B.data() + i));
            __m256i vr = simd_op(va, vb);
            _mm256_storeu_si256((__m256i*)(R.data() + i), vr);
        }
    }

    // Actual benchmark
    auto start = std::chrono::steady_clock::now();
    for (size_t iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i + 32 <= n; i += 32) {
            __m256i va = _mm256_loadu_si256((const __m256i*)(A.data() + i));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(B.data() + i));
            __m256i vr = simd_op(va, vb);
            _mm256_storeu_si256((__m256i*)(R.data() + i), vr);
        }
    }
    auto end = std::chrono::steady_clock::now();

    auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    return calculate_throughput(n, iterations, static_cast<double>(elapsed_ns));
}

// =============================================================================
// Benchmark: Scalar Operations
// =============================================================================

template <typename ScalarOp>
double bench_scalar_binary(const char* op_name, ScalarOp scalar_op,
                            const std::vector<uint8_t>& A,
                            const std::vector<uint8_t>& B,
                            size_t iterations) {
    size_t n = A.size();
    std::vector<uint8_t> R(n);

    // Warmup
    for (size_t iter = 0; iter < 100; ++iter) {
        for (size_t i = 0; i < n; ++i) {
            R[i] = scalar_op(A[i], B[i]);
        }
    }

    // Actual benchmark
    auto start = std::chrono::steady_clock::now();
    for (size_t iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < n; ++i) {
            R[i] = scalar_op(A[i], B[i]);
        }
    }
    auto end = std::chrono::steady_clock::now();

    auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    return calculate_throughput(n, iterations, static_cast<double>(elapsed_ns));
}

// =============================================================================
// Output Formatters
// =============================================================================

void print_header() {
    std::cout << "=============================================================================\n";
    std::cout << "Ternary SIMD Kernel Microbenchmarks (Phase 3)\n";
    std::cout << "=============================================================================\n\n";

    // System info
    SIMDLevel level = detect_best_simd();
    std::cout << "CPU SIMD Support: " << simd_level_name(level) << "\n";
    std::cout << "Compiler: ";
#ifdef __clang__
    std::cout << "Clang " << __clang_major__ << "." << __clang_minor__ << "\n";
#elif defined(__GNUC__)
    std::cout << "GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "\n";
#elif defined(_MSC_VER)
    std::cout << "MSVC " << _MSC_VER << "\n";
#else
    std::cout << "Unknown\n";
#endif
    std::cout << "\n";
}

void print_result(const char* operation, size_t n, double throughput_simd, double throughput_scalar) {
    double speedup = throughput_simd / throughput_scalar;
    std::cout << std::setw(8) << operation
              << " | N=" << std::setw(10) << n
              << " | SIMD: " << std::setw(8) << std::fixed << std::setprecision(2) << throughput_simd << " ME/s"
              << " | Scalar: " << std::setw(8) << throughput_scalar << " ME/s"
              << " | Speedup: " << std::setw(5) << std::setprecision(2) << speedup << "×\n";
}

void print_csv_header() {
    std::cout << "operation,size,throughput_simd_ME_s,throughput_scalar_ME_s,speedup\n";
}

void print_csv_result(const char* operation, size_t n, double throughput_simd, double throughput_scalar) {
    double speedup = throughput_simd / throughput_scalar;
    std::cout << operation << "," << n << ","
              << throughput_simd << "," << throughput_scalar << ","
              << speedup << "\n";
}

// =============================================================================
// Main Benchmark Runner
// =============================================================================

int main(int argc, char** argv) {
    BenchConfig config;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--csv") {
            config.output_csv = true;
        } else if (arg == "--json") {
            config.output_json = true;
        } else if (arg.rfind("--sizes=", 0) == 0) {
            // TODO: Parse custom sizes
        }
    }

    if (!config.output_csv && !config.output_json) {
        print_header();
    }

    if (config.output_csv) {
        print_csv_header();
    }

    // Run benchmarks for each size
    for (size_t n : config.sizes) {
        auto A = generate_random_trits(n, 42);
        auto B = generate_random_trits(n, 43);

        // Benchmark TADD
        double throughput_simd = bench_simd_binary("tadd", tadd_simd<true>, A, B, config.iterations);
        double throughput_scalar = bench_scalar_binary("tadd", tadd, A, B, config.iterations);

        if (config.output_csv) {
            print_csv_result("tadd", n, throughput_simd, throughput_scalar);
        } else {
            print_result("tadd", n, throughput_simd, throughput_scalar);
        }

        // Benchmark TMUL
        throughput_simd = bench_simd_binary("tmul", tmul_simd<true>, A, B, config.iterations);
        throughput_scalar = bench_scalar_binary("tmul", tmul, A, B, config.iterations);

        if (config.output_csv) {
            print_csv_result("tmul", n, throughput_simd, throughput_scalar);
        } else {
            print_result("tmul", n, throughput_simd, throughput_scalar);
        }

        // Benchmark TMIN
        throughput_simd = bench_simd_binary("tmin", tmin_simd<true>, A, B, config.iterations);
        throughput_scalar = bench_scalar_binary("tmin", tmin, A, B, config.iterations);

        if (config.output_csv) {
            print_csv_result("tmin", n, throughput_simd, throughput_scalar);
        } else {
            print_result("tmin", n, throughput_simd, throughput_scalar);
        }

        // Benchmark TMAX
        throughput_simd = bench_simd_binary("tmax", tmax_simd<true>, A, B, config.iterations);
        throughput_scalar = bench_scalar_binary("tmax", tmax, A, B, config.iterations);

        if (config.output_csv) {
            print_csv_result("tmax", n, throughput_simd, throughput_scalar);
        } else {
            print_result("tmax", n, throughput_simd, throughput_scalar);
        }

        if (!config.output_csv && !config.output_json) {
            std::cout << "\n";
        }
    }

    if (!config.output_csv && !config.output_json) {
        std::cout << "=============================================================================\n";
        std::cout << "Benchmark complete.\n";
    }

    return 0;
}
