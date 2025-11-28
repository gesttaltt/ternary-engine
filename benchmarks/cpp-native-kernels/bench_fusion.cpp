// bench_fusion.cpp — Native C++ benchmark for fused operations (Phase 4.0/4.1)
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
// This benchmark measures the performance of fused SIMD operations compared to
// separate operations. Fusion reduces memory bandwidth by combining operations
// that share intermediate results.
//
// Validated performance (2025-10-29):
//   - fused_tnot_tadd: 1.62-1.95× speedup
//   - fused_tnot_tmul: 1.53-1.86× speedup
//   - fused_tnot_tmin: 1.61-11.26× speedup
//   - fused_tnot_tmax: 1.65-9.50× speedup
//   - Average: 2.80× speedup
//
// COMPILATION (from benchmarks/cpp-native-kernels/ directory):
//   g++ -O3 -march=native -mavx2 -std=c++17 -I../../src bench_fusion.cpp -o bench_fusion
//   clang++ -O3 -march=native -mavx2 -std=c++17 -I../../src bench_fusion.cpp -o bench_fusion
//
//   # Windows (MSVC):
//   cl /O2 /arch:AVX2 /std:c++17 /EHsc /I..\..\src bench_fusion.cpp
//
// USAGE:
//   ./bench_fusion              # Run all benchmarks
//   ./bench_fusion --csv        # Output CSV format
//
// TARGET: src/core/simd/fused_binary_unary_ops.h - Fused SIMD operations (validated)
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

// Core kernel includes (src/core/)
#include "core/simd/fused_binary_unary_ops.h"
#include "core/simd/simd_avx2_32trit_ops.h"
#include "core/algebra/ternary_algebra.h"

using clock_type = std::chrono::steady_clock;

// =============================================================================
// Configuration
// =============================================================================

struct BenchConfig {
    std::vector<size_t> sizes = {1000, 10000, 100000, 1000000};
    size_t iterations = 100;
    size_t warmup_iters = 20;
    bool output_csv = false;
};

// =============================================================================
// Utilities
// =============================================================================

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

struct BenchStats {
    double mean;
    double min;
    double max;
    double cv;  // Coefficient of variation
};

BenchStats calculate_stats(const std::vector<double>& times) {
    BenchStats stats;
    stats.mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    stats.min = *std::min_element(times.begin(), times.end());
    stats.max = *std::max_element(times.begin(), times.end());

    double sq_sum = 0.0;
    for (double t : times) {
        sq_sum += (t - stats.mean) * (t - stats.mean);
    }
    double stddev = std::sqrt(sq_sum / times.size());
    stats.cv = (stats.mean > 0) ? (stddev / stats.mean * 100.0) : 0.0;

    return stats;
}

// =============================================================================
// Benchmark: Unfused operations (separate passes)
// =============================================================================

template <typename BinaryOp, typename UnaryOp>
BenchStats bench_unfused(
    const std::vector<uint8_t>& A,
    const std::vector<uint8_t>& B,
    BinaryOp binary_op,
    UnaryOp unary_op,
    size_t iterations,
    size_t warmup
) {
    size_t n = A.size();
    std::vector<uint8_t> temp(n);
    std::vector<uint8_t> result(n);
    std::vector<double> times;
    times.reserve(iterations);

    // Warmup
    for (size_t w = 0; w < warmup; ++w) {
        // First pass: binary operation
        for (size_t i = 0; i + 32 <= n; i += 32) {
            __m256i va = _mm256_loadu_si256((const __m256i*)(A.data() + i));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(B.data() + i));
            __m256i vr = binary_op(va, vb);
            _mm256_storeu_si256((__m256i*)(temp.data() + i), vr);
        }
        // Second pass: unary operation
        for (size_t i = 0; i + 32 <= n; i += 32) {
            __m256i vt = _mm256_loadu_si256((const __m256i*)(temp.data() + i));
            __m256i vr = unary_op(vt);
            _mm256_storeu_si256((__m256i*)(result.data() + i), vr);
        }
    }

    // Timed runs
    for (size_t iter = 0; iter < iterations; ++iter) {
        auto start = clock_type::now();

        // First pass: binary operation
        for (size_t i = 0; i + 32 <= n; i += 32) {
            __m256i va = _mm256_loadu_si256((const __m256i*)(A.data() + i));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(B.data() + i));
            __m256i vr = binary_op(va, vb);
            _mm256_storeu_si256((__m256i*)(temp.data() + i), vr);
        }
        // Second pass: unary operation
        for (size_t i = 0; i + 32 <= n; i += 32) {
            __m256i vt = _mm256_loadu_si256((const __m256i*)(temp.data() + i));
            __m256i vr = unary_op(vt);
            _mm256_storeu_si256((__m256i*)(result.data() + i), vr);
        }

        auto end = clock_type::now();
        double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        times.push_back(ns);
    }

    return calculate_stats(times);
}

// =============================================================================
// Benchmark: Fused operations (single pass)
// =============================================================================

template <typename FusedOp>
BenchStats bench_fused(
    const std::vector<uint8_t>& A,
    const std::vector<uint8_t>& B,
    FusedOp fused_op,
    size_t iterations,
    size_t warmup
) {
    size_t n = A.size();
    std::vector<uint8_t> result(n);
    std::vector<double> times;
    times.reserve(iterations);

    // Warmup
    for (size_t w = 0; w < warmup; ++w) {
        for (size_t i = 0; i + 32 <= n; i += 32) {
            __m256i va = _mm256_loadu_si256((const __m256i*)(A.data() + i));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(B.data() + i));
            __m256i vr = fused_op(va, vb);
            _mm256_storeu_si256((__m256i*)(result.data() + i), vr);
        }
    }

    // Timed runs
    for (size_t iter = 0; iter < iterations; ++iter) {
        auto start = clock_type::now();

        for (size_t i = 0; i + 32 <= n; i += 32) {
            __m256i va = _mm256_loadu_si256((const __m256i*)(A.data() + i));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(B.data() + i));
            __m256i vr = fused_op(va, vb);
            _mm256_storeu_si256((__m256i*)(result.data() + i), vr);
        }

        auto end = clock_type::now();
        double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        times.push_back(ns);
    }

    return calculate_stats(times);
}

// =============================================================================
// Output
// =============================================================================

void print_header() {
    std::cout << "=============================================================================\n";
    std::cout << "Fusion Operations Benchmark (src/core/simd/fused_binary_unary_ops.h)\n";
    std::cout << "=============================================================================\n\n";
    std::cout << "Comparing unfused (2-pass) vs fused (1-pass) operations\n";
    std::cout << "Expected speedup: 1.5-3.0× (micro-kernel, validated 2025-10-29)\n\n";
}

void print_result(const char* op, size_t n, BenchStats unfused, BenchStats fused) {
    double speedup = unfused.mean / fused.mean;
    std::cout << std::setw(16) << op
              << " | N=" << std::setw(8) << n
              << " | Unfused: " << std::setw(10) << std::fixed << std::setprecision(0) << unfused.mean << " ns"
              << " | Fused: " << std::setw(10) << fused.mean << " ns"
              << " | Speedup: " << std::setw(5) << std::setprecision(2) << speedup << "x"
              << " | CV: " << std::setw(5) << std::setprecision(1) << fused.cv << "%\n";
}

void print_csv_header() {
    std::cout << "operation,size,unfused_ns,fused_ns,speedup,cv_percent\n";
}

void print_csv_result(const char* op, size_t n, BenchStats unfused, BenchStats fused) {
    double speedup = unfused.mean / fused.mean;
    std::cout << op << "," << n << ","
              << std::fixed << std::setprecision(2)
              << unfused.mean << "," << fused.mean << ","
              << speedup << "," << fused.cv << "\n";
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    BenchConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--csv") {
            config.output_csv = true;
        }
    }

    if (!config.output_csv) {
        print_header();
    } else {
        print_csv_header();
    }

    for (size_t n : config.sizes) {
        auto A = generate_random_trits(n, 42);
        auto B = generate_random_trits(n, 43);

        // tnot(tadd(a, b))
        {
            auto unfused = bench_unfused(A, B,
                [](auto a, auto b) { return tadd_simd<true>(a, b); },
                [](auto x) { return tnot_simd<true>(x); },
                config.iterations, config.warmup_iters);
            auto fused = bench_fused(A, B,
                [](auto a, auto b) { return fused_tnot_tadd_simd<true>(a, b); },
                config.iterations, config.warmup_iters);

            if (config.output_csv) {
                print_csv_result("tnot_tadd", n, unfused, fused);
            } else {
                print_result("tnot(tadd)", n, unfused, fused);
            }
        }

        // tnot(tmul(a, b))
        {
            auto unfused = bench_unfused(A, B,
                [](auto a, auto b) { return tmul_simd<true>(a, b); },
                [](auto x) { return tnot_simd<true>(x); },
                config.iterations, config.warmup_iters);
            auto fused = bench_fused(A, B,
                [](auto a, auto b) { return fused_tnot_tmul_simd<true>(a, b); },
                config.iterations, config.warmup_iters);

            if (config.output_csv) {
                print_csv_result("tnot_tmul", n, unfused, fused);
            } else {
                print_result("tnot(tmul)", n, unfused, fused);
            }
        }

        // tnot(tmin(a, b))
        {
            auto unfused = bench_unfused(A, B,
                [](auto a, auto b) { return tmin_simd<true>(a, b); },
                [](auto x) { return tnot_simd<true>(x); },
                config.iterations, config.warmup_iters);
            auto fused = bench_fused(A, B,
                [](auto a, auto b) { return fused_tnot_tmin_simd<true>(a, b); },
                config.iterations, config.warmup_iters);

            if (config.output_csv) {
                print_csv_result("tnot_tmin", n, unfused, fused);
            } else {
                print_result("tnot(tmin)", n, unfused, fused);
            }
        }

        // tnot(tmax(a, b))
        {
            auto unfused = bench_unfused(A, B,
                [](auto a, auto b) { return tmax_simd<true>(a, b); },
                [](auto x) { return tnot_simd<true>(x); },
                config.iterations, config.warmup_iters);
            auto fused = bench_fused(A, B,
                [](auto a, auto b) { return fused_tnot_tmax_simd<true>(a, b); },
                config.iterations, config.warmup_iters);

            if (config.output_csv) {
                print_csv_result("tnot_tmax", n, unfused, fused);
            } else {
                print_result("tnot(tmax)", n, unfused, fused);
            }
        }

        if (!config.output_csv) {
            std::cout << "\n";
        }
    }

    if (!config.output_csv) {
        std::cout << "=============================================================================\n";
        std::cout << "Benchmark complete.\n";
        std::cout << "\nNote: Micro-kernel speedups. End-to-end application speedup typically 10-25% lower.\n";
    }

    return 0;
}
