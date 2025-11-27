// bench_dense243.cpp — Native C++ benchmark for Dense243 encoding/decoding
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
// This benchmark measures the raw performance of Dense243 encoding/decoding
// operations without Python/pybind11 overhead. Dense243 packs 5 trits into
// 1 byte (95.3% density vs 25% for 2-bit encoding).
//
// COMPILATION (from benchmarks/cpp-native-kernels/ directory):
//   g++ -O3 -march=native -mavx2 -std=c++17 -I../../src bench_dense243.cpp -o bench_dense243
//   clang++ -O3 -march=native -mavx2 -std=c++17 -I../../src bench_dense243.cpp -o bench_dense243
//
//   # Windows (MSVC):
//   cl /O2 /arch:AVX2 /std:c++17 /EHsc /I..\..\src bench_dense243.cpp
//
// USAGE:
//   ./bench_dense243              # Run all benchmarks
//   ./bench_dense243 --csv        # Output CSV format
//
// TARGET: src/engine/dense243/ - High-density encoding library
//
// =============================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <random>
#include <string>
#include <cmath>

// Engine library includes (src/engine/)
#include "engine/dense243/ternary_dense243.h"

// Core algebra for trit generation
#include "core/algebra/ternary_algebra.h"

using clock_type = std::chrono::steady_clock;

// =============================================================================
// Benchmark Configuration
// =============================================================================

struct BenchConfig {
    std::vector<size_t> sizes = {1000, 10000, 100000, 1000000, 10000000};
    size_t iterations = 100;
    size_t warmup_iters = 10;
    bool output_csv = false;
};

// =============================================================================
// Utilities
// =============================================================================

// Generate random trits (2-bit encoding)
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

// Generate random packed Dense243 bytes
std::vector<uint8_t> generate_random_dense243(size_t num_bytes, uint32_t seed = 42) {
    std::vector<uint8_t> data(num_bytes);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, 242);  // Valid Dense243 range

    for (size_t i = 0; i < num_bytes; ++i) {
        data[i] = static_cast<uint8_t>(dis(gen));
    }
    return data;
}

// Calculate throughput (Million Elements per second)
double calculate_throughput(size_t n, size_t iterations, double elapsed_ns) {
    return (n * iterations) / (elapsed_ns / 1e9) / 1e6;
}

// =============================================================================
// Benchmark: Pack (5 trits → 1 byte)
// =============================================================================

double bench_pack(const std::vector<uint8_t>& trits, size_t iterations) {
    size_t num_quintets = trits.size() / 5;
    std::vector<uint8_t> packed(num_quintets);

    // Warmup
    for (size_t iter = 0; iter < 10; ++iter) {
        for (size_t i = 0; i < num_quintets; ++i) {
            size_t base = i * 5;
            packed[i] = dense243_pack(
                trits[base], trits[base+1], trits[base+2],
                trits[base+3], trits[base+4]
            );
        }
    }

    // Timed runs
    auto start = clock_type::now();
    for (size_t iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < num_quintets; ++i) {
            size_t base = i * 5;
            packed[i] = dense243_pack(
                trits[base], trits[base+1], trits[base+2],
                trits[base+3], trits[base+4]
            );
        }
    }
    auto end = clock_type::now();

    auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    return calculate_throughput(trits.size(), iterations, static_cast<double>(elapsed_ns));
}

// =============================================================================
// Benchmark: Unpack (1 byte → 5 trits)
// =============================================================================

double bench_unpack(const std::vector<uint8_t>& packed, size_t iterations) {
    size_t num_trits = packed.size() * 5;
    std::vector<uint8_t> trits(num_trits);

    // Warmup
    for (size_t iter = 0; iter < 10; ++iter) {
        for (size_t i = 0; i < packed.size(); ++i) {
            Dense243Unpacked u = dense243_unpack(packed[i]);
            size_t base = i * 5;
            trits[base] = u.t0;
            trits[base+1] = u.t1;
            trits[base+2] = u.t2;
            trits[base+3] = u.t3;
            trits[base+4] = u.t4;
        }
    }

    // Timed runs
    auto start = clock_type::now();
    for (size_t iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < packed.size(); ++i) {
            Dense243Unpacked u = dense243_unpack(packed[i]);
            size_t base = i * 5;
            trits[base] = u.t0;
            trits[base+1] = u.t1;
            trits[base+2] = u.t2;
            trits[base+3] = u.t3;
            trits[base+4] = u.t4;
        }
    }
    auto end = clock_type::now();

    auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    return calculate_throughput(num_trits, iterations, static_cast<double>(elapsed_ns));
}

// =============================================================================
// Benchmark: Extract single trit
// =============================================================================

double bench_extract(const std::vector<uint8_t>& packed, size_t iterations) {
    volatile uint8_t sink = 0;  // Prevent optimization

    // Warmup
    for (size_t iter = 0; iter < 10; ++iter) {
        for (size_t i = 0; i < packed.size(); ++i) {
            for (size_t pos = 0; pos < 5; ++pos) {
                sink = dense243_extract_trit(packed[i], pos);
            }
        }
    }

    // Timed runs
    auto start = clock_type::now();
    for (size_t iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < packed.size(); ++i) {
            for (size_t pos = 0; pos < 5; ++pos) {
                sink = dense243_extract_trit(packed[i], pos);
            }
        }
    }
    auto end = clock_type::now();

    size_t total_extractions = packed.size() * 5;
    auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    return calculate_throughput(total_extractions, iterations, static_cast<double>(elapsed_ns));
}

// =============================================================================
// Output Formatters
// =============================================================================

void print_header() {
    std::cout << "=============================================================================\n";
    std::cout << "Dense243 Native C++ Benchmark (src/engine/dense243/)\n";
    std::cout << "=============================================================================\n\n";
    std::cout << "Packing: 5 trits/byte (95.3% density)\n";
    std::cout << "LUT-based extraction (256-byte LUTs, compile-time generated)\n\n";
}

void print_result(const char* operation, size_t n, double throughput) {
    std::cout << std::setw(12) << operation
              << " | N=" << std::setw(10) << n
              << " | " << std::setw(10) << std::fixed << std::setprecision(2) << throughput << " ME/s\n";
}

void print_csv_header() {
    std::cout << "operation,size,throughput_ME_s\n";
}

void print_csv_result(const char* operation, size_t n, double throughput) {
    std::cout << operation << "," << n << "," << throughput << "\n";
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    BenchConfig config;

    // Parse arguments
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
        // Generate test data
        auto trits = generate_random_trits(n, 42);
        auto packed = generate_random_dense243(n / 5, 43);

        // Run benchmarks
        double pack_throughput = bench_pack(trits, config.iterations);
        double unpack_throughput = bench_unpack(packed, config.iterations);
        double extract_throughput = bench_extract(packed, config.iterations);

        if (config.output_csv) {
            print_csv_result("pack", n, pack_throughput);
            print_csv_result("unpack", n, unpack_throughput);
            print_csv_result("extract", n, extract_throughput);
        } else {
            print_result("pack", n, pack_throughput);
            print_result("unpack", n, unpack_throughput);
            print_result("extract", n, extract_throughput);
            std::cout << "\n";
        }
    }

    if (!config.output_csv) {
        std::cout << "=============================================================================\n";
        std::cout << "Benchmark complete.\n";
    }

    return 0;
}
