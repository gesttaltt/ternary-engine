/**
 * bench_canonical_validation.cpp - Rigorous canonical indexing validation
 *
 * Copyright (c) 2025 Ternary Engine Contributors
 * Licensed under the Apache License, Version 2.0
 *
 * PURPOSE:
 * Validate the claimed "12-18% performance improvement" from canonical indexing.
 *
 * METHODOLOGY:
 * 1. Benchmark AVX2_v1 (traditional indexing: idx = (a<<2)|b)
 * 2. Benchmark AVX2_v2 (canonical indexing: idx = (a*3)+b via dual-shuffle)
 * 3. Compare across multiple array sizes
 * 4. Report actual speedup with statistical rigor
 *
 * EXPECTED OUTCOME:
 * - If canonical is faster: Validate and keep optimization
 * - If equivalent (±3%): Remove canonical (reduces complexity)
 * - If canonical is slower: Remove immediately (performance regression)
 *
 * USAGE:
 *   g++ -O3 -mavx2 -std=c++17 bench_canonical_validation.cpp -o bench_canonical
 *   ./bench_canonical
 *
 *   OR with MSVC:
 *   cl /O2 /arch:AVX2 /std:c++17 bench_canonical_validation.cpp
 *   bench_canonical_validation.exe
 */

#include "../../src/core/simd/backend_avx2_v1_baseline.cpp"
#include "../../src/core/simd/backend_avx2_v2_optimized.cpp"
#include "../../src/core/simd/backend_scalar_impl.cpp"
#include "../../src/core/simd/backend_registry_dispatch.cpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <cmath>

using namespace std::chrono;

// =============================================================================
// Benchmark Configuration
// =============================================================================

const int WARMUP_ITERATIONS = 10;
const int BENCHMARK_ITERATIONS = 100;
const size_t TEST_SIZES[] = {10000, 100000, 1000000, 10000000};
const int NUM_SIZES = 4;

// =============================================================================
// Statistical Utilities
// =============================================================================

struct BenchmarkResult {
    double mean_ms;
    double stddev_ms;
    double min_ms;
    double max_ms;
    double median_ms;
};

BenchmarkResult compute_statistics(std::vector<double>& times_ms) {
    BenchmarkResult result;

    // Sort for median calculation
    std::sort(times_ms.begin(), times_ms.end());

    // Mean
    double sum = 0.0;
    for (double t : times_ms) {
        sum += t;
    }
    result.mean_ms = sum / times_ms.size();

    // Standard deviation
    double variance = 0.0;
    for (double t : times_ms) {
        double diff = t - result.mean_ms;
        variance += diff * diff;
    }
    result.stddev_ms = std::sqrt(variance / times_ms.size());

    // Min/max/median
    result.min_ms = times_ms.front();
    result.max_ms = times_ms.back();
    result.median_ms = times_ms[times_ms.size() / 2];

    return result;
}

// =============================================================================
// Test Data Generation
// =============================================================================

void generate_random_trits(uint8_t* data, size_t count) {
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<> dis(0, 2);

    for (size_t i = 0; i < count; i++) {
        // Generate random trit: 0b00, 0b01, or 0b10
        int trit_val = dis(gen);
        if (trit_val == 0) data[i] = 0b00;
        else if (trit_val == 1) data[i] = 0b01;
        else data[i] = 0b10;
    }
}

// =============================================================================
// Benchmark Functions
// =============================================================================

BenchmarkResult benchmark_backend(
    const char* backend_name,
    void (*operation)(uint8_t*, const uint8_t*, const uint8_t*, size_t),
    const uint8_t* a,
    const uint8_t* b,
    uint8_t* out,
    size_t n
) {
    std::vector<double> times_ms;

    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        operation(out, a, b, n);
    }

    // Benchmark
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        auto start = high_resolution_clock::now();
        operation(out, a, b, n);
        auto end = high_resolution_clock::now();

        double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        times_ms.push_back(time_ms);
    }

    return compute_statistics(times_ms);
}

// =============================================================================
// Correctness Verification
// =============================================================================

bool verify_correctness(const uint8_t* a, const uint8_t* b, size_t n) {
    uint8_t* out_v1 = (uint8_t*)malloc(n);
    uint8_t* out_v2 = (uint8_t*)malloc(n);

    // Run both backends
    // Note: This requires direct access to backend functions
    // For now, we'll use the dispatch system
    ternary_backend_init();

    // Set to v1
    const TernaryBackend* v1 = ternary_backend_find("AVX2_v1");
    if (v1) {
        ternary_backend_set_active(v1);
        ternary_dispatch_tadd(out_v1, a, b, n);
    }

    // Set to v2
    const TernaryBackend* v2 = ternary_backend_find("AVX2_v2");
    if (v2) {
        ternary_backend_set_active(v2);
        ternary_dispatch_tadd(out_v2, a, b, n);
    }

    // Compare results
    bool match = true;
    for (size_t i = 0; i < n; i++) {
        if (out_v1[i] != out_v2[i]) {
            match = false;
            std::cerr << "Mismatch at index " << i << ": v1=" << (int)out_v1[i]
                      << ", v2=" << (int)out_v2[i] << std::endl;
            break;
        }
    }

    free(out_v1);
    free(out_v2);

    return match;
}

// =============================================================================
// Main Benchmark
// =============================================================================

int main() {
    std::cout << "\n";
    std::cout << "========================================================================\n";
    std::cout << "  Canonical Indexing Validation Benchmark\n";
    std::cout << "========================================================================\n";
    std::cout << "Compiler:   " << __VERSION__ << "\n";
    std::cout << "Iterations: " << BENCHMARK_ITERATIONS << " (per size)\n";
    std::cout << "Operation:  tadd (ternary addition)\n";
    std::cout << "\n";

    // Initialize backend system
    if (!ternary_backend_init()) {
        std::cerr << "ERROR: Failed to initialize backend system\n";
        return 1;
    }

    // Print available backends
    std::cout << "Available backends: " << ternary_backend_count() << "\n";
    for (size_t i = 0; i < ternary_backend_count(); i++) {
        const TernaryBackend* backend = ternary_backend_get(i);
        if (backend) {
            std::cout << "  - " << backend->info.name << ": " << backend->info.description << "\n";
        }
    }
    std::cout << "\n";

    // Get backend pointers
    const TernaryBackend* v1_backend = ternary_backend_find("AVX2_v1");
    const TernaryBackend* v2_backend = ternary_backend_find("AVX2_v2");

    if (!v1_backend || !v2_backend) {
        std::cerr << "ERROR: Required backends not available\n";
        return 1;
    }

    std::cout << "Comparing:\n";
    std::cout << "  AVX2_v1: " << v1_backend->info.description << "\n";
    std::cout << "  AVX2_v2: " << v2_backend->info.description << "\n";
    std::cout << "\n";

    // Results table header
    std::cout << "Size      | AVX2_v1 (ms)  | AVX2_v2 (ms)  | Speedup | Status\n";
    std::cout << "----------|---------------|---------------|---------|----------\n";

    // Test each size
    for (int size_idx = 0; size_idx < NUM_SIZES; size_idx++) {
        size_t n = TEST_SIZES[size_idx];

        // Allocate arrays
        uint8_t* a = (uint8_t*)malloc(n);
        uint8_t* b = (uint8_t*)malloc(n);
        uint8_t* out = (uint8_t*)malloc(n);

        // Generate test data
        generate_random_trits(a, n);
        generate_random_trits(b, n);

        // Verify correctness first
        bool correct = verify_correctness(a, b, n);

        // Benchmark AVX2_v1
        ternary_backend_set_active(v1_backend);
        BenchmarkResult v1_result = benchmark_backend(
            "AVX2_v1",
            v1_backend->tadd,
            a, b, out, n
        );

        // Benchmark AVX2_v2
        ternary_backend_set_active(v2_backend);
        BenchmarkResult v2_result = benchmark_backend(
            "AVX2_v2",
            v2_backend->tadd,
            a, b, out, n
        );

        // Calculate speedup
        double speedup = v1_result.median_ms / v2_result.median_ms;

        // Determine status
        const char* status;
        if (!correct) {
            status = "INCORRECT";
        } else if (speedup > 1.05) {
            status = "FASTER";
        } else if (speedup < 0.95) {
            status = "SLOWER";
        } else {
            status = "EQUIVALENT";
        }

        // Print result
        std::cout << std::setw(9) << n << " | "
                  << std::fixed << std::setprecision(3)
                  << std::setw(8) << v1_result.median_ms << " ±"
                  << std::setw(4) << v1_result.stddev_ms << " | "
                  << std::setw(8) << v2_result.median_ms << " ±"
                  << std::setw(4) << v2_result.stddev_ms << " | "
                  << std::setw(6) << speedup << "× | "
                  << status << "\n";

        // Cleanup
        free(a);
        free(b);
        free(out);
    }

    std::cout << "\n";
    std::cout << "========================================================================\n";
    std::cout << "  Interpretation Guide\n";
    std::cout << "========================================================================\n";
    std::cout << "Speedup > 1.05 (>5%):  Canonical is FASTER - Keep optimization\n";
    std::cout << "Speedup 0.95-1.05:     EQUIVALENT - Remove canonical (reduce complexity)\n";
    std::cout << "Speedup < 0.95 (<-5%): Canonical is SLOWER - Remove immediately\n";
    std::cout << "\n";
    std::cout << "RECOMMENDATION:\n";
    std::cout << "If most results are EQUIVALENT or SLOWER → Remove canonical indexing\n";
    std::cout << "If most results are FASTER → Keep canonical and update documentation\n";
    std::cout << "\n";

    return 0;
}
