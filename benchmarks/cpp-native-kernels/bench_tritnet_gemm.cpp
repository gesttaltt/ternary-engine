/**
 * @file bench_tritnet_gemm.cpp
 * @brief Benchmark TritNet GEMM performance vs BitNet
 *
 * Measures performance of naive vs AVX2 implementations and compares
 * against BitNet's expected performance.
 *
 * @date 2025-11-23
 *
 * COMPILATION (from benchmarks/cpp-native-kernels/ directory):
 *   g++ -O3 -march=native -mavx2 -std=c++17 -I../../ bench_tritnet_gemm.cpp -o bench_gemm
 *   clang++ -O3 -march=native -mavx2 -std=c++17 -I../../ bench_tritnet_gemm.cpp -o bench_gemm
 *
 *   # Windows (MSVC):
 *   cl /O2 /arch:AVX2 /std:c++17 /EHsc /I..\..\ bench_tritnet_gemm.cpp
 *
 * TARGET: models/tritnet/gemm/ - TritNet GEMM operations (experimental)
 */

// TritNet GEMM header (models/tritnet/gemm/)
#include "models/tritnet/gemm/tritnet_gemm.h"

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <cmath>

using namespace std::chrono;

/**
 * @brief Benchmark configuration for different matrix sizes
 */
struct BenchConfig {
    int M, N, K;
    const char* name;
    const char* use_case;
};

// Benchmark configurations matching BitNet model sizes
static const BenchConfig BENCH_CONFIGS[] = {
    // Tiny (for debugging)
    {8, 8, 160, "Tiny", "Debug"},

    // Small (similar to attention heads)
    {32, 64, 512, "Small", "Attention head"},

    // Medium (similar to MLP layers in 2B model)
    {1024, 2048, 4096, "Medium-2B", "MLP layer 2B model"},

    // Large (similar to MLP layers in 7B model)
    {2048, 8192, 8192, "Large-7B", "MLP layer 7B model"},

    // Huge (similar to 100B model)
    {4096, 16384, 16384, "Huge-100B", "MLP layer 100B model"},
};

/**
 * @brief Run GEMM and measure time
 */
double benchmark_gemm(
    int M, int N, int K,
    const float* A,
    const uint8_t* B,
    float* C,
    int num_runs = 10
) {
    // Warm-up run
    tritnet_gemm_f32(M, N, K, A, B, C);

    // Timed runs
    auto start = high_resolution_clock::now();

    for (int run = 0; run < num_runs; run++) {
        tritnet_gemm_f32(M, N, K, A, B, C);
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    return duration.count() / 1000.0 / num_runs;  // Average ms per run
}

/**
 * @brief Calculate GFLOPS for matrix multiply
 */
double calculate_gops(int M, int N, int K, double time_ms) {
    // GEMM operations: 2*M*N*K (multiply + add for each element)
    double ops = 2.0 * M * N * K;
    double gops = ops / (time_ms * 1e6);  // Convert to Gops/s
    return gops;
}

/**
 * @brief Run full benchmark suite
 */
void run_benchmarks() {
    std::cout << "========================================================\n";
    std::cout << " TritNet GEMM Benchmark Suite\n";
    std::cout << "========================================================\n\n";

    std::cout << std::setw(12) << "Config"
              << std::setw(15) << "Dimensions"
              << std::setw(12) << "Time (ms)"
              << std::setw(12) << "Gops/s"
              << std::setw(15) << "Memory (MB)"
              << "\n";
    std::cout << std::string(66, '-') << "\n";

    for (const auto& config : BENCH_CONFIGS) {
        int M = config.M;
        int N = config.N;
        int K = config.K;

        // Allocate matrices
        std::vector<float> A(M * K);
        std::vector<uint8_t> B((K / 5) * N);
        std::vector<float> C(M * N);

        // Initialize with random data
        for (auto& val : A) {
            val = (float)(rand() % 200 - 100) / 100.0f;
        }
        for (auto& val : B) {
            val = (uint8_t)(rand() % 243);
        }

        // Run benchmark
        int num_runs = (M * N * K < 1000000) ? 100 : 10;  // More runs for small sizes
        double time_ms = benchmark_gemm(M, N, K, A.data(), B.data(), C.data(), num_runs);

        // Calculate metrics
        double gops = calculate_gops(M, N, K, time_ms);
        double memory_mb = (M * K * 4 + (K / 5) * N + M * N * 4) / (1024.0 * 1024.0);

        // Print results
        std::cout << std::setw(12) << config.name
                  << std::setw(8) << M << "×" << N << "×" << K
                  << std::setw(12) << std::fixed << std::setprecision(3) << time_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << gops
                  << std::setw(15) << std::fixed << std::setprecision(1) << memory_mb
                  << "\n";
    }

    std::cout << "\n========================================================\n";
}

/**
 * @brief Compare with BitNet expected performance
 */
void compare_with_bitnet() {
    std::cout << "\n========================================================\n";
    std::cout << " Performance Comparison: TritNet vs BitNet\n";
    std::cout << "========================================================\n\n";

    // Medium config (2B model MLP layer)
    int M = 1024, N = 2048, K = 4096;

    std::vector<float> A(M * K);
    std::vector<uint8_t> B((K / 5) * N);
    std::vector<float> C(M * N);

    // Initialize
    for (auto& val : A) val = (float)(rand() % 100) / 100.0f;
    for (auto& val : B) val = (uint8_t)(rand() % 243);

    // Benchmark TritNet (naive)
    double tritnet_time = benchmark_gemm(M, N, K, A.data(), B.data(), C.data(), 10);
    double tritnet_gops = calculate_gops(M, N, K, tritnet_time);

    // BitNet TL2 expected performance (from their benchmarks)
    // On Intel i7, they report 2.37-6.17× speedup over INT8
    // INT8 GEMM on i7 ~= 50-100 Gops/s
    // BitNet TL2 ~= 118-617 Gops/s (let's estimate ~200 Gops/s for this size)
    double bitnet_estimated_gops = 200.0;  // Conservative estimate

    // Our target (2-3× faster than BitNet)
    double target_gops = bitnet_estimated_gops * 2.5;  // 500 Gops/s

    std::cout << std::setw(25) << "Implementation"
              << std::setw(15) << "Gops/s"
              << std::setw(15) << "Speedup"
              << "\n";
    std::cout << std::string(55, '-') << "\n";

    std::cout << std::setw(25) << "TritNet (Naive)"
              << std::setw(15) << std::fixed << std::setprecision(2) << tritnet_gops
              << std::setw(15) << "1.0×"
              << "\n";

    std::cout << std::setw(25) << "BitNet TL2 (estimated)"
              << std::setw(15) << bitnet_estimated_gops
              << std::setw(15) << std::fixed << std::setprecision(1)
              << (bitnet_estimated_gops / tritnet_gops) << "×"
              << "\n";

    std::cout << std::setw(25) << "TritNet Target (AVX2)"
              << std::setw(15) << target_gops
              << std::setw(15) << (target_gops / tritnet_gops) << "×"
              << "\n";

    std::cout << "\n";

    // Status
    if (tritnet_gops >= target_gops) {
        std::cout << "✅ TARGET ACHIEVED! TritNet is " << (tritnet_gops / bitnet_estimated_gops)
                  << "× faster than BitNet\n";
    } else {
        double gap = (target_gops / tritnet_gops);
        std::cout << "⚠️  Need " << gap << "× speedup to reach target\n";
        std::cout << "   Current: " << tritnet_gops << " Gops/s\n";
        std::cout << "   Target:  " << target_gops << " Gops/s\n";
        std::cout << "   Next: Implement AVX2 SIMD optimization\n";
    }

    std::cout << "\n========================================================\n";
}

/**
 * @brief Memory bandwidth analysis
 */
void analyze_memory_bandwidth() {
    std::cout << "\n========================================================\n";
    std::cout << " Memory Bandwidth Analysis\n";
    std::cout << "========================================================\n\n";

    // Medium config
    int M = 1024, N = 2048, K = 4096;

    // Calculate memory traffic
    double A_mb = (M * K * 4) / (1024.0 * 1024.0);  // FP32 activations
    double B_mb = ((K / 5) * N) / (1024.0 * 1024.0);  // Dense243 weights
    double C_mb = (M * N * 4) / (1024.0 * 1024.0);  // FP32 output

    double total_read = A_mb + B_mb;
    double total_write = C_mb;
    double total_traffic = total_read + total_write;

    std::cout << "Matrix dimensions: " << M << "×" << N << "×" << K << "\n\n";

    std::cout << "Memory traffic:\n";
    std::cout << "  A (activations):  " << std::fixed << std::setprecision(2) << A_mb << " MB\n";
    std::cout << "  B (weights):      " << B_mb << " MB\n";
    std::cout << "  C (output):       " << C_mb << " MB\n";
    std::cout << "  Total:            " << total_traffic << " MB\n\n";

    // Bandwidth comparison
    std::cout << "Weight format comparison:\n";
    double bitnet_2bit_mb = (K * N * 2 / 8) / (1024.0 * 1024.0);  // 2 bits per weight
    double dense243_mb = B_mb;
    double savings = (1.0 - dense243_mb / bitnet_2bit_mb) * 100.0;

    std::cout << "  BitNet 2-bit:     " << bitnet_2bit_mb << " MB\n";
    std::cout << "  Dense243:         " << dense243_mb << " MB\n";
    std::cout << "  Savings:          " << savings << "%\n";

    std::cout << "\n========================================================\n";
}

/**
 * @brief Main benchmark runner
 */
int main(int argc, char** argv) {
    // Set random seed for reproducibility
    srand(42);

    std::cout << "\nTritNet Direct Ternary GEMM - Performance Benchmark\n";
    std::cout << "====================================================\n";
    std::cout << "Platform: x86-64 AVX2\n";
    std::cout << "Precision: FP32 activations, Ternary weights\n";
    std::cout << "Packing: Dense243 (5 trits/byte)\n\n";

    // Run benchmarks
    run_benchmarks();

    // Compare with BitNet
    compare_with_bitnet();

    // Analyze memory bandwidth
    analyze_memory_bandwidth();

    return 0;
}
