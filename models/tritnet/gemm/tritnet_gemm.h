/**
 * @file tritnet_gemm.h
 * @brief TritNet Direct Ternary GEMM - Replacement for BitNet LUT Kernels
 *
 * This module provides optimized matrix multiplication for ternary weights
 * using our Dense243 packing and direct ternary operations instead of
 * BitNet's lookup table approach.
 *
 * Performance target: 2-3× faster than BitNet TL1/TL2 kernels
 *
 * @date 2025-11-23
 * @version 1.0
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Ternary GEMM: C = A × B where B contains ternary weights {-1, 0, +1}
 *
 * Computes matrix multiplication with ternary (1.58-bit) weights stored in
 * Dense243 format (5 trits/byte). This replaces BitNet's LUT-based approach
 * with direct ternary multiply-accumulate operations.
 *
 * @param M         Number of rows in A and C
 * @param N         Number of columns in B and C
 * @param K         Number of columns in A / rows in B
 * @param A         Input activations [M × K], row-major, float32
 * @param B_packed  Ternary weights [⌈K/5⌉ × N], Dense243-packed, uint8
 * @param C         Output [M × N], row-major, float32
 *
 * @note Weight packing: Every 5 consecutive weights in K dimension packed
 *       into 1 byte using Dense243 encoding (95.3% density).
 * @note K must be multiple of 5 for correct unpacking.
 * @note All pointers must be 64-byte aligned for SIMD efficiency.
 *
 * @performance
 *   - Naive (no SIMD): ~1-2 Gops/s
 *   - AVX2 optimized:  ~20-30 Gops/s
 *   - Expected speedup over BitNet TL2: 2-3×
 *
 * Example:
 * @code
 *   float A[1024 * 4096];     // Activations
 *   uint8_t B[819 * 2048];    // Weights (4096/5 = 819 rows)
 *   float C[1024 * 2048];     // Output
 *
 *   tritnet_gemm_f32(1024, 2048, 4096, A, B, C);
 * @endcode
 */
void tritnet_gemm_f32(
    int M,
    int N,
    int K,
    const float* A,
    const uint8_t* B_packed,
    float* C
);

/**
 * @brief Ternary GEMM with scaling factors (for per-block quantization)
 *
 * Same as tritnet_gemm_f32 but applies per-column scale factors to outputs.
 * This matches BitNet's per-block quantization scheme.
 *
 * @param M             Number of rows in A and C
 * @param N             Number of columns in B and C
 * @param K             Number of columns in A / rows in B
 * @param A             Input activations [M × K], row-major, float32
 * @param B_packed      Ternary weights [⌈K/5⌉ × N], Dense243-packed, uint8
 * @param scales        Per-column scale factors [N], float32
 * @param C             Output [M × N], row-major, float32
 *
 * @note Output computation: C[m,n] = scale[n] * sum_k(A[m,k] * B[k,n])
 * @note This matches BitNet's `ggml_qgemm_lut` with scale parameters.
 */
void tritnet_gemm_f32_scaled(
    int M,
    int N,
    int K,
    const float* A,
    const uint8_t* B_packed,
    const float* scales,
    float* C
);

/**
 * @brief Convert BitNet 2-bit weights to Dense243 format
 *
 * Converts weights from BitNet's 2-bit packing (8 trits/byte) to our
 * Dense243 format (5 trits/byte). This is a one-time conversion during
 * model loading.
 *
 * @param bitnet_weights  Input weights in BitNet 2-bit format [⌈K/4⌉ × N]
 * @param dense243_out    Output weights in Dense243 format [⌈K/5⌉ × N]
 * @param K               Number of rows (must be multiple of 20 for efficiency)
 * @param N               Number of columns
 *
 * @note BitNet encoding: -1→00, 0→01, +1→10 (2 bits/trit, 4 trits/byte)
 * @note Dense243 encoding: 5 trits/byte (see dense243.cpp for details)
 * @note Memory savings: Output is ~40% smaller than input
 *
 * Example:
 * @code
 *   uint8_t bitnet[1024 * 2048];    // BitNet format (K=4096, N=2048)
 *   uint8_t dense243[820 * 2048];   // Dense243 format (819.2 → 820)
 *
 *   convert_bitnet_to_dense243(bitnet, dense243, 4096, 2048);
 *   // dense243 now contains same weights, 40% less memory
 * @endcode
 */
void convert_bitnet_to_dense243(
    const uint8_t* bitnet_weights,
    uint8_t* dense243_out,
    int K,
    int N
);

/**
 * @brief Set number of threads for parallel GEMM
 *
 * Controls OpenMP parallelization across M dimension. Default is auto-detect.
 *
 * @param n_threads  Number of threads to use (0 = auto-detect)
 */
void tritnet_set_num_threads(int n_threads);

/**
 * @brief Get optimal block size for cache tiling
 *
 * Returns recommended tile size for loop tiling optimization based on
 * L1/L2 cache size. Used internally by GEMM kernel.
 *
 * @param cache_level  1 for L1, 2 for L2, 3 for L3
 * @return Recommended tile size in elements
 */
int tritnet_get_optimal_tile_size(int cache_level);

/**
 * @brief Benchmark GEMM performance
 *
 * Runs GEMM operation multiple times and returns average time in milliseconds.
 * Useful for profiling and comparing against BitNet.
 *
 * @param M            Matrix dimension
 * @param N            Matrix dimension
 * @param K            Matrix dimension
 * @param num_runs     Number of iterations for averaging
 * @return Average time per run in milliseconds
 */
double tritnet_benchmark_gemm(int M, int N, int K, int num_runs);

/**
 * @brief Validate GEMM correctness against reference implementation
 *
 * Compares TritNet GEMM output against naive reference implementation.
 * Returns maximum absolute error across all elements.
 *
 * @param M  Matrix dimension
 * @param N  Matrix dimension
 * @param K  Matrix dimension
 * @return Maximum absolute error (should be < 1e-6 for FP32)
 */
float tritnet_validate_gemm(int M, int N, int K);

// ============================================================================
// Internal API (not for external use)
// ============================================================================

#ifdef TRITNET_GEMM_INTERNAL

/**
 * @brief Unpack 5 ternary weights from Dense243 byte
 *
 * Internal helper for GEMM kernel. Unpacks one Dense243 byte into 5 trits.
 *
 * @param packed  Dense243-encoded byte
 * @param trits   Output array [5] for unpacked trits {-1, 0, +1}
 */
static inline void unpack_dense243_5(uint8_t packed, int8_t* trits);

/**
 * @brief SIMD-optimized inner loop for GEMM (AVX2)
 *
 * Processes 8 rows of A at once using AVX2 intrinsics.
 *
 * @param A          Activation block [8 × K]
 * @param B_packed   Weight block [⌈K/5⌉ × N]
 * @param C          Output block [8 × N]
 * @param K          Inner dimension
 * @param N          Number of columns
 */
void tritnet_gemm_kernel_avx2_8x(
    const float* A,
    const uint8_t* B_packed,
    float* C,
    int K,
    int N
);

/**
 * @brief SIMD-optimized inner loop for GEMM (ARM NEON)
 *
 * Processes 4 rows of A at once using NEON intrinsics.
 *
 * @param A          Activation block [4 × K]
 * @param B_packed   Weight block [⌈K/5⌉ × N]
 * @param C          Output block [4 × N]
 * @param K          Inner dimension
 * @param N          Number of columns
 */
void tritnet_gemm_kernel_neon_4x(
    const float* A,
    const uint8_t* B_packed,
    float* C,
    int K,
    int N
);

#endif // TRITNET_GEMM_INTERNAL

#ifdef __cplusplus
}
#endif

/**
 * @example tritnet_gemm_example.cpp
 * @code
 * #include "tritnet_gemm.h"
 * #include <vector>
 * #include <iostream>
 *
 * int main() {
 *     const int M = 1024;   // Batch size
 *     const int N = 2048;   // Output features
 *     const int K = 4096;   // Input features
 *
 *     // Allocate matrices
 *     std::vector<float> A(M * K);
 *     std::vector<uint8_t> B((K / 5) * N);  // Dense243 packed
 *     std::vector<float> C(M * N);
 *
 *     // Fill A with random activations
 *     for (auto& val : A) val = (rand() % 200 - 100) / 100.0f;
 *
 *     // Fill B with random ternary weights
 *     // (In practice, load from model file)
 *     for (auto& val : B) val = rand() % 243;  // Valid Dense243 values
 *
 *     // Run GEMM
 *     tritnet_gemm_f32(M, N, K, A.data(), B.data(), C.data());
 *
 *     // Benchmark performance
 *     double avg_time = tritnet_benchmark_gemm(M, N, K, 100);
 *     std::cout << "Average time: " << avg_time << " ms\n";
 *
 *     // Calculate throughput
 *     double gops = (2.0 * M * N * K) / (avg_time * 1e6);  // Operations in billions
 *     std::cout << "Performance: " << gops << " Gops/s\n";
 *
 *     return 0;
 * }
 * @endcode
 */

// End of tritnet_gemm.h
