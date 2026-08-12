/**
 * @file tritnet_gemm_avx2.cpp
 * @brief AVX2-optimized TritNet Direct Ternary GEMM implementation
 *
 * SIMD-optimized matrix multiplication for x86-64 with AVX2.
 * Performance target: 20-30 Gops/s (10-15× faster than naive)
 *
 * @date 2025-11-23
 */

#ifdef __AVX2__

#include "tritnet_gemm.h"
#include <immintrin.h>
#include <string.h>

/**
 * @brief Unpack 15 ternary weights from 3 Dense243 bytes
 *
 * Dense243 stores 5 trits/byte, so 3 bytes = 15 trits.
 * This unpacks them into an array for SIMD processing.
 *
 * @param packed      Pointer to the first of 3 Dense243 bytes
 * @param row_stride  Distance (in bytes) between consecutive K_packed rows
 *                     at the same output column — i.e. N, since B_packed is
 *                     stored row-major as [K_packed, N]. NOT 1: the 3 bytes
 *                     needed here are 3 consecutive *rows* (k_group/5,
 *                     k_group/5+1, k_group/5+2) at the same column, not 3
 *                     consecutive flat-array bytes.
 * @param trits       Output array [16] (padded to 16 for SIMD alignment)
 *
 * Found 2026-08-12 via tritnet_validate_gemm: the call site previously read
 * packed[0], packed[1], packed[2] (row_stride implicitly 1), which is only
 * correct when N==1. For any N>1 this silently read 3 bytes from the same
 * row at columns n, n+1, n+2 instead of 3 rows at column n — i.e. the wrong
 * weights entirely. This was undetected because tritnet_validate_gemm
 * compared the naive implementation against itself (see
 * tritnet_gemm_naive.cpp) and therefore could never have caught it.
 */
static inline void unpack_dense243_15_avx2(const uint8_t* packed, int row_stride, int8_t* trits) {
    // Unpack 3 bytes (15 trits) using same algorithm as naive version
    for (int b = 0; b < 3; b++) {
        int value = packed[b * row_stride];
        for (int i = 0; i < 5; i++) {
            int remainder = value % 3;
            value /= 3;
            trits[b * 5 + i] = (int8_t)(remainder - 1);
        }
    }
    // Pad last element to 0 for alignment
    trits[15] = 0;
}

/**
 * @brief AVX2 kernel: Process 8 rows of A at once
 *
 * This is the core SIMD kernel that computes 8 output elements in parallel.
 * Uses AVX2 to process 8 floats at once.
 *
 * Algorithm:
 *   For each group of 15 weights:
 *     1. Load 8 activations (AVX2 register)
 *     2. Unpack 15 ternary weights
 *     3. For each weight position:
 *        - Broadcast weight to all lanes
 *        - Conditional add/subtract based on weight value
 *     4. Accumulate results
 *
 * @param A          Activation block [8 × K]
 * @param B_packed   Weight block [⌈K/5⌉ × N]
 * @param C          Output block [8 × N]
 * @param K          Inner dimension (must be multiple of 15 for efficiency)
 * @param N          Number of output columns
 */
void tritnet_gemm_kernel_avx2_8x(
    const float* A,
    const uint8_t* B_packed,
    float* C,
    int K,
    int N
) {
    const int K_packed = (K + 4) / 5;  // Number of Dense243 bytes per column

    // Process 8 rows of A at once
    for (int n = 0; n < N; n++) {
        // Accumulator for 8 outputs (one per row)
        __m256 acc = _mm256_setzero_ps();

        // Process K dimension in groups of 15 (3 Dense243 bytes)
        for (int k_group = 0; k_group < K; k_group += 15) {
            // Unpack 15 weights for this column
            int8_t trits[16];  // Aligned to 16 for SIMD
            int pack_idx = (k_group / 5) * N + n;
            unpack_dense243_15_avx2(&B_packed[pack_idx], N, trits);

            // Process 15 weight positions (or remaining K)
            int k_max = (k_group + 15 < K) ? 15 : (K - k_group);
            for (int i = 0; i < k_max; i++) {
                int k = k_group + i;
                int8_t w = trits[i];

                if (w == 0) continue;  // Skip zero weights (free multiply)

                // Load 8 consecutive activations from column k
                // A is row-major: A[m, k] = A[m * K + k]
                // We want A[0:8, k]
                __m256 a_vec = _mm256_set_ps(
                    A[7 * K + k],
                    A[6 * K + k],
                    A[5 * K + k],
                    A[4 * K + k],
                    A[3 * K + k],
                    A[2 * K + k],
                    A[1 * K + k],
                    A[0 * K + k]
                );

                if (w == 1) {
                    // +1 weight: add activation
                    acc = _mm256_add_ps(acc, a_vec);
                } else {  // w == -1
                    // -1 weight: subtract activation
                    acc = _mm256_sub_ps(acc, a_vec);
                }
            }
        }

        // Store 8 accumulated results to output
        // C is row-major: C[m, n] = C[m * N + n]
        float results[8];
        _mm256_storeu_ps(results, acc);
        for (int m = 0; m < 8; m++) {
            C[m * N + n] = results[m];
        }
    }
}

/**
 * @brief AVX2-optimized GEMM main function
 *
 * Tiles the computation to process 8 rows at a time using AVX2.
 * Falls back to naive implementation for remaining rows.
 */
void tritnet_gemm_f32_avx2(
    int M,
    int N,
    int K,
    const float* A,
    const uint8_t* B_packed,
    float* C
) {
    // Zero output
    memset(C, 0, M * N * sizeof(float));

    // Process in blocks of 8 rows (AVX2 register size)
    int m_blocks = M / 8;
    int m_remainder = M % 8;

    // Process 8-row blocks
    for (int mb = 0; mb < m_blocks; mb++) {
        const float* A_block = A + (mb * 8) * K;
        float* C_block = C + (mb * 8) * N;

        tritnet_gemm_kernel_avx2_8x(A_block, B_packed, C_block, K, N);
    }

    // Handle remaining rows with scalar fallback
    if (m_remainder > 0) {
        int m_start = m_blocks * 8;
        for (int m = m_start; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float acc = 0.0f;
                for (int k_group = 0; k_group + 5 <= K; k_group += 5) {
                    int pack_idx = (k_group / 5) * N + n;
                    int8_t trits[5];
                    int value = B_packed[pack_idx];
                    for (int t = 0; t < 5; t++) {
                        trits[t] = (int8_t)(value % 3 - 1);
                        value /= 3;
                    }
                    for (int t = 0; t < 5; t++) {
                        acc += A[m * K + k_group + t] * trits[t];
                    }
                }
                C[m * N + n] = acc;
            }
        }
    }
}

/**
 * @brief Fused Ternary Multiply-Accumulate (Ternary FMA)
 *
 * Optimized inner loop that combines unpacking + multiply + accumulate.
 * Uses AVX2 gather/scatter for better memory bandwidth.
 *
 * @param A          Activations [M × K]
 * @param B_packed   Dense243 weights [⌈K/5⌉ × N]
 * @param C          Output [M × N]
 * @param M, N, K    Dimensions
 */
void tritnet_gemm_f32_avx2_fused(
    int M,
    int N,
    int K,
    const float* A,
    const uint8_t* B_packed,
    float* C
) {
    // Advanced optimization: Fuse unpacking with accumulation
    // Use AVX2 gather instructions to load non-contiguous activations
    // This will be implemented in Phase 2 after validating basic AVX2 version

    // For now, delegate to basic AVX2 version
    tritnet_gemm_f32_avx2(M, N, K, A, B_packed, C);
}

/**
 * @brief Cache-optimized tiled GEMM
 *
 * Tiles the computation for L1/L2 cache efficiency.
 * Uses 3-level tiling: L1 (32×32), L2 (128×128), L3 (512×512).
 *
 * *** BROKEN — not declared in tritnet_gemm.h, not called from anywhere in
 * this repo (confirmed 2026-08-12) ***. Has the same class of bug just
 * fixed in tritnet_gemm_f32_avx2/unpack_dense243_15_avx2: it calls
 * tritnet_gemm_kernel_avx2_8x(A_tile, B_tile, C_tile, k_block, n_block),
 * passing the TILE width (n_block) as that kernel's N — but the kernel
 * uses N both as its column loop bound AND as the row stride for indexing
 * into B_packed. B_tile is a sub-view into the full [K_packed, N_full]
 * matrix, so its row stride is N_full, not n_block; using n_block corrupts
 * the same way the just-fixed bug did. Not fixed here because nothing
 * exercises this path to verify a fix against (unlike the reachable bug,
 * which tritnet_validate_gemm now genuinely catches) — would need
 * tritnet_gemm_kernel_avx2_8x's signature split into a real N (loop
 * bound) and a separate row_stride parameter.
 */
void tritnet_gemm_f32_avx2_tiled(
    int M,
    int N,
    int K,
    const float* A,
    const uint8_t* B_packed,
    float* C
) {
    // Cache tiling parameters
    const int TILE_M = 32;   // L1 tile
    const int TILE_N = 32;
    const int TILE_K = 160;  // Multiple of 5 for Dense243

    // Zero output
    memset(C, 0, M * N * sizeof(float));

    // Tile over M and N dimensions
    for (int mt = 0; mt < M; mt += TILE_M) {
        int m_block = (mt + TILE_M < M) ? TILE_M : (M - mt);

        for (int nt = 0; nt < N; nt += TILE_N) {
            int n_block = (nt + TILE_N < N) ? TILE_N : (N - nt);

            for (int kt = 0; kt < K; kt += TILE_K) {
                int k_block = (kt + TILE_K < K) ? TILE_K : (K - kt);

                // Process tile
                const float* A_tile = A + mt * K + kt;
                const uint8_t* B_tile = B_packed + (kt / 5) * N + nt;
                float* C_tile = C + mt * N + nt;

                // Use AVX2 kernel on tile
                if (m_block >= 8) {
                    tritnet_gemm_kernel_avx2_8x(A_tile, B_tile, C_tile, k_block, n_block);
                }
                // TODO: Handle smaller tiles
            }
        }
    }
}

#endif // __AVX2__
