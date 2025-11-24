/**
 * @file tritnet_gemm_naive.cpp
 * @brief Naive (non-SIMD) implementation of TritNet Direct Ternary GEMM
 *
 * This is the reference implementation for correctness validation.
 * Performance: ~1-2 Gops/s (to be replaced with SIMD version)
 *
 * @date 2025-11-23
 */

#include "tritnet_gemm.h"
#include "ternary_dense243.h"  // Resolved via ternary_engine/lib/dense243 in include_dirs
#include <string.h>
#include <stdlib.h>

// Platform-specific aligned memory allocation (must be defined early)
#if defined(_WIN32)
    #define aligned_alloc_impl(alignment, size) _aligned_malloc(size, alignment)
    #define aligned_free_impl(ptr) _aligned_free(ptr)
#else
    static inline void* aligned_alloc_impl(size_t alignment, size_t size) {
        void* ptr = nullptr;
        posix_memalign(&ptr, alignment, size);
        return ptr;
    }
    #define aligned_free_impl(ptr) free(ptr)
#endif

// Configuration
static int g_num_threads = 0;  // 0 = auto-detect

/**
 * @brief Unpack 5 ternary weights from Dense243 byte
 *
 * This is the core unpacking operation that extracts 5 trits from our
 * Dense243 encoding.
 *
 * @param packed  Dense243-encoded byte (0-242)
 * @param trits   Output array [5] for unpacked trits {-1, 0, +1}
 */
static inline void unpack_dense243_5(uint8_t packed, int8_t* trits) {
    // Dense243 unpacking (see dense243.cpp for encoding details)
    // This maps 0-242 → 5 trits in base-3

    int value = packed;

    // Extract 5 trits using division by 3
    for (int i = 0; i < 5; i++) {
        int remainder = value % 3;
        value /= 3;

        // Map 0,1,2 → -1,0,+1
        trits[i] = (int8_t)(remainder - 1);
    }
}

/**
 * @brief Naive ternary GEMM implementation
 *
 * Triple-nested loop implementation for correctness validation.
 * No SIMD, no tiling, just straightforward computation.
 *
 * Algorithm:
 *   for m in 0..M:
 *     for n in 0..N:
 *       acc = 0
 *       for k in 0..K (step 5):
 *         unpack 5 weights from B_packed
 *         for each trit:
 *           if trit == +1: acc += A[m,k]
 *           if trit == -1: acc -= A[m,k]
 *           if trit ==  0: skip (free multiply by zero)
 *       C[m,n] = acc
 */
void tritnet_gemm_f32(
    int M,
    int N,
    int K,
    const float* A,
    const uint8_t* B_packed,
    float* C
) {
    // Zero output matrix
    memset(C, 0, M * N * sizeof(float));

    // Ensure K is multiple of 5 for correct unpacking
    if (K % 5 != 0) {
        // TODO: Handle K not multiple of 5 with padding
        return;
    }

    const int K_packed = K / 5;  // Number of Dense243 bytes per column

    // Triple-nested loop: M × N × K
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;

            // Inner loop: accumulate over K dimension
            for (int k_pack = 0; k_pack < K_packed; k_pack++) {
                // Get Dense243-packed byte for this position
                uint8_t packed = B_packed[k_pack * N + n];

                // Unpack 5 ternary weights
                int8_t trits[5];
                unpack_dense243_5(packed, trits);

                // Ternary multiply-accumulate
                for (int i = 0; i < 5; i++) {
                    int k = k_pack * 5 + i;
                    float a_val = A[m * K + k];

                    if (trits[i] == 1) {
                        acc += a_val;           // +1 weight
                    } else if (trits[i] == -1) {
                        acc -= a_val;           // -1 weight
                    }
                    // else trits[i] == 0: skip (free multiply by zero)
                }
            }

            C[m * N + n] = acc;
        }
    }
}

/**
 * @brief Ternary GEMM with per-column scaling
 *
 * Same as tritnet_gemm_f32 but applies scale factors to outputs.
 * Matches BitNet's per-block quantization scheme.
 */
void tritnet_gemm_f32_scaled(
    int M,
    int N,
    int K,
    const float* A,
    const uint8_t* B_packed,
    const float* scales,
    float* C
) {
    // First compute unscaled GEMM
    tritnet_gemm_f32(M, N, K, A, B_packed, C);

    // Apply per-column scales
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            C[m * N + n] *= scales[n];
        }
    }
}

/**
 * @brief Convert BitNet 2-bit weights to Dense243 format
 *
 * BitNet encoding: Each byte contains 4 trits (2 bits each)
 *   -1 → 00
 *    0 → 01
 *   +1 → 10
 *
 * Dense243 encoding: Each byte contains 5 trits
 *   value in 0-242 represents 5 trits in base-3
 */
void convert_bitnet_to_dense243(
    const uint8_t* bitnet_weights,
    uint8_t* dense243_out,
    int K,
    int N
) {
    // Process in groups of 20 weights (LCM of 4 and 5)
    // 20 weights: BitNet uses 5 bytes, Dense243 uses 4 bytes
    // Savings: 20% fewer bytes

    const int K_bitnet = (K + 3) / 4;  // Round up
    const int K_dense243 = (K + 4) / 5;  // Round up

    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k += 5) {
            // Collect 5 ternary weights from BitNet format
            int8_t trits[5] = {0};

            for (int i = 0; i < 5 && (k + i) < K; i++) {
                int k_idx = k + i;
                int byte_idx = k_idx / 4;
                int trit_idx = k_idx % 4;

                // Extract 2-bit value from BitNet byte
                uint8_t bitnet_byte = bitnet_weights[byte_idx * N + n];
                uint8_t two_bits = (bitnet_byte >> (trit_idx * 2)) & 0x03;

                // Convert BitNet encoding to trit {-1, 0, +1}
                switch (two_bits) {
                    case 0b00: trits[i] = -1; break;
                    case 0b01: trits[i] =  0; break;
                    case 0b10: trits[i] = +1; break;
                    default:   trits[i] =  0; break;  // Invalid, treat as 0
                }
            }

            // Pack 5 trits into Dense243 byte
            int value = 0;
            int multiplier = 1;
            for (int i = 0; i < 5; i++) {
                // Map {-1, 0, +1} → {0, 1, 2}
                int digit = trits[i] + 1;
                value += digit * multiplier;
                multiplier *= 3;
            }

            // Store Dense243 byte
            int k_pack = k / 5;
            dense243_out[k_pack * N + n] = (uint8_t)value;
        }
    }
}

/**
 * @brief Set number of threads for parallel GEMM
 */
void tritnet_set_num_threads(int n_threads) {
    g_num_threads = n_threads;
    // TODO: Actually use this for OpenMP parallelization
}

/**
 * @brief Get optimal tile size for cache tiling
 */
int tritnet_get_optimal_tile_size(int cache_level) {
    // Conservative tile sizes for L1/L2/L3
    switch (cache_level) {
        case 1: return 32;   // L1: 32KB typical
        case 2: return 128;  // L2: 256KB typical
        case 3: return 512;  // L3: 8MB typical
        default: return 64;
    }
}

/**
 * @brief Benchmark GEMM performance
 */
double tritnet_benchmark_gemm(int M, int N, int K, int num_runs) {
    // Allocate test matrices
    float* A = (float*)aligned_alloc_impl(64, M * K * sizeof(float));
    uint8_t* B = (uint8_t*)aligned_alloc_impl(64, (K / 5) * N * sizeof(uint8_t));
    float* C = (float*)aligned_alloc_impl(64, M * N * sizeof(float));

    // Initialize with random data
    for (int i = 0; i < M * K; i++) {
        A[i] = (float)(rand() % 200 - 100) / 100.0f;
    }
    for (int i = 0; i < (K / 5) * N; i++) {
        B[i] = (uint8_t)(rand() % 243);  // Valid Dense243 values
    }

    // Warm-up run
    tritnet_gemm_f32(M, N, K, A, B, C);

    // Benchmark runs
    // TODO: Add actual timing with high-resolution clock

    // Placeholder: return 0 for now
    double avg_time_ms = 0.0;

    // Cleanup
    aligned_free_impl(A);
    aligned_free_impl(B);
    aligned_free_impl(C);

    return avg_time_ms;
}

/**
 * @brief Validate GEMM correctness against reference
 */
float tritnet_validate_gemm(int M, int N, int K) {
    // Allocate test matrices
    float* A = (float*)aligned_alloc_impl(64, M * K * sizeof(float));
    uint8_t* B = (uint8_t*)aligned_alloc_impl(64, (K / 5) * N * sizeof(uint8_t));
    float* C_test = (float*)aligned_alloc_impl(64, M * N * sizeof(float));
    float* C_ref = (float*)aligned_alloc_impl(64, M * N * sizeof(float));

    // Initialize with known data
    for (int i = 0; i < M * K; i++) {
        A[i] = (float)(i % 10) / 10.0f;
    }
    for (int i = 0; i < (K / 5) * N; i++) {
        B[i] = (uint8_t)(i % 243);
    }

    // Compute reference (naive implementation)
    tritnet_gemm_f32(M, N, K, A, B, C_ref);

    // Compute test (same for now, but would be SIMD version later)
    tritnet_gemm_f32(M, N, K, A, B, C_test);

    // Find maximum absolute error
    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabs(C_test[i] - C_ref[i]);
        if (err > max_error) {
            max_error = err;
        }
    }

    // Cleanup
    aligned_free_impl(A);
    aligned_free_impl(B);
    aligned_free_impl(C_test);
    aligned_free_impl(C_ref);

    return max_error;
}
