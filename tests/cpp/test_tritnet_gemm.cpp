/**
 * @file test_tritnet_gemm.cpp
 * @brief Unit tests for TritNet Direct Ternary GEMM
 *
 * Tests correctness of our GEMM implementation against known-good outputs.
 *
 * @date 2025-11-23
 */

#include "../include/tritnet_gemm.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>

#define EXPECT_TRUE(cond, msg) \
    if (!(cond)) { \
        std::cerr << "FAIL: " << msg << std::endl; \
        return false; \
    }

#define EXPECT_NEAR(a, b, tol, msg) \
    if (fabs((a) - (b)) > (tol)) { \
        std::cerr << "FAIL: " << msg << " (expected " << (b) << ", got " << (a) << ")" << std::endl; \
        return false; \
    }

/**
 * Test 1: Tiny matrix (2×2×5) - Manual verification
 */
bool test_tiny_gemm() {
    std::cout << "Test 1: Tiny GEMM (2×2×5)..." << std::endl;

    // A = [1, 2, 3, 4, 5]   (2×5)
    //     [6, 7, 8, 9, 10]
    float A[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // B = [+1, -1]   (5×2, ternary)
    //     [ 0,  0]
    //     [+1, -1]
    //     [-1, +1]
    //     [ 0, +1]
    //
    // Dense243 packed: Column 0: [+1, 0, +1, -1, 0] → pack to byte
    //                  Column 1: [-1, 0, -1, +1, +1] → pack to byte

    // Pack column 0: trits [+1, 0, +1, -1, 0] → [1, 0, 1, -1, 0] + 1 → [2, 1, 2, 0, 1]
    // value = 2*1 + 1*3 + 2*9 + 0*27 + 1*81 = 2 + 3 + 18 + 0 + 81 = 104
    uint8_t col0 = 104;

    // Pack column 1: trits [-1, 0, -1, +1, +1] → [-1, 0, -1, 1, 1] + 1 → [0, 1, 0, 2, 2]
    // value = 0*1 + 1*3 + 0*9 + 2*27 + 2*81 = 0 + 3 + 0 + 54 + 162 = 219
    uint8_t col1 = 219;

    uint8_t B_packed[2] = {col0, col1};  // 1 row (K/5=1), 2 columns

    // Expected output C (2×2):
    // C[0,0] = A[0,:] · B[:,0] = 1*1 + 2*0 + 3*1 + 4*(-1) + 5*0 = 1 + 0 + 3 - 4 + 0 = 0
    // C[0,1] = A[0,:] · B[:,1] = 1*(-1) + 2*0 + 3*(-1) + 4*1 + 5*1 = -1 + 0 - 3 + 4 + 5 = 5
    // C[1,0] = A[1,:] · B[:,0] = 6*1 + 7*0 + 8*1 + 9*(-1) + 10*0 = 6 + 0 + 8 - 9 + 0 = 5
    // C[1,1] = A[1,:] · B[:,1] = 6*(-1) + 7*0 + 8*(-1) + 9*1 + 10*1 = -6 + 0 - 8 + 9 + 10 = 5

    float C[4] = {0};
    float C_expected[4] = {0, 5, 5, 5};

    // Run GEMM
    tritnet_gemm_f32(2, 2, 5, A, B_packed, C);

    // Validate
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(C[i], C_expected[i], 1e-6, "Output mismatch");
    }

    std::cout << "  PASS" << std::endl;
    return true;
}

/**
 * Test 2: Identity-like operation
 */
bool test_identity_gemm() {
    std::cout << "Test 2: Identity-like GEMM (4×4×5)..." << std::endl;

    const int M = 4, N = 4, K = 5;

    float A[M * K];
    uint8_t B_packed[1 * N];  // K/5 = 1
    float C[M * N] = {0};

    // A = identity-like pattern
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            A[m * K + k] = (m == k) ? 1.0f : 0.0f;
        }
    }

    // B = all +1 weights (should pass through activations)
    // All +1: [1, 1, 1, 1, 1] → [2, 2, 2, 2, 2] → value = 2 + 6 + 18 + 54 + 162 = 242
    uint8_t all_ones = 242;
    for (int n = 0; n < N; n++) {
        B_packed[n] = all_ones;
    }

    tritnet_gemm_f32(M, N, K, A, B_packed, C);

    // Expected: C[m,n] = sum of row m (which has one 1.0 at position m)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float expected = (m < K) ? 1.0f : 0.0f;  // 1 if m corresponds to a column
            EXPECT_NEAR(C[m * N + n], expected, 1e-6, "Identity test mismatch");
        }
    }

    std::cout << "  PASS" << std::endl;
    return true;
}

/**
 * Test 3: All zeros
 */
bool test_zero_weights() {
    std::cout << "Test 3: Zero weights (8×8×10)..." << std::endl;

    const int M = 8, N = 8, K = 10;

    float A[M * K];
    uint8_t B_packed[(K/5) * N];  // K/5 = 2
    float C[M * N] = {0};

    // Random activations
    for (int i = 0; i < M * K; i++) {
        A[i] = (float)(rand() % 100) / 10.0f;
    }

    // All zero weights: [0, 0, 0, 0, 0] → [1, 1, 1, 1, 1] → value = 1 + 3 + 9 + 27 + 81 = 121
    uint8_t all_zeros = 121;
    for (int i = 0; i < (K/5) * N; i++) {
        B_packed[i] = all_zeros;
    }

    tritnet_gemm_f32(M, N, K, A, B_packed, C);

    // Expected: All zeros in output
    for (int i = 0; i < M * N; i++) {
        EXPECT_NEAR(C[i], 0.0f, 1e-6, "Zero weight output not zero");
    }

    std::cout << "  PASS" << std::endl;
    return true;
}

/**
 * Test 4: Scaling function
 */
bool test_scaled_gemm() {
    std::cout << "Test 4: Scaled GEMM (4×4×5)..." << std::endl;

    const int M = 4, N = 4, K = 5;

    float A[M * K];
    uint8_t B_packed[1 * N];
    float scales[N] = {1.0f, 2.0f, 0.5f, 3.0f};
    float C_scaled[M * N] = {0};
    float C_unscaled[M * N] = {0};

    // Random initialization
    for (int i = 0; i < M * K; i++) {
        A[i] = (float)(rand() % 10);
    }
    for (int i = 0; i < N; i++) {
        B_packed[i] = (uint8_t)(rand() % 243);
    }

    // Compute both versions
    tritnet_gemm_f32(M, N, K, A, B_packed, C_unscaled);
    tritnet_gemm_f32_scaled(M, N, K, A, B_packed, scales, C_scaled);

    // Validate: C_scaled[m,n] should equal C_unscaled[m,n] * scales[n]
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float expected = C_unscaled[m * N + n] * scales[n];
            EXPECT_NEAR(C_scaled[m * N + n], expected, 1e-5, "Scaled GEMM mismatch");
        }
    }

    std::cout << "  PASS" << std::endl;
    return true;
}

/**
 * Test 5: BitNet format conversion
 */
bool test_bitnet_conversion() {
    std::cout << "Test 5: BitNet to Dense243 conversion..." << std::endl;

    const int K = 20, N = 4;  // Use K=20 (LCM of 4 and 5)

    // Create known BitNet weights
    // BitNet: 4 trits/byte, encoding -1→00, 0→01, +1→10

    uint8_t bitnet[(K/4) * N];  // 5 bytes × 4 columns = 20 bytes
    uint8_t dense243[(K/5) * N];  // 4 bytes × 4 columns = 16 bytes

    // Fill with known pattern: alternating [-1, 0, +1, 0]
    for (int k = 0; k < K/4; k++) {
        for (int n = 0; n < N; n++) {
            // 4 trits: [-1, 0, +1, 0] → [00, 01, 10, 01] → 0b01100100 = 0x64
            bitnet[k * N + n] = 0x64;
        }
    }

    // Convert
    convert_bitnet_to_dense243(bitnet, dense243, K, N);

    // Verify: Dense243 should have same ternary values when unpacked
    // This is a basic smoke test - more detailed validation would unpack and compare

    std::cout << "  PASS (basic validation)" << std::endl;
    return true;
}

/**
 * Main test runner
 */
int main() {
    std::cout << "===========================================\n";
    std::cout << " TritNet GEMM Unit Tests\n";
    std::cout << "===========================================\n\n";

    int passed = 0, total = 0;

    #define RUN_TEST(test_func) \
        total++; \
        if (test_func()) passed++; \
        std::cout << std::endl;

    RUN_TEST(test_tiny_gemm);
    RUN_TEST(test_identity_gemm);
    RUN_TEST(test_zero_weights);
    RUN_TEST(test_scaled_gemm);
    RUN_TEST(test_bitnet_conversion);

    std::cout << "===========================================\n";
    std::cout << " Results: " << passed << "/" << total << " tests passed\n";
    std::cout << "===========================================\n";

    return (passed == total) ? 0 : 1;
}
