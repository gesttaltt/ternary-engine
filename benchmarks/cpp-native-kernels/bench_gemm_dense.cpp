/**
 * @file bench_gemm_dense.cpp
 * @brief Native (pybind11-free) comparison of ternary GEMM kernels
 *
 * Isolates the kernel from Python/pybind11 overhead per this project's
 * ffi_isolation convention (.claude/CLAUDE.md: "Absolute performance claims
 * must be measured in native C++ ... to isolate pybind11 overhead"). No
 * such native benchmark existed for src/core/simd/ternary_gemm_zero_skip.*
 * or ternary_gemm_dense.* before this file (2026-08-20).
 *
 * Compares, at the real shapes benchmarks/python-with-interpreter-overhead/
 * bench_competitive.py's Phase 4 uses:
 *   - ternary_gemm_zero_skip_scalar / _avx2 (CSC, j-parallel)
 *   - ternary_gemm_zero_skip_tiled          (CSR, k-parallel, cache-tiled)
 *   - ternary_gemm_packed_scalar / _avx2    (dense-packed, added 2026-08-20)
 *
 * Background: the competitive suite's Phase 4 measured 0.189x avg vs NumPy
 * for the zero-skip kernels. Investigation found two structural problems:
 * (1) Phase 4 tests batch=1 (pure GEMV), where the zero-skip kernels'
 * 8-wide AVX2 SIMD -- vectorized over the batch/M dimension -- can never
 * fire; (2) CSC/CSR index storage (4B index + 1B sign per non-zero) is
 * LARGER than the dense int8 array itself at ternary's actual ~33-40%
 * zero density, so the "zero-skip" kernels move more bytes than a naive
 * dense kernel would, despite doing fewer FLOPs -- and these kernels are
 * memory-bandwidth-bound, not compute-bound. See ternary_gemm_dense.h for
 * the full analysis and reports/2026-08-20/GEMM_DENSE_PACKED_OPTIMIZATION.md
 * for the investigation.
 *
 * COMPILATION (from benchmarks/cpp-native-kernels/ directory):
 *   g++ -O3 -march=haswell -mavx2 -mfma -fopenmp -std=c++17 \
 *       -I../../src/core/simd \
 *       bench_gemm_dense.cpp \
 *       ../../src/core/simd/ternary_gemm_zero_skip.cpp \
 *       ../../src/core/simd/ternary_gemm_dense.cpp \
 *       -o bench_gemm_dense
 *
 * USAGE: ./bench_gemm_dense
 * OUTPUT: GOPS for each kernel at each (shape, batch) cell; correctness
 *         check against a float64 scalar reference before any timing.
 */

#include "ternary_gemm_zero_skip.h"
#include "ternary_gemm_dense.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <functional>

using Clock = std::chrono::high_resolution_clock;

static double reference_maxerr(int M, int N, int K,
                                const std::vector<float>& A,
                                const std::vector<int8_t>& B,
                                const std::vector<float>& C) {
    double maxerr = 0;
    for (int m = 0; m < M; m++) {
        for (int j = 0; j < N; j++) {
            double acc = 0;
            for (int k = 0; k < K; k++)
                acc += (double)A[(size_t)m * K + k] * (double)B[(size_t)k * N + j];
            maxerr = std::max(maxerr, std::fabs(acc - (double)C[(size_t)m * N + j]));
        }
    }
    return maxerr;
}

struct TimingResult { double ms; double gops; };

static TimingResult time_it(int M, int N, int K, int iters,
                             const std::function<void()>& fn) {
    for (int i = 0; i < std::min(5, iters); i++) fn();
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) fn();
    auto t1 = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    double gops = (double)M * N * K / (ms * 1e6);
    return {ms, gops};
}

int main() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> tri(-1, 1);
    std::normal_distribution<float> norm(0.0f, 1.0f);

    printf("================================================================================\n");
    printf("Native GEMM Kernel Comparison (pybind11-free) -- 2026-08-20\n");
    printf("================================================================================\n\n");

    struct Config { const char* name; int M_out, N_in; };
    // (M_out, N_in) match bench_competitive.py Phase 4's (weight rows, weight cols)
    Config configs[] = {
        {"Small MLP",       512,  512},
        {"Medium Layer",    2048, 2048},
        {"Large Layer",     4096, 4096},
        {"Attention Head",  8192, 1024},
    };
    int batches[] = {1, 8, 32, 128};

    for (auto& cfg : configs) {
        int N = cfg.M_out;   // weight output dim -> GEMM's N
        int K = cfg.N_in;    // weight input dim  -> GEMM's K
        std::vector<int8_t> B((size_t)K * N);
        for (auto& v : B) v = (int8_t)tri(rng);

        TernaryCSC* csc = build_ternary_csc(B.data(), K, N);
        TernaryCSR* csr = build_ternary_csr(B.data(), K, N);
        TernaryPacked* packed = pack_ternary_dense(B.data(), K, N);
        double zero_frac = 1.0 - (double)csc->nnz / ((double)K * N);
        long long csr_csc_bytes = (long long)csc->nnz * (4 + 1) + (long long)(N + 1) * 4;
        long long packed_bytes = (long long)packed->n_blocks * K * 8;

        printf("--- %s (N=%d, K=%d, zero_frac=%.3f) ---\n", cfg.name, N, K, zero_frac);
        printf("    CSC index storage: %lld bytes | packed dense storage: %lld bytes | ratio: %.2fx\n",
               csr_csc_bytes, packed_bytes, (double)csr_csc_bytes / packed_bytes);

        for (int M : batches) {
            std::vector<float> A((size_t)M * K);
            for (auto& v : A) v = norm(rng);
            std::vector<float> C((size_t)M * N);

            // Correctness (once per shape, cheap relative to timing loop)
            ternary_gemm_packed_avx2(M, N, K, A.data(), packed, C.data());
            double err = reference_maxerr(M, N, K, A, B, C);

            auto r_skip_avx2 = time_it(M, N, K, 100, [&]{
                ternary_gemm_zero_skip_avx2(M, N, K, A.data(), csc, C.data());
            });
            auto r_skip_tiled = time_it(M, N, K, 100, [&]{
                ternary_gemm_zero_skip_tiled(M, N, K, A.data(), csr, C.data());
            });
            auto r_dense_avx2 = time_it(M, N, K, 100, [&]{
                ternary_gemm_packed_avx2(M, N, K, A.data(), packed, C.data());
            });

            printf("    batch=%4d  skip_avx2=%9.4fms(%7.2f GOPS)  skip_tiled=%9.4fms(%7.2f GOPS)  "
                   "dense_avx2=%9.4fms(%7.2f GOPS)  dense_speedup_vs_best_skip=%.2fx  maxerr=%.2e\n",
                   M,
                   r_skip_avx2.ms, r_skip_avx2.gops,
                   r_skip_tiled.ms, r_skip_tiled.gops,
                   r_dense_avx2.ms, r_dense_avx2.gops,
                   std::min(r_skip_avx2.ms, r_skip_tiled.ms) / r_dense_avx2.ms,
                   err);
        }
        printf("\n");

        free_ternary_csc(csc);
        free_ternary_csr(csr);
        free_ternary_packed(packed);
    }

    return 0;
}
