/**
 * ternary_gemm_dense.cpp - Dense-Packed Ternary GEMM Implementation
 *
 * Copyright 2025 Ternary Engine Contributors
 * Licensed under the Apache License, Version 2.0
 *
 * See ternary_gemm_dense.h for the design rationale (why dense-packed
 * beats CSC/CSR at ternary's actual sparsity, and why packing fixes a
 * cache-line-stride problem the naive dense approach would otherwise have).
 */

#include "ternary_gemm_dense.h"
#include <stdlib.h>
#include <string.h>
#include <new>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* Batch rows processed per weight-tile load (register-blocking factor).
 * 8 accumulators + the loaded/widened weight vector + one broadcast
 * temporary comfortably fit AVX2's 16 ymm registers without spilling.
 *
 * Was 4 until 2026-08-22: the 2026-08-20 native benchmark
 * (benchmarks/cpp-native-kernels/bench_gemm_dense.cpp) found MB=4 lost
 * to the older CSC/CSR kernel at batch=128 for the smallest shape
 * (Small MLP, 0.45x -- reported, not hidden, in
 * reports/2026-08-20/GEMM_DENSE_PACKED_OPTIMIZATION.md). Swept MB in
 * {4,8,12,16} across all 4 Phase-4 shapes and every batch size the
 * suite tests (1,8,32,128,512): MB=8 is the best single fixed choice --
 * clearly best at the exact batch=128/Small-MLP regression case, tied
 * best or close-second everywhere else. MB=12/16 win narrowly at very
 * large batch (512) but lose more elsewhere, and 16 accumulators (MB=16)
 * plus the weight/broadcast temporaries exceeds AVX2's 16 ymm registers,
 * forcing spills -- consistent with its worse-everywhere results. */
#define TERNARY_GEMM_DENSE_MB 8

/* -------------------------------------------------------------------------
 * Pack / free
 * ---------------------------------------------------------------------- */

TernaryPacked* pack_ternary_dense(const int8_t* B, int K, int N) {
    TernaryPacked* p = (TernaryPacked*)malloc(sizeof(TernaryPacked));
    if (!p) return nullptr;

    p->K = K;
    p->N = N;
    p->n_blocks = (N + 7) / 8;

    size_t total = (size_t)p->n_blocks * K * 8;
    /* calloc: zero-pads the final block's tail columns when N % 8 != 0 */
    p->data = (int8_t*)calloc(total, 1);
    if (!p->data) { free(p); return nullptr; }

    for (int jb = 0; jb < p->n_blocks; jb++) {
        int j0 = jb * 8;
        int jn = std::min(8, N - j0);
        int8_t* dst = p->data + (size_t)jb * K * 8;
        for (int k = 0; k < K; k++)
            memcpy(dst + (size_t)k * 8, B + (size_t)k * N + j0, (size_t)jn);
    }

    return p;
}

void free_ternary_packed(TernaryPacked* p) {
    if (!p) return;
    free(p->data);
    free(p);
}

/* -------------------------------------------------------------------------
 * Scalar kernel
 * ---------------------------------------------------------------------- */

void ternary_gemm_packed_scalar(
    int M, int N, int K,
    const float*          A,
    const TernaryPacked*  packed,
    float*                C
) {
    memset(C, 0, (size_t)M * N * sizeof(float));

    #pragma omp parallel for schedule(static)
    for (int jb = 0; jb < packed->n_blocks; jb++) {
        int j0 = jb * 8;
        int jn = std::min(8, N - j0);
        const int8_t* base = packed->data + (size_t)jb * K * 8;

        for (int m = 0; m < M; m++) {
            const float* a_row = A + (size_t)m * K;
            float acc[8] = {0,0,0,0,0,0,0,0};
            for (int k = 0; k < K; k++) {
                const int8_t* w = base + (size_t)k * 8;
                const float av = a_row[k];
                for (int r = 0; r < jn; r++)
                    acc[r] += av * (float)w[r];
            }
            memcpy(C + (size_t)m * N + j0, acc, (size_t)jn * sizeof(float));
        }
    }
}

/* -------------------------------------------------------------------------
 * AVX2 kernel: N-vectorized (8 cols/register), M-blocked (4 rows/tile)
 * ---------------------------------------------------------------------- */

#ifdef __AVX2__

static inline __m256 widen8_i8_to_ps(const int8_t* p) {
    __m128i b = _mm_loadl_epi64((const __m128i*)p);
    __m256i i32 = _mm256_cvtepi8_epi32(b);
    return _mm256_cvtepi32_ps(i32);
}

void ternary_gemm_packed_avx2(
    int M, int N, int K,
    const float*          A,
    const TernaryPacked*  packed,
    float*                C
) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    const int MB = TERNARY_GEMM_DENSE_MB;

    #pragma omp parallel for schedule(static)
    for (int jb = 0; jb < packed->n_blocks; jb++) {
        int j0 = jb * 8;
        int jn = std::min(8, N - j0);
        const int8_t* base = packed->data + (size_t)jb * K * 8;

        for (int m0 = 0; m0 < M; m0 += MB) {
            int mn = std::min(MB, M - m0);
            __m256 acc[TERNARY_GEMM_DENSE_MB];
            for (int r = 0; r < MB; r++) acc[r] = _mm256_setzero_ps();

            for (int k = 0; k < K; k++) {
                /* Weight tile loaded/widened once per k, reused for all
                 * mn batch rows in this block -- the arithmetic-intensity
                 * gain over a naive per-row reload. */
                __m256 wf = widen8_i8_to_ps(base + (size_t)k * 8);
                for (int r = 0; r < mn; r++) {
                    __m256 av = _mm256_set1_ps(A[(size_t)(m0 + r) * K + k]);
                    acc[r] = _mm256_fmadd_ps(av, wf, acc[r]);
                }
            }

            for (int r = 0; r < mn; r++) {
                float* c_row = C + (size_t)(m0 + r) * N + j0;
                if (jn == 8) {
                    _mm256_storeu_ps(c_row, acc[r]);
                } else {
                    float buf[8];
                    _mm256_storeu_ps(buf, acc[r]);
                    memcpy(c_row, buf, (size_t)jn * sizeof(float));
                }
            }
        }
    }
}

#else /* !__AVX2__ */

void ternary_gemm_packed_avx2(
    int M, int N, int K,
    const float*          A,
    const TernaryPacked*  packed,
    float*                C
) {
    ternary_gemm_packed_scalar(M, N, K, A, packed, C);
}

#endif /* __AVX2__ */

/* -------------------------------------------------------------------------
 * Convenience all-in-one
 * ---------------------------------------------------------------------- */

void ternary_gemm_dense(
    int M, int N, int K,
    const float*  A,
    const int8_t* B,
    float*        C
) {
    TernaryPacked* packed = pack_ternary_dense(B, K, N);
    if (!packed) throw std::bad_alloc();
    ternary_gemm_packed_avx2(M, N, K, A, packed, C);
    free_ternary_packed(packed);
}
