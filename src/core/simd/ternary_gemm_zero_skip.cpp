/**
 * ternary_gemm_zero_skip.cpp - Zero-Skip Ternary GEMM Implementation
 *
 * Copyright 2025 Ternary Engine Contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Two paths:
 *   Scalar: clean loop, no SIMD.  Useful baseline and fallback.
 *   AVX2:   processes 8 M-rows simultaneously via gather + FMA.
 *
 * Both paths use the same CSC sparse index structure built by
 * build_ternary_csc().  The precomputed index eliminates:
 *   - Dense243 unpack cost (division-by-3 loop)
 *   - Per-weight branch (if trit == 0 continue)
 *   - ~33% of multiply-accumulate ops
 */

#include "ternary_gemm_zero_skip.h"
#include <stdlib.h>
#include <string.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* -------------------------------------------------------------------------
 * CSC build / free
 * ---------------------------------------------------------------------- */

TernaryCSC* build_ternary_csc(const int8_t* B, int K, int N) {
    TernaryCSC* csc = (TernaryCSC*)malloc(sizeof(TernaryCSC));
    csc->K = K;
    csc->N = N;

    /* Count non-zeros */
    int nnz = 0;
    for (int i = 0; i < K * N; i++) {
        if (B[i] != 0) nnz++;
    }
    csc->nnz = nnz;

    csc->col_ptr = (int*)malloc((N + 1) * sizeof(int));
    csc->row_idx = (int*)malloc(nnz * sizeof(int));
    csc->signs   = (int8_t*)malloc(nnz * sizeof(int8_t));

    /* Fill CSC: B is [K, N] row-major, so B[k, j] = B[k*N + j] */
    int ptr = 0;
    csc->col_ptr[0] = 0;
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < K; k++) {
            int8_t v = B[k * N + j];
            if (v != 0) {
                csc->row_idx[ptr] = k;
                csc->signs[ptr]   = v;
                ptr++;
            }
        }
        csc->col_ptr[j + 1] = ptr;
    }

    return csc;
}

void free_ternary_csc(TernaryCSC* csc) {
    if (!csc) return;
    free(csc->col_ptr);
    free(csc->row_idx);
    free(csc->signs);
    free(csc);
}

/* -------------------------------------------------------------------------
 * Scalar kernel
 *
 * Naive loop order (j → non-zero k → m) hits a cache problem: A[:,k] and
 * C[:,j] are stride-K and stride-N in row-major layout, causing one cache
 * miss per row for large K.
 *
 * Fix: transpose A → AT[K,M] and accumulate into CT[N,M].  Both inner
 * loops then read/write sequential memory.  The two transposes cost
 * O(M*K + M*N) and are dominated by the GEMM for large K.
 * ---------------------------------------------------------------------- */

void ternary_gemm_zero_skip_scalar(
    int M, int N, int K,
    const float*      A,
    const TernaryCSC* B_csc,
    float*            C
) {
    /* Transpose A[M,K] → AT[K,M] */
    float* AT = (float*)malloc((size_t)K * M * sizeof(float));
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++)
            AT[k * M + m] = A[m * K + k];

    /* Accumulate into CT[N,M] (column-major output) */
    float* CT = (float*)calloc((size_t)N * M, sizeof(float));

    for (int j = 0; j < N; j++) {
        const int p0 = B_csc->col_ptr[j];
        const int p1 = B_csc->col_ptr[j + 1];
        float* ct_row = CT + (size_t)j * M;

        for (int p = p0; p < p1; p++) {
            const int   k     = B_csc->row_idx[p];
            const float fsign = (float)B_csc->signs[p];
            const float* at_row = AT + (size_t)k * M;

            /* Sequential SAXPY: CT[j,:] += fsign * AT[k,:] */
            for (int m = 0; m < M; m++)
                ct_row[m] += fsign * at_row[m];
        }
    }

    /* Transpose CT[N,M] → C[M,N] */
    for (int m = 0; m < M; m++)
        for (int j = 0; j < N; j++)
            C[m * N + j] = CT[(size_t)j * M + m];

    free(AT);
    free(CT);
}

/* -------------------------------------------------------------------------
 * AVX2 kernel
 *
 * Same transpose strategy as scalar, but the inner SAXPY is vectorised:
 *   CT[j,:] += sign * AT[k,:]   →  8 floats per iteration via FMA
 *
 * All loads and stores are sequential, so the hardware prefetcher works
 * optimally.  No gather needed.
 * ---------------------------------------------------------------------- */

#ifdef __AVX2__

void ternary_gemm_zero_skip_avx2(
    int M, int N, int K,
    const float*      A,
    const TernaryCSC* B_csc,
    float*            C
) {
    /* Transpose A[M,K] → AT[K,M] */
    float* AT = (float*)malloc((size_t)K * M * sizeof(float));
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++)
            AT[k * M + m] = A[m * K + k];

    /* Accumulate into CT[N,M] */
    float* CT = (float*)calloc((size_t)N * M, sizeof(float));

    for (int j = 0; j < N; j++) {
        const int p0 = B_csc->col_ptr[j];
        const int p1 = B_csc->col_ptr[j + 1];
        float* ct_row = CT + (size_t)j * M;

        for (int p = p0; p < p1; p++) {
            const int    k     = B_csc->row_idx[p];
            const int8_t sign  = B_csc->signs[p];
            const __m256 vsign = (sign > 0) ? _mm256_set1_ps(1.0f)
                                            : _mm256_set1_ps(-1.0f);
            const float* at_row = AT + (size_t)k * M;

            int m = 0;
            /* 8-wide AVX2 FMA, fully sequential */
            for (; m + 8 <= M; m += 8) {
                __m256 av = _mm256_loadu_ps(at_row + m);
                __m256 cv = _mm256_loadu_ps(ct_row + m);
                cv = _mm256_fmadd_ps(vsign, av, cv);
                _mm256_storeu_ps(ct_row + m, cv);
            }
            /* Scalar tail */
            const float fsign = (float)sign;
            for (; m < M; m++)
                ct_row[m] += fsign * at_row[m];
        }
    }

    /* Transpose CT[N,M] → C[M,N] */
    for (int m = 0; m < M; m++)
        for (int j = 0; j < N; j++)
            C[m * N + j] = CT[(size_t)j * M + m];

    free(AT);
    free(CT);
}

#else  /* !__AVX2__ */

void ternary_gemm_zero_skip_avx2(
    int M, int N, int K,
    const float*      A,
    const TernaryCSC* B_csc,
    float*            C
) {
    /* Fallback on non-AVX2 builds */
    ternary_gemm_zero_skip_scalar(M, N, K, A, B_csc, C);
}

#endif

/* -------------------------------------------------------------------------
 * Convenience all-in-one
 * ---------------------------------------------------------------------- */

void ternary_gemm_zero_skip(
    int M, int N, int K,
    const float*  A,
    const int8_t* B,
    float*        C
) {
    TernaryCSC* csc = build_ternary_csc(B, K, N);
    ternary_gemm_zero_skip_avx2(M, N, K, A, csc, C);
    free_ternary_csc(csc);
}
