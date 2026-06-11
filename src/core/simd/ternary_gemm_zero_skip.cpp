/**
 * ternary_gemm_zero_skip.cpp - Zero-Skip Ternary GEMM Implementation
 *
 * Copyright 2025 Ternary Engine Contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Three kernels:
 *   Scalar:  baseline, no SIMD.
 *   AVX2:    j-parallel; all threads share AT in L3.  Scales ~3× from 8 cores
 *            at large M due to AT bandwidth saturation.
 *   Tiled:   k-parallel; each thread owns a k-stripe that fits in L2, then
 *            reduces thread-private CT_local arrays.  Avoids the shared-AT
 *            bottleneck to improve multi-core scaling.
 */

#include "ternary_gemm_zero_skip.h"
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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

    int nnz = 0;
    for (int i = 0; i < K * N; i++) {
        if (B[i] != 0) nnz++;
    }
    csc->nnz = nnz;

    csc->col_ptr = (int*)malloc((N + 1) * sizeof(int));
    csc->row_idx = (int*)malloc(nnz * sizeof(int));
    csc->signs   = (int8_t*)malloc(nnz * sizeof(int8_t));

    /* B is [K, N] row-major: B[k, j] = B[k*N + j] */
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
 * CSR build / free
 *
 * Row-indexed format for the tiled kernel.
 * row_ptr[k]..row_ptr[k+1]-1 indexes all non-zeros in activation row k.
 * col_idx[p] = j (output column), signs[p] = +1/-1.
 *
 * The tiled kernel loads AT[k,:] once per k-row and reuses it for every j
 * in that row, keeping the row hot in L1 across many SAXPY calls.
 * ---------------------------------------------------------------------- */

TernaryCSR* build_ternary_csr(const int8_t* B, int K, int N) {
    TernaryCSR* csr = (TernaryCSR*)malloc(sizeof(TernaryCSR));
    csr->K = K;
    csr->N = N;

    int nnz = 0;
    for (int i = 0; i < K * N; i++) {
        if (B[i] != 0) nnz++;
    }
    csr->nnz = nnz;

    csr->row_ptr = (int*)malloc((K + 1) * sizeof(int));
    csr->col_idx = (int*)malloc(nnz * sizeof(int));
    csr->signs   = (int8_t*)malloc(nnz * sizeof(int8_t));

    int ptr = 0;
    csr->row_ptr[0] = 0;
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < N; j++) {
            int8_t v = B[k * N + j];
            if (v != 0) {
                csr->col_idx[ptr] = j;
                csr->signs[ptr]   = v;
                ptr++;
            }
        }
        csr->row_ptr[k + 1] = ptr;
    }

    return csr;
}

void free_ternary_csr(TernaryCSR* csr) {
    if (!csr) return;
    free(csr->row_ptr);
    free(csr->col_idx);
    free(csr->signs);
    free(csr);
}

/* -------------------------------------------------------------------------
 * Scalar kernel (j-parallel, CSC)
 *
 * Transpose A → AT[K,M] and accumulate into CT[N,M] to convert strided
 * A[:,k] / C[:,j] accesses into sequential SAXPY rows.
 * ---------------------------------------------------------------------- */

void ternary_gemm_zero_skip_scalar(
    int M, int N, int K,
    const float*      A,
    const TernaryCSC* B_csc,
    float*            C
) {
    float* AT = (float*)malloc((size_t)K * M * sizeof(float));
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++)
            AT[k * M + m] = A[m * K + k];

    float* CT = (float*)calloc((size_t)N * M, sizeof(float));

    #pragma omp parallel for schedule(dynamic, 8)
    for (int j = 0; j < N; j++) {
        const int p0 = B_csc->col_ptr[j];
        const int p1 = B_csc->col_ptr[j + 1];
        float* ct_row = CT + (size_t)j * M;

        for (int p = p0; p < p1; p++) {
            const int   k     = B_csc->row_idx[p];
            const float fsign = (float)B_csc->signs[p];
            const float* at_row = AT + (size_t)k * M;
            for (int m = 0; m < M; m++)
                ct_row[m] += fsign * at_row[m];
        }
    }

    #pragma omp parallel for
    for (int m = 0; m < M; m++)
        for (int j = 0; j < N; j++)
            C[m * N + j] = CT[(size_t)j * M + m];

    free(AT);
    free(CT);
}

/* -------------------------------------------------------------------------
 * AVX2 j-parallel kernel (CSC, OpenMP over j)
 *
 * Same transpose strategy; inner SAXPY vectorised to 8 floats/iter via FMA.
 * Bottleneck at large M: all threads read the same AT[K,M] from L3.
 * ---------------------------------------------------------------------- */

#ifdef __AVX2__

void ternary_gemm_zero_skip_avx2(
    int M, int N, int K,
    const float*      A,
    const TernaryCSC* B_csc,
    float*            C
) {
    float* AT = (float*)malloc((size_t)K * M * sizeof(float));
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++)
            AT[k * M + m] = A[m * K + k];

    float* CT = (float*)calloc((size_t)N * M, sizeof(float));

    #pragma omp parallel for schedule(dynamic, 8)
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
            for (; m + 8 <= M; m += 8) {
                __m256 av = _mm256_loadu_ps(at_row + m);
                __m256 cv = _mm256_loadu_ps(ct_row + m);
                cv = _mm256_fmadd_ps(vsign, av, cv);
                _mm256_storeu_ps(ct_row + m, cv);
            }
            const float fsign = (float)sign;
            for (; m < M; m++)
                ct_row[m] += fsign * at_row[m];
        }
    }

    #pragma omp parallel for
    for (int m = 0; m < M; m++)
        for (int j = 0; j < N; j++)
            C[m * N + j] = CT[(size_t)j * M + m];

    free(AT);
    free(CT);
}

/* -------------------------------------------------------------------------
 * AVX2 k-parallel tiled kernel (CSR, OpenMP over k, thread-private CT)
 *
 * Why this beats j-parallel at large M:
 *   j-parallel: 8 threads each read random k-rows from AT[K,M] = 1 MB.
 *     All share the same AT in L3; ~3× scaling from 8 cores.
 *   k-parallel: each thread gets a static k-stripe.  For K=1024, 8 threads:
 *     AT stripe per thread = 128 rows × M×4B = 128 × 256 × 4 = 128 KB → L2.
 *     AT[k,:] (1 KB) stays in L1 for all N/3 non-zeros in that k-row.
 *     Thread-private CT_local[N×M] = 256 KB fits in L2.
 *     No AT bandwidth contention between threads.
 *
 * Trade-off: extra work = reduction of nthreads×N×M floats (AVX2 add).
 *   For 256×1024×256 with 8 threads: 8 × 256 × 256 = 524 K adds → fast.
 * ---------------------------------------------------------------------- */

void ternary_gemm_zero_skip_tiled(
    int M, int N, int K,
    const float*      A,
    const TernaryCSR* B_csr,
    float*            C
) {
#ifdef _OPENMP
    /* Transpose A[M,K] → AT[K,M] so each k-row is a contiguous M-vector */
    float* AT = (float*)malloc((size_t)K * M * sizeof(float));
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++)
            AT[k * M + m] = A[m * K + k];

    const int nthreads = omp_get_max_threads();

    /* Pre-allocate all thread-local CT arrays contiguously.
     * Thread t uses CT_all + t*N*M.  calloc zeros everything. */
    float* CT_all = (float*)calloc((size_t)nthreads * N * M, sizeof(float));

    /* Phase 1: k-parallel SAXPY into thread-private CT_local.
     *
     * schedule(static) gives each thread a contiguous k-range,
     * so AT[k_range,:] is a contiguous slab that fits in L2. */
    #pragma omp parallel
    {
        const int tid      = omp_get_thread_num();
        float* CT_local    = CT_all + (size_t)tid * N * M;

        #pragma omp for schedule(static)
        for (int k = 0; k < K; k++) {
            const float* at_row = AT + (size_t)k * M;
            const int p0 = B_csr->row_ptr[k];
            const int p1 = B_csr->row_ptr[k + 1];

            /* For this k-row, update all output columns j it touches.
             * AT[k,:] stays in L1 across the entire inner loop. */
            for (int p = p0; p < p1; p++) {
                const int    j     = B_csr->col_idx[p];
                const int8_t sign  = B_csr->signs[p];
                const __m256 vsign = (sign > 0) ? _mm256_set1_ps(1.0f)
                                                : _mm256_set1_ps(-1.0f);
                float* ct_row = CT_local + (size_t)j * M;

                int m = 0;
                for (; m + 8 <= M; m += 8) {
                    __m256 av = _mm256_loadu_ps(at_row + m);
                    __m256 cv = _mm256_loadu_ps(ct_row + m);
                    cv = _mm256_fmadd_ps(vsign, av, cv);
                    _mm256_storeu_ps(ct_row + m, cv);
                }
                const float fsign = (float)sign;
                for (; m < M; m++)
                    ct_row[m] += fsign * at_row[m];
            }
        }
    } /* end parallel: implicit barrier before reduction */

    /* Phase 2: reduce CT_all[nthreads × N × M] → CT[N, M].
     * Parallelise over j so each thread reduces its own output rows. */
    float* CT = (float*)calloc((size_t)N * M, sizeof(float));

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < N; j++) {
        float* ct_out = CT + (size_t)j * M;
        for (int t = 0; t < nthreads; t++) {
            const float* ct_local_row = CT_all + (size_t)t * N * M + (size_t)j * M;
            int m = 0;
            for (; m + 8 <= M; m += 8) {
                __m256 out = _mm256_loadu_ps(ct_out + m);
                __m256 loc = _mm256_loadu_ps(ct_local_row + m);
                out = _mm256_add_ps(out, loc);
                _mm256_storeu_ps(ct_out + m, out);
            }
            for (; m < M; m++)
                ct_out[m] += ct_local_row[m];
        }
    }

    /* Transpose CT[N,M] → C[M,N] */
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++)
        for (int j = 0; j < N; j++)
            C[m * N + j] = CT[(size_t)j * M + m];

    free(AT);
    free(CT_all);
    free(CT);

#else  /* !_OPENMP: single-threaded k-loop, same algorithm without reduction */
    float* AT = (float*)malloc((size_t)K * M * sizeof(float));
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++)
            AT[k * M + m] = A[m * K + k];

    float* CT = (float*)calloc((size_t)N * M, sizeof(float));

    for (int k = 0; k < K; k++) {
        const float* at_row = AT + (size_t)k * M;
        const int p0 = B_csr->row_ptr[k];
        const int p1 = B_csr->row_ptr[k + 1];
        for (int p = p0; p < p1; p++) {
            const int    j     = B_csr->col_idx[p];
            const int8_t sign  = B_csr->signs[p];
            const __m256 vsign = (sign > 0) ? _mm256_set1_ps(1.0f)
                                            : _mm256_set1_ps(-1.0f);
            float* ct_row = CT + (size_t)j * M;
            int m = 0;
            for (; m + 8 <= M; m += 8) {
                __m256 av = _mm256_loadu_ps(at_row + m);
                __m256 cv = _mm256_loadu_ps(ct_row + m);
                cv = _mm256_fmadd_ps(vsign, av, cv);
                _mm256_storeu_ps(ct_row + m, cv);
            }
            const float fsign = (float)sign;
            for (; m < M; m++)
                ct_row[m] += fsign * at_row[m];
        }
    }

    for (int m = 0; m < M; m++)
        for (int j = 0; j < N; j++)
            C[m * N + j] = CT[(size_t)j * M + m];

    free(AT);
    free(CT);
#endif
}

#else  /* !__AVX2__ */

void ternary_gemm_zero_skip_avx2(
    int M, int N, int K,
    const float*      A,
    const TernaryCSC* B_csc,
    float*            C
) {
    ternary_gemm_zero_skip_scalar(M, N, K, A, B_csc, C);
}

void ternary_gemm_zero_skip_tiled(
    int M, int N, int K,
    const float*      A,
    const TernaryCSR* B_csr,
    float*            C
) {
    /* No AVX2: build a CSC and run scalar */
    TernaryCSC* csc = (TernaryCSC*)malloc(sizeof(TernaryCSC));
    csc->K   = K;
    csc->N   = N;
    csc->nnz = B_csr->nnz;
    /* Rebuild col_ptr / row_idx from CSR */
    csc->col_ptr = (int*)calloc(N + 1, sizeof(int));
    csc->row_idx = (int*)malloc(B_csr->nnz * sizeof(int));
    csc->signs   = (int8_t*)malloc(B_csr->nnz * sizeof(int8_t));
    for (int k = 0; k < K; k++) {
        for (int p = B_csr->row_ptr[k]; p < B_csr->row_ptr[k + 1]; p++)
            csc->col_ptr[B_csr->col_idx[p] + 1]++;
    }
    for (int j = 0; j < N; j++)
        csc->col_ptr[j + 1] += csc->col_ptr[j];
    int* fill = (int*)calloc(N, sizeof(int));
    for (int k = 0; k < K; k++) {
        for (int p = B_csr->row_ptr[k]; p < B_csr->row_ptr[k + 1]; p++) {
            int j = B_csr->col_idx[p];
            int idx = csc->col_ptr[j] + fill[j]++;
            csc->row_idx[idx] = k;
            csc->signs[idx]   = B_csr->signs[p];
        }
    }
    free(fill);
    ternary_gemm_zero_skip_scalar(M, N, K, A, csc, C);
    free_ternary_csc(csc);
}

#endif  /* __AVX2__ */

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
