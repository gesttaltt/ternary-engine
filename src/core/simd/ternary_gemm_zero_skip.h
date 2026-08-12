/**
 * ternary_gemm_zero_skip.h - Zero-Skip Ternary GEMM
 *
 * Copyright 2025 Ternary Engine Contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Implements zero-skip GEMM for ternary weight matrices {-1, 0, +1}.
 *
 * Key optimization: ~33% of ternary weights are exactly zero (p-adic
 * structure, see research/FINDINGS.md).  By precomputing a sparse index
 * and iterating only over non-zero weights, we eliminate ~33% of
 * multiply-accumulate operations.
 *
 * Two sparse formats, two kernels:
 *
 *   CSC (column-indexed) + j-parallel kernel:
 *     For each output column j, SAXPY over non-zero k rows.
 *     All j threads share the full AT[K,M] transpose buffer in L3.
 *     Scales ~3× from 8 cores due to shared-AT bandwidth bottleneck.
 *
 *   CSR (row-indexed) + k-parallel tiled kernel:
 *     For each activation row k, SAXPY into all j columns it touches.
 *     Each thread owns a k-stripe; AT[k_stripe,:] fits in L2 (~128KB).
 *     Thread-private CT_local arrays eliminate AT bandwidth contention.
 *     Expected better scaling for bandwidth-bound large-M cases.
 *
 * API:
 *   TernaryCSC* build_ternary_csc(B, K, N)   -- column-indexed index
 *   TernaryCSR* build_ternary_csr(B, K, N)   -- row-indexed index
 *   void        ternary_gemm_zero_skip_scalar (M,N,K, A, csc, C)
 *   void        ternary_gemm_zero_skip_avx2   (M,N,K, A, csc, C)  [j-parallel]
 *   void        ternary_gemm_zero_skip_tiled  (M,N,K, A, csr, C)  [k-parallel]
 *   void        ternary_gemm_zero_skip        (M,N,K, A, B,   C)  [all-in-one]
 */

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- CSC: column-indexed (output-column outer loop) ---- */

typedef struct {
    int   K;         /* rows of B  (activation dimension) */
    int   N;         /* cols of B  (output dimension)     */
    int   nnz;       /* total non-zero count              */
    int*  col_ptr;   /* [N+1] column pointers             */
    int*  row_idx;   /* [nnz] non-zero row indices        */
    int8_t* signs;   /* [nnz] +1 or -1                    */
} TernaryCSC;

TernaryCSC* build_ternary_csc(const int8_t* B, int K, int N);
void        free_ternary_csc(TernaryCSC* csc);

/* ---- CSR: row-indexed (activation-row outer loop) ---- */

/* For the tiled kernel: each k-row lists the output columns j it connects to.
 * This enables AT[k,:] to be loaded once and reused for all j in that row,
 * keeping the k-stripe hot in L2 while threads work on their partition. */
typedef struct {
    int   K;         /* rows of B  (activation dimension) */
    int   N;         /* cols of B  (output dimension)     */
    int   nnz;       /* total non-zero count              */
    int*  row_ptr;   /* [K+1] row pointers                */
    int*  col_idx;   /* [nnz] non-zero column indices     */
    int8_t* signs;   /* [nnz] +1 or -1                    */
} TernaryCSR;

TernaryCSR* build_ternary_csr(const int8_t* B, int K, int N);
void        free_ternary_csr(TernaryCSR* csr);

/* ---- Kernels ---- */

/* Scalar j-parallel kernel (A: [M,K], C: [M,N], zeroed by callee). */
void ternary_gemm_zero_skip_scalar(
    int M, int N, int K,
    const float*       A,
    const TernaryCSC*  B_csc,
    float*             C
);

/* AVX2 j-parallel kernel (CSC, OpenMP over j).  Falls back to scalar
 * on non-AVX2 builds. */
void ternary_gemm_zero_skip_avx2(
    int M, int N, int K,
    const float*       A,
    const TernaryCSC*  B_csc,
    float*             C
);

/* AVX2 k-parallel tiled kernel (CSR, OpenMP over k, thread-private CT).
 * Parallelises over activation rows so each thread's AT slice fits in L2.
 * When OpenMP is absent, runs a single-threaded k-loop with the same CSR
 * SAXPY inner loop (not a call into ternary_gemm_zero_skip_avx2 — that
 * kernel takes a CSC, not a CSR, and is a separately maintained code path;
 * a fix to one SAXPY loop is not automatically applied to the other). */
void ternary_gemm_zero_skip_tiled(
    int M, int N, int K,
    const float*       A,
    const TernaryCSR*  B_csr,
    float*             C
);

/* Convenience: build CSC on the fly, run avx2, free. */
void ternary_gemm_zero_skip(
    int M, int N, int K,
    const float*   A,
    const int8_t*  B,
    float*         C
);

#ifdef __cplusplus
}
#endif
