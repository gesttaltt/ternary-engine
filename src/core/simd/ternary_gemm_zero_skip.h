/**
 * ternary_gemm_zero_skip.h - Zero-Skip Ternary GEMM
 *
 * Copyright 2025 Ternary Engine Contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Implements zero-skip GEMM for ternary weight matrices {-1, 0, +1}.
 *
 * Key optimization: ~33% of ternary weights are exactly zero (p-adic
 * structure, see research/FINDINGS.md).  By precomputing a CSC sparse
 * index and iterating only over non-zero weights, we eliminate ~33% of
 * multiply-accumulate operations.
 *
 * Unlike the Dense243-based tritnet_gemm, this kernel accepts B in plain
 * int8 format, avoiding the unpacking overhead.  The sparse structure is
 * built once per weight matrix and reused across forward passes.
 *
 * API:
 *   TernaryCSC* build_ternary_csc(B, K, N)   -- precompute sparse index
 *   void        free_ternary_csc(csc)
 *   void        ternary_gemm_zero_skip_scalar(M, N, K, A, csc, C)
 *   void        ternary_gemm_zero_skip_avx2  (M, N, K, A, csc, C)  [AVX2]
 *   void        ternary_gemm_zero_skip       (M, N, K, A, B, C)    [all-in-one]
 */

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* CSC (Compressed Sparse Column) representation of a ternary weight matrix.
 * col_ptr[j]..col_ptr[j+1]-1 indexes the non-zeros in column j.
 * row_idx[p] = k (row / activation dimension)
 * signs[p]   = +1 or -1 */
typedef struct {
    int   K;         /* rows of B  (activation dimension) */
    int   N;         /* cols of B  (output dimension)     */
    int   nnz;       /* total non-zero count              */
    int*  col_ptr;   /* [N+1] column pointers             */
    int*  row_idx;   /* [nnz] non-zero row indices        */
    int8_t* signs;   /* [nnz] +1 or -1                    */
} TernaryCSC;

/* Build CSC structure from dense int8 B [K, N] row-major.
 * Caller must free with free_ternary_csc(). */
TernaryCSC* build_ternary_csc(const int8_t* B, int K, int N);
void        free_ternary_csc(TernaryCSC* csc);

/* Zero-skip GEMM, scalar path.
 * A: [M, K] float32 row-major
 * C: [M, N] float32 row-major (zeroed by callee) */
void ternary_gemm_zero_skip_scalar(
    int M, int N, int K,
    const float*       A,
    const TernaryCSC*  B_csc,
    float*             C
);

/* Zero-skip GEMM, AVX2 path.  Falls back to scalar on non-AVX2 builds. */
void ternary_gemm_zero_skip_avx2(
    int M, int N, int K,
    const float*       A,
    const TernaryCSC*  B_csc,
    float*             C
);

/* Convenience: build CSC on the fly, compute, free.
 * Use for one-shot benchmarking; prefer the explicit API for inference. */
void ternary_gemm_zero_skip(
    int M, int N, int K,
    const float*   A,
    const int8_t*  B,
    float*         C
);

#ifdef __cplusplus
}
#endif
