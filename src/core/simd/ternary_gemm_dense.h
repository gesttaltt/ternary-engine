/**
 * ternary_gemm_dense.h - Dense-Packed Ternary GEMM
 *
 * Copyright 2025 Ternary Engine Contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Alternative to ternary_gemm_zero_skip.h's CSC/CSR sparse-index kernels.
 *
 * Motivation (found 2026-08-20 while investigating the competitive
 * benchmark's "0.189x avg matmul" figure, `.claude/CLAUDE.md` Critical
 * Gap #3): at ternary's actual ~33-40% zero density, a CSC/CSR index
 * (4-byte row/col index + 1-byte sign per non-zero) costs MORE memory
 * than just storing the dense int8 matrix directly (1 byte per weight,
 * zeros included) -- for 60-67% non-zero density, index storage is
 * ~0.6-0.67 * 5 = 3.0-3.35x the size of the dense array. Since these
 * kernels are memory-bandwidth-bound (SAXPY-style accumulation touches
 * every output element once per non-zero, no register reuse), reading
 * the smaller dense representation wins even though it does "extra"
 * multiply-adds on zero weights -- the FLOPs are nearly free, the bytes
 * moved are not.
 *
 * A second, independent problem in the naive dense approach: B is stored
 * row-major [K, N]. A kernel that blocks over N (for AVX2 width) and
 * loops K in the inner loop reads B[k*N + j0 .. j0+8) -- a stride of N
 * bytes between consecutive k, wasting most of every 64B cache line for
 * any N larger than ~64. Packing B once into [N/8][K][8] contiguous
 * blocks (mirroring how build_ternary_csc/csr precompute an index once
 * per weight matrix, since weights are fixed across inference calls)
 * turns that into fully sequential access.
 *
 * Combined effect, measured on this machine (AMD Ryzen, Linux x64,
 * 2026-08-20, `benchmarks/cpp-native-kernels/bench_gemm_dense.cpp`):
 * at the exact shapes/batch=1 the competitive suite's Phase 4 tests
 * (where the CSC/CSR kernels' 8-wide SIMD -- vectorized over the batch
 * dimension -- cannot fire at all), this kernel goes from a 0.06x-0.21x
 * loss against NumPy to a 4.5x-9.5x win, and stays competitive
 * (0.87x-3.3x) up through batch=512, where the existing tiled kernel's
 * per-thread CT_local buffer (nthreads * N * M floats) badly overflows
 * cache and degrades instead of improving.
 *
 * API:
 *   TernaryPacked* pack_ternary_dense(B, K, N)   -- pack once per weight matrix
 *   void   ternary_gemm_packed_scalar (M,N,K, A, packed, C)
 *   void   ternary_gemm_packed_avx2   (M,N,K, A, packed, C)
 *   void   ternary_gemm_dense         (M,N,K, A, B, C)  [all-in-one]
 */

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Packed dense weight layout: [n_blocks][K][8] int8, n_blocks = ceil(N/8).
 * A block is the 8 (possibly zero) weights for output columns
 * [8*b, 8*b+8) across all K input rows, stored contiguously so the GEMM
 * kernel's per-k weight load is sequential rather than strided by N.
 * The final block's tail columns (when N % 8 != 0) are zero-padded. */
typedef struct {
    int     K;
    int     N;
    int     n_blocks;
    int8_t* data;   /* [n_blocks * K * 8] */
} TernaryPacked;

TernaryPacked* pack_ternary_dense(const int8_t* B, int K, int N);
void           free_ternary_packed(TernaryPacked* p);

/* Scalar kernel (M-blocked, no SIMD). Reference / non-AVX2 fallback. */
void ternary_gemm_packed_scalar(
    int M, int N, int K,
    const float*         A,
    const TernaryPacked* packed,
    float*               C
);

/* AVX2 kernel: N-vectorized (8 output columns/register), M-blocked in
 * groups of TERNARY_GEMM_DENSE_MB (8 as of 2026-08-22, was 4 -- see
 * ternary_gemm_dense.cpp's #define comment for why) so each loaded/
 * widened weight tile is reused across that many batch rows before
 * moving on -- the register-blocking that gives batch>1 arithmetic-
 * intensity gains without needing zero-skip indices.
 * Falls back to ternary_gemm_packed_scalar on non-AVX2 builds. */
void ternary_gemm_packed_avx2(
    int M, int N, int K,
    const float*         A,
    const TernaryPacked* packed,
    float*               C
);

/* Convenience: pack B on the fly, run avx2, free. Prefer pack_ternary_dense()
 * + ternary_gemm_packed_avx2() directly when B is reused across many calls
 * (the normal inference case -- packing cost should not be paid per call). */
void ternary_gemm_dense(
    int M, int N, int K,
    const float*  A,
    const int8_t* B,
    float*        C
);

#ifdef __cplusplus
}
#endif
