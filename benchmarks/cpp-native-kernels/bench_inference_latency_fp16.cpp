/**
 * @file bench_inference_latency_fp16.cpp
 * @brief Commercial-viability criterion 3: ternary inference latency vs FP16
 *
 * Copyright 2025 Ternary Engine Contributors
 * Licensed under the Apache License, Version 2.0
 *
 * WHY THIS EXISTS. Of this project's five commercial-viability criteria
 * (.claude/CLAUDE.md -> "Commercial Viability Criteria"), criterion 3
 * ("Inference latency < 2x FP16") had never been measured at all -- see
 * docs/planning/ROADMAP.md -> "Where To Continue" (2026-08-29), which
 * identifies it as the only unvalidated criterion whose work is actually
 * possible in this environment (criterion 4, power, is blocked: RAPL
 * energy_uj is root-only here).
 *
 * WHAT "FP16 BASELINE" HONESTLY MEANS ON A CPU. This is the whole
 * methodological question, and getting it wrong would repeat the mistake
 * this project already retired with its "8,234x vs pure Python" headline.
 *
 * There is no fast native FP16 arithmetic on this class of x86 CPU (Zen 2
 * has F16C convert instructions but no FP16 FMA; AVX512-FP16 does not
 * exist here). A model whose weights are stored FP16 is therefore executed
 * on CPU by upconverting to FP32 and running an FP32 GEMM. So:
 *
 *   The honest FP16-model latency on this CPU IS a well-optimized FP32
 *   SGEMM. That is the baseline used for the criterion below.
 *
 * Benchmarking against *emulated* per-element FP16 arithmetic would be a
 * strawman that ternary wins by construction, exactly like comparing
 * compiled code against interpreted Python. This file measures that
 * emulated path too, but reports it ONLY as the strawman being rejected --
 * it is never used to substantiate the criterion.
 *
 * The FP32 reference is OpenBLAS (the one NumPy ships, 0.3.31, DYNAMIC_ARCH
 * so it dispatches a Zen-appropriate microkernel), not a hand-rolled loop.
 * A weak reference would make ternary look good for the wrong reason; this
 * project's own rule is that the baseline must be the strongest available.
 * Both sides are pinned to ONE thread so the comparison is per-core kernel
 * quality, not a threading-strategy accident.
 *
 * WHAT IS ACTUALLY COMPARED, per (shape, batch) cell:
 *   (a) ternary : pack_ternary_dense() once, then ternary_gemm_packed_avx2
 *                 -- fp32 activations x int8 ternary weights -> fp32.
 *                 Packing is amortized (weights are fixed at inference),
 *                 matching how the BLAS side also gets pre-laid-out weights.
 *   (b) fp32    : cblas_sgemm on the same logical matrices (THE BASELINE)
 *   (c) fp16dq  : dequantize FP16 weights -> FP32 every call, then sgemm.
 *                 The realistic cost if a deployment stores FP16 and cannot
 *                 keep an FP32 copy resident.
 *   (d) fp16emu : per-element FP16 arithmetic via F16C round-trips.
 *                 THE STRAWMAN. Reported, labelled, and not used.
 *
 * Shapes are TinyLlama-1.1B's real projection shapes, the model this
 * project's quantization work has used throughout, at batch sizes spanning
 * autoregressive decode (M=1, the latency-critical case) to prefill.
 *
 * Timing follows this project's interleaved_timing rule: contenders are
 * measured rep-by-rep back-to-back rather than in separate phases, so
 * clock/thermal drift hits both equally -- the exact unfairness that had to
 * be corrected in the TritNet Phase 3 session.
 *
 * COMPILATION (from benchmarks/cpp-native-kernels/):
 *   see build_inference_latency.py, which locates NumPy's OpenBLAS .so
 *   and its ILP64 symbol suffix automatically.
 *
 * USAGE: ./bench_inference_latency_fp16
 * OUTPUT: per-cell latency (ms) and the ternary/fp32 ratio, plus a verdict
 *         against the "< 2x FP16" criterion. Correctness of the ternary
 *         kernel is checked against a float64 reference before any timing.
 */

#include "ternary_gemm_dense.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <functional>
#include <immintrin.h>

using Clock = std::chrono::high_resolution_clock;

/* ------------------------------------------------------------------ */
/* OpenBLAS ILP64 cblas_sgemm. Declared by hand rather than including a
 * cblas.h, because the header that matches NumPy's build is not installed
 * system-wide; the symbol name (and its scipy_ prefix / 64_ suffix) is
 * resolved by build_inference_latency.py and passed in as CBLAS_SGEMM_SYM. */
#ifndef CBLAS_SGEMM_SYM
#define CBLAS_SGEMM_SYM cblas_sgemm64_
#endif
#ifndef OPENBLAS_SET_THREADS_SYM
#define OPENBLAS_SET_THREADS_SYM openblas_set_num_threads64_
#endif

#ifndef CBLAS_SGEMV_SYM
#define CBLAS_SGEMV_SYM cblas_sgemv64_
#endif

extern "C" {
void CBLAS_SGEMM_SYM(int order, int transa, int transb,
                     int64_t m, int64_t n, int64_t k,
                     float alpha, const float* a, int64_t lda,
                     const float* b, int64_t ldb,
                     float beta, float* c, int64_t ldc);
/* Needed for a FAIR batch=1 baseline: at M=1 the operation is a GEMV, and
 * OpenBLAS has a dedicated sgemv kernel that its general sgemm path does
 * not reduce to. Timing ternary against sgemm-at-M=1 while a faster BLAS
 * routine exists would hand ternary a win it did not earn, in exactly the
 * regime (autoregressive decode) where the criterion matters most. The
 * baseline used below is the BETTER of sgemm and sgemv. */
void CBLAS_SGEMV_SYM(int order, int trans, int64_t m, int64_t n,
                     float alpha, const float* a, int64_t lda,
                     const float* x, int64_t incx,
                     float beta, float* y, int64_t incy);
void OPENBLAS_SET_THREADS_SYM(int n);
}
#define CBLAS_ROW_MAJOR 101
#define CBLAS_NO_TRANS  111
#define CBLAS_TRANS     112

/* ------------------------------------------------------------------ */
/* FP16 helpers via F16C (present on Zen 2; the build script verifies).   */

static inline uint16_t f32_to_f16(float f) {
    return (uint16_t)_cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}
static inline float f16_to_f32(uint16_t h) { return _cvtsh_ss(h); }

/* (c) Dequantize an FP16 weight matrix to FP32, 8 lanes at a time. */
static void dequant_f16_to_f32(const uint16_t* src, float* dst, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i*)(src + i));
        _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(h));
    }
    for (; i < n; ++i) dst[i] = f16_to_f32(src[i]);
}

/* (d) THE STRAWMAN: genuine per-element FP16 arithmetic, i.e. every
 * multiply-add round-trips through FP16. No CPU in this class does this in
 * hardware, which is exactly the point -- it is what "FP16 on CPU" would
 * cost if taken literally, and why it must NOT be the baseline. */
static void gemm_fp16_emulated(int M, int N, int K,
                               const uint16_t* A, const uint16_t* B,
                               uint16_t* C) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            uint16_t acc = f32_to_f16(0.0f);
            for (int k = 0; k < K; ++k) {
                float prod = f16_to_f32(A[m * K + k]) * f16_to_f32(B[k * N + n]);
                acc = f32_to_f16(f16_to_f32(acc) + f32_to_f16(prod));
            }
            C[m * N + n] = acc;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Interleaved timing: contenders alternate rep-by-rep so clock and thermal
 * drift affect all of them equally. Returns the best (minimum) time per
 * contender, in seconds. Best-of rather than mean because we want the
 * machine's capability, not its scheduler noise. */
static std::vector<double> time_best_interleaved(
        const std::vector<std::function<void()>>& fns, int reps) {
    const size_t n = fns.size();
    std::vector<double> best(n, 1e300);
    for (auto& f : fns) f();                     /* warm caches/branch state */
    for (int r = 0; r < reps; ++r) {
        for (size_t i = 0; i < n; ++i) {
            auto t0 = Clock::now();
            fns[i]();
            double dt = std::chrono::duration<double>(Clock::now() - t0).count();
            if (dt < best[i]) best[i] = dt;
        }
    }
    return best;
}

/* Correctness of the ternary kernel against a float64 reference. Timing a
 * kernel that computes the wrong thing is the classic way to publish a
 * meaningless speedup, so this gates every cell. */
static double ternary_maxerr(int M, int N, int K, const float* A,
                             const int8_t* B, const float* C) {
    double worst = 0.0;
    const int ms = M < 2 ? M : 2, ns = N < 8 ? N : 8;
    for (int m = 0; m < ms; ++m)
        for (int n = 0; n < ns; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) acc += (double)A[m * K + k] * (double)B[k * N + n];
            worst = std::max(worst, std::fabs(acc - (double)C[m * N + n]));
        }
    return worst;
}

struct Shape { const char* name; int K; int N; };

int main() {
    /* One thread on both sides: this measures per-core kernel quality, not
     * whose threading strategy scales better on 6 cores. */
    OPENBLAS_SET_THREADS_SYM(1);

    /* Real projection shapes from two models:
     *   TinyLlama-1.1B  -- the model this project's (failed) quantization
     *                      experiments used throughout;
     *   BitNet b1.58 large (1bitLLM/bitnet_b1_58-large) -- a model actually
     *                      TRAINED ternary, added 2026-08-29. Its shapes are
     *                      what the engine would face running a real ternary
     *                      LLM, which is the engine's actual job; see
     *                      benchmarks/model_quantization/run_bitnet_on_engine.py.
     * Only shape and batch affect timing here (the dense kernel does the same
     * work regardless of which +-1 values the weights hold, since it does not
     * skip zeros), so synthetic ternary weights are representative. */
    const Shape shapes[] = {
        {"TinyLlama q/o_proj    [2048x2048]", 2048, 2048},
        {"TinyLlama k/v_proj    [2048x256] ", 2048,  256},
        {"TinyLlama gate/up     [2048x5632]", 2048, 5632},
        {"TinyLlama down_proj   [5632x2048]", 5632, 2048},
        {"BitNet    q/k/v/o     [1536x1536]", 1536, 1536},
        {"BitNet    gate/up     [1536x4096]", 1536, 4096},
        {"BitNet    down_proj   [4096x1536]", 4096, 1536},
    };
    const int batches[] = {1, 8, 32, 128};

    printf("=====================================================================\n");
    printf(" Criterion 3: ternary inference latency vs FP16 (CPU, 1 thread)\n");
    printf("=====================================================================\n");
    printf(" BASELINE = fp32 OpenBLAS sgemm.\n");
    printf(" On this CPU class an FP16-stored model is executed by upconverting\n");
    printf(" to FP32 (no FP16 FMA in hardware), so fp32 sgemm IS the honest\n");
    printf(" FP16-model latency. fp16emu below is the rejected strawman.\n\n");

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> trit(-1, 1);
    std::uniform_real_distribution<float> act(-1.0f, 1.0f);

    int cells = 0, cells_within_2x = 0;
    double ratio_sum = 0.0, ratio_worst = 0.0;

    for (const Shape& s : shapes) {
        const int K = s.K, N = s.N;

        std::vector<int8_t> B(  (size_t)K * N);
        std::vector<float>  Bf( (size_t)K * N);
        std::vector<uint16_t> Bh((size_t)K * N);
        for (size_t i = 0; i < B.size(); ++i) {
            B[i]  = (int8_t)trit(rng);
            Bf[i] = (float)B[i];
            Bh[i] = f32_to_f16(Bf[i]);
        }
        TernaryPacked* packed = pack_ternary_dense(B.data(), K, N);
        if (!packed) { fprintf(stderr, "pack_ternary_dense failed\n"); return 1; }

        printf("%s\n", s.name);
        printf("  %6s %11s %11s %11s   %10s %9s\n",
               "batch", "ternary_ms", "blas_ms", "fp16dq_ms", "ratio_t/blas", "verdict");

        for (int M : batches) {
            std::vector<float> A((size_t)M * K);
            for (auto& a : A) a = act(rng);
            std::vector<float>  Ct((size_t)M * N, 0.0f);
            std::vector<float>  Cf((size_t)M * N, 0.0f);
            std::vector<float>  Bdq((size_t)K * N);

            ternary_gemm_packed_avx2(M, N, K, A.data(), packed, Ct.data());
            double err = ternary_maxerr(M, N, K, A.data(), B.data(), Ct.data());
            double tol = 1e-3 * K;
            if (err > tol) {
                printf("  [FAIL] ternary kernel maxerr %.3g > tol %.3g -- not timing\n",
                       err, tol);
                continue;
            }

            std::vector<std::function<void()>> fns = {
                [&]{ ternary_gemm_packed_avx2(M, N, K, A.data(), packed, Ct.data()); },
                [&]{ CBLAS_SGEMM_SYM(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                                     M, N, K, 1.0f, A.data(), K, Bf.data(), N,
                                     0.0f, Cf.data(), N); },
                [&]{ dequant_f16_to_f32(Bh.data(), Bdq.data(), (size_t)K * N);
                     CBLAS_SGEMM_SYM(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                                     M, N, K, 1.0f, A.data(), K, Bdq.data(), N,
                                     0.0f, Cf.data(), N); },
            };
            /* At M=1 add the dedicated GEMV path, and use whichever BLAS
             * routine is faster as the baseline (strongest-baseline rule). */
            std::vector<float> Cv((size_t)N, 0.0f);
            if (M == 1) {
                fns.push_back([&]{
                    CBLAS_SGEMV_SYM(CBLAS_ROW_MAJOR, CBLAS_TRANS, K, N,
                                    1.0f, Bf.data(), N, A.data(), 1,
                                    0.0f, Cv.data(), 1); });
            }
            int reps = (M <= 8) ? 20 : 8;
            std::vector<double> t = time_best_interleaved(fns, reps);

            double blas_best = t[1];
            const char* blas_which = "sgemm";
            if (M == 1 && t.size() > 3 && t[3] < blas_best) {
                blas_best = t[3]; blas_which = "sgemv";
            }
            double ratio = t[0] / blas_best;
            bool ok = ratio < 2.0;
            cells++; if (ok) cells_within_2x++;
            ratio_sum += ratio; ratio_worst = std::max(ratio_worst, ratio);

            printf("  %6d %11.4f %11.4f %11.4f   %10.3fx %9s  (%s)\n",
                   M, t[0] * 1e3, blas_best * 1e3, t[2] * 1e3, ratio,
                   ok ? "PASS" : "FAIL", blas_which);
        }
        free_ternary_packed(packed);
        printf("\n");
    }

    /* The strawman, measured once at a small shape because it is O(MNK) in
     * scalar F16C round-trips and would otherwise dominate the runtime. */
    {
        const int M = 1, K = 2048, N = 256;
        std::vector<int8_t>   B((size_t)K * N);
        std::vector<uint16_t> Bh((size_t)K * N), Ah((size_t)M * K), Ch((size_t)M * N);
        std::vector<float>    A((size_t)M * K), Bf((size_t)K * N), Cf((size_t)M * N);
        for (size_t i = 0; i < B.size(); ++i) {
            B[i] = (int8_t)trit(rng); Bf[i] = (float)B[i]; Bh[i] = f32_to_f16(Bf[i]);
        }
        for (int i = 0; i < M * K; ++i) { A[i] = act(rng); Ah[i] = f32_to_f16(A[i]); }
        TernaryPacked* packed = pack_ternary_dense(B.data(), K, N);
        std::vector<float> Ct((size_t)M * N);

        std::vector<std::function<void()>> fns = {
            [&]{ ternary_gemm_packed_avx2(M, N, K, A.data(), packed, Ct.data()); },
            [&]{ CBLAS_SGEMM_SYM(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                                 M, N, K, 1.0f, A.data(), K, Bf.data(), N,
                                 0.0f, Cf.data(), N); },
            [&]{ gemm_fp16_emulated(M, N, K, Ah.data(), Bh.data(), Ch.data()); },
        };
        std::vector<double> t = time_best_interleaved(fns, 5);
        printf("Rejected strawman, for the record (M=1, K=2048, N=256):\n");
        printf("  ternary %.4f ms | fp32 sgemm %.4f ms | fp16 EMULATED %.4f ms\n",
               t[0] * 1e3, t[1] * 1e3, t[2] * 1e3);
        printf("  Ternary would look %.0fx 'better' against emulated FP16 vs %.2fx\n"
               "  against the honest fp32 baseline. This is why fp32 is used.\n\n",
               t[2] / t[0], t[0] / t[1]);
        free_ternary_packed(packed);
    }

    printf("=====================================================================\n");
    printf(" Criterion 3 (< 2x FP16): %d/%d cells pass | mean ratio %.3fx | worst %.3fx\n",
           cells_within_2x, cells, ratio_sum / (cells ? cells : 1), ratio_worst);
    printf(" VERDICT: %s\n", (cells_within_2x == cells) ? "PASS (all cells)"
                            : (cells_within_2x > 0 ? "PARTIAL" : "FAIL"));
    printf("=====================================================================\n");
    return 0;
}
