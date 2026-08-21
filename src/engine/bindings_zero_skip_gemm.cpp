/**
 * bindings_zero_skip_gemm.cpp - Python Bindings for Ternary GEMM
 *
 * Copyright 2025 Ternary Engine Contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Exposes two ternary GEMM strategies as one Python module:
 * ternary_zero_skip_gemm
 *
 *   ZeroSkipWeights: CSC/CSR sparse-index kernels (the original approach).
 *   DenseWeights:    dense-packed kernel, added 2026-08-20 -- despite the
 *                    module's name, this is the one to prefer. At
 *                    ternary's actual ~33-40% zero density, a CSC/CSR
 *                    index costs MORE memory than the dense int8 array
 *                    itself (see ternary_gemm_dense.h for the full
 *                    analysis), and these kernels are memory-bandwidth-
 *                    bound, not compute-bound. Measured 4.5x-9.5x faster
 *                    than NumPy at batch=1 (vs ZeroSkipWeights' 0.06x-0.21x
 *                    loss) at the exact shapes this project's competitive
 *                    benchmark's Phase 4 tests -- see
 *                    reports/2026-08-20/GEMM_DENSE_PACKED_OPTIMIZATION.md.
 *                    ZeroSkipWeights is kept for comparison / regression
 *                    tracking, not because it's the recommended path.
 *
 * Python API:
 *
 *   weights = DenseWeights(B_int8)       # pack dense weights once
 *   C = weights.gemm(A_float32)          # run GEMM (AVX2 or scalar)
 *
 *   weights = ZeroSkipWeights(B_int8)    # precompute sparse index once
 *   C = weights.gemm(A_float32)          # run GEMM (scalar or AVX2)
 *
 *   C = gemm(A_float32, B_int8)          # convenience all-in-one (zero-skip)
 *
 *   info = sparsity_info(B_int8)         # measure zero fraction, nnz, etc.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "ternary_gemm_zero_skip.h"
#include "ternary_gemm_dense.h"
#include "py_array_validate.h"
#include <stdexcept>
#include <string>

namespace py = pybind11;
using ternary_engine::validate_2d_contiguous;

/* -------------------------------------------------------------------------
 * ZeroSkipWeights class: holds precomputed CSC for one weight matrix
 * ---------------------------------------------------------------------- */

class ZeroSkipWeights {
public:
    ZeroSkipWeights(py::array_t<int8_t> B) {
        auto buf = validate_2d_contiguous(B, "B", "2-D [K, N] int8");

        K_   = (int)buf.shape[0];
        N_   = (int)buf.shape[1];
        const int8_t* data = static_cast<const int8_t*>(buf.ptr);
        csc_ = build_ternary_csc(data, K_, N_);
        csr_ = build_ternary_csr(data, K_, N_);
        if (!csc_ || !csr_)
            throw std::runtime_error("Failed to build sparse index");
    }

    ~ZeroSkipWeights() {
        free_ternary_csc(csc_);
        free_ternary_csr(csr_);
    }

    ZeroSkipWeights(const ZeroSkipWeights&) = delete;
    ZeroSkipWeights& operator=(const ZeroSkipWeights&) = delete;

    py::array_t<float> gemm(py::array_t<float> A, bool use_avx2 = true) {
        auto buf = validate_2d_contiguous(A, "A", "2-D [M, K] float32");
        int M = (int)buf.shape[0];
        int K = (int)buf.shape[1];
        if (K != K_)
            throw std::invalid_argument(
                "A columns (" + std::to_string(K) +
                ") must match weight rows (" + std::to_string(K_) + ")"
            );

        py::array_t<float> C({M, N_});
        auto c_buf = C.request();
        const float* a_ptr = static_cast<const float*>(buf.ptr);
        float*       c_ptr = static_cast<float*>(c_buf.ptr);

        if (use_avx2)
            ternary_gemm_zero_skip_avx2(M, N_, K_, a_ptr, csc_, c_ptr);
        else
            ternary_gemm_zero_skip_scalar(M, N_, K_, a_ptr, csc_, c_ptr);

        return C;
    }

    /* k-parallel tiled kernel: each thread owns a k-stripe → AT fits in L2. */
    py::array_t<float> gemm_tiled(py::array_t<float> A) {
        auto buf = validate_2d_contiguous(A, "A", "2-D [M, K] float32");
        int M = (int)buf.shape[0];
        int K = (int)buf.shape[1];
        if (K != K_)
            throw std::invalid_argument(
                "A columns (" + std::to_string(K) +
                ") must match weight rows (" + std::to_string(K_) + ")"
            );

        py::array_t<float> C({M, N_});
        auto c_buf = C.request();
        const float* a_ptr = static_cast<const float*>(buf.ptr);
        float*       c_ptr = static_cast<float*>(c_buf.ptr);

        ternary_gemm_zero_skip_tiled(M, N_, K_, a_ptr, csr_, c_ptr);
        return C;
    }

    py::dict info() const {
        py::dict d;
        d["K"]                = K_;
        d["N"]                = N_;
        d["nnz"]              = csc_->nnz;
        d["total"]            = K_ * N_;
        d["zero_fraction"]    = 1.0 - (double)csc_->nnz / (K_ * N_);
        d["nonzero_fraction"] = (double)csc_->nnz / (K_ * N_);
        return d;
    }

    int K() const { return K_; }
    int N() const { return N_; }
    int nnz() const { return csc_->nnz; }

private:
    int          K_, N_;
    TernaryCSC*  csc_;   /* column-indexed; used by gemm() */
    TernaryCSR*  csr_;   /* row-indexed;    used by gemm_tiled() */
};

/* -------------------------------------------------------------------------
 * DenseWeights class: holds a precomputed packed-dense layout for one
 * weight matrix. See ternary_gemm_dense.h for why this beats the
 * CSC/CSR-based ZeroSkipWeights above at ternary's actual sparsity.
 * ---------------------------------------------------------------------- */

class DenseWeights {
public:
    DenseWeights(py::array_t<int8_t> B) {
        auto buf = validate_2d_contiguous(B, "B", "2-D [K, N] int8");

        K_ = (int)buf.shape[0];
        N_ = (int)buf.shape[1];
        const int8_t* data = static_cast<const int8_t*>(buf.ptr);
        packed_ = pack_ternary_dense(data, K_, N_);
        if (!packed_)
            throw std::runtime_error("Failed to pack dense weight matrix");
    }

    ~DenseWeights() {
        free_ternary_packed(packed_);
    }

    DenseWeights(const DenseWeights&) = delete;
    DenseWeights& operator=(const DenseWeights&) = delete;

    py::array_t<float> gemm(py::array_t<float> A, bool use_avx2 = true) {
        auto buf = validate_2d_contiguous(A, "A", "2-D [M, K] float32");
        int M = (int)buf.shape[0];
        int K = (int)buf.shape[1];
        if (K != K_)
            throw std::invalid_argument(
                "A columns (" + std::to_string(K) +
                ") must match weight rows (" + std::to_string(K_) + ")"
            );

        py::array_t<float> C({M, N_});
        auto c_buf = C.request();
        const float* a_ptr = static_cast<const float*>(buf.ptr);
        float*       c_ptr = static_cast<float*>(c_buf.ptr);

        if (use_avx2)
            ternary_gemm_packed_avx2(M, N_, K_, a_ptr, packed_, c_ptr);
        else
            ternary_gemm_packed_scalar(M, N_, K_, a_ptr, packed_, c_ptr);

        return C;
    }

    py::dict info() const {
        py::dict d;
        d["K"] = K_;
        d["N"] = N_;
        d["n_blocks"] = packed_->n_blocks;
        /* Packed storage is always K*n_blocks*8 bytes (dense, zero-padded
         * tail) -- unlike ZeroSkipWeights, this does not vary with
         * sparsity, and is smaller than the CSC/CSR index whenever
         * non-zero density exceeds 1/5 (see ternary_gemm_dense.h). */
        d["packed_bytes"] = (long long)packed_->n_blocks * K_ * 8;
        return d;
    }

    int K() const { return K_; }
    int N() const { return N_; }

private:
    int             K_, N_;
    TernaryPacked*  packed_;
};

/* -------------------------------------------------------------------------
 * Module-level convenience functions
 * ---------------------------------------------------------------------- */

py::array_t<float> py_gemm(
    py::array_t<float>  A,
    py::array_t<int8_t> B
) {
    auto a_buf = validate_2d_contiguous(A, "A", "2-D [M, K] float32");
    auto b_buf = validate_2d_contiguous(B, "B", "2-D [K, N] int8");

    int M = (int)a_buf.shape[0];
    int K = (int)a_buf.shape[1];
    int K2= (int)b_buf.shape[0];
    int N = (int)b_buf.shape[1];

    if (K != K2)
        throw std::invalid_argument("A columns must match B rows");

    py::array_t<float> C({M, N});
    auto c_buf = C.request();

    ternary_gemm_zero_skip(
        M, N, K,
        static_cast<const float*>(a_buf.ptr),
        static_cast<const int8_t*>(b_buf.ptr),
        static_cast<float*>(c_buf.ptr)
    );

    return C;
}

py::dict py_sparsity_info(py::array_t<int8_t> B) {
    // Cluster C fix also caught a real latent bug here (2026-08-18): this
    // function indexes `data[i]` for i in [0, K*N) assuming row-major
    // C-contiguous layout, but never actually checked it -- the other 4
    // validation sites in this file all did. A non-contiguous B (e.g. a
    // transposed view) would have silently read wrong values instead of
    // raising a clear error. validate_2d_contiguous() closes that gap.
    auto buf = validate_2d_contiguous(B, "B", "2-D [K, N] int8");

    int K   = (int)buf.shape[0];
    int N   = (int)buf.shape[1];
    const int8_t* data = static_cast<const int8_t*>(buf.ptr);

    int nnz = 0, n_pos = 0, n_neg = 0;
    for (int i = 0; i < K * N; i++) {
        if      (data[i] > 0) { nnz++; n_pos++; }
        else if (data[i] < 0) { nnz++; n_neg++; }
    }

    py::dict d;
    d["K"]            = K;
    d["N"]            = N;
    d["total"]        = K * N;
    d["nnz"]          = nnz;
    d["n_pos"]        = n_pos;
    d["n_neg"]        = n_neg;
    d["n_zero"]       = K * N - nnz;
    d["zero_fraction"]    = (double)(K * N - nnz) / (K * N);
    d["nonzero_fraction"] = (double)nnz / (K * N);
    return d;
}

/* -------------------------------------------------------------------------
 * Module definition
 * ---------------------------------------------------------------------- */

PYBIND11_MODULE(ternary_zero_skip_gemm, m) {
    m.doc() = R"doc(
        Ternary GEMM: two strategies, DenseWeights and ZeroSkipWeights

        DenseWeights (added 2026-08-20, prefer this): packs the dense int8
        weight matrix once, no sparse index. At ternary's actual ~33-40%
        zero density, a CSC/CSR index costs MORE bytes than the dense array
        itself, and these kernels are memory-bandwidth-bound -- measured
        4.5x-9.5x faster than NumPy at batch=1 (vs ZeroSkipWeights' loss),
        see src/core/simd/ternary_gemm_dense.h and
        reports/2026-08-20/GEMM_DENSE_PACKED_OPTIMIZATION.md.

        ZeroSkipWeights (original): precomputed CSC/CSR sparse index,
        skips zero weights entirely. Kept for comparison / regression
        tracking. See: research/FINDINGS.md, § p-adic / 3-adic Structure

        Typical usage:
            import numpy as np
            import ternary_zero_skip_gemm as zs

            W = np.random.choice([-1, 0, 1], size=(K, N)).astype(np.int8)
            weights = zs.DenseWeights(W)      # pack once per weight matrix
            print(weights.info())

            A = np.random.randn(M, K).astype(np.float32)
            C = weights.gemm(A)               # [M, N] float32
    )doc";

    py::class_<ZeroSkipWeights>(m, "ZeroSkipWeights",
        "Precomputed CSC sparse structure for one ternary weight matrix.")
        .def(py::init<py::array_t<int8_t>>(), py::arg("B"),
             "Build sparse index from int8 B [K, N] with values in {-1, 0, +1}.")
        .def("gemm", &ZeroSkipWeights::gemm,
             py::arg("A"), py::arg("use_avx2") = true,
             "Compute C = A @ B (j-parallel, CSC).  Returns [M, N] float32.")
        .def("gemm_tiled", &ZeroSkipWeights::gemm_tiled,
             py::arg("A"),
             "Compute C = A @ B (k-parallel tiled, CSR).  Each thread's AT "
             "stripe fits in L2; reduces thread-private CT arrays.  Returns [M, N] float32.")
        .def("info", &ZeroSkipWeights::info,
             "Return dict with K, N, nnz, zero_fraction.")
        .def_property_readonly("K",   &ZeroSkipWeights::K)
        .def_property_readonly("N",   &ZeroSkipWeights::N)
        .def_property_readonly("nnz", &ZeroSkipWeights::nnz);

    m.def("gemm", &py_gemm,
          py::arg("A"), py::arg("B"),
          "Convenience: build sparse index on-the-fly and compute C = A @ B.");

    m.def("sparsity_info", &py_sparsity_info,
          py::arg("B"),
          "Measure zero fraction and non-zero counts for a ternary weight matrix.");

    py::class_<DenseWeights>(m, "DenseWeights",
        "Precomputed dense-packed structure for one ternary weight matrix. "
        "Prefer this over ZeroSkipWeights -- see module docstring.")
        .def(py::init<py::array_t<int8_t>>(), py::arg("B"),
             "Pack B [K, N] int8 with values in {-1, 0, +1} into contiguous "
             "8-column blocks once.")
        .def("gemm", &DenseWeights::gemm,
             py::arg("A"), py::arg("use_avx2") = true,
             "Compute C = A @ B (N-vectorized AVX2, M-blocked by 4). "
             "Returns [M, N] float32.")
        .def("info", &DenseWeights::info,
             "Return dict with K, N, n_blocks, packed_bytes.")
        .def_property_readonly("K", &DenseWeights::K)
        .def_property_readonly("N", &DenseWeights::N);

#ifdef __AVX2__
    m.attr("has_avx2") = true;
#else
    m.attr("has_avx2") = false;
#endif

#ifdef _OPENMP
    m.attr("has_openmp") = true;
#else
    m.attr("has_openmp") = false;
#endif
}
