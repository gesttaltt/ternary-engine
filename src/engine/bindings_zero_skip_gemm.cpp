/**
 * bindings_zero_skip_gemm.cpp - Python Bindings for Zero-Skip Ternary GEMM
 *
 * Copyright 2025 Ternary Engine Contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Exposes the zero-skip kernel as a Python module: ternary_zero_skip_gemm
 *
 * Python API:
 *
 *   weights = ZeroSkipWeights(B_int8)    # precompute sparse index once
 *   C = weights.gemm(A_float32)          # run GEMM (scalar or AVX2)
 *
 *   C = gemm(A_float32, B_int8)          # convenience all-in-one
 *
 *   info = sparsity_info(B_int8)         # measure zero fraction, nnz, etc.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "ternary_gemm_zero_skip.h"
#include <stdexcept>
#include <string>

namespace py = pybind11;

/* -------------------------------------------------------------------------
 * ZeroSkipWeights class: holds precomputed CSC for one weight matrix
 * ---------------------------------------------------------------------- */

class ZeroSkipWeights {
public:
    ZeroSkipWeights(py::array_t<int8_t> B) {
        auto buf = B.request();
        if (buf.ndim != 2)
            throw std::invalid_argument("B must be 2-D [K, N] int8 array");
        if (!B.attr("flags").attr("c_contiguous").cast<bool>())
            throw std::invalid_argument("B must be C-contiguous (row-major)");

        K_   = (int)buf.shape[0];
        N_   = (int)buf.shape[1];
        csc_ = build_ternary_csc(
            static_cast<const int8_t*>(buf.ptr), K_, N_
        );
        if (!csc_)
            throw std::runtime_error("Failed to build sparse index");
    }

    ~ZeroSkipWeights() { free_ternary_csc(csc_); }

    /* Disable copy; only move semantics make sense for an owned C struct */
    ZeroSkipWeights(const ZeroSkipWeights&) = delete;
    ZeroSkipWeights& operator=(const ZeroSkipWeights&) = delete;

    py::array_t<float> gemm(py::array_t<float> A, bool use_avx2 = true) {
        auto buf = A.request();
        if (buf.ndim != 2)
            throw std::invalid_argument("A must be 2-D [M, K] float32 array");
        int M = (int)buf.shape[0];
        int K = (int)buf.shape[1];
        if (K != K_)
            throw std::invalid_argument(
                "A columns (" + std::to_string(K) +
                ") must match weight rows (" + std::to_string(K_) + ")"
            );
        if (!A.attr("flags").attr("c_contiguous").cast<bool>())
            throw std::invalid_argument("A must be C-contiguous");

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

    py::dict info() const {
        py::dict d;
        d["K"]             = K_;
        d["N"]             = N_;
        d["nnz"]           = csc_->nnz;
        d["total"]         = K_ * N_;
        d["zero_fraction"] = 1.0 - (double)csc_->nnz / (K_ * N_);
        d["nonzero_fraction"] = (double)csc_->nnz / (K_ * N_);
        return d;
    }

    int K() const { return K_; }
    int N() const { return N_; }
    int nnz() const { return csc_->nnz; }

private:
    int          K_, N_;
    TernaryCSC*  csc_;
};

/* -------------------------------------------------------------------------
 * Module-level convenience functions
 * ---------------------------------------------------------------------- */

py::array_t<float> py_gemm(
    py::array_t<float>  A,
    py::array_t<int8_t> B
) {
    auto a_buf = A.request();
    auto b_buf = B.request();

    if (a_buf.ndim != 2)
        throw std::invalid_argument("A must be 2-D [M, K] float32");
    if (b_buf.ndim != 2)
        throw std::invalid_argument("B must be 2-D [K, N] int8");

    int M = (int)a_buf.shape[0];
    int K = (int)a_buf.shape[1];
    int K2= (int)b_buf.shape[0];
    int N = (int)b_buf.shape[1];

    if (K != K2)
        throw std::invalid_argument("A columns must match B rows");

    if (!A.attr("flags").attr("c_contiguous").cast<bool>())
        throw std::invalid_argument("A must be C-contiguous");
    if (!B.attr("flags").attr("c_contiguous").cast<bool>())
        throw std::invalid_argument("B must be C-contiguous");

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
    auto buf = B.request();
    if (buf.ndim != 2)
        throw std::invalid_argument("B must be 2-D [K, N] int8");

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
        Zero-Skip Ternary GEMM

        Implements matrix multiplication for ternary weight matrices {-1, 0, +1}
        by skipping zero weights entirely.  Uses a precomputed CSC sparse index
        to eliminate ~33% of multiply-accumulate operations.

        See: research/FINDINGS.md, § p-adic / 3-adic Structure

        Typical usage:
            import numpy as np
            import ternary_zero_skip_gemm as zs

            # Build sparse index once per weight matrix
            W = np.random.choice([-1, 0, 1], size=(K, N)).astype(np.int8)
            weights = zs.ZeroSkipWeights(W)
            print(weights.info())   # confirms ~33% zero fraction

            # Run GEMM for each batch
            A = np.random.randn(M, K).astype(np.float32)
            C = weights.gemm(A)     # [M, N] float32
    )doc";

    py::class_<ZeroSkipWeights>(m, "ZeroSkipWeights",
        "Precomputed CSC sparse structure for one ternary weight matrix.")
        .def(py::init<py::array_t<int8_t>>(), py::arg("B"),
             "Build sparse index from int8 B [K, N] with values in {-1, 0, +1}.")
        .def("gemm", &ZeroSkipWeights::gemm,
             py::arg("A"), py::arg("use_avx2") = true,
             "Compute C = A @ B using zero-skip kernel.  Returns [M, N] float32.")
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
