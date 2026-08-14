/**
 * bindings_tritnet_inference.cpp - Python Bindings for TritNet C++ Inference Engine
 *
 * Copyright 2025 Ternary Engine Contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Exposes models/tritnet/inference/{tritnet_inference.h,tritnet_inference_avx2.h}
 * (Phase 3: naive-scalar and AVX2 TritNet forward-pass engines) to Python,
 * batched over numpy arrays of 5-trit chunks. This is the last piece of Phase 3
 * flagged as outstanding in reports/2026-08-14/TRITNET_PHASE3_SESSION_REPORT.md
 * -- until now the engine only existed as standalone C++ headers plus a
 * dev-utility test/benchmark, neither reachable from Python.
 *
 * Runtime CPU dispatch (not just a compile-time flag): this module is always
 * built with the AVX2 kernel included (this project's stated baseline is
 * "AVX2 required", see CLAUDE.md Platform Requirements), but every call
 * checks has_avx2() and falls back to the scalar engine if the executing CPU
 * doesn't actually support AVX2 -- graceful degradation, matching this
 * project's existing convention (e.g. the AVX2-intrinsics-at-import-time
 * crash class of bug fixed in bindings_dense243.cpp, 2026-08-12).
 *
 * Reminder from the decisive benchmark (2026-08-14, Linux x64, AMD Ryzen 5
 * 7520U -- see CLAUDE.md "TritNet Development" -> Phase 3 and
 * reports/2026-08-14/TRITNET_PHASE3_SESSION_REPORT.md for full figures and
 * methodology): the LUT-based ternary_backend/ternary_simd_engine modules
 * win by ~950-1800x at the naive/scalar level and still ~150-210x against
 * this module's own AVX2 kernel. This binding exists to make TritNet
 * reachable and testable from Python -- not because it is competitive with
 * the LUT for this use case.
 *
 * Usage from Python:
 *   import numpy as np
 *   import ternary_tritnet_inference as tn
 *
 *   a = np.array([[0, 1, 2, 0, 1]], dtype=np.uint8)   # trit-encoded: 0b00=-1,0b01=0,0b10=+1
 *   b = np.array([[1, 1, 0, 2, 2]], dtype=np.uint8)
 *   out = tn.tadd(a, b)   # [1, 5] uint8
 *   print(tn.has_avx2())  # whether AVX2 dispatch is active on this machine
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>

#include "models/tritnet/inference/tritnet_inference.h"
#include "models/tritnet/inference/tritnet_inference_avx2.h"  // always compiled in; see file header
#include "core/simd/cpu_simd_capability.h"                    // runtime has_avx2()
#include "core/common/ternary_errors.h"                       // InvalidTritError

namespace py = pybind11;

namespace {

// Validate an [N, 5] uint8 trit-encoded array: shape, contiguity, and that
// every value is a valid 2-bit trit encoding (0b00, 0b01, or 0b10 -- 0b11 is
// unused/invalid). Matches this project's "input_validation - sanitize
// external inputs" convention (see ternary_errors.h's InvalidTritError) --
// this is the first binding to validate per-value trit range; the existing
// LUT bindings (bindings_core_ops.cpp) only validate array-size mismatches.
// Returns the buffer_info so callers reuse it instead of re-requesting it.
py::buffer_info validate_chunks(const py::array_t<uint8_t>& arr, const char* name) {
    auto buf = arr.request();
    if (buf.ndim != 2 || buf.shape[1] != 5) {
        throw std::invalid_argument(
            std::string(name) + " must be a [N, 5] uint8 array (5-trit chunks)");
    }
    if (!(arr.flags() & py::array::c_style)) {
        throw std::invalid_argument(std::string(name) + " must be C-contiguous");
    }
    const uint8_t* data = static_cast<const uint8_t*>(buf.ptr);
    ssize_t total = buf.shape[0] * buf.shape[1];
    for (ssize_t i = 0; i < total; ++i) {
        if (data[i] > 0b10) throw InvalidTritError(data[i]);
    }
    return buf;
}

py::array_t<uint8_t> make_output(ssize_t n) {
    return py::array_t<uint8_t>({n, static_cast<ssize_t>(5)});
}

py::array_t<uint8_t> py_tnot(py::array_t<uint8_t> chunks) {
    py::buffer_info buf = validate_chunks(chunks, "chunks");
    ssize_t n = buf.shape[0];
    auto out = make_output(n);

    const trit* in = static_cast<const trit*>(buf.ptr);
    trit* o = static_cast<trit*>(out.request().ptr);
    bool avx2 = has_avx2();

    for (ssize_t i = 0; i < n; ++i) {
        if (avx2) tritnet::avx2::tritnet_tnot(in + i * 5, o + i * 5);
        else      tritnet::tritnet_tnot(in + i * 5, o + i * 5);
    }
    return out;
}

// Shared implementation for the 4 binary ops -- template on the scalar and
// AVX2 function pointers, matching this project's "template_unification"
// convention (see CLAUDE.md Coding Conventions: one implementation instead
// of 4 hand-duplicated copies).
template<void (*ScalarFn)(const trit[5], const trit[5], trit[5]),
         void (*Avx2Fn)(const trit[5], const trit[5], trit[5])>
py::array_t<uint8_t> binary_op(py::array_t<uint8_t> a, py::array_t<uint8_t> b) {
    py::buffer_info buf_a = validate_chunks(a, "a");
    py::buffer_info buf_b = validate_chunks(b, "b");
    ssize_t n_a = buf_a.shape[0];
    ssize_t n_b = buf_b.shape[0];
    if (n_a != n_b) {
        throw std::invalid_argument(
            "a and b must have the same number of chunks (got " +
            std::to_string(n_a) + " and " + std::to_string(n_b) + ")");
    }

    auto out = make_output(n_a);
    const trit* pa = static_cast<const trit*>(buf_a.ptr);
    const trit* pb = static_cast<const trit*>(buf_b.ptr);
    trit* po = static_cast<trit*>(out.request().ptr);
    bool avx2 = has_avx2();

    for (ssize_t i = 0; i < n_a; ++i) {
        if (avx2) Avx2Fn(pa + i * 5, pb + i * 5, po + i * 5);
        else      ScalarFn(pa + i * 5, pb + i * 5, po + i * 5);
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(ternary_tritnet_inference, m) {
    m.doc() = R"doc(
        TritNet C++ Inference Engine (Phase 3) -- Python bindings

        Neural-network-based ternary arithmetic: replaces LUT lookup with a
        3-layer forward pass over 5-trit chunks. See CLAUDE.md "TritNet
        Development" for the full Phase 3 writeup, including the decisive
        benchmark result (2026-08-14: LUT wins by ~950-1800x at the naive/
        scalar level, still ~150-210x against this module's own AVX2 kernel
        -- this module exists for API completeness / Phase 4+ groundwork,
        not because it's competitive with the LUT-based ternary_backend
        module).

        All functions take/return [N, 5] uint8 arrays of 2-bit trit-encoded
        values (0b00=-1, 0b01=0, 0b10=+1), one row per 5-trit chunk.
        Dispatches to the AVX2 kernel when the running CPU supports it
        (see has_avx2()), falling back to the scalar kernel otherwise --
        both are correctness-verified bit-identical over the full input
        space (tests/cpp/test_tritnet_inference.cpp).

        Typical usage:
            import numpy as np
            import ternary_tritnet_inference as tn

            a = np.array([[0, 1, 2, 0, 1]], dtype=np.uint8)
            b = np.array([[1, 1, 0, 2, 2]], dtype=np.uint8)
            out = tn.tadd(a, b)          # [1, 5] uint8
            print(tn.has_avx2())         # runtime dispatch target on this machine
    )doc";

    m.def("tnot", &py_tnot, py::arg("chunks"),
          "Negate each 5-trit chunk. chunks: [N, 5] uint8. Returns [N, 5] uint8.");

    m.def("tadd", &binary_op<tritnet::tritnet_tadd, tritnet::avx2::tritnet_tadd>,
          py::arg("a"), py::arg("b"),
          "Saturating add, elementwise per trit. a, b: [N, 5] uint8. Returns [N, 5] uint8.");

    m.def("tmul", &binary_op<tritnet::tritnet_tmul, tritnet::avx2::tritnet_tmul>,
          py::arg("a"), py::arg("b"),
          "Multiply, elementwise per trit. a, b: [N, 5] uint8. Returns [N, 5] uint8.");

    m.def("tmin", &binary_op<tritnet::tritnet_tmin, tritnet::avx2::tritnet_tmin>,
          py::arg("a"), py::arg("b"),
          "Elementwise min per trit. a, b: [N, 5] uint8. Returns [N, 5] uint8.");

    m.def("tmax", &binary_op<tritnet::tritnet_tmax, tritnet::avx2::tritnet_tmax>,
          py::arg("a"), py::arg("b"),
          "Elementwise max per trit. a, b: [N, 5] uint8. Returns [N, 5] uint8.");

    m.def("has_avx2", &has_avx2,
          "Runtime check: does the executing CPU support AVX2? Determines "
          "which kernel (scalar or AVX2) the ops above actually dispatch to.");

    // Compile-time attribute: was this module built with the AVX2 kernel at
    // all (distinct from has_avx2()'s runtime check of the executing CPU).
    // Always true given this project's "AVX2 required" baseline (see
    // CLAUDE.md Platform Requirements) -- tritnet_inference_avx2.h is a hard
    // compile error without __AVX2__, so if this module built, this is true.
    m.attr("built_with_avx2") = true;
}
