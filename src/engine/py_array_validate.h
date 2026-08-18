// py_array_validate.h — Shared pybind11 array-validation helpers
//
// Copyright (c) 2026 Jonathan Verdun (Ternary Engine Project)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// =============================================================================
// PURPOSE
// =============================================================================
//
// "2-D, C-contiguous" is the one validation shape every GEMM binding in this
// project needs before touching a numpy array's raw pointer. Before this
// header existed, bindings_zero_skip_gemm.cpp hand-rolled it at 5 separate
// call sites and bindings_tritnet_gemm.cpp at 2 more, with zero sharing --
// CLAUDE.md gap #6, Cluster C (see reports/2026-08-18/GAP6_DUPLICATION_SCOPE.md).
// A visible cost of that duplication: the error wording had already drifted
// between two copies of the *same* check within bindings_zero_skip_gemm.cpp
// itself ("A must be 2-D [M, K] float32 array" vs. "...float32", no trailing
// word).
//
// This header factors out ONLY what's genuinely shared -- dimensionality and
// memory-layout checks. Each call site keeps its own domain-specific shape
// description (passed in as `shape_desc`) and any exact-size check against
// caller-supplied M/N/K (bindings_tritnet_gemm.cpp's py_gemm()/py_gemm_scaled()
// validate the array shape matches specific ints; bindings_zero_skip_gemm.cpp
// derives M/N/K from the array shape instead of checking against pre-supplied
// values) -- that difference is real, not duplicated, so it stays at each
// call site rather than being forced into one over-general helper.
//
// =============================================================================

#ifndef TERNARY_PY_ARRAY_VALIDATE_H
#define TERNARY_PY_ARRAY_VALIDATE_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

namespace ternary_engine {

namespace py = pybind11;

// Validates that `arr` is a 2-D, C-contiguous array. Returns the buffer_info
// (shape/ptr/etc.) for the caller to use -- callers needing an exact shape
// match beyond "2-D" (e.g. shape[0] must equal a specific M) check
// buf.shape themselves after calling this.
//
// `name` is the parameter name for error messages (e.g. "A"); `shape_desc`
// is a human-readable description of the expected shape/dtype (e.g.
// "2-D [M, K] float32") used only in the ndim error message, since dtype
// itself is enforced by `arr`'s static type (py::array_t<T>), not checked
// here.
template <typename T>
inline py::buffer_info validate_2d_contiguous(const py::array_t<T>& arr,
                                               const char* name,
                                               const char* shape_desc) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 2) {
        throw std::invalid_argument(std::string(name) + " must be " + shape_desc);
    }
    if (!arr.attr("flags").attr("c_contiguous").template cast<bool>()) {
        throw std::invalid_argument(std::string(name) + " must be C-contiguous (row-major)");
    }
    return buf;
}

// 1-D counterpart (used by bindings_tritnet_gemm.cpp's `scales` parameter).
template <typename T>
inline py::buffer_info validate_1d_contiguous(const py::array_t<T>& arr,
                                               const char* name,
                                               const char* shape_desc) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1) {
        throw std::invalid_argument(std::string(name) + " must be " + shape_desc);
    }
    if (!arr.attr("flags").attr("c_contiguous").template cast<bool>()) {
        throw std::invalid_argument(std::string(name) + " must be C-contiguous (row-major)");
    }
    return buf;
}

}  // namespace ternary_engine

#endif  // TERNARY_PY_ARRAY_VALIDATE_H
