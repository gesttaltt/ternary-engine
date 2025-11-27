// reference_cpp.cpp — Unoptimized C++ baseline for fair benchmarking
//
// Copyright (c) 2025 Jonathan Verdun (Ternary Engine Project)
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
// PURPOSE:
// This module provides an unoptimized C++ reference implementation for
// fair performance comparison. It uses:
// - NO SIMD (scalar operations only)
// - NO LUTs (conversion-based logic)
// - NO force inline
// - Minimal compiler optimizations (/O1 on MSVC, -O1 on GCC)
//
// This measures the actual benefit of optimizations (LUTs, SIMD, etc.)
// rather than just Python vs C++ differences.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <cstddef>

namespace py = pybind11;

// ssize_t compatibility for MSVC
#ifdef _MSC_VER
    #include <BaseTsd.h>
    typedef SSIZE_T ssize_t;
#endif

// =============================================================================
// Conversion Helpers (No LUTs)
// =============================================================================

// Convert 2-bit trit encoding to integer {-1, 0, +1}
inline int trit_to_int(uint8_t trit) {
    switch (trit & 0b11) {
        case 0b00: return -1;
        case 0b01: return  0;
        case 0b10: return +1;
        default:   return  0;  // Invalid (sanitize to 0)
    }
}

// Convert integer {-1, 0, +1} to 2-bit trit encoding
inline uint8_t int_to_trit(int value) {
    if (value < 0) return 0b00;
    if (value > 0) return 0b10;
    return 0b01;
}

// =============================================================================
// Unoptimized Scalar Operations (Conversion-based)
// =============================================================================

// Saturated addition (clamps to [-1, +1])
uint8_t tadd_scalar(uint8_t a, uint8_t b) {
    int ia = trit_to_int(a);
    int ib = trit_to_int(b);
    int sum = ia + ib;

    // Saturate to [-1, +1]
    if (sum < -1) sum = -1;
    if (sum > +1) sum = +1;

    return int_to_trit(sum);
}

// Multiplication
uint8_t tmul_scalar(uint8_t a, uint8_t b) {
    int ia = trit_to_int(a);
    int ib = trit_to_int(b);
    int product = ia * ib;
    return int_to_trit(product);
}

// Minimum
uint8_t tmin_scalar(uint8_t a, uint8_t b) {
    int ia = trit_to_int(a);
    int ib = trit_to_int(b);
    int minimum = (ia < ib) ? ia : ib;
    return int_to_trit(minimum);
}

// Maximum
uint8_t tmax_scalar(uint8_t a, uint8_t b) {
    int ia = trit_to_int(a);
    int ib = trit_to_int(b);
    int maximum = (ia > ib) ? ia : ib;
    return int_to_trit(maximum);
}

// Negation (sign flip)
uint8_t tnot_scalar(uint8_t a) {
    int ia = trit_to_int(a);
    int negated = -ia;
    return int_to_trit(negated);
}

// =============================================================================
// Array Processing (Pure Scalar, No Optimizations)
// =============================================================================

template <typename ScalarOp>
py::array_t<uint8_t> process_binary_array_ref(
    py::array_t<uint8_t> A,
    py::array_t<uint8_t> B,
    ScalarOp scalar_op
) {
    auto a = A.unchecked<1>();
    auto b = B.unchecked<1>();
    ssize_t n = A.size();

    // Validation
    if (n != B.size()) {
        throw std::runtime_error("Array size mismatch");
    }

    py::array_t<uint8_t> out(n);
    auto r = out.mutable_unchecked<1>();

    // Pure scalar loop (no SIMD, no unrolling, no optimizations)
    for (ssize_t i = 0; i < n; ++i) {
        r[i] = scalar_op(a[i], b[i]);
    }

    return out;
}

template <typename ScalarOp>
py::array_t<uint8_t> process_unary_array_ref(
    py::array_t<uint8_t> A,
    ScalarOp scalar_op
) {
    auto a = A.unchecked<1>();
    ssize_t n = A.size();

    py::array_t<uint8_t> out(n);
    auto r = out.mutable_unchecked<1>();

    // Pure scalar loop
    for (ssize_t i = 0; i < n; ++i) {
        r[i] = scalar_op(a[i]);
    }

    return out;
}

// =============================================================================
// Python Bindings
// =============================================================================

py::array_t<uint8_t> tadd_ref(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return process_binary_array_ref(A, B, tadd_scalar);
}

py::array_t<uint8_t> tmul_ref(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return process_binary_array_ref(A, B, tmul_scalar);
}

py::array_t<uint8_t> tmin_ref(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return process_binary_array_ref(A, B, tmin_scalar);
}

py::array_t<uint8_t> tmax_ref(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return process_binary_array_ref(A, B, tmax_scalar);
}

py::array_t<uint8_t> tnot_ref(py::array_t<uint8_t> A) {
    return process_unary_array_ref(A, tnot_scalar);
}

// =============================================================================
// Module Definition
// =============================================================================

PYBIND11_MODULE(reference_cpp, m) {
    m.doc() = "Unoptimized C++ reference implementation for fair benchmarking";

    m.def("tadd", &tadd_ref, "Saturated addition (unoptimized)",
          py::arg("a"), py::arg("b"));

    m.def("tmul", &tmul_ref, "Multiplication (unoptimized)",
          py::arg("a"), py::arg("b"));

    m.def("tmin", &tmin_ref, "Minimum (unoptimized)",
          py::arg("a"), py::arg("b"));

    m.def("tmax", &tmax_ref, "Maximum (unoptimized)",
          py::arg("a"), py::arg("b"));

    m.def("tnot", &tnot_ref, "Negation (unoptimized)",
          py::arg("a"));
}
