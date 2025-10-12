// reference_cpp.cpp - Unoptimized C++ reference implementations
//
// Copyright 2025 Ternary Core Contributors
// Licensed under the Apache License, Version 2.0
//
// PURPOSE: Provide baseline C++ implementations WITHOUT optimizations
// for fair performance comparisons. This measures the actual impact
// of Phase 0-1 optimizations, not Python vs C++ differences.
//
// IMPLEMENTATION: Uses conversion-based operations (pre-Phase 0 approach)
// - No LUTs (conversion overhead)
// - No force inline
// - No SIMD
// - Minimal compiler optimizations

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdint.h>

namespace py = pybind11;

// MSVC ssize_t compatibility
#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

// Trit encoding: 0b00=-1, 0b01=0, 0b10=+1
typedef uint8_t trit;

// --- Conversion functions (overhead intentionally included) ---
static inline int trit_to_int(trit t) {
    if (t == 0b00) return -1;
    if (t == 0b10) return 1;
    return 0;
}

static inline trit int_to_trit(int v) {
    if (v < 0) return 0b00;
    if (v > 0) return 0b10;
    return 0b01;
}

// --- Unoptimized scalar operations (pre-Phase 0) ---
// These use conversion overhead and branches, no LUTs

static trit ref_tadd(trit a, trit b) {
    int ai = trit_to_int(a);
    int bi = trit_to_int(b);
    int sum = ai + bi;
    // Saturate to [-1, 1]
    if (sum > 1) sum = 1;
    if (sum < -1) sum = -1;
    return int_to_trit(sum);
}

static trit ref_tmul(trit a, trit b) {
    int ai = trit_to_int(a);
    int bi = trit_to_int(b);
    int prod = ai * bi;
    return int_to_trit(prod);
}

static trit ref_tmin(trit a, trit b) {
    int ai = trit_to_int(a);
    int bi = trit_to_int(b);
    return (ai < bi) ? a : b;
}

static trit ref_tmax(trit a, trit b) {
    int ai = trit_to_int(a);
    int bi = trit_to_int(b);
    return (ai > bi) ? a : b;
}

static trit ref_tnot(trit a) {
    // Pre-Phase 0 approach: conditional chain
    if (a == 0b00) return 0b10;  // -1 → +1
    if (a == 0b10) return 0b00;  // +1 → -1
    return 0b01;                 // 0 → 0
}

// --- Array wrappers (pure scalar, no SIMD) ---

py::array_t<uint8_t> ref_tadd_array(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    auto a = A.unchecked<1>();
    auto b = B.unchecked<1>();
    ssize_t n = A.size();
    if (n != B.size()) throw std::runtime_error("Arrays must match");

    py::array_t<uint8_t> out(n);
    auto r = out.mutable_unchecked<1>();

    // Pure scalar loop, no SIMD, no unrolling
    for (ssize_t i = 0; i < n; ++i) {
        r[i] = ref_tadd(a[i], b[i]);
    }
    return out;
}

py::array_t<uint8_t> ref_tmul_array(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    auto a = A.unchecked<1>();
    auto b = B.unchecked<1>();
    ssize_t n = A.size();
    if (n != B.size()) throw std::runtime_error("Arrays must match");

    py::array_t<uint8_t> out(n);
    auto r = out.mutable_unchecked<1>();

    for (ssize_t i = 0; i < n; ++i) {
        r[i] = ref_tmul(a[i], b[i]);
    }
    return out;
}

py::array_t<uint8_t> ref_tmin_array(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    auto a = A.unchecked<1>();
    auto b = B.unchecked<1>();
    ssize_t n = A.size();
    if (n != B.size()) throw std::runtime_error("Arrays must match");

    py::array_t<uint8_t> out(n);
    auto r = out.mutable_unchecked<1>();

    for (ssize_t i = 0; i < n; ++i) {
        r[i] = ref_tmin(a[i], b[i]);
    }
    return out;
}

py::array_t<uint8_t> ref_tmax_array(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    auto a = A.unchecked<1>();
    auto b = B.unchecked<1>();
    ssize_t n = A.size();
    if (n != B.size()) throw std::runtime_error("Arrays must match");

    py::array_t<uint8_t> out(n);
    auto r = out.mutable_unchecked<1>();

    for (ssize_t i = 0; i < n; ++i) {
        r[i] = ref_tmax(a[i], b[i]);
    }
    return out;
}

py::array_t<uint8_t> ref_tnot_array(py::array_t<uint8_t> A) {
    auto a = A.unchecked<1>();
    ssize_t n = A.size();

    py::array_t<uint8_t> out(n);
    auto r = out.mutable_unchecked<1>();

    for (ssize_t i = 0; i < n; ++i) {
        r[i] = ref_tnot(a[i]);
    }
    return out;
}

PYBIND11_MODULE(reference_cpp, m) {
    m.doc() = "Unoptimized C++ reference implementations for fair benchmarking";

    m.def("tadd", &ref_tadd_array, "Unoptimized ternary addition");
    m.def("tmul", &ref_tmul_array, "Unoptimized ternary multiplication");
    m.def("tmin", &ref_tmin_array, "Unoptimized ternary minimum");
    m.def("tmax", &ref_tmax_array, "Unoptimized ternary maximum");
    m.def("tnot", &ref_tnot_array, "Unoptimized ternary negation");
}
