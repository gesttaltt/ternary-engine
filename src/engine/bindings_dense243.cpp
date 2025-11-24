// ternary_dense243_module.cpp — Dense243 Python Module with TritNet Architecture
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
// =============================================================================
// MODULE DESIGN: TritNet-Ready Architecture
// =============================================================================
//
// This module is designed to support future TritNet integration:
//
// PHASE 1 (Current): LUT-based dense243 operations
//   - Pack/unpack: 5 trits ↔ 1 byte (95.3% density)
//   - Operations: LUT-based arithmetic on dense243 format
//
// PHASE 2 (TritNet): Neural network replacement
//   - Train BitNet on dense243 truth tables (243 states)
//   - Distill to ternary weights {-1, 0, +1}
//   - Replace LUT calls with tiny matmul operations
//
// ARCHITECTURE:
//   - Operation interface: Allows swapping LUT ↔ matmul backends
//   - Modular design: Pack/unpack separate from operations
//   - TritNet hooks: Backend selection at runtime
//
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <string>
#include <cstdint>

// Dense243 implementation
#include "dense243/ternary_dense243.h"
#include "dense243/ternary_dense243_simd.h"

// Core ternary algebra (for 2-bit ↔ dense243 conversion)
#include "core/algebra/ternary_algebra.h"
#include "core/common/ternary_errors.h"

namespace py = pybind11;

// MSVC doesn't have ssize_t, use Py_ssize_t from pybind11
#ifdef _MSC_VER
using ssize_t = py::ssize_t;
#endif

// =============================================================================
// Operation Backend Selection (TritNet-Ready)
// =============================================================================

enum class OperationBackend {
    LUT,      // Current: Lookup table based (default)
    TRITNET   // Future: Tiny neural network based
};

static OperationBackend g_backend = OperationBackend::LUT;

// Backend selection API
void set_operation_backend(const std::string& backend) {
    if (backend == "lut") {
        g_backend = OperationBackend::LUT;
    } else if (backend == "tritnet") {
        // Future: Load TritNet weights and prepare for inference
        throw std::runtime_error("TritNet backend not yet implemented. Stay tuned!");
    } else {
        throw std::invalid_argument("Unknown backend: " + backend + ". Use 'lut' or 'tritnet'");
    }
}

std::string get_operation_backend() {
    return (g_backend == OperationBackend::LUT) ? "lut" : "tritnet";
}

// =============================================================================
// Dense243 Packing/Unpacking (2-bit ↔ Dense243 Conversion)
// =============================================================================

py::array_t<uint8_t> pack_to_dense243(py::array_t<uint8_t> trits_2bit) {
    auto trits = trits_2bit.unchecked<1>();
    ssize_t n = trits_2bit.size();

    // Calculate required bytes (round up to nearest 5-trit group)
    size_t num_bytes = dense243_bytes_for_trits(n);

    py::array_t<uint8_t> packed(num_bytes);
    auto packed_data = packed.mutable_unchecked<1>();

    // Pack 5 trits at a time
    size_t byte_idx = 0;
    for (ssize_t i = 0; i < n; i += 5) {
        uint8_t t0 = (i + 0 < n) ? trits[i + 0] : 0b01;  // Pad with zeros
        uint8_t t1 = (i + 1 < n) ? trits[i + 1] : 0b01;
        uint8_t t2 = (i + 2 < n) ? trits[i + 2] : 0b01;
        uint8_t t3 = (i + 3 < n) ? trits[i + 3] : 0b01;
        uint8_t t4 = (i + 4 < n) ? trits[i + 4] : 0b01;

        packed_data[byte_idx++] = dense243_pack(t0, t1, t2, t3, t4);
    }

    return packed;
}

py::array_t<uint8_t> unpack_from_dense243(py::array_t<uint8_t> packed, ssize_t num_trits) {
    auto packed_data = packed.unchecked<1>();
    size_t num_bytes = packed.size();

    // If num_trits not specified, use maximum (5 trits per byte)
    if (num_trits < 0) {
        num_trits = num_bytes * 5;
    }

    py::array_t<uint8_t> trits_2bit(num_trits);
    auto trits = trits_2bit.mutable_unchecked<1>();

    // Unpack byte by byte
    size_t trit_idx = 0;
    for (size_t byte_idx = 0; byte_idx < num_bytes && trit_idx < num_trits; ++byte_idx) {
        Dense243Unpacked unpacked = dense243_unpack(packed_data[byte_idx]);

        if (trit_idx + 0 < num_trits) trits[trit_idx++] = unpacked.t0;
        if (trit_idx + 0 < num_trits) trits[trit_idx++] = unpacked.t1;
        if (trit_idx + 0 < num_trits) trits[trit_idx++] = unpacked.t2;
        if (trit_idx + 0 < num_trits) trits[trit_idx++] = unpacked.t3;
        if (trit_idx + 0 < num_trits) trits[trit_idx++] = unpacked.t4;
    }

    return trits_2bit;
}

// =============================================================================
// Dense243 Direct Operations (TritNet-Ready Interface)
// =============================================================================
// These functions will eventually call TritNet instead of unpacking to 2-bit

// Binary operation template (future: swap LUT for TritNet matmul)
template<typename ScalarOp>
py::array_t<uint8_t> binary_op_dense243(
    py::array_t<uint8_t> A_dense,
    py::array_t<uint8_t> B_dense,
    ScalarOp scalar_op,
    const char* op_name
) {
    auto a_data = A_dense.unchecked<1>();
    auto b_data = B_dense.unchecked<1>();
    ssize_t n = A_dense.size();

    if (n != B_dense.size()) {
        throw ArraySizeMismatchError(n, B_dense.size());
    }

    py::array_t<uint8_t> result(n);
    auto r_data = result.mutable_unchecked<1>();

    // Current: LUT-based implementation
    // Future: Check g_backend and call TritNet if enabled

    for (ssize_t i = 0; i < n; ++i) {
        // Unpack both bytes
        Dense243Unpacked a = dense243_unpack(a_data[i]);
        Dense243Unpacked b = dense243_unpack(b_data[i]);

        // Perform operation on each trit position
        uint8_t r0 = scalar_op(a.t0, b.t0);
        uint8_t r1 = scalar_op(a.t1, b.t1);
        uint8_t r2 = scalar_op(a.t2, b.t2);
        uint8_t r3 = scalar_op(a.t3, b.t3);
        uint8_t r4 = scalar_op(a.t4, b.t4);

        // Pack result
        r_data[i] = dense243_pack(r0, r1, r2, r3, r4);
    }

    return result;
}

// Unary operation template
template<typename ScalarOp>
py::array_t<uint8_t> unary_op_dense243(
    py::array_t<uint8_t> A_dense,
    ScalarOp scalar_op,
    const char* op_name
) {
    auto a_data = A_dense.unchecked<1>();
    ssize_t n = A_dense.size();

    py::array_t<uint8_t> result(n);
    auto r_data = result.mutable_unchecked<1>();

    for (ssize_t i = 0; i < n; ++i) {
        Dense243Unpacked a = dense243_unpack(a_data[i]);

        uint8_t r0 = scalar_op(a.t0);
        uint8_t r1 = scalar_op(a.t1);
        uint8_t r2 = scalar_op(a.t2);
        uint8_t r3 = scalar_op(a.t3);
        uint8_t r4 = scalar_op(a.t4);

        r_data[i] = dense243_pack(r0, r1, r2, r3, r4);
    }

    return result;
}

// Exposed operations
py::array_t<uint8_t> tadd_dense243(py::array_t<uint8_t> a, py::array_t<uint8_t> b) {
    return binary_op_dense243(a, b, tadd, "tadd");
}

py::array_t<uint8_t> tmul_dense243(py::array_t<uint8_t> a, py::array_t<uint8_t> b) {
    return binary_op_dense243(a, b, tmul, "tmul");
}

py::array_t<uint8_t> tmin_dense243(py::array_t<uint8_t> a, py::array_t<uint8_t> b) {
    return binary_op_dense243(a, b, tmin, "tmin");
}

py::array_t<uint8_t> tmax_dense243(py::array_t<uint8_t> a, py::array_t<uint8_t> b) {
    return binary_op_dense243(a, b, tmax, "tmax");
}

py::array_t<uint8_t> tnot_dense243(py::array_t<uint8_t> a) {
    return unary_op_dense243(a, tnot, "tnot");
}

// =============================================================================
// Utility Functions
// =============================================================================

size_t calculate_dense243_size(size_t num_trits) {
    return dense243_bytes_for_trits(num_trits);
}

size_t calculate_trits_from_dense243(size_t num_bytes) {
    return dense243_trits_in_bytes(num_bytes);
}

bool validate_dense243_byte(uint8_t byte_val) {
    return dense243_is_valid(byte_val);
}

size_t validate_dense243_array(py::array_t<uint8_t> data) {
    auto arr = data.unchecked<1>();
    size_t invalid_count = 0;

    for (ssize_t i = 0; i < arr.size(); ++i) {
        if (!dense243_is_valid(arr[i])) {
            invalid_count++;
        }
    }

    return invalid_count;
}

// =============================================================================
// Module Definition
// =============================================================================

PYBIND11_MODULE(ternary_dense243_module, m) {
    m.doc() = R"pbdoc(
        Ternary Dense243 Module - High-Density Ternary Encoding

        Provides 5 trits/byte encoding (95.3% density) with TritNet-ready architecture.

        Features:
        - Pack/unpack: Convert between 2-bit and dense243 formats
        - Direct operations: Operate on dense243-encoded data
        - TritNet hooks: Backend selection for future neural network integration

        Encoding:
        - 2-bit format: 4 trits/byte (standard, fast operations)
        - Dense243 format: 5 trits/byte (compact storage, slower operations)

        Use cases:
        - Persistent storage: 20% space savings
        - Network transmission: 20% bandwidth reduction
        - Large datasets: Memory-bound workloads (>10M elements)

        Future TritNet Integration:
        - Train tiny BitNet on dense243 truth tables
        - Distill to ternary weights {-1, 0, +1}
        - Replace LUT operations with matmul calls
    )pbdoc";

    // Backend selection
    m.def("set_backend", &set_operation_backend,
          py::arg("backend"),
          "Set operation backend: 'lut' (current) or 'tritnet' (future)");

    m.def("get_backend", &get_operation_backend,
          "Get current operation backend");

    // Packing/unpacking
    m.def("pack", &pack_to_dense243,
          py::arg("trits_2bit"),
          "Pack 2-bit trits to dense243 format (5 trits → 1 byte)");

    m.def("unpack", &unpack_from_dense243,
          py::arg("packed"),
          py::arg("num_trits") = -1,
          "Unpack dense243 to 2-bit trits (1 byte → 5 trits)");

    // Direct operations on dense243 format
    m.def("tadd", &tadd_dense243,
          py::arg("a"), py::arg("b"),
          "Ternary addition on dense243-encoded data");

    m.def("tmul", &tmul_dense243,
          py::arg("a"), py::arg("b"),
          "Ternary multiplication on dense243-encoded data");

    m.def("tmin", &tmin_dense243,
          py::arg("a"), py::arg("b"),
          "Ternary minimum on dense243-encoded data");

    m.def("tmax", &tmax_dense243,
          py::arg("a"), py::arg("b"),
          "Ternary maximum on dense243-encoded data");

    m.def("tnot", &tnot_dense243,
          py::arg("a"),
          "Ternary negation on dense243-encoded data");

    // Utilities
    m.def("bytes_for_trits", &calculate_dense243_size,
          py::arg("num_trits"),
          "Calculate dense243 bytes needed for N trits");

    m.def("trits_in_bytes", &calculate_trits_from_dense243,
          py::arg("num_bytes"),
          "Calculate number of trits in N dense243 bytes");

    m.def("is_valid_byte", &validate_dense243_byte,
          py::arg("byte_value"),
          "Check if byte is valid dense243 value (0-242)");

    m.def("validate_array", &validate_dense243_array,
          py::arg("data"),
          "Count invalid bytes in dense243 array");

    // Module metadata
    m.attr("__version__") = "1.0.0";
    m.attr("DENSITY") = 5.0 / 1.0;  // 5 trits per byte
    m.attr("STATE_UTILIZATION") = 243.0 / 256.0;  // 95.3%
}
