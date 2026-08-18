/**
 * bindings_backend_api.cpp - Python Bindings for Backend System
 *
 * Copyright 2025 Ternary Engine Contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Exposes the v1.2.0 backend system to Python:
 * - Backend initialization and discovery
 * - Backend selection and querying
 * - Dispatch operations using selected backend
 *
 * Usage from Python:
 *   import ternary_backend
 *   ternary_backend.init()
 *   backends = ternary_backend.list_backends()
 *   ternary_backend.set_backend("AVX2_v2")
 *   result = ternary_backend.tadd(a, b)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "core/simd/backend_plugin_api.h"
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>

namespace py = pybind11;

// ============================================================================
// Backend Info Wrapper (for Python access)
// ============================================================================

struct BackendInfoPython {
    std::string name;
    std::string description;
    std::string version;
    uint32_t capabilities;
    size_t preferred_batch_size;
    bool is_available;
    bool is_active;
};

// ============================================================================
// Initialization and Management
// ============================================================================

/**
 * Initialize backend system
 *
 * Call once at program startup to detect and register all backends
 */
bool init_backends() {
    return ternary_backend_init();
}

/**
 * Shutdown backend system
 */
void shutdown_backends() {
    ternary_backend_shutdown();
}

/**
 * List all available backends
 *
 * Returns list of backend info dictionaries
 */
std::vector<BackendInfoPython> list_backends() {
    std::vector<BackendInfoPython> backends;

    const TernaryBackend* active = ternary_backend_get_active();
    size_t count = ternary_backend_count();

    for (size_t i = 0; i < count; i++) {
        const TernaryBackend* backend = ternary_backend_get(i);
        if (!backend) continue;

        BackendInfoPython info;
        info.name = backend->info.name;
        info.description = backend->info.description;

        // Format version string
        uint32_t ver = backend->info.version;
        std::ostringstream version_ss;
        version_ss << ((ver >> 16) & 0xFF) << "."
                  << ((ver >> 8) & 0xFF) << "."
                  << (ver & 0xFF);
        info.version = version_ss.str();

        info.capabilities = backend->info.capabilities;
        info.preferred_batch_size = backend->info.preferred_batch_size;
        info.is_available = backend->info.is_available ? backend->info.is_available() : true;
        info.is_active = (backend == active);

        backends.push_back(info);
    }

    return backends;
}

/**
 * Get active backend name
 */
std::string get_active_backend() {
    const TernaryBackend* backend = ternary_backend_get_active();
    if (backend) {
        return backend->info.name;
    }
    return "None";
}

/**
 * Set active backend by name
 */
bool set_active_backend(const std::string& name) {
    const TernaryBackend* backend = ternary_backend_find(name.c_str());
    if (!backend) {
        return false;
    }
    ternary_backend_set_active(backend);
    return true;
}

/**
 * Get backend capabilities as string
 */
std::string get_capabilities_string(uint32_t capabilities) {
    char buffer[512];
    ternary_backend_capabilities_to_string(capabilities, buffer, sizeof(buffer));
    return std::string(buffer);
}

// ============================================================================
// Dispatch Operations (NumPy Interface)
// ============================================================================
//
// The 9 dispatch_* functions below (5 core ops + 4 fused ops) used to each
// hand-copy the same validate/allocate/call-function-pointer/throw skeleton,
// differing only in which ternary_dispatch_* function to call, the op name
// embedded in the error strings, and (fused ops only) a different failure
// message. Factored into dispatch_unary()/dispatch_binary() below --
// CLAUDE.md gap #6, Cluster B (see reports/2026-08-18/GAP6_DUPLICATION_SCOPE.md
// for the ~140 duplicated lines this replaces). Error text is preserved
// exactly, including the fused-ops' distinct failure message.

using UnaryDispatchFn  = bool (*)(uint8_t*, const uint8_t*, size_t);
using BinaryDispatchFn = bool (*)(uint8_t*, const uint8_t*, const uint8_t*, size_t);

static const char* const NO_ACTIVE_BACKEND_MSG =
    "no active backend (call init()/set_backend() first)";
static const char* const NO_FUSED_BACKEND_MSG =
    "no active backend, or active backend does not implement this fused op";

static py::array_t<uint8_t> dispatch_unary(
    const char* op_name, UnaryDispatchFn fn, const char* fail_msg,
    py::array_t<uint8_t> A
) {
    auto a = A.unchecked<1>();
    py::ssize_t n = A.size();

    py::array_t<uint8_t> result(n);
    auto r = result.mutable_unchecked<1>();

    if (!fn(r.mutable_data(0), a.data(0), static_cast<size_t>(n)))
        throw std::runtime_error(std::string(op_name) + ": " + fail_msg);

    return result;
}

static py::array_t<uint8_t> dispatch_binary(
    const char* op_name, BinaryDispatchFn fn, const char* fail_msg,
    py::array_t<uint8_t> A, py::array_t<uint8_t> B
) {
    auto a = A.unchecked<1>();
    auto b = B.unchecked<1>();
    py::ssize_t n = A.size();

    if (A.size() != B.size())
        throw std::invalid_argument(std::string(op_name) + ": array size mismatch");

    py::array_t<uint8_t> result(n);
    auto r = result.mutable_unchecked<1>();

    if (!fn(r.mutable_data(0), a.data(0), b.data(0), static_cast<size_t>(n)))
        throw std::runtime_error(std::string(op_name) + ": " + fail_msg);

    return result;
}

py::array_t<uint8_t> dispatch_tnot(py::array_t<uint8_t> A) {
    return dispatch_unary("tnot", ternary_dispatch_tnot, NO_ACTIVE_BACKEND_MSG, A);
}

py::array_t<uint8_t> dispatch_tadd(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return dispatch_binary("tadd", ternary_dispatch_tadd, NO_ACTIVE_BACKEND_MSG, A, B);
}

py::array_t<uint8_t> dispatch_tmul(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return dispatch_binary("tmul", ternary_dispatch_tmul, NO_ACTIVE_BACKEND_MSG, A, B);
}

py::array_t<uint8_t> dispatch_tmax(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return dispatch_binary("tmax", ternary_dispatch_tmax, NO_ACTIVE_BACKEND_MSG, A, B);
}

py::array_t<uint8_t> dispatch_tmin(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return dispatch_binary("tmin", ternary_dispatch_tmin, NO_ACTIVE_BACKEND_MSG, A, B);
}

// ============================================================================
// Fusion Operations Dispatch (Phase 4.1)
// ============================================================================

py::array_t<uint8_t> dispatch_fused_tnot_tadd(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return dispatch_binary("fused_tnot_tadd", ternary_dispatch_fused_tnot_tadd, NO_FUSED_BACKEND_MSG, A, B);
}

py::array_t<uint8_t> dispatch_fused_tnot_tmul(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return dispatch_binary("fused_tnot_tmul", ternary_dispatch_fused_tnot_tmul, NO_FUSED_BACKEND_MSG, A, B);
}

py::array_t<uint8_t> dispatch_fused_tnot_tmin(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return dispatch_binary("fused_tnot_tmin", ternary_dispatch_fused_tnot_tmin, NO_FUSED_BACKEND_MSG, A, B);
}

py::array_t<uint8_t> dispatch_fused_tnot_tmax(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return dispatch_binary("fused_tnot_tmax", ternary_dispatch_fused_tnot_tmax, NO_FUSED_BACKEND_MSG, A, B);
}

// ============================================================================
// Module Definition
// ============================================================================

PYBIND11_MODULE(ternary_backend, m) {
    m.doc() = "Ternary Engine Backend System (v1.2.0)\n\n"
              "Backend management and dispatch API for ternary operations.\n"
              "Provides access to multiple backend implementations:\n"
              "  - Scalar: Portable reference (always available)\n"
              "  - AVX2_v1: Baseline AVX2 (28-35 Gops/s)\n"
              "  - AVX2_v2: Optimized AVX2 with canonical indexing (35-45 Gops/s target)\n\n"
              "Usage:\n"
              "  import ternary_backend as tb\n"
              "  tb.init()\n"
              "  print(tb.list_backends())\n"
              "  tb.set_backend('AVX2_v2')\n"
              "  result = tb.tadd(a, b)";

    // Backend management
    m.def("init", &init_backends,
          "Initialize backend system (call once at startup)");

    m.def("shutdown", &shutdown_backends,
          "Shutdown backend system");

    m.def("list_backends", &list_backends,
          "List all available backends");

    m.def("get_active", &get_active_backend,
          "Get name of currently active backend");

    m.def("set_backend", &set_active_backend,
          py::arg("name"),
          "Set active backend by name");

    m.def("get_capabilities_string", &get_capabilities_string,
          py::arg("capabilities"),
          "Convert capability flags to human-readable string");

    // Dispatch operations
    m.def("tnot", &dispatch_tnot,
          py::arg("a"),
          "Ternary NOT via backend dispatch");

    m.def("tadd", &dispatch_tadd,
          py::arg("a"), py::arg("b"),
          "Ternary ADD via backend dispatch");

    m.def("tmul", &dispatch_tmul,
          py::arg("a"), py::arg("b"),
          "Ternary MUL via backend dispatch");

    m.def("tmax", &dispatch_tmax,
          py::arg("a"), py::arg("b"),
          "Ternary MAX via backend dispatch");

    m.def("tmin", &dispatch_tmin,
          py::arg("a"), py::arg("b"),
          "Ternary MIN via backend dispatch");

    // Fusion operations (Phase 4.1)
    m.def("fused_tnot_tadd", &dispatch_fused_tnot_tadd,
          py::arg("a"), py::arg("b"),
          "Fused tnot(tadd(a, b)) - eliminates intermediate array");

    m.def("fused_tnot_tmul", &dispatch_fused_tnot_tmul,
          py::arg("a"), py::arg("b"),
          "Fused tnot(tmul(a, b)) - eliminates intermediate array");

    m.def("fused_tnot_tmin", &dispatch_fused_tnot_tmin,
          py::arg("a"), py::arg("b"),
          "Fused tnot(tmin(a, b)) - eliminates intermediate array");

    m.def("fused_tnot_tmax", &dispatch_fused_tnot_tmax,
          py::arg("a"), py::arg("b"),
          "Fused tnot(tmax(a, b)) - eliminates intermediate array");

    // BackendInfo class for Python
    py::class_<BackendInfoPython>(m, "BackendInfo")
        .def_readonly("name", &BackendInfoPython::name)
        .def_readonly("description", &BackendInfoPython::description)
        .def_readonly("version", &BackendInfoPython::version)
        .def_readonly("capabilities", &BackendInfoPython::capabilities)
        .def_readonly("preferred_batch_size", &BackendInfoPython::preferred_batch_size)
        .def_readonly("is_available", &BackendInfoPython::is_available)
        .def_readonly("is_active", &BackendInfoPython::is_active)
        .def("__repr__", [](const BackendInfoPython& info) {
            std::ostringstream ss;
            ss << "<BackendInfo name='" << info.name << "' "
               << "version=" << info.version << " "
               << "active=" << (info.is_active ? "True" : "False") << ">";
            return ss.str();
        });

    // Version info
    m.attr("__version__") = "1.2.0";
}
