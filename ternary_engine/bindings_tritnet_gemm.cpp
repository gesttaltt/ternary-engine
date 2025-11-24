/**
 * @file ternary_tritnet_gemm_module.cpp
 * @brief Python bindings for TritNet Direct Ternary GEMM
 *
 * Exposes optimized ternary matrix multiplication to Python via pybind11.
 * Provides numpy-compatible interface for integration with TritNet training.
 *
 * @date 2025-11-23
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "tritnet_gemm.h"  // Resolved via include_dirs in build script
#include <stdexcept>

namespace py = pybind11;

/**
 * @brief Python wrapper for tritnet_gemm_f32
 *
 * Performs C = A @ B where B contains ternary weights {-1, 0, +1}
 * stored in Dense243 format (5 trits/byte).
 *
 * @param A         Input activations [M, K], numpy float32 array, row-major
 * @param B_packed  Ternary weights [K/5, N], numpy uint8 array, Dense243-packed
 * @param M         Number of rows in A and C
 * @param N         Number of columns in B and C
 * @param K         Number of columns in A / rows in B (must be multiple of 5)
 * @return          Output [M, N], numpy float32 array
 */
py::array_t<float> py_gemm(
    py::array_t<float> A,
    py::array_t<uint8_t> B_packed,
    int M, int N, int K
) {
    // Validate inputs
    if (K % 5 != 0) {
        throw std::invalid_argument("K must be multiple of 5 for Dense243 packing");
    }

    // Check array dimensions
    auto A_buf = A.request();
    auto B_buf = B_packed.request();

    if (A_buf.ndim != 2 || A_buf.shape[0] != M || A_buf.shape[1] != K) {
        throw std::invalid_argument("A must be [M, K] float32 array");
    }

    int K_packed = K / 5;
    if (B_buf.ndim != 2 || B_buf.shape[0] != K_packed || B_buf.shape[1] != N) {
        throw std::invalid_argument("B_packed must be [K/5, N] uint8 array");
    }

    // Check contiguity (C-order/row-major)
    if (!A.attr("flags").attr("c_contiguous").cast<bool>()) {
        throw std::invalid_argument("A must be C-contiguous (row-major)");
    }
    if (!B_packed.attr("flags").attr("c_contiguous").cast<bool>()) {
        throw std::invalid_argument("B_packed must be C-contiguous (row-major)");
    }

    // Allocate output array
    py::array_t<float> C({M, N});
    auto C_buf = C.request();

    // Call C function
    tritnet_gemm_f32(
        M, N, K,
        static_cast<const float*>(A_buf.ptr),
        static_cast<const uint8_t*>(B_buf.ptr),
        static_cast<float*>(C_buf.ptr)
    );

    return C;
}

/**
 * @brief Python wrapper for tritnet_gemm_f32_scaled
 *
 * Performs C = A @ B with per-column scaling: C[m,n] = scale[n] * sum_k(A[m,k] * B[k,n])
 * This matches BitNet's per-block quantization scheme.
 *
 * @param A         Input activations [M, K], numpy float32 array
 * @param B_packed  Ternary weights [K/5, N], numpy uint8 array
 * @param scales    Per-column scale factors [N], numpy float32 array
 * @param M         Number of rows in A and C
 * @param N         Number of columns in B and C
 * @param K         Number of columns in A / rows in B (must be multiple of 5)
 * @return          Output [M, N], numpy float32 array
 */
py::array_t<float> py_gemm_scaled(
    py::array_t<float> A,
    py::array_t<uint8_t> B_packed,
    py::array_t<float> scales,
    int M, int N, int K
) {
    // Validate inputs
    if (K % 5 != 0) {
        throw std::invalid_argument("K must be multiple of 5 for Dense243 packing");
    }

    // Check array dimensions
    auto A_buf = A.request();
    auto B_buf = B_packed.request();
    auto scales_buf = scales.request();

    if (A_buf.ndim != 2 || A_buf.shape[0] != M || A_buf.shape[1] != K) {
        throw std::invalid_argument("A must be [M, K] float32 array");
    }

    int K_packed = K / 5;
    if (B_buf.ndim != 2 || B_buf.shape[0] != K_packed || B_buf.shape[1] != N) {
        throw std::invalid_argument("B_packed must be [K/5, N] uint8 array");
    }

    if (scales_buf.ndim != 1 || scales_buf.shape[0] != N) {
        throw std::invalid_argument("scales must be [N] float32 array");
    }

    // Check contiguity
    if (!A.attr("flags").attr("c_contiguous").cast<bool>()) {
        throw std::invalid_argument("A must be C-contiguous");
    }
    if (!B_packed.attr("flags").attr("c_contiguous").cast<bool>()) {
        throw std::invalid_argument("B_packed must be C-contiguous");
    }
    if (!scales.attr("flags").attr("c_contiguous").cast<bool>()) {
        throw std::invalid_argument("scales must be C-contiguous");
    }

    // Allocate output array
    py::array_t<float> C({M, N});
    auto C_buf = C.request();

    // Call C function
    tritnet_gemm_f32_scaled(
        M, N, K,
        static_cast<const float*>(A_buf.ptr),
        static_cast<const uint8_t*>(B_buf.ptr),
        static_cast<const float*>(scales_buf.ptr),
        static_cast<float*>(C_buf.ptr)
    );

    return C;
}

/**
 * @brief Python wrapper for convert_bitnet_to_dense243
 *
 * Converts BitNet 2-bit weights to Dense243 format.
 * One-time conversion during model loading.
 *
 * BitNet encoding: -1→00, 0→01, +1→10 (4 trits/byte)
 * Dense243 encoding: 5 trits/byte (20% memory savings)
 *
 * @param bitnet_weights  Input [K/4, N], numpy uint8 array (BitNet format)
 * @param K               Number of rows (must be multiple of 20 for efficiency)
 * @param N               Number of columns
 * @return                Output [K/5, N], numpy uint8 array (Dense243 format)
 */
py::array_t<uint8_t> py_convert_from_bitnet(
    py::array_t<uint8_t> bitnet_weights,
    int K, int N
) {
    // Validate K is multiple of 20 for efficiency (LCM of 4 and 5)
    if (K % 20 != 0) {
        // Warning: K should be multiple of 20, but we can still proceed
        py::print("[Warning] K should be multiple of 20 for optimal conversion");
    }

    // Check array dimensions
    auto bitnet_buf = bitnet_weights.request();
    int K_bitnet = (K + 3) / 4;  // Round up

    if (bitnet_buf.ndim != 2 || bitnet_buf.shape[0] != K_bitnet || bitnet_buf.shape[1] != N) {
        throw std::invalid_argument("bitnet_weights must be [K/4, N] uint8 array");
    }

    if (!bitnet_weights.attr("flags").attr("c_contiguous").cast<bool>()) {
        throw std::invalid_argument("bitnet_weights must be C-contiguous");
    }

    // Allocate output array
    int K_dense243 = (K + 4) / 5;  // Round up
    py::array_t<uint8_t> dense243_out({K_dense243, N});
    auto dense243_buf = dense243_out.request();

    // Call C function
    convert_bitnet_to_dense243(
        static_cast<const uint8_t*>(bitnet_buf.ptr),
        static_cast<uint8_t*>(dense243_buf.ptr),
        K, N
    );

    return dense243_out;
}

/**
 * @brief Set number of OpenMP threads for parallel GEMM
 */
void py_set_num_threads(int n_threads) {
    if (n_threads < 0) {
        throw std::invalid_argument("n_threads must be >= 0 (0 = auto-detect)");
    }
    tritnet_set_num_threads(n_threads);
}

/**
 * @brief Get optimal cache tile size
 */
int py_get_tile_size(int cache_level) {
    if (cache_level < 1 || cache_level > 3) {
        throw std::invalid_argument("cache_level must be 1 (L1), 2 (L2), or 3 (L3)");
    }
    return tritnet_get_optimal_tile_size(cache_level);
}

/**
 * @brief Benchmark GEMM performance
 *
 * @param M         Matrix dimension
 * @param N         Matrix dimension
 * @param K         Matrix dimension
 * @param num_runs  Number of iterations for averaging
 * @return          Average time per run in milliseconds
 */
double py_benchmark(int M, int N, int K, int num_runs = 10) {
    if (K % 5 != 0) {
        throw std::invalid_argument("K must be multiple of 5");
    }
    if (num_runs < 1) {
        throw std::invalid_argument("num_runs must be >= 1");
    }
    return tritnet_benchmark_gemm(M, N, K, num_runs);
}

/**
 * @brief Validate GEMM correctness
 *
 * @param M  Matrix dimension
 * @param N  Matrix dimension
 * @param K  Matrix dimension
 * @return   Maximum absolute error (should be < 1e-6 for FP32)
 */
float py_validate(int M, int N, int K) {
    if (K % 5 != 0) {
        throw std::invalid_argument("K must be multiple of 5");
    }
    return tritnet_validate_gemm(M, N, K);
}

/**
 * @brief pybind11 module definition
 */
PYBIND11_MODULE(ternary_tritnet_gemm, m) {
    m.doc() = R"doc(
        TritNet Direct Ternary GEMM - Python Interface

        Optimized matrix multiplication for ternary weights {-1, 0, +1}
        stored in Dense243 format (5 trits/byte).

        Performance target: 2-3× faster than BitNet TL2 kernels

        Core Functions:
            gemm(A, B_packed, M, N, K) -> C
                Compute C = A @ B with ternary weights

            gemm_scaled(A, B_packed, scales, M, N, K) -> C
                Compute C with per-column scaling

            convert_from_bitnet(bitnet_weights, K, N) -> dense243_weights
                Convert BitNet 2-bit format to Dense243

        Utility Functions:
            set_num_threads(n)
                Control OpenMP parallelization

            get_tile_size(cache_level)
                Get optimal tile size for cache

            benchmark(M, N, K, num_runs=10)
                Measure GEMM performance

            validate(M, N, K)
                Validate correctness vs reference

        Example Usage:
            import numpy as np
            import ternary_tritnet_gemm as gemm

            # Activations (FP32)
            A = np.random.randn(1024, 4096).astype(np.float32)

            # Ternary weights (Dense243-packed)
            B_packed = np.random.randint(0, 243, (4096//5, 2048), dtype=np.uint8)

            # Matrix multiply
            C = gemm.gemm(A, B_packed, 1024, 2048, 4096)

            # Benchmark performance
            time_ms = gemm.benchmark(1024, 2048, 4096, num_runs=10)
            print(f"GEMM: {time_ms:.2f} ms")
    )doc";

    // Core GEMM operations
    m.def("gemm", &py_gemm,
          "Ternary GEMM: C = A @ B with Dense243-packed weights",
          py::arg("A"), py::arg("B_packed"), py::arg("M"), py::arg("N"), py::arg("K"));

    m.def("gemm_scaled", &py_gemm_scaled,
          "Ternary GEMM with per-column scaling",
          py::arg("A"), py::arg("B_packed"), py::arg("scales"),
          py::arg("M"), py::arg("N"), py::arg("K"));

    // Weight conversion
    m.def("convert_from_bitnet", &py_convert_from_bitnet,
          "Convert BitNet 2-bit weights to Dense243 format",
          py::arg("bitnet_weights"), py::arg("K"), py::arg("N"));

    // Configuration
    m.def("set_num_threads", &py_set_num_threads,
          "Set number of OpenMP threads (0 = auto-detect)",
          py::arg("n_threads"));

    m.def("get_tile_size", &py_get_tile_size,
          "Get optimal cache tile size (1=L1, 2=L2, 3=L3)",
          py::arg("cache_level"));

    // Benchmarking and validation
    m.def("benchmark", &py_benchmark,
          "Benchmark GEMM performance (returns avg time in ms)",
          py::arg("M"), py::arg("N"), py::arg("K"), py::arg("num_runs") = 10);

    m.def("validate", &py_validate,
          "Validate GEMM correctness (returns max absolute error)",
          py::arg("M"), py::arg("N"), py::arg("K"));
}
