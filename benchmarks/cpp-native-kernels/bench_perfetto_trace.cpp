/**
 * @file bench_perfetto_trace.cpp
 * @brief First real Perfetto trace of a ternary hot path (pybind11-free)
 *
 * Closes part of Critical Gap #10 (profiler integration): the call sites
 * in bindings_core_ops.cpp (TERNARY_PROFILE_TASK_BEGIN/END around the
 * OpenMP/serial-SIMD/scalar-tail sections) were genuinely wired but no
 * build script ever defined TERNARY_ENABLE_PERFETTO, VTune, or NVTX, so
 * only the no-op stub had ever been built or verified. Of the three,
 * only Perfetto needs no proprietary tool or GPU to build AND verify
 * against -- see src/core/profiling/ternary_profiler.h's Perfetto
 * section and third_party/perfetto/README.md.
 *
 * This program replicates bindings_core_ops.cpp's exact profiling
 * pattern (same domain, same 3 task names, same OMP/SIMD/tail
 * structure) on a real tadd workload, using the actual AVX2 kernel
 * (simd_avx2_32trit_ops.h), so the resulting trace reflects genuine
 * hot-path timing, not a synthetic sleep() loop.
 *
 * COMPILATION (from benchmarks/cpp-native-kernels/ directory):
 *   g++ -O3 -march=haswell -mavx2 -mfma -fopenmp -std=c++17 \
 *       -DTERNARY_ENABLE_PERFETTO \
 *       -I../../src/core -I../.. \
 *       bench_perfetto_trace.cpp \
 *       ../../src/core/profiling/ternary_profiler_perfetto.cc \
 *       ../../third_party/perfetto/perfetto.cc \
 *       -o bench_perfetto_trace -lpthread
 *
 * USAGE: ./bench_perfetto_trace [output.perfetto-trace]
 * OUTPUT: a real Perfetto trace file, openable at https://ui.perfetto.dev
 *         or queryable with trace_processor_shell.
 */

#include "profiling/ternary_profiler.h"
#include "simd/simd_avx2_32trit_ops.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// Exact same domain/task-name pattern as bindings_core_ops.cpp's real
// call sites, so this trace is representative of the real hot path.
TERNARY_PROFILE_DOMAIN(g_ternary_domain, "TernaryCore");
TERNARY_PROFILE_TASK_NAME(g_task_omp, "OpenMP_Parallel");
TERNARY_PROFILE_TASK_NAME(g_task_simd, "Serial_SIMD");
TERNARY_PROFILE_TASK_NAME(g_task_tail, "Scalar_Tail");

static void run_tadd_traced(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t n) {
    const size_t simd_width = 32;
    const size_t omp_threshold = 262144;  // matches this project's OMP_THRESHOLD

    size_t simd_count = n / simd_width;
    size_t simd_end = simd_count * simd_width;

    if (n >= omp_threshold) {
        TERNARY_PROFILE_TASK_BEGIN(g_ternary_domain, g_task_omp);
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < (long long)simd_count; i++) {
            size_t off = (size_t)i * simd_width;
            __m256i va = _mm256_loadu_si256((const __m256i*)(a + off));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(b + off));
            __m256i vr = tadd_simd<true>(va, vb);
            _mm256_storeu_si256((__m256i*)(out + off), vr);
        }
        TERNARY_PROFILE_TASK_END(g_ternary_domain);
    } else {
        TERNARY_PROFILE_TASK_BEGIN(g_ternary_domain, g_task_simd);
        for (size_t off = 0; off < simd_end; off += simd_width) {
            __m256i va = _mm256_loadu_si256((const __m256i*)(a + off));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(b + off));
            __m256i vr = tadd_simd<true>(va, vb);
            _mm256_storeu_si256((__m256i*)(out + off), vr);
        }
        TERNARY_PROFILE_TASK_END(g_ternary_domain);
    }

    if (simd_end < n) {
        TERNARY_PROFILE_TASK_BEGIN(g_ternary_domain, g_task_tail);
        for (size_t i = simd_end; i < n; i++) {
            // Scalar balanced-ternary saturated add, matching tadd's
            // semantics (0b00=-1, 0b01=0, 0b10=+1): direct LUT-free
            // version is fine here, this is intentionally the "tail"
            // path (small element count) being profiled, not a hot loop.
            int av = (int)a[i] - 1, bv = (int)b[i] - 1;
            int sum = av + bv;
            if (sum > 1) sum = 1;
            if (sum < -1) sum = -1;
            out[i] = (uint8_t)(sum + 1);
        }
        TERNARY_PROFILE_TASK_END(g_ternary_domain);
    }
}

int main(int argc, char** argv) {
    const char* trace_path = (argc > 1) ? argv[1] : "ternary_hotpath.perfetto-trace";

    if (!ternary_profiler_perfetto_start(trace_path)) {
        fprintf(stderr, "Failed to start Perfetto tracing session (path=%s)\n", trace_path);
        return 1;
    }

    // A few representative sizes: below and above the OMP threshold, so
    // the trace contains real examples of both the "Serial_SIMD" and
    // "OpenMP_Parallel" task types, plus "Scalar_Tail" from sizes not a
    // multiple of 32.
    size_t sizes[] = {1000, 50000, 500000, 2000000};
    for (size_t n : sizes) {
        std::vector<uint8_t> a(n), b(n), out(n);
        for (size_t i = 0; i < n; i++) {
            a[i] = (uint8_t)(rand() % 3);
            b[i] = (uint8_t)(rand() % 3);
        }
        // Repeat each size a few times so the trace has enough events to
        // be a meaningful timeline, not a single instantaneous blip.
        for (int rep = 0; rep < 5; rep++) {
            run_tadd_traced(a.data(), b.data(), out.data(), n);
        }
        printf("Traced tadd at n=%zu (5 reps)\n", n);
    }

    ternary_profiler_perfetto_stop();
    printf("Trace written to %s\n", trace_path);
    return 0;
}
