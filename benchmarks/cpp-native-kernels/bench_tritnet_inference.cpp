/**
 * @file bench_tritnet_inference.cpp
 * @brief TritNet-vs-LUT throughput benchmark (Phase 3 decisive experiment)
 *
 * CLAUDE.md's TritNet Phase 3 explicitly calls this out as the experiment that
 * decides whether TritNet beats LUTs in practice, noting the LUT does ~20K
 * fewer MACs per 5-trit op than the TritNet forward pass. This benchmark
 * measures all three paths doing the SAME work -- computing one binary/unary
 * op across a 5-trit chunk -- and reports real throughput, not an estimate:
 *   - LUT path:    5x scalar calls into ternary_algebra.h's tadd/tmul/tmin/tmax/tnot
 *   - Scalar path: 1x naive forward pass, models/tritnet/inference/tritnet_inference.h
 *   - AVX2 path:   1x vectorized forward pass, tritnet_inference_avx2.h (only
 *                  built/run when compiled with -mavx2; see COMPILATION below)
 *
 * COMPILATION (from benchmarks/cpp-native-kernels/ directory):
 *   g++ -O3 -march=native -mavx2 -mfma -std=c++17 -I../../ bench_tritnet_inference.cpp -o bench_tritnet_inference
 *   clang++ -O3 -march=native -mavx2 -mfma -std=c++17 -I../../ bench_tritnet_inference.cpp -o bench_tritnet_inference
 *
 *   Omit -mavx2 -mfma to benchmark LUT vs scalar-TritNet only (AVX2 column
 *   is skipped, not zero-filled, so it can't be misread as "AVX2 measured
 *   as 0 Mops/s").
 *
 *   # Windows (MSVC):
 *   cl /O2 /arch:AVX2 /std:c++17 /EHsc /I..\..\ bench_tritnet_inference.cpp
 *
 * TARGET: models/tritnet/inference/ - TritNet C++ inference engine (Phase 3)
 */

#include "models/tritnet/inference/tritnet_inference.h"
#ifdef __AVX2__
#include "models/tritnet/inference/tritnet_inference_avx2.h"
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace std::chrono;

// ---------------------------------------------------------------------------
// Fixed-seed PRNG (not rand()/srand() -- reproducible across platforms/libc)
// ---------------------------------------------------------------------------

struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t seed) : state(seed) {}
    uint64_t next() {
        uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
};

static void random_chunk(SplitMix64& rng, trit out[5]) {
    for (int i = 0; i < 5; ++i) {
        int v = static_cast<int>(rng.next() % 3) - 1;  // {-1,0,1}
        out[i] = int_to_trit(v);
    }
}

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

constexpr int N_CHUNKS = 200000;
constexpr int N_REPEATS = 5;

struct Result {
    double lut_mops;
    double scalar_mops;
    double avx2_mops;  // -1 if not measured (compiled without -mavx2)
};

static double best_seconds_of(const std::vector<double>& samples) {
    double best = samples[0];
    for (double s : samples) if (s < best) best = s;
    return best;
}

// Times `fn` (called once per chunk, args supplied by `call`) over N_REPEATS
// passes and returns the best (lowest) wall time in seconds.
template<typename Fn>
static double time_best(int n_chunks, Fn&& call) {
    std::vector<double> times;
    times.reserve(N_REPEATS);
    for (int rep = 0; rep < N_REPEATS; ++rep) {
        auto t0 = high_resolution_clock::now();
        for (int c = 0; c < n_chunks; ++c) call(c);
        auto t1 = high_resolution_clock::now();
        times.push_back(duration<double>(t1 - t0).count());
    }
    return best_seconds_of(times);
}

static Result bench_unary(trit (*lut_fn)(trit),
                           void (*scalar_fn)(const trit[5], trit[5])
#ifdef __AVX2__
                           , void (*avx2_fn)(const trit[5], trit[5])
#endif
) {
    SplitMix64 rng(42);
    std::vector<trit> chunks(static_cast<size_t>(N_CHUNKS) * 5);
    for (int c = 0; c < N_CHUNKS; ++c) random_chunk(rng, &chunks[c * 5]);

    volatile int sink = 0;  // prevent the optimizer from eliding the loops
    trit out[5];

    double lut_s = time_best(N_CHUNKS, [&](int c) {
        const trit* a = &chunks[c * 5];
        for (int i = 0; i < 5; ++i) out[i] = lut_fn(a[i]);
        sink += out[0];
    });

    double scalar_s = time_best(N_CHUNKS, [&](int c) {
        scalar_fn(&chunks[c * 5], out);
        sink += out[0];
    });

    Result r{(N_CHUNKS / lut_s) / 1e6, (N_CHUNKS / scalar_s) / 1e6, -1.0};

#ifdef __AVX2__
    double avx2_s = time_best(N_CHUNKS, [&](int c) {
        avx2_fn(&chunks[c * 5], out);
        sink += out[0];
    });
    r.avx2_mops = (N_CHUNKS / avx2_s) / 1e6;
#endif

    (void)sink;
    return r;
}

static Result bench_binary(trit (*lut_fn)(trit, trit),
                            void (*scalar_fn)(const trit[5], const trit[5], trit[5])
#ifdef __AVX2__
                            , void (*avx2_fn)(const trit[5], const trit[5], trit[5])
#endif
) {
    SplitMix64 rng(42);
    std::vector<trit> chunks_a(static_cast<size_t>(N_CHUNKS) * 5);
    std::vector<trit> chunks_b(static_cast<size_t>(N_CHUNKS) * 5);
    for (int c = 0; c < N_CHUNKS; ++c) {
        random_chunk(rng, &chunks_a[c * 5]);
        random_chunk(rng, &chunks_b[c * 5]);
    }

    volatile int sink = 0;
    trit out[5];

    double lut_s = time_best(N_CHUNKS, [&](int c) {
        const trit* a = &chunks_a[c * 5];
        const trit* b = &chunks_b[c * 5];
        for (int i = 0; i < 5; ++i) out[i] = lut_fn(a[i], b[i]);
        sink += out[0];
    });

    double scalar_s = time_best(N_CHUNKS, [&](int c) {
        scalar_fn(&chunks_a[c * 5], &chunks_b[c * 5], out);
        sink += out[0];
    });

    Result r{(N_CHUNKS / lut_s) / 1e6, (N_CHUNKS / scalar_s) / 1e6, -1.0};

#ifdef __AVX2__
    double avx2_s = time_best(N_CHUNKS, [&](int c) {
        avx2_fn(&chunks_a[c * 5], &chunks_b[c * 5], out);
        sink += out[0];
    });
    r.avx2_mops = (N_CHUNKS / avx2_s) / 1e6;
#endif

    (void)sink;
    return r;
}

static void print_row(const char* name, const Result& r) {
#ifdef __AVX2__
    printf("%-6s %14.2f %14.4f %14.4f %12.1fx %12.1fx\n",
           name, r.lut_mops, r.scalar_mops, r.avx2_mops,
           r.lut_mops / r.scalar_mops, r.lut_mops / r.avx2_mops);
#else
    printf("%-6s %14.2f %14.4f %12.1fx\n",
           name, r.lut_mops, r.scalar_mops, r.lut_mops / r.scalar_mops);
#endif
}

int main() {
    printf("======================================================================\n");
    printf("TritNet vs LUT Throughput Benchmark (Phase 3 decisive experiment)\n");
    printf("======================================================================\n");
    printf("Chunks per op: %d, repeats: %d (best-of reported)\n", N_CHUNKS, N_REPEATS);
    printf("Unit: Mops/s = millions of 5-trit-chunk operations/sec\n\n");

#ifdef __AVX2__
    printf("%-6s %14s %14s %14s %13s %13s\n",
           "Op", "LUT", "Scalar", "AVX2", "LUT/Scalar", "LUT/AVX2");
    printf("--------------------------------------------------------------------------------------\n");
#else
    printf("%-6s %14s %14s %13s\n", "Op", "LUT", "Scalar", "LUT/Scalar");
    printf("---------------------------------------------------------\n");
    printf("(compiled without -mavx2 -- AVX2 column skipped)\n");
#endif

#ifdef __AVX2__
    print_row("tnot", bench_unary(tnot, tritnet::tritnet_tnot, tritnet::avx2::tritnet_tnot));
    print_row("tadd", bench_binary(tadd, tritnet::tritnet_tadd, tritnet::avx2::tritnet_tadd));
    print_row("tmul", bench_binary(tmul, tritnet::tritnet_tmul, tritnet::avx2::tritnet_tmul));
    print_row("tmin", bench_binary(tmin, tritnet::tritnet_tmin, tritnet::avx2::tritnet_tmin));
    print_row("tmax", bench_binary(tmax, tritnet::tritnet_tmax, tritnet::avx2::tritnet_tmax));
#else
    print_row("tnot", bench_unary(tnot, tritnet::tritnet_tnot));
    print_row("tadd", bench_binary(tadd, tritnet::tritnet_tadd));
    print_row("tmul", bench_binary(tmul, tritnet::tritnet_tmul));
    print_row("tmin", bench_binary(tmin, tritnet::tritnet_tmin));
    print_row("tmax", bench_binary(tmax, tritnet::tritnet_tmax));
#endif

    printf("\nScalar = naive forward pass, no SIMD. AVX2 = 8-wide vectorized forward\n");
    printf("pass (tritnet_inference_avx2.h), correctness-verified bit-identical to\n");
    printf("scalar over the full input space (tests/cpp/test_tritnet_inference.cpp).\n");

    return 0;
}
