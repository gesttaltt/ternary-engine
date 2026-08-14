/**
 * @file bench_tritnet_inference.cpp
 * @brief TritNet-vs-LUT throughput benchmark (Phase 3 decisive experiment)
 *
 * CLAUDE.md's TritNet Phase 3 explicitly calls this out as the experiment that
 * decides whether TritNet beats LUTs in practice, noting the LUT does ~20K
 * fewer MACs per 5-trit op than the TritNet forward pass. This benchmark
 * measures both paths doing the SAME work -- computing one binary/unary op
 * across a 5-trit chunk -- and reports real throughput, not an estimate:
 *   - LUT path: 5x scalar calls into ternary_algebra.h's tadd/tmul/tmin/tmax/tnot
 *   - TritNet path: 1x forward pass through models/tritnet/inference/tritnet_inference.h
 *
 * COMPILATION (from benchmarks/cpp-native-kernels/ directory):
 *   g++ -O3 -march=native -std=c++17 -I../../ bench_tritnet_inference.cpp -o bench_tritnet_inference
 *   clang++ -O3 -march=native -std=c++17 -I../../ bench_tritnet_inference.cpp -o bench_tritnet_inference
 *
 *   # Windows (MSVC):
 *   cl /O2 /std:c++17 /EHsc /I..\..\ bench_tritnet_inference.cpp
 *
 * TARGET: models/tritnet/inference/ - TritNet C++ inference engine (Phase 3, naive/scalar)
 */

#include "models/tritnet/inference/tritnet_inference.h"

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
    double tritnet_mops;
};

static double best_seconds_of(const std::vector<double>& samples) {
    double best = samples[0];
    for (double s : samples) if (s < best) best = s;
    return best;
}

static Result bench_unary(trit (*lut_fn)(trit),
                           void (*net_fn)(const trit[5], trit[5])) {
    SplitMix64 rng(42);
    std::vector<trit> chunks(static_cast<size_t>(N_CHUNKS) * 5);
    for (int c = 0; c < N_CHUNKS; ++c) random_chunk(rng, &chunks[c * 5]);

    volatile int sink = 0;  // prevent the optimizer from eliding the loops

    std::vector<double> lut_times, net_times;
    trit out[5];

    for (int rep = 0; rep < N_REPEATS; ++rep) {
        auto t0 = high_resolution_clock::now();
        for (int c = 0; c < N_CHUNKS; ++c) {
            const trit* a = &chunks[c * 5];
            for (int i = 0; i < 5; ++i) out[i] = lut_fn(a[i]);
            sink += out[0];
        }
        auto t1 = high_resolution_clock::now();
        lut_times.push_back(duration<double>(t1 - t0).count());
    }

    for (int rep = 0; rep < N_REPEATS; ++rep) {
        auto t0 = high_resolution_clock::now();
        for (int c = 0; c < N_CHUNKS; ++c) {
            net_fn(&chunks[c * 5], out);
            sink += out[0];
        }
        auto t1 = high_resolution_clock::now();
        net_times.push_back(duration<double>(t1 - t0).count());
    }

    (void)sink;
    double lut_s = best_seconds_of(lut_times);
    double net_s = best_seconds_of(net_times);
    // "ops" = 5-trit-chunk operations (1 op = negating/combining all 5 trits)
    return {
        (N_CHUNKS / lut_s) / 1e6,
        (N_CHUNKS / net_s) / 1e6,
    };
}

static Result bench_binary(trit (*lut_fn)(trit, trit),
                            void (*net_fn)(const trit[5], const trit[5], trit[5])) {
    SplitMix64 rng(42);
    std::vector<trit> chunks_a(static_cast<size_t>(N_CHUNKS) * 5);
    std::vector<trit> chunks_b(static_cast<size_t>(N_CHUNKS) * 5);
    for (int c = 0; c < N_CHUNKS; ++c) {
        random_chunk(rng, &chunks_a[c * 5]);
        random_chunk(rng, &chunks_b[c * 5]);
    }

    volatile int sink = 0;

    std::vector<double> lut_times, net_times;
    trit out[5];

    for (int rep = 0; rep < N_REPEATS; ++rep) {
        auto t0 = high_resolution_clock::now();
        for (int c = 0; c < N_CHUNKS; ++c) {
            const trit* a = &chunks_a[c * 5];
            const trit* b = &chunks_b[c * 5];
            for (int i = 0; i < 5; ++i) out[i] = lut_fn(a[i], b[i]);
            sink += out[0];
        }
        auto t1 = high_resolution_clock::now();
        lut_times.push_back(duration<double>(t1 - t0).count());
    }

    for (int rep = 0; rep < N_REPEATS; ++rep) {
        auto t0 = high_resolution_clock::now();
        for (int c = 0; c < N_CHUNKS; ++c) {
            net_fn(&chunks_a[c * 5], &chunks_b[c * 5], out);
            sink += out[0];
        }
        auto t1 = high_resolution_clock::now();
        net_times.push_back(duration<double>(t1 - t0).count());
    }

    (void)sink;
    double lut_s = best_seconds_of(lut_times);
    double net_s = best_seconds_of(net_times);
    return {
        (N_CHUNKS / lut_s) / 1e6,
        (N_CHUNKS / net_s) / 1e6,
    };
}

int main() {
    printf("======================================================================\n");
    printf("TritNet vs LUT Throughput Benchmark (Phase 3 decisive experiment)\n");
    printf("======================================================================\n");
    printf("Chunks per op: %d, repeats: %d (best-of reported)\n", N_CHUNKS, N_REPEATS);
    printf("Unit: Mops/s = millions of 5-trit-chunk operations/sec\n\n");

    printf("%-6s %14s %14s %10s\n", "Op", "LUT (Mops/s)", "TritNet (Mops/s)", "LUT/TritNet");
    printf("----------------------------------------------------------------------\n");

    Result r;

    r = bench_unary(tnot, tritnet::tritnet_tnot);
    printf("%-6s %14.2f %14.4f %9.1fx\n", "tnot", r.lut_mops, r.tritnet_mops, r.lut_mops / r.tritnet_mops);

    r = bench_binary(tadd, tritnet::tritnet_tadd);
    printf("%-6s %14.2f %14.4f %9.1fx\n", "tadd", r.lut_mops, r.tritnet_mops, r.lut_mops / r.tritnet_mops);

    r = bench_binary(tmul, tritnet::tritnet_tmul);
    printf("%-6s %14.2f %14.4f %9.1fx\n", "tmul", r.lut_mops, r.tritnet_mops, r.lut_mops / r.tritnet_mops);

    r = bench_binary(tmin, tritnet::tritnet_tmin);
    printf("%-6s %14.2f %14.4f %9.1fx\n", "tmin", r.lut_mops, r.tritnet_mops, r.lut_mops / r.tritnet_mops);

    r = bench_binary(tmax, tritnet::tritnet_tmax);
    printf("%-6s %14.2f %14.4f %9.1fx\n", "tmax", r.lut_mops, r.tritnet_mops, r.lut_mops / r.tritnet_mops);

    printf("\nNote: this is the naive/scalar TritNet forward pass (no SIMD). AVX2\n");
    printf("vectorization of the matmuls is a later Phase 3 step, not yet done.\n");

    return 0;
}
