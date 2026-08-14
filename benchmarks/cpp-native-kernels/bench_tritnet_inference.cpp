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

// Times two functions INTERLEAVED (A, B, A, B, ... one rep of each at a
// time) rather than back-to-back-in-two-blocks, and returns each side's best
// wall time in seconds. Found 2026-08-14: comparing a number measured early
// in the run (the main table's baseline.avx2_mops) against a number measured
// much later (after already running a full 5-op x 3-path sweep) is itself an
// unfair comparison if the CPU's clock state drifts between them -- which it
// did, reproducibly, on this laptop chip. Interleaving means both sides see
// the same drift pattern rep-by-rep, so "best of 5" for each is comparing
// apples to apples regardless of any slow thermal/turbo decay over the run.
template<typename FnA, typename FnB>
static void time_best_interleaved(int n_chunks, FnA&& call_a, FnB&& call_b,
                                   double& seconds_a, double& seconds_b) {
    std::vector<double> times_a, times_b;
    times_a.reserve(N_REPEATS);
    times_b.reserve(N_REPEATS);
    for (int rep = 0; rep < N_REPEATS; ++rep) {
        auto t0 = high_resolution_clock::now();
        for (int c = 0; c < n_chunks; ++c) call_a(c);
        auto t1 = high_resolution_clock::now();
        times_a.push_back(duration<double>(t1 - t0).count());

        auto t2 = high_resolution_clock::now();
        for (int c = 0; c < n_chunks; ++c) call_b(c);
        auto t3 = high_resolution_clock::now();
        times_b.push_back(duration<double>(t3 - t2).count());
    }
    seconds_a = best_seconds_of(times_a);
    seconds_b = best_seconds_of(times_b);
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

#ifdef __AVX2__
// ---------------------------------------------------------------------------
// Amortized-weight-conversion experiment
// ---------------------------------------------------------------------------
// tritnet_inference_avx2.h's layer_avx2() re-runs the int8->float weight
// conversion (_mm256_cvtepi8_epi32 -> _mm256_cvtepi32_ps) on every single
// forward-pass call, even though the SAME weight bytes are converted every
// time -- weights don't change between calls. That's a fixed cost being
// re-paid per-call instead of hoisted out and amortized across a batch,
// which is the same class of measurement distortion as this project's
// retired "8,234x vs pure Python" claim (see README.md's historical note):
// paying a cost repeatedly that a fair/realistic comparison would pay once.
// This block isolates exactly how much of the measured AVX2 throughput is
// attributable to that redundant reconversion, by pre-converting weights
// ONCE outside the timed loop and reusing them across all N_CHUNKS calls.
// Deliberately kept local to this benchmark (not promoted into
// tritnet_inference_avx2.h) -- it answers a measurement question, it isn't
// a verified/shipped inference path.

namespace preconv {

template<int IN, int HID>
void convert_weights(const int8_t W[IN][HID], float out[IN][HID]) {
    for (int i = 0; i < IN; ++i) {
        for (int j = 0; j < HID; j += 8) {
            __m128i w8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(W[i] + j));
            __m256i w32 = _mm256_cvtepi8_epi32(w8);
            __m256 wf = _mm256_cvtepi32_ps(w32);
            _mm256_storeu_ps(&out[i][j], wf);
        }
    }
}

template<int IN, int HID, bool ApplyReLU>
inline void layer_preconv(const float x[IN], const float W[IN][HID],
                           const float B[HID], float out[HID]) {
    for (int j = 0; j < HID; j += 8) _mm256_storeu_ps(&out[j], _mm256_loadu_ps(&B[j]));
    for (int i = 0; i < IN; ++i) {
        const __m256 xi = _mm256_set1_ps(x[i]);
        const float* row = W[i];
        for (int j = 0; j < HID; j += 8) {
            __m256 wf = _mm256_loadu_ps(&row[j]);
            __m256 acc = _mm256_loadu_ps(&out[j]);
            acc = _mm256_fmadd_ps(xi, wf, acc);
            _mm256_storeu_ps(&out[j], acc);
        }
    }
    if (ApplyReLU) {
        const __m256 zero = _mm256_setzero_ps();
        for (int j = 0; j < HID; j += 8) {
            __m256 v = _mm256_max_ps(_mm256_loadu_ps(&out[j]), zero);
            _mm256_storeu_ps(&out[j], v);
        }
    }
}

// tadd's shapes: IN1=10, HID=128, OUT_PADDED=16 (matches tmul/tmin/tmax --
// all 4 binary ops share this architecture, so tadd stands in for all of them).
struct TaddWeightsF32 {
    float W1[10][128];
    float W2[128][128];
    float W3[128][16];
};

static void prepare_tadd(TaddWeightsF32& w) {
    using namespace tritnet::weights::tadd_weights;
    convert_weights<10, 128>(W1, w.W1);
    convert_weights<128, 128>(W2, w.W2);
    convert_weights<128, 16>(W3, w.W3);
}

static void forward_tadd_preconv(const TaddWeightsF32& w, const float x[10], int8_t out_trits[5]) {
    using namespace tritnet::weights::tadd_weights;
    float h1[128];
    layer_preconv<10, 128, true>(x, w.W1, B1, h1);
    float h2[128];
    layer_preconv<128, 128, true>(h1, w.W2, B2, h2);
    float logits[16];
    layer_preconv<128, 16, false>(h2, w.W3, B3, logits);
    for (int k = 0; k < 5; ++k) {
        int best = 0;
        float best_val = logits[k * 3];
        for (int c = 1; c < 3; ++c) {
            if (logits[k * 3 + c] > best_val) { best_val = logits[k * 3 + c]; best = c; }
        }
        out_trits[k] = static_cast<int8_t>(best - 1);
    }
}

// tmul reuses TaddWeightsF32's shape (same 10/128/16 architecture as all 4
// binary ops) -- checking it separately from tadd because tmul's checkpoint
// is the weakest of the four (99.5% vs tadd's 100%), so it's worth confirming
// the amortization ratio doesn't depend on how saturated/clean the learned
// weights are.
static void prepare_tmul(TaddWeightsF32& w) {
    using namespace tritnet::weights::tmul_weights;
    convert_weights<10, 128>(W1, w.W1);
    convert_weights<128, 128>(W2, w.W2);
    convert_weights<128, 16>(W3, w.W3);
}

static void forward_tmul_preconv(const TaddWeightsF32& w, const float x[10], int8_t out_trits[5]) {
    using namespace tritnet::weights::tmul_weights;
    float h1[128];
    layer_preconv<10, 128, true>(x, w.W1, B1, h1);
    float h2[128];
    layer_preconv<128, 128, true>(h1, w.W2, B2, h2);
    float logits[16];
    layer_preconv<128, 16, false>(h2, w.W3, B3, logits);
    for (int k = 0; k < 5; ++k) {
        int best = 0;
        float best_val = logits[k * 3];
        for (int c = 1; c < 3; ++c) {
            if (logits[k * 3 + c] > best_val) { best_val = logits[k * 3 + c]; best = c; }
        }
        out_trits[k] = static_cast<int8_t>(best - 1);
    }
}

// tmin, same 10/128/16 architecture, checkpoint accuracy 99.9% -- between
// tadd's 100% and tmul's 99.5%, rounding out the accuracy range checked.
static void prepare_tmin(TaddWeightsF32& w) {
    using namespace tritnet::weights::tmin_weights;
    convert_weights<10, 128>(W1, w.W1);
    convert_weights<128, 128>(W2, w.W2);
    convert_weights<128, 16>(W3, w.W3);
}

static void forward_tmin_preconv(const TaddWeightsF32& w, const float x[10], int8_t out_trits[5]) {
    using namespace tritnet::weights::tmin_weights;
    float h1[128];
    layer_preconv<10, 128, true>(x, w.W1, B1, h1);
    float h2[128];
    layer_preconv<128, 128, true>(h1, w.W2, B2, h2);
    float logits[16];
    layer_preconv<128, 16, false>(h2, w.W3, B3, logits);
    for (int k = 0; k < 5; ++k) {
        int best = 0;
        float best_val = logits[k * 3];
        for (int c = 1; c < 3; ++c) {
            if (logits[k * 3 + c] > best_val) { best_val = logits[k * 3 + c]; best = c; }
        }
        out_trits[k] = static_cast<int8_t>(best - 1);
    }
}

// tmax, same 10/128/16 architecture, 99.9% checkpoint -- last of the 4
// binary ops, completing the set (tadd, tmul, tmin already checked with the
// corrected interleaved-timing methodology; tmax was the one the
// 2026-08-14 session report flagged as not yet independently confirmed).
static void prepare_tmax(TaddWeightsF32& w) {
    using namespace tritnet::weights::tmax_weights;
    convert_weights<10, 128>(W1, w.W1);
    convert_weights<128, 128>(W2, w.W2);
    convert_weights<128, 16>(W3, w.W3);
}

static void forward_tmax_preconv(const TaddWeightsF32& w, const float x[10], int8_t out_trits[5]) {
    using namespace tritnet::weights::tmax_weights;
    float h1[128];
    layer_preconv<10, 128, true>(x, w.W1, B1, h1);
    float h2[128];
    layer_preconv<128, 128, true>(h1, w.W2, B2, h2);
    float logits[16];
    layer_preconv<128, 16, false>(h2, w.W3, B3, logits);
    for (int k = 0; k < 5; ++k) {
        int best = 0;
        float best_val = logits[k * 3];
        for (int c = 1; c < 3; ++c) {
            if (logits[k * 3 + c] > best_val) { best_val = logits[k * 3 + c]; best = c; }
        }
        out_trits[k] = static_cast<int8_t>(best - 1);
    }
}

// tnot's shapes: IN=5, HID=64, OUT_PADDED=16 -- the one unary op, smaller
// hidden layer than the binary ops (64 vs 128), so it's worth checking
// separately rather than assuming tadd's ratio generalizes.
struct TnotWeightsF32 {
    float W1[5][64];
    float W2[64][64];
    float W3[64][16];
};

static void prepare_tnot(TnotWeightsF32& w) {
    using namespace tritnet::weights::tnot_weights;
    convert_weights<5, 64>(W1, w.W1);
    convert_weights<64, 64>(W2, w.W2);
    convert_weights<64, 16>(W3, w.W3);
}

static void forward_tnot_preconv(const TnotWeightsF32& w, const float x[5], int8_t out_trits[5]) {
    using namespace tritnet::weights::tnot_weights;
    float h1[64];
    layer_preconv<5, 64, true>(x, w.W1, B1, h1);
    float h2[64];
    layer_preconv<64, 64, true>(h1, w.W2, B2, h2);
    float logits[16];
    layer_preconv<64, 16, false>(h2, w.W3, B3, logits);
    for (int k = 0; k < 5; ++k) {
        int best = 0;
        float best_val = logits[k * 3];
        for (int c = 1; c < 3; ++c) {
            if (logits[k * 3 + c] > best_val) { best_val = logits[k * 3 + c]; best = c; }
        }
        out_trits[k] = static_cast<int8_t>(best - 1);
    }
}

}  // namespace preconv

// `baseline` is passed in (already computed once for the main table) rather
// than recomputed here -- bench_unary/bench_binary's scalar-path timing
// alone is ~15-20s per op (naive TritNet is ~0.07-0.27 Mops/s, so 200K
// chunks x 5 repeats takes a while), and re-running that a second time for
// every single amortized-check function was needlessly pushing total
// runtime past two minutes. `baseline` is used only for the (cheap, stable)
// LUT reference number, `.lut_mops`.
//
// The AVX2-reconvert-vs-preconv comparison itself is measured fresh, HERE,
// interleaved rep-by-rep via time_best_interleaved() -- not by reusing
// baseline.avx2_mops. Found 2026-08-14: an earlier version of this function
// reused baseline.avx2_mops (measured early in the run, in the main table)
// against a preconv_mops measured much later (after the whole table already
// ran) -- comparing two numbers from different points in a long-running
// binary is itself unfair if the CPU's clock state drifts between them,
// which it reproducibly did on this laptop chip (tadd/tmul/tmin's
// amortization ratio collapsed from ~1.8x to ~0.97x purely from being
// measured later, while tnot -- measured first -- stayed consistent).
// Interleaving both sides' timing loops fixes this at the root.
static void bench_amortized_tnot(const Result& baseline) {
    printf("\n-- Amortized-weight-conversion experiment (tnot, the one unary op) --\n");

    SplitMix64 rng(42);
    std::vector<trit> chunks(static_cast<size_t>(N_CHUNKS) * 5);
    for (int c = 0; c < N_CHUNKS; ++c) random_chunk(rng, &chunks[c * 5]);

    preconv::TnotWeightsF32 w;
    preconv::prepare_tnot(w);  // NOT timed -- this is the "convert once" step

    // Correctness spot-check against the verified AVX2 path first.
    {
        trit out_ref[5];
        float x[5];
        for (int i = 0; i < 5; ++i) x[i] = static_cast<float>(trit_to_int(chunks[i]));
        tritnet::avx2::tritnet_tnot(&chunks[0], out_ref);
        int8_t y[5];
        preconv::forward_tnot_preconv(w, x, y);
        trit out_pc[5];
        for (int i = 0; i < 5; ++i) out_pc[i] = int_to_trit(y[i]);
        bool ok = true;
        for (int i = 0; i < 5; ++i) if (out_ref[i] != out_pc[i]) ok = false;
        printf("  correctness vs verified AVX2 path: %s\n", ok ? "MATCH" : "MISMATCH (bug -- do not trust numbers below)");
        if (!ok) return;
    }

    volatile int sink = 0;
    trit out_reconv[5];
    int8_t out_preconv[5];
    std::vector<float> xs(static_cast<size_t>(N_CHUNKS) * 5);
    for (int c = 0; c < N_CHUNKS; ++c) {
        for (int i = 0; i < 5; ++i) xs[c * 5 + i] = static_cast<float>(trit_to_int(chunks[c * 5 + i]));
    }

    double reconv_s, preconv_s;
    time_best_interleaved(
        N_CHUNKS,
        [&](int c) { tritnet::avx2::tritnet_tnot(&chunks[c * 5], out_reconv); sink += out_reconv[0]; },
        [&](int c) { preconv::forward_tnot_preconv(w, &xs[c * 5], out_preconv); sink += out_preconv[0]; },
        reconv_s, preconv_s);
    (void)sink;

    double reconv_mops = (N_CHUNKS / reconv_s) / 1e6;

    double preconv_mops = (N_CHUNKS / preconv_s) / 1e6;

    printf("  %-28s %10.4f Mops/s\n", "AVX2 (reconverts weights every call, interleaved timing)", reconv_mops);
    printf("  %-28s %10.4f Mops/s\n", "AVX2 (weights preconverted once)", preconv_mops);
    printf("  speedup from amortizing conversion: %.2fx\n", preconv_mops / reconv_mops);
    printf("  LUT/AVX2-preconverted: %.1fx (was %.1fx against the per-call-reconvert AVX2)\n",
           baseline.lut_mops / preconv_mops, baseline.lut_mops / reconv_mops);
}

static void bench_amortized_tadd(const Result& baseline) {
    printf("\n-- Amortized-weight-conversion experiment (tadd, representative of the 4 binary ops) --\n");

    SplitMix64 rng(42);
    std::vector<trit> chunks_a(static_cast<size_t>(N_CHUNKS) * 5);
    std::vector<trit> chunks_b(static_cast<size_t>(N_CHUNKS) * 5);
    for (int c = 0; c < N_CHUNKS; ++c) {
        random_chunk(rng, &chunks_a[c * 5]);
        random_chunk(rng, &chunks_b[c * 5]);
    }

    preconv::TaddWeightsF32 w;
    preconv::prepare_tadd(w);  // NOT timed -- this is the "convert once" step

    // Correctness spot-check: preconverted path must match the verified AVX2
    // path exactly (both compute the same float multiply, just with the
    // int8->float conversion done at a different time).
    {
        trit out_ref[5], out_pc[5];
        float x[10];
        for (int i = 0; i < 5; ++i) x[i] = static_cast<float>(trit_to_int(chunks_a[i]));
        for (int i = 0; i < 5; ++i) x[5 + i] = static_cast<float>(trit_to_int(chunks_b[i]));
        tritnet::avx2::tritnet_tadd(&chunks_a[0], &chunks_b[0], out_ref);
        int8_t y[5];
        preconv::forward_tadd_preconv(w, x, y);
        for (int i = 0; i < 5; ++i) out_pc[i] = int_to_trit(y[i]);
        bool ok = true;
        for (int i = 0; i < 5; ++i) if (out_ref[i] != out_pc[i]) ok = false;
        printf("  correctness vs verified AVX2 path: %s\n", ok ? "MATCH" : "MISMATCH (bug -- do not trust numbers below)");
        if (!ok) return;
    }

    volatile int sink = 0;
    trit out_reconv[5];
    int8_t out_preconv[5];
    std::vector<float> xs(static_cast<size_t>(N_CHUNKS) * 10);
    for (int c = 0; c < N_CHUNKS; ++c) {
        for (int i = 0; i < 5; ++i) xs[c * 10 + i] = static_cast<float>(trit_to_int(chunks_a[c * 5 + i]));
        for (int i = 0; i < 5; ++i) xs[c * 10 + 5 + i] = static_cast<float>(trit_to_int(chunks_b[c * 5 + i]));
    }

    double reconv_s, preconv_s;
    time_best_interleaved(
        N_CHUNKS,
        [&](int c) { tritnet::avx2::tritnet_tadd(&chunks_a[c * 5], &chunks_b[c * 5], out_reconv); sink += out_reconv[0]; },
        [&](int c) { preconv::forward_tadd_preconv(w, &xs[c * 10], out_preconv); sink += out_preconv[0]; },
        reconv_s, preconv_s);
    (void)sink;

    double reconv_mops = (N_CHUNKS / reconv_s) / 1e6;
    double preconv_mops = (N_CHUNKS / preconv_s) / 1e6;

    printf("  %-28s %10.4f Mops/s\n", "AVX2 (reconverts weights every call, interleaved timing)", reconv_mops);
    printf("  %-28s %10.4f Mops/s\n", "AVX2 (weights preconverted once)", preconv_mops);
    printf("  speedup from amortizing conversion: %.2fx\n", preconv_mops / reconv_mops);
    printf("  LUT/AVX2-preconverted: %.1fx (was %.1fx against the per-call-reconvert AVX2)\n",
           baseline.lut_mops / preconv_mops, baseline.lut_mops / reconv_mops);
}

static void bench_amortized_tmul(const Result& baseline) {
    printf("\n-- Amortized-weight-conversion experiment (tmul, weakest checkpoint at 99.5%%) --\n");

    SplitMix64 rng(42);
    std::vector<trit> chunks_a(static_cast<size_t>(N_CHUNKS) * 5);
    std::vector<trit> chunks_b(static_cast<size_t>(N_CHUNKS) * 5);
    for (int c = 0; c < N_CHUNKS; ++c) {
        random_chunk(rng, &chunks_a[c * 5]);
        random_chunk(rng, &chunks_b[c * 5]);
    }

    preconv::TaddWeightsF32 w;
    preconv::prepare_tmul(w);  // NOT timed -- this is the "convert once" step

    // Correctness spot-check: preconverted path must match the verified AVX2
    // path exactly (both compute the same float multiply, just with the
    // int8->float conversion done at a different time).
    {
        trit out_ref[5], out_pc[5];
        float x[10];
        for (int i = 0; i < 5; ++i) x[i] = static_cast<float>(trit_to_int(chunks_a[i]));
        for (int i = 0; i < 5; ++i) x[5 + i] = static_cast<float>(trit_to_int(chunks_b[i]));
        tritnet::avx2::tritnet_tmul(&chunks_a[0], &chunks_b[0], out_ref);
        int8_t y[5];
        preconv::forward_tmul_preconv(w, x, y);
        for (int i = 0; i < 5; ++i) out_pc[i] = int_to_trit(y[i]);
        bool ok = true;
        for (int i = 0; i < 5; ++i) if (out_ref[i] != out_pc[i]) ok = false;
        printf("  correctness vs verified AVX2 path: %s\n", ok ? "MATCH" : "MISMATCH (bug -- do not trust numbers below)");
        if (!ok) return;
    }

    volatile int sink = 0;
    trit out_reconv[5];
    int8_t out_preconv[5];
    std::vector<float> xs(static_cast<size_t>(N_CHUNKS) * 10);
    for (int c = 0; c < N_CHUNKS; ++c) {
        for (int i = 0; i < 5; ++i) xs[c * 10 + i] = static_cast<float>(trit_to_int(chunks_a[c * 5 + i]));
        for (int i = 0; i < 5; ++i) xs[c * 10 + 5 + i] = static_cast<float>(trit_to_int(chunks_b[c * 5 + i]));
    }

    double reconv_s, preconv_s;
    time_best_interleaved(
        N_CHUNKS,
        [&](int c) { tritnet::avx2::tritnet_tmul(&chunks_a[c * 5], &chunks_b[c * 5], out_reconv); sink += out_reconv[0]; },
        [&](int c) { preconv::forward_tmul_preconv(w, &xs[c * 10], out_preconv); sink += out_preconv[0]; },
        reconv_s, preconv_s);
    (void)sink;

    double reconv_mops = (N_CHUNKS / reconv_s) / 1e6;
    double preconv_mops = (N_CHUNKS / preconv_s) / 1e6;

    printf("  %-28s %10.4f Mops/s\n", "AVX2 (reconverts weights every call, interleaved timing)", reconv_mops);
    printf("  %-28s %10.4f Mops/s\n", "AVX2 (weights preconverted once)", preconv_mops);
    printf("  speedup from amortizing conversion: %.2fx\n", preconv_mops / reconv_mops);
    printf("  LUT/AVX2-preconverted: %.1fx (was %.1fx against the per-call-reconvert AVX2)\n",
           baseline.lut_mops / preconv_mops, baseline.lut_mops / reconv_mops);
}

static void bench_amortized_tmin(const Result& baseline) {
    printf("\n-- Amortized-weight-conversion experiment (tmin, 99.9%% checkpoint) --\n");

    SplitMix64 rng(42);
    std::vector<trit> chunks_a(static_cast<size_t>(N_CHUNKS) * 5);
    std::vector<trit> chunks_b(static_cast<size_t>(N_CHUNKS) * 5);
    for (int c = 0; c < N_CHUNKS; ++c) {
        random_chunk(rng, &chunks_a[c * 5]);
        random_chunk(rng, &chunks_b[c * 5]);
    }

    preconv::TaddWeightsF32 w;
    preconv::prepare_tmin(w);  // NOT timed -- this is the "convert once" step

    // Correctness spot-check: preconverted path must match the verified AVX2
    // path exactly (both compute the same float multiply, just with the
    // int8->float conversion done at a different time).
    {
        trit out_ref[5], out_pc[5];
        float x[10];
        for (int i = 0; i < 5; ++i) x[i] = static_cast<float>(trit_to_int(chunks_a[i]));
        for (int i = 0; i < 5; ++i) x[5 + i] = static_cast<float>(trit_to_int(chunks_b[i]));
        tritnet::avx2::tritnet_tmin(&chunks_a[0], &chunks_b[0], out_ref);
        int8_t y[5];
        preconv::forward_tmin_preconv(w, x, y);
        for (int i = 0; i < 5; ++i) out_pc[i] = int_to_trit(y[i]);
        bool ok = true;
        for (int i = 0; i < 5; ++i) if (out_ref[i] != out_pc[i]) ok = false;
        printf("  correctness vs verified AVX2 path: %s\n", ok ? "MATCH" : "MISMATCH (bug -- do not trust numbers below)");
        if (!ok) return;
    }

    volatile int sink = 0;
    trit out_reconv[5];
    int8_t out_preconv[5];
    std::vector<float> xs(static_cast<size_t>(N_CHUNKS) * 10);
    for (int c = 0; c < N_CHUNKS; ++c) {
        for (int i = 0; i < 5; ++i) xs[c * 10 + i] = static_cast<float>(trit_to_int(chunks_a[c * 5 + i]));
        for (int i = 0; i < 5; ++i) xs[c * 10 + 5 + i] = static_cast<float>(trit_to_int(chunks_b[c * 5 + i]));
    }

    double reconv_s, preconv_s;
    time_best_interleaved(
        N_CHUNKS,
        [&](int c) { tritnet::avx2::tritnet_tmin(&chunks_a[c * 5], &chunks_b[c * 5], out_reconv); sink += out_reconv[0]; },
        [&](int c) { preconv::forward_tmin_preconv(w, &xs[c * 10], out_preconv); sink += out_preconv[0]; },
        reconv_s, preconv_s);
    (void)sink;

    double reconv_mops = (N_CHUNKS / reconv_s) / 1e6;
    double preconv_mops = (N_CHUNKS / preconv_s) / 1e6;

    printf("  %-28s %10.4f Mops/s\n", "AVX2 (reconverts weights every call, interleaved timing)", reconv_mops);
    printf("  %-28s %10.4f Mops/s\n", "AVX2 (weights preconverted once)", preconv_mops);
    printf("  speedup from amortizing conversion: %.2fx\n", preconv_mops / reconv_mops);
    printf("  LUT/AVX2-preconverted: %.1fx (was %.1fx against the per-call-reconvert AVX2)\n",
           baseline.lut_mops / preconv_mops, baseline.lut_mops / reconv_mops);
}

static void bench_amortized_tmax(const Result& baseline) {
    printf("\n-- Amortized-weight-conversion experiment (tmax, 99.9%% checkpoint, last of the 4 binary ops) --\n");

    SplitMix64 rng(42);
    std::vector<trit> chunks_a(static_cast<size_t>(N_CHUNKS) * 5);
    std::vector<trit> chunks_b(static_cast<size_t>(N_CHUNKS) * 5);
    for (int c = 0; c < N_CHUNKS; ++c) {
        random_chunk(rng, &chunks_a[c * 5]);
        random_chunk(rng, &chunks_b[c * 5]);
    }

    preconv::TaddWeightsF32 w;
    preconv::prepare_tmax(w);  // NOT timed -- this is the "convert once" step

    // Correctness spot-check: preconverted path must match the verified AVX2
    // path exactly (both compute the same float multiply, just with the
    // int8->float conversion done at a different time).
    {
        trit out_ref[5], out_pc[5];
        float x[10];
        for (int i = 0; i < 5; ++i) x[i] = static_cast<float>(trit_to_int(chunks_a[i]));
        for (int i = 0; i < 5; ++i) x[5 + i] = static_cast<float>(trit_to_int(chunks_b[i]));
        tritnet::avx2::tritnet_tmax(&chunks_a[0], &chunks_b[0], out_ref);
        int8_t y[5];
        preconv::forward_tmax_preconv(w, x, y);
        for (int i = 0; i < 5; ++i) out_pc[i] = int_to_trit(y[i]);
        bool ok = true;
        for (int i = 0; i < 5; ++i) if (out_ref[i] != out_pc[i]) ok = false;
        printf("  correctness vs verified AVX2 path: %s\n", ok ? "MATCH" : "MISMATCH (bug -- do not trust numbers below)");
        if (!ok) return;
    }

    volatile int sink = 0;
    trit out_reconv[5];
    int8_t out_preconv[5];
    std::vector<float> xs(static_cast<size_t>(N_CHUNKS) * 10);
    for (int c = 0; c < N_CHUNKS; ++c) {
        for (int i = 0; i < 5; ++i) xs[c * 10 + i] = static_cast<float>(trit_to_int(chunks_a[c * 5 + i]));
        for (int i = 0; i < 5; ++i) xs[c * 10 + 5 + i] = static_cast<float>(trit_to_int(chunks_b[c * 5 + i]));
    }

    double reconv_s, preconv_s;
    time_best_interleaved(
        N_CHUNKS,
        [&](int c) { tritnet::avx2::tritnet_tmax(&chunks_a[c * 5], &chunks_b[c * 5], out_reconv); sink += out_reconv[0]; },
        [&](int c) { preconv::forward_tmax_preconv(w, &xs[c * 10], out_preconv); sink += out_preconv[0]; },
        reconv_s, preconv_s);
    (void)sink;

    double reconv_mops = (N_CHUNKS / reconv_s) / 1e6;
    double preconv_mops = (N_CHUNKS / preconv_s) / 1e6;

    printf("  %-28s %10.4f Mops/s\n", "AVX2 (reconverts weights every call, interleaved timing)", reconv_mops);
    printf("  %-28s %10.4f Mops/s\n", "AVX2 (weights preconverted once)", preconv_mops);
    printf("  speedup from amortizing conversion: %.2fx\n", preconv_mops / reconv_mops);
    printf("  LUT/AVX2-preconverted: %.1fx (was %.1fx against the per-call-reconvert AVX2)\n",
           baseline.lut_mops / preconv_mops, baseline.lut_mops / reconv_mops);
}
#endif  // __AVX2__

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
    Result r_tnot = bench_unary(tnot, tritnet::tritnet_tnot, tritnet::avx2::tritnet_tnot);
    Result r_tadd = bench_binary(tadd, tritnet::tritnet_tadd, tritnet::avx2::tritnet_tadd);
    Result r_tmul = bench_binary(tmul, tritnet::tritnet_tmul, tritnet::avx2::tritnet_tmul);
    Result r_tmin = bench_binary(tmin, tritnet::tritnet_tmin, tritnet::avx2::tritnet_tmin);
    Result r_tmax = bench_binary(tmax, tritnet::tritnet_tmax, tritnet::avx2::tritnet_tmax);
    print_row("tnot", r_tnot);
    print_row("tadd", r_tadd);
    print_row("tmul", r_tmul);
    print_row("tmin", r_tmin);
    print_row("tmax", r_tmax);
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

#ifdef __AVX2__
    // Reuse the Results already computed above -- see bench_amortized_tnot's
    // comment for why recomputing them here would be both wasteful and a
    // thermal-throttling risk.
    bench_amortized_tnot(r_tnot);
    bench_amortized_tadd(r_tadd);
    bench_amortized_tmul(r_tmul);
    bench_amortized_tmin(r_tmin);
    bench_amortized_tmax(r_tmax);
#endif

    return 0;
}
