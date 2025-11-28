// benchmark_main.cpp — Production-grade benchmark suite for ternary_simd_engine
//
// Compile (from benchmarks/cpp-native-kernels/ directory):
//   clang++ -O3 -march=native -fopenmp -std=c++17 -I../../src benchmark_main.cpp -o bench
//   g++ -O3 -march=native -fopenmp -std=c++17 -I../../src benchmark_main.cpp -o bench
//
// Windows (MSVC):
//   cl /O2 /arch:AVX2 /openmp /std:c++17 /EHsc /I..\..\src benchmark_main.cpp
//
// Run:
//   ./bench --repeat=5 --threads=12 --out=results/bench.json
//
// JSON + CSV telemetry outputs for CI dashboards (Grafana/Plotly/Sheets).
// Deterministic RNG ensures reproducibility.
//
// TARGET: src/core/ - Production SIMD kernels (validated, stable)

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <thread>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <cstring>
#include <omp.h>

// Core kernel includes (src/core/)
#include "core/algebra/ternary_algebra.h"
#include "core/simd/simd_avx2_32trit_ops.h"

// Simple JSON builder (no external dependency)
namespace json {
    struct Value {
        std::string data;
        Value(const std::string& s) : data("\"" + s + "\"") {}
        Value(const char* s) : data("\"" + std::string(s) + "\"") {}
        Value(int v) : data(std::to_string(v)) {}
        Value(size_t v) : data(std::to_string(v)) {}
        Value(double v) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << v;
            data = oss.str();
        }
        Value(bool v) : data(v ? "true" : "false") {}
    };

    struct Object {
        std::map<std::string, Value> fields;
        void add(const std::string& key, const Value& val) { fields[key] = val; }
        std::string str() const {
            std::string s = "{";
            bool first = true;
            for (const auto& [k, v] : fields) {
                if (!first) s += ",";
                s += "\"" + k + "\":" + v.data;
                first = false;
            }
            s += "}";
            return s;
        }
    };

    struct Array {
        std::vector<Object> items;
        void add(const Object& obj) { items.push_back(obj); }
        std::string str() const {
            std::string s = "[";
            for (size_t i = 0; i < items.size(); ++i) {
                if (i > 0) s += ",";
                s += items[i].str();
            }
            s += "]";
            return s;
        }
    };

    struct Document {
        Object meta;
        Array runs;
        std::string str() const {
            return "{\"meta\":" + meta.str() + ",\"runs\":" + runs.str() + "}";
        }
    };
}

using clock_t = std::chrono::steady_clock;

// ─────────────────────────────────────────────────────────────────────────────
// Utility: deterministic RNG for trits
// ─────────────────────────────────────────────────────────────────────────────
std::vector<uint8_t> make_random_trits(size_t n, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 2);
    std::vector<uint8_t> data(n);
    for (auto &v : data) {
        int x = dist(rng);
        v = (x == 0) ? 0b00 : (x == 1) ? 0b01 : 0b10;
    }
    return data;
}

// ─────────────────────────────────────────────────────────────────────────────
// Timer helper
// ─────────────────────────────────────────────────────────────────────────────
template <typename F>
double time_ns(F&& fn, int repeats = 5) {
    double total = 0;
    for (int i = 0; i < repeats; ++i) {
        auto s = clock_t::now();
        fn();
        auto e = clock_t::now();
        total += std::chrono::duration<double, std::nano>(e - s).count();
    }
    return total / repeats;
}

// ─────────────────────────────────────────────────────────────────────────────
// Generic benchmark kernel
// ─────────────────────────────────────────────────────────────────────────────
struct BenchResult {
    std::string op;
    std::string mode;
    bool sanitize;
    size_t n;
    double ns_total;
    double ns_per_elem;
};

template <typename Fn>
BenchResult bench_generic(const std::string& op, const std::string& mode, bool sanitize,
                          size_t n, Fn&& fn, int repeats) {
    double ns = time_ns(fn, repeats);
    return {op, mode, sanitize, n, ns, ns / n};
}

// ─────────────────────────────────────────────────────────────────────────────
// SIMD wrappers (direct C++ invocation, no pybind11)
// ─────────────────────────────────────────────────────────────────────────────
template <bool Sanitize = true>
void run_simd_binary(const uint8_t* a, const uint8_t* b, uint8_t* r, size_t n,
                     __m256i (*simd_fn)(__m256i, __m256i)) {
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((__m256i const*)(a + i));
        __m256i vb = _mm256_loadu_si256((__m256i const*)(b + i));
        __m256i vr = simd_fn(va, vb);
        _mm256_storeu_si256((__m256i*)(r + i), vr);
    }
    // Scalar tail
    for (; i < n; ++i) r[i] = 0;
}

template <bool Sanitize = true>
void run_simd_unary(const uint8_t* a, uint8_t* r, size_t n,
                    __m256i (*simd_fn)(__m256i)) {
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((__m256i const*)(a + i));
        __m256i vr = simd_fn(va);
        _mm256_storeu_si256((__m256i*)(r + i), vr);
    }
    // Scalar tail
    for (; i < n; ++i) r[i] = 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI argument parsing
// ─────────────────────────────────────────────────────────────────────────────
std::map<std::string, std::string> parse_args(int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string s(argv[i]);
        auto pos = s.find('=');
        if (pos != std::string::npos) {
            args[s.substr(2, pos - 2)] = s.substr(pos + 1);
        }
    }
    return args;
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);
    int repeats = args.count("repeat") ? std::stoi(args["repeat"]) : 5;
    int threads = args.count("threads") ? std::stoi(args["threads"]) : std::thread::hardware_concurrency();
    std::string outpath = args.count("out") ? args["out"] : "results/bench.json";

    omp_set_num_threads(threads);

    std::vector<size_t> sizes = {1000, 10000, 100000, 1000000};
    std::vector<BenchResult> results;

    // Warm-up LUTs
    volatile auto warm = tadd(0b00, 0b10);

    std::cout << "🧪 Ternary SIMD Benchmark Suite\n";
    std::cout << "   Threads: " << threads << "\n";
    std::cout << "   Repeats: " << repeats << "\n";
    std::cout << "   Sizes: ";
    for (auto s : sizes) std::cout << s << " ";
    std::cout << "\n\n";

    // ─────────────────────────────────────────────────────────────────────────
    // BENCHMARK LOOP
    // ─────────────────────────────────────────────────────────────────────────
    for (auto n : sizes) {
        std::cout << "Testing n=" << n << "... " << std::flush;

        auto A = make_random_trits(n, 42);
        auto B = make_random_trits(n, 1337);
        std::vector<uint8_t> R(n);

        // ---- tadd ----
        results.push_back(bench_generic("tadd", "SIMD_san", true, n, [&] {
            run_simd_binary<true>(A.data(), B.data(), R.data(), n, tadd_simd<true>);
        }, repeats));

        results.push_back(bench_generic("tadd", "SIMD_raw", false, n, [&] {
            run_simd_binary<false>(A.data(), B.data(), R.data(), n, tadd_simd<false>);
        }, repeats));

        results.push_back(bench_generic("tadd", "Scalar", true, n, [&] {
            for (size_t i = 0; i < n; ++i) R[i] = tadd(A[i], B[i]);
        }, repeats));

        // ---- tmul ----
        results.push_back(bench_generic("tmul", "SIMD_san", true, n, [&] {
            run_simd_binary<true>(A.data(), B.data(), R.data(), n, tmul_simd<true>);
        }, repeats));

        results.push_back(bench_generic("tmul", "SIMD_raw", false, n, [&] {
            run_simd_binary<false>(A.data(), B.data(), R.data(), n, tmul_simd<false>);
        }, repeats));

        results.push_back(bench_generic("tmul", "Scalar", true, n, [&] {
            for (size_t i = 0; i < n; ++i) R[i] = tmul(A[i], B[i]);
        }, repeats));

        // ---- tmin ----
        results.push_back(bench_generic("tmin", "SIMD_san", true, n, [&] {
            run_simd_binary<true>(A.data(), B.data(), R.data(), n, tmin_simd<true>);
        }, repeats));

        results.push_back(bench_generic("tmin", "SIMD_raw", false, n, [&] {
            run_simd_binary<false>(A.data(), B.data(), R.data(), n, tmin_simd<false>);
        }, repeats));

        results.push_back(bench_generic("tmin", "Scalar", true, n, [&] {
            for (size_t i = 0; i < n; ++i) R[i] = tmin(A[i], B[i]);
        }, repeats));

        // ---- tmax ----
        results.push_back(bench_generic("tmax", "SIMD_san", true, n, [&] {
            run_simd_binary<true>(A.data(), B.data(), R.data(), n, tmax_simd<true>);
        }, repeats));

        results.push_back(bench_generic("tmax", "SIMD_raw", false, n, [&] {
            run_simd_binary<false>(A.data(), B.data(), R.data(), n, tmax_simd<false>);
        }, repeats));

        results.push_back(bench_generic("tmax", "Scalar", true, n, [&] {
            for (size_t i = 0; i < n; ++i) R[i] = tmax(A[i], B[i]);
        }, repeats));

        // ---- tnot ----
        results.push_back(bench_generic("tnot", "SIMD_san", true, n, [&] {
            run_simd_unary<true>(A.data(), R.data(), n, tnot_simd<true>);
        }, repeats));

        results.push_back(bench_generic("tnot", "SIMD_raw", false, n, [&] {
            run_simd_unary<false>(A.data(), R.data(), n, tnot_simd<false>);
        }, repeats));

        results.push_back(bench_generic("tnot", "Scalar", true, n, [&] {
            for (size_t i = 0; i < n; ++i) R[i] = tnot(A[i]);
        }, repeats));

        std::cout << "done\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // JSON + CSV output
    // ─────────────────────────────────────────────────────────────────────────
    json::Document doc;
    doc.meta.add("compiler", "clang++ -O3 -march=native");
    doc.meta.add("threads", threads);
    doc.meta.add("repeat", repeats);
    doc.meta.add("cpu_threads", (int)std::thread::hardware_concurrency());
    doc.meta.add("timestamp", (int)std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));

    std::ofstream csv("results/bench.csv");
    csv << "op,mode,sanitize,n,ns_total,ns_per_elem\n";

    for (auto& r : results) {
        json::Object obj;
        obj.add("op", r.op);
        obj.add("mode", r.mode);
        obj.add("sanitize", r.sanitize);
        obj.add("n", r.n);
        obj.add("ns_total", r.ns_total);
        obj.add("ns_per_elem", r.ns_per_elem);
        doc.runs.add(obj);

        csv << r.op << "," << r.mode << "," << r.sanitize << ","
            << r.n << "," << std::fixed << std::setprecision(3)
            << r.ns_total << "," << r.ns_per_elem << "\n";
    }
    csv.close();

    std::ofstream f(outpath);
    f << doc.str();
    f.close();

    std::cout << "\n✅ Benchmark complete: " << results.size() << " runs\n"
              << "   JSON → " << outpath << "\n"
              << "   CSV  → results/bench.csv\n";

    return 0;
}
