// timer.h — High-resolution timer for benchmarking
//
// Copyright (c) 2025 Jonathan Verdun (Ternary Core Experimental Project)
// Licensed under the Apache License, Version 2.0

#pragma once

#include <chrono>
#include <functional>
#include <algorithm>  // std::min_element, std::max_element — used below, wasn't included
#include <numeric>    // std::accumulate — used below, wasn't included
#include <tuple>       // std::tuple — used below, wasn't included
#include <vector>      // std::vector — used below, wasn't included

// Named bench_clock_t, not clock_t: `using clock_t = ...` collides with the
// C standard library's own ::clock_t (typedef'd in <time.h>, transitively
// pulled in by <chrono> on glibc/libstdc++) — "conflicting declaration"
// compile error, reproduced 2026-08-12 on Linux/GCC. This file has no
// Linux/macOS build script (only .bat files exist alongside it), so this
// was presumably never compiled outside MSVC, where <chrono> apparently
// doesn't transitively expose a conflicting ::clock_t the same way.
using bench_clock_t = std::chrono::steady_clock;

/**
 * Time a function's execution with multiple repeats
 *
 * @param fn Function to time (can be lambda)
 * @param repeats Number of repetitions (default: 5)
 * @return Average execution time in nanoseconds
 */
template <typename F>
double time_ns(F&& fn, int repeats = 5) {
    double total = 0.0;
    for (int i = 0; i < repeats; ++i) {
        auto start = bench_clock_t::now();
        fn();
        auto end = bench_clock_t::now();
        total += std::chrono::duration<double, std::nano>(end - start).count();
    }
    return total / repeats;
}

/**
 * Time a function and return min/max/mean statistics
 *
 * @param fn Function to time
 * @param repeats Number of repetitions
 * @return Tuple of (min_ns, max_ns, mean_ns)
 */
template <typename F>
std::tuple<double, double, double> time_stats(F&& fn, int repeats = 5) {
    std::vector<double> times;
    times.reserve(repeats);

    for (int i = 0; i < repeats; ++i) {
        auto start = bench_clock_t::now();
        fn();
        auto end = bench_clock_t::now();
        times.push_back(std::chrono::duration<double, std::nano>(end - start).count());
    }

    double min = *std::min_element(times.begin(), times.end());
    double max = *std::max_element(times.begin(), times.end());
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / times.size();

    return {min, max, mean};
}
