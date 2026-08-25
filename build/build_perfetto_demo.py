#!/usr/bin/env python3
"""
build_perfetto_demo.py - Build and run the Perfetto profiler integration demo

Closes part of Critical Gap #10 (profiler integration): no build script
anywhere in build/ ever defined TERNARY_ENABLE_VTUNE/_NVTX/_PERFETTO, so
every existing build only ever exercised ternary_profiler.h's no-op stub.
Of the three backends, only Perfetto needs no proprietary tool (VTune) or
GPU (NVTX) to build AND verify against -- see third_party/perfetto/README.md
and reports/2026-08-25/PERFETTO_PROFILER_INTEGRATION.md.

Builds benchmarks/cpp-native-kernels/bench_perfetto_trace.cpp with
-DTERNARY_ENABLE_PERFETTO, linking third_party/perfetto/perfetto.cc and
src/core/profiling/ternary_profiler_perfetto.cc, then runs it once to
produce a real .perfetto-trace file -- open at https://ui.perfetto.dev,
or query with trace_processor_shell (fetched separately; not vendored,
~14MB standalone binary, not needed to build or use this demo).

USAGE:
    python build/build_perfetto_demo.py
    python build/build_perfetto_demo.py --no-run

OUTPUT:
    benchmarks/cpp-native-kernels/bench_perfetto_trace (Linux/macOS) or
    .exe (Windows), plus (unless --no-run) a trace file at
    benchmarks/results/ternary_hotpath.perfetto-trace

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0
"""

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
KERNELS_DIR = ROOT_DIR / "benchmarks" / "cpp-native-kernels"
RESULTS_DIR = ROOT_DIR / "benchmarks" / "results"


def main():
    parser = argparse.ArgumentParser(description="Build the Perfetto profiler demo")
    parser.add_argument("--no-run", action="store_true", help="Build only, don't run")
    args = parser.parse_args()

    if platform.system() == "Windows":
        print("This demo's build command targets GCC/Clang (Linux/macOS).")
        print("On Windows, compile bench_perfetto_trace.cpp manually with MSVC")
        print("(see the file's own header comment for the reference g++ command")
        print("to translate) -- not attempted here, matching this project's")
        print("verify-by-execution discipline (no Windows machine in this session).")
        return 1

    compiler = "g++" if shutil.which("g++") else ("clang++" if shutil.which("clang++") else None)
    if compiler is None:
        print("ERROR: no g++ or clang++ found")
        return 1

    output = KERNELS_DIR / "bench_perfetto_trace"
    cmd = [
        compiler, "-O3", "-march=haswell", "-mavx2", "-mfma", "-fopenmp", "-std=c++17",
        "-DTERNARY_ENABLE_PERFETTO",
        f"-I{ROOT_DIR / 'src' / 'core'}", f"-I{ROOT_DIR}",
        str(KERNELS_DIR / "bench_perfetto_trace.cpp"),
        str(ROOT_DIR / "src" / "core" / "profiling" / "ternary_profiler_perfetto.cc"),
        str(ROOT_DIR / "third_party" / "perfetto" / "perfetto.cc"),
        "-o", str(output), "-lpthread",
    ]

    print("=" * 70)
    print("Building Perfetto profiler demo")
    print("=" * 70)
    print(f"Compiler: {compiler}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("BUILD FAILED")
        print(result.stderr[-4000:])
        return 1
    print(f"Built: {output}")

    if args.no_run:
        return 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = RESULTS_DIR / "ternary_hotpath.perfetto-trace"
    print("\nRunning demo (traces a real tadd hot path)...")
    run_result = subprocess.run([str(output), str(trace_path)])
    if run_result.returncode != 0:
        print("DEMO RUN FAILED")
        return 1

    print(f"\nTrace written to {trace_path}")
    print("Open at https://ui.perfetto.dev, or query with trace_processor_shell")
    print("(https://github.com/google/perfetto/releases -- not vendored here).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
