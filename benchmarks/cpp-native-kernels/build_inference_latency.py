"""
build_inference_latency.py - build bench_inference_latency_fp16.cpp

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Builds the criterion-3 benchmark (ternary inference latency vs FP16), which
needs a genuinely strong FP32 GEMM as its baseline. Rather than requiring a
system BLAS dev package, this links the OpenBLAS that NumPy already ships.

That library is built ILP64 with a vendor prefix, so its symbols are named
e.g. `scipy_cblas_sgemm64_` rather than `cblas_sgemm`. The exact spelling
varies by NumPy/SciPy build, so it is discovered from the .so's dynamic
symbol table here and passed to the compiler as -DCBLAS_SGEMM_SYM=... --
hardcoding one spelling would silently break on a different wheel.

USAGE: python benchmarks/cpp-native-kernels/build_inference_latency.py [--run]
OUTPUT: ./bench_inference_latency_fp16 in this directory
"""

import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent          # benchmarks/cpp-native-kernels -> repo root


def find_openblas() -> str:
    import numpy
    pats = [
        os.path.join(os.path.dirname(numpy.__file__), "..", "numpy.libs",
                     "libscipy_openblas*.so*"),
        os.path.join(os.path.dirname(numpy.__file__), "..", "scipy.libs",
                     "libscipy_openblas*.so*"),
        os.path.join(os.path.dirname(numpy.__file__), ".libs", "libopenblas*.so*"),
    ]
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits:
            return os.path.realpath(hits[0])
    raise SystemExit(
        "Could not locate NumPy's bundled OpenBLAS. This benchmark needs a\n"
        "strong FP32 GEMM as its baseline -- a hand-rolled loop would be a\n"
        "weak reference and would flatter the ternary kernel for the wrong\n"
        "reason. Install a system OpenBLAS and adapt this script if needed.")


def find_symbols(lib: str) -> tuple:
    """Returns (sgemm, sgemv, set_num_threads) symbols as actually exported."""
    out = subprocess.run(["nm", "-D", "--defined-only", lib],
                         capture_output=True, text=True).stdout
    names = re.findall(r"\s[TW]\s+(\S+)", out)

    def pick(cands, what):
        for c in cands:
            if c in names:
                return c
        raise SystemExit(f"No {what} symbol found in {lib}.\n"
                         f"Tried: {cands}")

    sgemm = pick(["scipy_cblas_sgemm64_", "cblas_sgemm64_",
                  "cblas_sgemm_", "cblas_sgemm"], "cblas_sgemm")
    sgemv = pick(["scipy_cblas_sgemv64_", "cblas_sgemv64_",
                  "cblas_sgemv_", "cblas_sgemv"], "cblas_sgemv")
    threads = pick(["scipy_openblas_set_num_threads64_",
                    "openblas_set_num_threads64_",
                    "scipy_openblas_set_num_threads_64_",
                    "openblas_set_num_threads"], "openblas_set_num_threads")
    return sgemm, sgemv, threads


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true", help="run after building")
    args = ap.parse_args()

    lib = find_openblas()
    sgemm, sgemv, threads = find_symbols(lib)
    print(f"OpenBLAS : {lib}")
    print(f"  sgemm  : {sgemm}")
    print(f"  sgemv  : {sgemv}")
    print(f"  threads: {threads}")

    out = HERE / "bench_inference_latency_fp16"
    cmd = [
        "g++", "-O3", "-march=haswell", "-mavx2", "-mfma", "-mf16c",
        "-std=c++17",
        f"-DCBLAS_SGEMM_SYM={sgemm}",
        f"-DCBLAS_SGEMV_SYM={sgemv}",
        f"-DOPENBLAS_SET_THREADS_SYM={threads}",
        f"-I{ROOT / 'src' / 'core' / 'simd'}",
        str(HERE / "bench_inference_latency_fp16.cpp"),
        str(ROOT / "src" / "core" / "simd" / "ternary_gemm_dense.cpp"),
        lib,
        f"-Wl,-rpath,{os.path.dirname(lib)}",
        "-o", str(out),
    ]
    print("\n" + " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("[FAIL] compilation failed")
        return 1
    print(f"[OK] built {out}")

    if args.run:
        print()
        return subprocess.run([str(out)]).returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
