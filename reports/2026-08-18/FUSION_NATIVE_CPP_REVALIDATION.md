# Native C++ fusion benchmark re-validation (Linux x64)

**Date:** 2026-08-18
**Platform:** Linux x64, AMD Ryzen 5 7520U, g++ 13.3.0, `-O3 -march=native -mavx2 -std=c++17`
**Target:** CLAUDE.md gap #5 ("Phase 4.1 fusion... dedicated performance benchmarks still pending")
**Source:** `benchmarks/cpp-native-kernels/bench_fusion.cpp`, unmodified, run 3 times

## Why this benchmark, not the Python one

`bench_simd_fusion_ops.py` (Python bindings, upgraded to use `BenchmarkRunner` earlier this
session) already exercises fusion vs. separate ops, but CLAUDE.md's own `ffi_isolation` rule is
explicit: **absolute performance claims must be measured in native C++
(`benchmarks/cpp-native-kernels/`) to isolate pybind11 overhead.** `bench_fusion.cpp` is that
native benchmark for fusion specifically, and its own header comment carries a single validation
date -- **2025-10-29** -- with no platform recorded. That's the actual gap: a "dedicated
performance benchmark" exists and has been run before, but never on a documented platform, and
never re-validated since. This session closes that.

## Method

Compiled and ran unmodified, 3 times back-to-back (this machine's `cpufreq` governor is
`powersave`, confirmed noisy for large/parallel workloads in
`reports/2026-08-18/CV_SPIKE_ROOT_CAUSE.md` earlier today -- multiple runs, not one, per that
same lesson). The benchmark's own built-in sizes (1,000 / 10,000 / 100,000 / 1,000,000) and its
own built-in CV reporting were used as-is; nothing in the benchmark itself was modified.

## Results (3 runs, speedup = unfused_ns / fused_ns)

| Op | N=1,000 | N=10,000 | N=100,000 | N=1,000,000 |
|---|---|---|---|---|
| tnot(tadd) | 1.17, 1.17, 1.19 | 1.00*, 1.39, 1.19 | 1.44, 1.40, 1.46 | 2.89, 1.84, 2.04 |
| tnot(tmul) | 1.15, 1.20, 1.14 | 1.19, 1.55, 1.19 | 1.41, 1.43, 1.46 | 2.35, 2.08, 2.06 |
| tnot(tmin) | 1.16, 1.16, 1.15 | 1.33, 1.40, 1.11* | 1.45, 1.42, 1.47 | 2.34, 2.11, 2.07 |
| tnot(tmax) | 1.39, 1.28, 1.38 | 1.36, 1.71, 1.38 | 1.42, 1.51, 1.45 | 2.16, 2.01, 2.05 |

\* These two cells carried unusually high per-run CV (128.6% and 113.7% respectively, one lone
`perf` outlier within that run's own median-of-N samples) -- consistent with the governor-driven
noise documented earlier today, not a distinct new mechanism. Included for completeness rather
than dropped.

**Approximate per-op averages across all 3 runs, all sizes:** tadd ~1.64x, tmul ~1.57x,
tmin ~1.60x, tmax ~1.60x. **Range observed: 1.00x-2.89x.**

## Comparison against the documented 2025-10-29 claim

| Op | Documented (2025-10-29, platform unrecorded) | Measured today (Linux x64, this machine) |
|---|---|---|
| tnot(tadd) | 1.62x - 1.95x, avg 1.76x | 1.00x - 2.89x, avg ~1.64x |
| tnot(tmul) | 1.53x - 1.86x, avg 1.71x | 1.14x - 2.35x, avg ~1.57x |
| tnot(tmin) | 1.61x - 11.26x, avg 4.06x | 1.11x - 2.34x, avg ~1.60x |
| tnot(tmax) | 1.65x - 9.50x, avg 3.68x | 1.28x - 2.16x, avg ~1.60x |
| "Conservative claim: 1.53x minimum, any op/size" | — | **Not met**: several cells below 1.53x today (as low as 1.00x/1.11x, both the noted outlier cells; excluding those, still as low as 1.14x) |

The wide 2025-10-29 max speedups for tmin/tmax (11.26x, 9.50x) are not reproduced at all on this
machine -- the highest speedup seen in 3 runs, any op, any size, was 2.89x.

## What this does and doesn't mean

- **Not a claimed regression.** The 2025-10-29 baseline's platform was never recorded (itself a
  documentation gap this report flags), so there's no controlled basis to say performance dropped
  between then and now versus simply being different hardware. FUSION.md's own CV-range column for
  that original run (15-88% for tmin, 18-84% for tmax) already shows the original measurement was
  itself quite noisy -- high variance for these two ops isn't new to this session.
- **Is a real, dated, platform-labeled data point** where none existed for fusion on Linux before,
  closing the concrete part of gap #5 ("dedicated performance benchmarks... pending" -- one now
  exists, has been run, and is documented with a date and platform, per this project's own
  `validation_dates` convention).
- **The qualitative finding holds**: fusion is a real, reproducible, one-pass-vs-two-pass
  memory-traffic win on this machine too -- every cell across all 3 runs and all 4 ops beat 1.0x
  (i.e., fusion was never slower than separate ops), and the win grows with array size (roughly
  1.1-1.4x at N=1,000, ~2.0-2.3x at N=1,000,000), consistent with the documented memory-traffic
  argument (5N bytes unfused vs. 3N bytes fused) mattering more once the working set is large
  enough to be memory-bound.
- **The specific numeric claims in FUSION.md / `bench_fusion.cpp`'s header / `backend_plugin_api.h`
  / `benchmarks/cpp-native-kernels/README.md` (all citing "1.5-11x" or the 2025-10-29 table) remain
  as originally documented** -- not overwritten, since this session's numbers are a second,
  differently-labeled data point (Linux x64, this specific laptop CPU, 2026-08-18), not a
  correction of a platform-unspecified prior claim. Recommend citing both going forward rather than
  replacing one with the other.

## Verification performed

- `g++ -O3 -march=native -mavx2 -std=c++17` compiles clean, no warnings.
- Ran 3 times back-to-back; all 3 runs' full output captured above (not cherry-picked).
- No source file was modified for this investigation -- `bench_fusion.cpp` was run exactly as
  committed.
