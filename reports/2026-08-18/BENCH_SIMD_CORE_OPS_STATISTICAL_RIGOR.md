# bench_simd_core_ops.py statistical-rigor upgrade

**Date:** 2026-08-18
**Platform:** Linux x64, AMD Ryzen 5 7520U, 8 logical CPUs, AVX2 (shared, non-isolated dev container)
**Scope:** `benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py` only
**Status:** Methodology upgrade complete. **Not** a re-validation of the "35,042 Mops/s peak
throughput" headline figure (that number is a Windows x64 claim per CLAUDE.md's platform-support
policy; this session's numbers are Linux-local and are reported here only to demonstrate and
validate the new methodology, not to supersede or reproduce that figure).

## Why

CLAUDE.md gap #8 ("`BenchmarkRunner` unused") was closed for `bench_fair_baseline.py` and
`bench_simd_fusion_ops.py` earlier in this session's parent thread, but `bench_simd_core_ops.py`
was deliberately left untouched at the time: it's the source script for this project's headline
throughput claim, and this doc's own "Modifying Hot Paths" rule requires a dated,
platform-validated comparison before changing a cited metric's methodology -- not something to
fold into an unrelated dedup pass. This is that dedicated follow-up session.

## What changed

The old `benchmark_operation()` bracketed **one** block of `MEASURED_ITERATIONS=1000` back-to-back
calls with a single `perf_counter_ns()` pair and divided -- a single sample. No variance, CV, or
confidence interval was ever computable from that design, no matter how large the block was: a
lone block can't distinguish "consistently fast" from "fast on average but drifted mid-run."

New design: `BATCH_ITERATIONS=200` calls form one timed block (same amortization purpose the old
block served), and `MEASUREMENT_RUNS=10` independent blocks are each timed separately. Per-call
times across all 10 blocks feed `compute_timing_statistics()` -- the same statistics engine
`bench_fair_baseline.py` and `bench_simd_fusion_ops.py` now use (promoted out of
`BenchmarkRunner._compute_statistics` earlier this session) -- giving median, stdev, CV%, and a
95% CI on every reported cell for the first time. A `[WARN]` block, matching
`bench_fair_baseline.py`'s convention, now flags any cell with CV > 15% instead of silently
reporting it as trustworthy.

The JSON schema is additive-only: all 7 original `results_optimized`/`results_baseline` keys
(`operation`, `size`, `time_ns_total`, `iterations`, `time_ns_per_op`, `time_ns_per_elem`,
`throughput_mops`) are unchanged in meaning, so the three confirmed downstream consumers
(`benchmark_validator.py`'s `extract_performance()`, `bench_regression_detect.py`'s
`compare_benchmarks()`, `run_all_benchmarks.py`'s file glob) keep working without modification --
verified by actually running both tools against a real before/after pair (see below), not just by
inspection.

## The finding: this shared dev container is far noisier than the old methodology could show

Running the full (non-`--quick`) suite exposed real, substantial block-to-block variance that the
single-block design was structurally incapable of surfacing:

| Size | Worst CV this run | Note |
|---|---|---|
| 32 - 100,000 | ≤ 4.4% | Stable |
| 1,000,000 | **114.2%** (tmax) | tadd 28.7%, tmul 38.1%, tmin 72.8%, tnot 20.1% |
| 10,000,000 | 18.2% (tmin) | tadd 15.6% also over threshold |

At 1,000,000 elements specifically, every one of the 5 ops exceeded the 15% CV warning threshold
in this run, with `tmax` reaching 114% -- meaning the block-to-block spread is larger than the
median itself. Two independent `--quick` runs of the *same* script, minutes apart on the *same*
idle-looking machine, produced peak-throughput figures at 100,000 elements of 9,190 Mops/s and
18,740 Mops/s respectively -- a 2x swing with no code change in between, consistent with
contention in this shared, non-isolated container (background load, thermal/frequency-scaling
state, or OS scheduler noise) rather than any real performance difference.

This is exactly the failure mode the old methodology could never have shown: a single block,
however large, silently blends whatever happened during that one measurement window into one
number with no way to tell "clean measurement" from "measurement caught mid-contention." The new
design surfaces it automatically via the `[WARN]` block, per-cell.

**Consequence for `bench_regression_detect.py`**: comparing an old-methodology run against a
new-methodology run at 1,000,000 elements shows apparent "improvements" of up to +129% (tnot) and
+90% (tmax) -- these are **not** real speedups. They're the visible symptom of the same
high-variance cells above: the old single block happened to land on a slower moment, the new
median-of-10 happened to land higher, and neither tool had any way to know that without the CV
figures the new methodology now provides. Anyone comparing runs across this methodology change at
1,000,000+ elements should look at the CV columns before trusting a delta.

## What this does NOT claim

- **Not** a re-validation of "35,042 Mops/s peak throughput" -- that figure is a Windows x64
  claim; nothing here was run on Windows.
- **Not** a performance regression or improvement -- no engine code changed, only the measurement
  script.
- **Not** a recommendation to cite any of the numbers in this report as a new headline --
  several of the most interesting cells (1,000,000 elements) are exactly the ones flagged
  unreliable by the tool's own new CV check, consistent with this project's own transparency rule
  (`SKEPTICAL_METRICS.md`): publish the finding, including the parts that are unflattering to the
  measurement's own reliability, rather than quietly re-running until a clean number appears.

## Verification performed

- `py_compile` clean.
- `--quick` and full (all `TEST_SIZES`, up to 10,000,000 elements) runs both complete without
  error or timeout.
- JSON schema spot-checked: all 7 original keys present with unchanged meaning, 5 new keys
  additive (`measurement_runs`, `cv_percent`, `stdev_ns_per_op`, `throughput_mops_ci_lower/upper`).
- `bench_regression_detect.py` run against a real pre-refactor JSON vs. a real post-refactor JSON
  -- completed without error, correctly compared all 20 (op, size) cells.
- `benchmark_validator.py` run against the same pair with `--baseline`/`--current` -- completed
  without error, correct PASS/FAIL verdicts, report + JSON both generated.
- `tests/run_tests.py`: 15/15 (this script isn't wired into that suite; unaffected either way).
