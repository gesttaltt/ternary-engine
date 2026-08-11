# Linux x64 Validation Report

**Doc-Type:** Formal Validation Report · Version 1.0 (DRAFT) · 2026-08-11
**Platform:** Linux 7.0.0-28-generic x86_64, 8 cores, AVX2, Python 3.12.3, NumPy 2.5.1, GCC 13
**Status:** DRAFT — competitive suite results pending (running); this report
becomes the formal artifact required by the project standard for Linux
production claims once complete.

---

## Purpose

Per project standard (benchmark_everything, statistical_rigor, reports with
validation dates), production claims require a formal benchmark report in
`reports/`. The 2026-03-19 "Production-ready" Linux claim lacked one and was
downgraded on 2026-08-11 (commit a01924a). This report is the path back.

## 1. Correctness — COMPLETE ✅

| Suite | Result | Evidence |
|-------|--------|----------|
| Unified test runner (7 suites) | 7/7 pass | tests/run_tests.py, 2026-08-11 |
| Phase 0 correctness (50 cases) | pass | test_phase0.py |
| OpenMP scaling (25 cases) | pass | test_omp.py |
| Error handling / fusion | pass | test_errors.py, test_fusion.py |
| TritNet GEMM integration | pass | test_tritnet_gemm_integration.py |
| Dense243 (new 2026-08-11) | pass | test_dense243.py + 10/10 C++ tests |
| Zero-skip GEMM (new 2026-08-11) | pass | test_zero_skip_gemm.py |
| CI (GitHub Actions, ubuntu-latest) | green on every push since 34b697e | .github/workflows/ci.yml |

## 2. Fair Baseline vs NumPy — COMPLETE ✅

Methodology: same-semantics NumPy implementations, median of 100 repeats,
geomean over cells with CV ≤ 15% on both sides, preallocated outputs
favoring NumPy. Evidence: `benchmarks/results/fair_baseline_20260811_104629.json`.

| Group | Result |
|-------|--------|
| tadd (saturated add) | 1.7–3.5× vs NumPy |
| Single element-wise ops | 0.84× geomean (~parity, NumPy slightly ahead) |
| Fused tnot(op) | 1.43× geomean, up to 6× (tnot∘tadd) |
| Measurement caveat | 1M-element zone bimodal on this machine (turbo/OpenMP); excluded cells disclosed in JSON |

## 3. Competitive Suite (6 phases) — PARTIAL ⏳ (interrupted, resume next session)

Run started 2026-08-11, interrupted after Phase 1–2 complete and Phase 3
started (killed at user request to end the session; full raw output
preserved in `reports/2026-08-11/competitive_partial_20260811.log`).

**Two false starts before this data, both process-management errors, not
benchmark errors:**
1. First run was piped through `tail` inside a backgrounded shell — stdout
   buffered in the pipe and was never flushed to disk. Killed after the
   agreed 90-minute deadline with 0 recoverable lines (~91 min of compute
   lost to an instrumentation mistake).
2. Second run used `python3 -u` to an unbuffered log file — this one
   worked (below) but was itself killed early when the user ended the
   session, this time with output correctly salvaged.

**Phase 1 (arithmetic vs NumPy INT8) — COMPLETE:**
Speedup range 0.30×–0.90× across sizes/ops, average add 0.63×, average
mul 0.68×. Suite's own verdict: **"✗ NEEDS WORK"**. Consistent with the
fair-baseline finding (single-op geomean 0.84×, NumPy ufuncs are already
single AVX instructions) — two independent methodologies agree the
engine does not beat NumPy on raw single-op throughput.

**Phase 2 (memory footprint) — COMPLETE:**
Ternary vs INT8: **4.0×** advantage. Ternary vs INT4: **2.0×**. Dense243
vs INT4: **2.5×**. Suite's own verdict: **"✓ SIGNIFICANT ADVANTAGE"**.
This is the strongest validated claim in the whole suite and matches the
README's memory-density claims.

**Phase 3 (throughput at equivalent bit-width) — STARTED, not completed.**
Setup logged (1GB footprint: ternary/INT2 4B elements, INT4 2B elements)
before interruption.

**Phase 4–6 — NOT RUN.** Phase 5 in this script is a descriptive framework
only (prints strategy, no actual quantization); real measurement needs
`bench_model_quantization.py` (requires `transformers` + model downloads,
not yet run on Linux). Phase 6 needs root for RAPL (`energy_uj` permission
denied as non-root on this machine).

**TODO next session:** resume via
`python3 -u benchmarks/python-with-interpreter-overhead/bench_competitive.py
> <logfile> 2>&1 &` (always unbuffered + real file, never piped through
`tail` in a background shell), let Phase 3–6 complete, fold results in here.

## 4. Absolute Throughput — PARTIAL ⚠️

The 45.3 Gops/s peak / 39.1 Gops/s element-wise figures are Windows x64
(2025-11-28). Linux equivalents from the fair-baseline run (informal, this
machine): engine peaks around 44 Gops/s (tnot @ 100K, stable cells). A
dedicated bench_phase0-style run with statistical comparison against the
Windows baseline remains TODO before quoting Linux absolute numbers.

## 5. Verdict (to be finalized)

- Correctness: Linux x64 fully validated (tests + CI).
- Fair-baseline: measured and published with disclosed methodology.
- Production status: remains "tests + CI validated" until sections 3 and 4
  are complete and reviewed.

**TritNet Phase 2B (same date, same platform):** GO — 4/4 binary ops ≥99%
exact with ternary weights (tadd 100%, tmul 99.5%, tmin 99.9%, tmax 99.9%),
trained on complete 59,049-sample truth tables; checkpoints and per-op
result.json in models/tritnet/phase2b/.

**IP chain:** OpenTimestamps manifests 20260811_121728 and 20260811_135314
(Bitcoin confirmation of the latter pending at time of writing).
