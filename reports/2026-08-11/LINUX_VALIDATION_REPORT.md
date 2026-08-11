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

## 3. Competitive Suite (6 phases) — PENDING ⏳

Running 2026-08-11. Results to be inserted here:
- Phase 1: arithmetic vs NumPy INT8 — TBD
- Phase 2: memory efficiency (7B/13B/70B models) — TBD
- Phase 3: throughput at equivalent bit-width — TBD
- Phase 4: neural workload patterns — TBD
- Phase 5: model quantization — framework only in this suite; real
  measurement requires bench_model_quantization.py (transformers + model
  downloads; not yet run on Linux)
- Phase 6: power consumption — RAPL present but requires root; not measured

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
