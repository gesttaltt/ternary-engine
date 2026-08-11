# Linux x64 Validation Report

**Doc-Type:** Formal Validation Report · Version 1.1 (DRAFT) · 2026-08-11
**Platform:** Linux 7.0.0-28-generic x86_64, 12 cores, AVX2, Python 3.12.3, NumPy 2.4.4, GCC 13.3.0
**Status:** DRAFT — competitive suite (§3) now fully run across all 6 phases;
this report becomes the formal artifact required by the project standard for
Linux production claims once §4 (absolute throughput vs Windows baseline) is
also closed out.

**Note on core count/NumPy version above:** this session's own environment
probe (`nproc`, `numpy.__version__`) reports 12 cores / NumPy 2.4.4, which
differs from the v1.0 draft's "8 cores / NumPy 2.5.1". Corrected here to the
directly-observed values; the discrepancy is unexplained (possibly a
different machine or stale note in the earlier session) and not itself a
finding.

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

## 3. Competitive Suite (6 phases) — COMPLETE ✅ (all 6 phases run, mixed results)

Phase 1–2 ran in the prior session (2026-08-11, interrupted mid-Phase-3).
Phase 3–6 resumed and completed in this session (2026-08-11), each run
individually via `--phase N` with the Bash tool's own `run_in_background`
(harness-tracked completion notification) rather than manual shell
backgrounding — avoiding both process-management failures below. Raw logs:
`reports/2026-08-11/competitive_phase{3,4,5,6}_20260811.log`; JSON results
under `benchmarks/python-with-interpreter-overhead/results/competitive/`.

**Two false starts in the prior session, both process-management errors,
not benchmark errors (kept for the record):**
1. First run was piped through `tail` inside a backgrounded shell — stdout
   buffered in the pipe and was never flushed to disk. Killed after the
   agreed 90-minute deadline with 0 recoverable lines (~91 min of compute
   lost to an instrumentation mistake).
2. Second run used `python3 -u` to an unbuffered log file — this one
   worked but was itself killed early when the user ended the session.

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

**Phase 3 (throughput at equivalent bit-width) — COMPLETE:**
1.0GB footprint (ternary/INT2: 4B elements @ 2 bits; INT4: 2B elements @ 4
bits). Ternary measured at **4.32 GOPS** (924.88ms/op, 4,324,907,808
elements/sec). Suite's own verdict: **"⚠ NEEDS INT2/INT4 REFERENCE FOR
COMPARISON"** — the script measures only the ternary side and never
actually benchmarks an INT2/INT4 baseline to compare against, so this
phase produces a number but not the comparison its own name promises.
Treat as a baseline data point, not a validated advantage claim.

**Phase 4 (neural workload patterns) — COMPLETE:**
Matmul speedup vs NumPy across 4 shapes (512×512 → 8192×1024): **0.06×,
0.20×, 0.35×, 0.11×**, average **0.18×**. Suite's own verdict: **"✗ TOO
SLOW FOR AI"**, with its own caveat that this path uses Python loops, not
the compiled zero-skip/TritNet GEMM kernels already built and validated
elsewhere in this repo (§1). This phase is measuring the wrong
implementation for a fair comparison — the 0.18× figure characterizes the
benchmark script's naive Python path, not the engine's actual GEMM
capability. Re-running Phase 4 against `ternary_zero_skip_gemm` or
`ternary_tritnet_gemm` directly (not through this script) would be needed
before drawing any conclusion about matmul competitiveness.

**Phase 5 (model quantization) — COMPLETE, descriptive only:**
Confirmed framework-only as documented in the prior draft: prints strategy,
target models (TinyLlama/Phi-2/Gemma), and success criteria; no
`quantize_to_ternary()` implementation, no actual model loaded or measured.
Zero new evidence. Real measurement still requires
`bench_model_quantization.py` (needs `transformers` + model downloads, not
run on this machine).

**Phase 6 (power consumption) — COMPLETE, descriptive only:**
Confirmed framework-only: prints target hardware platforms and metrics to
measure, no RAPL/nvidia-smi call attempted (unlike the earlier assumption
that it would fail on permissions — it doesn't try). Zero new evidence.

**Net effect on commercial viability criteria: still 2/5 validated**
(memory efficiency ✓, throughput baseline measured but not comparative ⚠).
Phases 3 and 4 as implemented in this script do not move the needle —
Phase 3 lacks a comparison baseline and Phase 4 benchmarks a Python
reference path instead of the engine's compiled kernels. Closing this gap
for real needs either fixing the script (add INT2/INT4 reference arrays to
Phase 3; call the compiled GEMM modules in Phase 4) or accepting the
existing fair-baseline (§2) and zero-skip GEMM correctness suite (§1) as
the more informative sources of truth on those two questions.

## 4. Absolute Throughput — PARTIAL ⚠️

The 45.3 Gops/s peak / 39.1 Gops/s element-wise figures are Windows x64
(2025-11-28). Linux equivalents from the fair-baseline run (informal, this
machine): engine peaks around 44 Gops/s (tnot @ 100K, stable cells). A
dedicated bench_phase0-style run with statistical comparison against the
Windows baseline remains TODO before quoting Linux absolute numbers.

## 5. Verdict (to be finalized)

- Correctness: Linux x64 fully validated (tests + CI).
- Fair-baseline: measured and published with disclosed methodology.
- Competitive suite: all 6 phases now run. 2/5 commercial-viability
  criteria validated (memory efficiency, and throughput baseline measured
  without a comparative reference). Phases 3–4 surfaced script limitations
  (missing INT2/INT4 reference, naive Python GEMM path) rather than
  engine limitations — worth fixing the script before re-quoting those
  numbers as engine performance claims.
- Production status: remains "tests + CI validated" until section 4
  (absolute throughput vs Windows baseline) is also closed out.

**TritNet Phase 2B (same date, same platform):** GO — 4/4 binary ops ≥99%
exact with ternary weights (tadd 100%, tmul 99.5%, tmin 99.9%, tmax 99.9%),
trained on complete 59,049-sample truth tables; checkpoints and per-op
result.json in models/tritnet/phase2b/.

**IP chain:** OpenTimestamps manifests 20260811_121728 and 20260811_135314
(Bitcoin confirmation of the latter pending at time of writing).
