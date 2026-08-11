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

**Phase 3 (throughput at equivalent bit-width) — COMPLETE, script fixed
2026-08-11 (second pass, same session):**
The original run (documented in the prior revision of this section) only
ever measured the ternary side and printed "NEEDS INT2/INT4 REFERENCE FOR
COMPARISON" without ever building one — plus a real bug: sizing
`np.uint8` arrays as `bytes / 0.25` actually allocates 1 byte per element
in that dtype, so the "1GB" test was silently 4× over budget (this is
what drove the ~7.5GB RSS spike observed during that run). Both are fixed
in `benchmarks/python-with-interpreter-overhead/bench_competitive.py`
(see its Phase 3 docstring for full rationale). The fixed version measures
FOUR representations, each genuinely occupying ~1GB:

| Representation | Elements | GOPS |
|---|---|---|
| Ternary (engine native, 1 byte/trit) | 1.0B | **4.45** |
| Ternary (Dense243, ~1.6 bits/trit, compiled) | 5.0B trits | **0.34** |
| INT4 packed (2 lanes/byte, NumPy reference) | 2.0B | 0.035 |
| INT2 packed (4 lanes/byte, NumPy reference) | 4.0B | 0.035 |

INT2/INT4 are real bit-packed references (genuine unpack → saturating-add
→ repack per op in NumPy — no dedicated compiled kernel exists for them
here, same standard `bench_fair_baseline.py` already applies: "fastest
reasonable NumPy implementation of the same semantics"). Comparing
like-for-like density (Dense243 ≈1.6 bits/trit vs. INT2 exactly 2
bits/element), suite's own computed verdict: **"✓ Dense243 is 9.6×
FASTER than the INT2 NumPy reference at equivalent (~2-bit) density."**
This is the strongest new evidence from this session — it validates the
core thesis (a dedicated compiled ternary kernel beats naive sub-byte bit
manipulation in plain NumPy) using an honestly-built comparison instead of
a hand-waved one. Caveat: this is a NumPy-vs-compiled-C++ comparison, not
NumPy-vs-hand-tuned-SIMD-INT2/INT4 — a dedicated INT2/INT4 SIMD kernel
would likely close some of that 9.6× gap, so don't read this as "ternary
beats INT2/INT4 in general," only "beats the NumPy reference we can
actually build without one."

**Phase 4 (neural workload patterns) — COMPLETE, script fixed 2026-08-11
(second pass, same session):**
The original per-row Python loop (`for i in range(M): tc.tmul(...);
np.sum(...)`) mostly measured CPython dispatch overhead, not the engine's
GEMM capability. Fixed to call the compiled, AVX2-vectorized
`ternary_zero_skip_gemm.ZeroSkipWeights` kernel instead (sparse index
built once per weight matrix, outside the timed loop — realistic for
inference where weights are fixed and only activations vary), with every
result checked against the NumPy reference for correctness (max abs
error ≤ 9.92e-05 across all four shapes, float32-appropriate).

| Shape | Ternary (compiled) | NumPy | Speedup |
|---|---|---|---|
| Small MLP 512×512 | 0.192ms | 0.019ms | 0.099× |
| Medium 2048×2048 | 3.561ms | 0.734ms | 0.206× |
| Large 4096×4096 | 9.493ms | 2.839ms | 0.299× |
| Attention 8192×1024 | 5.269ms | 1.276ms | 0.242× |

Average **0.21×**, suite verdict still **"✗ TOO SLOW FOR AI"** — but this
number is now trustworthy: it's the actual compiled GEMM kernel,
correctness-verified, at batch=1 (single-token decode) against random
~33% -sparsity ternary weights. Two caveats worth testing before treating
this as final: (1) real trained-model sparsity is closer to ~40% (see
CLAUDE.md falsification notes on 3-adic sparsity in learned weights,
H14/TritNet) vs. the ~33% a uniform random {-1,0,1} draw produces here,
and (2) NumPy's BLAS is extremely well-optimized at batch=1 GEMV
specifically — larger batches change the ratio (tested informally during
development: batch 32 brought some shapes as high as 0.84×, still not a
consistent win). Bottom line unchanged in direction (ternary GEMM is not
yet competitive with NumPy/BLAS on this hardware) but now for the right
reason — a real kernel-vs-kernel comparison, not an interpreter artifact.

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
(memory efficiency ✓, throughput baseline measured but not comparative ⚠) —
the "Throughput at equivalent bit-width" criterion's checkbox status is
unchanged even though the underlying Phase 3/4 script gaps are now fixed,
because neither result crosses into an unambiguous win: Phase 3 shows
ternary beating a NumPy INT2 *reference implementation* (9.6×) but that's
not the same claim as beating a hardware-comparable INT2/INT4 kernel, and
Phase 4's compiled-GEMM number (0.21× avg) is still sub-parity with NumPy.
Both are now real, correctness-checked measurements rather than script
artifacts, which is the actual deliverable here — the criteria table
should be read as "honestly measured, not yet a validated advantage" for
these two, not "still broken."

## 4. Absolute Throughput — PARTIAL ⚠️

The 45.3 Gops/s peak / 39.1 Gops/s element-wise figures are Windows x64
(2025-11-28). Linux equivalents from the fair-baseline run (informal, this
machine): engine peaks around 44 Gops/s (tnot @ 100K, stable cells). A
dedicated bench_phase0-style run with statistical comparison against the
Windows baseline remains TODO before quoting Linux absolute numbers.

## 5. Verdict (to be finalized)

- Correctness: Linux x64 fully validated (tests + CI).
- Fair-baseline: measured and published with disclosed methodology.
- Competitive suite: all 6 phases now run, and the Phase 3/4 script gaps
  (missing INT2/INT4 reference, naive Python GEMM path) are fixed and
  re-run. 2/5 commercial-viability criteria formally validated. New
  evidence from the fixed phases: Dense243 beats a real NumPy INT2
  reference 9.6× at equivalent bit density (Phase 3, strong positive);
  the compiled zero-skip GEMM kernel is correctness-verified but still
  0.21× NumPy on this hardware at batch=1 (Phase 4, negative but now
  trustworthy). Neither flips a criterion to ✓ on its own, but both
  replace guesswork with a real number.
- Production status: remains "tests + CI validated" until section 4
  (absolute throughput vs Windows baseline) is also closed out.

**TritNet Phase 2B (same date, same platform):** GO — 4/4 binary ops ≥99%
exact with ternary weights (tadd 100%, tmul 99.5%, tmin 99.9%, tmax 99.9%),
trained on complete 59,049-sample truth tables; checkpoints and per-op
result.json in models/tritnet/phase2b/.

**IP chain:** OpenTimestamps manifests 20260811_121728 and 20260811_135314
(Bitcoin confirmation of the latter pending at time of writing).
