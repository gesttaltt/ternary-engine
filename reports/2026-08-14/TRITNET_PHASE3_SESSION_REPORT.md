# TritNet Phase 3 — 2026-08-14

**Scope:** Continue the TritNet roadmap from CLAUDE.md's Critical Gaps #2
("TritNet Phase 3 pending"). Went from weight-export blocked to Phase 3
complete: weight export, naive C++ inference engine, AVX2 vectorization, and
the decisive TritNet-vs-LUT benchmark — plus three rounds of user-requested
fairness review, one of which caught a real bug in the benchmark itself.

**Net result:** 11 commits, all on `main`. `tests/run_tests.py` (15/15) still
passes throughout — the C++-only work (headers, dev-utility test/benchmark)
isn't wired into that suite, matching this project's existing convention for
`tests/cpp/`, but the Python bindings added in §8 are. Every numeric claim
below was independently re-measured, not carried forward from an earlier
commit's message.

| Commit | Summary |
|---|---|
| `ccf4d72` | Unblock weight export — the real blocker was two disjoint model architectures, not a save-format mismatch |
| `69af832` | Naive/scalar C++ inference engine + first TritNet-vs-LUT benchmark |
| `1b0c6c6` | AVX2 vectorization of the forward pass |
| `416f261` | Fairness review (Python-overhead angle) — found and fixed a real redundant-computation bug, fixed a stale claim in CLAUDE.md |
| `cee9e8f` | Extended fairness check to tnot |
| `91f4d20` | Extended fairness check to tmul |
| `7b56d5c` | **Correction** — the fairness check itself was unfair; fixed and re-measured, conclusion partially reverses for the binary ops |
| `ed82c94` | This report, first draft (7 commits, tmax not yet independently confirmed) |
| `00fd5e3` | Completed the set — tmax confirmed under the corrected methodology (§7) |
| `a0a4482` | Python bindings for the inference engine (§8) + this report, updated |

---

## 1. Weight export was blocked by two disjoint architectures, not a format bug (`ccf4d72`)

CLAUDE.md described the blocker as a "checkpoint format incompatibility
between `tritnet_model.py` and `train_phase2b.py`'s saves." It was worse:
`models/tritnet/src/tritnet_model.py`'s `TritNetUnary`/`TritNetBinary`
(backed by `ternary_layers.TernaryLinear`, no bias, direct-regression
output, with a working `export_weights_to_numpy()`) is a **stale,
abandoned pipeline** — its only surviving checkpoint, `tritnet_tadd.tritnet`,
is 15.8% accurate, not the 100% the roadmap documents.

The real GO checkpoints (tadd 100%, tmul 99.5%, tmin 99.9%, tmax 99.9%,
confirmed by matching each `result.json`'s `best_acc` to CLAUDE.md's cited
figures) were trained by `train_phase2a.py`/`train_phase2b.py`'s own local
`TritClassifier`/`TernaryLinearQAT` — bias included, CrossEntropy
classification head, ReLU hidden layers, structurally incompatible with
`tritnet_model.py`'s classes. Worse: `train_phase2a.py` (tnot) **never
called `torch.save` anywhere** — the tnot GO checkpoint, documented complete
since Phase 2A, had never existed on disk.

**Fixed:**
- Added checkpoint save/resume to `train_phase2a.py` (mirroring
  `train_phase2b.py`'s pattern), re-ran it: 100% reproduced,
  `models/tritnet/phase2a/tnot/`.
- Wrote `models/tritnet/export_weights.py` targeting the real
  `TritClassifier` architecture directly, exporting all 5 ops' quantized
  int8 weights + biases to `models/tritnet/phase2b_export/<op>/*.npy`.
- Added `tests/python/test_tritnet_export.py` — pure-NumPy replay of the
  exported weights over the **full** input space (243/59,049 samples per
  op), verified bit-for-bit against each op's recorded checkpoint accuracy.
  Wired into `run_tests.py` (13 → 14 suites).

---

## 2. Naive C++ inference engine + first benchmark (`69af832`)

- `models/tritnet/inference/generate_weights_header.py`: codegen
  (NumPy-only, no PyTorch dependency) turning the exported `.npy` files into
  a single constexpr C++ header (`tritnet_weights.h`), matching this
  project's existing compile-time-LUT-generation convention
  (`src/core/algebra/ternary_lut_gen.h`).
- `models/tritnet/inference/tritnet_inference.h`: naive/scalar forward-pass
  engine for all 5 ops, one shared templated 3-layer-ReLU-argmax
  implementation instantiated per op. Operates on `ternary_algebra.h`'s
  `trit` type so it's directly comparable to 5× LUT calls for the same
  5-trit chunk.
- `tests/cpp/test_tritnet_inference.cpp`: iterates the full input space per
  op and verifies the C++ engine's output matches each op's recorded
  checkpoint accuracy exactly (bit-for-bit sample count) — a cross-language
  check on the export, independent of the Python-side test.
- `benchmarks/cpp-native-kernels/bench_tritnet_inference.cpp`: the decisive
  experiment CLAUDE.md had been flagging as the outstanding Phase 3
  question.

**Result (Linux x64, AMD Ryzen 5 7520U, g++ 13.3.0 `-O3 -march=native`):
LUT wins by 950×–1776×** over the naive/scalar TritNet forward pass (tnot
265.1 vs 0.279 Mops/s; tadd 109.4 vs 0.072; tmul 126.7 vs 0.072; tmin 110.3
vs 0.072; tmax 129.0 vs 0.073 — all Mops/s of 5-trit-chunk operations).
Consistent with the ~20K-more-MACs-per-op cost `research/PRIOR_ART_TERNARY_LANDSCAPE.md`
already flagged.

---

## 3. AVX2 vectorization (`1b0c6c6`)

`models/tritnet/inference/tritnet_inference_avx2.h`: vectorizes across the
**output** dimension rather than the reduction dimension — weights are
stored `[IN][HID]` row-major, so for a fixed input index `i`,
`W[i][0:HID]` is contiguous. Broadcasts one input scalar at a time, does an
8-wide FMA against 8 contiguous output lanes, accumulating directly in the
output buffer (an outer-product GEMV pattern). Ternary int8 weights widened
to float via `_mm256_cvtepi8_epi32` → `_mm256_cvtepi32_ps`.

`generate_weights_header.py` was extended to zero-pad the output layer
(15 → `OUT_PADDED=16`) so the AVX2 path never needs a scalar tail; this
also caught a codegen bug the padding exposed (`%.9g` on an exact `0.0`
prints `"0"` with no decimal point, and `"0f"` isn't a valid C++ float
literal).

**Correctness:** AVX2 verified bit-identical to scalar over the full input
space (not just matching aggregate accuracy) in
`tests/cpp/test_tritnet_inference.cpp`.

**Result:** AVX2 gives a real **~10.2×–10.9× speedup over scalar**,
landing at the low end of this repo's usual ~10–30× AVX2 gain — but **LUT
still wins by 169×–195×** even against the vectorized path. AVX2 recovered
roughly 1 of the 3 orders of magnitude, exactly as predicted: TritNet's
practical case has to rest on Phase 4/5 (GPU/TPU batch throughput, or
learned generalization beyond what a LUT can express), not on beating a
LUT at this op width on a CPU.

---

## 4. Fairness review, round 1 — a real bug and a stale claim (`416f261`)

**User's question:** "Review if this is compared fairly against python,
remember python overhead."

**Literal answer:** no Python was involved — verified zero Python/pybind
references in the benchmark file. But the review applied the same *class*
of skepticism this project already learned the hard way from its own
retired "8,234× vs pure Python" claim (paying a cost repeatedly that a fair
comparison would pay once), and found a real analog:

`tritnet_inference_avx2.h`'s `layer_avx2()` reconverts the *same* int8
weights to float on **every single call**, even though weights are
invariant across calls. Isolated it with a diagnostic pre-converted-weights
variant (tadd, correctness-checked against the shipped AVX2 path first):
**~1.8× further speedup** (0.78 → 1.41 Mops/s at the time), narrowing
LUT's win from ~169× to ~93×.

**Separately, also found and fixed:** `CLAUDE.md`'s `core_innovation`
bullets still stated "8,234× average speedup over pure Python" as a live
headline number. `README.md` had already retired this exact claim with a
historical note calling it a strawman (`benchmarks/SKEPTICAL_METRICS.md`);
`CLAUDE.md` had never caught up. Fixed to match.

---

## 5. Extending the check — tnot and tmul (`cee9e8f`, `91f4d20`)

User asked to check tnot (different architecture: hidden=64 vs the binary
ops' hidden=128) and then tmul (weakest checkpoint, 99.5%, to rule out the
ratio depending on weight quality rather than architecture). Both initially
appeared to confirm the pattern: tnot ~1.72–1.74×, tmul ~1.82× (identical
to tadd). **These numbers turned out to be wrong — see §6.**

---

## 6. The correction — the fairness check was itself unfair (`7b56d5c`)

**User's question:** "check tmin too."

Checking tmin surfaced that the amortization check from §4–5 had the exact
same class of bug it was built to catch, one level deeper: each
`bench_amortized_X()` compared `baseline.avx2_mops` (measured **early**, in
the main table, at the start of the run) against a freshly-timed
`preconv_mops` (measured **much later**, after an increasingly long binary
— by this point several minutes — had already run sustained AVX2/FMA
load). Two numbers from different points in a long-running program are an
unfair comparison if the CPU's clock/thermal state drifts between them —
which it reproducibly did on this laptop chip.

**Fixed** with `time_best_interleaved()`: both the reconvert-every-call and
preconvert-once paths are now timed rep-by-rep, alternating, so both see
the same drift pattern within each of 5 repeats rather than being measured
in two separate blocks minutes apart.

**The result reverses for the binary ops:**

| Op | Previously claimed (§4–5) | Corrected (interleaved timing) |
|---|---|---|
| tadd / tmul / tmin | ~1.82× speedup, LUT win narrows to ~93–103× | **~0.94–0.95×, i.e. no benefit** (reproducible across repeated runs) — LUT win stands at ~150–190× |
| tnot | ~1.72–1.74× | **~1.58–1.74×, confirmed real** |

**Root cause, mechanistically confirmed** (via `lscpu`: 32KB L1d on this
machine): tnot's largest layer (hidden=64) is small enough that both its
int8 (4KB) and converted-float (16KB) forms fit in L1, so skipping the
redundant conversion is a pure compute win. The binary ops' largest layer
is 4× bigger (hidden=128): the int8 form (16KB) fits L1, but the
converted-float form (64KB) overflows 32KB L1 by 2×, forcing L2 traffic
that costs more than the cheap `cvt` instruction it was meant to save.
**Amortizing conversion trades compute for memory footprint — it only pays
off when the wider representation still fits cache.**

`CLAUDE.md` was corrected (not appended to) — the ~93–103× figures from
commits `416f261`/`91f4d20` are wrong and superseded.

---

## Final, corrected numbers (2026-08-14, Linux x64, AMD Ryzen 5 7520U)

| Op | LUT (Mops/s) | Scalar TritNet | AVX2 TritNet | AVX2 + amortized* | LUT wins by |
|---|---|---|---|---|---|
| tnot | ~265–535 | 0.27 | 2.73 | 4.72 (real, ~1.6–1.7×) | ~66–195× |
| tadd | ~110–150 | 0.07 | 0.79 | *no benefit* | ~150–190× |
| tmul | ~115–147 | 0.07 | 0.78 | *no benefit* | ~150–190× |
| tmin | ~112–153 | 0.07 | 0.78 | *no benefit* | ~150–190× |
| tmax | ~114–153 | 0.07 | 0.78 | *no benefit* (~0.89–0.92×, confirmed 2026-08-14 §7) | ~150–210× |

*Weight-conversion amortization; see §6 for why it only helps tnot.
LUT and AVX2 absolute throughput vary run-to-run (thermal/turbo state on a
laptop chip) — ranges above reflect the spread observed across this
session's repeated measurements, not single-run point estimates.

**Qualitative conclusion, unchanged by any of the above:** LUT wins by two
orders of magnitude at this operation width on CPU, naive or vectorized,
amortized or not. TritNet's practical case has to rest on Phase 4 (GPU/TPU
batch throughput) or Phase 5 (learned generalization beyond what a LUT can
express) — not on beating a LUT at 5-trit-chunk width on a CPU.

---

## What's left

- **Phase 4 (GPU/TPU batch inference)** — the natural next step, and the
  one place TritNet's structural advantage (a real batched GEMM, amortizing
  weight loads and exploiting parallelism a per-element LUT gather can't)
  could plausibly close the gap this session measured on CPU.

(Python bindings — the other item originally listed here — were wired up
same-day, §8 below.)

---

## 7. Completing the set — tmax (2026-08-14, same-day follow-up)

User asked to check tmax too, the one binary op §6 hadn't independently
confirmed. Added it to the corrected (interleaved-timing) benchmark,
correctness-checked against the shipped AVX2 path first (MATCH), then
re-run twice for stability: **~0.89–0.92×, no benefit** — consistent with
tadd/tmul/tmin. All 5 ops are now confirmed under the corrected
methodology: only tnot benefits from amortizing weight conversion; all 4
binary ops (sharing the same hidden=128 architecture and the same
L1-cache-overflow mechanism from §6) do not. This closes the "what's left"
item from the first draft of this report.

---

## 8. Python bindings (2026-08-14, same-day follow-up)

User asked to wire up Python bindings for the inference engine — the other
"what's left" item from the first draft of this report. Until now the
engine existed only as standalone C++ headers plus a dev-utility
test/benchmark, neither reachable from Python.

- `src/engine/bindings_tritnet_inference.cpp`: exposes `tnot`/`tadd`/`tmul`/
  `tmin`/`tmax` as `ternary_tritnet_inference`, batched over `[N, 5]` uint8
  trit-encoded numpy arrays. Runtime `has_avx2()` dispatch between the
  scalar and AVX2 engines per call — not a compile-time-only choice — so
  the module degrades gracefully on a CPU without AVX2 at *execution* time
  even though it's always *built* with AVX2 codegen (matching this
  project's "AVX2 required" baseline). Input validation reuses the existing
  `InvalidTritError`/shape-check conventions from the LUT-based bindings.
- `build/build_tritnet_inference.py`: mirrors `build_tritnet_gemm.py`'s
  structure and compiler flags.
- `tests/python/test_tritnet_inference_bindings.py`: verifies all 5 ops
  against the full input space (243/59,049 samples per op, a third
  independent cross-check alongside the C++ and pure-NumPy-export
  verifications from earlier in this report) plus input validation (wrong
  shape, mismatched batch size, invalid trit value). Wired into
  `run_tests.py` (14 → 15 suites).

Correctness: all 5 ops PASS, bit-for-bit matching each op's recorded
checkpoint accuracy, same as every other verification path in this report.

---

## Verification discipline used throughout

- Every correctness claim was checked against the full input space (243 or
  59,049 samples), not spot-checked — both cross-language (C++ vs the
  Python-recorded checkpoint accuracy) and cross-implementation (AVX2 vs
  scalar, bit-identical).
- Every amortization/fairness number was re-run at least twice before being
  reported, checking for run-to-run stability before drawing a conclusion.
- When a number *didn't* reproduce cleanly (tadd/tmul/tmin's amortization
  ratio, once tmin's check forced closer scrutiny), the response was to
  find the actual mechanism (interleaved timing, then the L1-cache-fit
  explanation, confirmed against this machine's real cache size via
  `lscpu`) rather than to average away the discrepancy or pick the more
  favorable number.
- `tests/run_tests.py` (14/14) re-run after every commit in this session to
  confirm no regression to the existing suite.
