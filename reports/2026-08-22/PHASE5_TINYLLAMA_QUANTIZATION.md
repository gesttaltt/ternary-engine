# Phase 5: Real Model Quantization — TinyLlama-1.1B, 2026-08-22

**Scope:** Direct follow-up to the same day's Phase 6 session. Picked Phase
5 (real model quantization) — the last deferred item from the original
recommendation menu. Scoped to TinyLlama-1.1B-Chat-v1.0 only (not the
originally-planned TinyLlama+Phi-2+Gemma-2B trio), per explicit user
direction: on a CPU-only sandbox, downloading and evaluating all three
wasn't worth it for a first real data point. Extended same day (§5) with
a per-channel quantization follow-up, per a second user choice among
three offered next steps.

## 1. What was built

`benchmarks/model_quantization/quantize_tinyllama.py` — a real,
end-to-end pipeline, not a framework description:

1. Loads TinyLlama-1.1B-Chat-v1.0 in fp32.
2. Measures baseline perplexity on the WikiText-2 raw test split (the
   standard corpus for this class of measurement — fetched directly via
   `huggingface_hub` + `pyarrow`, avoiding a `datasets` package
   dependency), 8,192 tokens across 16 non-overlapping 512-token blocks.
3. Quantizes every `nn.Linear` weight in the attention/MLP blocks
   (q/k/v/o/gate/up/down projections) to ternary in place, BitNet-style
   per-tensor absmean scale:
   `scale = mean(|W|)`, `W_ternary = round(clip(W/scale, -1, 1))`,
   dequantized back to float32 so the same unmodified model/forward pass
   measures the accuracy cost. Embedding and `lm_head` left at full
   precision (standard practice for this quantization class).
4. Re-measures perplexity on the identical token window.
5. Computes real memory footprint from the run's own measured parameter
   counts (not assumed model sizes).

Wired into `bench_competitive.py`'s Phase 5 (graceful skip if
`torch`/`transformers` unavailable, matching the Phase 4/6 pattern).

## 2. Results (Linux x64, CPU-only, this sandbox)

| Metric | fp32 baseline | Ternary (quantized) | Change |
|---|---|---|---|
| Perplexity (WikiText-2, 8,192 tokens) | 12.780 | 89,100.682 | +697,074% |
| Mean NLL (per-token cross-entropy) | 2.548 | 11.398 | — |
| Memory (quantized layers only, 968.9M/1.1B params) | 2,200.1 MB | 456.1 MB (Dense243) | 4.82× smaller |

- **Accuracy criterion (<5% loss): FAIL, decisively.** Not a marginal miss.
- **Memory criterion (<25% of FP16): PASS.** 456.1/2200.1 = 20.7%.
- **Zero fraction across the 154 quantized layers:** mean 0.330,
  range 0.312-0.537 — consistent with this project's own repeatedly-cited
  ~33-40% ternary sparsity figure, a real internal cross-check that the
  quantization function is behaving as expected, not producing degenerate
  output.
- **Latency ratio (quantized-simulated / fp32 forward pass): 0.982×** —
  i.e. no measurable speed difference, exactly as expected: both paths run
  the identical PyTorch float32 matmul ops; ternary values are dequantized
  back to float before the forward pass, so there is no reason for this
  number to differ. **This is not a measurement of this project's own
  ternary kernels** (`src/core/simd/ternary_gemm_dense.h`) — integrating
  those into an HF transformer's forward pass is a separate, much larger
  project, out of scope here. The kernel-speed question has already been
  answered independently (TritNet Phases 3-5, the 2026-08-20 GEMM work).

## 3. Sanity-checked before writing this up

A +697,074% perplexity increase is a large enough number to warrant
checking it isn't a bug before reporting it:

- **Both NLL values are finite** (2.548 baseline, 11.398 quantized) —
  `exp(11.398) ≈ 89,100` is the correct, deterministic consequence of a
  real (if catastrophic) cross-entropy increase, not a NaN/Inf explosion
  masquerading as a large finite number.
- **Zero fraction matches this project's own established ternary
  sparsity range** (~33-40%) across all 154 quantized layers — the
  quantization function is producing the expected statistical structure,
  not a degenerate all-zero or all-saturated result.
- **No embedding/lm_head accidentally quantized** — `nn.Embedding` isn't
  `nn.Linear` (skipped automatically by the `isinstance` check), and
  `lm_head` is explicitly excluded by name.

## 4. Why this happened — a real finding, not a surprise once examined

Per-tensor absmean ternary quantization applied uniformly to every
attention/MLP layer, **with no fine-tuning, calibration, or QAT**, is
known in the quantization literature to badly break an already-converged
pretrained checkpoint. The published techniques that make ternary/1.58-bit
weights work (BitNet, etc.) train the model **from scratch** with
quantization-aware training; naive post-training quantization of an
off-the-shelf fp16 checkpoint is a fundamentally different, much harsher
regime. This result is exactly what that literature would predict, not
an anomaly.

This directly informs this project's own framework text
(`bench_competitive.py`'s Phase 5 docstring, and `README.md`'s
"Quantization Strategy: Simple threshold-based") — as literally specified
(a per-tensor threshold, no further sophistication), that scheme is **not
viable** for real off-the-shelf model quantization.

## 5. Follow-up (same day): does per-channel scaling fix it?

The obvious next question — is the per-tensor result a *scheme artifact*
(too coarse a scale) rather than something more fundamental? Extended the
script to also quantize with **one scale per output row** (`nn.Linear`
weight shape is `[out_features, in_features]`, so this is per-output-
channel absmean scaling) instead of one scale for the whole matrix — the
natural next-cheapest refinement, still no calibration data or retraining.

**Result: per-channel scaling made it WORSE, not better.**

| Scheme | Perplexity | Mean NLL | Zero fraction | vs baseline |
|---|---|---|---|---|
| Baseline (fp32) | 12.780 | 2.548 | — | — |
| Per-tensor (one scale, whole matrix) | 89,100.682 | 11.398 | 33.0% (31.2-53.7%) | +697,074% |
| Per-channel (one scale per row) | **132,590.254** | **11.795** | 31.8% (31.1-46.7%) | **+1,037,361%** |

Sanity-checked the same way as the first result: both NLL values finite
(11.398, 11.795 — not NaN/Inf), and the per-tensor numbers reproduced
**exactly** (89,100.682, zero fraction 0.330) on a second, independent
run that reloaded the model fresh — confirms full determinism, not a
fluke of one run.

**This falsifies the natural hypothesis** ("coarser scale = the
problem") and sharpens the actual conclusion: granularity of the scale
is not the bottleneck. A plausible mechanism, offered as a hypothesis
rather than a proven cause (not independently verified further in this
session): per-tensor scaling uses one large, global scale dominated by
the average across all rows — for a row whose true weight magnitudes are
much smaller than that global average, dividing by the larger global
scale pushes nearly all of that row toward zero, effectively *pruning*
it (a mild, almost benign failure mode for a row that was already
contributing a small correction). Per-channel scaling instead recalibrates
every row to its own local statistics and forces even naturally
small/precise rows into a full ±(their own scale) ternary pattern — with
no bias or error-compensation step (unlike GPTQ/AWQ-style calibrated
quantization, which spreads rounding error to not-yet-quantized weights),
this can inject *larger* absolute error into channels the per-tensor
scheme would have simply zeroed out. Neither scheme has any calibration
data or retraining; the real bottleneck is that omission, not the choice
between per-tensor and per-channel.

## 6. What this does and doesn't claim

- Does NOT claim ternary quantization is unworkable in general — TritNet's
  own from-scratch-trained ternary networks reach ≥99% accuracy on their
  target operations (see `.claude/CLAUDE.md`'s TritNet section). This
  result is specific to *naive post-training* quantization of an
  *already-trained* checkpoint, a much harder regime than training ternary
  from scratch.
- Does claim: this project's own stated "Simple threshold-based"
  quantization strategy, applied literally with no further sophistication,
  fails badly and honestly on a real model — a genuine, previously-unknown
  (framework was never actually run before) data point for this project's
  commercial-viability story.
- Does claim, from the follow-up: the failure is not fixed, and is in
  fact made worse, by simply making the scale finer-grained — a real,
  informative negative result that rules out the cheapest possible
  refinement and points at calibration/retraining (QAT, GPTQ/AWQ-style
  error compensation, selective layer exclusion) as the actual next step,
  not attempted here (substantially larger undertaking, deliberately
  scoped out per this project's established discipline of flagging
  follow-up work rather than chasing every extension in one sitting).
- Memory footprint claim stands on its own regardless of the accuracy
  result: real ternary weights genuinely take 4.82× less space than fp16
  for the quantized fraction of a real model (identical for both schemes,
  since neither changes which weights are quantized), computed from this
  run's actual measured parameter counts.
- "2/5 commercial-viability criteria" (Phases 1-4) is unaffected — Phase 5
  was never part of that count; this is the first real data Phase 5 has
  ever produced (previously purely descriptive).

## Files changed

- `benchmarks/model_quantization/quantize_tinyllama.py` (new; extended
  same day to add the per-channel scheme and a 3-way baseline/per-tensor/
  per-channel comparison, reloading the model fresh per scheme to stay
  within this machine's 7GB RAM rather than holding multiple copies)
- `benchmarks/python-with-interpreter-overhead/bench_competitive.py`
  (Phase 5 wired to call the new script)

Raw results: `benchmarks/results/model_quantization/tinyllama_20260822_144000.json`
(supersedes the first run's `tinyllama_20260822_134719.json`, which only
has the per-tensor comparison — kept for reference, not deleted).

`tests/run_tests.py`: unaffected (neither file is wired into that suite —
both are benchmarks).
