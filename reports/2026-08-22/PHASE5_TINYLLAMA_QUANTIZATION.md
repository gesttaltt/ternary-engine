# Phase 5: Real Model Quantization — TinyLlama-1.1B, 2026-08-22

**Scope:** Direct follow-up to the same day's Phase 6 session. Picked Phase
5 (real model quantization) — the last deferred item from the original
recommendation menu. Scoped to TinyLlama-1.1B-Chat-v1.0 only (not the
originally-planned TinyLlama+Phi-2+Gemma-2B trio), per explicit user
direction: on a CPU-only sandbox, downloading and evaluating all three
wasn't worth it for a first real data point.

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
off-the-shelf fp16 checkpoint — especially with coarse per-tensor (not
per-channel) scaling and no outlier handling — is a fundamentally
different, much harsher regime. This result is exactly what that
literature would predict, not an anomaly.

This directly informs this project's own framework text
(`bench_competitive.py`'s Phase 5 docstring, and `README.md`'s
"Quantization Strategy: Simple threshold-based") — as literally specified
(a per-tensor threshold, no further sophistication), that scheme is **not
viable** for real off-the-shelf model quantization. A fairer/more
realistic result would need per-channel scaling, calibration data,
selective layer exclusion (e.g. skip the first/last transformer blocks,
commonly the most quantization-sensitive), or actual QAT/fine-tuning —
each a substantially larger undertaking, not attempted in this session
per the same scope discipline already applied elsewhere in this project's
history (flag follow-up work, don't chase every extension in one sitting).

## 5. What this does and doesn't claim

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
- Memory footprint claim stands on its own regardless of the accuracy
  result: real ternary weights genuinely take 4.82× less space than fp16
  for the quantized fraction of a real model, computed from this run's
  actual measured parameter counts.
- "2/5 commercial-viability criteria" (Phases 1-4) is unaffected — Phase 5
  was never part of that count; this is the first real data point Phase 5
  has ever produced (previously purely descriptive).

## Files changed

- `benchmarks/model_quantization/quantize_tinyllama.py` (new)
- `benchmarks/python-with-interpreter-overhead/bench_competitive.py`
  (Phase 5 wired to call the new script)

Raw results: `benchmarks/results/model_quantization/tinyllama_20260822_134719.json`

`tests/run_tests.py`: unaffected (neither file is wired into that suite —
both are benchmarks).
