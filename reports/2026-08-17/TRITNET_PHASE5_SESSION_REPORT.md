# TritNet Phase 5 — 2026-08-17

**Scope:** First two Phase 5 ("Learned Generalization") experiments from
`.claude/CLAUDE.md`'s TritNet roadmap, picked up immediately after Phase 4
closed with "TritNet's remaining practical case rests on Phase 5 —
capabilities a LUT structurally cannot offer — not on throughput." Two of
Phase 5's three roadmap bullets are addressed below: "explore approximate
arithmetic" (§1-4) and "research applications" (§5).

**Net result, §1-4 (approximate arithmetic):** Structured, not noise — but
not "graceful" either. Errors cluster extremely strongly by input
**sparsity** (chi-square p < 1e-100 for all three imperfect ops, aggregate
effect driven by well-populated bins, not small-n artifacts), with the
sparsity extremes (near-all-zero or near-no-zero inputs) 2-40x more
error-prone than the well-populated middle of the distribution. Valuation
depth clustering is present for tmin/tmax (p < 1e-90) but not tmul
(p = 0.31) — a real, reported split, not rounded away. Margin analysis
shows the model is usually **confidently** wrong when it errs (mean margin
-15 to -23 logits), not hovering near a decision boundary — only 1-6% of
wrong positions are near-misses (margin > -0.5).

**Net result, §5 (research applications):** Decisive negative. A trivial
GPU-native direct-arithmetic kernel (no weights, no training, no LUT — just
`tadd = clamp(a+b,-1,1)`, `tmul = a*b`, etc.) beats TritNet-GPU's
compute-only throughput by **46-57x** and is 100% exact on all 5 ops (vs
TritNet's 99.5-99.9% on 3 of them). There is no real GPU pipeline where
TritNet's learned forward pass is preferable to just computing these five
specific operations directly — the finding holds on CPU (Phase 3) and GPU
(Phase 4/5) alike.

New files: `models/tritnet/phase5_error_characterization.py`,
`models/tritnet/phase5_gpu_application_context.py`. Results:
`models/tritnet/phase5_error_characterization_results.json`,
`models/tritnet/phase5_gpu_application_context_results.json`.

---

## 1. Method

Per-op, over the **full input space** (243 samples for tnot, 59,049 for the
4 binary ops — the complete ground truth, not a sample), using the exported
weights from `models/tritnet/phase2b_export/` (same source Phases 3 and 4
already validated bit/decision-exact against the recorded checkpoint
accuracy):

- **Valuation depth** v3(x): reuses the *exact* convention already
  established in `research/scripts/falsify.py`'s `load_ultrametric()`
  (v3(n) = largest k such that 3^k | n), applied to each 5-trit operand read
  as a balanced-ternary integer `sum(t_i * 3^i)`. For binary ops, the
  "joint depth" used for binning is `min(v3(A), v3(B))`. A 5-trit vector
  gives v3 in {0,1,2,3,4}, plus a 6th "all-zero" bin standing in for that
  function's v3(0)=999 sentinel (5 trits can't reach 999).
- **Sparsity**: fraction of input trits equal to 0 (10 trits for binary ops,
  5 for tnot), binned into deciles.
- **Clustering test**: chi-square goodness-of-fit — is the wrong-count per
  bin proportional to that bin's share of the full input space, or not?
- **Margin**: for every wrong output-trit position, `true_class_logit -
  max(other_two_logits)` (negative by construction). Compared against the
  same margin computed on correct positions, for scale.

tadd and tnot (both 100% exact) are included as a **zero-error control** —
correctly report "0 errors, nothing to test" rather than a spurious
p-value.

## 2. Results

### tadd, tnot — control (0 errors, as expected)

Both 100% exact over the full input space; the script reports this
directly rather than manufacturing a statistic from an empty error set.

### tmul (99.4919%, 300/59,049 wrong)

| Metric | chi2 | p | Verdict |
|---|---|---|---|
| Valuation depth | 5.99 | 0.3069 | consistent with uniform |
| Sparsity (deciles) | 622.55 | 3.0e-128 | **non-uniform, strongly clustered** |

Sparsity decile 0 (least sparse, n=1024): error rate 5.27% (**10.4x**
overall). Decile 9 (most sparse, n=21): 14.29% (**28.1x**), though that
bin's n is small enough that the multiplier itself is noisy — the decile-0
result (n=1024, still 10.4x) is the well-powered version of the same
pattern. Middle deciles (2-5, n=8,064-15,360 each) sit at 0.54-0.83x
overall — genuinely *more* reliable than average, not just "less bad."

Margin: mean -15.04, median -10.89 (confidently wrong). Near-miss rate
(margin > -0.5): 2.31%.

### tmin (99.8882%, 66/59,049 wrong)

| Metric | chi2 | p | Verdict |
|---|---|---|---|
| Valuation depth | 493.33 | 2.2e-104 | **non-uniform, strongly clustered** |
| Sparsity (deciles) | 4115.21 | ~0 | **non-uniform, strongly clustered** |

Same U-shape as tmul, sharper: sparsity decile 0 (n=1024) 2.05% (18.4x);
decile 9 (n=21) 42.86% (383x, small-n caveat applies again but decile 0's
n=1024 result alone is a solid, well-powered confirmation of the same
shape). Valuation-depth bin 4 (n=8, i.e. joint-depth-4 operand pairs, the
rarest combinatorially) shows 25% error rate — real signal, but on a
sample small enough (8) that it's reported, not leaned on.

Margin: mean -21.24, median -11.68. Near-miss rate: 6.06% — the highest of
the three ops, but still means >93% of tmin's errors are confident, not
marginal.

### tmax (99.8510%, 88/59,049 wrong)

| Metric | chi2 | p | Verdict |
|---|---|---|---|
| Valuation depth | 437.09 | 3.0e-92 | **non-uniform, strongly clustered** |
| Sparsity (deciles) | 4219.28 | ~0 | **non-uniform, strongly clustered** |

Same pattern again: decile 0 (n=1024) 2.44% (16.4x), decile 9 (n=21) 47.6%
(320x). Margin: mean -22.95, near-miss rate 1.14% — the lowest of the
three, i.e. tmax's rare errors are its most confidently-wrong of the set.

## 3. What this means

**Sparsity is the dominant, universal signal.** All three imperfect ops
show the same qualitative shape: inputs near the sparsity extremes (almost
all zero trits, or almost none) are markedly harder than the well-populated
middle of the distribution, where the network is if anything *more*
reliable than its own headline accuracy suggests. This isn't an artifact of
tiny bins — decile 0 alone (n=1024, one of the larger deciles) shows a
10-18x elevated error rate for all three ops, independently of the noisier
extreme-tail deciles.

**Valuation depth is real for two of three ops, not one.** tmin and tmax
both show strong depth-clustering (p < 1e-90); tmul does not (p = 0.31).
Reported as found, not smoothed into a single "valuation depth predicts
errors" headline — the ternary-native-metrics framing CLAUDE.md mandates
doesn't mean every metric has to matter equally for every operation.

**Errors are confident, not marginal.** This is the most important
qualifier on "approximate arithmetic": if TritNet's mistakes were near-tied
decision-boundary calls, that would support a "graceful degradation"
reading — the network almost getting it right. That is not what happens.
93-98% of wrong positions have large negative margins (mean -15 to -23
logits against typical correct-position margins of +200 to +360) — the
network commits firmly to the wrong trit in a specific, structurally
identifiable region of input space (the sparsity extremes), rather than
hedging near a boundary. "Structured but not graceful" is the accurate
summary, not "fuzzy logic" in the sense Phase 5's roadmap language
gestures at.

## 4. Implication for the rest of Phase 5

This result narrows, rather than opens, the "approximate arithmetic"
angle: the errors are a *predictable, sparsity-linked blind spot* in these
specific checkpoints (plausibly a training-data-density effect — the
sparsity extremes are combinatorially rare corners of the full 59,049-row
truth table even though every row was included in training), not a
tunable knob toward useful fuzzy/probabilistic ternary logic. A
follow-up worth scoping later, not attempted here: does oversampling the
sparsity-extreme rows during training close this gap, and if so does
accuracy reach 100% the same way tadd/tnot already did? That would confirm
the density-effect explanation rather than a representational limit of the
architecture itself.

## 5. Research applications: does TritNet-on-GPU ever beat the obvious alternative?

Phase 4 compared TritNet-GPU against **CPU LUT** and found it short by
4-10x even in the best case. That's the wrong comparison for a "real GPU
pipeline" question, though: if an application already has its ternary data
resident on GPU — the only scenario where TritNet-GPU's 15-47x edge over
AVX2-CPU (Phase 4) would actually matter, since there'd be no CPU round
trip either way — the real competing implementation isn't a CPU LUT. It's
whatever a GPU-native version of the same operation looks like.

And tadd/tmul/tmin/tmax/tnot all have trivial closed forms: `tnot(a)=-a`,
`tadd(a,b)=clamp(a+b,-1,1)`, `tmul(a,b)=a*b`, `tmin/tmax` are literally
`min`/`max`. No lookup, no learned weights, no training step, and (unlike
3 of TritNet's 5 checkpoints) no accuracy loss at all — these are exact by
construction.

`models/tritnet/phase5_gpu_application_context.py` benchmarks that
direct-arithmetic GPU kernel against TritNet-GPU (Phase 4), same hardware,
same methodology:

| Op | TritNet-GPU compute-only (best) | Direct-arithmetic-GPU compute-only (best) | Ratio |
|---|---|---|---|
| tnot | 82.19 Mops/s | 3,809.08 Mops/s | **46.3x** |
| tadd | 44.58 Mops/s | 1,533.81-1,457.38 Mops/s (batch-dependent) | **~33-46x** |
| tmul | 44.58 Mops/s | 2,566.27 Mops/s | **57.6x** |
| tmin | 44.54 Mops/s | 2,564.56 Mops/s | **57.6x** |
| tmax | 44.58 Mops/s | 2,568.49 Mops/s | **57.6x** |

Correctness: 100.0000% exact for all 5 ops (n=243/59,049 each) — including
tmul, tmin, tmax, where TritNet tops out at 99.49-99.89%.

End-to-end (including H2D/D2H), direct arithmetic even closes most of the
gap to CPU LUT that TritNet-GPU couldn't: best case reaches 0.65-0.76x of
LUT throughput for the binary ops (vs TritNet-GPU's 0.25-0.27x) and 0.29x
for tnot (vs TritNet-GPU's 0.13x) — still short of beating the LUT outright
on this hardware, but a fundamentally different, much smaller gap, closed
by removing the neural network rather than by optimizing it.

**Answer to "research applications":** none exists, for these five specific
operations, on any hardware tested in Phases 3-5. The reason is structural,
not an implementation gap to optimize away: `tadd`/`tmul`/`tmin`/`tmax`/`tnot`
cost O(1) FLOPs per element in closed form, while TritNet's forward pass
spends ~240-5,376 ternary MACs computing the same answer (less accurately,
for 3 of the 5). No amount of batching, precision reduction, or kernel
fusion changes that ratio, because it isn't a throughput-tuning problem —
it's spending a neural network on a problem that never needed one. If
TritNet has a genuine niche, it is not as a replacement for exact
per-trit-chunk arithmetic; it would have to be for operations that lack a
cheap closed form in the first place — which is exactly Phase 5's other
still-unstarted bullet, "discover novel ternary operations."

---

**Reproduce:** `python models/tritnet/phase5_error_characterization.py`
(pure NumPy + SciPy, no GPU or PyTorch required; reuses
`models/tritnet/phase2b_export/` from Phase 3).
`python models/tritnet/phase5_gpu_application_context.py` (requires PyTorch
with CUDA; skips gracefully with exit 0 if unavailable).
