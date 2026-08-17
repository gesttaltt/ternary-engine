# TritNet Phase 5 — 2026-08-17

**Scope:** First Phase 5 ("Learned Generalization") experiment from
`.claude/CLAUDE.md`'s TritNet roadmap, picked up immediately after Phase 4
closed with "TritNet's remaining practical case rests on Phase 5 —
capabilities a LUT structurally cannot offer — not on throughput." Phase 5
lists "explore approximate arithmetic" as its first bullet; this asks that
question in a falsifiable form: **are the three imperfect checkpoints'
errors (tmul 99.49%, tmin 99.89%, tmax 99.85%) structured, or statistically
indistinguishable from noise scattered uniformly across the input space?**

**Net result:** Structured, not noise — but not "graceful" either. Errors
cluster extremely strongly by input **sparsity** (chi-square p < 1e-100 for
all three imperfect ops, aggregate effect driven by well-populated bins, not
small-n artifacts), with the sparsity extremes (near-all-zero or
near-no-zero inputs) 2-40x more error-prone than the well-populated middle
of the distribution. Valuation depth clustering is present for tmin/tmax
(p < 1e-90) but not tmul (p = 0.31) — a real, reported split, not rounded
away. Margin analysis shows the model is usually **confidently** wrong when
it errs (mean margin -15 to -23 logits), not hovering near a decision
boundary — only 1-6% of wrong positions are near-misses (margin > -0.5).

New file: `models/tritnet/phase5_error_characterization.py`. Results:
`models/tritnet/phase5_error_characterization_results.json`.

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

"Discover novel ternary operations" and "research applications" (Phase 5's
other two roadmap bullets) remain unstarted.

---

**Reproduce:** `python models/tritnet/phase5_error_characterization.py`
(pure NumPy + SciPy, no GPU or PyTorch required; reuses
`models/tritnet/phase2b_export/` from Phase 3).
