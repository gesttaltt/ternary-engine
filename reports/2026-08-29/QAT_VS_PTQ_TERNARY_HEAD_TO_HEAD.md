# QAT vs PTQ for ternary TinyLlama: the hypothesis holds, the criterion does not

**Date:** 2026-08-29 · **Platform:** Linux x64, AMD Ryzen 5 4500, NVIDIA
GeForce RTX 3050 (6GB, compute 8.6), CUDA 13.0 driver, PyTorch 2.5.1+cu121,
transformers 5.16.1 · **Author:** Ternary Engine Team

---

## 1. The question

`docs/planning/ROADMAP.md`'s accuracy-retention criterion ("<5% loss") has
defeated four post-training-quantization techniques, each failing in the
same direction rather than a better one:

| Technique | Perplexity | Scope |
|---|---|---|
| Naive per-tensor absmean | 12.780 → 89,100.682 (+697,074%) | 100% |
| Naive per-channel absmean | 12.780 → 132,590.254 (+1,037,361%) | 100% |
| GPTQ (Hessian error compensation) | 12.780 → 18,565.469 (+145,167%) | 100% |
| GPTQ + mixed precision (3+3 protected) | 12.780 → 14,285.862 (+111,681%) | 73% |

The roadmap's own decision rule: once mixed-precision PTQ fails at
full-model scale — it did, 2026-08-28 — "the real fix is QAT or training
from scratch", matching the observation that every published
ternary/1.58-bit success (BitNet b1.58 and similar) trains **with the
quantizer in the loop** rather than post-hoc-quantizing a converged
checkpoint.

**The hypothesis, stated so it can fail:** if the PTQ failure mode really
is *"3 discrete levels cannot represent what a converged checkpoint's
weights need"* rather than *"insufficiently compensated rounding error"*,
then letting gradients move the weights while the quantizer sits in the
forward pass should recover substantially more accuracy than any amount of
post-hoc rounding **at the same scope**. If QAT lands in the same range as
GPTQ, the hypothesis is wrong and ternary is simply not representationally
sufficient here.

---

## 2. Design: one variable, deliberately

`benchmarks/model_quantization/qat_tinyllama.py` mirrors
`quantize_tinyllama_gptq.py` structurally so the comparison isolates
exactly one thing.

Held identical: the block-sequential harness (a Catcher captures block 0's
real inputs once; blocks are processed in order on cached activations; each
block's post-quantization output becomes the next block's input, so error
compounds forward), the calibration and eval windows, the corpus, and —
critically — **the quantizer itself**: per-tensor absmean scale with
round-and-clamp to {-1,0,+1}·scale.

The single difference: whether the weights are **trained** under that
quantizer (QAT, via a straight-through estimator) or **rounded** into it
(GPTQ). Per block, the fp16 teacher output is captured *first*, the block's
`nn.Linear` layers are swapped for STE-backed ternary equivalents
initialized from those same fp16 weights, and Adam minimizes MSE against
the teacher. Only one block is ever on the backward graph — that is what
makes this fit alongside a resident 1.1B model on a 6GB card.

`models/tritnet/qat_common.py`'s `TernaryLinearQAT` is credited as prior
art for the STE pattern but not reused directly: it thresholds on absolute
magnitude (default 0.3), tuned for TritNet's deliberately wide
`nn.init.normal_(std=1.0)` weights, which would zero essentially every
weight of a real transformer (std ≈0.02). The quantizer here has to be the
scale-aware absmean one regardless, to match the PTQ baseline.

---

## 3. Result: QAT beats PTQ decisively at matched scope

Every row below uses the same quantizer, the same windows, and the same
harness. Only "rounded" vs "trained" differs.

**Blocks 0–1 (14/154 layers), 8192-token eval, fp16 baseline 12.780:**

| Method | Perplexity | vs baseline | vs GPTQ |
|---|---|---|---|
| GPTQ (rounded) | 1,336.567 | +10,358% | — |
| **QAT, 200 steps/block** | **313.019** | +2,349% | **4.27× better** |
| **QAT, 600 steps/block, lr 3e-4** | **280.973** | +2,098% | **4.76× better** |

**Block 0 only (7/154 layers), 2048-token eval, fp16 baseline 10.320:**

| Method | Perplexity | vs baseline | vs GPTQ |
|---|---|---|---|
| GPTQ (rounded) | 78.847 | +664% | — |
| QAT, 20 steps/block | 35.541 | +244% | 2.22× better |
| **QAT, 200 steps/block** | **25.546** | +148% | **3.09× better** |

**The hypothesis is supported.** At identical scope, with an identical
quantizer, training the weights recovers 3–4.8× more accuracy than the best
available post-hoc rounding. This is the first technique tried against this
criterion that improves on its predecessor rather than failing in the same
direction — the previous four each landed at or worse than the one before.

**And the criterion is still missed by a wide margin.** 280.973 against a
12.780 baseline is +2,098%, not <5%.

---

## 4. The more interesting finding: block-local MSE is a bad proxy

The distillation objective is essentially solved, and it does not matter
nearly enough.

| Scope | final training loss | baked block-output MSE vs teacher | resulting perplexity |
|---|---|---|---|
| block 0, 200 steps | 0.00001 | 0.00002 | 25.546 (2.5× baseline) |
| blocks 0–1, 200 steps | 0.00005 | 0.00005 | 313.019 (24× baseline) |
| blocks 0–1, 600 steps | 0.00003 | 0.00003 | 280.973 (22× baseline) |

Each block's output is being reproduced to within ~1e-5 MSE of its fp16
teacher, and end-to-end perplexity is still 22–24× the baseline. Tripling
the training budget (200 → 600 steps) drives the loss down another ~40% but
buys only a 10% perplexity improvement (313.019 → 280.973) — clear
diminishing returns against an objective that is already nearly saturated.

**Reading:** the bottleneck at this point is not optimization effort and not
the quantizer. It is that *matching each block's output locally is not the
same as preserving the model's next-token distribution.* Tiny per-block
residuals compound across 22 blocks in a way the local MSE objective cannot
see, because no block is ever told what its error costs downstream.

This is a concrete, actionable result rather than a dead end: it says the
next experiment should change the **objective**, not the budget. End-to-end
QAT — backpropagating the actual language-modeling loss through the whole
quantized model — is the thing block-local distillation is a cheap
approximation of, and it is precisely what the published ternary successes
do. Spending more compute on block-local distillation would be spending it
in the wrong place.

---

## 5. Honest limits of this result

- **This is the weakest form of QAT**, chosen because it fits a 6GB card:
  block-local distillation against a teacher, not end-to-end training
  against the LM loss, and calibrated on 32 sequences of 512 tokens rather
  than a real training corpus. It should be read as a *lower bound* on what
  QAT can do, which makes the 3–4.8× margin over PTQ more notable, not less.
- **It does not show ternary is viable for this model.** It shows training
  beats rounding by a large factor while both remain far from usable.
- The blocks 0–1 rows above are the *controlled* head-to-head, because that
  is the scope where a directly comparable GPTQ number exists; §6 gives the
  full-model result.
- Same-environment caveat as always: perplexity here is only comparable at
  identical `seq_len` **and** `max_tokens` (see
  [PERPLEXITY_BASELINE_REPRODUCIBILITY.md](PERPLEXITY_BASELINE_REPRODUCIBILITY.md)).
  Every comparison in this document holds both fixed.

---

## 6. Full model: QAT beats every PTQ result by 6–8×

All 154 Linear layers ternarized, 200 steps/block, 8192-token eval,
19.1 minutes on the RTX 3050:

| Method | Layers ternarized | Perplexity | vs baseline |
|---|---|---|---|
| fp16 baseline | 0 / 154 | 12.780 | — |
| GPTQ | 154 / 154 | 18,565.469 | +145,167% |
| GPTQ + mixed precision (3+3 protected) | 112 / 154 (73%) | 14,285.862 | +111,681% |
| **QAT (block-local)** | **154 / 154 (100%)** | **2,237.038** | **+17,404%** |

QAT is **8.3× better than GPTQ** at identical 100% coverage, and **6.4×
better than GPTQ-with-mixed-precision** while ternarizing *more* of the
model (100% vs 73%) — mixed precision's whole advantage was leaving 27% of
the layers untouched, and QAT beats it without that concession.

**The error-compounding mechanism is directly visible** in the per-block
teacher-matching MSE, which grows monotonically with depth across the run:

| Block | 1 | 5 | 10 | 15 | 20 | 22 |
|---|---|---|---|---|---|---|
| baked MSE vs teacher | 0.00002 | 0.00017 | 0.00018 | 0.00070 | 0.00318 | 0.02668 |

A ~1,300× growth from first block to last, with every block getting the
same 200 steps and the same objective. Each block is handed an input that
is already degraded by all the quantization upstream of it, so the teacher
target it is chasing gets progressively harder to hit. This is the same
compounding the GPTQ session observed at 2-block scale ("super-additive"),
now measured continuously across all 22 — and it is exactly the effect a
block-local objective is blind to, since no block is told what its residual
costs downstream.

**Net: the roadmap's QAT hypothesis is confirmed in direction and
magnitude, and the criterion is still missed by orders of magnitude.**
Training beats rounding decisively and consistently — 3.1× at one block,
4.3–4.8× at two, 8.3× at the full model, with the margin *growing* as
scope grows. That is the opposite of the pattern the four PTQ techniques
showed, where each more sophisticated attempt failed in the same direction.
But 2,237.038 against a 12.780 baseline is not a usable model.

---

## 7. Reproduction

```bash
# Controlled head-to-head at blocks 0-1 (GPTQ reference: 1,336.567)
python3 benchmarks/model_quantization/qat_tinyllama.py --layers 14 --steps 200
python3 benchmarks/model_quantization/qat_tinyllama.py --layers 14 --steps 600 --lr 3e-4

# Block 0 only, 2048-token eval (GPTQ reference: 78.847)
python3 benchmarks/model_quantization/qat_tinyllama.py --layers 7 --steps 200 --max-tokens 2048

# Full model
python3 benchmarks/model_quantization/qat_tinyllama.py --steps 200

# The PTQ side of every comparison above
python3 benchmarks/model_quantization/quantize_tinyllama_gptq.py --device cuda --layers 14
python3 benchmarks/model_quantization/quantize_tinyllama_gptq.py --device cuda --layers 7 --max-tokens 2048
```

Result JSONs land in `benchmarks/results/model_quantization/`
(`tinyllama_qat_*.json`), which is gitignored — the tables above are the
durable record.
