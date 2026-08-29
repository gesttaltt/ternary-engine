# A recorded baseline that does not reproduce, and how that was established

**Date:** 2026-08-29 (continues the 2026-08-28 session) · **Platform:** Linux
x64, AMD Ryzen 5 4500, NVIDIA GeForce RTX 3050 (6GB, compute 8.6), CUDA 13.0
driver, PyTorch 2.5.1+cu121 · **Author:** Ternary Engine Team

**Scope note.** This document exists separately from
[reports/2026-08-28/GPTQ_GPU_ENABLEMENT_AND_FULL_MODEL_MIXED_PRECISION.md](../2026-08-28/GPTQ_GPU_ENABLEMENT_AND_FULL_MODEL_MIXED_PRECISION.md)
(§5.1 of which covers the same investigation) because a reproducibility
failure in a recorded result is a different kind of artifact from a
performance/accuracy session report, and someone looking for "why don't the
v1.56.0 GPTQ numbers reproduce" should not have to know it happened during
a GPU-enablement session to find the answer.

---

## 1. The discrepancy

`CLAUDE.md` v1.56.0 and `docs/planning/ROADMAP.md` record, for
`benchmarks/model_quantization/quantize_tinyllama_gptq.py`:

| Quantity | Recorded (v1.56.0) |
|---|---|
| fp16 baseline perplexity, TinyLlama-1.1B, WikiText-2 | **7.172** |
| blocks 0 quantized (7/154 layers) | 26.139 (+264%) |
| blocks 0–1 quantized (14/154 layers) | 4,776.805 (+66,502%) |
| blocks 0–1 protected, 2–3 quantized (14/154) | 261.189 (+3,542%) |
| improvement attributed to mixed precision | **~18.8x** |

Re-running the same script with its documented flags on 2026-08-28/29
yields a baseline of **12.780**, and pilots of **1,336.567** and
**656.236** (a 2.04x improvement from protection).

This is not a cosmetic difference. **Every degradation percentage in that
table is computed against the baseline**, and the "~18.8x better"
mixed-precision claim — which is the evidence ROADMAP.md cited when
recommending mixed precision as step (1) — is a ratio of two numbers that
both depend on it. A wrong baseline silently invalidates the whole row set.

---

## 2. What was tested, and what it excludes

The tempting answer is "different environment, moving on." That was the
first explanation offered (CLAUDE.md v1.57.0) and it was **wrong** — or at
least unverified, which for this project's purposes is the same thing.
Three hypotheses were tested.

### (a) GPU vs CPU — excluded

The GPU path was newly added in the same session, making it the obvious
suspect. A controlled A/B at identical scope, changing only `--device`:

| Metric | CPU | CUDA | agreement |
|---|---|---|---|
| baseline perplexity | 10.320309776 | 10.319646371 | 6.4e-05 rel |
| max \|zero_fraction diff\| over 7 layers | — | — | 1.8e-05 |
| max relative \|scale diff\| over 7 layers | — | — | 1.4e-07 |

Not the cause.

### (b) `transformers` version — excluded, twice over

`transformers` and `pyarrow` were absent from this machine at the session's
start and had to be reinstalled (5.16.1), so the earlier session
demonstrably ran *some* other version. That made this the leading
hypothesis.

It was tested directly, by installing 4.46.3 into a throwaway directory and
shadowing it via `PYTHONPATH` so the machine's own environment stayed
untouched:

```bash
pip install --target /tmp/tf4 "transformers==4.46.3"
PYTHONPATH=/tmp/tf4 python3 benchmarks/model_quantization/probe_perplexity_reproducibility.py --mode grid
```

| seq_len / max_tokens | transformers 4.46.3 | transformers 5.16.1 |
|---|---|---|
| **512 / 8192 (the script's own defaults)** | **12.796** | **12.780** |
| 512 / 2048 | 10.350 | 10.320 |
| 1024 / 8192 | 10.855 | 10.881 |
| 2048 / 2048 | 6.122 | 6.119 |

Agreement is ~0.1% everywhere — nowhere near the gap being explained.

The hypothesis is excluded a second, independent way. The committed script
calls `AutoModelForCausalLM.from_pretrained(..., dtype=...)`. `dtype=` is
the **transformers v5** spelling; v4.46.3 rejects it outright:

```
TypeError: LlamaForCausalLM.__init__() got an unexpected keyword argument 'dtype'
```

So the earlier session cannot have been running a v4-era `transformers` —
the script would not have started. It must have been on v5, which is
precisely where 12.780 is measured.

(The probe loads with either spelling and reports **the dtype the weights
actually landed in**, so the nastier variant of this failure — a version
that accepts an unknown kwarg, ignores it, loads at `config.json`'s dtype,
and lets the script's banner keep printing "fp16" — could not have hidden
here.)

### (c) Eval window — excluded

This was the strongest hypothesis, because perplexity from this formulation
really is very sensitive to `seq_len`: longer blocks give more context per
prediction, so the number falls.

| seq_len | 2048 tok | 4096 | 8192 | 16384 |
|---|---|---|---|---|
| 512 | 10.320 | 11.249 | **12.780** | 10.281 |
| 1024 | 7.942 | 9.297 | 10.881 | 8.699 |
| 2048 | 6.119 | 7.816 | 9.618 | 7.707 |

7.172 sits inside that range, so "they used a different window" was
plausible. A dense scan of **54** `(seq_len, max_tokens)` combinations
(seq_len 128–2048, max_tokens 1024–16384) settles it:

| seq_len | max_tokens | perplexity | \|diff\| from 7.172 |
|---|---|---|---|
| 1536 | 3072 | 6.936 | 0.236 |
| 1536 | 4096 | 6.936 | 0.236 |
| 768 | 2048 | 7.436 | 0.264 |
| 2048 | 16384 | 7.707 | 0.535 |

**No configuration lands within 0.236.** Not a windowing difference.

### (d) Code drift — excluded

`git log` shows exactly two commits have ever touched the file: `3edfd2d`
(which added it, and whose changelog records 7.172) and `401b397` (this
session's GPU work). `SEQ_LEN = 512`, `DEFAULT_MAX_TOKENS = 8192`,
`CALIB_TOKEN_OFFSET`, `CALIB_SEQS`, `CALIB_SEQ_LEN`, `MODEL_NAME`, and the
body of `compute_perplexity()` are byte-identical between the committed
version and the version measured here.

---

## 3. Conclusion

Running the committed code with its documented flags yields a baseline of
**12.78, robustly** — across two `transformers` major versions, across CPU
and GPU, and agreeing to 5 decimal places with the **fp32** baseline
(12.780) that the sibling script `quantize_tinyllama.py` established
independently in an earlier session.

**The 7.172 figure cannot be reproduced from the committed code in any
configuration tested.**

Stated at that strength deliberately: this is *not* proof the earlier run
was wrong. Two explanations remain, and neither is testable after the fact:

- That session developed the script in stages — its own changelog describes
  a first per-layer implementation, later rewritten to block-sequential —
  so intermediate runs may have used code differing from what was finally
  committed.
- The HuggingFace model cache on this machine was **empty** at this
  session's start and TinyLlama had to be re-downloaded, so the exact model
  snapshot the earlier session used cannot be inspected.

### Consequences

1. The v1.56.0 GPTQ rows and the **"~18.8x better"** mixed-precision claim
   are marked **not reproducible**, superseded by the same-environment
   numbers in the 2026-08-28 report.
2. The *qualitative* finding those numbers supported — early transformer
   blocks are disproportionately sensitive to ternary quantization — **does**
   reproduce, at **2.04x** rather than ~18.8x.
3. The headline conclusion is unaffected and, if anything, firmer:
   mixed-precision PTQ at full-model scale fails the <5% criterion by ~4
   orders of magnitude (12.780 → 14,285.862).

### Methodology note worth keeping

Perplexity numbers from these scripts are **only comparable at identical
`seq_len` AND `max_tokens`**. The 512→2048 `seq_len` swing alone moves the
baseline from 12.780 to 9.618 on the same corpus window — larger than many
of the effects these scripts are used to measure. Any future report quoting
a perplexity should quote both settings alongside it.

---

## 4. Reproduction

```bash
# Which library version, which kwarg, and what dtype the weights REALLY are
python3 benchmarks/model_quantization/probe_perplexity_reproducibility.py --mode version

# The (seq_len x max_tokens) sensitivity grid
python3 benchmarks/model_quantization/probe_perplexity_reproducibility.py --mode grid

# Does ANY window setting reproduce the recorded figure? (exit 1 = no)
python3 benchmarks/model_quantization/probe_perplexity_reproducibility.py --mode hunt --target 7.172

# Cross-version check, without disturbing the machine's own environment
pip install --target /tmp/tf4 "transformers==4.46.3"
PYTHONPATH=/tmp/tf4 python3 benchmarks/model_quantization/probe_perplexity_reproducibility.py --mode grid
```
