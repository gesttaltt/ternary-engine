# GPTQ on GPU, and the full-model mixed-precision PTQ answer

**Date:** 2026-08-28 · **Platform:** Linux x64, AMD Ryzen 5 4500, NVIDIA GeForce
RTX 3050 (6GB, compute 8.6), CUDA 13.0 driver, PyTorch 2.5.1+cu121,
transformers 5.16.1 · **Author:** Ternary Engine Team

---

## 1. Why this session happened

`docs/planning/ROADMAP.md`'s "Status Reality Check" gives an explicit,
ordered recommendation for the accuracy-retention criterion (the one
commercial-viability criterion that looks structural rather than merely
under-engineered):

1. **Cheap, low-risk, in progress:** mixed-precision PTQ at full-model
   scale. Marked *"Next step (not yet run)"*.
2. **If (1) also fails at full-model scale:** the real fix is QAT or
   training from scratch — explicitly deferred, in part because *"no GPU
   access has been available on any checked peer session."*

Two facts made step (1) actionable this session:

- **A CUDA GPU is available on this host.** The roadmap's deferral of the
  whole direction was written against the premise that none was. That
  premise is stale (an RTX 3050 is present and functional; TritNet Phase 4
  had already used it on 2026-08-17).
- **`quantize_tinyllama_gptq.py` was 100% CPU-bound** — no `.to("cuda")`
  anywhere in the file. That is the direct, mechanical reason step (1) had
  never been run: a full 154-layer pass costs ~2.5–3.5h on this machine's
  CPU, and the machine reboots often enough mid-run for that to be a real
  obstacle (the reason the script's checkpoint/resume machinery exists at
  all).

So the highest-value GPU work available was **not** a new GPU compute
backend — TritNet Phases 4 and 5 already closed that question, finding no
niche — but putting the existing GPTQ pipeline on the GPU in order to
unblock the roadmap's own next step.

---

## 2. What changed in the code

`benchmarks/model_quantization/quantize_tinyllama_gptq.py` gained
`--device {auto,cuda,cpu}`. **The quantization mathematics is unchanged;
only where it executes changed.** Beyond the obvious `model.to(device)` and
moving calibration/eval token tensors, three changes were necessary and are
worth recording because each was a real defect on the GPU path:

1. **`H` is allocated on the layer's own device.** The Hessian hook fires
   once per calibration sequence per layer; had `H` stayed on CPU, every
   one of those calls would have forced an activation round trip across
   PCIe, which would have dominated the CUDA path's runtime.

2. **The per-column `.item()` was removed from the GPTQ column loop.** The
   original `zero_count += int((q_col == 0).sum().item())` forced a
   GPU→CPU synchronization on *every column* — 5,632 of them for
   `down_proj` alone — stalling the pipeline on each one. Now accumulated
   into a device-resident `int64` and read back once at the end. Identical
   value, no per-column sync.

3. **Hessian ownership transfer instead of `.clone()`, plus per-block
   `empty_cache()`.** `self.H` is never read after `quantize()` consumes
   it, so cloning it momentarily doubled the largest Hessian
   (`down_proj` is 5,632 columns ≈ 127MB fp32) for no benefit. On a 6GB
   card shared with a live desktop session that headroom matters. A guard
   raises if `quantize()` is ever called twice, so the ownership transfer
   cannot silently produce a wrong answer.

`--device cuda` is an explicit request that **fails loudly** if CUDA is
unavailable rather than silently falling back to CPU. A CPU run mislabeled
as a GPU run would produce a meaningless timing comparison — the same
silent-degradation class this project has already fixed in
`bench_competitive.py` (mock arithmetic fallback) and
`test_falsification.py` (NumPy substituted for the real engine). The saved
JSON also records `device`/`device_name` so no result file can be
misattributed after the fact.

---

## 3. Verification: the GPU path was proven equivalent before being trusted

Per this project's own discipline — the same discipline that caught the
unfair early-vs-late timing comparison in TritNet Phase 3, and that
verified the block-sequential GPTQ rewrite bit-for-bit before accepting
its ~21x speedup — the GPU path was **not** trusted on the basis that it
ran. A controlled A/B was run at identical scope (`--layers 7`,
`--max-tokens 2048`, `--no-checkpoint`; block 0 only), changing nothing but
`--device`:

| Metric | CPU | CUDA | Agreement |
|---|---|---|---|
| baseline perplexity | 10.320309776 | 10.319646371 | 6.4e-05 rel |
| quantized perplexity | 79.558420929 | 78.847440200 | 8.9e-03 rel |
| max \|zero_fraction diff\| across the 7 layers | — | — | **1.8e-05** |
| max relative \|scale diff\| across the 7 layers | — | — | **1.4e-07** |
| calibration + quantization | 651.8s | 15.4s | **42.2x** |
| perplexity eval (2048 tokens) | 1387.0s | 1.1s | ~1314x |
| **end-to-end** | **3294s** | **17.1s** | **193x** |

`dead_cols=0` on all 7 layers in both arms.

**Honest caveat, stated rather than smoothed over:** the two arms are *not*
bit-identical. Quantized perplexity differs by 0.9% relative. This is
expected and has a specific mechanism: fp16 forward kernels differ between
CPU and GPU, and GPTQ chains each block's quantized output into the next
block's calibration inputs, so a small kernel difference compounds
forward. The quantities that represent *the quantizer's own decisions* —
`zero_fraction` and `scale` — agree to 5–7 significant digits, which is
what actually had to be verified. The eval-time speedup (~1314x) is
inflated by fp16 matmul being emulated and effectively single-threaded on
CPU; the **42.2x on calibration+quantization is the honest figure** for the
work this script exists to do.

**Confound recorded, not hidden:** an unrelated 3D game was running
throughout, consuming ~245% CPU and ~1.8GB VRAM. This inflates the CPU
arm's wall time and slightly depresses the GPU's. These are not clean-room
measurements. The order-of-magnitude conclusion is robust to it; the exact
ratios are not.

---

## 4. The result: mixed-precision PTQ at full-model scale FAILS

All runs below: `--device cuda`, fp16, 8192-token eval window, calibration
window [83630, 100014) — disjoint from the eval window, unchanged from
prior sessions.

| Run | Blocks protected | Layers ternarized | Perplexity | vs baseline | Wall |
|---|---|---|---|---|---|
| fp16 baseline | — | 0 / 154 | **12.780** | — | — |
| pilot: blocks 0–1 quantized | none | 14 / 154 (9%) | 1,336.567 | +10,358% | 21.7s |
| pilot: blocks 0–1 protected, 2–3 quantized | 0,1 | 14 / 154 (9%) | 656.236 | +5,035% | 22.0s |
| **full model, no protection** (control) | none | **154 / 154** | **18,565.469** | **+145,167%** | 239.5s |
| **full model, `--protect-first 3 --protect-last 3`** | 0,1,2,19,20,21 | **112 / 154 (73%)** | **14,285.862** | **+111,681%** | 174.7s |

**The roadmap's open question is now answered, and the answer is no.**
Mixed-precision PTQ does not reach the <5% accuracy-retention criterion at
full-model scale. It misses by roughly four orders of magnitude.

Two secondary findings, both worth stating precisely:

- **Protection helps, but far less than the pilot scale suggested.** At
  the 2-block pilot scale, protecting blocks 0–1 and quantizing 2–3
  instead improves perplexity 1,336.567 → 656.236, a **2.04x** gain. At
  full-model scale, protecting 6 of 22 blocks improves 18,565.469 →
  14,285.862, only a **1.30x** gain — while still leaving 27% of the
  model's Linear layers at fp16. The benefit of protection *shrinks* as
  coverage grows, which is the opposite of what a "protect the sensitive
  layers and the rest is fine" hypothesis predicts.
- **The direction of the early-block-sensitivity effect reproduces; the
  magnitude does not** (see §5).

---

## 5. ⚠ The roadmap's recorded GPTQ numbers do not reproduce in this environment

This needs flagging prominently, because it affects how the roadmap's own
table should be read.

`docs/planning/ROADMAP.md` records, from earlier the same day (v1.56.0):
an fp16 baseline of **7.172**, blocks 0–1 quantized giving **4,776.805**,
and blocks 0–1 protected / 2–3 quantized giving **261.189** (an ~18.8x
improvement from protection).

Re-running the **same script with the same flags** in this session gives
an fp16 baseline of **12.780**, and the corresponding pilots give
**1,336.567** and **656.236** (a 2.04x improvement from protection).

| Quantity | Roadmap (earlier 2026-08-28) | This session | 
|---|---|---|
| fp16 baseline, 8192 tokens | 7.172 | **12.780** |
| blocks 0–1 quantized (14/154) | 4,776.805 | **1,336.567** |
| blocks 0–1 protected, 2–3 quantized (14/154) | 261.189 | **656.236** |
| improvement from protection | ~18.8x | **2.04x** |

### 5.1 Follow-up investigation: the 7.172 baseline is not reproducible

The discrepancy was chased down rather than left as "probably an
environment difference". Three candidate explanations were tested; all
three are excluded.

**(a) Not a GPU artifact.** The §3 A/B shows CPU and CUDA agree on baseline
perplexity to 6.4e-05 relative within this session.

**(b) Not a `transformers` version difference.** The same measurement was
re-run under `transformers` 4.46.3, installed into an isolated directory
and shadowed via `PYTHONPATH` so the machine's own environment was left
untouched. The two major versions agree to ~0.1% everywhere:

| seq_len / max_tokens | transformers 4.46.3 | transformers 5.16.1 |
|---|---|---|
| **512 / 8192 (the script's own defaults)** | **12.796** | **12.780** |
| 512 / 2048 | 10.350 | 10.320 |
| 1024 / 8192 | 10.855 | 10.881 |
| 2048 / 2048 | 6.122 | 6.119 |

This is doubly excluded, because the committed script calls
`AutoModelForCausalLM.from_pretrained(..., dtype=...)`, and `dtype=` is a
**v5-only** kwarg -- 4.46.3 raises `TypeError: LlamaForCausalLM.__init__()
got an unexpected keyword argument 'dtype'` (it wants `torch_dtype=`). So
the earlier session must itself have run a v5-era `transformers`, which is
exactly where 12.780 is measured. The probe used for the table above was
made version-adaptive and reports the dtype the weights actually landed in,
so a silently-ignored dtype kwarg could not masquerade as fp16.

**(c) Not an eval-window difference.** Perplexity here is genuinely very
sensitive to `seq_len` -- longer blocks give more context per prediction,
so the number drops (12.780 at seq_len 512 vs 9.618 at seq_len 2048, same
8192-token window). That made "they used a different window" a serious
hypothesis. It does not survive: a scan of **54** `(seq_len, max_tokens)`
combinations (seq_len 128-2048, max_tokens 1024-16384) found **no
configuration within 0.236 of 7.172**; the closest were 6.936 (1536/3072)
and 7.436 (768/2048).

**And the code itself has not drifted.** `git log` shows only two commits
have ever touched this file -- the one that added it (`3edfd2d`, the
v1.56.0 commit whose changelog records 7.172) and this session's
(`401b397`). `SEQ_LEN = 512`, `DEFAULT_MAX_TOKENS = 8192`,
`CALIB_TOKEN_OFFSET`, `CALIB_SEQS`, `CALIB_SEQ_LEN`, `MODEL_NAME` and the
body of `compute_perplexity()` are byte-identical between the committed
version and the version measured here.

**Conclusion, stated at the strength the evidence supports:** running the
committed code with its documented flags yields a baseline of **12.78**,
robustly -- across two `transformers` major versions, across CPU and GPU,
and it independently agrees to 5 decimal places with the **fp32** baseline
(12.780) that the sibling script `quantize_tinyllama.py` established in an
earlier session. **The 7.172 figure cannot currently be reproduced from the
committed code in any configuration tested.**

This is *not* proof that the earlier run was wrong. Two explanations remain
open and are not testable after the fact:

- That session developed this script in stages (its own changelog describes
  a first per-layer implementation, later rewritten to block-sequential).
  Intermediate runs may have used code that differs from what was finally
  committed.
- The HuggingFace model cache on this machine was **empty** at the start of
  this session and TinyLlama had to be re-downloaded, so the exact model
  snapshot the earlier session used cannot be inspected.

**Consequence for the documentation:** the roadmap's GPTQ rows (7.172 ->
26.139, -> 4,776.805, -> 261.189) and the "**~18.8x better**"
mixed-precision claim derived from them should be treated as **not
reproducible**, rather than merely "measured in a different environment".
They are superseded by this session's same-environment, internally
consistent numbers. The *qualitative* finding they were used to support --
that early transformer blocks are disproportionately sensitive to ternary
quantization -- does reproduce here, at 2.04x rather than ~18.8x.

---

## 6. What this means for direction

Per the roadmap's own stated decision rule — *"If (1) also fails at
full-model scale: the real fix is QAT or training from scratch"* — step (1)
is now exhausted. Four PTQ techniques have now been tried and all four
failed in the same direction, not a better one:

| Technique | Result | Scope |
|---|---|---|
| Naive per-tensor absmean | +697,074% | 100% |
| Naive per-channel absmean | +1,037,361% (worse) | 100% |
| GPTQ (Hessian error compensation) | +145,167% | 100% |
| GPTQ + mixed precision (3+3 protected) | +111,681% | 73% |

This is consistent with the hypothesis already recorded in ROADMAP.md: the
failure mode is not "insufficiently compensated rounding error," it is that
3 discrete values cannot represent what an already-converged checkpoint's
weights need without retraining. Every published ternary/1.58-bit success
(BitNet b1.58 and similar) trains **from scratch** with QAT; none
post-hoc-quantizes a converged checkpoint the way all four attempts here
have.

**The blocker on step (2) has partly lifted.** The roadmap deferred QAT as
"not recommended as an opportunistic pivot inside this environment,"
citing the absence of GPU compute and unstable sessions. GPU compute is now
demonstrated to work here: this session ran a full 154-layer GPTQ pass in
**4.0 minutes** that would have cost hours on CPU. That does not make
training a 1.1B model from scratch tractable on a 6GB card — it is not —
but it does make **small-scale QAT experiments** tractable, and this
project already has proven QAT building blocks (`models/tritnet/
qat_common.py`'s `TernaryLinearQAT`, validated on TritNet's own ops).

**Recommended next step:** a scoped QAT feasibility experiment — take a
small transformer (or a few layers of TinyLlama), fine-tune with
`TernaryLinearQAT` in the loop on the GPU, and measure whether perplexity
recovers toward the fp16 baseline in a way PTQ demonstrably cannot. That is
a falsifiable question answerable in this environment, and it directly
tests the hypothesis the last four negative results all point at. It is
explicitly *not* a commitment to training a full model from scratch.

**Also worth doing, cheaply:** pin the `transformers` version and re-run
the two pilots to resolve §5 before either number set is cited further.

---

## 7. Reproduction

```bash
# A/B verification (§3) -- identical but for --device
python3 benchmarks/model_quantization/quantize_tinyllama_gptq.py \
    --device cpu  --layers 7 --max-tokens 2048 --no-checkpoint
python3 benchmarks/model_quantization/quantize_tinyllama_gptq.py \
    --device cuda --layers 7 --max-tokens 2048 --no-checkpoint

# The roadmap's step (§4)
python3 benchmarks/model_quantization/quantize_tinyllama_gptq.py \
    --device cuda --protect-first 3 --protect-last 3 --no-checkpoint

# Control, and the two pilot reproductions (§4, §5)
python3 benchmarks/model_quantization/quantize_tinyllama_gptq.py --device cuda --no-checkpoint
python3 benchmarks/model_quantization/quantize_tinyllama_gptq.py --device cuda --layers 14 --no-checkpoint
python3 benchmarks/model_quantization/quantize_tinyllama_gptq.py --device cuda --protect-first 2 --layers 14 --no-checkpoint
```

`--no-checkpoint` is used throughout: the checkpoint/resume machinery exists
for multi-hour CPU runs, and on GPU these runs take 20s–4min, where a stale
checkpoint from a different configuration is a larger risk than an
interrupted run.

Result JSONs (each records its own `device`, `device_name`,
`protect_first`/`protect_last`, and calibration/eval windows). Note these
live under `benchmarks/results/`, which is gitignored, so they are local
artifacts of this session rather than tracked files -- the tables above are
the durable record:

- `benchmarks/results/model_quantization/tinyllama_gptq_20260828_174829.json` — A/B, CUDA
- `benchmarks/results/model_quantization/tinyllama_gptq_20260828_184417.json` — A/B, CPU
- `benchmarks/results/model_quantization/tinyllama_gptq_20260828_184820.json` — full model, protect 3+3
- `benchmarks/results/model_quantization/tinyllama_gptq_20260828_185328.json` — full model, control
- `benchmarks/results/model_quantization/tinyllama_gptq_20260828_185404.json` — pilot, blocks 0–1
- `benchmarks/results/model_quantization/tinyllama_gptq_20260828_185441.json` — pilot, blocks 2–3
