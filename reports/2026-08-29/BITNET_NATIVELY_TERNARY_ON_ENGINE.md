# Running a natively-ternary LLM on the engine

**Date:** 2026-08-29 · **Platform:** Linux x64, AMD Ryzen 5 4500 (Zen 2),
OpenBLAS 0.3.31 · **Model:** `1bitLLM/bitnet_b1_58-large` (0.7B, the public
BitNet b1.58 reproduction) · **Author:** Ternary Engine Team

---

## 1. The reframing

Every prior quantization experiment in this repo tried to **manufacture** a
ternary model by crushing an already-converged fp16 checkpoint
(TinyLlama-1.1B). Five techniques, all failing in the same direction:

| Technique | vs 12.780 baseline |
|---|---|
| Naive per-tensor absmean | +697,074% |
| Naive per-channel absmean | +1,037,361% |
| GPTQ | +145,167% |
| GPTQ + mixed precision | +111,681% |
| Block-local QAT | +17,404% |

The project's own recorded conclusion was that published ternary successes
train **from scratch** with the quantizer in the loop. That reframes what
this repo should be asking:

> The engine's job is not to invent a good ternary model.
> It is to run one correctly and quickly.

Criteria 1, 2, and 3 are all **engine** properties, and all three pass.
Criterion 5 (accuracy) is a **model** property. So this session took a model
that was genuinely trained ternary and asked whether the engine can run it.

New: `benchmarks/model_quantization/run_bitnet_on_engine.py`, plus BitNet's
real shapes added to `bench_inference_latency_fp16.cpp`.

---

## 2. Verified first: the checkpoint's weights are NOT ternary on disk

This mattered and was checked rather than assumed. Layer 0's `q_proj` holds
**25,824 distinct values** in fp32. The 1bitLLM release ships *latent*
full-precision weights; BitNet's `BitLinear` quantizes them in the forward
pass.

The deployed ternary weights therefore have to be materialized by applying
BitNet's own quantizer — per-tensor absmean scale, round, clamp to
{-1, 0, +1}. That is the *same functional form* this project already uses.
The difference that matters is that **this model was trained under it**, so
materializing it is a readout of what the model already is, not a
post-training quantization of something that never expected it.

After materialization, every sampled projection has exactly **3 unique
values**, as it must.

---

## 3. Result A — the engine is faithful

`ternary_zero_skip_gemm.DenseWeights` was checked against a float reference
matmul of the same ternary weights, across 3 layers × 7 projections × 3
batch sizes:

**63/63 cells agree, worst relative deviation 2.21e-06.**

### A correction to my own first attempt, recorded rather than quietly fixed

The first version of this script asserted **bit-exact** agreement, on the
reasoning that ternary × fp32 products are exactly representable. The
products are — but the *sum* of 1,536–4,096 fp32 terms is order-dependent,
and the engine accumulates N-vectorized/M-blocked while the reference does
not. That version reported **0/63 bit-exact** with a worst deviation of
2.2e-06, which is the *correct* numerical outcome misread as a failure.

The test is now a tolerance appropriate to fp32 accumulation over K terms
(`64·K·eps`). A genuine kernel defect — wrong stride, dropped term, sign
error — lands orders of magnitude above that, not marginally outside it.

(Also fixed on the way: a `numpy.bool_` leak into `json.dump`, the same bug
class this project already fixed once in `test_falsification.py`, CLAUDE.md
v1.23.0.)

---

## 4. Result B — 35.2% zeros, an independent check on a long-standing claim

CLAUDE.md has asserted "**40% of products are ZERO**" since 2025-12-30,
derived from synthetic/experimental data.

Measured across **84,934,656 real weights** of a trained ternary LLM:

| | zero fraction |
|---|---|
| per-projection range | 33.0% – 40.1% |
| **overall** | **35.2%** |

The claim holds in kind and is slightly optimistic in degree: the real
figure is ~35%, not ~40%, with 40% reached only by the most sparse
individual projections (`k_proj` in the middle layers). Zero-skip
optimization remains viable; the headline number should be stated as
"~⅓ of weights are zero" rather than 40%.

This also independently corroborates the reasoning in
`ternary_gemm_dense.h`: at ~35% zeros, a CSC/CSR sparse index (4B index +
1B sign per non-zero) costs more memory than the dense int8 array, which is
exactly why the dense-packed kernel beat the zero-skip kernels.

---

## 5. Result C — speed at BitNet's real shapes

**Measured natively**, not through Python. The first attempt timed the
engine through pybind11 and produced visibly untrustworthy numbers — the
same shape gave 0.177 ms and 5.97 ms in different rows, because the engine's
OpenMP pool and NumPy's OpenBLAS pool oversubscribe each other, with CV
26–72% on this shared desktop. That violated the project's own
`ffi_isolation` rule as well. Those numbers were discarded, not reported,
and the Python script no longer measures speed at all.

BitNet's shapes were instead added to
`benchmarks/cpp-native-kernels/bench_inference_latency_fp16.cpp`
(single-threaded both sides, interleaved timing, best-of `sgemm`/`sgemv`
baseline). Ternary ÷ best-BLAS; **< 1.0 means ternary is faster**:

| BitNet shape | batch=1 | batch=8 | batch=32 | batch=128 |
|---|---|---|---|---|
| q/k/v/o [1536×1536] | 1.101× | 0.747× | 1.376× | 1.784× |
| gate/up [1536×4096] | 1.267× | 0.737× | 1.346× | 1.763× |
| down_proj [4096×1536] | 1.175× | 0.734× | 1.419× | 1.824× |

With BitNet's shapes included, criterion 3 now stands at **28/28 cells
passing, mean 1.254×, worst 1.826×**.

Note BitNet's batch=1 ratios (1.10–1.27×) are slightly *worse* than
TinyLlama's (0.89–0.999×): these matrices are smaller, so OpenBLAS's `sgemv`
keeps more of the working set resident and ternary's bandwidth advantage has
less to bite on. Still comfortably inside the 2× bound, and still with the
same shape: ternary wins at batch 8, loses ground as batch grows.

---

## 6. What this does and does not establish

**Does:** the engine runs a real, trained-ternary LLM's weights correctly
(63/63 within fp32 tolerance) and within the latency criterion at that
model's own shapes (12/12 BitNet cells pass). The "~40% zeros" premise is
independently confirmed at ~35% on real trained weights.

**Does not:** this is weight-level and GEMM-level. A full forward pass was
not run — BitNet's `BitLinear` also quantizes *activations* to 8 bits
(`input_bits: 8` in its config), which this engine's fp32-activation kernels
do not implement, so end-to-end perplexity through the engine would require
building that path first. No perplexity number is claimed here.

**Honest position on criterion 5:** this work does not make the accuracy
criterion pass. It relocates it. "<5% retention" presumes quantizing an fp16
original, which is the thing that demonstrably fails. For a natively-ternary
model there is no retention to lose, and the meaningful question becomes
whether the engine is faithful to the reference — which it is. Whether to
restate criterion 5 in those terms is a project decision, not something this
report assumes.

---

## 7. Reproduction

```bash
python3 -c "from huggingface_hub import snapshot_download; \
  snapshot_download('1bitLLM/bitnet_b1_58-large', \
  allow_patterns=['*.json','*.model','*.safetensors'])"

python3 benchmarks/model_quantization/run_bitnet_on_engine.py
python3 benchmarks/cpp-native-kernels/build_inference_latency.py --run
```
