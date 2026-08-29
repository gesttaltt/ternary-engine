# Criterion 3: ternary inference latency vs FP16 — passes where latency means something

**Date:** 2026-08-29 · **Platform:** Linux x64, AMD Ryzen 5 4500 (Zen 2, 6C/12T),
g++ 13.3.0 `-O3 -march=haswell -mavx2 -mfma -mf16c`, OpenBLAS 0.3.31
(NumPy-bundled, DYNAMIC_ARCH), single-threaded both sides ·
**Author:** Ternary Engine Team

---

## 1. Why this, and why now

`docs/planning/ROADMAP.md` → "Where To Continue" identified criterion 3
("Inference latency < 2× FP16") as **the only unvalidated
commercial-viability criterion whose work is possible in this environment**
— criterion 4 (power) is blocked because RAPL's `energy_uj` is root-only
here, and criterion 5 (accuracy) has now been thoroughly characterized as
failing across five quantization techniques.

It had never been measured. This closes it.

New: `benchmarks/cpp-native-kernels/bench_inference_latency_fp16.cpp` +
`build_inference_latency.py`. Native C++, no pybind11, per this project's
`ffi_isolation` rule.

---

## 2. The methodological problem, and how it was resolved

**There is no fast FP16 arithmetic on this CPU.** Zen 2 has F16C convert
instructions but no FP16 FMA; AVX512-FP16 does not exist here. A model
stored in FP16 is executed on CPU by upconverting to FP32 and running an
FP32 GEMM.

So the honest FP16-model latency on this CPU **is a well-optimized FP32
SGEMM**, and that is the baseline used. Three consequences, all deliberate:

- **The reference is OpenBLAS, not a hand-rolled loop.** A weak baseline
  would flatter the ternary kernel for the wrong reason. OpenBLAS 0.3.31
  with `DYNAMIC_ARCH` dispatches a Zen-appropriate microkernel.
- **Both sides pinned to one thread.** This measures per-core kernel
  quality, not whose threading strategy happens to scale better on 6 cores.
- **Weight packing is amortized on both sides.** `pack_ternary_dense()` runs
  once, matching the fact that BLAS also receives pre-laid-out weights.
  Weights are fixed at inference; charging ternary a per-call packing cost
  would be as unfair as charging BLAS a per-call transpose.

**The strawman was measured and rejected, not quietly avoided.** Literal
per-element FP16 arithmetic (F16C round-trips per multiply-add) costs
2.91 ms at M=1/K=2048/N=256, against ternary's 0.083 ms. Ternary would look
**35× better** against that number, versus **0.31×** against the honest FP32
baseline. Publishing the 35× would have been the same category error this
project already retired with its "8,234× vs pure Python" headline. It
appears in the output labelled as the rejected strawman.

### A fairness bug caught mid-session, worth recording

The first working version used `cblas_sgemm` as the baseline at **every**
batch size, including M=1. But at M=1 the operation is a GEMV, and OpenBLAS
has a dedicated `cblas_sgemv` kernel its general SGEMM path does not reduce
to. Timing ternary against the slower BLAS routine — in precisely the
regime where the criterion matters most — would have handed ternary a win
it did not earn.

The effect was large:

| batch=1 cell | vs `sgemm` (unfair) | vs `sgemv` (fair) |
|---|---|---|
| q_proj/o_proj | 0.409× | 0.999× |
| k_proj/v_proj | 0.296× | 0.749× |
| gate/up_proj | 0.420× | 0.950× |
| down_proj | 0.398× | 0.918× |

The unfair version would have supported a headline of *"2.4–3.4× faster
than FP16 at decode"*. The honest number is **parity to ~1.3× faster**. The
benchmark now takes the **better of `sgemm` and `sgemv`** as the baseline
and prints which one won each cell.

---

## 3. Results

TinyLlama-1.1B's real projection shapes. Ternary kernel correctness is
checked against a float64 reference before any cell is timed. Contenders
are timed **interleaved rep-by-rep** per this project's `interleaved_timing`
rule, so clock/thermal drift hits both equally.

Ratios are ternary ÷ best-BLAS; **< 1.0 means ternary is faster**. Ranges
are across 4 independent runs.

| Shape | batch=1 | batch=8 | batch=32 | batch=128 |
|---|---|---|---|---|
| q_proj / o_proj [2048×2048] | 0.935–0.999× | 0.720× | 1.486× | 1.812–1.909× |
| k_proj / v_proj [2048×256] | 0.503–0.749× | 0.499× | 1.177× | 1.647–1.684× |
| gate/up_proj [2048×5632] | 0.899–0.950× | 0.696× | 1.332× | 1.842–**1.996×** |
| down_proj [5632×2048] | 0.893–0.998× | 0.649× | 1.323× | 1.769–1.834× |

**Absolute latency, batch=1 (the autoregressive decode case):**

| Shape | ternary | best BLAS | FP16-stored + dequant/call |
|---|---|---|---|
| q_proj/o_proj | 0.65 ms | 0.66 ms | 3.70 ms |
| k_proj/v_proj | 0.081 ms | 0.108 ms | 0.29 ms |
| gate/up_proj | 1.80 ms | 1.90 ms | 10.04 ms |
| down_proj | 1.92 ms | 2.09 ms | 10.73 ms |

---

## 4. Verdict, stated at the strength the data supports

**Criterion 3 passes — but the honest statement is scoped by batch size,
not a flat "16/16".**

| Regime | Ratio range | Robust? |
|---|---|---|
| **batch 1–8** (autoregressive decode — what "latency" means) | **0.50–0.999×** | ✅ yes, ternary at parity or faster |
| **batch 32** (small prefill) | 1.18–1.49× | ✅ yes, comfortable margin |
| **batch 128** (throughput/prefill) | 1.65–**2.00×** | ⚠️ **marginal — brushes the threshold** |

Three of four runs report 16/16 cells passing; the worst cell
(gate/up_proj @ batch=128) measured **1.996×** in one run, and an earlier
run under the unfair-but-stricter `sgemm`-only baseline measured 2.026× on
k_proj@128. **The batch=128 result sits on the 2× line and is not robust
run-to-run.** It should not be quoted as a clean pass.

Since inference *latency* is fundamentally a low-batch property —
autoregressive decode processes one token at a time — the criterion is met
in the regime it exists to describe, with margin, and reproducibly. The
high-batch cells are a throughput regime where the criterion is borderline.

**Recommended wording for the criteria table:** validated for batch ≤ 32;
borderline (≈1.7–2.0×) at batch 128.

### Why the ratio degrades with batch, mechanistically

This matches `ternary_gemm_dense.h`'s own analysis. At batch=1 the GEMM is
memory-bandwidth-bound, and ternary moves **1 byte per weight against FP32's
4** — a 4× traffic advantage that dominates. As batch grows, arithmetic
intensity rises, the operation becomes compute-bound, and OpenBLAS's heavily
tuned FP32 microkernel (register blocking, software pipelining, Zen-specific
scheduling) wins on raw FLOP throughput against a comparatively simple AVX2
kernel. Ternary's structural advantage is bandwidth, so it shines exactly
where latency lives and fades where throughput lives.

**A genuinely favourable secondary case:** if a deployment stores FP16 and
cannot keep an FP32 copy resident, it pays dequantization every call —
3.70/0.29/10.04/10.73 ms against ternary's 0.65/0.081/1.80/1.92 ms, i.e.
**ternary is 3.6–5.6× faster**. Whether that applies depends on whether the
memory the FP32 copy would occupy is available, which is precisely the
constraint ternary exists to relieve.

---

## 5. Honest limits

- **This is a GEMM-level result, not end-to-end model latency.** Real
  inference also spends time in attention, softmax, RMSNorm, and KV-cache
  traffic. The projections measured here are the dominant GEMM cost, not the
  whole forward pass.
- **Single-threaded.** Multi-threaded behaviour is not characterized; both
  sides would change, and not necessarily by the same factor.
- **Weight-only quantization.** Activations are FP32 on both sides, which is
  the standard weight-quantized inference pattern and matches what this
  project's kernels implement.
- **Accuracy is a separate criterion and it fails.** This result says a
  ternary model would be *fast enough*; criterion 5 says the ternary model
  produced so far is not *accurate enough*. Both are true simultaneously and
  neither rescues the other.
- Zen 2 has no FP16 FMA. On hardware with native FP16 (AVX512-FP16, or ARM
  with FP16 SIMD) the baseline would be faster and this margin would shrink.
  The result is platform-scoped, as this project's conventions require.

---

## 6. Reproduction

```bash
python3 benchmarks/cpp-native-kernels/build_inference_latency.py --run
```

The build script locates NumPy's bundled OpenBLAS and resolves its ILP64
symbol spelling (`scipy_cblas_sgemm64_`, `scipy_cblas_sgemv64_`) from the
`.so`'s dynamic symbol table rather than hardcoding one, since the prefix
varies by wheel.
