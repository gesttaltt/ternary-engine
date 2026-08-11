# Prior Art Landscape: Ternary Computing for AI

**Doc-Type:** IP Strategy Research Note · Version 1.0 · 2026-08-11
**Status:** Preliminary mapping — NOT a legal freedom-to-operate analysis.
A professional patent search is required before filing anything.

---

## Purpose

The project invests in OpenTimestamps snapshots (first: 2025-11-23) as
evidence of invention dates. That evidence is only as strong as our
understanding of what already existed. This note maps the most obvious
prior art in ternary AI computing and identifies which of this project's
components are anticipated versus plausibly differentiated.

---

## Part I: The Landscape

### BitNet b1.58 (Microsoft, 2024)

- Weights constrained to {-1, 0, +1} ("1.58 bits"), absmean quantization,
  BitLinear layers, straight-through estimators, trained from scratch.
- At 3B+ parameters matches FP16 LLaMA perplexity with ~3.55× less GPU
  memory ([technical report](https://arxiv.org/pdf/2504.12285),
  [JMLR paper](https://jmlr.org/papers/volume26/24-2050/24-2050.pdf)).
- [bitnet.cpp](https://arxiv.org/pdf/2502.11880) ships **LUT-based
  ternary matmul kernels** for edge CPUs.
- Released model: [bitnet-b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T).

### llama.cpp ternary formats (2024)

- **TQ1_0: 1.6875 bits/weight, packs 5 ternary elements per byte using
  3⁵ = 243 < 256** ([quantization docs](https://deepwiki.com/ggml-org/llama.cpp/7.3-quantization-techniques)).
- TQ2_0: 2 bits/element (~2.06 bpw), same 2-bit-code idea as our core encoding.

### Ternary NN research (2016+)

- [Ternary Weight Networks](https://arxiv.org/pdf/1605.04711) (2016):
  weights in {-1, 0, +1}, 2-bit storage, threshold-based quantization.
- [TNNs for resource-efficient AI](https://arxiv.org/pdf/1609.00222) (2016),
  [Soft Threshold Ternary Networks](https://arxiv.org/pdf/2204.01234),
  custom FPGA adder trees ([ACM TRETS](https://dl.acm.org/doi/10.1145/3270764)).

### Ternary hardware patents (examples found in a first-pass search)

- [KR102540226B1](https://patents.google.com/patent/KR102540226B1/en) /
  [US 2024/0037381](https://patents.justia.com/patent/20240037381):
  ternary neural network accelerator, ternary weight × ternary input.
- [US 2021/0089272](https://patents.google.com/patent/US20210089272A1/en):
  ternary in-memory accelerator.

### Historical

- Balanced ternary hardware: Setun (Moscow State University, 1958).
  Balanced ternary arithmetic itself is centuries old (Fowler, Knuth
  vol. 2 discussion) and unpatentable as such.

---

## Part II: What This Means for Our Components

### Anticipated (do NOT claim novelty)

| Component | Anticipated by |
|-----------|----------------|
| Ternary {-1,0,+1} weights for NNs | TWN 2016, BitNet 2024 |
| 2-bit trit encoding | TQ2_0, standard practice |
| **Dense243 5-trits/byte packing** | **llama.cpp TQ1_0 (2024) uses the identical base-243 construction, predating our first timestamp (2025-11-23)** |
| LUT-based ternary matmul kernels | bitnet.cpp TL1/TL2 |
| Ternary accelerator hardware (generic) | Patents above |
| "1.58-bit" memory-advantage claims | BitNet marketing |

**Action:** Dense243 documentation should cite TQ1_0 as prior art and
stop implying the packing itself is novel. Our packing may still differ
in implementation details (SIMD unpack path, operations directly on
packed data) — those details, not the base-243 idea, are the claimable
surface, if any.

### Plausibly differentiated (worth timestamping and developing)

1. **Saturated balanced-ternary elementwise algebra as an engine.**
   BitNet does mixed ternary-weight × int8-activation matmul; it does
   not define or accelerate a closed elementwise algebra (tadd with
   saturation, tmul, tmin, tmax, tnot) over trits. Our fusion engine
   and canonical-indexing SIMD are specific machinery for that algebra.
2. **The falsification corpus and non-associativity characterization.**
   Exhaustive 19,683-corpus proof that saturated tadd is non-associative
   for 79.6% of triplets, with the algorithm-design consequences
   (research/FINDINGS.md). Research contribution; also defensive
   publication value.
3. **TritNet: distilling exact arithmetic into ternary-weight networks.**
   Learning the arithmetic operations themselves (tnot 100%, tadd 100%
   with ternary weights) inverts the usual direction (networks using
   ternary arithmetic). First-pass search found no direct precedent —
   needs a dedicated literature search before any claim.
4. **3-adic / ultrametric algorithm-design framing** for ternary GEMM
   (valuation-based sparsity analysis, zero-skip justified by 3-adic
   measure). Framing and metrics, likely publishable; patentability
   doubtful (mathematics as such is excluded).

### Honest overlap warning

Our own TRITNET_ROADMAP.md says TritNet distills "using the BitNet
b1.58 pipeline" — the QAT/STE training machinery is BitNet's method.
The differentiator is the *target* (exact arithmetic operators), not
the training technique.

---

## Part III: Recommended Actions

1. Cite TQ1_0 in Dense243 docs (src/engine/dense243/, README claims).
2. Timestamp this note and the differentiated components explicitly.
3. Before any patent filing: professional prior-art search focused on
   (a) elementwise saturated ternary SIMD engines, (b) arithmetic-
   learning networks.
4. Consider defensive publication of the falsification findings —
   fast, cheap, and blocks others from patenting the same results.

**Sources:**
- [BitNet b1.58 2B4T Technical Report](https://arxiv.org/pdf/2504.12285)
- [BitNet: 1-bit Pre-training (JMLR)](https://jmlr.org/papers/volume26/24-2050/24-2050.pdf)
- [Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://arxiv.org/pdf/2502.11880)
- [llama.cpp Quantization Techniques](https://deepwiki.com/ggml-org/llama.cpp/7.3-quantization-techniques)
- [Ternary Weight Networks](https://arxiv.org/pdf/1605.04711)
- [Ternary Neural Networks for Resource-Efficient AI](https://arxiv.org/pdf/1609.00222)
- [KR102540226B1](https://patents.google.com/patent/KR102540226B1/en) ·
  [US 2024/0037381](https://patents.justia.com/patent/20240037381) ·
  [US 2021/0089272](https://patents.google.com/patent/US20210089272A1/en)
- [High-Efficiency Convolutional TNNs (ACM TRETS)](https://dl.acm.org/doi/10.1145/3270764)
