# Ternary Engine Roadmap

**Last Updated:** 2026-08-28
**Current Version:** v1.1.0 "ktr" (doc below still describes the v1.2.0 plan as upcoming; see the reality-check section immediately below — most of it has since shipped)
**Target Version:** v1.2.0 → v2.0 → v3.0+

---

## Status Reality Check (2026-08-17)

This document's body (dated 2025-11-24) was written before most of the
v1.2.0 plan below was implemented, and has not been kept in sync. **CLAUDE.md
is the authoritative, actively-maintained source of truth for current status**
(see its "Critical Gaps & Known Issues" and "TritNet Development" sections);
this box summarizes where the plan below actually stands as of this date so
the rest of the document can be read as history rather than a live plan.

**v1.2.0 plan — mostly shipped, unevenly:**
| Planned item | Actual status |
|---|---|
| Encoding layer (Dense243, TriadSextet) | ✅ Done, validated |
| Backend interface (TCBI) | ✅ Done — `ternary_backend` (Scalar/AVX2_v1/AVX2_v2), in CI since 2026-08-12 |
| Canonical index LUT | ✅ Done |
| LUT-256B expansion | Dead/unreachable code — never wired into a live dispatch path |
| Dual-shuffle XOR | **Not implemented** — explicitly labeled a "future enhancement" in `backend_avx2_v2_optimized.cpp`; its test (`test_dual_shuffle_validation.py`) correctly reports failure rather than being faked or skipped |
| Multi-platform CI | Linux x64 CI builds + runs the full test suite (15/15); no formal benchmark/performance validation on Linux yet, and macOS is untested |

**TritNet — the plan below undercounts what's real:** Phase 2A (tnot) was
listed above as "Pending Validation" in this doc's original body; in reality
Phases 1–3 are all complete (truth tables → QAT training → C++ inference,
scalar and AVX2 → Python bindings), a scope well beyond this document's
tracking. The load-bearing open item is **Phase 4 (GPU acceleration)**,
which has not been started — no CUDA/ROCm code exists in the tree despite
the v3.0 target below. This matters because Phase 3's own benchmark found
LUT beats AVX2-TritNet by 169×–195× on CPU; TritNet's practical case now
rests entirely on the unstarted Phase 4/5 work, not on anything shipped so
far. See CLAUDE.md → "TritNet Development" for the authoritative phase
table.

**v2.0 → v4.0 (index-arithmetic elimination, AVX-512, ARM NEON/SVE, RISC-V
Vector, GPU/CUDA, FPGA/ASIC):** none of this has been started. No code
exists for any backend beyond the current AVX2 (Scalar/AVX2_v1/AVX2_v2)
system.

**Success criteria below:** note that the "✅" marks in the "v1.2.0/v2.0/v3.0
Success Criteria" sections further down are the *original authors' target
list*, not a record of achievement — several of those criteria (e.g. "Builds
on Windows/Linux/macOS", GPU backend, TritNet 100% *and* GPU-competitive)
are still open per the table above. Read them as goals, not checkmarks.

**Other open production gaps** (full detail in CLAUDE.md → "Critical Gaps &
Known Issues"): competitive-benchmark results need a clean-environment
re-run before being cited further; Dense243's Windows `/arch:AVX2` flag is
unvalidated on actual Windows hardware; `train_phase2a.py`/`train_phase2b.py`
still duplicate their QAT training code instead of sharing a module;
`BenchmarkRunner` remains built but unused by the 3 active benchmark
scripts; VTune/NVTX/Perfetto profiler backends have call sites but no build
ever defines the macros to activate them; `test_falsification.py`'s one
completed run reported FALSIFIED (2/5 criteria) in a noisy shared container
and needs an isolated re-run before that number means anything either way.

**⚠️ IMPORTANT — strategic note on the accuracy-retention criterion
(2026-08-28): PTQ may be structurally the wrong tool, not just
under-tuned.** Of the project's 5 commercial-viability criteria (see
CLAUDE.md → "Commercial Viability Criteria"), 2/5 are validated (memory
efficiency 4x vs INT8; throughput at equivalent bit-width, Dense243 8x
faster than an INT2 reference). Of the 3 remaining, "inference latency
<2x FP16" and "power consumption 2-4x better" are testing/infrastructure
gaps (need a real fp16 baseline comparison, and a machine with RAPL/perf
permissions this sandbox lacks) — tractable with more engineering effort
alone. **"Accuracy retention <5% loss" looks different: every
post-training-quantization (PTQ) technique tried so far has failed, and
each more sophisticated attempt failed in the same direction, not a
better one:**

| Technique | Perplexity (baseline → quantized) | Scope |
|---|---|---|
| Naive per-tensor absmean | 12.780 → 89,100.682 (+697,074%) | 100% of layers |
| Naive per-channel absmean | 12.780 → 132,590.254 (+1,037,361%, WORSE) | 100% of layers |
| GPTQ-style (Hessian error compensation) | 7.172 → 26.139 (+264%) | 1 of 22 blocks (~5% of layers) |
| GPTQ-style (Hessian error compensation) | 7.172 → 4,776.805 (+66,502%) | 2 of 22 blocks (~9% of layers) |

(GPTQ rows use a fresh fp16 baseline, not directly comparable to the
fp32 baseline in the naive rows — see
`benchmarks/model_quantization/quantize_tinyllama_gptq.py`.) The GPTQ
row-to-row jump is the concerning part: in log-perplexity (NLL) terms,
block 0 alone adds +1.29 nats but block 1 adds +5.21 more on top of
that — degradation compounding *faster than additively* as more blocks
are touched, at only 9% of the model quantized. This matches this
project's own prior observation (Phase 5 TinyLlama session,
`reports/2026-08-22/PHASE5_TINYLLAMA_QUANTIZATION.md`) that published
ternary/1.58-bit successes (BitNet b1.58 and similar) train **from
scratch** with quantization-aware training (QAT); none of them
post-hoc-quantize an already-converged checkpoint the way every attempt
here has. Ternary is a 3-level quantizer — a much more aggressive
compression than the int4/int8 targets GPTQ's error-compensation math
was originally built for — so it's plausible no amount of better PTQ
math closes this gap, because the failure mode isn't "insufficiently
compensated rounding error," it's "3 discrete values can't represent
what this checkpoint's weights need without retraining."

**UPDATE (2026-08-28, same day): mixed-precision PTQ tried, and it genuinely
helps — direction confirmed, magnitude not yet enough.** Ran the cheapest
possible controlled comparison: quantize the same 14/154 layers (2 whole
blocks) either as blocks 0-1 directly, or with blocks 0-1 protected at fp16
and blocks 2-3 quantized in their place instead (`--protect-first 2 --layers
14`, same script, same calibration data):

| Scope (14/154 layers quantized either way) | Perplexity | vs baseline |
|---|---|---|
| Blocks 0-1 quantized (no protection) | 7.172 → 4,776.805 | +66,502% |
| Blocks 0-1 protected, blocks 2-3 quantized instead | 7.172 → 261.189 | +3,542% |

**~18.8x better** for the identical amount of the model touched, just by
moving *which* blocks get quantized — confirms the hypothesis that the
earliest transformer blocks are disproportionately sensitive to ternary
quantization, exactly as most published low-bit techniques already assume
when they protect early/late layers. Still nowhere near the <5% criterion
at this scale (14/154 layers, 9%), so this is a real, positive, but
partial result, not a solved problem: worth extending (protect more of the
front AND back, e.g. `--protect-first 3 --protect-last 3` over a
full-model run) before concluding either way. `--protect-first`/
`--protect-last` are now real flags on
`quantize_tinyllama_gptq.py` (mixed-precision blocks are simply forward-
passed at their original fp16 weights, never hooked/quantized/
checkpointed as quantized).

**RESOLVED (2026-08-28, later the same day): step (1) was run at
full-model scale on GPU. It FAILS.** The pipeline was CPU-only, which is
precisely why this had never been run; adding `--device cuda` to
`quantize_tinyllama_gptq.py` (quantization math unchanged, verified
equivalent to the CPU path to 1.8e-05 on zero-fraction and 1.4e-07 on
scale before being trusted) made a full 154-layer pass cost **4.0 minutes
instead of ~2.5-3.5h** — 42x on calibration+quantization, 193x
end-to-end. Results, all fp16 / 8192-token eval / this environment:

| Run | Layers ternarized | Perplexity | vs baseline |
|---|---|---|---|
| fp16 baseline | 0 / 154 | 12.780 | — |
| full model, no protection (control) | 154 / 154 | 18,565.469 | +145,167% |
| full model, `--protect-first 3 --protect-last 3` | 112 / 154 (73%) | 14,285.862 | +111,681% |

Mixed-precision PTQ misses the <5% criterion by ~4 orders of magnitude.
Worse for the hypothesis: **the benefit of protection shrinks as coverage
grows** — 2.04x at the 2-block pilot scale, only 1.30x at full-model
scale — the opposite of what "protect the sensitive layers and the rest
is fine" predicts. Per the decision rule stated below, step (1) is now
exhausted.

**⚠ The GPTQ numbers recorded in the v1.56.0 note above do not reproduce
in the current environment.** Re-running the same script with the same
flags now gives an fp16 baseline of 12.780 (not 7.172), blocks 0-1
quantized -> 1,336.567 (not 4,776.805), and blocks 0-1 protected / 2-3
quantized -> 656.236 (not 261.189) — an improvement from protection of
2.04x, not ~18.8x. Three candidate explanations were tested and **all three
are excluded**: it is not a GPU artifact (CPU and CUDA agree to 6.4e-05 on
baseline perplexity), not a `transformers` version difference (v4.46.3 and
v5.16.1 agree to ~0.1% on every cell of a window grid — and the committed
script's `dtype=` kwarg is v5-only, so the earlier session must itself have
run v5, which is exactly where 12.780 is measured), and not an
eval-window difference (a scan of 54 `(seq_len, max_tokens)` combinations
found nothing within 0.236 of 7.172). The code has not drifted either:
only two commits have ever touched the file, and every relevant constant
plus the body of `compute_perplexity()` is byte-identical between them.
This session's *fp16* baseline also agrees to 5 decimals with the *fp32*
baseline (12.780) that the sibling script `quantize_tinyllama.py`
established independently.

**Therefore: treat the 7.172-based GPTQ rows above — and the "~18.8x
better" claim derived from them — as NOT REPRODUCIBLE from the committed
code, not merely as "measured in a different environment".** They are
superseded by this session's internally consistent numbers. This is not
proof the earlier run was wrong: that session developed the script in
stages, so intermediate runs may have used code differing from what was
finally committed, and the HuggingFace cache was empty at the start of
this session so the earlier model snapshot cannot be inspected. The
*qualitative* early-block-sensitivity finding does reproduce, at 2.04x
rather than ~18.8x. Full detail:
reports/2026-08-28/GPTQ_GPU_ENABLEMENT_AND_FULL_MODEL_MIXED_PRECISION.md
section 5.1

**Recommendation, in order, before concluding PTQ is a dead end or
committing to a costlier pivot:**
1. **Cheap, low-risk, in progress:** mixed-precision PTQ — keep the most
   sensitive layers (embeddings, first/last transformer blocks) at
   fp16/int8 and ternarize only the bulk of the middle, exactly what
   nearly every published low-bit quantization technique actually does
   (none of them ternarize 100% of the model uniformly). This reuses
   100% of the existing GPTQ pipeline. **DONE 2026-08-28, and it FAILS**
   (see the RESOLVED box above): at full-model scale with 6 of 22 blocks
   protected, perplexity is 14,285.862 against a 12.780 baseline
   (+111,681%), and the benefit of protection shrinks rather than grows
   with coverage. Mixed precision delays the compounding failure; it does
   not avert it.
2. **If (1) also fails at full-model scale: the real fix is QAT or
   training from scratch**,
   matching what actually-successful ternary models do — this project
   already has proven QAT building blocks at small scale
   (`models/tritnet/qat_common.py`'s `TernaryLinearQAT`, validated on
   TritNet's own ops) that could in principle extend to real transformer
   layers, lowering some of the engineering risk. **This is now the
   active recommendation, since (1) is exhausted as of 2026-08-28.**

   **The GPU half of this step's stated blocker has lifted.** This
   paragraph previously deferred QAT partly because "no GPU access has
   been available on any checked peer session" — that premise was
   stale: a CUDA GPU (RTX 3050, 6GB, compute 8.6) is present and working
   on this host, and the 2026-08-28 session used it to run a full
   154-layer GPTQ pass in 4.0 minutes. That does **not** make training a
   1.1B model from scratch tractable on a 6GB card — it is not — but
   it does make **small-scale QAT experiments** tractable here. The
   session-instability half of the blocker stands (3 machine reboots
   during the earlier GPTQ session), which argues for short, checkpointed
   experiments rather than long unattended ones.

   **Recommended concrete next step:** a scoped QAT feasibility
   experiment — a small transformer, or a few layers of TinyLlama,
   fine-tuned with `TernaryLinearQAT` in the loop on the GPU, measuring
   whether perplexity recovers toward the fp16 baseline in a way PTQ
   demonstrably cannot. That is falsifiable and answerable in this
   environment. It is explicitly *not* a commitment to full training from
   scratch, which remains a separately-resourced decision.

Net (updated 2026-08-28): step (1) is done and negative, so the PTQ
direction **is** now exhausted by this document's own rule. Four
techniques have been tried — naive per-tensor (+697,074%), naive
per-channel (+1,037,361%), GPTQ (+145,167%), GPTQ + mixed precision
(+111,681%) — and all four failed in the same direction, not a better
one. Do not invest in a fifth PTQ variant. The open question worth
spending on is whether QAT behaves differently, which is step (2).

---

## Vision

Build a **universal ternary computing platform** with:
- **Portable scalar core** (reference implementation for all platforms)
- **Platform-specific backends** (AVX2, AVX-512, ARM SVE, RISC-V Vector, GPU)
- **Dense encoding layers** (Sixtet, Octet, Dense243)
- **Stable 25-50 Gops/s** on consumer CPUs (v1.2.0)
- **100-150 Gops/s** on AVX-512 (v2.0)
- **1-5 TOps** on GPU with TritNet (v3.0+)

---

## Architecture Principles

### **Principle 1: Separation of Concerns**
- **Mathematical core** (scalar, portable, truth-table-based)
- **Encoding layers** (Sixtet/Octet/Dense243 for I/O and cache optimization)
- **Compute backends** (SIMD-specific optimizations as plugins)

### **Principle 2: Platform Agnostic**
- Scalar reference runs everywhere (C99)
- Backends are optional performance layers
- No ISA lock-in (support x86, ARM, RISC-V, FPGA, ASIC, GPU)

### **Principle 3: Future-Proof**
- Layered design supports new backends
- TritNet enables neural network-based arithmetic
- Ready for custom silicon (FPGA/ASIC)

---

## Current Status (v1.1.0 "ktr")

### **Production-Ready** ✅
- Scalar ternary algebra (16 tests passing)
- AVX2 SIMD kernels (28.6-35.0 Gops/s)
- Dense243 encoding (5 trits/byte, validated)
- Operation fusion Phase 4.0 (1.59× - 21.65× speedup)
- Windows x64 platform (fully validated)

### **Validated & Ready** ✅
- TriadSextet encoding (6 trits in 2 bytes)
- TritNet GEMM integration (AVX2 matmul)
- Competitive benchmarks (vs NumPy INT8)

### **Pending Validation** ⚠️
- Linux/macOS builds (untested)
- Multi-platform CI (disabled for OpenMP)
- TritNet Phase 2A (tnot learning)

---

## Roadmap

### **v1.2.0 "Encoding-Aware Pipeline"** (Target: Next Release)

**Goal:** Add Sixtet/Octet layers for cache optimization and I/O efficiency without compromising portability.

**Major Features:**
1. **Encoding Layer (TEL)** - Ternary Encoding Layer
   - Sixtet pack/unpack (3 trits → 6 bits)
   - Octet pack/unpack (2 trits → 3 bits)
   - Dense243 integration
   - Portable scalar implementation

2. **Backend Interface (TCBI)** - Ternary Compute Backend Interface
   - Clean separation between scalar core and SIMD backends
   - Backend registration system
   - Runtime backend selection

3. **Safe SIMD Optimizations**
   - Canonical index LUT (removes shift/OR arithmetic)
   - LUT-256B (256-byte expanded lookup tables)
   - Dual-shuffle XOR (parallel execution on separate ports)
   - Selective interleaving (portable subset)

**Performance Targets:**
- **Sustained:** 20-35 Gops/s (stable under load)
- **Peak:** 45 Gops/s (ideal conditions)
- **Cache improvements:** +15-25% from Sixtet strip-mining

**Platform Support:**
- Windows x64 (primary)
- Linux x64 (build validation)
- macOS ARM64 (build validation)

**Breaking Changes:** None (external API unchanged)

---

### **v2.0 "SIMD Kernel v2.0"** (Future)

**Goal:** Maximum AVX2 performance through microarchitectural optimization.

**Major Features:**
1. **Index Arithmetic Elimination**
   - Remove all `(a << 2) | b` operations
   - Pure LUT-based index generation
   - Zero integer ALU pressure

2. **Pipeline Port Saturation**
   - Permute + Shuffle + XOR interleaving
   - Utilize 3 execution ports simultaneously
   - Reduce thermal variance

3. **Multi-LUT Fusion**
   - 2-op fusion (1.9× - 3.4× speedup)
   - 3-op fusion (2.8× - 5.5× speedup)
   - Fused LUTs for common patterns

**Performance Targets:**
- **Sustained:** 25-50 Gops/s (stable)
- **Peak:** 55-70 Gops/s (burst)
- **Variance:** <2× (vs 7× in v1.0)

**Platform:** x86-64 AVX2 optimized (Intel Skylake+, AMD Zen2+)

---

### **v2.5 "Multi-Platform Backends"** (Future)

**Goal:** Add backends for AVX-512, ARM NEON/SVE, RISC-V Vector.

**Backends:**
1. **AVX-512** (Intel Ice Lake+)
   - 64-element vectors
   - 100-150 Gops/s sustained

2. **ARM NEON** (Apple Silicon, ARM Cortex-A)
   - 16-element vectors
   - Mobile/embedded deployment

3. **ARM SVE/SVE2** (ARM servers, Fujitsu A64FX)
   - Variable-length vectors (128-2048 bits)
   - 200-400 Gops/s sustained

4. **RISC-V Vector** (SiFive, Alibaba T-Head)
   - Variable-length vectors
   - Future-proof open ISA

---

### **v3.0 "TritNet GPU Acceleration"** (Future)

**Goal:** Replace memory-bound LUTs with compute-bound neural network operations.

**Major Features:**
1. **TritNet Training**
   - Neural networks learn ternary operations
   - Ternary weights (1.58 bits/weight)
   - 100% accuracy requirement

2. **GPU Inference**
   - CUDA/ROCm kernels
   - Tensor core acceleration
   - Batch processing

3. **BitNet Integration**
   - Hybrid binary/ternary matmul
   - Popcount + XOR tricks
   - 10× faster than NumPy matmul

**Performance Targets:**
- **GPU:** 1-5 TOps (trillion ops/sec)
- **Batch efficiency:** 30-50× speedup vs scalar
- **Model quantization:** TinyLlama, Phi-2, Gemma

---

### **v4.0+ "Hardware Acceleration"** (Long-term)

**Goal:** Custom silicon and FPGA implementations.

**Platforms:**
1. **FPGA** (Xilinx/Altera)
   - 100-300 Gops/s
   - HDL generation from scalar core
   - Reconfigurable logic

2. **ASIC** (Custom silicon)
   - 2-10 TOps
   - Ternary ALU units
   - In-memory LUT arrays

3. **NPU/TPU** Integration
   - Google Edge TPU
   - Qualcomm Hexagon
   - Apple Neural Engine

---

## Detailed v1.2.0 Implementation Plan

### **Phase 1: Encoding Layer**

**Sixtet Implementation**
- 3 trits → 6 bits packing
- LUT-based pack/unpack (64-entry tables)
- Strip-mining for L1 cache optimization
- Branchless encoding/decoding

**Octet Implementation**
- 2 trits → 3 bits with canonical mapping
- 7 valid states + 1 sentinel
- Byte-aligned for DMA/GPU transfers
- Error detection support

**Dense243 Integration**
- Existing implementation (5 trits/byte)
- Positional base-3 encoding
- Storage/network transport optimized

**Files:**
- `src/core/packing/sixtet_pack.h`
- `src/core/packing/octet_pack.h`
- `src/core/packing/pack.h`
- `src/core/packing/unpack.h`

---

### **Phase 2: Backend Interface**

**Ternary Compute Backend Interface (TCBI)**
```cpp
struct TernaryBackend {
    const char* name;
    bool (*detect)(void);
    void (*tadd)(const uint8_t*, const uint8_t*, uint8_t*, size_t);
    void (*tmul)(const uint8_t*, const uint8_t*, uint8_t*, size_t);
    void (*tmin)(const uint8_t*, const uint8_t*, uint8_t*, size_t);
    void (*tmax)(const uint8_t*, const uint8_t*, uint8_t*, size_t);
    void (*tnot)(const uint8_t*, uint8_t*, size_t);
};
```

**Backend Registration**
- Runtime detection (CPUID for AVX2/AVX-512)
- Backend priority system
- Fallback to scalar reference

**Backends:**
- `scalar` (C99 reference, always available)
- `avx2_v1` (existing SIMD, portable subset)
- `avx2_v2` (future optimized, x86-specific)

**Files:**
- `src/core/backend/backend_interface.h`
- `src/core/backend/backend_scalar.c`
- `src/core/backend/backend_avx2_v1.cpp`
- `src/core/backend/backend_registry.c`

---

### **Phase 3: Safe SIMD Optimizations**

**Canonical Index LUT**
- Pre-computed index mapping
- Eliminates `(a << 2) | b` at runtime
- 16-byte or 256-byte LUT options

**LUT-256B Expansion**
- 256-byte lookup tables (4 cache lines)
- Direct byte indexing (no bit manipulation)
- Fits in L1 data cache

**Dual-Shuffle XOR**
```cpp
__m256i lo = _mm256_shuffle_epi8(LUT_LO, a);
__m256i hi = _mm256_shuffle_epi8(LUT_HI, b);
__m256i out = _mm256_xor_si256(lo, hi);
```
- Two parallel shuffles (separate execution ports)
- Zero-latency XOR fusion
- 1.5-1.7× speedup on Zen CPUs

**Files:**
- `src/core/simd/ternary_simd_kernels_v2.h`
- `src/core/algebra/ternary_lut_256.h`
- `src/core/simd/ternary_canonical_index.h`

---

### **Phase 4: Testing & Validation**

**Correctness Tests**
- Sixtet pack/unpack round-trip
- Octet encoding validation
- Backend equivalence tests
- Cross-platform consistency

**Performance Benchmarks**
- Cache pressure reduction (Sixtet)
- Backend selection overhead
- Canonical index speedup
- Dual-shuffle XOR gains

**Platform Validation**
- Windows x64 (primary)
- Linux x64 (Docker/CI)
- macOS ARM64 (CI)

**Files:**
- `tests/cpp/test_sixtet.cpp`
- `tests/cpp/test_octet.cpp`
- `tests/python/test_backends.py`
- `benchmarks/bench_encoding.py`

---

### **Phase 5: Documentation**

**Technical Documentation**
- Encoding layer design document
- Backend interface specification
- SIMD optimization guide
- Platform support matrix

**API Documentation**
- Sixtet/Octet usage examples
- Backend selection API
- Performance tuning guide
- Migration from v1.1.0

**Files:**
- `docs/architecture/encoding-layer.md`
- `docs/architecture/backend-interface.md`
- `docs/performance/simd-optimizations.md`
- `docs/migration/v1.1-to-v1.2.md`

---

## Performance Projections

### **AVX2 Theoretical Limits**

| Version | Ops/Cycle | Clock | Theoretical | Measured Stable | Measured Peak |
|:--------|----------:|------:|------------:|----------------:|--------------:|
| v1.0    | 32        | 3.5 GHz | 112 Gops/s | 12-28 Gops/s | 35 Gops/s |
| v1.1    | 32        | 3.8 GHz | 122 Gops/s | 28 Gops/s | 35 Gops/s |
| v1.2    | 48        | 3.8 GHz | 182 Gops/s | 30 Gops/s | 45 Gops/s |
| v2.0    | 64        | 4.0 GHz | 256 Gops/s | 40 Gops/s | 65 Gops/s |

### **Platform Projections**

| Platform | Version | Sustained | Peak | Efficiency |
|:---------|:--------|----------:|-----:|-----------:|
| AVX2 (Zen2) | v1.2 | 30 Gops/s | 45 Gops/s | 25% |
| AVX2 (Zen3) | v2.0 | 45 Gops/s | 65 Gops/s | 35% |
| AVX-512 (Ice Lake) | v2.5 | 120 Gops/s | 180 Gops/s | 60% |
| ARM SVE (A64FX) | v2.5 | 250 Gops/s | 400 Gops/s | 50% |
| GPU (RTX 3050) | v3.0 | 2 TOps | 5 TOps | ~40% |

---

## Dependencies & Requirements

### **Core Requirements**
- C++17 compiler (MSVC, GCC 9+, Clang 10+)
- Python 3.7+ (for bindings)
- pybind11 2.6+
- NumPy 1.19+

### **Platform-Specific**
- **AVX2:** Intel Haswell (2013+), AMD Excavator (2015+)
- **AVX-512:** Intel Ice Lake (2019+), AMD Zen4 (2022+)
- **ARM NEON:** ARMv7-A+, Apple M1+
- **ARM SVE:** ARMv8.2-A+, Fujitsu A64FX

### **Optional**
- PyTorch 2.0+ (for TritNet)
- CUDA 11.0+ (for GPU backend)
- OpenMP (for multi-threading)

---

## Success Metrics

### **v1.2.0 Success Criteria**
- ✅ Sixtet/Octet encoding implementations complete
- ✅ Backend interface working (scalar + AVX2)
- ✅ No performance regression from v1.1.0
- ✅ +15% sustained throughput from cache optimization
- ✅ Builds on Windows/Linux/macOS
- ✅ All tests passing
- ✅ Documentation complete

### **v2.0 Success Criteria**
- ✅ 40+ Gops/s sustained on Zen3/Skylake
- ✅ <2× performance variance (vs 7× in v1.0)
- ✅ Multi-LUT fusion operational
- ✅ Thermal stability improvements
- ✅ Comprehensive benchmarks

### **v3.0 Success Criteria**
- ✅ TritNet training reaches 100% accuracy
- ✅ GPU backend operational
- ✅ 1+ TOps sustained on consumer GPU
- ✅ Model quantization (TinyLlama, Phi-2)

---

## Timeline & Milestones

**v1.2.0 Development:**
- Phase 1 (Encoding): 6 weeks
- Phase 2 (Backend Interface): 4 weeks
- Phase 3 (SIMD Optimizations): 8 weeks
- Phase 4 (Testing): 4 weeks
- Phase 5 (Documentation): 2 weeks
- **Total:** ~24 weeks (~6 months)

**v2.0 Development:**
- Advanced SIMD: 12 weeks
- Multi-LUT fusion: 8 weeks
- Thermal optimization: 4 weeks
- **Total:** ~24 weeks (~6 months)

**v3.0 Development:**
- TritNet training: 12 weeks
- GPU kernels: 16 weeks
- Model integration: 8 weeks
- **Total:** ~36 weeks (~9 months)

---

## Open Questions

1. **Sixtet vs TriadSextet:** Which to prioritize in v1.2.0?
   - Sixtet: 3 trits → 6 bits (canonical, simpler)
   - TriadSextet: 6 trits → 16 bits (higher density, more complex)

2. **Backend dispatch overhead:** Acceptable cost for runtime selection?
   - Virtual function calls: ~2-5 cycles
   - Function pointers: ~1-3 cycles
   - Static dispatch: 0 cycles (compile-time only)

3. **Multi-platform CI:** Re-enable OpenMP tests after fixes?
   - OpenMP crashes on CI (root cause fixed in v1.0)
   - Needs validation across platforms

4. **TritNet accuracy requirement:** 100% or 99%+ acceptable?
   - 100%: Perfect arithmetic (required for exact computation)
   - 99%+: Approximate arithmetic (sufficient for ML/AI)

---

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

**Roadmap discussions:** [GitHub Discussions](https://github.com/gesttaltt/ternary-engine/discussions)

**Issue tracking:** [GitHub Issues](https://github.com/gesttaltt/ternary-engine/issues)

---

## References

### **Internal Documentation**
- [TRITNET_ROADMAP.md](../research/tritnet/TRITNET_ROADMAP.md) - TritNet neural network learning
- [architecture/optimization-roadmap.md](../architecture/optimization-roadmap.md) - SIMD optimizations
- [TECHNICAL_DEBT_CATALOG.md](../historical/audits/TECHNICAL_DEBT_CATALOG.md) - Known issues
- [BITNET_INTEGRATION_STRATEGY.md](../research/bitnet/BITNET_INTEGRATION_STRATEGY.md) - BitNet hybrid approach

### **Local Reports** (not in git)
- `local-reports/opt.md` - Comprehensive SIMD optimization guide v2.0
- `local-reports/tpo.md` - v1.2.0 architecture proposal
- `local-reports/2025-11-24/PERFORMANCE_INVESTIGATION.md` - 35 Gops/s validation

### **External Resources**
- [Balanced Ternary](https://en.wikipedia.org/wiki/Balanced_ternary) - Wikipedia
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/) - AVX2/AVX-512
- [ARM SVE Programming](https://developer.arm.com/documentation/102476/latest/) - Scalable Vector Extension
- [RISC-V Vector Extension](https://github.com/riscv/riscv-v-spec) - RVV specification

---

**Last Updated:** 2026-08-17 (reality-check box added at top; body below is historical, dated 2025-11-24)
**Maintainers:** Jonathan Verdun, Ternary Engine Contributors
**License:** Apache 2.0
