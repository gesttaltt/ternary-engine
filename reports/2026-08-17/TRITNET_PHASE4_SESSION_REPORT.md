# TritNet Phase 4 — 2026-08-17

**Scope:** Continue the TritNet roadmap from `.claude/CLAUDE.md`'s "TritNet
Development" section after Phase 3's GPU-shaped question was left open:
"Remaining work is Phase 4 (GPU/TPU), which is where TritNet's actual case
has to be made." This machine has a real CUDA GPU (NVIDIA GeForce RTX 3050,
6GB, compute capability 8.6) — no TPU path exists in this repo, so Phase 4's
scope here is GPU-only, via PyTorch/CUDA (batched inference against the same
exported ternary weights Phase 3's C++ engine uses).

**Net result:** Phase 4's GPU experiment is complete and reaches the same
qualitative conclusion Phase 3 reached on CPU: **the LUT still wins.** GPU
batching, fp16, and kernel fusion all measurably help TritNet's throughput,
but none of them — even stacked — close the gap. Best case (fp16, batch
1-3M, end-to-end): binary ops reach ~0.25-0.27x of the LUT's throughput on
this hardware; tnot reaches ~0.10-0.13x. The honest read is unchanged from
Phase 3's: TritNet's case, if it exists, rests on Phase 5 (capabilities a
LUT structurally can't offer), not on raw throughput.

New file: `models/tritnet/phase4_gpu_benchmark.py`. Results snapshot:
`models/tritnet/phase4_gpu_results.json`.

---

## 1. Methodology

The benchmark reuses the *exact* forward-pass recipe `export_weights.py` and
`tests/python/test_tritnet_export.py` already validate on CPU (`h = ReLU(x @
Wi + bi)` for the two hidden layers, then a plain linear output layer,
argmax-1 decode per trit), executed as batched PyTorch tensor ops on CUDA
instead of NumPy/AVX2 on CPU. Weights are loaded straight from
`models/tritnet/phase2b_export/<op>/*.npy` — the same files Phase 3's C++
engine consumes, so a GPU number here is comparable to a CPU number there in
principle, not just in spirit.

Two throughput numbers are reported per (op, batch size, dtype), for the
same reason Phase 3 ended up splitting "reconverts weights every call" from
"weights preconverted once":

- **compute-only** — input already resident on GPU; times only the matmuls
  + ReLU + argmax. The number that matters if TritNet is one stage inside a
  larger GPU-resident pipeline.
- **end-to-end** — a single round trip from CPU: host→device copy of the
  input batch, compute, device→host copy of the result. The number that
  matters if this replaces a CPU LUT call site as-is.

Correctness is checked first, over the **full input space** per op (243
samples for tnot, 59,049 for the 4 binary ops) — the same check
`test_tritnet_export.py` runs on CPU, replayed on GPU. Throughput numbers
are only printed after every op passes.

## 2. Correctness: fp32 exact, fp16 has a small, real, documented drift

| Op | fp32 GPU acc | Recorded (CPU) acc | fp16 GPU acc | fp16 drift |
|---|---|---|---|---|
| tnot | 100.0000% | 100.0000% | 100.0000% | +0.0000pp |
| tadd | 100.0000% | 100.0000% | 100.0000% | +0.0000pp |
| tmul | 99.4919% | 99.4919% | 99.4919% | +0.0000pp |
| tmin | 99.8882% | 99.8882% | 99.8916% | +0.0034pp |
| tmax | 99.8510% | 99.8510% | 99.8493% | -0.0017pp |

fp32 on GPU reproduces the CPU-recorded checkpoint accuracy exactly for all
5 ops, over the full input space — the same fidelity guarantee Phase 3
established for the C++ engine now holds for the PyTorch/CUDA path too.

fp16 does **not** always reproduce it exactly: tmin and tmax each flip a
handful of samples relative to fp32 (net drift under 0.004 percentage
points either op). This isn't a bug — it's reduced-precision logits
crossing an argmax tie near a decision boundary — but it's a real,
measured tradeoff that fp16's speedup (below) doesn't come for free. tnot,
tadd, and tmul happened not to have any near-tied logits close enough to
flip at fp16 precision on this run.

## 3. Throughput: GPU beats AVX2-CPU by an order of magnitude, LUT still wins by 4-10x

Same-host baselines (AMD Ryzen 5 4500, this machine), re-measured today via
`benchmarks/cpp-native-kernels/bench_tritnet_inference.cpp` rather than
reused from Phase 3's numbers — Phase 3's CPU baselines were measured on a
*different* machine (AMD Ryzen 5 7520U), and comparing today's GPU numbers
against a different host's CPU numbers would repeat the exact cross-run
timing-fairness mistake Phase 3 already caught and fixed once, one level
up. AVX2 uses the "reconverts weights every call" figure — Phase 3's
headline comparison point, not the amortized one:

| Op | LUT (Mops/s) | AVX2 (Mops/s) |
|---|---|---|
| tnot | 517.12 | 3.3478 |
| tadd | 148.88 | 1.0038 |
| tmul | 145.87 | 1.0035 |
| tmin | 134.80 | 1.0040 |
| tmax | 139.02 | 1.0033 |

GPU results, best configuration per op (fp16, largest batch that fit in
6GB VRAM before CUDA OOM; end-to-end, i.e. including H2D/D2H transfer):

| Op | Batch | GPU e2e (Mops/s) | vs LUT | vs AVX2 |
|---|---|---|---|---|
| tnot | 1,000,000 | 64.60 | 0.125x | 19.29x |
| tadd | 3,000,000 | 37.03 | 0.249x | 36.89x |
| tmul | 1,000,000 | 36.69 | 0.252x | 36.56x |
| tmin | 3,000,000 | 36.97 | 0.274x | 36.82x |
| tmax | 1,000,000 | 36.72 | 0.264x | 36.60x |

Compute-only (excluding transfer) tops out a bit higher — ~82 Mops/s for
tnot, ~44.5 Mops/s for the binary ops — but even that best case reaches
only ~0.32x of LUT for the binary ops and ~0.16x for tnot. Full sweep
(fp32 and fp16, 3-5 batch sizes per op) is in the script's stdout and
`phase4_gpu_results.json`.

**fp16 vs fp32:** a real, consistent ~1.5-2x throughput gain (e.g. tadd
compute-only: 21.85 → 44.52 Mops/s at batch=1M), at the cost of the small
accuracy drift in §2. Worth taking if GPU inference is used at all; not
enough on its own to change the LUT-wins conclusion.

**OOM ceiling:** on this 6GB laptop-class GPU, tnot (hidden=64) fits up to
5M samples in fp32 / 10M in fp16; the binary ops (hidden=128, so ~2x the
activation memory per sample) fit up to 1M in fp32 / 3M in fp16 before
`torch.cuda.OutOfMemoryError`. This is a hardware ceiling specific to this
GPU, not a fundamental limit — a GPU with more VRAM would push it out, but
would not change the underlying Mops/s ceiling (see §4).

## 4. Why: the network is too small to be GPU-compute-bound at any batch size

Throughput barely rises between batch=100K and batch=1-3M (e.g. tadd fp32
compute-only: 15.05 → 21.85 Mops/s from 100K to 1M, then flat) — the
signature of a workload that's already saturated whatever it's going to
saturate well before the GPU's parallelism ceiling, rather than one that's
overhead-bound at small batches and scales up with N. At batch=1M each
forward call already takes tens of milliseconds, so Python/kernel-launch
dispatch overhead (microseconds) isn't the bottleneck — the GPU's own
execution time is. The hidden layers (64 or 128 neurons) and layer count
(3 matmuls + 2 ReLU) are simply too small and too memory-bandwidth-bound
(elementwise ReLU between each tiny matmul) to reach a compute-bound
regime where an RTX 3050's ~9 TFLOPS FP32 would tell.

## 5. `torch.compile` explored, not wired into the sweep

A standalone check (tadd, fp16, batch=1M, `torch.compile(mode="reduce-overhead")`,
fusing the 3-matmul/2-ReLU/1-argmax forward into fewer kernel launches via
CUDA graphs) pushed compute-only throughput from 44.6 → 68.6 Mops/s — a
further ~1.5x on top of fp16's own ~2x gain, for a compile step costing
~3.8s one-time (amortizable in a long-running batch pipeline, not free for
a cold start). Still short of LUT's 133-149 Mops/s. This was not folded
into `phase4_gpu_benchmark.py`'s automated sweep — compiling separately
per op per batch size would multiply the script's run time for a result
that narrows but does not reverse the gap, and per this project's
`phase_coherence` standard (only add complexity for >10% *closing* the
actual question, not just any measured speedup) it isn't worth the
added surface area yet. Documented here as a data point for whoever picks
up Phase 5.

## 6. Conclusion

Phase 4 does not reverse Phase 3's finding — it extends it to real GPU
hardware and confirms it under the best conditions this experiment could
give TritNet (batching across 3-4 orders of magnitude, fp16, and a
kernel-fusion spot-check), the same fairness-first posture Phase 3 applied
to AVX2. The LUT's ~2ns-per-sample memory lookup remains fundamentally
cheaper than any version of the neural-network forward pass tried here,
GPU included: batching amortizes per-call *overhead*, but the LUT never
paid meaningful per-call overhead to begin with, and the networks trained
in Phase 2B are too small to reach a regime where a GPU's raw FLOPS
advantage would overcome that.

Per `.claude/CLAUDE.md`'s own framing, TritNet's remaining practical case
rests on **Phase 5 (Learned Generalization)** — capabilities a LUT
structurally cannot offer (approximate arithmetic, novel operations,
generalization beyond the trained truth table) — not on beating a LUT at
throughput for these exact per-op networks, on this or any hardware.

---

**Reproduce:** `python models/tritnet/phase4_gpu_benchmark.py` (requires
PyTorch with CUDA; skips gracefully with exit 0 if unavailable). CPU
baselines: `g++ -O3 -march=native -mavx2 -mfma -std=c++17 -I../../
bench_tritnet_inference.cpp -o bench_tritnet_inference` from
`benchmarks/cpp-native-kernels/`, then run the resulting binary.
