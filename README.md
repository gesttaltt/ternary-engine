# Ternary Engine

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![C++ Standard](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![Performance](https://img.shields.io/badge/peak-45300%20Mops/s-brightgreen)](https://github.com/gesttaltt/ternary-engine#performance)
[![Fair baseline](https://img.shields.io/badge/fused%20vs%20NumPy-1.4x-brightgreen)](https://github.com/gesttaltt/ternary-engine#performance)
[![Platform](https://img.shields.io/badge/production-Windows%20x64-blue)](https://github.com/gesttaltt/ternary-engine#production-status)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

Production-grade balanced ternary arithmetic library with AVX2 SIMD vectorization, operation fusion, and Python bindings.

## Production Status

✅ **Windows x64:** Production-ready (validated 2025-11-28)
🟡 **Linux x64:** Tests fully validated + CI on every push (2026-08-11); formal benchmark validation pending — no production performance claims yet per project standard

## Overview

Ternary Engine implements high-performance balanced ternary logic operations using lookup table optimization, AVX2 SIMD vectorization (32 parallel operations), and operation fusion. Achieves **peak throughput of 45,300 Mops/s** (45.3 Gops/s) with fusion operations (validated 2025-11-28, Windows x64). Against NumPy implementing the same ternary semantics — the honest baseline — the engine reaches **1.7–3.5× on saturated addition** and **1.43× geometric mean on fused operations** (up to 6× on tnot∘tadd), with 0.84× geomean on single element-wise ops (Linux x64, 2026-08-11).

> **Benchmark Methodology Note:** Performance metrics for ternary operations are *subject to analysis* as there is no standardized benchmarking methodology for trit-based computing. Measurements follow best practices (statistical rigor, load-aware benchmarking, reproducibility validation) but direct comparison with binary operations requires careful interpretation. Results represent actual measured throughput on validated test systems.

**Balanced Ternary**: Three-valued logic system using {-1, 0, +1} with symmetric negative/positive representation. Applications include fractal generation, modulo-3 arithmetic, and specialized computational workflows. **Future potential**: Computer vision edge detection (experimental POC in development - see roadmap).

### Features

- **2-bit trit encoding** - Compact representation (0b00=-1, 0b01=0, 0b10=+1)
- **Branch-free operations** - Pre-computed lookup tables eliminate conditional logic
- **AVX2 vectorization** - Process 32 trits per operation via `_mm256_shuffle_epi8`
- **OpenMP parallelization** - Automatic multi-threading for arrays ≥100K elements
- **NumPy integration** - Zero-copy array processing via pybind11

### Supported Operations

| Operation | Function | Description |
|-----------|----------|-------------|
| Addition | `tadd(a, b)` | Saturated addition (clamps to [-1, +1]) |
| Multiplication | `tmul(a, b)` | Standard multiplication |
| Minimum | `tmin(a, b)` | Element-wise minimum |
| Maximum | `tmax(a, b)` | Element-wise maximum |
| Negation | `tnot(a)` | Sign flip (0 unchanged) |

### Dense243 High-Density Module (Experimental)

**Separate module for 20% storage savings with TritNet-ready architecture**

```python
import ternary_dense243_module as td

# Pack 5 trits into 1 byte (vs 5 bytes in standard encoding)
trits = np.array([0b00, 0b01, 0b10, 0b10, 0b01], dtype=np.uint8)
packed = td.pack(trits)  # 5 → 1 byte (80% space savings)

# Future: Neural network-based operations
td.set_backend('tritnet')  # Switch from LUT to trained model
result = td.tadd(packed_a, packed_b)  # Uses matmul instead of lookup
```

**Features:**
- **Density:** 5 trits/byte (95.3% utilization) vs 4 trits/byte (standard)
- **Performance:** Pack 0.25ns, Unpack 0.91ns (validated, all 243 states tested)
- **Use cases:** Persistent storage, network transmission, memory-bound workloads
- **TritNet roadmap:** Train BitNet on truth tables → distill to ternary weights → replace LUT with matmul
- **Build:** `python build/build_dense243.py`
- **Docs:** `docs/research/tritnet/TRITNET_ROADMAP.md`

### TritNet - Neural Network-Based Ternary Arithmetic (Experimental)

**Revolutionary approach: Replace lookup tables with learned matrix multiplication**

```python
# Traditional LUT approach: Memory-bound
result = TADD_LUT[(a << 2) | b]  # 243-entry lookup table

# TritNet approach: Compute-bound, hardware-accelerated
result = tritnet_model(input)  # 2-layer ternary matmul
```

**Core Innovation:**
- Train tiny neural networks with pure ternary weights {-1, 0, +1} on complete truth tables
- Achieve 100% accuracy on balanced ternary arithmetic operations
- Replace memory lookups with matrix multiplication (GPU/TPU friendly)
- Enable hardware acceleration via tensor cores instead of memory access

**Implementation Status - Phases 1-5 Complete (2026-08-18):**
- ✅ Phase 1: Truth table generation for all operations (243 samples for unary, 59,049 for binary)
- ✅ Phase 2: Trained to ≥99% accuracy with ternary weights (tnot/tadd 100%, tmul 99.5%, tmin/tmax 99.9%)
- ✅ Phase 3: C++ inference engine (scalar + AVX2) benchmarked against the LUT — **LUT wins by 169×-195×**, AVX2 recovers only ~10× of that gap
- ✅ Phase 4: GPU (CUDA/PyTorch) batch inference — confirms rather than reverses Phase 3 (best case ~0.10-0.27× of LUT throughput; networks too small to reach GPU-compute-bound at any batch size that fits in VRAM)
- ✅ Phase 5: Learned generalization — errors on the 3 imperfect ops are structured, not noise; a genuinely novel, closed-form-resistant, fully-associative operation was discovered and learned to 99.52%, but a bounded-domain operation always admits a cheap LUT, so the verdict is unchanged
- **Overall verdict:** LUT wins by 1-2 orders of magnitude regardless of operation, hardware target, or how the operation was discovered. See `.claude/CLAUDE.md`'s "TritNet Development" section for the full phase-by-phase writeup and report links.

**Operations:**
- **tnot** - Unary negation (243 samples, 8 hidden neurons)
- **tadd** - Binary addition (59,049 samples, 16 hidden neurons)
- **tmul** - Binary multiplication (59,049 samples, 16 hidden neurons)
- **tmin** - Binary minimum (59,049 samples, 16 hidden neurons)
- **tmax** - Binary maximum (59,049 samples, 16 hidden neurons)

**Architecture:**
```
Input: 5 or 10 trits {-1, 0, +1}
  ↓
Layer 1: TernaryLinear [in → hidden_size]
  Weights: Quantized to {-1, 0, +1}
  ↓
Layer 2: TernaryLinear [hidden_size → hidden_size]
  Weights: Quantized to {-1, 0, +1}
  ↓
Output: TernaryLinear [hidden_size → 5]
  Activation: sign() → {-1, 0, +1}
```

**Usage:**
```bash
# Generate truth tables for all operations
python models/tritnet/src/generate_truth_tables.py --output-dir models/datasets/tritnet

# Train tnot operation (proof-of-concept)
python models/tritnet/src/train_tritnet.py --operation tnot --hidden-size 8

# Train all binary operations (use run_tritnet.py for full workflow)
python models/tritnet/run_tritnet.py --all
```

**Performance (measured, not goals — see `.claude/CLAUDE.md` for dates/platforms):**
- **LUT:** ~517-535 Mops/s (tnot), ~133-149 Mops/s (binary ops) — memory-bound, wins decisively
- **TritNet AVX2 (CPU):** ~2.7-3.4 Mops/s (tnot), ~0.78-1.0 Mops/s (binary) — compute-bound, ~10× over scalar but 169×-195× behind LUT
- **TritNet GPU (RTX 3050, fp16, largest batch fitting in 6GB):** ~65 Mops/s (tnot), ~37 Mops/s (binary) — beats AVX2-CPU by 15-47× but still only 0.10-0.27× of LUT
- **Conclusion:** the honest niche for TritNet, if one exists, is discovering operations without a cheap closed form — not raw throughput on any hardware tried so far (see Phase 5 above)

**Roadmap:**
- Phase 1: Truth table generation ✅ COMPLETE
- Phase 2: Train and validate ≥99% accuracy on all operations ✅ COMPLETE
- Phase 3: C++ integration and benchmarking vs LUT ✅ COMPLETE
- Phase 4: GPU acceleration and batch inference ✅ COMPLETE (GPU only; no TPU path exists in this repo)
- Phase 5: Learned generalization beyond exact truth tables ✅ COMPLETE

**Documentation:**
- **[docs/research/tritnet/TRITNET_ROADMAP.md](docs/research/tritnet/TRITNET_ROADMAP.md)** - Implementation roadmap and technical architecture
- **[docs/research/tritnet/TRITNET_VISION.md](docs/research/tritnet/TRITNET_VISION.md)** - Long-term vision and research goals
- **[models/tritnet/src/](models/tritnet/src/)** - Training scripts and model definitions
- **[models/tritnet/run_tritnet.py](models/tritnet/run_tritnet.py)** - Unified TritNet workflow orchestration

**Why This Matters:**
Moving ternary computing from memory-bound (LUT) to compute-bound (matmul) enables:
- Leveraging $100B+ investment in ML hardware (GPUs, TPUs, tensor cores)
- Batch processing for massive throughput gains
- Discovering learned patterns beyond hand-coded arithmetic
- Path to custom ternary hardware accelerators

## Installation

### Requirements

- **Python** 3.7+
- **Compiler** C++17 (MSVC/GCC/Clang)
- **CPU** x86-64 with AVX2 (Intel Haswell 2013+, AMD Excavator 2015+)
- **Dependencies** pybind11, NumPy

### Build

```bash
pip install pybind11 numpy
python build/build.py
python -c "import ternary_simd_engine; print('Success')"
```

### Manual Compilation

⚠️ **Warning:** Manual compilation commands below are provided for reference. **Prefer `python build/build.py`** (or the other `build/build_*.py` scripts) — it handles per-platform flags correctly and is what CI actually runs. Windows remains the only platform with formal, statistically-validated production benchmark claims; Linux x64 has full test-suite + CI validation (2026-08-11) but benchmark validation is still pending per project standard.

**Windows (MSVC) - VALIDATED:**
```bash
cl /O2 /GL /arch:AVX2 /std:c++17 /EHsc /LD ^
   src/engine/bindings_core_ops.cpp /link /LTCG
```

**Linux/macOS - locally functional, not formally benchmark-validated:**
```bash
c++ -O3 -march=haswell -mavx2 -mfma -fopenmp -flto -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    src/engine/bindings_core_ops.cpp \
    -o ternary_simd_engine$(python3-config --extension-suffix)
```

Note: OpenMP (`-fopenmp`) is enabled by default in `build/build.py` on GCC/Clang (disabled only on ARM and Apple Clang) — validated passing on Linux x64 (2026-07-23) via `tests/python/test_omp.py`. For any real build, use the validated build script: `python build/build.py`

## Usage

### Basic Example

```python
import numpy as np
import ternary_simd_engine as tc

# Encoding constants
MINUS_ONE = 0b00
ZERO      = 0b01
PLUS_ONE  = 0b10

# Create arrays
a = np.array([MINUS_ONE, ZERO, PLUS_ONE], dtype=np.uint8)
b = np.array([PLUS_ONE, ZERO, MINUS_ONE], dtype=np.uint8)

# Operations
result = tc.tadd(a, b)  # [0, 0, 0]
```

### Conversion Helpers

```python
def int_to_trit(value):
    return 0b00 if value < 0 else 0b10 if value > 0 else 0b01

def trit_to_int(trit):
    return -1 if trit == 0b00 else 1 if trit == 0b10 else 0

# Convert integer arrays
values = [-1, 0, 1, -1, 1]
trits = np.array([int_to_trit(v) for v in values], dtype=np.uint8)
result = tc.tadd(trits, trits)
```

## Performance

### Ternary SIMD Engine (AVX2) with Fusion

- **Peak throughput (fusion)**: **45.3 Gops/s** (fused operations @ 1M elements)
- **Peak throughput (element-wise)**: **39.1 Gops/s** (tnot @ 1M elements)
- **Sustained throughput (typical)**: ~20-22 Gops/s
- **Fair baseline vs NumPy (same semantics)**: tadd 1.7–3.5×, fused 1.43× geomean, single ops 0.84× geomean (Linux x64, 2026-08-11)

Performance validated with system load monitoring and statistical rigor.
See [docs/historical/benchmarks/](docs/historical/benchmarks/) for detailed methodology.

> **Note:** Benchmark results are subject to analysis - see methodology note in Overview section.

### Validated Benchmarks (2025-11-28, Windows x64)

**Peak Throughput - Backend AVX2 with Canonical Indexing:**

| Category | Operation | Throughput | Array Size | Notes |
|----------|-----------|------------|------------|-------|
| **Fusion** | fused operations | **45,300 Mops/s** | 1M | Best overall (canonical indexing) |
| **Element-wise** | tnot | **39,100 Mops/s** | 1M | Best non-fusion |
| | tadd | ~21,500 Mops/s | 1M | Stable |
| | tmul | ~21,300 Mops/s | 100K | Stable |

**Peak Performance: 45,300 Mops/s** (45.3 billion operations/second)
**Canonical Indexing Gain: 33%** via dual-shuffle + ADD optimization

*(Mops/s = Million operations/second)*

### Fair Baseline vs NumPy (2026-08-11, Linux x64)

Each operation measured against the fastest NumPy implementation of the
same ternary semantics on int8 (e.g. tadd → `np.clip(a+b,-1,1)`, with
preallocated outputs favoring NumPy), median of 100 repeats, geometric-mean
summary over cells where both sides had CV ≤ 15%
(`benchmarks/python-with-interpreter-overhead/bench_fair_baseline.py`):

| Group | Result | Why |
|-------|--------|-----|
| tadd (saturated add) | **1.7–3.5×** | Saturation costs NumPy an extra pass; free in the LUT |
| tmul / tmin / tmax / tnot | 0.84× geomean (~parity, NumPy slightly ahead) | NumPy int8 ufuncs are already single AVX instructions |
| Fused tnot(op(a,b)) | **1.43× geomean**, up to 6× on tnot∘tadd | One memory pass vs NumPy's 2–3 |
| Memory density | 4× vs INT8 (5 trits/byte with Dense243) | 2-bit encoding |

Measurement notes: the 1M-element transition zone is bimodal on the test
machine (turbo/OpenMP variance) and cells with CV > 15% are excluded from
the geomeans; all per-cell data including exclusions is in
`benchmarks/results/fair_baseline_20260811_104629.json`.

> **Historical note:** earlier releases advertised "8,234× average speedup
> vs pure Python". That baseline is a strawman (any compiled code beats
> interpreted Python by 10³–10⁴×; see benchmarks/SKEPTICAL_METRICS.md) and
> is retained only as this historical footnote.

**Scaling Behavior:**
- Small arrays (1K elements): 500-833 Mops/s (function call overhead dominates)
- Medium arrays (10K elements): 5,263-7,143 Mops/s (L2 cache-resident)
- Large arrays (100K elements): 21,277-29,412 Mops/s (peak regular throughput)
- Very large (1M elements): 17,621-37,244 Mops/s (OpenMP effective, fusion shines)
- Huge arrays (10M elements): 6,578-8,608 Mops/s (memory bandwidth limited)

### Competitive Analysis vs NumPy INT8 (re-validated 2026-08-18, Phase 4 updated 2026-08-20, Linux x64)

**⚠️ SUPERSEDES the numbers below this note — the 6-phase competitive suite was re-run under a verified-clean environment (`PYTHONPATH` unset, cwd outside the repo) 2026-08-18, closing a caveat that the phase 1-4 numbers this section originally shipped with had never proven themselves independent of a forgiving local `PYTHONPATH`. Phase 4 specifically was re-measured again 2026-08-20 after a GEMM kernel fix (see row below). See `.claude/CLAUDE.md` Critical Gap #3 for the full history.**

| Phase | Result | Notes |
|:------|:-------|:------|
| 1. Arithmetic vs NumPy INT8 | 0.70×/0.69× avg | "Needs work" — engine ~parity with NumPy on single ops, not a clean win |
| 2. Memory efficiency | **4.0× vs INT8** (exact match to README claim) | ✅ Proven |
| 3. Throughput @ equivalent bit-width | Dense243 8.0× faster than an INT2 reference | Real kernel-vs-kernel comparison, sized to a genuine 1GB footprint |
| 4. Neural workload (matmul) | **~1.1×-1.2× avg** (was 0.189×) | "Viable for AI" — `DenseWeights` dense-packed kernel replaced the CSC/CSR `ZeroSkipWeights` kernel this phase used before; see note below |
| 5. Model quantization | **Real measurement, TinyLlama-1.1B (2026-08-22)** | Memory: ✅ 4.82× smaller than fp16 (PASS <25% target). Accuracy: ❌ perplexity 12.8 → 89,101 — naive post-training ternary quantization (no fine-tuning/QAT), as literally specified by this project's own "simple threshold-based" scheme, catastrophically breaks the model. See note below |
| 6. Power consumption | Real measurement code (2026-08-22) | Auto-detects a real hardware monitor (Intel RAPL / NVIDIA `nvidia-smi` / Windows) where available and reports honestly if none is (falls back to a clearly-labeled simulated `MockPowerMonitor`, never silently) — see note below |

**Commercial viability: 2/5 criteria validated** — see the table further below. This is a confirmed, re-verified number, not a first-pass estimate. The Phase 4 improvement does **not** change this count — it fixes a stale/wrong matmul figure, but the "< 2× FP16" criterion Phase 4 is a proxy for compares against fp32 NumPy here, not fp16, so it still isn't a direct measurement of that specific criterion. Phase 5 now producing a real result also does **not** change this count — its accuracy criterion fails decisively (see below), though its memory criterion genuinely passes. Phase 6 being wired to real measurement code also does **not** change this count on its own — a run needs an actual reachable hardware power monitor to produce a citable number, which this project's own dev sandbox does not have (confirmed: no RAPL read permission, no `perf` access, no sudo); the fix means Phase 6 will produce a genuine number the first time it's run somewhere that does have one, instead of remaining unwired.

**Phase 4 GEMM kernel fix (2026-08-20):** investigation found the CSC/CSR "zero-skip" kernel wasn't naive (already AVX2 + OpenMP + cache-aware) but had two structural problems: it vectorizes over the batch dimension while Phase 4 tests batch=1 (so AVX2 never engaged), and at ternary's real ~33-40% zero density, CSC/CSR index storage is ~3.3× *larger* than just the dense int8 weight array — "zero-skip" was optimizing the wrong resource on a memory-bandwidth-bound kernel. A new dense-packed kernel (`src/core/simd/ternary_gemm_dense.h`, `DenseWeights` in `ternary_zero_skip_gemm`) fixes both: measured **19×-32× faster than the best existing kernel at batch=1**, native and pybind-free (`benchmarks/cpp-native-kernels/bench_gemm_dense.cpp`). Fixing this also surfaced a benchmark-methodology bug (a 3-call warmup that was fine for the old ~1-5ms/call kernel left this machine's CPU frequency-scaling cold for the new ~10-30µs/call one, causing up to 50× run-to-run variance) — fixed with wall-clock warmup and interleaved median timing; stable across repeated runs. Full investigation: [reports/2026-08-20/GEMM_DENSE_PACKED_OPTIMIZATION.md](reports/2026-08-20/GEMM_DENSE_PACKED_OPTIMIZATION.md).

**Phase 5 model quantization, real result (2026-08-22):** `benchmarks/model_quantization/quantize_tinyllama.py` quantizes every attention/MLP weight in TinyLlama-1.1B-Chat-v1.0 to ternary (BitNet-style per-tensor absmean scale — exactly this project's own documented "simple threshold-based" scheme, no fine-tuning/QAT/calibration) and measures WikiText-2 perplexity before/after. Result: **memory genuinely drops 4.82× (2200→456MB for the quantized layers, passes the <25%-of-fp16 target)**, but **perplexity explodes from 12.78 to 89,101** (both cross-entropy values finite and sanity-checked, not a NaN/Inf bug) — a decisive failure of the accuracy criterion. This is a known, expected regime, not a surprise on inspection: published ternary/1.58-bit techniques train from scratch with quantization-aware training; naively post-training-quantizing an already-converged checkpoint with no per-channel scaling, calibration, or retraining is a much harsher setting, and this result is what that literature would predict. Zero fraction across quantized layers (33.0% mean, 31.2-53.7% range) matches this project's own established ~33-40% ternary sparsity figure — a real internal consistency check, not a coincidence. Full writeup: [reports/2026-08-22/PHASE5_TINYLLAMA_QUANTIZATION.md](reports/2026-08-22/PHASE5_TINYLLAMA_QUANTIZATION.md).

**Phase 6 power measurement wiring (2026-08-22):** `bench_competitive.py`'s Phase 6 previously only printed a static description; now it calls the real (and bugfixed) `PowerConsumptionBenchmark`/monitor stack in `bench_power_efficiency.py`. Along the way, found and fixed a real silent-failure bug: `IntelRAPLMonitor.is_available()` checked only that the RAPL directory existed, not that its `energy_uj` counter was actually readable — root-only by default on a stock Linux install (confirmed concretely on this machine), the common unprivileged-user case, not a sandbox quirk. It would have silently reported "available" and measured 0.0J for everything behind one easy-to-miss warning; fixed to attempt a real read, so it now honestly falls through to a clearly-labeled `MockPowerMonitor` instead. Also fixed an independent RAPL counter-wraparound bug in `get_energy_joules()`. Full writeup: [reports/2026-08-22/PHASE6_POWER_MEASUREMENT_WIRED_UP.md](reports/2026-08-22/PHASE6_POWER_MEASUREMENT_WIRED_UP.md).

For the honest, apples-to-apples single-op comparison (same ternary semantics, fair NumPy baseline rather than a strawman), see [Fair Baseline vs NumPy](#fair-baseline-vs-numpy-2026-08-11-linux-x64) above: **tadd 1.7-3.5×, fused ops up to 6×, single non-saturating ops ~parity with NumPy (0.84× geomean)**.

### Operation Fusion (Phase 4.0 - Validated)

**Fused Operations** combine multiple operations into a single pass, reducing memory traffic:

**fused_tnot_tadd** - Validated speedup (rigorous benchmarking):
- **Contiguous arrays:** 1.80× to 4.78× speedup
- **Non-contiguous arrays:** 1.78× to 15.52× speedup
- **Cold cache:** 1.62× to 2.56× speedup
- **Conservative estimate:** 1.94× minimum speedup

Performance validated with statistical rigor (variance, confidence intervals, coefficient of variation).

### Latency (per element)

| Implementation | Time | CPU Cycles |
|----------------|------|------------|
| Python | 10 ns | ~30 |
| C++ LUT | 0.5 ns | ~2 |
| **C++ SIMD** | **0.077 ns** | **~0.23** |
| **C++ Fused** | **0.040 ns** | **~0.12** |

## Architecture

### Project Structure (current — see `.claude/CLAUDE.md` "Code Organization" for the authoritative version)

```
src/core/                  # Production-ready kernel (mathematically stable)
├─ algebra/                # Core ternary operations
│   ├─ ternary_algebra.h      # Scalar operations + LUTs
│   └─ ternary_lut_gen.h      # Compile-time LUT generation
├─ simd/                   # SIMD acceleration
│   ├─ ternary_simd_kernels.h # AVX2 vectorization
│   ├─ ternary_cpu_detect.h   # Runtime CPU detection
│   ├─ ternary_fusion.h       # Operation fusion
│   └─ backend_*.{h,cpp}      # Pluggable Scalar/AVX2_v1/AVX2_v2 backend system
├─ ffi/                    # Cross-language FFI
├─ common/                 # Error types, shared utilities
├─ config/, packing/, profiling/
└─ core_api.h              # Unified entry point

src/engine/                # Python bindings and library code
├─ bindings_core_ops.cpp          # Core SIMD ops (ternary_simd_engine)
├─ bindings_dense243.cpp          # Dense243 encoding (ternary_dense243_module)
├─ bindings_tritnet_gemm.cpp      # TritNet GEMM
├─ bindings_zero_skip_gemm.cpp    # Zero-skip ternary GEMM (used by competitive bench Phase 4)
├─ bindings_backend_api.cpp       # Pluggable backend system (ternary_backend)
├─ bindings_tritnet_inference.cpp # TritNet C++ inference engine, scalar+AVX2
├─ py_array_validate.h            # Shared GEMM input-validation helpers
└─ lib/dense243/                  # High-density encoding library

models/tritnet/            # TritNet training pipeline (Phases 1-5, all complete)
├─ src/                    # Truth tables, ternary layers, model architectures
├─ qat_common.py           # Shared QAT training code
├─ inference/              # C++ inference engine (scalar + AVX2), weight export
└─ phase{4,5}_*.py         # GPU benchmark, error characterization, novel-op discovery

build/                     # Build scripts (build.py, build_dense243.py,
                            # build_backend.py, build_zero_skip_gemm.py,
                            # build_tritnet_gemm.py, build_tritnet_inference.py,
                            # build_pgo*.py, build_all.py, clean_all.py)

benchmarks/                # Competitive analysis suite
tests/                     # tests/run_tests.py (unified runner), python/, cpp/
docs/                      # API reference and architecture documentation
opentimestamps/            # IP protection (timestamp_create.py, timestamp_verify.py)
reports/                   # Dated session/validation reports (source of truth for history)
```

**Total kernel + bindings implementation:** ~9,700 lines of validated C++17 code (src/core/ + src/engine/)

### Intellectual Property Protection

**OpenTimestamps SHA512-based IP protection system (Added 2025-11-23)**

```bash
# Generate IP protection timestamp for snapshot (runs immediately -- no --help/--dry-run)
python opentimestamps/timestamp_create.py

# Verify existing timestamp
python opentimestamps/timestamp_verify.py opentimestamps/timestamps/manifest_YYYYMMDD_HHMMSS.json.ots
```

**How it works:**
- Creates SHA512 hash of all source files (88 files tracked)
- Submits hash to OpenTimestamps Bitcoin blockchain
- Generates verifiable proof of existence at specific date/time
- Immutable, tamper-proof record of IP creation date

**Timestamped snapshots:**
- **2025-11-23 (ce39331):** Initial snapshot - 88 files including TritNet Phase 1, competitive benchmarks, Dense243

**Purpose:** Establishes provable date of invention for patent applications and IP disputes

**Documentation:** See `.ots` files in `timestamps/` directory and OpenTimestamps verification tools

### Design Layers

**Layer 0**: Constexpr LUT generation - Compile-time table construction
**Layer 1**: Scalar operations - Branch-free lookup table operations
**Layer 2**: SIMD vectorization - 32-wide parallel processing via AVX2
**Layer 3**: Python bindings - Zero-copy NumPy integration
**Layer 4**: Runtime safety - CPU detection, alignment validation, ISA dispatch

## Kernel Architecture Deep Dive

### Trit Encoding: 2-Bit Representation

**Core Concept**: Each balanced ternary trit {-1, 0, +1} is encoded in 2 bits:

```
Value    | Binary | Decimal
---------|--------|--------
   -1    |  0b00  |   0
    0    |  0b01  |   1
   +1    |  0b10  |   2
 (invalid)| 0b11  |   3 (reserved/undefined)
```

**Why 2 bits?**
- Minimum bits needed to represent 3 states (log₂(3) ≈ 1.58, round up to 2)
- Enables efficient SIMD operations via byte-level shuffles
- Wastes 25% of bit space (3/4 states used) but optimizes for CPU instructions
- Alternative: Dense243 packing (5 trits/byte) trades CPU efficiency for storage density

**Memory Layout Example**:
```
Array: [-1, 0, +1, -1]
Bytes: [0b00, 0b01, 0b10, 0b00]
Memory: 4 bytes (1 trit/byte)
```

### Dense243 Encoding: 5 Trits per Byte

**Mathematical Foundation**: 3⁵ = 243 states < 256 (1 byte capacity)

**Base-3 Positional Encoding**:
```
packed_byte = trit[0]×(3⁰) + trit[1]×(3¹) + trit[2]×(3²) +
              trit[3]×(3³) + trit[4]×(3⁴)

Where each trit ∈ {0, 1, 2} (mapped from {-1, 0, +1})
```

**Example Encoding**:
```
Input trits:  [-1,  0, +1, +1,  0]
Map to 0-2:   [ 0,  1,  2,  2,  1]
Calculate:     0×1 + 1×3 + 2×9 + 2×27 + 1×81
             = 0 + 3 + 18 + 54 + 81
             = 156 (stored as single byte 0x9C)
```

**Unpacking Algorithm**:
```python
def dense243_unpack(byte_value):
    trits = []
    remainder = byte_value
    for i in range(5):
        trit_012 = remainder % 3  # Extract trit in [0,1,2]
        trits.append(trit_012)
        remainder //= 3           # Divide by base-3
    return trits  # [-1,0,+1] after remapping
```

**Space Savings**:
- **Standard 2-bit**: 5 trits = 5 bytes (1 trit/byte)
- **Dense243**: 5 trits = 1 byte (5 trits/byte)
- **Compression**: 80% space reduction
- **Density**: 95.3% utilization (243/256 states used)

**Performance Trade-offs**:
```
Operation     | 2-bit   | Dense243  | Ratio
--------------|---------|-----------|-------
Pack (5 trits)| N/A     | 0.25 ns   | -
Unpack        | N/A     | 0.91 ns   | -
Storage       | 5 bytes | 1 byte    | 5.0×
SIMD ops      | 32/vec  | Scalar    | 0.03×
```

**Implementation** (`src/engine/dense243/ternary_dense243.h`):
- Compile-time LUT generation for fast div/mod by 3
- Constexpr base-3 arithmetic
- All 243 states validated in comprehensive test suite

### TriadSextet Encoding: 3+3 Trits Split

**Design**: Split 6 trits into two 3-trit groups (triads), each encoded separately

**Mathematics**: 3³ = 27 states < 32 (5 bits capacity)

**Encoding Structure**:
```
┌─────────────────────────────────┐
│  Byte (8 bits)                  │
├──────────────────┬──────────────┤
│ Triad 1 (5 bits) │ Triad 2 (3b) │
│  trits [0,1,2]   │ trits [3,4,5]│
└──────────────────┴──────────────┘
```

**Packed Layout**:
```
Bit positions:  7  6  5  4  3  2  1  0
               [  Triad 1   ][ Tri2  ]
                5 bits used   3 bits overflow!
```

**Problem**: 5+5 = 10 bits needed, but only 8 bits available!

**Solution**: Use 2 bytes for 2 triads
```
Byte 0: [  5 bits: triad 0   ][3 bits: triad 1 (LSBs)]
Byte 1: [  2 bits: triad 1 (MSBs) ][5 bits: triad 2  ][ unused ]
```

**Actual Implementation** (Optimized):
```cpp
// Pack 6 trits → triadsextet_t (single uint16_t)
triadsextet_t pack_triadsextet(uint8_t t[6]) {
    // First triad (trits 0-2): Base-3 encoding
    uint8_t triad0 = t[0] + t[1]*3 + t[2]*9;  // 0-26

    // Second triad (trits 3-5): Base-3 encoding
    uint8_t triad1 = t[3] + t[4]*3 + t[5]*9;  // 0-26

    // Combine: triad0 in bits [0-4], triad1 in bits [5-9]
    return (triad1 << 5) | triad0;  // 10 bits used of 16
}
```

**Space Efficiency**:
- **Theoretical**: 6 trits = 10 bits (1.67 bits/trit)
- **Actual**: 6 trits = 2 bytes = 16 bits (2.67 bits/trit)
- **Density**: 62.5% utilization (10/16 bits used)
- **vs Standard**: 6 bytes → 2 bytes = 3× compression
- **vs Dense243**: Less dense (62.5% vs 95.3%) but faster pack/unpack

**Performance**:
```
Operation        | Time (ns) | Note
-----------------|-----------|---------------------------
Pack (6 trits)   | 0.16 ns   | 5.6× faster than Dense243
Unpack (6 trits) | 0.66 ns   | 1.4× faster than Dense243
```

**Use Cases**:
- Intermediate format between 2-bit and Dense243
- When pack/unpack speed matters more than storage
- Hardware implementations with 16-bit registers

**Implementation** (`src/engine/dense243/ternary_triadsextet.h`):
- Validated all 27³ = 19,683 state combinations
- Optimized div/mod-3 operations via compile-time LUTs
- Integrated with Dense243 for flexible encoding strategies

### SIMD Kernel: AVX2 Vectorization

**Core Technique**: Lookup Table Shuffle with `_mm256_shuffle_epi8`

**Algorithm**:
```cpp
// Pre-computed 16-byte LUT for operation (e.g., TADD)
alignas(16) uint8_t TADD_LUT[16] = {
    // Index: (a << 2) | b → result
    0b10, 0b01, 0b10, 0b11,  // a=-1: tadd(-1,-1)=+1, ...
    0b01, 0b01, 0b10, 0b11,  // a= 0: tadd( 0,-1)= 0, ...
    0b10, 0b10, 0b10, 0b11,  // a=+1: tadd(+1,-1)=+1, ...
    0b11, 0b11, 0b11, 0b11   // Invalid entries
};

// SIMD operation (32 trits in parallel)
__m256i tadd_simd(__m256i a, __m256i b) {
    // Build lookup indices: (a << 2) | b
    __m256i hi = _mm256_slli_epi16(a, 2);  // Shift a left by 2
    __m256i indices = _mm256_or_si256(hi, b); // Combine with b

    // Broadcast 16-byte LUT to 32-byte vector
    __m128i lut_128 = _mm_loadu_si128((__m128i*)TADD_LUT);
    __m256i lut_256 = _mm256_broadcastsi128_si256(lut_128);

    // Parallel lookup: 32 lookups in single instruction!
    return _mm256_shuffle_epi8(lut_256, indices);
}
```

**Why This Works**:
1. **2-bit encoding** → max index = (0b10 << 2) | 0b10 = 0b1010 = 10 < 16
2. **All indices fit in 4 bits** → perfect for byte shuffle
3. **32 bytes per AVX2 vector** → 32 parallel operations
4. **Single instruction latency** → ~3 cycles on modern CPUs

**Memory Layout**:
```
Input arrays (aligned to 32 bytes):
a: [trit₀, trit₁, ..., trit₃₁] (32 bytes)
b: [trit₀, trit₁, ..., trit₃₁] (32 bytes)

AVX2 loads:
__m256i va = _mm256_load_si256(a);  // Load 32 trits
__m256i vb = _mm256_load_si256(b);  // Load 32 trits

Result:
__m256i vr = tadd_simd(va, vb);     // Process all 32
```

**Performance Breakdown**:
```
Operation              | Cycles | Notes
-----------------------|--------|------------------------
Shift (_mm256_slli)    | 1      | Instruction-level parallelism
OR (_mm256_or)         | 1      | Can execute in parallel
Broadcast              | 1-3    | Depends on μarch
Shuffle (_mm256_shuffle)| 1     | Single-cycle on modern CPUs
Total latency          | ~3-5   | Pipeline overlaps
Throughput             | 0.077 ns/trit | 32 trits per ~2.5ns
```

**Comparison vs Scalar**:
```
Method          | ns/trit | Speedup
----------------|---------|--------
Python loop     | 10.0    | 1×
C++ scalar LUT  | 0.5     | 20×
C++ SIMD AVX2   | 0.077   | 130×
C++ Fused SIMD  | 0.040   | 250×
```

**Implementation** (`src/core/simd/ternary_simd_kernels.h`):
- Template-based for all operations (tadd, tmul, tmin, tmax, tnot)
- Runtime CPU detection (AVX2 check, graceful fallback)
- Alignment validation (32-byte boundaries for streaming stores)
- OpenMP parallelization for arrays ≥100K elements

### Kernel Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│ Layer 4: Python Bindings (pybind11)                    │
│  - NumPy array ↔ C++ uint8_t* zero-copy                │
│  - Exception translation, GIL management                │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 3: Runtime Dispatch & Safety                     │
│  - CPU feature detection (AVX2, alignment)              │
│  - Array size routing (SIMD threshold: 1024 elements)   │
│  - OpenMP parallelization (threshold: 100K elements)    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 2: SIMD Vectorization (AVX2)                     │
│  - Process 32 trits per instruction                     │
│  - LUT-based via _mm256_shuffle_epi8                    │
│  - Streaming stores for large arrays                    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 1: Scalar Operations (Branch-Free LUT)           │
│  - Compile-time LUT generation (constexpr)              │
│  - 16-entry tables for each operation                   │
│  - Used for: tail elements, small arrays, fallback      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 0: Mathematical Specification                    │
│  - Pure functions: tadd(-1,+1)=0, tmul(+1,-1)=-1        │
│  - Truth tables (9 entries for binary, 3 for unary)     │
│  - Validated against balanced ternary algebra           │
└─────────────────────────────────────────────────────────┘
```

**Execution Flow Example** (tadd with 100K elements):
```
Python: tc.tadd(a, b)
  ↓
Layer 4: Extract NumPy pointers, validate shapes
  ↓
Layer 3: Detect AVX2 ✓, size=100K → enable OpenMP
  ↓
Layer 2: Split into 8 threads, each processes:
         - Main loop: 3,125 SIMD iterations (32 elements each)
         - Tail loop: Handle remaining elements
  ↓
Layer 1: Tail elements use scalar LUT (< 32 elements)
  ↓
Result: 100K results in ~9.1 μs (11,000 Mops/s)
```

### Implementation Files

**Core Kernel** (`src/core/`):
- `algebra/ternary_lut_gen.h` (181 lines) - Compile-time LUT generation
- `algebra/ternary_algebra.h` (200 lines) - Scalar operations
- `simd/simd_avx2_32trit_ops.h` (117 lines) - AVX2 vectorization
- `simd/cpu_simd_capability.h` (185 lines) - Runtime CPU detection
- `simd/fused_binary_unary_ops.h` (252 lines) - Operation fusion
- `simd/backend_*.{h,cpp}` - Pluggable Scalar/AVX2_v1/AVX2_v2 backend system
- `common/ternary_errors.h` (164 lines) - Error handling
- `core_api.h` (87 lines) - Unified API

**High-Density Encodings** (`src/engine/dense243/`):
- `ternary_dense243.h` (260 lines) - Dense243 pack/unpack
- `ternary_dense243_simd.h` (411 lines) - SIMD-accelerated Dense243
- `ternary_triadsextet.h` (397 lines) - TriadSextet encoding

**Python Bindings** (`src/engine/`):
- `bindings_core_ops.cpp` (580 lines) - Main SIMD operations
- `bindings_dense243.cpp` (370 lines) - Dense243 module
- `bindings_tritnet_gemm.cpp` (343 lines) - TritNet GEMM
- `bindings_backend_api.cpp` (328 lines) - Pluggable backend system
- `bindings_zero_skip_gemm.cpp` (247 lines) - Zero-skip ternary GEMM
- `bindings_tritnet_inference.cpp` (189 lines) - TritNet C++ inference engine

**Total kernel + bindings**: ~9,700 lines of validated C++17 code (`src/core/` + `src/engine/`)

### Deployment Status

✅ **Production-Ready** (src/core/, Windows x64 only):
- Core algebra system (16 test functions, all passing)
- SIMD kernels (AVX2, validated 2025-11-28)
- CPU feature detection (runtime ISA dispatch)
- C FFI layer (cross-language ready)
- Operation fusion (7-35× validated speedup)
- Canonical indexing optimization (33% SIMD improvement)
- Performance validated: 45,300 Mops/s peak throughput

✅ **Validated & Ready** (`src/engine/lib/dense243/`):
- **Dense243 encoding** (all 243 states validated, 0.25 ns pack, 0.91 ns unpack)
- **TriadSextet encoding** (all 27 states validated, 0.16 ns pack, 0.66 ns unpack)
- **fused_tnot_tadd** (rigorous benchmarks: 1.94× conservative, up to 15.52× speedup, Windows x64 2025-10-29)

✅ **Phase 4.1 fusion operations** (`fused_tnot_tmul`/`tmin`/`tmax`) - re-validated on Linux x64 native C++ (2026-08-18): 1.00×-2.89× speedup range, ~1.6× average, every measured cell beat 1.0×. The 2025-10-29 Windows table's higher tmin/tmax ceiling (9-11×) wasn't reproduced here — not asserted as a regression given the different, unverified original hardware; both numbers are kept side by side in `src/core/simd/docs/FUSION.md`.

See dated validation reports in [reports/](reports/) (the authoritative history of every fix, benchmark, and re-validation).

## Testing

```bash
# Run all tests (unified test runner)
python tests/run_tests.py

# Run individual test suites
python tests/python/test_phase0.py     # Correctness
python tests/python/test_omp.py         # OpenMP scaling
python tests/python/test_errors.py      # Error handling

# Performance benchmarks
python benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py
```

See **[tests/README.md](tests/README.md)** for comprehensive testing and CI/CD documentation.

## Competitive Benchmarking Suite

**Prove whether ternary has commercial value by comparing against industry standards**

Comprehensive 6-phase benchmark suite comparing ternary operations against NumPy INT8, INT4, FP16, and real quantized models.

### Quick Start

```bash
# Run full competitive benchmark suite (6 phases)
python benchmarks/python-with-interpreter-overhead/bench_competitive.py --all

# Run specific phase
python benchmarks/python-with-interpreter-overhead/bench_competitive.py --phase 1  # vs NumPy
python benchmarks/python-with-interpreter-overhead/bench_competitive.py --phase 4  # Neural workloads
python benchmarks/python-with-interpreter-overhead/bench_competitive.py --phase 5  # Model quantization

# Generate visualization report
python benchmarks/utils/visualization.py results/competitive_results_*.json
```

### Benchmark Phases

**Phase 1: Arithmetic Operations vs NumPy INT8**
- Direct performance comparison at equivalent information density
- Measures operations/second, throughput (GB/s), speedup
- **Goal:** Prove ternary is competitive or faster than NumPy INT8

**Phase 2: Memory Efficiency Analysis**
- Compare storage requirements for 7B, 13B, 70B parameter models
- Targets: FP16 (baseline), INT8, INT4, Ternary (2-bit), Dense243 (1.6-bit)
- **Result:** 8× smaller than FP16, 4× smaller than INT8

**Phase 3: Throughput at Equivalent Bit-Width**
- Operations/second when memory footprint is equal (1GB target)
- Real competition: Ternary (2-bit) vs INT2 (2-bit) vs INT4 (4-bit)
- **Goal:** Prove ternary outperforms other ultra-low bit schemes

**Phase 4: Neural Network Workload Patterns**
- Matrix operations typical in AI (512×512, 2048×2048, 4096×4096, 8192×1024)
- Simulates actual inference patterns (matmul, activations, batching)
- **Critical:** Must achieve >0.5× NumPy performance to be viable for AI

**Phase 5: Real Model Quantization**
- Quantize pre-trained models (TinyLlama-1.1B, Phi-2, Gemma-2B) to ternary
- Measure perplexity degradation, accuracy, inference latency, memory
- **Success:** <5% accuracy loss, <2× latency, <25% memory vs FP16

**Phase 6: Power Consumption**
- Energy efficiency (operations/Joule) on x86, ARM, GPU
- Platforms: Intel RAPL, nvidia-smi, USB power meters
- **Expected:** 2-4× lower power consumption vs INT8

### Commercial Viability Criteria

**What proves we have a product:**

| Criterion | Target | Status |
|:----------|:-------|:-------|
| Memory efficiency at same capacity | 4× vs INT8 | ✅ **PROVEN** (4.0× validated) |
| Throughput at equivalent bit-width | > INT2 | ✅ **PROVEN** (Dense243 8.0× faster than a real INT2 reference) |
| Inference latency in real models | < 2× FP16 | ⚠️ ~1.1×-1.2× avg vs fp32 NumPy matmul (dense-packed kernel, 2026-08-20 — was 0.189× against the CSC/CSR kernel) — genuinely improved, but not a measurement of fp16 specifically, so this criterion stays unproven either way |
| Power consumption on edge | 2-4× better | ⚠️ Needs hardware |
| Accuracy retention after quantization | < 5% loss | ❌ Tested 2026-08-22 (TinyLlama-1.1B, naive post-training quantization) — perplexity degraded +697,074%, decisive fail as literally specified; QAT/fine-tuning or per-channel scaling would be needed for a fairer test, not attempted |

**Current Status:** 2/5 criteria validated (40%) — re-confirmed 2026-08-18 under a verified-clean environment, see `.claude/CLAUDE.md` Critical Gap #3 for the full history.

**Latest Full Results:** [reports/archive/benchmarks/2025-11-23/BENCHMARK_SUMMARY.md](reports/archive/benchmarks/2025-11-23/BENCHMARK_SUMMARY.md) (original 2025-11-23 run; see [reports/2026-08-18/COMPETITIVE_BENCHMARK_REVALIDATION.md](reports/2026-08-18/COMPETITIVE_BENCHMARK_REVALIDATION.md) for the current re-validated numbers)

### Results Structure

```json
{
  "metadata": {
    "timestamp": "2025-11-23T...",
    "platform": "win32",
    "numpy_version": "1.24.0"
  },
  "phase1_arithmetic_comparison": {
    "size": [1000, 10000, 100000, 1000000],
    "ternary_add_ns": [...],
    "numpy_int8_add_ns": [...],
    "speedup": [...]
  },
  "phase2_memory_efficiency": {...},
  "phase4_neural_workload_patterns": {...},
  "phase5_model_quantization": {...}
}
```

### Installation Requirements

**Core (Phases 1-4):**
```bash
pip install numpy matplotlib
```

**Model Quantization (Phase 5):**
```bash
pip install torch transformers pyarrow
```
(`pyarrow` is for fetching the WikiText-2 perplexity corpus directly, without pulling in the full `datasets` package.)

**Power Monitoring (Phase 6):**
- Intel RAPL: Linux with `/sys/class/powercap/intel-rapl/` access
- NVIDIA: `nvidia-smi` installed
- ARM: USB power meter hardware

### Documentation

- **[benchmarks/COMPETITIVE_BENCHMARKS.md](benchmarks/COMPETITIVE_BENCHMARKS.md)** - Complete suite documentation
- **[benchmarks/README.md](benchmarks/README.md)** - Standard benchmark documentation

## Documentation

**Core Documentation:**
- **[tests/README.md](tests/README.md)** - Testing and CI/CD guide
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development guidelines
- **[CHANGELOG.md](CHANGELOG.md)** - Version history
- **[docs/](docs/)** - Complete API reference and architecture docs
- **[docs/build-system/README.md](docs/build-system/README.md)** - Build system documentation
- **[tests/README.md](tests/README.md)** - Test suite documentation

**TritNet (Neural Network-Based Arithmetic):** ⭐ New!
- **[docs/research/tritnet/TRITNET_ROADMAP.md](docs/research/tritnet/TRITNET_ROADMAP.md)** - Implementation roadmap and technical architecture (phase numbering superseded — see `.claude/CLAUDE.md`'s "TritNet Development" section for current phase status)
- **[docs/research/tritnet/TRITNET_VISION.md](docs/research/tritnet/TRITNET_VISION.md)** - Long-term vision and research goals
- **[models/tritnet/src/](models/tritnet/src/)** - Training scripts and model definitions

**Competitive Benchmarking:** ⭐ New!
- **[benchmarks/COMPETITIVE_BENCHMARKS.md](benchmarks/COMPETITIVE_BENCHMARKS.md)** - 6-phase competitive benchmark suite
- **[benchmarks/README.md](benchmarks/README.md)** - Standard benchmark documentation

## Current Limitations & Status

### Validated & Production-Ready (Windows x64)

**✅ What Works Excellently:**
- Element-wise operations (tadd, tmul, tmin, tmax, tnot)
- 45.3 Gops/s peak throughput with fusion, 39.1 Gops/s element-wise
- Fair baseline vs NumPy: fused 1.43×, tadd 1.7–3.5× (2026-08-11)
- 4× memory advantage over INT8, 8× over FP16
- Operation fusion (7-35× speedup)
- Canonical indexing (33% SIMD improvement)
- Dense243 high-density encoding
- Build system and benchmarking infrastructure

**Use Cases Ready for Production:**
- ✅ Modulo-3 arithmetic and number theory
- ✅ Fractal generation with ternary coordinates
- ✅ Memory-constrained embedded systems
- ✅ Element-wise array operations
- ✅ Edge detection algorithms (experimental POC)

### Known Limitations & Ongoing Work

**Platform Support:**
- ✅ **Windows x64**: Production-ready (validated 2025-11-28)
- 🟡 **Linux x64**: Test suite + CI validated (2026-08-11); formal benchmark validation pending (project standard requires statistical benchmark reports before production claims)
- ⚠️ **ARM/NEON**: Not yet supported (planned for future)

**Technical Constraints:**
- **Arrays**: any shape/dimensionality (multi-dimensional support added 2026-08-18 — `process_binary_array`/`process_unary_array` generalized from 1D-only; output preserves input shape, both inputs must be C-contiguous)
- **CPU requirement**: AVX2 instruction set (Intel Haswell 2013+, AMD Excavator 2015+)
  - Module performs runtime detection and fails gracefully on unsupported CPUs
- **Size matching**: Binary operations require identical array shapes (a dedicated `ArrayShapeMismatchError` distinguishes "same size, different shape" from a plain size mismatch)
- **Invalid encoding**: 0b11 is reserved/undefined
- **Alignment**: Streaming stores require 32-byte alignment (automatically detected)

**AI/ML Workload — Matrix Multiplication Status:**

The original GEMM v1.0.0 (derived from BitNet b1.58) described in earlier releases of this README was superseded by two different, better-validated paths:
- **`ternary_zero_skip_gemm`** — two kernel strategies for ternary matmul, used by the competitive benchmark's Phase 4. `ZeroSkipWeights` (original, CSC/CSR sparse index) measured 0.189× avg vs NumPy ("too slow for AI") — investigated 2026-08-20 and found to be memory-bandwidth-bound with an index that's actually *larger* than the dense weight array at ternary's real sparsity, and SIMD that never engages at Phase 4's batch=1. `DenseWeights` (added 2026-08-20, now the kernel Phase 4 uses) fixes both: **~1.1×-1.2× avg vs NumPy** ("viable for AI" by the suite's own threshold), 19×-32× faster than `ZeroSkipWeights` at batch=1 specifically. See the Competitive Analysis table above and [reports/2026-08-20/GEMM_DENSE_PACKED_OPTIMIZATION.md](reports/2026-08-20/GEMM_DENSE_PACKED_OPTIMIZATION.md).
- **TritNet's own GEMM/inference path** (Phases 1-5, complete 2026-08-18) — explored whether a learned-matmul approach could out-run a LUT for ternary arithmetic itself; conclusion was a decisive no (LUT wins by 169×-195× even against AVX2-vectorized TritNet, and by 46×-58× against direct closed-form GPU arithmetic). See the TritNet section above.

**What This Means:**
- ✅ **Excellent for element-wise operations** - 45,300 Mops/s peak validated (fused), 39,100 Mops/s (element-wise), Windows x64
- ✅ **Proven memory advantage** - 4× smaller than INT8, Dense243 format working
- ✅ **Matrix multiplication improved substantially** - ~1.1×-1.2× avg vs NumPy with the dense-packed kernel (2026-08-20), up from 0.189×; genuinely re-measured on the same shapes/batch size, not a tuning claim
- ⚠️ **Still not a full "AI-ready" claim** by this project's own commercial-viability criteria (2/5 validated) — this specific figure compares against fp32 NumPy, not the fp16 baseline that criterion actually names, and power/accuracy-retention criteria remain unmeasured

**Historical root-cause analysis:** `reports/performance/gemm_gap_root_cause.md` (statistical analysis of the original GEMM v1.0.0 gap — missing SIMD/OpenMP/cache-blocking; superseded first by the zero-skip kernel, then by the dense-packed kernel above, kept as historical record).

## Advanced Features

### Profile-Guided Optimization

Additional 5-15% performance gain using Clang PGO (recommended) or MSVC fallback:

```bash
# Clang PGO (recommended - works with Python extensions)
python build/build_pgo_unified.py --clang

# Auto-detect (prefers Clang if available)
python build/build_pgo_unified.py

# MSVC fallback (has known limitations)
python build/build_pgo.py full
```

See [docs/pgo/README.md](docs/pgo/README.md) and [docs/pgo/CLANG_INSTALLATION.md](docs/pgo/CLANG_INSTALLATION.md) for details.

### Compile-Time Options

```cpp
// Disable input sanitization for validated data pipelines (3-5% gain)
#define TERNARY_NO_SANITIZE
```

## Roadmap

**Status Reality Check:** `.claude/CLAUDE.md` is this project's actively-maintained, dated source of truth for what's done vs. pending — it has a fuller and more current changelog than this section. What follows is a summary, current as of 2026-08-18.

**Completed:**

**Core Engine:**
- ✅ Kernel/engine separation (`src/core/` vs `src/engine/`)
- ✅ Runtime CPU detection and graceful fallback
- ✅ Alignment validation for streaming stores, hardware concurrency clamping
- ✅ **Dense243** (all 243 states validated) and **TriadSextet** (all 27 states validated) high-density encodings
- ✅ Canonical-indexing SIMD optimization (33% gain) and operation fusion (1.0×-2.9× native C++, geomean ~1.6×, re-validated 2026-08-18)
- ✅ Pluggable backend system (`ternary_backend`: Scalar/AVX2_v1/AVX2_v2, runtime-selectable)
- ✅ Multi-dimensional array support (any shape, not just 1D — added 2026-08-18)
- ✅ C FFI layer; OpenMP enabled by default on GCC/Clang (ARM/Apple-Clang excluded)
- ✅ Test suite: 16 wired suites via `tests/run_tests.py`, Linux x64 CI on every push
- ✅ Code-duplication cleanup between binding files (3 clusters, ~330-400 lines removed, 2026-08-18) and TritNet training scripts (shared `qat_common.py`)
- ✅ Performance benchmarking with statistical rigor (`BenchmarkRunner`, CV/95% CI) adopted across the fusion, fair-baseline, and core-ops suites

**TritNet (Neural Network-Based Arithmetic) — Phases 1-5, all complete (2026-08-18):**
- ✅ Truth tables → ≥99% ternary-weight accuracy (Phase 2) → C++ scalar+AVX2 inference engine, benchmarked against the LUT (Phase 3: **LUT wins by 169×-195×**) → GPU/CUDA batch inference (Phase 4: confirms rather than reverses Phase 3) → learned generalization and novel-operation discovery (Phase 5: errors are structured not noise; a genuinely novel, closed-form-resistant operation is learnable to 99.52% but still loses to a trivial LUT)
- **Verdict, unchanged across every phase and every hardware target tried:** LUT wins by 1-2 orders of magnitude. See the TritNet section above and `.claude/CLAUDE.md`'s "TritNet Development" section for the full writeup and report links.

**Competitive Benchmarking:**
- ✅ 6-phase suite implemented; Phases 1-4 have real measurement code (Phase 5-6 remain framework/descriptive-only)
- ✅ Re-validated 2026-08-18 under a verified-clean environment: **2/5 commercial-viability criteria validated** (memory efficiency, throughput-at-bit-width; latency/power/accuracy-retention still open)

**Infrastructure:**
- ✅ OpenTimestamps IP protection (SHA512 + Bitcoin blockchain timestamping)
- ✅ Documentation debt pass across `docs/`, `tests/`, `reports/`, `benchmarks/`, `models/`, `research/`, `build/`, `scripts/`, `CONTRIBUTING.md` — recurring finding was stale paths/claims left behind by earlier reorganizations, see `.claude/CLAUDE.md`'s "Critical Gaps" for the full list of sessions

**Open / Deferred:**
- **ARM/NEON support** - deferred; no ARM cross-compiler, hardware, or emulator available in the environment this was last attempted from, and writing untested NEON intrinsics would violate this project's verify-by-execution discipline
- **AVX-512** - not started; not evaluated on any AVX-512-capable hardware yet
- **RISC-V, FPGA/ASIC** - not started, long-term vision only
- **Profiler integration (VTune ITT, NVTX, Perfetto)** - the call-site framework exists in `src/core/profiling/ternary_profiler.h` and is genuinely wired into `bindings_core_ops.cpp`'s hot paths, but no build script defines `TERNARY_ENABLE_VTUNE`/`_NVTX`/`_PERFETTO`, so only the no-op stub has ever been built
- **Windows x64 re-validation** - most 2026-08 fixes (multi-dim arrays, code dedup, TritNet Phases 3-5, competitive-benchmark re-validation) have been done and verified on Linux x64 only; Windows remains the only platform with a *production* performance claim, dated 2025-11-28, predating this work
- **Real model quantization (Phase 5)** wired to real measurement code 2026-08-22 (TinyLlama-1.1B, naive post-training ternary quantization) — memory criterion passes (4.82× smaller than fp16), accuracy criterion fails decisively (perplexity 12.8 → 89,101); extending to Phi-2/Gemma-2B or a more sophisticated quantization scheme (QAT, per-channel scaling, calibration) would need further work, not attempted. **Power measurement (Phase 6)** wired to real code 2026-08-22 (auto-detects Intel RAPL / NVIDIA / Windows monitors, honest Mock fallback) but this project's dev sandbox has no reachable hardware monitor to actually validate against (no RAPL read permission, no `perf` access, no sudo) — needs running somewhere with real access before a citable number exists

An earlier exploratory plan to integrate TritNet with BitNet's 1.58-bit matmul kernels (a 4-phase research path, A through D) was never executed — TritNet's actual roadmap took a different, self-contained path (native C++ engine → AVX2 → GPU → novel-operation discovery, documented above) and reached a definitive verdict without it. Kept here as historical context, not a live plan.

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development workflow
- Coding standards
- Testing requirements
- Performance guidelines

## License

Apache License 2.0 - See [LICENSE](LICENSE)

Copyright 2025 Jonathan Verdun (Ternary Engine Project)

Developed by Jonathan Verdun with grateful acknowledgment to Ivan Weiss Van der Pol and Kyrian Weiss Van der Pol for their support.

## Citation

```bibtex
@software{ternary_engine,
  title={Ternary Engine: High-Performance Balanced Ternary Arithmetic},
  author={Jonathan Verdun},
  year={2025},
  version={1.0.0},
  url={https://github.com/gesttaltt/ternary-engine}
}
```

## References

- [Balanced Ternary (Wikipedia)](https://en.wikipedia.org/wiki/Balanced_ternary)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- [pybind11 Documentation](https://pybind11.readthedocs.io/)

---

**Version**: See `.claude/CLAUDE.md` (currently v1.49.0) for the authoritative, dated version history — this README summarizes rather than tracks it line-by-line
**Status**: Production (Windows x64, dated 2025-11-28), Tests+CI validated (Linux x64), TritNet roadmap (Phases 1-5) complete, Experimental (macOS, ARM/NEON, AVX-512)
**Updated**: 2026-08-22
**Platform**: Windows x64 (production perf claims, validated 2025-11-28), Linux x64 (tests + CI + most 2026-08 fixes/re-validations, formal benchmark re-validation on Windows still pending), macOS (untested)

**Recent Additions (2026-08):**
- ✅ **TritNet Phases 1-5 complete** - truth tables → ≥99% accuracy → C++ engine (LUT wins 169×-195×) → GPU (confirms, doesn't reverse) → novel-operation discovery (99.52% on a genuinely new op, still loses to a LUT)
- ✅ **Multi-dimensional array support** - any shape, not just 1D
- ✅ **Competitive benchmark re-validation** - 2/5 commercial-viability criteria, confirmed under a verified-clean environment
- ✅ **Code-duplication cleanup** - ~330-400 lines removed across binding files; shared TritNet training module
- ✅ **Extensive documentation-debt pass** - stale paths and claims fixed across docs/, tests/, reports/, benchmarks/, models/, build/, and this file
- ✅ **Dense-packed GEMM kernel** - fixed the "0.189× avg / too slow for AI" matmul verdict; new kernel is 19×-32× faster than the old CSC/CSR one at the tested batch size, flips the competitive suite's Phase 4 verdict to "viable for AI"
- ✅ **Phase 6 power measurement wired up** - was purely descriptive; now runs a real hardware-monitor-based comparison (Intel RAPL / NVIDIA / Windows, auto-detected, honest simulated fallback) — also fixed a real silent-failure bug in the RAPL permission check found along the way
- ✅ **Phase 5 real model quantization** - was purely descriptive; now runs a real TinyLlama-1.1B ternary quantization + WikiText-2 perplexity measurement. Memory passes (4.82× smaller than fp16); accuracy fails decisively (naive post-training quantization, as this project's own "simple threshold-based" scheme literally specifies, catastrophically breaks the model — a genuine, informative negative result)

**Performance Summary (Windows x64, validated 2025-11-28 — the most recent formal production benchmark run):**
- ✅ **45.3 Gops/s peak** throughput with fusion operations
- ✅ **39.1 Gops/s peak** throughput for element-wise operations
- ✅ **33% canonical indexing gain** via dual-shuffle + ADD optimization
- ✅ **1.43× fused / 1.7–3.5× tadd** vs same-semantics NumPy (fair baseline, Linux x64, 2026-08-11)
- ✅ **4× memory advantage** over INT8, 8× over FP16 (validated on 7B-405B models)
- ✅ **Matmul**: ~1.1×-1.2× avg vs NumPy (dense-packed kernel, 2026-08-20 — was 0.189× against the CSC/CSR kernel, re-validated 2026-08-18) - see Critical Gap #3 for the fp16-vs-fp32 caveat

> **Note:** Performance metrics are *subject to analysis* - no standardized benchmarking exists for trit operations. Element-wise/fusion figures above are the most recent formal Windows x64 production run; matmul and competitive-suite figures are Linux x64, re-validated more recently and more rigorously (see the Competitive Analysis and Commercial Viability tables above).

---

## The P-Adic Ecosystem

This repository is part of a tri-fold ecosystem exploring the intersection of p-adic mathematics, ternary logic, and high-performance computing:

*   **[3-Adic ML](https://github.com/gesttaltt/3-adic-ml)**: Mathematical foundation and framework for p-adic Variational Autoencoders and geometric deep learning.
*   **[3-Adic Bioinformatics](https://github.com/gesttaltt/3-adic-bioinformatics)**: Application of ultrametric geometry to genomic sequences, protein folding, and biological hierarchy analysis.
*   **[Ternary Engine](https://github.com/gesttaltt/ternary-engine)**: (This Repo) High-performance C++/C backend for native ternary arithmetic and efficient p-adic valuation processing.

## Status & Engagement

**Current Phase**: Active Low-Profile Research

This engine provides the computational backbone for our p-adic research. It is designed for researchers who require deterministic, high-efficiency ternary logic.

*   **Proposals**: We are focused on technical excellence and scientific utility. We are not entertaining commercial acquisition or mass-market investment at this stage.
*   **Contributions**: Technical contributions that improve the efficiency of the C++ core or broaden the C-header compatibility are highly valued.
