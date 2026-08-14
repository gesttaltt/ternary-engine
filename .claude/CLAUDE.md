# Claude Code Configuration - Ternary Neural Network Engine

**Doc-Type:** Project-Level Configuration · Version 1.17 · Updated 2026-08-14 · Author Ternary Engine Team

Project-specific Claude Code configuration for the Ternary Neural Network Engine - a production-grade balanced ternary arithmetic library with SIMD acceleration, TritNet neural network-based operations, and competitive benchmarking suite.

---

## Purpose & Scope

This configuration defines **project-level standards** for the Ternary Engine codebase. It establishes coding conventions, architectural principles, and workflow guidelines specific to this high-performance computing project.

**scope** - Project-level configuration for ternary-engine repository
**inheritance** - Extends user-level ~/.claude/CLAUDE.md with project-specific standards
**audience** - Contributors to the Ternary Engine project

---

## About This Project

**what_this_is**:
- Production-grade balanced ternary arithmetic library (Windows x64 validated)
- SIMD-accelerated operations with AVX2 vectorization (32 parallel trits)
- TritNet: Revolutionary neural network-based arithmetic (replacing LUTs with matmul)
- Competitive benchmarking suite (6 phases) proving commercial viability
- IP-protected with OpenTimestamps blockchain verification

**core_innovation**:
- 35,042 Mops/s peak throughput (35 billion operations/second) — the historical "8,234× vs pure Python" headline is retired (see README.md's historical note): comparing compiled code against interpreted Python overhead is a strawman that any compiled language wins by 10³–10⁴×, not a measure of this engine's actual advantage. See `benchmarks/SKEPTICAL_METRICS.md` and `benchmarks/python-with-interpreter-overhead/bench_fair_baseline.py`'s fair NumPy-baseline numbers instead: engine ~parity with NumPy on single ops, real wins are saturation-for-free (tadd 1.7–3.5×), fusion (1.43× geomean, up to 6×), and 4× memory density.
- 2-bit trit encoding enabling 8× memory reduction vs FP16 for AI models
- Neural network-based arithmetic learning (TritNet) for GPU/TPU hardware acceleration
- Rigorous validation with 65/65 tests passing on Windows x64

**target_applications**:
- Edge AI deployment (memory-constrained devices)
- Ultra-low power AI inference
- Model quantization beyond INT4/INT8
- Custom ternary hardware accelerators
- Computer vision edge detection

**documentation**:
- Quick start: [README.md](../README.md)
- Testing guide: [TESTING.md](../TESTING.md)
- Contributing: [CONTRIBUTING.md](../CONTRIBUTING.md)
- API reference: [docs/](../docs/)
- TritNet roadmap: [docs/research/tritnet/TRITNET_ROADMAP.md](../docs/research/tritnet/TRITNET_ROADMAP.md) (2025-11-23; phase numbering superseded — CLAUDE.md "TritNet Development" section is the source of truth for phase status)

---

## CRITICAL: Ternary vs Binary Assumptions

**THIS SECTION IS MANDATORY READING FOR ALL ALGORITHM DESIGN WORK**

### Fundamental Principle: 1 Trit ≠ 2 Bits

Ternary arithmetic operates on trits {-1, 0, +1} which are fundamentally different from binary bits:

- **Algebraic structure** - Ternary has 3-adic valuation, not 2-adic
- **Cost model** - "Number of multiplications" is a BINARY metric, not ternary
- **Optimality criteria** - Binary-optimal algorithms are NOT ternary-optimal

### Strassen is NOT Optimal for Ternary

**Strassen's algorithm** (7 multiplications for 2×2 matrix multiply) is:
- Binary-optimal: minimizes multiplication count for real/binary matrices
- NOT ternary-optimal: multiplication count is the WRONG metric in ternary
- One real-valued embedding among many, not a gold standard

**Why this matters:**
- In ternary {-1, 0, +1}, the cost structure is fundamentally different
- 3-adic valuation depth matters more than operation count
- Sparsity patterns have different significance
- Ultrametric structure defines natural basins, not Euclidean distance

### WRONG Metrics (DO NOT USE for Ternary)

| Metric | Why Wrong |
|--------|-----------|
| "Number of multiplications" | Binary cost model, irrelevant in ternary |
| "Strassen orbit penalty" | Strassen is not special in ternary space |
| "Novel = not Strassen" | Wrong definition of novelty |
| "7 factors optimal" | Binary optimality, not ternary |
| "Factor count" | Ignores ternary sparsity structure |

### CORRECT Ternary-Native Metrics (USE THESE)

| Metric | Description |
|--------|-------------|
| **Valuation depth** | 3-adic depth of coefficients |
| **Sparsity entropy** | Information-theoretic sparsity measure |
| **Ultrametric transition cost** | Cost of p-adic tree traversal |
| **Ternary operation count** | tadd, tmul in native units |
| **Dense243 packing efficiency** | Actual bits used (5 trits/byte) |

### Archived Example: GEMM Discovery (2024-12-29)

A complete GEMM algorithm discovery framework was built with the wrong assumption that Strassen was the gold standard to either rediscover or escape from. This approach was fundamentally flawed because:

1. All "discoveries" were Strassen gauge-equivalents (expected, given the binary-centric framing)
2. The search was optimizing for binary metrics in a ternary space
3. "Escaping Strassen orbit" is meaningless when Strassen isn't special

**See**: `models/gemm_discovery/ARCHIVE_2024-12-29_binary_assumption_error.md`

**Reusable components**:
- `gauge_canonical.py` - Gauge reduction is still valid math
- `ultrametric_actions.py` - Hierarchy-altering actions still valid
- `validate_independent.py` - Bilinear validity checking still correct

**Flawed components** (archived, do not use):
- `ultrametric_energy.py` - Remove Strassen penalty
- `surgical_analysis.py` - Compares to Strassen
- `run_*_discovery.py` - All Strassen-centric

---

## Trained Models for Falsification Testing

### Available Checkpoints

| Model | Path | Purpose | Key Properties |
|-------|------|---------|----------------|
| **v5_11_3** | `models/company-flagships/ternary-multiVAE/ternary_v5_11_3.pt` | Add/sub arithmetic centering | Hyperbolic embedding, operation-aware |
| **homeostasis** | `models/company-flagships/v5_11_homeostasis/best.pt` | Radial hierarchy, p-adic valuation | VRC target: -0.83, coverage: 100% — **unverified as of 2026-08-13, see caveat below** |
| **codon_encoder** | `models/company-flagships/hierarchy-encoder-codon-inference/codon-predictor/codon_encoder_3adic.pt` | Hierarchy neural network | 3-adic valuation structure |

### Model Capabilities

**v5_11_3 (Arithmetic Centering)**
- Trained for tadd/tsub operation embeddings
- Strong hyperbolic structure in Poincare ball
- Use for: H3 (hyperbolic), H10 (group theory) tests

**homeostasis (Radial Hierarchy)**
- Trained for valuation-radius correlation (VRC)
- Target: high valuation → small radius (near center)
- Use for: H1 (p-adic), H2 (ultrametric) tests

**codon_encoder (Hierarchy Network)**
- 3-adic valuation hierarchy encoder
- Can be repurposed for arithmetic hierarchy testing
- Use for: H2 (ultrametric tree), H24 (sui generis) tests

**CAVEAT (added 2026-08-13, see reports/2026-08-13/MODELS_RESEARCH_REVIEW.md):**
`models/company-flagships/validate_checkpoints.py` — the tool that computes
`hierarchy_A`/`hierarchy_B` = `Spearman(valuation, radius)`, the exact metric
behind the homeostasis checkpoint's documented "VRC target: -0.83" — had an
inverted-3-adic-valuation bug (computed on the raw corpus index instead of the
decoded balanced-ternary value, same bug class found in `research/scripts/
falsify.py` and 9 other places across `models/`, all fixed the same day). The
training script that originally produced `v5_11_homeostasis/best.pt` isn't in this
repo, so whether the original -0.83 figure was itself affected can't be determined
retroactively — but re-running `validate_checkpoints.py` against this checkpoint
before the fix would have reproduced an inverted value, not the correct one.
Treat -0.83 as unverified until re-run against the now-fixed script.

### Falsification Test Integration

```python
# Load models in falsify.py
models = {
    'v5_11_3': 'models/company-flagships/ternary-multiVAE/ternary_v5_11_3.pt',
    'homeostasis': 'models/company-flagships/v5_11_homeostasis/best.pt',
    'codon': 'models/company-flagships/hierarchy-encoder-codon-inference/codon-predictor/codon_encoder_3adic.pt',
}
```

### Falsification Results Summary (updated 2026-08-11)

**15 hypotheses tested, 0 falsified** — `research/FINDINGS.md` is the source of truth (full table, scores, and per-hypothesis analysis). H5 (Clifford) and H7 (Quantum) were REMOVED as non-informative furniture tests — do not reimplement them.

Condensed status:
- **INTRINSIC (5):** H1 p-adic, H2 ultrametric, H6 three-valued logic, H11 lattice, H13 topological (Cantor/3-adic)
- **Supported (9):** H3 hyperbolic, H4 tropical, H8 category, H9 information, H10 group theory, H17 F₃ field, H23 modular (rewritten for saturation), H24 sui generis, H14 neural (TritNet QAT)
- **Weak (1):** H12 dynamical (tmul period-2 collapses near zero trits)

**Key discoveries:**
- **tadd is non-associative for 79.6% of triplets** — balanced ternary with saturation is NOT a group; the structure is a non-associative ring (H24: tmul distributes over tadd 100% both sides)
- **tmul = F₃ multiplication exactly; tadd ≠ F₃ addition** at saturation (H17)
- **H14 Neural:** tnot 100% with ternary weights (Phase 2A GO), extended 2026-08-11 by Phase 2B GO: all four binary ops ≥99% (tadd 100%, tmul 99.5%, tmin 99.9%, tmax 99.9%), learned weights ~40% zeros matching 3-adic sparsity

Remaining untested: H15-H16, H18-H22 (Tier 4 research; evaluate value before implementing).

---

## Communication Style

**professional** - B2B focused, avoid emojis unless documenting existing code with them
**concise** - Technical precision over verbosity
**formatted** - Use markdown for clarity, code blocks for examples
**objective** - Prioritize accuracy and measurable performance over claims
**thinking_mode** - Enabled for complex optimization and architectural decisions

---

## Code Quality Standards

### Design Philosophy

**YAGNI_principle** - No speculative code, only proven optimizations
**phase_coherence** - Only add complexity if >10% performance gain measured
**algorithm_as_documentation** - LUTs generated from high-level logic, not hardcoded
**single_source_of_truth** - Mathematical rules defined once (compile-time generation)

### Performance Requirements

**benchmark_everything** - All optimizations must be validated with benchmarks
**regression_threshold** - 5% slowdown triggers investigation and revert
**statistical_rigor** - Report variance, confidence intervals, coefficient of variation
**validate_on_windows_x64** - Only platform proven for production claims

### Security & Safety

**no_undefined_behavior** - Strict C++17 compliance
**input_validation** - Sanitize external inputs, trust internal code paths
**graceful_degradation** - Runtime CPU detection with clear error messages
**alignment_safety** - Validate 32-byte alignment for streaming stores

---

## Code Organization

### Directory Structure

**src/core/** - Production-ready kernel (mathematically stable, validated)
- algebra/ - Scalar operations + LUTs
- simd/ - SIMD acceleration (AVX2)
- ffi/ - Cross-language FFI
- core_api.h - Unified entry point

**src/engine/** - Python bindings and library code
- bindings_core_ops.cpp - Python module for core SIMD operations (ternary_simd_engine)
- bindings_dense243.cpp - Python module for Dense243 encoding (ternary_dense243_module)
- bindings_tritnet_gemm.cpp - Python module for TritNet GEMM operations
- bindings_zero_skip_gemm.cpp - Python module for zero-skip ternary GEMM (ternary_zero_skip_gemm; used by competitive benchmark Phase 4, build/build_zero_skip_gemm.py, in CI)
- bindings_backend_api.cpp - Python module for the pluggable backend system (ternary_backend; Scalar/AVX2_v1/AVX2_v2 selectable at runtime, build/build_backend.py, added to CI 2026-08-12 — was previously undocumented and failed to compile, see "Critical Gaps" #1)
- lib/dense243/ - High-density encoding library (5 trits/byte, validated)

**scripts/** - Build and development automation
- build/ - Build scripts (all platforms)
- tritnet/ - TritNet training pipeline
- orchestration/ - High-level workflows

**benchmarks/** - Competitive analysis suite
**tests/** - Test suite, 14 suites via `run_tests.py` as of 2026-08-14 (expanded from 7/"65 tests" — see "Critical Gaps" #1)
**docs/** - API reference and architecture documentation

### Deployment Status Markers

**Production-Ready (src/core/)** - Windows x64 validated, mathematically stable
**Validated & Ready (experimental/)** - All tests passing, awaiting integration
**Pending Validation (experimental/)** - Implementation complete, benchmarks pending
**Broken/Deprecated (legacy/)** - Archived, do not use

---

## Coding Conventions

### Naming Conventions

**operations** - Lowercase, 4-char prefixed with 't': tadd, tmul, tmin, tmax, tnot
**build_scripts** - build_<target>.py (e.g., build_dense243.py)
**tritnet_scripts** - <verb>_tritnet.py or <noun>.py (e.g., train_tritnet.py)
**test_files** - test_<feature>.py (e.g., test_phase0.py)
**constants** - ALL_CAPS with underscores: MINUS_ONE, ZERO, PLUS_ONE

### C++ Style

**headers** - .h extension (not .hpp)
**force_inline** - Use FORCE_INLINE macro for critical paths
**constexpr_everything** - Compile-time computation for LUTs
**template_unification** - Single template replaces multiple code paths
**namespace** - No global namespace pollution

Example:
```cpp
// Good: Template-based unification
template<bool Sanitize>
static py::array_t<uint8_t> process_binary_array(/* ... */) {
    // Single implementation handles both sanitized and unsanitized paths
}

// Bad: Code duplication
static py::array_t<uint8_t> process_binary_array_sanitized(/* ... */) { /* ... */ }
static py::array_t<uint8_t> process_binary_array_fast(/* ... */) { /* ... */ }
```

### Python Style

**docstrings** - Comprehensive module, class, and function docstrings with examples
**type_hints** - Use typing module for function signatures
**test_at_bottom** - Include test code in `if __name__ == "__main__"` block
**imports** - Group: stdlib, third-party, local (separated by blank lines)

Example:
```python
"""
Module-level docstring explaining purpose.

Usage:
    from ternary_layers import TernaryLinear
    layer = TernaryLinear(10, 16)
"""

import sys
from pathlib import Path

import torch
import numpy as np

from local_module import helper


def quantize_ternary(weights: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Quantize weights to ternary values {-1, 0, +1}.

    Args:
        weights: Full-precision weight tensor
        threshold: Threshold for zero region

    Returns:
        Ternary weights with values in {-1, 0, +1}
    """
    # Implementation with inline comments explaining "why"
```

### File Headers

**copyright** - Apache 2.0 license header for all source files
**description** - Brief file purpose in header comment

```python
"""
filename.py - Brief description of file purpose

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Detailed description of what this file does.

USAGE: python filename.py [args]
OUTPUT: Description of outputs
"""
```

---

## Architecture Principles

### Kernel vs Engine Separation

**src/core/** - Production-ready, mathematically stable code
- Only validated, proven implementations
- Comprehensive test coverage required
- Performance validated with benchmarks
- Clear deployment status documentation

**src/engine/lib/** - Reusable library code for Python modules
- Implementation complete with validated performance
- Contains headers and utilities used by multiple bindings
- Example: lib/dense243/ for high-density encoding

### Template-Based Unification

**prefer** - Single template over multiple code paths
**reduces** - Code duplication, maintenance burden
**maintains** - Zero-cost abstraction via compile-time resolution

Example: `process_binary_array<Sanitize>()` replaced 6 separate implementations

### Compile-Time vs Runtime

**compile_time** - LUT generation, template instantiation, constexpr
**runtime** - CPU detection, alignment validation, graceful degradation

### Error Handling

**domain_specific_exceptions** - Custom error types in ternary_errors.h
**graceful_degradation** - Clear error messages, fallback paths
**python_integration** - C++ exceptions mapped to Python types

---

## Performance Optimization

### Measurement Requirements

**before_optimization** - Establish baseline with benchmark
**after_optimization** - Measure actual improvement
**statistical_rigor** - Multiple runs, variance, confidence intervals
**regression_detection** - Automated comparison against previous best

### Optimization Hierarchy

1. **Algorithm choice** - Choose correct algorithm first
2. **Compile-time optimization** - Constexpr, templates, LUT generation
3. **SIMD vectorization** - AVX2 for 32-wide parallelism
4. **Operation fusion** - Reduce memory traffic
5. **OpenMP parallelization** - Multi-threading for large arrays (≥100K elements)
6. **Profile-Guided Optimization** - Clang PGO for 5-15% additional gain

### Critical Performance Paths

**DO NOT modify without benchmarking:**
- src/core/algebra/ternary_algebra.h (scalar operations)
- src/core/simd/ternary_simd_kernels.h (SIMD operations)
- src/engine/bindings_core_ops.cpp (Python bindings for core operations)

**Benchmark methodology:**
- Use build/build.py for standard optimized build
- Run benchmarks/bench_phase0.py for comprehensive suite
- Compare against previous reports in reports/YYYY-MM-DD/
- Document results with validation date and platform

---

## Testing Requirements

### Test Coverage

**correctness_first** - Correctness tests before performance tests
**comprehensive** - 100% test coverage for production kernel
**both_layers** - C++ unit tests AND Python integration tests
**regression_prevention** - All bugs get regression tests

### Test Categories

**test_phase0.py** - Correctness (50 test cases)
**test_omp.py** - OpenMP scaling validation (25 test cases)
**test_errors.py** - Error handling
**test_fusion.py** - Fusion operation validation
**test_luts.cpp** - C++ unit tests (20 test cases)

### Running Tests

```bash
# Unified test runner (recommended)
python run_tests.py

# Individual test suites
python tests/test_phase0.py
python tests/test_omp.py

# Performance benchmarks (NOT unit tests)
python benchmarks/bench_phase0.py
```

### Test Writing Guidelines

**arrange_act_assert** - Clear test structure
**descriptive_names** - Test name describes what it validates
**single_responsibility** - One assertion per test when possible
**reproducible** - Fixed random seeds for deterministic tests

---

## TritNet Development

### Vision

**core_innovation** - Replace memory-bound LUTs with compute-bound neural networks
**hardware_acceleration** - Enable GPU/TPU via matmul instead of memory lookup
**learned_patterns** - Discover patterns beyond hand-coded arithmetic

### Architecture

**TritNetUnary** - For unary operations (tnot)
- Input: 5 trits {-1, 0, +1}
- Hidden: 8 neurons, ternary weights
- Output: 5 trits

**TritNetBinary** - For binary operations (tadd, tmul, tmin, tmax)
- Input: 10 trits (5 from A, 5 from B)
- Hidden: 16 neurons, ternary weights
- Output: 5 trits

### Development Phases

**Phase 1** - Truth table generation ✅ COMPLETE
- 243 samples for unary operations
- 59,049 samples for binary operations
- All operations: tnot, tadd, tmul, tmin, tmax

**Phase 2A** - Proof-of-concept (tnot) ✅ COMPLETE — GO (commit 0dfa6af)
- tnot trained to 100% accuracy with ternary weights (QAT)
- Go/No-Go decision point passed

**Phase 2B** - Scale to all operations ✅ COMPLETE — GO (2026-08-11)
- tadd: 100% exact | tmul: 99.5% | tmin: 99.9% | tmax: 99.9% — 4/4 PASSED
- GO criterion met: ≥3/4 ops ≥99% with ≥1 at 100%
- All ops with ternary weights, ~40% zeros (consistent with 3-adic sparsity)
- Checkpoints + result.json per op in models/tritnet/phase2b/

**Phase 3** - C++ Integration ✅ COMPLETE (naive + AVX2 both measured) — TritNet loses badly either way, see caveat
- Export ternary weights to binary format ✅ `models/tritnet/inference/generate_weights_header.py` → `tritnet_weights.h` (constexpr, compile-time, matches project's LUT-codegen convention; output layer zero-padded to a multiple of 8 so both scalar and AVX2 share one generated header with no scalar tail)
- Implement C++ inference engine ✅ naive/scalar `models/tritnet/inference/tritnet_inference.h` **and** AVX2 `tritnet_inference_avx2.h` (2026-08-14), all 5 ops both paths. AVX2 vectorizes across the output dimension (outer-product GEMV: broadcast one input scalar, FMA into 8 contiguous output lanes, int8 ternary weights widened via `_mm256_cvtepi8_epi32`→`_mm256_cvtepi32_ps`) rather than the reduction dimension, since weights are stored `[IN][HID]` row-major. Correctness: both paths verified bit-exact against every op's recorded checkpoint accuracy over the FULL input space (243/59,049 samples), and AVX2 verified **bit-identical** to scalar output over that same full space — all in `tests/cpp/test_tritnet_inference.cpp`.
- Benchmark TritNet vs LUT performance ✅ run 2026-08-14, Linux x64, AMD Ryzen 5 7520U, g++ 13.3.0 `-O3 -march=native -mavx2 -mfma` (`benchmarks/cpp-native-kernels/bench_tritnet_inference.cpp`): AVX2 gives a real **~10.2×–10.9× speedup over scalar** (tnot 2.75 vs 0.268 Mops/s; tadd 0.787 vs 0.072; tmul 0.783 vs 0.073; tmin 0.785 vs 0.073; tmax 0.782 vs 0.073 — landing at the low end of this repo's usual ~10–30× AVX2 gain, as predicted), but **LUT still wins by 169×–195×** even against the vectorized path (LUT throughput itself: tnot ~535 Mops/s, binary ops ~133 Mops/s — 5-trit-chunk ops/sec, consistent run-to-run). Matches the ~20K-more-MACs-per-op cost noted in research/PRIOR_ART_TERNARY_LANDSCAPE.md; AVX2 recovered roughly 1 order of magnitude, not the 3 needed to close the gap, exactly as anticipated. **Honest read confirmed, not just predicted**: TritNet's practical case rests on Phase 4/5 (GPU/TPU throughput at batch scale, or learned generalization beyond what a LUT can express), not on beating a LUT at this per-op width on a CPU — AVX2 does not change that conclusion, it was never going to. **Fairness check (2026-08-14, user-requested — same lesson this project already learned from the retired "8,234× vs Python" claim, see `core_innovation` above), CORRECTED same day after a second pass**: the AVX2 kernel re-runs the int8→float weight conversion on every single forward-pass call even though the weights never change — worth checking whether amortizing it (convert once, reuse) changes the picture. **First attempt was itself methodologically unfair and its numbers are wrong, superseded by the paragraph below**: it compared `baseline.avx2_mops` (measured early, in the main table) against a freshly-measured `preconv_mops` (measured much later, after an increasingly long binary had already run several minutes of sustained AVX2/FMA load) — two numbers from different points in a long-running program, exactly the kind of unfair timing this check was supposed to be hunting for, just one level deeper. That version reported tadd/tmul/tmin all amortizing to ~1.8×, narrowing LUT's win to ~93–113×. **Re-measured with both sides timed interleaved, rep-by-rep, back-to-back** (`time_best_interleaved()`, so both see the same clock/thermal drift pattern) — the result reverses for the binary ops: tadd/tmul/tmin/tmax (all 4 checked, tmax last) show **no benefit from amortizing (~0.89–0.95×, if anything marginally slower)**, reproducible across repeated runs; only **tnot shows a real, robust benefit (~1.58–1.74×)**. The mechanism, confirmed against this CPU's actual cache size (`lscpu`: 32KB L1d): tnot's weights are small enough that both the int8 (4KB) and converted-float (16KB) forms of its largest layer fit in L1, so amortizing is pure compute win. The binary ops' largest layer is 4× bigger (hidden=128 vs 64): int8 form (16KB) fits L1, but the converted float form (64KB) does **not** — it overflows 32KB L1 by 2×, forcing L2 traffic that costs more than the cheap `cvt` instructions it was meant to save. Amortizing isn't free; it trades compute for memory footprint, and that trade only pays off when the wider representation still fits cache. **Conclusion**: LUT's win over AVX2-TritNet stands at the original ~150–190× for the four binary ops (not narrowed by amortization), and ~66–114× for tnot (real, if variable run-to-run, benefit from amortizing). Either way the qualitative Phase 3 conclusion is unchanged: LUT still wins by two orders of magnitude. Full session writeup: [reports/2026-08-14/TRITNET_PHASE3_SESSION_REPORT.md](../reports/2026-08-14/TRITNET_PHASE3_SESSION_REPORT.md).

**Phase 4** - GPU Acceleration
- Batch inference optimization
- GPU/TPU deployment
- Measure actual throughput gains

**Phase 5** - Learned Generalization
- Explore approximate arithmetic
- Discover novel ternary operations
- Research applications

### Training Guidelines

**dataset** - Use complete truth tables from datasets/tritnet/
**optimizer** - Adam with default PyTorch settings
**target_accuracy** - 100% for exact arithmetic (99%+ acceptable)
**validation** - Hold-out test set from truth tables
**checkpointing** - Save models to models/tritnet/ with .tritnet extension

### File Organization

**models/tritnet/src/generate_truth_tables.py** - Dataset generation
**models/tritnet/src/ternary_layers.py** - PyTorch ternary layers
**models/tritnet/src/tritnet_model.py** - Model architectures
**models/tritnet/src/train_tritnet.py** - Training orchestration
**models/tritnet/run_tritnet.py** - Unified workflow

---

## Hyperbolic GEMM Research (3-vae-gemm-v1)

### Core Insight

**The ternary operation space is NOT Euclidean** - it's a p-adic, ultrametric, hyperbolic topology:

| Property | Description |
|----------|-------------|
| **p-adic (3-adic)** | Distance based on divisibility by 3 |
| **Non-Archimedean** | \|a + b\| ≤ max(\|a\|, \|b\|) |
| **Ultrametric** | All triangles isoceles (strong triangle inequality) |
| **Hyperbolic** | Negative curvature, tree-like hierarchy |

**Why Euclidean approaches fail:**
- MLP treats latent space as Euclidean
- `midpoint = (emb_a + emb_b) / 2` is NOT equidistant in hyperbolic space
- Classification into 19,683 bins ignores geometric structure
- Operations create TRAJECTORIES to attractor basins, not classifications

**Proof - Geodesic vs Euclidean Midpoint:**
```
x = [0.3, 0.2, 0.1], y = [0.1, 0.4, 0.2]

Euclidean midpoint: d(x,mid)=0.3496, d(y,mid)=0.3646  <- NOT EQUAL
Geodesic midpoint:  d(x,mid)=0.3564, d(y,mid)=0.3564  <- EQUAL
```

### Current Model Status

**Model:** HyperbolicOperationModel (330,971 parameters)
**Location:** `models/3-vae-gemm-v1/`
**Checkpoint:** `models/3-vae-gemm-v1/checkpoints_hyperbolic/best_model.pt`

| Metric | Value | Notes |
|--------|-------|-------|
| Last Epoch | 1 | Paused for resource management |
| Train Loss | 1.02 | Still learning |
| Val Accuracy | 0.11% | Early training |
| VRC | 0.03 | Correct direction (should be negative) |
| Trajectory Length | 1.15 | Flow dynamics active |

### How to Continue Training

```bash
# Resume from checkpoint
cd models/3-vae-gemm-v1
python train_hyperbolic.py --resume

# Or start fresh with more epochs
python train_hyperbolic.py --epochs 50

# Quick test (10 epochs)
python train_hyperbolic.py --epochs 10 --batch-size 1024
```

**Training time:** ~89 minutes per epoch (due to 19,683 attractor distance computation)

**Key files:**
- `models/3-vae-gemm-v1/hyperbolic_ops.py` - Poincare ball operations, Mobius addition, geodesic midpoint
- `models/3-vae-gemm-v1/train_hyperbolic.py` - Training pipeline with checkpoint support
- `models/3-vae-gemm-v1/checkpoints_hyperbolic/` - Saved model checkpoints

### Falsification Experiment Results (2025-12-30)

Quick experiments to test ternary GEMM optimization hypotheses:

| Hypothesis | Result | Implication |
|------------|--------|-------------|
| **H1: Ternary Strassen** | WRONG METRIC | Rank is binary thinking; need ultrametric equivalence |
| **H2: Ultrametric GEMM** | 22% match | Works for sparse (66.7%), fails for dense |
| **H3: Valuation Sparsity** | SUPPORTED | 40% of products are zero |
| **H4: Geodesic Interpolation** | PARTIAL | MIN works (100%), others need training |

**Run experiments:**
```bash
python models/gemm_discovery/experiments/ternary_gemm_falsification.py --experiment all
python models/gemm_discovery/experiments/ternary_gemm_falsification.py --experiment B  # Ultrametric only
```

### Key Takeaways

1. **Zero-skip optimization viable** - 40% of matrix product entries are zero
2. **Sparse matrices benefit from ultrametric** - 66.7% accuracy for sparse inputs
3. **Rank is a BINARY metric** - No rank-6 found, but "rank" itself is wrong metric for ternary
4. **Hyperbolic training should continue** - VRC learning correct direction

### Critical Insight: Strassen Equivalence Classes

**DO NOT say "Strassen is optimal"** - this applies binary thinking to ternary space.

What we actually found: The rank-7 decomposition exists as **one embedding** of a deeper ultrametric equivalence class. Different "Strassen variants" are the **same ontological structure** viewed from different points in the p-adic tree - multiple semantic minima that converge to the same attractor basin when viewed hierarchically.

The correct framing:
- Binary: "7 multiplications is optimal" (Euclidean, count-based)
- Ternary: "Hierarchical depth and ultrametric transitions define efficiency" (p-adic, topology-based)

Future work should explore:
- Ultrametric equivalence classes of decompositions
- Hierarchical depth as the true cost metric
- p-adic attractor basins that unify "different" algorithms

### Documentation

- **Research notes:** [docs/HYPERBOLIC_GEMM_RESEARCH.md](../docs/HYPERBOLIC_GEMM_RESEARCH.md)
- **Falsification code:** `models/gemm_discovery/experiments/ternary_gemm_falsification.py`
- **Ultrametric energy:** `models/gemm_discovery/ebm/ultrametric_energy.py`

---

## Build System

### Build Scripts

**Standard build** - `python build/build.py`
**Dense243 module** - `python build/build_dense243.py`
**Zero-skip GEMM module** - `python build/build_zero_skip_gemm.py`
**Backend module** - `python build/build_backend.py` (pluggable Scalar/AVX2_v1/AVX2_v2 backend system)
**PGO (Clang)** - `python build/build_pgo_unified.py --clang`
**PGO (MSVC)** - `python build/build_pgo.py full`
**Cleanup** - `python build/clean_all.py`

### Build Artifacts

**Output** - Root directory for easy import
**Versioned** - build/artifacts/<target>/<timestamp>/
**Latest symlink** - build/artifacts/<target>/latest (may fail on Windows without admin)

### Platform Requirements

**Windows (MSVC)** - Production-ready, fully validated
**Linux/macOS** - Experimental only, builds untested, CI disabled

### Compiler Flags

**MSVC** - /O2 /GL /arch:AVX2 /std:c++17 /LTCG
**GCC/Clang** - -O3 -march=native -mavx2 -flto -std=c++17

**OpenMP** - Enabled by default (-fopenmp / /openmp in build/build.py; disabled only on ARM and Apple Clang). Validated passing on Linux x64 (2026-07-23) via tests/python/test_omp.py.
**AVX2 required** - Runtime CPU detection with graceful failure

---

## Benchmarking

### Competitive Benchmarking Suite

**Purpose** - Prove commercial viability against industry standards

**6 Phases:**
1. Arithmetic vs NumPy INT8 - Direct performance comparison
2. Memory efficiency - Storage requirements for 7B/13B/70B models
3. Throughput at equivalent bit-width - vs INT2/INT4
4. Neural workload patterns - Matrix operations for AI
5. Real model quantization - TinyLlama, Phi-2, Gemma
6. Power consumption - Energy efficiency on x86/ARM/GPU

**Run benchmarks:**
```bash
# Full suite
python benchmarks/bench_competitive.py --all

# Specific phase
python benchmarks/bench_competitive.py --phase 1

# Generate report
python benchmarks/utils/visualization.py results/competitive_results_*.json
```

### Commercial Viability Criteria

**Target metrics:**
- Memory efficiency: 4× vs INT8 ✅ Proven
- Throughput at equivalent bit-width: > INT2 ⚠️ Needs INT2 reference
- Inference latency: < 2× FP16 ⚠️ Needs testing
- Power consumption: 2-4× better ⚠️ Needs hardware
- Accuracy retention: < 5% loss ⚠️ Needs models

**Current status:** 2/5 criteria validated

### Standard Benchmarks

**bench_phase0.py** - Core performance suite
**bench_competitive.py** - 6-phase competitive analysis
**bench_model_quantization.py** - Real model testing
**bench_power_consumption.py** - Energy efficiency

---

## IP Protection

### OpenTimestamps System

**Purpose** - Establish provable date of invention for patent applications
**Method** - SHA512 hash + Bitcoin blockchain timestamping
**Coverage** - 88 files tracked in initial snapshot (2025-11-23)

**Create timestamp:**
```bash
python opentimestamps/timestamp_create.py
# NOTE: runs immediately on invocation — it has no --help/--dry-run flags
```

**Verify timestamp:**
```bash
python opentimestamps/timestamp_verify.py opentimestamps/timestamps/manifest_YYYYMMDD_HHMMSS.json.ots
```

**Before major releases:**
- Novel innovations
- Production milestones
- Weekly snapshots during active development

---

## Git Workflow

### Commit Message Format

```
<type>: <subject>

<body>

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Types:**
- FEAT: New feature
- FIX: Bug fix
- PERF: Performance improvement
- REFACTOR: Code restructuring
- DOCS: Documentation
- TEST: Testing
- BUILD: Build system

### Commit Guidelines

**check_status_first** - Always run `git status` before commits
**follow_style** - Match repository's existing commit format
**no_force_push** - Never force push to main without explicit request
**preserve_hooks** - Never skip hooks (--no-verify) unless explicitly requested

### Recent Commit Pattern Examples

```
ce39331 TIMESTAMP: Initial IP protection snapshot - 88 files
af669be FEAT: Add OpenTimestamps IP protection system with SHA512
4bbd0c7 REFACTOR: Reorganize scripts directory with clear separation and orchestration
ed2bfdc FEAT: Add comprehensive build cleanup system and fix path inconsistencies
7628698 FEAT: TritNet Phase 1 - Complete truth table generation for all operations
```

---

## Development Workflow

### Starting New Work

1. **Read existing code first** - Understand before modifying
2. **Check for duplicates** - Search codebase for similar functionality
3. **Plan optimization** - Establish baseline benchmark
4. **Write tests** - Correctness tests before implementation
5. **Implement** - Follow coding conventions
6. **Benchmark** - Measure actual performance
7. **Validate** - Run full test suite
8. **Document** - Update relevant docs with validation date

### Adding New Operations

1. **Algorithm definition** - Mathematical specification
2. **Scalar implementation** - Branch-free LUT in ternary_algebra.h
3. **LUT generation** - Compile-time constexpr in ternary_lut_gen.h
4. **SIMD kernel** - AVX2 vectorization in ternary_simd_kernels.h
5. **Python binding** - pybind11 wrapper in bindings_core_ops.cpp
6. **Correctness tests** - Comprehensive test coverage
7. **Performance benchmark** - Validate speedup vs Python
8. **Documentation** - API docs with examples

### Modifying Hot Paths

**NEVER modify without:**
- Current baseline benchmark
- Post-modification benchmark
- Statistical comparison (variance, confidence intervals)
- Regression threshold check (5% max slowdown)
- Documentation of changes with validation date

---

## Documentation Standards

### Documentation Types

**README.md** - Project overview, quick start, performance claims
**TESTING.md** - Testing and CI/CD guide
**CONTRIBUTING.md** - Development guidelines
**CHANGELOG.md** - Version history with dates
**docs/** - API reference and architecture
**build/README.md** - Build system comprehensive docs

### Documentation Requirements

**performance_claims** - Must cite validation reports with dates and platforms
**code_examples** - Include working examples in docstrings
**api_docs** - Comprehensive parameter and return value descriptions
**cross_references** - Link to related documentation
**validation_dates** - All benchmarks include validation date and platform

### Documentation Format

**markdown** - GitHub-flavored markdown
**code_blocks** - Fenced code blocks with language specifiers
**headers** - Hierarchical structure (max 3 levels)
**badges** - Status badges for quick scanning (performance, platform, license)

Example:
```markdown
### Function Name

**Purpose** - Brief description

**Arguments:**
- param1 (type) - Description
- param2 (type) - Description

**Returns:**
- type - Description

**Example:**
```python
result = function_name(arg1, arg2)
```

**Performance** - X Mops/s (validated YYYY-MM-DD, platform)
```

---

## Slash Commands

**Available commands** (via .claude/commands/):
- /build - Build standard module
- /build-dense243 - Build Dense243 module
- /test - Run test suite
- /benchmark - Run performance benchmarks
- /tritnet-train - Train TritNet models
- /competitive - Run competitive benchmarks
- /clean - Clean build artifacts
- /timestamp - Create IP timestamp

See .claude/commands/ for implementation details.

---

## Dependencies

### Core Runtime

**Python** - 3.7+ (validated on 3.12)
**C++ Compiler** - MSVC (Windows), GCC/Clang (Linux/macOS) with C++17 support
**CPU** - x86-64 with AVX2 (Intel Haswell 2013+, AMD Excavator 2015+)
**Libraries:**
- pybind11 (C++/Python binding)
- NumPy (array operations)

### TritNet

**PyTorch** - 2.0+ (custom ternary quantization)
**NumPy** - 1.19+
**No CUDA required** - CPU-only training for Phase 2

### Benchmarking

**Core** - NumPy for baseline comparisons
**Visualization** - Matplotlib (optional)
**Model quantization** - PyTorch + Transformers (for Phase 5)
**Power monitoring** - Platform-specific (Intel RAPL, nvidia-smi)

### Installation

```bash
# Core dependencies
pip install pybind11 numpy

# TritNet dependencies
pip install torch

# Benchmarking dependencies (optional)
pip install matplotlib transformers
```

---

## Platform Support

### Production Ready

**Windows x64** - Fully validated (2025-11-23)
- 65/65 tests passing
- 35,042 Mops/s peak throughput validated
- MSVC build system proven
- OpenMP enabled by default (see Build System section)

### Experimental

**Linux/macOS** - Not production-validated per project standard, but locally functional
- Local Linux x64 run (2026-07-23): all 5 test suites pass (65 tests), including OpenMP, via tests/run_tests.py
- Build scripts provided but no formal benchmark/CI validation
- No production claims until formally validated per project standard

### Target Platforms (Future)

**ARM NEON/SVE** - Planned for mobile/edge deployment
**AVX-512** - Planned for latest Intel/AMD CPUs
**WebAssembly SIMD** - Planned for browser deployment

---

## Critical Gaps & Known Issues

### Production Gaps

1. **Multi-platform validation** - Only Windows x64 formally proven for benchmarks; Linux x64 CI added 2026-08-11 (.github/workflows/ci.yml builds engine + TritNet GEMM + Dense243 + zero-skip GEMM + backend, runs full suite); benchmark validation on Linux still pending. Suite expanded 2026-08-12 from 7 to 13 wired suites — `tests/run_tests.py` and CI previously ran only 7 of ~17 real test files under `tests/python/` (including `quality_gates.py`, whose own header says it "must pass before any release", and `test_backend_integration.py`, the largest Python test file). Triaged all un-wired files: 6 passed and are now wired (`test_canonical_lut.py`, `test_simd_validation.py`, `test_fusion_correctness.py`, `test_backend_integration.py`, `test_fused_op_bug.py`, `quality_gates.py`); 2 were genuinely dead and removed (`test_simd_python.py` — truncated mid-docstring, never valid Python; `test_path_fixes.py` — its own `PROJECT_ROOT` path math was off by 2 `.parent` levels, and it referenced a `build_fusion.py`/`ternary_fusion_engine` that no longer exist post-fusion-merge, so every check in it either silently skipped or hard-failed); `test_dual_shuffle_validation.py` correctly reports failure — it tests dual-shuffle XOR, an explicitly-labeled "future enhancement" not yet implemented (see `backend_avx2_v2_optimized.cpp` header), left un-wired as a known limitation rather than a bug; `compile_test.py`/`setup_test.py`/`run_simd_harness.py` are dev utilities, not suites, left as-is. Wiring `test_backend_integration.py` required first fixing why `build/build_backend.py` (undocumented, not previously in CI) failed to compile at all on Linux — see TritNet/backend note below.
2. **TritNet Phase 3 pending** - Phase 2B GO achieved 2026-08-11: 4/4 ops ≥99% with ternary weights (tadd 100%, tmul 99.5%, tmin 99.9%, tmax 99.9%). Next: Phase 3 C++ inference engine (weight export, C++ inference, TritNet-vs-LUT benchmark — the experiment that decides whether TritNet beats LUTs in practice; note LUT does ~20K fewer MACs per 5-trit op, see research/PRIOR_ART_TERNARY_LANDSCAPE.md context). Blocking bugs fixed 2026-08-12 (see reports/2026-08-12/CODE_REVIEW_SESSION_REPORT.md): the AVX2 TritNet GEMM kernel computed wrong results for any output width >1 (row-stride bug, masked because the validation function compared the naive kernel against itself and could never have caught it) and was never actually reachable from the Python `gemm()` API regardless; both fixed and verified. `tritnet_gemm_f32_avx2_tiled` has the same class of bug but is still unreachable/unfixed (no test coverage to verify a fix against). **Weight export unblocked 2026-08-14**: the "checkpoint format incompatibility" was actually two fully disjoint architectures, not a save-wrapper mismatch — `tritnet_model.py`'s `TritNetUnary`/`TritNetBinary` (backed by `ternary_layers.TernaryLinear`, no bias, direct-regression output, has `export_weights_to_numpy()`) turned out to be a stale/abandoned pipeline: its only surviving checkpoint (`tritnet_tadd.tritnet`) is 15.8% accurate, not the 100% the roadmap documents. The real GO checkpoints were trained by `train_phase2a.py`/`train_phase2b.py`'s own local `TritClassifier`/`TernaryLinearQAT` (bias included, CrossEntropy classification head, ReLU hidden layers) — structurally incompatible with `tritnet_model.py`'s classes, and `train_phase2a.py` additionally never called `torch.save` anywhere, so tnot's GO model (documented complete since Phase 2A) had no on-disk checkpoint at all despite `train_phase2b.py`'s 4 binary ops already having theirs. Fixed: added checkpoint save/resume to `train_phase2a.py` (now saves to `models/tritnet/phase2a/tnot/`, mirroring `train_phase2b.py`'s pattern) and re-ran it (100% reproduced); wrote `models/tritnet/export_weights.py` targeting the real `TritClassifier` architecture directly, exporting all 5 ops' quantized int8 weights + biases to `models/tritnet/phase2b_export/<op>/*.npy`; added `tests/python/test_tritnet_export.py` (wired into `run_tests.py`, suite now 14/14) — a pure-NumPy replay of the exported weights over the full input space (243 / 59,049 samples per op) that reproduces each op's recorded checkpoint accuracy bit-for-bit. **C++ inference engine done 2026-08-14** (`models/tritnet/inference/tritnet_inference.h` naive/scalar + `tritnet_inference_avx2.h` AVX2, all 5 ops both paths; correctness verified bit-exact against every op's recorded accuracy over the full input space, and AVX2 verified bit-identical to scalar, via `tests/cpp/test_tritnet_inference.cpp`) — and the decisive TritNet-vs-LUT benchmark now has real numbers for both: naive scalar loses by 950×–1776×; **AVX2 recovers ~10.2×–10.9× (as predicted, the low end of this repo's usual AVX2 gain) but LUT still wins by 169×–195×** (`benchmarks/cpp-native-kernels/bench_tritnet_inference.cpp`, 2026-08-14, Linux x64, AMD Ryzen 5 7520U). See "TritNet Development" → Phase 3 above for the full figures. Phase 3 is now complete; remaining work is Python bindings for the inference engine (not yet written; only standalone C++ headers + dev-utility test/benchmark exist, neither wired into `run_tests.py` or CI, matching gap #1's convention for `tests/cpp/`) and Phase 4 (GPU/TPU), which is where TritNet's actual case has to be made. Two rounds of user-requested fairness review followed the same day, one of which caught and corrected a real bug in the benchmark itself (an unfair early-vs-late timing comparison, same class of issue as the retired "8,234× vs Python" claim) — see "TritNet Development" → Phase 3 above for the corrected numbers and full session report: reports/2026-08-14/TRITNET_PHASE3_SESSION_REPORT.md.
3. **Competitive benchmarking** - Only 2/5 criteria validated. Full 6-phase Linux x64 run COMPLETE, including a same-session fix for the Phase 3/4 script gaps (2026-08-11, see reports/2026-08-11/LINUX_VALIDATION_REPORT.md §3): Phase 1 (arithmetic) "NEEDS WORK" (0.63-0.68x avg, matches fair-baseline finding that single ops are ~parity/behind NumPy); Phase 2 (memory) "SIGNIFICANT ADVANTAGE" (4.0x vs INT8, matches README claim); Phase 3 (throughput @ bit-width) FIXED — now benchmarks real INT2/INT4 NumPy-packed references alongside engine-native and Dense243 ternary, all sized to a genuine 1GB footprint (the old version silently allocated 4x that); verdict: Dense243 is 9.6x faster than the INT2 reference at equivalent (~2-bit) density; Phase 4 (neural workload) FIXED — now calls the compiled `ternary_zero_skip_gemm.ZeroSkipWeights` kernel (sparse index precomputed, correctness-checked against NumPy, max err ~1e-4) instead of a naive Python loop; result: still "TOO SLOW FOR AI" at 0.21x avg (batch=1, ~33% random sparsity vs ~40% in real trained weights) but now a trustworthy kernel-vs-kernel number; Phase 5-6 remain descriptive-only (no measurement code, confirmed by inspection). Net: criteria count unchanged at 2/5 (neither fixed phase crosses into an unambiguous "beats NumPy" claim), but both numbers are now real measurements instead of script artifacts. **CAVEAT added 2026-08-12**: `bench_competitive.py` (this script) was found to have a path-resolution bug that made it silently fall back to mock `(a+b)%3` arithmetic instead of the real engine whenever `PYTHONPATH` didn't happen to already contain the repo root — fixed and reproduced/re-verified with a clean environment, but there's no way to audit retroactively whether the 2026-08-11 run above was affected (unknown what environment produced it). The numbers above are worth a fresh re-run under a verified-clean environment before being cited further; see reports/2026-08-12/CODE_REVIEW_SESSION_REPORT.md for the full explanation.
4. **Dense243 integration** - RESOLVED on Linux (2026-08-11): build failed because build_dense243.py lacked AVX2 flags on GCC (-march=haswell -mavx2); fixed, module builds, 10/10 C++ tests pass, and new tests/python/test_dense243.py validates round-trip + all 5 ops against the reference engine (suite now 6/6). Windows /arch:AVX2 added but not yet re-validated on Windows. 2026-08-12: also fixed a global-static AVX2-intrinsics-at-import-time crash risk (no `has_avx2()` gate) and a broken SIMD trit-extraction path (misuse of `_mm256_shuffle_epi8`, confirmed dead/unreachable) in `ternary_dense243_simd.h` — see session report.

### Important Improvements

5. **Phase 4.1 fusion** - Implementation complete; test_fusion.py passes, but dedicated performance benchmarks still pending
6. **Code duplication** - Between engines, needs refactoring
7. **TritNet training pipeline duplication** - `train_phase2a.py`/`train_phase2b.py` duplicate their QAT training code (STE, TernaryLinearQAT, TritClassifier, rescale_weights_for_qat) near-verbatim instead of sharing a module. Demonstrated concretely 2026-08-12: a checkpoint-resume correctness bug had to be found and fixed separately in each file because there's no shared code a single fix would apply to.
8. **`BenchmarkRunner` unused** - `benchmarks/python-with-interpreter-overhead/benchmark_framework.py`'s `BenchmarkRunner` class is imported nowhere except itself and a deprecated file; `bench_fair_baseline.py`, `bench_simd_core_ops.py`, and `bench_simd_fusion_ops.py` each hand-roll their own warmup/measure/stats logic instead, with inconsistent statistical rigor between them (found 2026-08-12 while fixing a crash in `BenchmarkRunner` that none of the three active scripts would have hit, since none of them use it).

### 2026-08-12/13 code review session — closed out

Full reports: reports/2026-08-12/CODE_REVIEW_SESSION_REPORT.md (original session)
and reports/2026-08-13/CODE_REVIEW_SESSION_REPORT.md (continuation, condensed
below). The four areas left pending on
2026-08-12 (`benchmarks/utils/` 4 of 6 files, `benchmarks/macro/`, `research/`,
`opentimestamps/`) and the broader path-resolution sweep have all now been
reviewed; nothing remains queued from this review effort. Findings:

- **`benchmarks/utils/geometric_metrics.py`** — `autocorrelation(lag=0)` crashed
  with a broadcast `ValueError`: `x[:-lag]` with `lag=0` is Python's
  negative-zero-slicing gotcha (`x[:-0] == x[:0]`, empty array) instead of the
  full array. Fixed with an explicit `lag > 0` branch; verified across 4 cases.
- **`benchmarks/utils/benchmark_validator.py`** — two related bugs, both
  root-caused to `extract_performance()` returning `0.0` for both "genuinely
  zero" and "not found in this JSON's schema": (1) `compare_performance()`
  reported a false -100% regression instead of "no data" whenever a JSON's
  schema didn't match one of the 3 hardcoded formats (e.g. this repo's real
  `fair_baseline_*.json`, keyed under `'cells'`); (2) `generate_report()` had
  no way to know `load_data()` had failed, so a missing/corrupt input file
  rendered as "Total Benchmarks: 0, Failed: 0, Status: PASS, Action: proceed
  with merge" instead of a failure report. Fixed: `extract_performance()` now
  returns `Optional[float]` (`None` = not found), comparisons get an explicit
  `NO_DATA` status, and a `data_loaded` flag gates report generation. Both
  reproduced with synthetic inputs and verified fixed.
- **`benchmarks/utils/visualization.py`** — `_format_phase4`/
  `_generate_phase4_html` crashed with `TypeError` on the `{'error': '...'}`
  dict `bench_competitive.py` emits when `ternary_zero_skip_gemm` isn't built
  (a non-empty dict is truthy, so execution fell through to iterating its
  string keys). `_format_phase3` checked for a `'ternary_gops'` key that no
  longer exists in Phase 3's output since its 2026-08-11 fix (gap #3) — always
  reported "No data available" even with real data present. Phase 3 was also
  never rendered in the HTML report at all (no `_generate_phase3_html`
  existed). All three fixed; verified together against a synthetic JSON
  matching the real schemas.
- **`benchmarks/utils/windows_power.py`** — `benchmark_operation()` sampled
  power (blocking PowerShell calls, up to their own 5-10s timeouts) inside the
  same timed loop used to measure `ops_per_sec`, deflating throughput by
  however much sampling overhead landed mid-benchmark. Fixed by tracking
  operation-only elapsed time separately. Also confirmed and documented (not
  merged, YAGNI): this entire module is unimported anywhere in the repo —
  `bench_power_efficiency.py`, its only conceivable caller, reimplements an
  independent `WindowsPowerMonitor` from scratch instead (same duplication
  shape as gap #7).
- **`benchmarks/macro/`** (`bench_image_pipeline.py`, `bench_layer_forward.py`)
  — reviewed and run end-to-end; both correct, no bugs found.
- **`opentimestamps/{timestamp_create,timestamp_verify}.py`** — reviewed;
  `timestamp_create.py` has real side effects (can submit to the public
  Bitcoin blockchain via the `ots` CLI) so was read-reviewed only, not
  executed. `timestamp_verify.py` is read-only and was actually run against a
  real manifest, correctly flagging genuine file changes from earlier in this
  session. Both had `datetime.utcnow()` deprecation warnings (Python 3.12+);
  fixed in both, re-verified clean with `-W error::DeprecationWarning`.
- **`research/scripts/falsify.py`** — the highest-value findings of this
  continuation, all reproduced against the real pipeline before and after:
  - `ComponentLoader.build_corpus()` computed `valuations[i] = v3(i)` on the
    **raw** corpus encoding index instead of `v3(i - idx_offset)` on the
    **decoded** balanced-ternary value (`idx_offset = 9841`, the all-zero-trits
    index). This silently swapped which indices looked "near zero": the true
    ternary zero (raw index 9841) got valuation 0, while the most negative
    value (raw index 0) got valuation 999. Corrupted every pointwise lookup
    (H3's valuation-radius-correlation test, H24's associativity-vs-valuation
    bucketing); aggregate histograms were unaffected (invariant under a
    constant shift over a complete residue system).
  - `test_H9_information`'s "zero is special" check called `v3(9841)` (the raw
    index) instead of `v3(0)` (the decoded value, via the same
    `idx - idx_offset` pattern H23 already used elsewhere in the file) —
    always failed regardless of any real structure. Verified: H9 now scores
    100% with `zero_is_special=True` (was unconditionally `False` before).
  - `test_H1_padic` sliced `all_results[:10000]` from a concatenation of ~4
    operations' LUT results (~50K samples each) — drew entirely from
    whichever operation's dict-insertion order came first, not a
    representative mix. Fixed with a fixed-seed uniform random sample;
    verified H1 now shows the expected cross-operation valuation distribution.
  - `main()`'s pre-flight guard checked only `status['data']`/
    `status['corpus']`, not `status['luts']`/`status['hyperbolic']` — since
    corpus-building doesn't depend on either, the guard could pass while they
    failed silently, and `test_H1_padic`/`test_H9_information`/
    `test_H3_hyperbolic` (each with an unconditional `self.c['luts']` or
    `self.c['hyperbolic']` access) would then raise a bare `KeyError`,
    masked by the generic exception handler into an opaque `grade='E'`. Added
    descriptive guards to all three tests and an upfront `[WARN]` in `main()`.
  - `test_H24_sui_generis`'s right-distributivity block computed a value from
    the wrong operands, discarded it, then recomputed correctly from
    duplicate calls of values already available from the left-distributivity
    test just above — 3 wasted SIMD calls per run, including one fully dead
    computation with a leftover "Wait: need tmul(a,c)..." comment marking the
    in-place bug-fix. Cleaned up; verified byte-for-byte identical output.
  - Also fixed: docstring referenced a `--tier` flag never wired into
    `argparse` (only `--hypothesis`/`--all` exist); missing `OUTPUT:` line and
    return-type hint on `main()` per CLAUDE.md conventions.
  - Documented but not fixed (structural/methodology, not wrong-answer bugs,
    same policy as gaps #7/#8): `compute_3adic_valuation` is independently
    reimplemented in 10 places across the repo with drifting clamp values and
    no canonical source (the `ebm` module the comment cites no longer exists
    in the tree); all 14 `test_H*()` methods hand-roll identical
    scoring/result-construction boilerplate; `H4`/`H10`/`H11`/`H23` each
    reseed the global RNG to the same state, making their samples
    bit-for-bit identical (not independent) when run together via `--all`;
    `H8`'s categorical tests call the SIMD op ~1300 times on single-element
    arrays instead of batching, ~185x more calls than necessary.
- **Broader path-resolution sweep** — systematically checked every
  `ROOT`/`PROJECT_ROOT`-style `Path(__file__).parent` chain in the repo (~50
  assignments across 48 files using `sys.path` manipulation) against each
  file's actual directory depth, not just the two idioms the 2026-08-12 sweep
  covered. Found 2 more instances of the same off-by-one bug in
  `tests/python/compile_test.py` and `tests/python/run_simd_harness.py` (both
  2 directories deep but computed with only 2 `.parent` calls instead of 3,
  doubling `"tests"` into a nonexistent `tests/tests/...` path); fixed. Both
  scripts' target file, `test_simd_correctness.cpp`, had also independently
  moved to `tests/cpp/` — fixed that too, though it made no practical
  difference before the fix either way (the file didn't resolve under any
  path prior to this). Both remain Windows/MSVC-only dev utilities, not
  wired into `tests/run_tests.py`, per gap #1. No other instances found; a
  couple of other flagged candidates (`build_gops_bench.py`, `build_kernels.py`
  in `benchmarks/cpp-native-kernels/`) were checked and confirmed already
  correct.
- `tests/run_tests.py`: 13/13 still pass after every fix in this session.

### 2026-08-13 models/ and research/ review — the inverted-valuation bug, ten times over

Full report: reports/2026-08-13/MODELS_RESEARCH_REVIEW.md. User-requested
follow-up ("review research/ and models/ for more bugs") after the above session
closed out. Covered `models/3-vae-gemm-v1/` (~3,500 lines) and
`models/company-flagships/` (~4,800 lines) — the two remaining substantial code
directories (`models/tritnet/` already reviewed 2026-08-12; `models/bitnet/` is an
empty placeholder; `research/` has no further Python beyond `falsify.py`, already
covered above).

**Headline finding:** the inverted-3-adic-valuation bug from `falsify.py`'s
`build_corpus()` (computing `v3(idx)` on the raw corpus encoding index instead of
`v3(idx - idx_offset)` on the decoded balanced-ternary value) turned up
**independently reimplemented 9 more times**: 4 in `models/3-vae-gemm-v1/`
(`data.py`, `model.py`, `hyperbolic_ops.py` ×2 — one of which seeds the initial
position of all 19,683 learnable attractor parameters) and 5 in
`models/company-flagships/` (`validate_checkpoints.py`, `create_embedding_lut.py`,
`explore_gemm_space.py`, `embedding_exactitude_score.py` ×2 sites,
`explore_gemm_extended.py`). All fixed and verified end-to-end via real object
instantiation (including geometrically, for `UltrametricAttractorField`: the
true-zero attractor now correctly sits near the Poincaré ball center instead of
the boundary). **Most consequential instance:** `validate_checkpoints.py` computes
`hierarchy_A`/`hierarchy_B` — the exact VRC metric behind the homeostasis
checkpoint's documented "-0.83" claim (see caveat in "Trained Models" section
above). A final repo-wide grep for the bug pattern after all fixes confirmed no
further instances exist.

**Other real bugs found and fixed** (see full report for details): a fully
missing metric (`AlgebraicMetrics.tadd_associativity`/`tmul_associativity`
declared but never computed, silently defaulting to 0.0 for every checkpoint ever
validated); non-reproducible `hash()`-based RNG seeding (3 sites, Python's string
`hash()` is randomized per-process); a dead metric branch
(`dendrogram_correlation` gated behind a threshold its only caller never
satisfies); a silent-wrong-result footgun (`generate_soft_gemm_map` returning
identity for unrecognized operations instead of raising); an unguarded `KeyError`
reachable against a real checkpoint architecture in the same directory; dead
per-operation result computation discarded instead of saved; a "test" with zero
assertions that always reported success; 3 hardcoded checkpoint paths that don't
match CLAUDE.md's documented path; a misleading stale comment; an O(n⁴) manual
cophenetic-distance loop replaced with the vectorized scipy equivalent already
used correctly in a sibling file; a NaN-risk unguarded division in a training
loss; stale cached embeddings used as ranking negatives for the entire back half
of training after an unfreeze boundary with no invalidation; and `models/3-vae-
gemm-v1/test_hyperbolic.py` rewritten from a zero-assertion always-passes script
into one with real structural and deterministic assertions (which, when first run,
caught an off-by-one in the review's own first-draft assertion — corrected in the
same pass).

**Documented but not fixed** (design/architecture-level, needs a judgment call —
see full report): `train.py` never calls the model's VAE forward path, so
`EESLoss`'s reconstruction/KL branches and the `Decoder` are permanently untrained
despite being fully wired; the package's exported public API
(`__init__.py` → `VAEGemmV1`) is the discredited plain-Euclidean-midpoint model,
not the geodesic/hyperbolic one that exists in the same directory; CLAUDE.md
documents a `train_hyperbolic.py --resume` workflow that doesn't exist in the
script's argparse, and neither training script has any checkpoint-resume
capability; a metric/loss mismatch where `predicted_idx` is derived from 3 extra
geodesic-flow steps beyond `predicted_emb` (what the loss actually optimizes); an
O(n³) unbatched attractor-ultrametric loss likely responsible for the documented
~89-min/epoch training cost; three independently-drifting VRC/radial-target
formulas across `loss.py` and `hyperbolic_ops.py`.

Commits: `e7572e7`, `d2c4396` (company-flagships), `c4426c1` (3-vae-gemm-v1).
`tests/run_tests.py`: 13/13 still pass (none of these files are wired into that
suite; verified individually via direct execution/`py_compile`).

### Nice to Have

7. **Multi-dimensional arrays** - Currently 1D only
8. **ARM/NEON support** - x86-64 AVX2 only
9. **GPU/TPU acceleration** - For TritNet Phase 4+
10. **Profiler integration** - Corrected 2026-08-12: framework IS integrated — `TERNARY_PROFILE_TASK_BEGIN`/`END` call sites genuinely exist in `bindings_core_ops.cpp`'s hot paths, contradicting this gap's prior wording. What's actually missing: no build script in `build/` ever defines `TERNARY_ENABLE_VTUNE`/`_NVTX`/`_PERFETTO`, so every current build only exercises the no-op stub — the backends (VTune ITT API, NVTX, Perfetto) are unbuilt and unverified, not "not integrated." `src/core/profiling/ternary_profiler.h`'s own former claim of "tested with Intel VTune Profiler" was corrected to reflect this at the same time.

---

## Questions to Ask Before Major Changes

**Architecture:**
- Is this change consistent with the kernel/engine separation?
- Does this belong in src/core/ (proven) or src/engine/experimental/ (pending)?
- Have we validated this approach on Windows x64?

**Performance:**
- What is the baseline performance?
- What is the expected performance improvement?
- Does this meet the >10% threshold for adding complexity?
- How will we measure and validate the improvement?

**Testing:**
- What correctness tests are needed?
- What performance benchmarks are needed?
- Have we considered regression tests?

**Documentation:**
- What documentation needs updating?
- What validation date and platform should we cite?
- Are there cross-references to update?

**Deployment:**
- What is the deployment status (production/validated/pending)?
- What phase is this (for fusion operations)?
- When should we promote to production?
- Does this belong in src/core/ (kernel) or src/engine/lib/ (library)?

---

## Learning Resources

### Internal Documentation

**Quick start** - [README.md](../README.md)
**Testing** - [TESTING.md](../TESTING.md)
**Contributing** - [CONTRIBUTING.md](../CONTRIBUTING.md)
**API reference** - [docs/](../docs/)
**Architecture** - [docs/architecture/](../docs/architecture/)
**Build system** - [docs/build-system/](../docs/build-system/)
**PGO guide** - [docs/pgo/](../docs/pgo/)
**TritNet** - [models/tritnet/src/](../models/tritnet/src/)
**Competitive benchmarks** - [benchmarks/COMPETITIVE_BENCHMARKS.md](../benchmarks/COMPETITIVE_BENCHMARKS.md)

### External Resources

**Balanced ternary** - https://en.wikipedia.org/wiki/Balanced_ternary
**Intel intrinsics** - https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
**pybind11** - https://pybind11.readthedocs.io/
**PyTorch** - https://pytorch.org/docs/
**AVX2 programming** - https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avxnewtechs=AVX2

---

## Support & Contact

**Issues** - https://github.com/gesttaltt/ternary-engine/issues
**Discussions** - https://github.com/gesttaltt/ternary-engine/discussions
**Author** - Jonathan Verdun (jonathan.verdun707@gmail.com)
**License** - Apache 2.0

**Acknowledgments:**
- Ivan Weiss Van der Pol
- Kyrian Weiss Van der Pol

---

## Changelog

| Date       | Version | Description                                    |
|:-----------|:--------|:-----------------------------------------------|
| 2026-08-14 | v1.17.0 | Completed the amortization check (gap flagged in the session report): tmax, the one binary op not yet independently re-measured under the corrected interleaved-timing methodology, confirmed to match tadd/tmul/tmin -- no benefit from amortizing weight conversion (~0.89-0.92x), reproducible. All 5 ops now confirmed. |
| 2026-08-14 | v1.16.0 | Session handoff: wrote reports/2026-08-14/TRITNET_PHASE3_SESSION_REPORT.md consolidating the full day's TritNet Phase 3 work (7 commits: weight export unblock, naive C++ engine, AVX2 vectorization, two rounds of fairness review including the self-correction) into one document with a single corrected numbers table, cross-referenced from both the TritNet Development and Critical Gaps sections above. |
| 2026-08-14 | v1.15.0 | CORRECTION: user asked to check tmin too, which surfaced that the amortized-weight-conversion check itself (v1.12.0-v1.14.0) was methodologically unfair -- it compared an early-measured baseline against a much-later-measured preconv number, the same class of bug it was built to catch, one level deeper. Fixed with interleaved rep-by-rep timing (both sides see the same clock/thermal drift); result reverses for the binary ops: tadd/tmul/tmin show NO benefit from amortizing (~0.94-0.95x, reproducible), only tnot does (~1.58-1.74x, real and robust). Root-caused to L1 cache fit (confirmed via lscpu, 32KB L1d): tnot's largest layer's converted-float form (16KB) still fits L1, the binary ops' (64KB) doesn't (int8 form, 16KB, does) -- amortizing trades compute for memory footprint, and only pays off when the wider form still fits cache. LUT's win over AVX2-TritNet reverts to the original ~150-190x for the 4 binary ops (not narrowed as previously claimed); tnot's ~66-114x stands. Qualitative Phase 3 conclusion unchanged either way. |
| 2026-08-14 | v1.14.0 | Extended the amortized-weight-conversion check to tmul (99.5% checkpoint, the weakest of the 4 binary ops) per user request. Holds identically to tadd: ~1.82x speedup (0.78->1.43 Mops/s), narrowing its LUT win from ~187x to ~103x -- confirms the amortization ratio is a function of the shared architecture (hidden=128), not of checkpoint accuracy/weight quality. All 3 checked ops (tadd, tmul, tnot) now converge on the same finding: ~1.7-1.8x from amortizing, LUT still wins by two orders of magnitude either way. |
| 2026-08-14 | v1.13.0 | Extended the amortized-weight-conversion check to tnot (the one unary op, hidden=64 vs the binary ops' hidden=128), per user request to confirm the finding generalizes. It does: tnot amortizes to ~1.72x (2.73->4.72 Mops/s), narrowing its LUT win from ~195x to ~113x -- consistent magnitude with tadd's ~1.8x/~93x across both hidden-layer sizes in the model family. Both checked correctness-bit-identical to the shipped AVX2 path before trusting either timing. |
| 2026-08-14 | v1.12.0 | User-requested fairness review of the AVX2 benchmark ("remember Python overhead"): no Python was involved, but found the same class of bias -- layer_avx2() reconverts the same invariant int8 weights to float on every call instead of amortizing. Isolated it (tadd): pre-converting once gives ~1.8x further speedup (0.78->1.41 Mops/s), narrowing LUT's win from ~169x to ~93x -- real and reproducible, doesn't change the qualitative conclusion (LUT still wins by two orders of magnitude). Also fixed CLAUDE.md's own stale "8,234x vs Python" headline claim in `core_innovation`, which README.md had already retired as a strawman (compiled-vs-interpreted) but CLAUDE.md never caught up on. |
| 2026-08-14 | v1.11.0 | Added AVX2 vectorization to the TritNet Phase 3 inference engine (`models/tritnet/inference/tritnet_inference_avx2.h`, all 5 ops), vectorizing across the output dimension (outer-product GEMV, int8 weights widened via `_mm256_cvtepi8_epi32`) since weights are stored row-major `[IN][HID]`. Correctness: AVX2 verified bit-identical to scalar over the full input space, not just matching aggregate accuracy. Benchmark result: AVX2 gives ~10.2x-10.9x over scalar (as predicted, low end of this repo's usual AVX2 range) but LUT still wins by 169x-195x even against the vectorized path -- confirms, not just predicts, that closing this gap needs Phase 4 (GPU/batch) or Phase 5 (learned generalization), not more CPU SIMD. Phase 3 is now complete. |
| 2026-08-14 | v1.10.0 | Finished the naive/scalar TritNet Phase 3 C++ inference engine (`models/tritnet/inference/tritnet_inference.h`, all 5 ops, weights compiled in via a generated header) and ran the decisive TritNet-vs-LUT benchmark CLAUDE.md had been flagging as the outstanding question: **LUT wins by 950x-1776x** at the naive/scalar level (Linux x64, AMD Ryzen 5 7520U, `benchmarks/cpp-native-kernels/bench_tritnet_inference.cpp`). Correctness verified bit-exact against every op's recorded checkpoint accuracy over the full input space (`tests/cpp/test_tritnet_inference.cpp`). Documented the honest read: closing a 3-orders-of-magnitude gap via AVX2 alone (~10-30x elsewhere in this repo) is not realistic -- TritNet's case has to rest on Phase 4/5, not beating a LUT at this op width on CPU. |
| 2026-08-14 | v1.9.0  | Continued the TritNet roadmap (gap #2): the documented "checkpoint format incompatibility" blocking Phase 3 weight export turned out to be two disjoint model architectures, not a save-wrapper mismatch -- `tritnet_model.py`'s exporter targets a stale/abandoned pipeline (its only surviving tadd checkpoint is 15.8% accurate, not the 100% the roadmap documents), while the real GO checkpoints (train_phase2a.py/train_phase2b.py's local TritClassifier) had zero export tooling and, for tnot specifically, no saved checkpoint on disk at all (train_phase2a.py never called torch.save). Fixed: added checkpoint save/resume to train_phase2a.py and reran it (100% reproduced, models/tritnet/phase2a/tnot/); wrote models/tritnet/export_weights.py targeting the real architecture, exporting all 5 ops' quantized weights to models/tritnet/phase2b_export/; added tests/python/test_tritnet_export.py (wired into run_tests.py, suite now 14/14) verifying the export bit-for-bit against each op's recorded accuracy via pure-NumPy replay over the full input space. C++ inference engine is the remaining Phase 3 step. |
| 2026-08-13 | v1.8.0  | Reviewed models/3-vae-gemm-v1/ and models/company-flagships/ (user request): found the inverted-3-adic-valuation bug (raw index vs. decoded value) independently reimplemented 9 more times beyond falsify.py, including in validate_checkpoints.py's hierarchy_A/hierarchy_B -- the exact metric behind the documented homeostasis "VRC -0.83" claim, now flagged unverified pending re-run; fixed all 9, plus a missing associativity metric, non-reproducible hash()-based seeding (3 sites), a dead metric branch, a silent-wrong-result footgun, an unguarded KeyError, discarded computation, a fake test, 3 hardcoded checkpoint-path mismatches, a misleading comment, an O(n⁴) loop replaced with vectorized scipy, a NaN-risk division in a training loss, and stale cached embeddings used as training negatives after an unfreeze boundary -- every fix reproduced and verified via real object instantiation. Full report: reports/2026-08-13/MODELS_RESEARCH_REVIEW.md |
| 2026-08-13 | v1.7.0  | Closed out the 2026-08-12 session's pending scope (benchmarks/utils/, benchmarks/macro/, research/, opentimestamps/, broader path sweep): fixed a real correctness bug in falsify.py's build_corpus() that silently swapped which corpus index looked "near zero" (raw vs. decoded valuation), plus 3 related falsify.py bugs (H9's zero-check, H1's skewed sample, main()'s missing luts/hyperbolic pre-flight guard) and a dead-code cleanup, all reproduced and verified against the real pipeline; fixed a false-regression bug in benchmark_validator.py (ambiguous 0.0 for "not found" vs. genuine zero) and a false-PASS-on-load-failure bug; fixed two visualization.py crashes (phase4 error-dict, stale phase3 schema) and phase3's total absence from HTML reports; fixed 2 more instances of the path off-by-one bug (tests/python/compile_test.py, run_simd_harness.py) plus a stale subdirectory reference. Full reports: reports/2026-08-12/CODE_REVIEW_SESSION_REPORT.md (original) + reports/2026-08-13/CODE_REVIEW_SESSION_REPORT.md (continuation) |
| 2026-08-12 | v1.6.0  | Code review session (19 commits, a527da0..32eada2): removed foreign .claude/ config dump; fixed a real correctness bug in the TritNet AVX2 GEMM kernel (was also unreachable from Python); found the TritNet orchestrator never actually ran the pipeline that achieved Phase 2B GO; found a systemic path-resolution bug in 12 files including bench_competitive.py (silent mock-fallback risk, flagged in gap #3); test suite expanded 7→13 wired suites. Full report: reports/2026-08-12/CODE_REVIEW_SESSION_REPORT.md |
| 2026-07-23 | v1.5.0  | Corrected stale gaps: OpenMP is enabled by default (not disabled), TRITNET_VISION.md/TRITNET_ROADMAP.md exist, TritNet Phase 2A is GO (Phase 2B in progress, 2/4 ops passed), Linux x64 passes all tests locally |
| 2025-01-02 | v1.4.0  | Updated falsification results (9/24 hypotheses), added H4 Tropical and H9 Information Theory |
| 2025-12-31 | v1.3.0  | Added trained models section for falsification testing |
| 2025-12-30 | v1.2.0  | Added Hyperbolic GEMM research section with 3-vae-gemm-v1 status, training instructions, falsification results |
| 2025-12-29 | v1.1.0  | CRITICAL: Added ternary vs binary assumptions section, archived GEMM discovery flawed approach |
| 2025-11-23 | v1.0.0  | Initial .claude configuration for Ternary Engine project |

---

**Remember:**
- **1 TRIT ≠ 2 BITS** - Ternary has different algebraic structure than binary
- **tadd is NON-ASSOCIATIVE** - Only 20% associativity, balanced ternary is NOT a group
- **Strassen is NOT optimal for ternary** - Use ternary-native metrics (valuation depth, sparsity entropy)
- **Ternary space is HYPERBOLIC** - Use geodesic midpoints, not Euclidean
- **40% of products are ZERO** - Zero-skip optimization is viable
- YAGNI: Only proven optimizations
- Benchmark everything before claiming performance gains
- Validate on Windows x64 before production claims
- Document with validation dates and platforms
- TritNet is the future: memory-bound → compute-bound

---

**Version:** 1.17.0 · **Updated:** 2026-08-14 · **Project:** Ternary Engine · **Repository:** https://github.com/gesttaltt/ternary-engine
