# Claude Code Configuration - Ternary Neural Network Engine

**Doc-Type:** Project-Level Configuration · Version 1.31 · Updated 2026-08-17 · Author Ternary Engine Team

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

**Note (2026-08-16):** `models/gemm_discovery/` does not exist anywhere in
this repository (confirmed via `git log --all` — it was never committed,
not merely deleted), so every path below is currently unreachable. The
lessons this section documents are still the project's position; found
while reviewing `research/` for path-resolution bugs (`research/scripts/
falsify.py` still does a no-op `sys.path.insert()` for this directory,
harmlessly, since nothing actually imports from it).

**Reusable components** (paths below are dead; kept for historical record):
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
**tests/** - Test suite, 15 suites via `run_tests.py` as of 2026-08-14 (expanded from 7/"65 tests" — see "Critical Gaps" #1)
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
**interleaved_timing** - Use interleaved reps (`time_best_interleaved()`) to avoid thermal/clock drift bias between runs.
**ffi_isolation** - Absolute performance claims must be measured in native C++ (`benchmarks/cpp-native-kernels/`) to isolate pybind11 overhead.

### Optimization Hierarchy

1. **Algorithm choice** - Choose correct algorithm first
2. **Compile-time optimization** - Constexpr, templates, LUT generation
3. **SIMD vectorization** - AVX2 for 32-wide parallelism
4. **Operation fusion** - Reduce memory traffic
5. **OpenMP parallelization** - Multi-threading for large arrays (≥100K elements)
6. **Profile-Guided Optimization** - Clang PGO for 5-15% additional gain
7. **Cache boundary verification** - Compute memory footprint against L1d/L2/L3 cache sizes to prevent conversion-widening optimizations from triggering cache thrashing.

### Critical Performance Paths

**DO NOT modify without benchmarking:**
- src/core/algebra/ternary_algebra.h (scalar operations)
- src/core/simd/ternary_simd_kernels.h (SIMD operations)
- src/engine/bindings_core_ops.cpp (Python bindings for core operations)

**Benchmark methodology:**
- Use build/build.py for standard optimized build
- Run benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py for comprehensive suite
- Compare against previous reports in reports/YYYY-MM-DD/
- Document results with validation date and platform
- Require unified `BenchmarkRunner` in all Python benchmarks (no hand-rolled loops)


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
python benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py
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
- Python bindings ✅ (2026-08-14) `src/engine/bindings_tritnet_inference.cpp` → `ternary_tritnet_inference`, `tnot`/`tadd`/`tmul`/`tmin`/`tmax` batched over `[N, 5]` uint8 trit-encoded numpy arrays, runtime `has_avx2()` dispatch between scalar and AVX2 kernels (graceful degradation on a CPU without AVX2, not a compile-time-only choice — matches this project's established convention, e.g. the AVX2-at-import-time crash class of bug already fixed in `bindings_dense243.cpp`). `build/build_tritnet_inference.py` (mirrors `build_tritnet_gemm.py`). Correctness verified against the full input space (243/59,049 samples per op) plus input validation (shape, dtype range, batch-size mismatch) in `tests/python/test_tritnet_inference_bindings.py`, wired into `run_tests.py` (15/15).

**Phase 4** - GPU Acceleration ✅ COMPLETE (2026-08-17) — LUT still wins, GPU confirms rather than reverses Phase 3
- Batch inference optimization ✅ `models/tritnet/phase4_gpu_benchmark.py` — batched PyTorch/CUDA forward pass over the same `models/tritnet/phase2b_export/<op>/*.npy` weights Phase 3's C++ engine uses; correctness verified exact (fp32) / near-exact (fp16, tiny documented argmax-tie drift on tmin/tmax, <0.004pp) over the FULL input space per op. fp16 gives a real ~1.5-2x throughput gain over fp32; a standalone `torch.compile(mode="reduce-overhead")` spot-check on tadd added a further ~1.5x (44.6→68.6 Mops/s compute-only) but wasn't wired into the automated sweep (narrows, doesn't reverse, the gap — not worth the added complexity per `phase_coherence`).
- GPU/TPU deployment ✅ (GPU only — no TPU path exists in this repo) NVIDIA GeForce RTX 3050 (6GB, compute 8.6), CUDA 12.1 via PyTorch 2.5.1
- Measure actual throughput gains ✅ Same-host CPU baselines re-measured 2026-08-17 (AMD Ryzen 5 4500, not reused from Phase 3's different-host numbers — repeating that cross-run mistake once was enough): LUT tnot 517/binary ~135-149 Mops/s, AVX2 tnot 3.35/binary ~1.00 Mops/s. Best GPU case (fp16, largest batch fitting in 6GB VRAM before OOM, end-to-end incl. H2D/D2H): binary ops ~37 Mops/s (**0.25-0.27x of LUT, but 37-47x of AVX2-CPU**), tnot ~65 Mops/s (**0.10-0.13x of LUT, ~15-19x of AVX2-CPU**). Root cause: these networks (5→64→64→15 / 10→128→128→15) are too small to be GPU-compute-bound at any batch size that fits in 6GB — throughput plateaus by batch~1M instead of continuing to scale, meaning the RTX 3050's FLOPS headroom is never reached; the LUT's ~2ns lookup has no per-call overhead left for batching to amortize away. Full writeup: [reports/2026-08-17/TRITNET_PHASE4_SESSION_REPORT.md](../reports/2026-08-17/TRITNET_PHASE4_SESSION_REPORT.md).

**Phase 5** - Learned Generalization — remaining place TritNet's practical case could rest (per Phase 3 and Phase 4 conclusions: not raw throughput, on any hardware tried so far)
- Explore approximate arithmetic ✅ FIRST RESULT (2026-08-17) — `models/tritnet/phase5_error_characterization.py` asks whether the 3 imperfect checkpoints' errors (tmul 99.49%, tmin 99.89%, tmax 99.85%; tadd/tnot both 100% serve as zero-error controls) are structured or noise, using this project's mandated ternary-native metrics (valuation depth v3, reusing `research/scripts/falsify.py`'s exact convention; sparsity). Finding: **structured, not noise, but not "graceful" either.** All 3 imperfect ops show errors clustering strongly by input sparsity (chi-square p<1e-100, driven by well-populated bins not small-n artifacts) — the sparsity extremes (near-all-zero or near-no-zero inputs) are 10-40x more error-prone than the well-populated middle, which is if anything *more* reliable than the headline accuracy. Valuation-depth clustering holds for tmin/tmax (p<1e-90) but NOT tmul (p=0.31) — reported as a real split, not smoothed over. Margin analysis: errors are usually confident, not near-miss (93-98% of wrong positions have large negative logit margins vs strongly positive margins on correct positions) — this is a predictable sparsity-linked blind spot, not fuzzy/probabilistic behavior near a decision boundary. Full writeup: [reports/2026-08-17/TRITNET_PHASE5_SESSION_REPORT.md](../reports/2026-08-17/TRITNET_PHASE5_SESSION_REPORT.md).
- Discover novel ternary operations — not started
- Research applications — not started

### Training Guidelines

**dataset** - Use complete truth tables from datasets/tritnet/
**optimizer** - Adam with default PyTorch settings
**target_accuracy** - 100% for exact arithmetic (99%+ acceptable)
**validation** - Hold-out test set from truth tables
**checkpointing** - Save models to models/tritnet/ with .tritnet extension
**code_unification** - Avoid duplicating QAT layers and training loops across `train_phase2a.py` and `train_phase2b.py`. Share common modules (like `ternary_layers.py` or training orchestration functions) to ensure bug fixes apply globally.

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

**Run experiments:** not currently possible — `models/gemm_discovery/` does
not exist in this repository (see note above); these commands are kept as a
historical record of what was run to produce the table above, not a
runnable reference.
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
- **Falsification code:** `models/gemm_discovery/experiments/ternary_gemm_falsification.py` (does not exist — see note above)
- **Ultrametric energy:** `models/gemm_discovery/ebm/ultrametric_energy.py` (does not exist — see note above)

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

All under `benchmarks/python-with-interpreter-overhead/` (corrected 2026-08-16;
`bench_phase0.py`/`bench_power_consumption.py` were renamed Nov 2025 and this
section hadn't caught up):

**bench_simd_core_ops.py** - Core performance suite (was bench_phase0.py)
**bench_competitive.py** - 6-phase competitive analysis
**bench_model_quantization.py** - Real model testing
**bench_power_efficiency.py** - Energy efficiency (was bench_power_consumption.py)

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
2. **TritNet Phase 3 pending** - Phase 2B GO achieved 2026-08-11: 4/4 ops ≥99% with ternary weights (tadd 100%, tmul 99.5%, tmin 99.9%, tmax 99.9%). Next: Phase 3 C++ inference engine (weight export, C++ inference, TritNet-vs-LUT benchmark — the experiment that decides whether TritNet beats LUTs in practice; note LUT does ~20K fewer MACs per 5-trit op, see research/PRIOR_ART_TERNARY_LANDSCAPE.md context). Blocking bugs fixed 2026-08-12 (see reports/2026-08-12/CODE_REVIEW_SESSION_REPORT.md): the AVX2 TritNet GEMM kernel computed wrong results for any output width >1 (row-stride bug, masked because the validation function compared the naive kernel against itself and could never have caught it) and was never actually reachable from the Python `gemm()` API regardless; both fixed and verified. `tritnet_gemm_f32_avx2_tiled` has the same class of bug but is still unreachable/unfixed (no test coverage to verify a fix against). **Weight export unblocked 2026-08-14**: the "checkpoint format incompatibility" was actually two fully disjoint architectures, not a save-wrapper mismatch — `tritnet_model.py`'s `TritNetUnary`/`TritNetBinary` (backed by `ternary_layers.TernaryLinear`, no bias, direct-regression output, has `export_weights_to_numpy()`) turned out to be a stale/abandoned pipeline: its only surviving checkpoint (`tritnet_tadd.tritnet`) is 15.8% accurate, not the 100% the roadmap documents. The real GO checkpoints were trained by `train_phase2a.py`/`train_phase2b.py`'s own local `TritClassifier`/`TernaryLinearQAT` (bias included, CrossEntropy classification head, ReLU hidden layers) — structurally incompatible with `tritnet_model.py`'s classes, and `train_phase2a.py` additionally never called `torch.save` anywhere, so tnot's GO model (documented complete since Phase 2A) had no on-disk checkpoint at all despite `train_phase2b.py`'s 4 binary ops already having theirs. Fixed: added checkpoint save/resume to `train_phase2a.py` (now saves to `models/tritnet/phase2a/tnot/`, mirroring `train_phase2b.py`'s pattern) and re-ran it (100% reproduced); wrote `models/tritnet/export_weights.py` targeting the real `TritClassifier` architecture directly, exporting all 5 ops' quantized int8 weights + biases to `models/tritnet/phase2b_export/<op>/*.npy`; added `tests/python/test_tritnet_export.py` (wired into `run_tests.py`, suite now 14/14) — a pure-NumPy replay of the exported weights over the full input space (243 / 59,049 samples per op) that reproduces each op's recorded checkpoint accuracy bit-for-bit. **C++ inference engine done 2026-08-14** (`models/tritnet/inference/tritnet_inference.h` naive/scalar + `tritnet_inference_avx2.h` AVX2, all 5 ops both paths; correctness verified bit-exact against every op's recorded accuracy over the full input space, and AVX2 verified bit-identical to scalar, via `tests/cpp/test_tritnet_inference.cpp`) — and the decisive TritNet-vs-LUT benchmark now has real numbers for both: naive scalar loses by 950×–1776×; **AVX2 recovers ~10.2×–10.9× (as predicted, the low end of this repo's usual AVX2 gain) but LUT still wins by 169×–195×** (`benchmarks/cpp-native-kernels/bench_tritnet_inference.cpp`, 2026-08-14, Linux x64, AMD Ryzen 5 7520U). See "TritNet Development" → Phase 3 above for the full figures. Phase 3 is now complete. Three rounds of user-requested fairness review followed the same day, one of which caught and corrected a real bug in the benchmark itself (an unfair early-vs-late timing comparison, same class of issue as the retired "8,234× vs Python" claim); a fourth extended the corrected check to the last op (tmax) — see "TritNet Development" → Phase 3 above for the corrected numbers and full session report: reports/2026-08-14/TRITNET_PHASE3_SESSION_REPORT.md. **Python bindings wired up 2026-08-14**: `src/engine/bindings_tritnet_inference.cpp` exposes `tnot`/`tadd`/`tmul`/`tmin`/`tmax` as `ternary_tritnet_inference`, batched over `[N, 5]` uint8 trit-encoded numpy arrays, with runtime `has_avx2()` dispatch between the scalar and AVX2 engines (graceful degradation, matching this project's established convention rather than a compile-time-only flag). `build/build_tritnet_inference.py` builds it (mirrors `build_tritnet_gemm.py`). `tests/python/test_tritnet_inference_bindings.py` verifies all 5 ops against the full input space plus input validation, wired into `run_tests.py` (15/15). **Phase 4 (GPU) complete 2026-08-17**: batched PyTorch/CUDA inference on a real GPU (RTX 3050) confirms rather than reverses Phase 3 — best case (fp16, largest batch fitting in 6GB VRAM) reaches only ~0.25-0.27x of LUT throughput for the binary ops and ~0.10-0.13x for tnot, though it does beat AVX2-CPU by 15-47x. Root cause: the trained networks are too small to reach a GPU-compute-bound regime at any batch size that fits in VRAM (throughput plateaus around batch~1M rather than continuing to scale). See "TritNet Development" → Phase 4 above and reports/2026-08-17/TRITNET_PHASE4_SESSION_REPORT.md. Remaining work is Phase 5 (Learned Generalization) — per both Phase 3 and Phase 4, TritNet's practical case has to rest on capabilities a LUT can't offer, not on throughput.
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

### 2026-08-14 src/core/ + src/engine/ bug hunt — first dedicated pass

User-requested ("check the engine for bug hunting session"). `src/core/` and
`src/engine/` — the production kernel and Python bindings — had not been the
subject of a dedicated review pass before; prior sessions covered
`benchmarks/`, `research/`, `models/`, and `opentimestamps/`. Two rounds:
first pass (code-review skill, high effort) scoped itself to only the diff in
the immediately-preceding commit (the new TritNet Python bindings, see gap #2)
rather than the full directories — found and fixed 4 real issues there
(commit `14157d1`): an ISA-portability bug (`build_tritnet_inference.py` used
`-march=native` despite the module's own docstring promising graceful
degradation via `has_avx2()` on any AVX2-only machine — switched to
`-march=haswell`, matching `build.py`'s/`build_dense243.py`'s convention, the
same crash class already fixed once for `bindings_dense243.cpp` 2026-08-12);
an imprecise blended performance claim (fixed to cite the real, separately-
labeled scalar/AVX2 figures with date/platform); a misleading comment
claiming prior art for `InvalidTritError`'s per-value validation that doesn't
exist elsewhere in the codebase; and redundant `py::array::request()` calls.

A second, explicitly broader pass (background general-purpose subagent,
verify-by-execution discipline) actually covered the full production kernel:
`ternary_algebra.h`/`ternary_lut_gen.h`, all of `src/core/simd/` (AVX2
kernels, backend plugin system, CPU detection), `src/core/packing/`,
`src/core/common/`, `src/core/config/`, `src/core/profiling/`,
`src/core/ffi/`, and `src/engine/bindings_core_ops.cpp`/
`bindings_dense243.cpp`/`bindings_zero_skip_gemm.cpp`/`bindings_backend_api.cpp`.
Traced the 2-bit trit encoding, canonical `a*3+b` indexing, and the int8↔uint8
bridge isomorphism end-to-end across every LUT generator and AVX2 kernel and
confirmed they're mutually consistent — no wrong-numeric-result bugs found in
any load-bearing path (Dense243/TriadSextet pack/unpack LUTs both have
exhaustive compile-time round-trip `static_assert`s that hold; OpenMP
streaming-store fencing in `bindings_core_ops.cpp`/
`backend_avx2_v2_optimized.cpp` correctly re-checks alignment and fences
per-thread, matching each file's own prior fix comments). Two real findings,
both fixed (commit `d5b792c`):

- **`ternary_gemm_zero_skip.cpp`**: `ternary_gemm_zero_skip()` (the
  convenience wrapper reachable from Python via
  `ternary_zero_skip_gemm.gemm()`) called `build_ternary_csc()` and passed
  the result straight into `ternary_gemm_zero_skip_avx2()` without checking
  for the `nullptr` that function explicitly returns on malloc failure —
  every other call site in the file, and `ZeroSkipWeights`'s constructor in
  `bindings_zero_skip_gemm.cpp`, already checked this. Added the same guard.
- **`backend_avx2_v2_optimized.cpp`**: `init_canonical_luts()` lazily
  initialized five global `__m256i` LUTs guarded by a plain non-atomic
  `bool` — a data race (UB under the C++ memory model) if two threads called
  any `avx2_v2_*` dispatch function for the first time concurrently.
  Confirmed unreachable through any Python entry point that exists today
  (`bindings_backend_api.cpp` never releases the GIL around dispatch calls),
  but fixed anyway per this doc's own "no_undefined_behavior - Strict C++17
  compliance" principle — it becomes live UB the moment anyone adds
  `gil_scoped_release` for parallelism. Replaced with `std::call_once`.

Both rebuilt (`ternary_zero_skip_gemm`, `ternary_backend`) and re-verified:
`tests/run_tests.py` 15/15, plus a direct functional smoke test of each
module's real output against a NumPy reference. One dead-code finding
(`opt_dual_shuffle_xor.h`/`opt_lut_256byte_expanded.h`) matched what this
doc already documents (gap #1's `test_dual_shuffle_validation.py` note) — no
new action needed.

### 2026-08-15 build/ + scripts/ bug hunt — first dedicated pass

User-requested ("review the project and commit it"), scoped to `scripts/`
and `build/*.py` (`scripts/` turned out to only hold 2 dev-utility files —
its `build/`, `tritnet/`, `orchestration/` subdirectories referenced
elsewhere in this doc no longer exist there; the actual build automation
lives in `build/` at repo root, 12 real scripts / ~3,700 lines). Neither
directory had a prior dedicated pass (earlier sessions covered
`benchmarks/`, `research/`, `models/`, `opentimestamps/`, `src/core/`,
`src/engine/`). Ran via the code-review skill (high effort, 8 finder
angles, 1-vote verify); the orchestrating fork got tangled tracking which
of its own 9 async verification sub-agents had reported back and stalled
mid-run without delivering a final report — findings were pulled directly
from the verification sub-agents' own completed transcripts (all genuinely
CONFIRMED) and applied by hand. 14 bugs fixed across 11 files, commit
`4e72be2`:

- **`build_test_packing.py`**: test file path was stale
  (`tests/test_packing.cpp` → `tests/cpp/test_packing.cpp`, moved in an
  earlier reorg) — script always reported "not found"; also quoted every
  interpolated path in the `shell=True` MSVC command (a space anywhere in
  the repo path broke it).
- **`clean_all.py`**: `clean_benchmark_results()` globbed for
  `bench_results_*`, a prefix no real file in `benchmarks/results/` uses
  (real files are `fair_baseline_*`, `zero_skip_*`, `canonical_fix_*`,
  etc.) — cleanup silently did nothing. Matched on extension instead;
  verified with `--dry-run` (now finds and would clean 14 real files).
- **`build_pgo_unified.py`**: `--clang` on Windows silently built plain
  non-instrumented MSVC (`build.py`'s Windows path is hardcoded to
  `MSVCCompiler`, which never consults `CPPFLAGS`/`LDFLAGS` or any
  `--compiler` override), then failed at Phase 3 for a confusing reason
  ("no .profraw files"). Now fails fast with a clear explanation.
- **`build_reference.py`**: `extra_compile_args` was MSVC-only syntax with
  zero platform branching (unlike every sibling `build_*.py`) — build
  failed outright on Linux/macOS; added Unix flags matching `build.py`'s
  pattern. `copy_to_latest()`/`print_summary()` also only globbed `*.pyd`,
  never `*.so` — the built module silently wasn't copied to the project
  root on Linux/macOS.
- **`build_all.py`**: the "unified" build entry point never called
  `build_backend.py`, `build_zero_skip_gemm.py`, or
  `build_tritnet_inference.py` (3 of 8 `build_<target>.py` scripts), unlike
  CI which builds all of them individually. Added all three with matching
  `--no-*` skip flags.
- **`setup_dev_environment.py`**: `main()` discarded
  `build_modules()`/`run_tests()`'s return values and always printed
  "Setup Complete!"/exited 0 even on failure; now propagates failure to the
  exit code. `check_prerequisites()` looked for a `"python"` binary
  specifically, absent on a typical Linux system without
  `python-is-python3` — switched to `sys.executable`.
- **`build_backend.py`**: GCC/Clang flags used `-march=native` (SIGILL risk
  if built on a newer-ISA machine and redistributed to this project's
  documented AVX2-only baseline — the crash class already fixed once for
  `bindings_dense243.cpp`); switched to `-march=haswell`, matching
  `build.py`/`build_dense243.py`/`build_tritnet_inference.py`. This module
  is built directly in CI, so the risk was live. Also wrapped the primary
  `shutil.copy2()` in try/except (the very next block already degraded
  gracefully; this one didn't). Rebuilt and validated on Linux.
- **`build_backend.py` + `build_zero_skip_gemm.py`**: `-fopenmp` was added
  unconditionally with no ARM/Apple-Clang guard, contradicting this doc's
  own documented rule ("disabled only on ARM and Apple Clang") which
  `build.py` already implements correctly. Added the same guard to both;
  rebuilt and validated on Linux (OpenMP still enabled here, as expected on
  x86_64).
- **`build.py`**: `copy_to_latest()` printed `"[ERROR] No module files
  found!"` and returned, but `main()` never checked the return value —
  `print_summary()` unconditionally printed `"[SUCCESS] BUILD COMPLETE"`
  and the script exited 0 regardless. Now propagates the failure and exits
  1.
- **`.github/workflows/ci.yml`**: `build_tritnet_inference.py` was never
  run in CI, so `ternary_tritnet_inference` was never built there;
  `test_tritnet_inference_bindings.py`'s import-guard makes it self-skip on
  `ImportError` but `run_tests.py`'s bookkeeping counts that as PASSED, not
  SKIPPED — CI's "15/15" tally silently included a suite that never ran its
  real assertions. Added the missing build step.

Verified: all edited files `py_compile` clean; `ci.yml` YAML-parses;
`build_backend.py`/`build_zero_skip_gemm.py` rebuilt clean on Linux with the
new flags (AVX2 validated, zero-skip GEMM correctness checks pass);
`build_all.py --help` shows the 3 new skip flags; `tests/run_tests.py` 15/15
passing throughout.

### 2026-08-16 docs/ + tests/ review — and a real "15/15 was never real" finding

User-requested ("review docs/ and tests/ next"), immediate follow-up to the
scripts/+build/ session above. Two parallel passes: `tests/` (~7,800 lines)
via the code-review skill (high effort, --fix); `docs/` (90 markdown files)
via a general-purpose fact-checking agent, since a link/claim audit isn't
the diff-oriented code-review skill's job. Commits `a291ebb`, `bd3e6f6`,
`24104b1`.

**tests/ findings** (10, one documented skip): `test_errors.py`'s
`test_invalid_trit_values()` unconditionally returned `True` regardless of
outcome; `test_simd_validation.py`'s 1000-trial fuzz loop's bare `except:`
counted a crash as a pass; `run_tests.py`'s capability-skip branch only
matched `'openmp'` (not the already-supported `'fusion'`), and
`run_test_suite()` counted any exit-code-0 subprocess as "Passed" even
suites that self-skip via `sys.exit(0)` after printing `"[SKIP]"`; every
suite's `required` field was defined but never read, so an optional suite's
failure flipped the overall exit code like a required one;
`test_tritnet_gemm_integration.py` hard-failed instead of self-skipping when
its optional module wasn't built; `test_backend_integration.py`'s
tadd/tmul dispatch tests only range-checked output instead of checking
exact values (unlike their tnot/tmax/tmin siblings in the same file);
`test_capabilities.py`'s `_detect_openmp()` conflated "module not built"
with "probe call crashed" under one blanket `except`; `test_omp.py` had no
`np.random.seed()` despite this project's own documented convention.
Skipped: `test_canonical_lut.py`'s fake pass/fail accounting (the file's own
comment says it deliberately doesn't test the real C++ engine yet) —
documented, not fixed, as a scope expansion beyond a contained bug.

**The real find**: re-verifying the `run_tests.py` skip-detection fix in a
clean environment (`env -u PYTHONPATH`) surfaced that `test_dense243.py`,
`test_zero_skip_gemm.py`, and `test_tritnet_inference_bindings.py` were
missing the `sys.path.insert(0, str(PROJECT_ROOT))` that
`test_backend_integration.py` already has — without it, `import
ternary_dense243_module` (etc.) can't find the compiled module in the
project root when the script runs from `tests/python/`, so all three
self-skip via their own `"[SKIP] ... not built"` branch **even when the
module is genuinely built**. The now-fixed counting bug had been silently
counting that self-skip as "Passed" — meaning this session's own earlier
"15/15" claims (both today's build/scripts session and, per grep, every
prior session's CI badge) were never a real pass for these 3 suites. Fixed
by adding the same `sys.path` setup used elsewhere; re-verified genuinely
PASSED (not skipped) with all 6 optional modules built, in a clean
environment. `tests/run_tests.py`: 15/15, 0 skipped.

**docs/ findings** (16 files fixed): systemic stale file-name references in
`FEATURES.md` (SIMD/backend/canonical-index/dual-shuffle/fusion files all
renamed in earlier reorgs this doc never caught up on) — rewritten
wholesale; 8 broken links in `api-reference/source-code-overview.md` (the
doc `docs/README.md` itself labels "START HERE"); 58 occurrences of a
nonexistent `build/scripts/setup*.py` path across `build-system/{README,
setup-standard,setup-pgo,setup-reference}.md` (real scripts are
`build/build*.py`, no `scripts/` subdirectory — `setup-standard.md` even
contradicted itself, one line saying `build.py` is at the project root, the
rest of the file saying `build/scripts/setup.py`); a nonexistent
`TESTING.md` referenced from 2 environment docs; broken cross-references
between the two encoding spec docs; 5 broken links in `planning/ROADMAP.md`;
a factually-reversed `InvalidTritError` claim in `error-handling.md`
("not thrown in production" — it has been, since 2026-08-14); an overstated
VTune "fully integrated and tested" claim in `profiling/README.md` (the
call sites are real, the backend itself is unbuilt, matching CLAUDE.md's
own already-corrected framing — also flagged that the documented
CPPFLAGS/LDFLAGS VTune-build workflow doesn't work against the current
`build.py`, which hardcodes flags and never reads those env vars); a stale
"canonical indexing deferred" note in `planning/backend_api_design.md`
(shipped in v1.3.0); a stale "DESIGN COMPLETE, Next: Implementation of
Phase 1" status in `architecture/BRIDGE_LAYER_ARCHITECTURE.md` (Phase 1 is
implemented and shipped); a retired-style "7,315× avg speedup" claim in
`encoding_ecosystem_overview.md` (same compiled-vs-Python strawman class
already retired project-wide). Two of the fact-checking agent's findings
were mis-attributed to the wrong file (broken links actually in
`FEATURES.md` were reported against `README.md`) — caught and corrected
during application rather than applied blindly. Historical/point-in-time
docs (`historical/`, `audits/`, dated analyses) were left alone per their
own carve-out except where they had an active broken link presented as
current, in which case a staleness note was added rather than editing the
historical narrative.

**Process note**: mid-session, the `tests/` code-review agent (sharing the
same working tree as the concurrent `docs/` fact-checking agent) saw an
in-progress `docs/FEATURES.md` diff, assumed it was stray/unrelated to its
own scope, and ran `git checkout -- docs/FEATURES.md` — silently reverting
most of an in-progress edit sequence. Recovered by rewriting the file once
both background agents had finished, with no other agent running
concurrently. Lesson: don't run two `--fix`-capable background agents
against the same working tree at once, or expect one might "clean up" the
other's uncommitted changes.

### 2026-08-16 — same bug class, checked src/core/ and src/engine/

User-requested direct follow-up to the tests/ sys.path finding above:
"review src/core/ and src/engine/ for the same sys.path class of bug."
Neither directory contains any Python (pure C++), so the literal pattern
can't recur there; searched instead for the structural analog -- a
resolution step (capability dispatch, #include paths, dynamic loading,
best-of selection) that silently degrades/fails, currently non-triggering
only because of an unstated invariant elsewhere, with nothing testing the
edge case. Checked: runtime `has_avx2()` dispatch across
`bindings_core_ops.cpp`/`bindings_tritnet_gemm.cpp`/
`bindings_tritnet_inference.cpp` (all loud -- throw or return a checkable
bool, not silent); backend registry availability filtering (an intentional
capability filter, not a masking bug); duplicate header basenames across
`src/core/`+`src/engine/` (none); runtime file I/O / `dlopen` (none present
in either directory). Commit `1d6eefd`.

**Found one real match**: `backend_registry_dispatch.cpp`'s
`ternary_backend_select_best()` scored each registered backend and kept
the highest via `if (score > best_score)`, starting from
`best = NULL, best_score = 0`. If every registered backend scored exactly
0 (none of the scored capability bits set), `best` stays `NULL` for the
whole loop -- spuriously reporting "no backends available" even with a
valid, registered backend, exactly the same shape as the tests/ bug: a
resolution step that silently fails today only because of an assumption
nothing enforces (here, that Scalar's capabilities bitmask always includes
`TERNARY_CAP_FUSION`, worth 25 points -- not because the function itself
guarantees a nonzero-scoring candidate exists). The downstream error paths
(`ternary_backend_init()`, the Python `tadd()`/etc. dispatch wrappers) are
all properly loud already (fprintf + false, surfaced as a thrown Python
exception) -- this fix closes the actual silent-failure point rather than
a downstream symptom. Fixed with an explicit `have_best` flag so the first
registered candidate is always selected regardless of score. Rebuilt
`ternary_backend`; verified AVX2_v2 is still correctly selected as active
(unchanged real-world outcome, since it always outscores the alternatives
today), `tests/run_tests.py` 15/15.

### 2026-08-16 — same bug class, checked benchmarks/

Direct follow-up to the two reviews above ("review benchmarks/ for the same
bug class"). Unlike `src/core/`/`src/engine/`, `benchmarks/` genuinely has
Python, so both the literal `sys.path` pattern and its silent-degradation
cousin were worth checking — and both turned up real, live instances.
Commit `73d0aeb`.

**sys.path off-by-one (5 files, `benchmarks/deprecated/`)**: all computed
`PROJECT_ROOT` as `Path(__file__).parent.parent`, correct for files
directly in `benchmarks/` but 1 `.parent` short for this subdirectory (2
levels deep) — the exact bug already fixed in ~10 other files across this
repo in the 2026-08-12 session (sibling files in this same directory
already carry a "# fixed 2026-08-12" comment for the identical bug; these 5
were missed then). Fixed; verified all 5 now resolve correctly and actually
import/run.

Fixing these surfaced that `benchmarks/deprecated/README.md`'s core claim —
"depend on the deprecated `ternary_backend` module, replaced by
`ternary_simd_engine`" — is now stale: a different, unrelated
`ternary_backend` module has since been built from scratch (the v1.2.0
pluggable backend system, in CI since 2026-08-12) and turns out to be
API-compatible with these old scripts by coincidence, not design. Verified
all 5 fixed scripts run successfully against it today. Added a correction
note; the directory remains deprecated (a successful run isn't validation
of stale methodology).

**Silent mock-fallback (`test_falsification.py`)**: this file's own
docstring says "Tests que DEBEN pasar o el proyecto no tiene valor" (tests
that MUST pass or the project has no value) — yet its `ImportError` handler
silently substituted plain NumPy for `ternary_add`/`ternary_mul` when
`ternary_simd_engine` isn't importable, then continued to a full "VALIDATED"
verdict from comparing NumPy against itself. A `ternary_available` flag was
already computed at import time but never read again — the only trace of
degraded mode was one `[WARN]` line, easy to miss, with no marker in the
saved JSON or final verdict. The same shape as the already-fixed
`bench_competitive.py` bug (gap #3): a resolution failure silently
substituting fake data that looks real. Fixed by propagating the flag into
the saved JSON (`using_real_engine` + a WARNING field), the console summary
(an unmissable banner + a verdict string reading "NOT RUN AGAINST REAL
ENGINE" instead of "VALIDATED"), and the exit code (`2`, distinct from
PASS=0/FAIL=1, so a caller checking only the exit code can't mistake a mock
run for real either). Verified both paths explicitly (real engine, and
mock via temporarily hiding the `.so`).

**Bonus find while verifying the above**: `save_results()` crashed with
`TypeError: Object of type bool is not JSON serializable` on every single
real run — `np.array_equal()` returns `numpy.bool_`, and
`correctness_passed and matches` (Python `and` returns an operand, not a
coerced bool) let that leak into a dataclass field needing JSON
serialization; the same leak class existed at 2 more sites (`cv < 0.3`,
`overhead < 2.0` — any numpy-scalar-vs-Python-literal comparison yields
`numpy.bool_`). This file had apparently never been run to a successful,
complete finish before. Fixed once, defensively, at
`record_falsification()` (`bool(passed)`) rather than chasing each site.

Running the now-fixed script for real reports **FALSIFIED (2/5 criteria)**
in this environment — flagged here per this project's own stated rule
(`SKEPTICAL_METRICS.md`: "Si un test de falsificacion falla, PUBLICARLO, no
ocultarlo") rather than omitted, though it's a single run in a shared,
non-isolated dev container (CV 64-76%, well above this project's own rigor
bar), not a controlled benchmark — a data point that needs a properly
isolated re-run to mean anything, not a claim.

### 2026-08-16 — same bug class, checked research/ — clean bill of health

Direct follow-up ("review research/ for the same bug class"). `research/`
is small (just `research/scripts/falsify.py`, 2,815 lines) and had already
had a thorough dedicated pass in the 2026-08-12/13 sessions (the inverted-
3adic-valuation bug, H9's zero-check, H1's skewed sample, missing pre-flight
guards). Re-checked specifically for the sys.path class and its
silent-degradation cousin. Commit `9acefb4`.

Genuinely clean this time: `ROOT` computation (3 `.parent` calls for a file
2 levels deep) is correct, verified both by inspection and by actually
running `--hypothesis H6` end-to-end (exit 0). Every component-load failure
path (data, hyperbolic, ultrametric, LUTs, trained model, corpus) already
prints an explicit status marker and propagates a real failure signal — no
silent mock-fallback anywhere, unlike the `benchmarks/` findings above.
Documented-not-fixed: `self._load_status` and `FalsificationRunner.config`
are both write-only (set, never read) — harmless, since the real control
flow uses each load method's return value / a local `status` dict instead,
not these fields.

**Found while checking the `sys.path.insert` lines**: `models/gemm_discovery/`
does not exist anywhere in this repo — confirmed via `git log --all` it was
never actually committed, not deleted later. `falsify.py`'s
`sys.path.insert()` for it is a genuine no-op (nothing imports from it).
More significantly, CLAUDE.md's own "Archived Example: GEMM Discovery"
section above (§ Ternary vs Binary Assumptions) presented 2 runnable
`python models/gemm_discovery/...` commands and 3 more file-path references
as if they work today. Added notes marking all 5 as currently unreachable
— kept as historical record, not deleted, since the section's actual
lessons about ternary-native metrics are still the project's position.

### 2026-08-16 — same bug class, checked models/

Direct follow-up ("review models/ for the same bug class"). `models/` had
a thorough dedicated pass in the 2026-08-13 session (`3-vae-gemm-v1/` +
`company-flagships/`: the inverted-valuation bug found 9 more times,
missing metrics, non-reproducible seeding, NaN risk, stale training
embeddings, fake tests); `tritnet/` reviewed 2026-08-12; `bitnet/` is an
empty placeholder. Checked all 26 Python files for both halves of the bug
class. Commit `723893c`.

`sys.path`/`PROJECT_ROOT` depth math: verified correct at all 7 sites that
compute one, cross-checked programmatically against each file's real
directory depth. `export_weights.py`/`generate_weights_header.py`
intentionally use a local `models/tritnet/` root rather than the repo
root, correctly (they only need sibling modules / the `phase2b_export/`
output dir). No bug found. Every `except ImportError` site (7 total) was
already loud and correct: pre-flight checkers, hard `sys.exit(1)`, or an
explicit `warnings.warn()` + a genuinely equivalent fallback (not fake
data — `data.py`'s Python fallback computes the same ternary ops, just
slower; `ternary_layers.py`'s GEMM fallback is PyTorch's own real
`F.linear`).

**Found 2 real instances of the deeper pattern** — a fallback producing a
fake-but-plausible value with zero indication anything went wrong, same
shape as the mock-fallback bugs found in `benchmarks/` — both in
`models/company-flagships/embedding_exactitude_score.py`:
- `compute_hierarchy_metrics()`: `except ImportError: ami = 0.0` when
  scikit-learn isn't installed — not a documented project dependency
  anywhere, so a live risk, not hypothetical. `0.0` is also a real,
  legitimate point on AMI's own scale ("no mutual information"), making a
  missing-dependency run indistinguishable from a genuine null finding,
  and it feeds directly into `hc_score`, a composite checkpoint-quality
  score. Added a loud warning.
- `compute_linearity_metrics()`'s inner loop: a bare `except: r2 = 0.0`
  around `np.linalg.lstsq` — same ambiguity (`r2=0.0` is a real "no linear
  relationship" value), and a bare except also swallows genuine bugs.
  Narrowed to `except Exception as e` with the same style of loud warning.

### 2026-08-16 — same bug class, checked docs/ (round 2)

Direct follow-up ("review docs/ for the same bug class"). `docs/` can't
have a runtime `sys.path` bug (it's markdown), so this searched for the
structural analog: documented paths/commands presented as current that
don't actually resolve — the same failure-to-verify shape as every other
fix in this chain, manifesting as prose instead of code. Commit `34d3be0`.

Found two categories. **A residual bug from this session's own earlier
docs/ pass**: `setup-standard.md`'s "Build Script Structure" snippet had
its header comment corrected from `build/scripts/setup.py` to
`build/build.py` earlier this session, but the accompanying path-depth
math was left as-is — correct for the old assumed location, now one level
too many for the corrected one. A reminder that "fix the label, not the
logic" is exactly how this bug class recurs.

**A genuine gap in the original docs/ pass**: its automated link-checker
only verified `[text](url)` markdown syntax, never plain backtick-quoted
prose paths — how most script paths in these docs are actually written. A
broader regex sweep found `artifact-organization.md` was never covered by
the original pass at all (wrong `build/scripts/` layout, a nonexistent
`latest/output/` subfolder — verified against a real `python build/build.py`
run that `latest/` holds the module directly, no subfolder, and standard
builds use `t`/`o` not `temp`/`output` — that naming genuinely differs by
build type), plus 2 more files the link-checker's syntax gap let slip
through. Most significantly: **`bench_phase0.py` doesn't exist anywhere in
the repo** — renamed to `bench_simd_core_ops.py` in a Nov 2025 refactor
(confirmed via `git log`) — yet was still referenced as current in 9 active
docs/ files (58 occurrences) plus this very file's own "Standard
Benchmarks" section (which also had `bench_power_consumption.py`, renamed
the same commit to `bench_power_efficiency.py`). The rename touches 30+
files project-wide; scoped this pass to docs/ + CLAUDE.md per explicit user
confirmation — README.md, CONTRIBUTING.md, `benchmarks/README.md`,
`tests/README.md`, and historical/archived docs (which accurately describe
what was true when written) are still untouched.

Bonus: `benchmarks/python-with-interpreter-overhead/run_all_benchmarks.py`'s
own docstring had the identical missing-subdirectory bug in its own Usage
examples — fixed directly in the script, not just the docs describing it.

### 2026-08-16 — same bug class, checked reports/

Direct follow-up ("review reports/ for the same bug class"). No Python
there either (36 markdown files + data), so again the structural analog:
documented paths presented as current that don't resolve. Unlike docs/,
`reports/` is fundamentally a point-in-time record by design — every
subdirectory checked (including the undated-looking `architecture/`,
`roadmaps/`, `process/`, `research/`) turned out to carry an explicit
date/status header, so the historical-narrative carve-out applies far more
broadly here. Most of the ~140 backtick-quoted report paths repo-wide
already resolved correctly. Commit `c8c1095`.

Found one well-defined sub-pattern: several reports were renamed/moved into
subdirectories during a later reorganization, but their own internal self-
citations (typically "`reports/OLD_NAME.md` (this document)") were never
updated — mechanical citation breaks, not a rewrite of what the report
found, so fixing them doesn't touch the historical narrative. Fixed 4 such
self-citations (`phase1_invariant_measurement_complete.md`,
`dtype_bug_investigation.md`, `session_2025-11-26_phase1_merge.md`,
`mandatory_benchmarking_policy.md`) plus every place they cross-reference
each other under the old names, keeping the old name as an "originally X"
annotation rather than deleting it. Also fixed 2 archive-internal sibling
references still pointing at the pre-archival flat path, and a genuinely
broken (non-self-referential) citation — both `MISSING_FEATURES.md` and
`README.md` cited a `reports/reasons.md` that doesn't exist under that name
anywhere; the real file, verified by content match, is
`reports/performance/gemm_gap_root_cause.md`. `CHANGELOG.md`'s own mention
of the same old name was deliberately left alone — a changelog "Added:"
line describing what a specific past commit added, not a live pointer.

Investigated and deliberately left alone: 2 ambiguous self-references in
dated planning docs (unclear whether they mean a sibling file, a not-yet-
written deliverable, or a stale name); an explicit unchecked `- [ ]` TODO
item; and a literal `YYYY-MM-DD` template placeholder.

### 2026-08-16 — same bug class, checked benchmarks/README.md

Direct follow-up ("review benchmarks/ README.md for the same bug class") —
this specific file was one of the 30+ identified during the docs/-round-2
review but explicitly deferred (that fix was scoped to docs/ + CLAUDE.md
only). Dated "Last Updated: 2025-10-14", predating the Nov 2025
reorganization entirely — effectively every script path in it was stale
(18 `bench_phase0.py` occurrences alone). Commit `6ecc4f0`.

Comprehensive rewrite, verified against the real current scripts rather
than pattern-matched: `bench_phase0.py` → `bench_simd_core_ops.py`,
`bench_compare.py` → `bench_regression_detect.py` (confirmed identical
CLI), `run_all_benchmarks.py` path + real flags (`--with-pgo`, `--quick`,
`--clean`, `--skip-build`), `python build.py` → `python build/build.py`.
The "Structure" diagram was completely wrong (a flat 3-script layout with
`results/standard`/`results/pgo`/`results/validation` subdirectories that
don't exist) — replaced with the real structure, verified by listing the
actual directory (flat, timestamped-filename `results/`, not build-type
subfolders). Two dead documentation links (`../build/README.md`,
`../docs/PGO_README.md`) fixed to their real current locations; added a
pointer to `python-with-interpreter-overhead/README.md`, this project's
current and more precise framing of Python-benchmark timing reliability
that this file predates entirely. Softened 2025-era Python-baseline
speedup claims with a pointer to the already-retired compiled-vs-
interpreted framing and the current fair-NumPy-baseline alternative.
Relabeled a CI/CD YAML example as illustrative rather than implying it
describes the real (different-purpose) `.github/workflows/ci.yml`.

### 2026-08-16 — same bug class, CONTRIBUTING.md + tests/README.md

Direct follow-up ("review CONTRIBUTING.md and tests/README.md for the same
bug class"). Both dated 2025-10-13 — older even than `benchmarks/README.md`
— predating not just the Nov 2025 script renames but the
`src/core/`+`src/engine/` kernel reorganization and the entire
`tests/python/`+`tests/cpp/` split. Commit `e969a02`.

**The significant finding**: `CONTRIBUTING.md`'s own "Import Path
Convention" section — written specifically to teach contributors the
correct `sys.path` depth-math pattern and prevent exactly the bug class
this whole review chain has been hunting — had wrong depth math in 2 of
its 3 worked examples. It labeled `tests/python/test_phase0.py` as needing
"Depth 3" but showed code with only 2 `.parent` calls (that file is 2
subdirectories deep, genuinely needs 3); labeled
`models/tritnet/src/train_tritnet.py` as "Depth 4" but showed only 3
`.parent` calls (needs 4). Only the `build/build.py` example (depth 2) was
correct. **The project's own guidance was teaching the off-by-one bug it
exists to prevent.** Fixed both, verified programmatically against each
file's real directory depth, and added a compact "count subdirectories,
match `.parent` calls exactly" rule.

Also fixed throughout both files: `python build.py` → `python build/build.py`;
`tests/test_phase0.py`/`tests/test_omp.py` → `tests/python/...`;
`benchmarks/bench_phase0.py` → the real current path; a "Build scripts in
`build/scripts/`" claim (no such subdirectory exists); "keep all .h/.cpp
files at root level (no nesting)" (actively wrong about where a new
contributor should put a new file — real convention is `src/core/`/
`src/engine/`); every `ternary_simd_engine.cpp` source reference → the real
file, `bindings_core_ops.cpp`; `tests/README.md`'s entire "Structure"
section (described a 3-file suite that hasn't existed since the
`tests/python/`+`tests/cpp/` split, and never mentioned
`tests/run_tests.py` at all despite it being the actual current single
source of truth — confirmed against the real CI workflow); a nonexistent
`benchmarks/bench_fair.py` → the real `bench_fair_baseline.py`.

### 2026-08-16 — same bug class, checked scripts/ (round 2)

Direct follow-up ("review scripts/ for the same bug class"). `scripts/`
only has 2 files; both already had a pass in this session's very first
review (2026-08-15, commit `4e72be2`), but that pass only touched
`setup_dev_environment.py`'s `build_modules()`/`run_tests()` return-value
tracking and its `"python"` binary check —
`scripts/generate_compile_commands.py` was never reviewed, and
`setup_dev_environment.py`'s *own separate* `generate_compile_commands()`
function was missed entirely. Commit `5fe5d17`.

`scripts/generate_compile_commands.py` itself: reviewed fresh, genuinely
clean (correct depth math, correct platform branching, globs verified
against every real file under `src/`, runs successfully).

**`setup_dev_environment.py`'s `generate_compile_commands()`**: a real,
live instance of the silent-wrong-result pattern — a *second*,
independently-maintained, drifted copy of the same generator, hardcoded to
MSVC-only flags and `/I`-style includes with zero platform branching
(unlike the real standalone script). On Linux/macOS this silently wrote a
`compile_commands.json` full of `cl /O2 ... /c file.cpp` commands — a
Windows-only compiler invoked with Windows-only syntax — while printing
`"[OK] Generated ... N entries"`, a genuine success message for content
that can't work on the platform it just ran on. Verified concretely by
running the buggy flag-construction logic directly on this Linux machine.
Fixed by deleting the duplicate and delegating to the real script via
subprocess, per this project's own documented single-source-of-truth
principle (the same lesson gap #7 already names for
`train_phase2a.py`/`train_phase2b.py`'s duplicated QAT code).

Also found the identical "discarded return value" bug this exact file's
first pass already fixed for `build_modules()`/`run_tests()`, but missed
for `generate_compile_commands()` — wired into the same gating pattern,
non-fatal but now visibly warning instead of silently discarding a
failure.

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
| 2026-08-17 | v1.33.0 | TritNet Phase 5 (Learned Generalization), first result: "explore approximate arithmetic" asked as a falsifiable question -- are the 3 imperfect checkpoints' errors (tmul 99.49%, tmin 99.89%, tmax 99.85%) structured or noise? New `models/tritnet/phase5_error_characterization.py`, full input space per op, using this project's mandated ternary-native metrics (valuation depth v3 reusing research/scripts/falsify.py's exact convention, plus sparsity), tadd/tnot (both 100%) as zero-error controls. Finding: structured, not noise, but not graceful -- sparsity-extreme inputs are 10-40x more error-prone (chi2 p<1e-100, driven by well-populated bins) while the well-populated middle is more reliable than the headline accuracy suggests; valuation-depth clustering holds for tmin/tmax (p<1e-90) but not tmul (p=0.31), reported as a real split rather than smoothed over; margin analysis shows 93-98% of wrong positions are confidently wrong, not near-miss -- a predictable sparsity-linked blind spot, not fuzzy/probabilistic behavior. Full writeup: reports/2026-08-17/TRITNET_PHASE5_SESSION_REPORT.md. Commit pending. |
| 2026-08-17 | v1.32.0 | TritNet Phase 4 (GPU Acceleration), completing the "actual load-bearing open item" flagged by v1.31.0's roadmap review. Real CUDA GPU available on this machine (RTX 3050, 6GB, compute 8.6) -- no TPU path exists in this repo, so scope was GPU-only. New `models/tritnet/phase4_gpu_benchmark.py`: batched PyTorch/CUDA forward pass over the same exported weights Phase 3's C++ engine uses; correctness verified exact (fp32) / near-exact (fp16, <0.004pp drift on tmin/tmax) over the full input space per op. Result confirms rather than reverses Phase 3: best case (fp16, largest batch fitting in VRAM, end-to-end) reaches only 0.10-0.27x of LUT throughput despite beating AVX2-CPU by 15-47x -- these networks are too small to reach a GPU-compute-bound regime at any batch size that fits in 6GB. Re-measured CPU baselines on this machine rather than reusing Phase 3's different-host numbers (repeating that cross-run timing mistake once, at the CLAUDE.md level, would have been ironic given Phase 3's own internal fairness correction). Full writeup: reports/2026-08-17/TRITNET_PHASE4_SESSION_REPORT.md. Commit pending. |
| 2026-08-17 | v1.31.0 | User request ("review the project roadmap and evaluate what's left"), followed by "document this on the roadmap and push it". docs/planning/ROADMAP.md was dated 2025-11-24 and had drifted badly out of sync: most of its v1.2.0 plan has since shipped (Dense243, TriadSextet, backend interface / TCBI as ternary_backend, canonical index LUT), dual-shuffle XOR is explicitly not implemented (not a gap -- documented as a future enhancement, matching gap #1's test note), and TritNet is far past the doc's "Phase 2A pending" status -- Phases 1-3 are complete, with Phase 4 (GPU) the actual load-bearing open item given Phase 3's finding that LUT beats AVX2-TritNet by 169x-195x on CPU. v2.0-v4.0 (AVX-512, ARM, RISC-V, GPU, FPGA) remain fully unstarted, no code exists for any of them. Added a "Status Reality Check" section to the top of ROADMAP.md cross-referencing this file as the authoritative live source, and flagged that its "Success Criteria" checkmarks further down are the original target list, not a record of achievement. Left the rest of that document as historical plan/context rather than a wholesale rewrite. Commit `e1a93f1`. |
| 2026-08-16 | v1.30.0 | Direct follow-up ("review scripts/ for the same bug class") -- round 2 of scripts/, since round 1 (2026-08-15) missed generate_compile_commands.py entirely and setup_dev_environment.py's own separate, same-named function. Found a real live bug: setup_dev_environment.py maintained a second, drifted copy of the compile_commands.json generator, hardcoded to MSVC-only flags with zero platform branching -- on Linux/macOS it silently wrote a compile_commands.json full of broken `cl` commands while printing a success message. Fixed by deleting the duplicate and delegating to the real standalone script (single-source-of-truth). Also fixed the identical "discarded return value" bug this file's first pass already fixed for 2 other functions, but missed for this one. Commit `5fe5d17`. |
| 2026-08-16 | v1.29.0 | Direct follow-up ("review CONTRIBUTING.md and tests/README.md for the same bug class"). Both dated 2025-10-13, predating the src/core/+src/engine/ reorganization and the tests/python/+tests/cpp/ split entirely. Significant finding: CONTRIBUTING.md's own "Import Path Convention" section -- written to teach contributors the correct sys.path depth-math pattern -- had wrong depth math in 2 of its 3 worked examples (off by exactly one .parent call each, verified programmatically), i.e. the project's own guidance was teaching the bug this whole review chain has been hunting. Fixed both, plus every stale build.py/tests/*.py/benchmarks/*.py path throughout both files, a false "keep .h/.cpp files at root level" claim, and tests/README.md's entire "Structure" section (described a 3-file suite superseded by the tests/python/+tests/cpp/ split, never mentioned tests/run_tests.py despite it being the real current entry point). Commit `e969a02`. |
| 2026-08-16 | v1.28.0 | Direct follow-up ("review benchmarks/ README.md for the same bug class") -- one of the 30+ files identified but deferred during the docs/-round-2 review. Dated 2025-10-14, predating the Nov 2025 script reorganization entirely (18 bench_phase0.py occurrences alone). Comprehensive rewrite verified against the real current scripts: bench_phase0.py -> bench_simd_core_ops.py, bench_compare.py -> bench_regression_detect.py, correct run_all_benchmarks.py flags, a completely wrong "Structure" diagram replaced with the real directory layout, 2 dead doc links fixed, 2025-era Python-baseline speedup claims softened with a pointer to this project's already-retired compiled-vs-interpreted framing, and a CI/CD YAML example relabeled as illustrative rather than implying it describes the real ci.yml. Commit `6ecc4f0`. |
| 2026-08-16 | v1.27.0 | Direct follow-up ("review reports/ for the same bug class"). reports/ is a point-in-time record by design (every subdirectory carries an explicit date/status header, even the undated-looking ones), so the historical-narrative carve-out applied far more broadly than in docs/ -- most of the ~140 report paths repo-wide already resolved. Found one well-defined sub-pattern: 4 reports renamed/moved into subdirectories during a later reorganization still had internal self-citations under their old names ("this document" references broken by the move, not a rewrite of what was found) -- fixed those plus their cross-references to each other, keeping old names as "originally X" annotations rather than deleting them. Also fixed 2 archive-internal sibling references and a genuinely broken citation (`reports/reasons.md`, cited from MISSING_FEATURES.md and README.md, doesn't exist under that name -- real file is `reports/performance/gemm_gap_root_cause.md`, confirmed by content match); left CHANGELOG.md's mention of the same old name alone since it's describing what a specific past commit added, not a live pointer. Commit `c8c1095`. |
| 2026-08-16 | v1.26.0 | Direct follow-up ("review docs/ for the same bug class") -- round 2 of docs/, since docs/ can't have a runtime path bug, so this hunted the structural analog: documented paths/commands presented as current that don't resolve. Fixed a residual bug from this session's own earlier docs/ pass (a path-depth-math snippet left unfixed after its accompanying label was corrected); found the original pass's link-checker only verified markdown `[text](url)` syntax, missing plain backtick prose paths, which let `artifact-organization.md` (never covered at all) and 2 more files slip through; and found `bench_phase0.py` doesn't exist anywhere in the repo (renamed `bench_simd_core_ops.py`, Nov 2025) yet was referenced as current in 9 docs/ files (58 occurrences) plus this file's own "Standard Benchmarks" section. Fixed all of them (docs/ + CLAUDE.md, per user-confirmed scope; the rename is 30+ files project-wide, README.md/CONTRIBUTING.md/benchmarks/README.md left for a separate pass). Also fixed the identical bug in run_all_benchmarks.py's own docstring. Commit `34d3be0`. |
| 2026-08-16 | v1.25.0 | Direct follow-up ("review models/ for the same bug class"). All 26 Python files across 3-vae-gemm-v1/, tritnet/, company-flagships/ checked. sys.path/PROJECT_ROOT math verified correct everywhere (cross-checked programmatically); every ImportError fallback already loud and functionally-equivalent (not fake data). Found 2 real instances of the deeper silent-fallback pattern in embedding_exactitude_score.py: an `ami = 0.0` default when scikit-learn (not a documented dependency anywhere) is missing, and a bare `except: r2 = 0.0` around a linear-algebra solve -- both defaults are legitimate real values on their own metric's scale, making a silent failure indistinguishable from a genuine null finding, and the AMI one feeds a composite checkpoint-quality score. Both fixed with loud warnings. Commit `723893c`. |
| 2026-08-16 | v1.24.0 | Direct follow-up ("review research/ for the same bug class"). research/scripts/falsify.py already had a thorough 2026-08-12/13 pass; re-checked specifically and came back clean -- ROOT computation correct, every component-load failure path already loud, verified by actually running --hypothesis H6 end-to-end. Found one adjacent issue while checking the sys.path.insert lines: models/gemm_discovery/ doesn't exist anywhere in the repo (git log --all confirms it was never committed), yet CLAUDE.md's own "Archived Example: GEMM Discovery" section presented 5 references to it as if current -- annotated all 5 as unreachable rather than deleting the section's still-valid lessons. **Also corrects a date error this session introduced**: the v1.21.0/v1.22.0/v1.23.0 rows below were labeled 2026-08-15, but git log shows their actual commits landed 2026-08-16 (the working session crossed midnight after the v1.20.0 build/scripts commits, which genuinely are 2026-08-15) -- the 3 section headers and changelog dates were corrected to match. Commit `9acefb4`. |
| 2026-08-16 | v1.23.0 | Direct follow-up ("review benchmarks/ for the same bug class"). Unlike src/core/+src/engine/, benchmarks/ has real Python, so both the sys.path pattern and its silent-degradation cousin applied. Found: 5 files in benchmarks/deprecated/ with the exact off-by-one PROJECT_ROOT bug already fixed elsewhere in 2026-08-12 (fixed; also surfaced a stale README claiming ternary_backend was deprecated when a different, unrelated ternary_backend module is now live and API-compatible with these old scripts); test_falsification.py silently substituted NumPy for the real engine on ImportError and still produced a "VALIDATED" verdict with no marker anywhere except one easy-to-miss [WARN] line -- fixed to propagate the flag into the saved JSON, an unmissable console banner, and a distinct exit code. Bonus find while verifying: the script crashed on every real run (numpy.bool_ leaking into JSON serialization) -- fixed, and the now-actually-completable run reports FALSIFIED (2/5) in this environment, published per the project's own transparency rule rather than hidden, with the caveat that it's one noisy non-isolated run, not a controlled benchmark. Commit `73d0aeb`. |
| 2026-08-16 | v1.22.0 | Direct follow-up to the tests/ sys.path finding below (user request: "review src/core/ and src/engine/ for the same sys.path class of bug"). Neither directory has any Python, so searched for the structural analog instead -- a resolution step that silently degrades, masked by an unstated invariant. Found one real match: `backend_registry_dispatch.cpp`'s best-backend scoring loop (`score > best_score` from a `best_score = 0` start) would silently return NULL if the only registered backend scored exactly 0, currently non-triggering only because Scalar's capabilities bitmask happens to always include TERNARY_CAP_FUSION. Fixed with an explicit `have_best` flag. Everything else checked (has_avx2() dispatch, backend availability filtering, duplicate headers, dlopen/file I/O) was already loud or non-applicable. Commit `1d6eefd`. |
| 2026-08-16 | v1.21.0 | First dedicated review of docs/ and tests/ (user request, "review docs/ and tests/ next"), immediate follow-up to the scripts/+build/ session below. tests/: 10 findings fixed via code-review skill (fake pass/fail accounting, a fuzz loop that counted crashes as passes, run_tests.py counting self-skips as passed and ignoring the `required` field, a hard-fail instead of self-skip, range-only instead of exact-value assertions, a probe that conflated "not built" with "crashed", missing random seed) -- plus a real, non-cosmetic find: re-verifying the run_tests.py fix in a clean environment exposed that 3 test files were missing a `sys.path` entry needed to find their compiled module, meaning they'd been silently self-skipping and *every* prior "15/15" claim (this session's and, per CLAUDE.md's own history, prior sessions') was never a real pass for those 3 suites; fixed, now genuinely 15/15 with 0 skipped. docs/: 16 files fixed for stale paths, broken links, and factually-reversed claims (an "InvalidTritError not thrown in production" claim that's been false since 2026-08-14; an overstated VTune integration claim; a stale "canonical indexing deferred" note for a feature that shipped 2 versions ago), found via a dedicated fact-checking agent rather than the diff-oriented code-review skill. Also: a background code-review agent auto-reverted an in-progress edit to a file outside its own scope, assuming it was stray -- recovered, and noted as a lesson against running two `--fix` agents against the same working tree concurrently. Commits `a291ebb`, `bd3e6f6`, `24104b1`. |
| 2026-08-15 | v1.20.0 | First dedicated bug hunt of scripts/ and build/*.py (user request, "review the project and commit it"). 14 bugs fixed across 11 files: stale test path + unquoted shell command (build_test_packing.py), a benchmark-cleanup glob that matched zero real files (clean_all.py), a silent MSVC-instead-of-clang-cl fallback on Windows PGO (build_pgo_unified.py), missing Linux/macOS platform branching + a *.pyd-only copy glob (build_reference.py), 3 of 8 build scripts missing from the unified build_all.py entry point, discarded return values + a Linux-incompatible "python" binary check (setup_dev_environment.py), -march=native SIGILL risk + an unguarded shutil.copy2 (build_backend.py), a missing ARM/Apple-Clang OpenMP guard (build_backend.py + build_zero_skip_gemm.py), an unpropagated build failure that always reported SUCCESS (build.py), and a CI gap that let a self-skipping test suite count as passed (ci.yml). The review-orchestrating agent fork itself stalled mid-run confusing which of its own async verification sub-agents had reported back; findings were recovered directly from the sub-agents' completed transcripts (all genuinely CONFIRMED) and applied by hand. All fixes rebuilt/re-verified on Linux, tests/run_tests.py 15/15. Commit `4e72be2`. |
| 2026-08-14 | v1.19.0 | First dedicated bug hunt of src/core/ and src/engine/ (the production kernel and bindings -- user request, "check the engine"), prior sessions had covered benchmarks/research/models/opentimestamps but not this. Fixed a null-deref-on-OOM in ternary_gemm_zero_skip()'s convenience wrapper (reachable from Python) and an unsynchronized lazy-init data race in backend_avx2_v2_optimized.cpp's canonical LUT init (confirmed unreachable via any current Python entry point, fixed anyway per this doc's own no-UB principle, replaced with std::call_once). Also fixed 4 issues found in the immediately-preceding commit's new TritNet bindings (ISA-portability bug, imprecise perf claim, misleading comment, redundant buffer requests). See "2026-08-14 src/core/ + src/engine/ bug hunt" above for full details. Commits `14157d1`, `d5b792c`. |
| 2026-08-14 | v1.18.0 | Wired up Python bindings for the TritNet inference engine, closing the last "what's left" item from the Phase 3 session report: `src/engine/bindings_tritnet_inference.cpp` -> `ternary_tritnet_inference` module, batched over [N,5] uint8 trit-encoded numpy arrays, runtime has_avx2() dispatch (graceful degradation, not compile-time-only). `build/build_tritnet_inference.py` mirrors build_tritnet_gemm.py. tests/python/test_tritnet_inference_bindings.py verifies all 5 ops against the full input space plus input validation, wired into run_tests.py (15/15). |
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

**Version:** 1.33.0 · **Updated:** 2026-08-17 · **Project:** Ternary Engine · **Repository:** https://github.com/gesttaltt/ternary-engine
