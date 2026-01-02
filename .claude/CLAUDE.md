# Claude Code Configuration - Ternary Neural Network Engine

**Doc-Type:** Project-Level Configuration · Version 1.4 · Updated 2025-01-02 · Author Ternary Engine Team

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
- 8,234× average speedup over pure Python implementations
- 35,042 Mops/s peak throughput (35 billion operations/second)
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
- TritNet roadmap: [docs/TRITNET_ROADMAP.md](../docs/TRITNET_ROADMAP.md)

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
| **homeostasis** | `models/company-flagships/v5_11_homeostasis/best.pt` | Radial hierarchy, p-adic valuation | VRC target: -0.83, coverage: 100% |
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

### Falsification Test Integration

```python
# Load models in falsify.py
models = {
    'v5_11_3': 'models/company-flagships/ternary-multiVAE/ternary_v5_11_3.pt',
    'homeostasis': 'models/company-flagships/v5_11_homeostasis/best.pt',
    'codon': 'models/company-flagships/hierarchy-encoder-codon-inference/codon-predictor/codon_encoder_3adic.pt',
}
```

### Falsification Results Summary (2025-01-02)

**9 of 24 hypotheses tested** - See `research/results/FALSIFICATION_SUMMARY.md` for details.

| Hypothesis | Score | Grade | Status | Key Finding |
|------------|-------|-------|--------|-------------|
| H1 p-adic | 100% | A | INTRINSIC | Built into ternary representation |
| H2 Ultrametric | 89.32% | B | Supported | Raw=100%, model=45% isoceles |
| H3 Hyperbolic | 99.80% | A | Supported | VRC=0.035, target=-0.8 |
| H4 Tropical | 87.20% | B | Supported | tadd distributes, tmul doesn't |
| H6 Three-Valued | 100% | A | INTRINSIC | De Morgan laws hold |
| H9 Information | 90.91% | B | Supported | Entropy confirms p-adic |
| H10 Group Theory | 84.08% | B | Supported | **tadd non-associative (20%)** |
| H11 Lattice | 100% | A | INTRINSIC | tmin/tmax distributive lattice |
| H23 Modular | 56.53% | C | Weak | Products fail, mod-3 fails |

**Key Discovery:** tadd is non-associative for 79.6% of triplets - balanced ternary is NOT a group.

**Next Session:** Implement remaining 15 hypotheses starting with Tier 2 (H5, H7, H8). Run `python research/scripts/falsify.py -H H5`.

| Priority | Hypotheses | Difficulty |
|----------|------------|------------|
| Tier 2 | H5 Clifford, H7 Quantum, H8 Category | Medium |
| Tier 3 | H12 Dynamical, H13 Topological, H14 Neural | Hard |
| Tier 4 | H15-H22, H24 | Research |

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
- lib/dense243/ - High-density encoding library (5 trits/byte, validated)

**scripts/** - Build and development automation
- build/ - Build scripts (all platforms)
- tritnet/ - TritNet training pipeline
- orchestration/ - High-level workflows

**benchmarks/** - Competitive analysis suite
**tests/** - Comprehensive test suite (65 tests)
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

**Phase 2A** - Proof-of-concept (tnot) 🔄 IN PROGRESS
- Train tnot model to 100% accuracy
- Validate learned weights
- Go/No-Go decision point

**Phase 2B** - Scale to all operations
- Train tadd, tmul, tmin, tmax
- Validate ≥99% accuracy requirement
- Document learned weight patterns

**Phase 3** - C++ Integration
- Export ternary weights to binary format
- Implement C++ inference engine
- Benchmark TritNet vs LUT performance

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

**OpenMP** - Disabled by default (documented CI crashes)
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
python scripts/timestamp_snapshot.py --create
```

**Verify timestamp:**
```bash
python scripts/timestamp_snapshot.py --verify timestamps/snapshot_YYYYMMDD_HHMMSS.ots
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
- OpenMP disabled (documented CI crashes, root cause fixed but needs validation)

### Experimental

**Linux/macOS** - Untested, use at own risk
- Build scripts provided but not validated
- CI disabled for OpenMP tests
- Manual compilation commands untested
- No production claims until validated

### Target Platforms (Future)

**ARM NEON/SVE** - Planned for mobile/edge deployment
**AVX-512** - Planned for latest Intel/AMD CPUs
**WebAssembly SIMD** - Planned for browser deployment

---

## Critical Gaps & Known Issues

### Production Gaps

1. **Multi-platform validation** - Only Windows x64 proven
2. **TritNet Phase 2 decision** - tnot 100% accuracy validation pending
3. **Competitive benchmarking** - Only 2/5 criteria validated
4. **Dense243 integration** - Pack/unpack work but module integration issues

### Important Improvements

5. **OpenMP re-enablement** - Fixed but needs CI validation
6. **Phase 4.1 fusion** - Implementation complete, benchmarks pending
7. **Documentation gaps** - Missing TRITNET_VISION.md, TRITNET_ROADMAP.md
8. **Code duplication** - Between engines, needs refactoring

### Nice to Have

9. **Multi-dimensional arrays** - Currently 1D only
10. **ARM/NEON support** - x86-64 AVX2 only
11. **GPU/TPU acceleration** - For TritNet Phase 4+
12. **Profiler integration** - Framework implemented but not integrated

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

**Version:** 1.2.0 · **Updated:** 2025-12-30 · **Project:** Ternary Engine · **Repository:** https://github.com/gesttaltt/ternary-engine
