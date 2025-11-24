# Claude Code Configuration - Ternary Neural Network Engine

**Doc-Type:** Project-Level Configuration · Version 1.0 · Updated 2025-11-23 · Author Ternary Engine Team

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
| 2025-11-23 | v1.0.0  | Initial .claude configuration for Ternary Engine project |

---

**Remember:**
- YAGNI: Only proven optimizations
- Benchmark everything before claiming performance gains
- Validate on Windows x64 before production claims
- Document with validation dates and platforms
- TritNet is the future: memory-bound → compute-bound

---

**Version:** 1.0.0 · **Updated:** 2025-11-23 · **Project:** Ternary Engine · **Repository:** https://github.com/gesttaltt/ternary-engine
