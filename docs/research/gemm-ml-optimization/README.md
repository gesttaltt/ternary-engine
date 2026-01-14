# GEMM ML-Driven Optimization Research

**Doc-Type:** Research Survey · Version 1.0 · Updated 2025-12-29 · Author Ternary Engine Team

Survey of machine learning approaches to GEMM (General Matrix-Matrix Multiplication) optimization, covering Google and Intel solutions from January 2021 to December 2025.

---

## Executive Summary

Traditional GEMM optimization relies on human-designed heuristics that fail to capture hardware complexity. Recent ML-driven approaches have demonstrated that:

1. **RL can discover novel algorithms** - AlphaTensor found matrix multiplication algorithms faster than 50-year-old human discoveries
2. **Autotuning outperforms expert heuristics** - 5-15% average speedup, up to 2.4x on specific workloads
3. **Learned cost models beat analytical models** - More accurate prediction of kernel performance

**Key insight**: Human semantics impose artificial constraints on the optimization space. ML explores configurations humans wouldn't consider.

---

## ML-Driven Methods (80%)

### 1. AlphaTensor (Google DeepMind, 2022)

**Paper**: "Discovering faster matrix multiplication algorithms with reinforcement learning" - Nature 610, 2022

**Approach**: Deep reinforcement learning based on AlphaZero to discover tensor decompositions

**Key Innovation**:
- Frames algorithm discovery as a single-player game
- Board state = 3D tensor representing distance from correct algorithm
- Moves = algorithm instructions that modify tensor
- Goal = zero out all tensor entries

**Results**:
| Matrix Size | Human Best | AlphaTensor | Improvement |
|-------------|------------|-------------|-------------|
| 4×4 | 49 (Strassen²) | 47 | First improvement in 50 years |
| 4×5 × 5×5 | 80 | 76 | 5% fewer multiplications |
| Various | Strassen | Novel | 10-20% faster on GPU/TPU |

**Hardware-Specific Discovery**:
- Adapted to find algorithms optimized for specific hardware
- Nvidia V100 GPU: 10-20% faster than standard algorithms
- Google TPU v2: Similar improvements
- Generated 14,236 nonequivalent algorithms for 4×4 case

**Source**: [DeepMind Blog](https://deepmind.google/discover/blog/discovering-novel-algorithms-with-alphatensor/) | [GitHub](https://github.com/google-deepmind/alphatensor) | [Nature Paper](https://www.nature.com/articles/s41586-022-05172-4)

---

### 2. XLA Autotuning (Google, 2021)

**Paper**: "A Flexible Approach to Autotuning Multi-Pass Machine Learning Compilers" - PACT 2021

**Approach**: Multi-pass autotuning with ML cost models for tile sizes, fusion, and layout optimization

**Key Innovation**:
- Tunes multiple compiler passes jointly
- ML-based cost model predicts execution time
- Explores tile sizes, fusion decisions, memory layouts, compiler flags

**Results on 150 Google Production ML Models**:
| Model | Speedup |
|-------|---------|
| AVSpeech Inference | 2.4× |
| Translate Transformer | 1.5× |
| MLPerf DLRM Training | 14% |
| MLPerf Mask RCNN | 13% |
| MLPerf SSD | 11% |
| Average | 5% |
| 9 models | >15% |

**Components Contributing to Speedup**:
1. Tile size autotuning (largest contribution)
2. Fusion autotuning
3. Layout optimization
4. Compiler flags

**Source**: [PACT 2021 Paper](https://mangpo.net/papers/xla-autotuning-pact2021.pdf) | [OpenXLA](https://openxla.org/xla)

---

### 3. TVM Ansor Auto-Scheduler (Apache/UC Berkeley/Alibaba, 2021)

**Paper**: "Ansor: Generating High-Performance Tensor Programs for Deep Learning" - OSDI 2020, integrated 2021

**Approach**: Hierarchical search with sketch generation + annotation tuning

**Key Innovation**:
- No manual templates required (unlike AutoTVM)
- Sketch generation captures high-level structure
- Annotation tuning fills in details
- Evolutionary search + learned cost model

**Advantages over AutoTVM**:
1. **No expert templates** - Rules encoded programmatically
2. **Larger search space** - Hierarchical decomposition
3. **Faster search** - Evolutionary algorithm + ML cost model

**Results**:
- Evaluated on convolution, GEMM, group conv, dilated conv, depthwise conv
- Competitive with vendor libraries (cuDNN, cuBLAS)
- Reduces tuning time vs AutoTVM

**Source**: [TVM Blog](https://tvm.apache.org/2021/03/03/intro-auto-scheduler) | [Ansor Paper](https://arxiv.org/pdf/2006.06762)

---

### 4. Meta Schedule (Apache TVM, 2022)

**Approach**: Unifies AutoTVM and Ansor approaches with extensible DSL

**Key Innovation**:
- 3rd generation TVM auto-scheduling
- Supports tensorization, loop partitioning
- Extensible to new hardware primitives

**Source**: [TVM RFC](https://github.com/apache/tvm-rfcs/blob/main/rfcs/0005-meta-schedule-autotensorir.md)

---

### 5. BaCO - Bayesian Compiler Optimization (2024)

**Paper**: "BaCO: A Fast and Portable Bayesian Compiler Optimization Framework" - ASPLOS 2024

**Approach**: Bayesian optimization for compiler autotuning

**Key Innovation**:
- Faster convergence than random/grid search
- Portable across different compilers
- Handles high-dimensional discrete spaces

---

### 6. LLM-Based Compiler Optimization (2023)

**Paper**: "Large Language Models for Compiler Optimization" - CC 2023 (Chris Cummins et al.)

**Approach**: Use LLMs to predict optimal compiler flags/transforms

**Key Innovation**:
- Leverages code understanding from pretraining
- Can reason about optimization strategies
- Generates optimization suggestions in natural language

---

### 7. Reinforcement Learning for GEMM (2022-2023)

**Papers**: Various works on RL-based GEMM optimization

**Approaches**:
1. **G-BFS** (Greedy Best First Search) - Heuristic search on TVM
2. **N-A2C** (Neighborhood Actor Advantage Critic) - RL for schedule selection

**Results**: Competitive with Ansor on standard benchmarks

---

### 8. MLIR PEAK (2023)

**Paper**: "PEAK: Generating High-Performance Schedules in MLIR"

**Results vs TVM/Ansor**:
- Higher performance for matrix-vector products
- Comparable for matrix-matrix multiplication
- Lower for convolutions

**Source**: [Springer](https://link.springer.com/chapter/10.1007/978-3-032-02436-7_13)

---

## Traditional/Hybrid Methods (20%)

### 1. Intel oneDNN (2021-2024)

**Approach**: Expert-designed kernels with runtime dispatch

**Key Features**:
- Hand-optimized GEMM for Intel CPUs/GPUs
- Automatic memory format selection
- Primitive fusion (Conv+ReLU, etc.)
- INT8/BF16/FP16 quantization

**2024.1 Release Improvements**:
- Improved MATMUL for LLM/transformer workloads
- Better Intel Xeon support
- Memory efficiency optimizations

**Source**: [Intel oneDNN](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onednn.html) | [GitHub](https://github.com/oneapi-src/oneDNN)

---

### 2. cuDNN/cuBLAS (NVIDIA)

**Approach**: Vendor-optimized libraries with heuristic kernel selection

**Relationship to ML methods**:
- XLA/TVM often call cuDNN/cuBLAS for complex ops
- ML autotuning selects among cuDNN algorithms
- Fusion decisions happen at compiler level

---

### 3. OpenXLA Consortium (2022)

**Members**: Google, Alibaba, AWS, AMD, Apple, Arm, Cerebras, Graphcore, Hugging Face, Intel, Meta, NVIDIA, SiFive

**Approach**: Unified ML compiler infrastructure

**Key Components**:
- XLA compiler
- StableHLO portable operator set
- Shared autotuning infrastructure

**Source**: [OpenXLA](https://openxla.org/)

---

## Key Insights for Ternary Engine

### Why Human Heuristics Fail

1. **Combinatorial explosion** - Search space too large for manual exploration
2. **Hardware complexity** - Cache hierarchies, SIMD widths, memory bandwidth interactions
3. **Workload diversity** - Optimal strategy varies by matrix size, sparsity, hardware
4. **Local optima** - Human intuition gets stuck in familiar patterns

### What ML Discovers

1. **Non-intuitive decompositions** - AlphaTensor's 47-step 4×4 algorithm
2. **Hardware-specific tricks** - Tile sizes that exploit cache geometry
3. **Unexpected fusions** - Combining ops humans wouldn't consider
4. **Adaptive strategies** - Different algorithms for different input sizes

### Application to Ternary GEMM

**Opportunity**: Apply AlphaTensor/Ansor approaches to ternary matrix multiplication

1. **AlphaTensor-style**: Discover novel ternary matmul algorithms
   - Search space: ternary tensor decompositions
   - Reward: minimize operation count while preserving correctness
   - Constraint: operations must use {-1, 0, +1} arithmetic

2. **Ansor-style**: Auto-schedule ternary GEMM kernels
   - Define tensor expression for ternary matmul
   - Let Ansor find optimal tile sizes, loop orders, parallelization
   - Train cost model on ternary-specific hardware characteristics

3. **Hybrid approach**:
   - Use ML to discover algorithm structure
   - Use traditional optimization for low-level SIMD

---

## Timeline of Key Developments

| Date | Development | Source |
|------|-------------|--------|
| 2021-03 | TVM Ansor integrated | Apache TVM |
| 2021-10 | XLA autotuning PACT paper | Google |
| 2022-03 | OpenXLA consortium formed | Industry |
| 2022-10 | AlphaTensor published in Nature | DeepMind |
| 2022-11 | Meta Schedule RFC | Apache TVM |
| 2023-02 | LLM compiler optimization | CC 2023 |
| 2023-10 | MLIR PEAK published | Academic |
| 2024-01 | BaCO at ASPLOS | Academic |
| 2024-03 | oneDNN 2024.1 with LLM focus | Intel |
| 2024-10 | bitnet.cpp with ternary kernels | Microsoft |

---

## References

### Primary Sources

1. [AlphaTensor - Nature](https://www.nature.com/articles/s41586-022-05172-4)
2. [XLA Autotuning - PACT 2021](https://mangpo.net/papers/xla-autotuning-pact2021.pdf)
3. [Ansor Paper](https://arxiv.org/pdf/2006.06762)
4. [Intel oneDNN](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onednn.html)
5. [OpenXLA Project](https://openxla.org/xla)

### Curated Resource Lists

1. [ML in Compilers - GitHub](https://github.com/zwang4/awesome-machine-learning-in-compilers)
2. [AI in Compiler Optimization - GitHub](https://github.com/shrutisaxena51/Artificial-Intelligence-in-Compiler-Optimization)

### DeepMind AlphaTensor

1. [Blog Post](https://deepmind.google/discover/blog/discovering-novel-algorithms-with-alphatensor/)
2. [GitHub Repository](https://github.com/google-deepmind/alphatensor)
3. [IEEE Spectrum Coverage](https://spectrum.ieee.org/matrix-multiplication-deepmind)

---

## Next Steps for Ternary Engine

1. **Evaluate Ansor on ternary ops** - Define ternary GEMM in TVM tensor expression
2. **Study AlphaTensor codebase** - Understand RL approach for algorithm discovery
3. **Profile current GEMM gap** - Identify where we lose to bitnet.cpp
4. **Prototype ML-guided tuning** - Start with tile size selection

---

**Version**: 1.0 · **Updated**: 2025-12-29 · **Scope**: GEMM ML Optimization 2021-2025
