# GEMM Optimization Methods Comparison

**Doc-Type:** Quick Reference · Version 1.0 · Updated 2025-12-29

---

## ML-Driven Methods (80%)

| Method | Organization | Year | Approach | Key Result | Applicability to Ternary |
|--------|--------------|------|----------|------------|--------------------------|
| **AlphaTensor** | Google DeepMind | 2022 | RL algorithm discovery | First 4×4 improvement in 50 years | HIGH - discover ternary-specific algorithms |
| **XLA Autotuning** | Google | 2021 | Multi-pass ML tuning | 5% avg, up to 2.4× | MEDIUM - tune existing kernels |
| **Ansor** | Apache TVM | 2021 | Hierarchical search + ML cost model | No templates needed | HIGH - auto-schedule ternary GEMM |
| **Meta Schedule** | Apache TVM | 2022 | Unified DSL | Extensible to tensorization | HIGH - custom ternary primitives |
| **BaCO** | Academic | 2024 | Bayesian optimization | Fast convergence | MEDIUM - compiler flag tuning |
| **LLM Optimization** | Academic | 2023 | Language model reasoning | Novel optimization strategies | LOW - research stage |
| **N-A2C / G-BFS** | Academic | 2022 | RL schedule selection | Competitive with Ansor | MEDIUM - alternative to Ansor |

---

## Traditional/Hybrid Methods (20%)

| Method | Organization | Year | Approach | Key Result | Applicability to Ternary |
|--------|--------------|------|----------|------------|--------------------------|
| **oneDNN** | Intel | 2021-24 | Expert-designed + runtime dispatch | LLM/transformer optimized | LOW - INT8 focus, not ternary |
| **cuDNN/cuBLAS** | NVIDIA | Ongoing | Vendor-optimized | De facto standard | LOW - GPU only, FP/INT focus |
| **OpenXLA** | Consortium | 2022 | Unified infrastructure | Industry standard | MEDIUM - use as compilation target |
| **bitnet.cpp** | Microsoft | 2024 | Ternary lookup tables | 1.37-6.25× on ARM/x86 | HIGH - direct competitor |

---

## Performance Comparison

| Method | Speedup Range | Hardware | Notes |
|--------|--------------|----------|-------|
| AlphaTensor | 10-20% vs Strassen | GPU/TPU | Algorithm-level improvement |
| XLA Autotuning | 5-240% | TPU | Workload dependent |
| Ansor | Comparable to cuDNN | GPU/CPU | No templates required |
| oneDNN | Baseline | Intel CPU/GPU | Reference implementation |
| bitnet.cpp | 1.37-6.25× | ARM/x86 CPU | Ternary-specific |
| **Ternary Engine** | **5-13×** | **x86 AVX2** | **Element-wise only** |

---

## Recommendation for Ternary Engine

### Immediate (1-2 weeks)
- Study bitnet.cpp kernel implementation
- Profile GEMM gap (0.37 vs 20+ Gops/s)

### Short-term (1-2 months)
- Implement Ansor tensor expression for ternary GEMM
- Let ML find optimal tile sizes and loop orders

### Long-term (3-6 months)
- Explore AlphaTensor-style discovery for ternary algorithms
- Custom ternary tensor decompositions

---

## Key Insight

> "Human semantics impose artificial constraints. AlphaTensor discovered algorithms humans never considered because it wasn't biased by 50 years of mathematical intuition."

The same applies to GEMM scheduling - ML explores configurations humans dismiss as "obviously suboptimal" but which exploit hardware quirks we don't understand.
