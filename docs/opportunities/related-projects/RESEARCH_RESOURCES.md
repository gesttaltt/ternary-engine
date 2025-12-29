# Research Resources & Paper Collections

**Doc-Type:** Ecosystem Analysis · Version 1.0 · Generated 2025-12-09

This document catalogs research papers, awesome lists, and academic resources relevant to Ternary Engine development.

---

## Table of Contents

1. [Overview](#overview)
2. [Awesome Lists](#awesome-lists)
3. [Key Papers by Topic](#key-papers-by-topic)
4. [Research Trends](#research-trends)
5. [Academic Implementations](#academic-implementations)
6. [Research Opportunities](#research-opportunities)

---

## Overview

Staying current with research helps:
- Identify new algorithms to implement
- Understand accuracy-efficiency trade-offs
- Find collaboration opportunities
- Benchmark against state-of-the-art

---

## Awesome Lists

### 1. Efficient-ML/Awesome-Model-Quantization

- **URL:** https://github.com/Efficient-ML/Awesome-Model-Quantization
- **Stars:** 3,000+
- **Coverage:** Comprehensive quantization papers and code

**Contents:**
```
Categories:
├── Overview/Survey Papers
├── Post-Training Quantization (PTQ)
│   ├── Layer-wise Quantization
│   ├── Block-wise Quantization
│   └── Data-Free Quantization
├── Quantization-Aware Training (QAT)
│   ├── Low-bit Training
│   ├── Mixed Precision
│   └── Gradient Quantization
├── Binary/Ternary Networks
│   ├── XNOR-Net variants
│   ├── TTQ variants
│   └── Recent advances
└── Hardware/Deployment
    ├── FPGA implementations
    ├── ASIC designs
    └── Edge deployment
```

**Ternary-relevant entries:**
- Trained Ternary Quantization (TTQ)
- Ternary Weight Networks
- TernaryBERT
- TRQ: Ternary Neural Networks with Residual Quantization

---

### 2. Kai-Liu001/Awesome-Model-Quantization

- **URL:** https://github.com/Kai-Liu001/Awesome-Model-Quantization
- **Stars:** 500+
- **Coverage:** 2020-2025 papers from top conferences

**Notable 2024-2025 Papers:**
```
NeurIPS 2025:
├── LoTA-QAF: Lossless Ternary Adaptation for QAT (with code!)
├── Extreme quantization advances
└── LLM quantization methods

NeurIPS 2024:
├── Advanced binary/ternary methods
├── Hardware-aware quantization
└── Training stability improvements

ECCV 2024:
├── Vision model quantization
├── Mixed precision strategies
└── Deployment optimizations
```

---

### 3. Zhen-Dong/Awesome-Quantization-Papers

- **URL:** https://github.com/Zhen-Dong/Awesome-Quantization-Papers
- **Stars:** 1,000+
- **Coverage:** Categorized by model type and application

**Categories:**
```
By Model Type:
├── CNN Quantization
├── Transformer Quantization
├── GAN Quantization
├── GNN Quantization
└── Object Detection

By Method:
├── PTQ Methods
├── QAT Methods
├── Mixed Precision
└── Extreme Quantization (binary/ternary) ← Most relevant

By Application:
├── Image Classification
├── Object Detection
├── NLP/LLM
└── Generative Models
```

---

## Key Papers by Topic

### Ternary Neural Networks

| Paper | Venue | Year | Key Contribution |
|-------|-------|------|------------------|
| **Trained Ternary Quantization** | ICLR | 2017 | Learned scales, threshold |
| **Ternary Weight Networks** | arXiv | 2016 | First practical ternary |
| **TernaryBERT** | EMNLP | 2020 | Ternary for transformers |
| **TRQ** | AAAI | 2021 | Residual quantization |
| **LoTA-QAF** | NeurIPS | 2025 | Lossless ternary adaptation |

### Seminal Papers

#### 1. Trained Ternary Quantization (TTQ)
```
@inproceedings{zhu2017trained,
  title={Trained Ternary Quantization},
  author={Zhu, Chenzhuo and Han, Song and Mao, Huizi and Dally, William J},
  booktitle={ICLR},
  year={2017}
}

Key Ideas:
- Learn separate positive/negative scales
- Threshold at 0.7 × mean(|W|)
- Straight-Through Estimator for gradients
- ~2% accuracy drop on ImageNet
```

#### 2. Ternary Weight Networks (TWN)
```
@article{li2016ternary,
  title={Ternary Weight Networks},
  author={Li, Fengfu and Zhang, Bo and Liu, Bin},
  journal={arXiv:1605.04711},
  year={2016}
}

Key Ideas:
- First to constrain weights to {-1, 0, +1}
- Approximation of full-precision weights
- 16× model compression
- Foundation for later work
```

#### 3. TernaryBERT
```
@inproceedings{zhang2020ternarybert,
  title={TernaryBERT: Distillation-aware Ultra-low Bit BERT},
  author={Zhang, Wei and Hou, Lu and Yin, Yichun and Shang, Lifeng and
          Chen, Xiao and Jiang, Xin and Liu, Qun},
  booktitle={EMNLP},
  year={2020}
}

Key Ideas:
- Ternary weights for BERT
- Knowledge distillation from teacher
- 14.9× compression with <3% accuracy drop
- Applicable to transformer architectures
```

### Binary Neural Networks (Related)

| Paper | Venue | Year | Key Contribution |
|-------|-------|------|------------------|
| **BinaryConnect** | NeurIPS | 2015 | First binary training |
| **XNOR-Net** | ECCV | 2016 | Binary with scaling |
| **Bi-Real Net** | ECCV | 2018 | Residual connections |
| **ReActNet** | ECCV | 2020 | State-of-the-art binary |

### LLM Quantization

| Paper | Venue | Year | Key Contribution |
|-------|-------|------|------------------|
| **GPTQ** | ICLR | 2023 | One-shot PTQ for LLMs |
| **AWQ** | MLSys | 2024 | Activation-aware |
| **SmoothQuant** | ICML | 2023 | Smooth activations |
| **LLM.int8()** | NeurIPS | 2022 | Outlier handling |
| **QuIP#** | arXiv | 2024 | Near-lossless 2-bit |

---

## Research Trends

### Trend 1: Extreme Quantization Revival

```
Interest in Sub-4-bit Quantization:

2020  ████                          Binary/Ternary (limited)
2021  █████                         Some progress
2022  ██████                        LLM drives interest
2023  ████████████                  GPTQ, AWQ, boom
2024  ████████████████████          2-bit LLMs emerging
2025  ████████████████████████████  Ternary for LLMs? ← Opportunity

Driver: LLM size explosion makes extreme compression necessary
```

### Trend 2: Hardware-Aware Quantization

```
Research Focus Shift:

Early:    "Can we quantize without accuracy loss?"
Now:      "How do we maximize hardware utilization?"

Key Papers:
- HAQ: Hardware-Aware Automated Quantization
- APQ: Joint Search for Network Architecture and Quantization
- Once-for-All: Train One Network for All Hardware
```

### Trend 3: Training Efficiency

```
Training Cost Reduction:

Full QAT:           100% training time
PTQ with calibration: 1-5% training time
Data-free PTQ:        0% training time

Trend: Move toward PTQ and efficient QAT
```

---

## Academic Implementations

### With Code Available

| Paper | Code | Framework | Quality |
|-------|------|-----------|---------|
| TTQ | [GitHub](https://github.com/TropComplique/trained-ternary-quantization) | PyTorch | Good |
| XNOR-Net | [GitHub](https://github.com/allenai/XNOR-Net) | Torch7 | Research |
| DoReFa-Net | [GitHub](https://github.com/tensorpack/tensorpack/tree/master/examples/DoReFa-Net) | TensorFlow | Good |
| ReActNet | [GitHub](https://github.com/liuzechun/ReActNet) | PyTorch | Good |
| Brevitas | [GitHub](https://github.com/Xilinx/brevitas) | PyTorch | Production |
| Larq | [GitHub](https://github.com/larq/larq) | TensorFlow | Production |

### Implementation Quality Guide

```
Code Quality Assessment:

★★★★★ Production:  Well-tested, documented, maintained
         Examples: Brevitas, Larq, bitsandbytes

★★★★☆ Good:       Works, some docs, occasional updates
         Examples: TTQ, ReActNet

★★★☆☆ Research:   Works for paper results, minimal docs
         Examples: Most academic code

★★☆☆☆ Legacy:     Old framework, may not run
         Examples: Torch7 implementations

★☆☆☆☆ Abandoned:  Broken dependencies, no support
         (Avoid these)
```

---

## Research Opportunities

### Gap 1: Production-Ready Ternary

```
Current State:
- TTQ, TernaryBERT exist as research code
- No production ternary library
- No HuggingFace integration
- No pip install

Opportunity:
Ternary Engine can be the "bitsandbytes for ternary"
```

### Gap 2: Ternary for Modern LLMs

```
Missing Research:
- Ternary quantization for Llama, Mistral, Qwen
- Ternary KV-cache compression
- Ternary attention weights
- Comparison with INT2/QuIP#

Opportunity:
Apply ternary to LLMs with modern techniques
```

### Gap 3: Hardware-Ternary Co-Design

```
Current Hardware:
- Binary: XNOR on FPGA (well-explored)
- INT4/INT8: Tensor Cores (well-supported)
- Ternary: No dedicated hardware

Opportunity:
- Design ternary-optimized operations
- FPGA implementations
- Custom accelerator research
```

### Gap 4: TritNet Innovation

```
Ternary Engine's Unique Direction:
- Replace LUTs with learned neural operations
- No existing research in this direction
- Could discover novel ternary operations

Research Questions:
1. Can neural networks learn ternary arithmetic?
2. What novel operations emerge?
3. Does this enable better accuracy?
```

---

## Recommended Reading Order

### For Understanding Ternary Networks

1. **Start:** Ternary Weight Networks (TWN) - Foundation
2. **Then:** Trained Ternary Quantization (TTQ) - Training
3. **Next:** TernaryBERT - Modern applications
4. **Finally:** LoTA-QAF - Latest advances

### For Implementation

1. **Start:** Brevitas documentation - QAT patterns
2. **Then:** bitsandbytes code - Integration patterns
3. **Next:** llama.cpp GGML - Kernel optimization
4. **Finally:** XNOR-Net - Low-bit tricks

### For Innovation

1. **Read:** Survey papers on quantization
2. **Study:** QuIP# and 2-bit methods
3. **Explore:** Hardware-aware quantization
4. **Innovate:** TritNet direction

---

## Conferences to Follow

| Conference | Focus | Submission | Website |
|------------|-------|------------|---------|
| NeurIPS | ML general | May | neurips.cc |
| ICML | ML general | Jan | icml.cc |
| ICLR | Representations | Sep | iclr.cc |
| CVPR | Computer vision | Nov | cvpr.org |
| ECCV | Computer vision | Mar | eccv.ecva.net |
| EMNLP | NLP | May | emnlp.org |
| MLSys | ML systems | Oct | mlsys.org |

---

## Action Items

### Research Monitoring

1. [ ] Set up Google Scholar alerts for "ternary neural network"
2. [ ] Monitor Awesome lists monthly
3. [ ] Track NeurIPS/ICML proceedings for quantization papers

### Paper Implementation

1. [ ] Implement TTQ training in Ternary Engine PyTorch layer
2. [ ] Replicate TernaryBERT results on small model
3. [ ] Compare accuracy against recent 2-bit methods

### Novel Research

1. [ ] Document TritNet hypothesis and approach
2. [ ] Design experiments for learned ternary operations
3. [ ] Write technical report on findings

---

## Bibliography Template

For tracking papers as you read them:

```markdown
## [Paper Title]

**Citation:** Authors, Venue, Year
**Link:** [Paper](url) | [Code](url)
**Read:** YYYY-MM-DD
**Relevance:** ★★★★★

### Summary
(2-3 sentences on what the paper does)

### Key Ideas
- Idea 1
- Idea 2
- Idea 3

### Applicable to Ternary Engine
- [ ] Training technique
- [ ] Inference optimization
- [ ] Architecture design
- [ ] Evaluation method

### Notes
(Your thoughts, questions, ideas)
```

---

**Document Version:** 1.0
**Generated:** 2025-12-09
**Author:** Claude Code Analysis
