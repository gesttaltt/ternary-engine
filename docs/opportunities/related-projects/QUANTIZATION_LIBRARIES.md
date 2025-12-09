# Quantization Libraries Analysis

**Doc-Type:** Ecosystem Analysis · Version 1.0 · Generated 2025-12-09

This document provides detailed analysis of mainstream quantization libraries that represent both competition and learning opportunities for Ternary Engine.

---

## Table of Contents

1. [Overview](#overview)
2. [bitsandbytes](#1-bitsandbytes)
3. [AutoGPTQ](#2-autogptq)
4. [GPTQModel](#3-gptqmodel)
5. [PyTorch AO](#4-pytorch-ao)
6. [Intel Neural Compressor](#5-intel-neural-compressor)
7. [Brevitas](#6-brevitas)
8. [Comparative Analysis](#comparative-analysis)
9. [Lessons for Ternary Engine](#lessons-for-ternary-engine)

---

## Overview

These libraries dominate the 4-bit and 8-bit quantization space. Understanding their architecture, APIs, and limitations helps position Ternary Engine in the market.

| Library | Primary Focus | Bit Widths | Framework | Maintainer |
|---------|---------------|------------|-----------|------------|
| bitsandbytes | LLM quantization | 4, 8 | PyTorch | Community |
| AutoGPTQ | GPTQ algorithm | 2, 3, 4, 8 | PyTorch | Community |
| GPTQModel | Production GPTQ | 4, 8 | PyTorch | ModelCloud |
| PyTorch AO | Native quantization | 4, 8 | PyTorch | Meta |
| Neural Compressor | Multi-framework | 4, 8, FP8 | PT/TF/ONNX | Intel |
| Brevitas | QAT training | Any | PyTorch | AMD/Xilinx |

---

## 1. bitsandbytes

### Repository Information

- **URL:** https://github.com/bitsandbytes-foundation/bitsandbytes
- **Stars:** ~6,000
- **Language:** Python, CUDA
- **License:** MIT
- **Last Updated:** Actively maintained (2025)

### What It Does

bitsandbytes provides accessible large language model quantization via k-bit compression for PyTorch. It's the de facto standard for quick LLM quantization.

### Key Features

```python
# 8-bit inference (LLM.int8())
from transformers import AutoModelForCausalLM
import bitsandbytes as bnb

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    load_in_8bit=True,
    device_map="auto"
)

# 4-bit inference (QLoRA/NF4)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # Normal Float 4
    bnb_4bit_compute_dtype=torch.float16
)

# 8-bit optimizers for training
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-4)
```

### Architecture

```
bitsandbytes Architecture:

┌─────────────────────────────────────────────────────┐
│                    Python API                        │
│  (load_in_8bit, load_in_4bit, optimizers)           │
└─────────────────────────┬───────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────┐
│                 Quantization Core                    │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │ LLM.int8()   │  │ QLoRA/NF4    │  │ 8-bit     │  │
│  │ (inference)  │  │ (training)   │  │ Optimizers│  │
│  └──────────────┘  └──────────────┘  └───────────┘  │
└─────────────────────────┬───────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────┐
│                   CUDA Kernels                       │
│  (Custom int8/int4 matrix multiplication)           │
└─────────────────────────────────────────────────────┘
```

### Why It Matters for Ternary Engine

**Learn from:**
- Simple HuggingFace integration (`load_in_Xbit=True`)
- Transparent quantization (users don't manage low-level details)
- CUDA kernel organization

**Compete with:**
- Ternary offers 8× compression vs bitsandbytes' 4×
- Memory savings are the key differentiator

**Gap:**
- bitsandbytes is GPU-only; Ternary Engine has CPU SIMD advantage

### Code Patterns to Study

```python
# bitsandbytes Linear layer replacement
class Linear8bitLt(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super().__init__()
        self.weight = Int8Params(...)  # Custom parameter class
        self.bias = nn.Parameter(...) if bias else None

    def forward(self, x):
        # Dequantize -> matmul -> requantize
        return bnb.matmul(x, self.weight.data, ...)
```

---

## 2. AutoGPTQ

### Repository Information

- **URL:** https://github.com/AutoGPTQ/AutoGPTQ
- **Stars:** ~4,000
- **Language:** Python, CUDA, Triton
- **License:** MIT
- **Last Updated:** Actively maintained

### What It Does

AutoGPTQ implements the GPTQ algorithm for post-training quantization, enabling 2/3/4/8-bit weight-only quantization with calibration.

### Key Features

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Define quantization config
quantize_config = BaseQuantizeConfig(
    bits=4,                    # 2, 3, 4, or 8
    group_size=128,            # Quantization group size
    desc_act=True,             # Activation order
    damp_percent=0.1           # Dampening for Hessian
)

# Load and quantize
model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantize_config
)

# Calibrate with sample data
model.quantize(calibration_data)

# Save quantized model
model.save_quantized("llama-2-7b-gptq-4bit")

# Load for inference
quantized_model = AutoGPTQForCausalLM.from_quantized(
    "llama-2-7b-gptq-4bit",
    device="cuda:0"
)
```

### GPTQ Algorithm

```
GPTQ Quantization Process:

1. Collect Calibration Data
   ┌─────────────────────────┐
   │ Run ~128-1024 samples   │
   │ through the model       │
   └───────────┬─────────────┘
               │
2. Compute Hessian for Each Layer
   ┌───────────▼─────────────┐
   │ H = 2 * X^T * X         │
   │ (activation covariance) │
   └───────────┬─────────────┘
               │
3. Quantize Weights Column by Column
   ┌───────────▼─────────────┐
   │ For each column:        │
   │   q = quantize(w)       │
   │   error = w - q         │
   │   Propagate error to    │
   │   remaining columns     │
   └───────────┬─────────────┘
               │
4. Output Quantized Weights
   ┌───────────▼─────────────┐
   │ INT4 weights + scales   │
   │ + zero points           │
   └─────────────────────────┘
```

### Why It Matters for Ternary Engine

**Learn from:**
- Calibration-based quantization approach
- Group-wise quantization for accuracy
- HuggingFace model integration patterns

**Potential application:**
- GPTQ-style calibration could improve ternary accuracy
- Group-wise ternary quantization

**Key insight:**
- GPTQ's 2-bit mode exists but rarely used due to accuracy loss
- Ternary (1.58-bit) could fill the gap between 2-bit and 1-bit

---

## 3. GPTQModel

### Repository Information

- **URL:** https://github.com/ModelCloud/GPTQModel
- **Stars:** ~1,000
- **Language:** Python, CUDA
- **License:** Apache 2.0
- **Last Updated:** Actively maintained

### What It Does

GPTQModel is a production-ready fork/evolution of AutoGPTQ with additional features and hardware support.

### Key Features

```python
from gptqmodel import GPTQModel, QuantizeConfig

# Supports multiple quantization methods
config = QuantizeConfig(
    bits=4,
    group_size=128,
    format="gptq",  # or "awq", "qqq", "marlin"
)

# Multi-backend support
model = GPTQModel.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantize_config=config
)

# Hardware-specific optimizations
# - NVIDIA CUDA
# - AMD ROCm
# - Intel XPU
# - Apple Metal (experimental)
```

### Why It Matters for Ternary Engine

**Learn from:**
- Multi-hardware backend architecture
- Production-ready error handling
- Multiple quantization format support

**Pattern to adopt:**
```python
# GPTQModel's backend abstraction
class TernaryBackend:
    @staticmethod
    def get_backend(device_type):
        if device_type == "cuda":
            return TernaryCUDABackend()
        elif device_type == "cpu":
            return TernarySIMDBackend()  # Our AVX2 implementation
        elif device_type == "metal":
            return TernaryMetalBackend()
```

---

## 4. PyTorch AO

### Repository Information

- **URL:** https://github.com/pytorch/ao
- **Stars:** ~2,000
- **Language:** Python, C++, CUDA
- **License:** BSD
- **Last Updated:** Very active (Meta-maintained)

### What It Does

PyTorch AO (Architecture Optimization) is PyTorch's official library for quantization and sparsity, providing native integration with the PyTorch ecosystem.

### Key Features

```python
import torch
from torchao.quantization import quantize_, int4_weight_only

# Simple API for weight-only quantization
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
quantize_(model, int4_weight_only())

# More control with config
from torchao.quantization import Int4WeightOnlyConfig

config = Int4WeightOnlyConfig(
    group_size=128,
    inner_k_tiles=8
)
quantize_(model, config)

# QAT support
from torchao.quantization import int8_dynamic_activation_int4_weight
quantize_(model, int8_dynamic_activation_int4_weight())
```

### Architecture

```
PyTorch AO Architecture:

┌─────────────────────────────────────────────────────────────┐
│                      torchao API                             │
│  quantize_(), sparsify_(), autoquant()                      │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                   Quantization Primitives                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ AffineQuant │  │ Tensor      │  │ Layout              │  │
│  │ izedTensor  │  │ Subclasses  │  │ (PackedWeight, etc) │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                   Backend Kernels                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ CUDA        │  │ CPU (AVX)   │  │ torch.compile       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Why It Matters for Ternary Engine

**Critical learning opportunity:**
- This is PyTorch's official approach - follow their patterns
- `quantize_()` API is the target interface style
- Tensor subclassing is the modern PyTorch way

**Target integration:**
```python
# Goal: Ternary Engine integration with torchao
from torchao.quantization import quantize_
from ternary_engine.torchao import ternary_weight_only

model = load_model()
quantize_(model, ternary_weight_only())  # Uses our SIMD kernels
```

---

## 5. Intel Neural Compressor

### Repository Information

- **URL:** https://github.com/intel/neural-compressor
- **Stars:** ~2,000
- **Language:** Python
- **License:** Apache 2.0
- **Last Updated:** Actively maintained

### What It Does

Intel Neural Compressor provides comprehensive model compression for PyTorch, TensorFlow, and ONNX, optimized for Intel hardware but works on any x86.

### Key Features

```python
from neural_compressor import quantization

# Post-training quantization
config = quantization.PostTrainingQuantConfig(
    approach="static",  # or "dynamic", "weight_only"
    backend="ipex",     # Intel Extension for PyTorch
    op_type_dict={
        "Linear": {
            "weight": {"dtype": "int4", "algorithm": "RTN"},
            "activation": {"dtype": "fp32"}
        }
    }
)

q_model = quantization.fit(model, config, calib_dataloader=calib_loader)

# Weight-only quantization for LLMs
from neural_compressor import WeightOnlyQuantConfig

woq_config = WeightOnlyQuantConfig(
    weight_dtype="int4",
    weight_algorithm="GPTQ",  # or "AWQ", "RTN", "AutoRound"
    group_size=128
)
```

### Supported Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| RTN | Round-to-Nearest | Fast, moderate accuracy |
| GPTQ | Gradient-based | High accuracy |
| AWQ | Activation-aware | Balanced |
| AutoRound | Automatic tuning | Best accuracy |
| SmoothQuant | Activation smoothing | INT8 |

### Why It Matters for Ternary Engine

**Learn from:**
- Multi-framework abstraction layer
- Algorithm-agnostic configuration
- Intel CPU optimization patterns (relevant for our AVX2)

**Potential integration:**
```python
# Neural Compressor could support ternary as a weight_dtype
config = WeightOnlyQuantConfig(
    weight_dtype="ternary",  # New dtype
    weight_algorithm="TernaryRound",  # New algorithm
)
```

---

## 6. Brevitas

### Repository Information

- **URL:** https://github.com/Xilinx/brevitas
- **Stars:** ~1,000
- **Language:** Python
- **License:** BSD
- **Last Updated:** Actively maintained

### What It Does

Brevitas is AMD/Xilinx's library for quantization-aware training (QAT), supporting arbitrary bit-widths including ternary.

### Key Features

```python
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat

# Quantized layers with configurable bit-width
class QuantizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = qnn.QuantConv2d(
            3, 64, 3,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None
        )
        self.relu1 = qnn.QuantReLU(act_quant=Int8ActPerTensorFloat)
        self.fc1 = qnn.QuantLinear(
            64 * 7 * 7, 10,
            weight_quant=Int8WeightPerTensorFloat
        )

# Custom ternary quantizer
from brevitas.quant.base import NarrowIntQuant

class TernaryWeightQuant(NarrowIntQuant):
    bit_width = 2  # Represents {-1, 0, +1}
    narrow_range = True
    signed = True

class TernaryLinear(qnn.QuantLinear):
    def __init__(self, in_features, out_features):
        super().__init__(
            in_features, out_features,
            weight_quant=TernaryWeightQuant,
            bias=False
        )
```

### QAT Training Flow

```
Brevitas QAT Training:

┌─────────────────────────────────────────────────────────────┐
│                    Training Loop                             │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    Forward Pass                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 1. Quantize weights: W_q = quantize(W_fp)           │    │
│  │ 2. Forward with quantized: y = W_q @ x              │    │
│  │ 3. Quantize activations if configured               │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    Backward Pass (STE)                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Straight-Through Estimator:                          │    │
│  │ grad_W_fp = grad_W_q  (pass gradient through)       │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    Update FP Weights                         │
│  W_fp = W_fp - lr * grad_W_fp                               │
└─────────────────────────────────────────────────────────────┘
```

### Why It Matters for Ternary Engine

**Most directly relevant** for training ternary models:
- Already supports arbitrary bit-widths
- Clean quantizer abstraction
- Proven QAT implementation

**Pattern to adopt:**
```python
# Ternary Engine could provide Brevitas-compatible quantizers
from ternary_engine.brevitas import TernaryWeightQuant, TernaryActQuant

model = qnn.QuantLinear(
    in_features, out_features,
    weight_quant=TernaryWeightQuant,  # From Ternary Engine
)
# Train with QAT, then export to Ternary Engine format for inference
```

---

## Comparative Analysis

### Feature Comparison

| Feature | bits&bytes | AutoGPTQ | GPTQModel | PyTorch AO | Neural Comp | Brevitas |
|---------|------------|----------|-----------|------------|-------------|----------|
| PTQ | Yes | Yes | Yes | Yes | Yes | Limited |
| QAT | No | No | No | Yes | Yes | **Yes** |
| INT8 | Yes | Yes | Yes | Yes | Yes | Yes |
| INT4 | Yes | Yes | Yes | Yes | Yes | Yes |
| INT2 | No | Yes | Yes | No | No | Yes |
| **Ternary** | No | No | No | No | No | **Possible** |
| HF Integration | **Best** | Good | Good | Emerging | Good | Manual |
| GPU Required | Yes | Yes | Yes | Optional | Optional | Optional |
| CPU SIMD | No | No | No | Some | **Yes** | No |

### Performance Comparison

| Library | Quantization Speed | Inference Speed | Memory Use |
|---------|-------------------|-----------------|------------|
| bitsandbytes | Fast (no calib) | Good | Good |
| AutoGPTQ | Slow (calibration) | **Best** | **Best** |
| GPTQModel | Medium | Very Good | Very Good |
| PyTorch AO | Fast | Good | Good |
| Neural Compressor | Slow | Good (Intel) | Good |
| Brevitas | Training time | Depends | Depends |

### Accuracy Comparison (Llama-2-7B, Perplexity)

| Method | Bit Width | WikiText-2 PPL |
|--------|-----------|----------------|
| FP16 | 16 | 5.47 |
| bitsandbytes NF4 | 4 | 5.78 |
| GPTQ | 4 | 5.63 |
| AWQ | 4 | 5.60 |
| GPTQ | 2 | ~8-10 |
| **Ternary (target)** | 1.58 | **TBD** |

---

## Lessons for Ternary Engine

### API Design Lessons

**From bitsandbytes:**
```python
# Simple, HuggingFace-native API
model = AutoModel.from_pretrained(
    "model-name",
    load_in_ternary=True,  # <-- Our goal
    device_map="auto"
)
```

**From PyTorch AO:**
```python
# Modern PyTorch patterns
from ternary_engine import ternary_weight_only
quantize_(model, ternary_weight_only())
```

**From Brevitas:**
```python
# Composable quantizers
class TernaryLinear(qnn.QuantLinear):
    weight_quant = TernaryWeightQuant
```

### Architecture Lessons

1. **Separate concerns:**
   - Quantization algorithm (how to convert weights)
   - Storage format (how to pack bits)
   - Compute kernel (how to do math)
   - Framework integration (how to use in PyTorch)

2. **Backend abstraction:**
   - Support multiple hardware (CUDA, CPU SIMD, etc.)
   - Automatic backend selection

3. **Calibration infrastructure:**
   - Data collection utilities
   - Threshold computation
   - Per-channel/per-group options

### Technical Lessons

1. **Memory layout matters:**
   - Pack ternary values efficiently (current: 2 bits each)
   - Consider Dense243 (5 trits/byte) for storage

2. **Kernel optimization:**
   - Study llama.cpp GGML for SIMD patterns
   - Study TensorRT-LLM for CUDA patterns

3. **Gradient handling:**
   - STE is standard for training
   - Consider learned gradients (TritNet direction)

---

## Action Items for Ternary Engine

### Immediate (Phase 1-2)

1. [ ] Study bitsandbytes HuggingFace integration code
2. [ ] Implement `TernaryLinear` following Brevitas patterns
3. [ ] Add `load_in_ternary=True` to transformers integration
4. [ ] Create calibration utilities similar to GPTQ

### Medium-term (Phase 3)

1. [ ] Implement torchao-compatible `ternary_weight_only()`
2. [ ] Add group-wise ternary quantization
3. [ ] Benchmark against bitsandbytes INT4

### Long-term (Phase 4)

1. [ ] Neural Compressor backend for ternary
2. [ ] ONNX export with ternary ops
3. [ ] Multi-hardware backends

---

**Document Version:** 1.0
**Generated:** 2025-12-09
**Author:** Claude Code Analysis
