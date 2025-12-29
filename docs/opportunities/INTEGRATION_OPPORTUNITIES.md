# Integration Opportunities

**Doc-Type:** Strategic Analysis · Version 1.0 · Generated 2025-12-09

This document details opportunities for integrating the Ternary Engine with popular ML/AI frameworks, NumPy, and the broader Python ecosystem.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [PyTorch Integration](#1-pytorch-integration)
3. [TensorFlow/JAX Integration](#2-tensorflowjax-integration)
4. [NumPy Integration](#3-numpy-integration)
5. [HuggingFace Integration](#4-huggingface-integration)
6. [ONNX Runtime Integration](#5-onnx-runtime-integration)
7. [Scikit-learn Integration](#6-scikit-learn-integration)
8. [GPU/CUDA Support](#7-gpucuda-support)
9. [Implementation Recommendations](#implementation-recommendations)

---

## Executive Summary

| Integration | Strategic Value | Effort | Priority |
|-------------|-----------------|--------|----------|
| **PyTorch** | Enables ML research/production | High | CRITICAL |
| **NumPy ufuncs** | Native array operations | Medium | HIGH |
| **HuggingFace** | Model hub access | Medium | HIGH |
| **TensorFlow** | Production deployment | High | MEDIUM |
| **ONNX** | Model interchange | Medium | MEDIUM |
| **CUDA/GPU** | Performance at scale | Very High | MEDIUM |
| **Scikit-learn** | Traditional ML | Low | LOW |
| **JAX** | Research community | High | LOW |

### Current State

```
Integration Status:
PyTorch        ░░░░░░░░░░░░░░░░░░░░ 0%  (No integration)
NumPy          ██████░░░░░░░░░░░░░░ 30% (Array conversions only)
TensorFlow     ░░░░░░░░░░░░░░░░░░░░ 0%  (No integration)
HuggingFace    ░░░░░░░░░░░░░░░░░░░░ 0%  (No integration)
CUDA/GPU       ░░░░░░░░░░░░░░░░░░░░ 0%  (CPU only)
```

---

## 1. PyTorch Integration

### Why This Matters

PyTorch is the dominant framework for ML research and increasingly for production. Without PyTorch integration, Ternary Engine cannot participate in the ML ecosystem.

### What's Missing

```python
# DESIRED: Native PyTorch integration
import torch
from ternary_engine.torch import TernaryLinear, TernaryConv2d, ternary_quantize

# Quantize existing model to ternary
model = torchvision.models.resnet18(pretrained=True)
ternary_model = ternary_quantize(model, threshold=0.5)

# Or build from scratch
class TernaryMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = TernaryLinear(784, 256)
        self.fc2 = TernaryLinear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Training with ternary-aware gradients
model = TernaryMLP()
optimizer = torch.optim.Adam(model.parameters())
# ... training loop with STE gradients
```

### Required Components

#### 1.1 Ternary Layer Types

```python
# src/engine/torch/layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TernaryLinear(nn.Module):
    """
    Linear layer with ternary weights {-1, 0, +1}.

    Uses Straight-Through Estimator (STE) for gradient computation.
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, threshold: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold

        # Full-precision shadow weights for training
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def quantize_weights(self, w: torch.Tensor) -> torch.Tensor:
        """Quantize to ternary with STE gradients."""
        # Forward: quantize to {-1, 0, +1}
        ternary = torch.zeros_like(w)
        ternary[w > self.threshold] = 1.0
        ternary[w < -self.threshold] = -1.0

        # STE: gradient passes through unchanged
        return w + (ternary - w).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ternary_weight = self.quantize_weights(self.weight)
        return F.linear(x, ternary_weight, self.bias)

    def get_ternary_weights(self) -> torch.Tensor:
        """Export final ternary weights for inference."""
        with torch.no_grad():
            return self.quantize_weights(self.weight).to(torch.int8)


class TernaryConv2d(nn.Module):
    """2D Convolution with ternary weights."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, padding: int = 0,
                 bias: bool = True, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Same quantization as TernaryLinear
        ternary_weight = self._quantize(self.weight)
        return F.conv2d(x, ternary_weight, self.bias,
                       self.stride, self.padding)
```

#### 1.2 Custom Autograd Functions

```python
# src/engine/torch/autograd.py
import torch
from torch.autograd import Function
import ternary_simd_engine as tse

class TernaryMatmulFunction(Function):
    """
    Custom autograd function for ternary matrix multiplication.

    Uses SIMD-accelerated forward pass with STE backward.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor):
        # Save for backward
        ctx.save_for_backward(input, weight)

        # Quantize weights to ternary
        ternary_weight = torch.zeros_like(weight)
        ternary_weight[weight > 0.5] = 1.0
        ternary_weight[weight < -0.5] = -1.0

        # Convert to packed format for SIMD
        packed_weight = pack_ternary(ternary_weight)

        # Use SIMD-accelerated matmul
        output = tse.ternary_matmul(input.numpy(), packed_weight)

        return torch.from_numpy(output)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight = ctx.saved_tensors

        # STE: pass gradients through quantization
        grad_input = grad_output @ weight
        grad_weight = input.t() @ grad_output

        return grad_input, grad_weight
```

#### 1.3 Model Quantization Utilities

```python
# src/engine/torch/quantization.py
import torch
import torch.nn as nn
from typing import Dict, Optional

def quantize_model_to_ternary(
    model: nn.Module,
    threshold: float = 0.5,
    skip_layers: Optional[list] = None
) -> nn.Module:
    """
    Convert a PyTorch model to use ternary weights.

    Args:
        model: Source model with FP32/FP16 weights
        threshold: Threshold for zero region
        skip_layers: Layer names to keep in full precision

    Returns:
        Model with ternary weights

    Example:
        >>> model = torchvision.models.resnet18(pretrained=True)
        >>> ternary_model = quantize_model_to_ternary(model)
        >>> # Original: 44.6 MB, Ternary: 5.6 MB (8× smaller)
    """
    skip_layers = skip_layers or []

    for name, module in model.named_modules():
        if name in skip_layers:
            continue

        if isinstance(module, nn.Linear):
            # Replace with TernaryLinear
            ternary_layer = TernaryLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                threshold=threshold
            )
            # Copy weights
            ternary_layer.weight.data = module.weight.data.clone()
            if module.bias is not None:
                ternary_layer.bias.data = module.bias.data.clone()

            # Replace in model
            _replace_module(model, name, ternary_layer)

        elif isinstance(module, nn.Conv2d):
            # Replace with TernaryConv2d
            # ... similar replacement logic
            pass

    return model


def export_ternary_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Export quantized weights for efficient inference.

    Returns:
        Dictionary mapping layer names to packed ternary tensors
    """
    weights = {}

    for name, module in model.named_modules():
        if hasattr(module, 'get_ternary_weights'):
            weights[name] = module.get_ternary_weights()

    return weights
```

### Implementation Plan

| Phase | Task | Deliverable |
|-------|------|-------------|
| 1 | Core layers | TernaryLinear, TernaryConv2d |
| 2 | Autograd | Custom backward with STE |
| 3 | Quantization | Model conversion utilities |
| 4 | SIMD bridge | PyTorch → C++ SIMD kernel calls |
| 5 | Examples | MNIST, CIFAR, pretrained models |

### Estimated Impact

- **Memory reduction:** 8× smaller models
- **Research adoption:** Access to ML research community
- **Production path:** Clear deployment story

---

## 2. TensorFlow/JAX Integration

### Why This Matters

TensorFlow dominates production ML deployment. JAX is growing in research. Supporting both expands market reach.

### What's Missing

```python
# DESIRED: TensorFlow Keras layers
import tensorflow as tf
from ternary_engine.tensorflow import TernaryDense, TernaryConv2D

model = tf.keras.Sequential([
    TernaryDense(128, activation='relu'),
    TernaryDense(10, activation='softmax')
])

# DESIRED: JAX integration
import jax
from ternary_engine.jax import ternary_linear

@jax.jit
def forward(params, x):
    return ternary_linear(params, x)
```

### Required Components

#### 2.1 TensorFlow Custom Ops

```python
# src/engine/tensorflow/ops.py
import tensorflow as tf
from tensorflow.python.framework import ops

# Load custom SIMD kernel
_ternary_ops = tf.load_op_library('./ternary_tf_ops.so')

@tf.custom_gradient
def ternary_matmul(x, w, threshold=0.5):
    """
    Ternary matrix multiplication with custom gradient.
    """
    # Quantize weights
    ternary_w = tf.where(w > threshold, tf.ones_like(w),
                 tf.where(w < -threshold, -tf.ones_like(w),
                         tf.zeros_like(w)))

    # Forward pass (uses custom kernel if available)
    result = tf.matmul(x, ternary_w)

    def grad(dy):
        # STE backward
        grad_x = tf.matmul(dy, tf.transpose(ternary_w))
        grad_w = tf.matmul(tf.transpose(x), dy)
        return grad_x, grad_w

    return result, grad


class TernaryDense(tf.keras.layers.Layer):
    """Dense layer with ternary weights."""

    def __init__(self, units, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.threshold = threshold

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )

    def call(self, inputs):
        return ternary_matmul(inputs, self.w, self.threshold) + self.b
```

#### 2.2 JAX Custom Primitives

```python
# src/engine/jax/primitives.py
import jax
import jax.numpy as jnp
from jax import custom_vjp

@custom_vjp
def ternary_quantize(w, threshold=0.5):
    """Quantize to ternary values with custom gradient."""
    return jnp.where(w > threshold, 1.0,
                    jnp.where(w < -threshold, -1.0, 0.0))

def ternary_quantize_fwd(w, threshold):
    result = ternary_quantize(w, threshold)
    return result, (w,)

def ternary_quantize_bwd(res, g):
    # STE: pass gradient through
    return (g, None)

ternary_quantize.defvjp(ternary_quantize_fwd, ternary_quantize_bwd)
```

### Implementation Priority

**TensorFlow:** Medium priority - Focus on PyTorch first, then port patterns

**JAX:** Low priority - Niche audience, but patterns similar to TensorFlow

---

## 3. NumPy Integration

### Why This Matters

NumPy is the foundation of Python scientific computing. Native integration enables seamless interoperability.

### What's Missing

Currently, Ternary Engine requires explicit conversion:

```python
# CURRENT: Manual conversion
import numpy as np
import ternary_simd_engine as tse

a = np.array([0, 1, 2, 1, 0], dtype=np.uint8)
b = np.array([2, 1, 0, 1, 2], dtype=np.uint8)
result = np.frombuffer(tse.tadd(a.tobytes(), b.tobytes()), dtype=np.uint8)

# DESIRED: NumPy ufunc integration
import numpy as np
from ternary_engine import ternary as tn

a = tn.array([0, 1, 2, 1, 0])
b = tn.array([2, 1, 0, 1, 2])
result = a + b  # Uses ternary addition automatically
result = tn.min(a, b)  # ternary min
result = ~a  # ternary NOT
```

### Required Components

#### 3.1 Ternary Array Class

```python
# src/engine/numpy/array.py
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

class TernaryArray(NDArrayOperatorsMixin):
    """
    NumPy-compatible array with ternary values.

    Supports NumPy ufuncs and array operations with automatic
    dispatch to SIMD-accelerated kernels.
    """

    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __init__(self, data, dtype=np.uint8):
        if isinstance(data, np.ndarray):
            self._data = data.astype(dtype)
        else:
            self._data = np.array(data, dtype=dtype)

        # Validate ternary values
        if not np.all((self._data >= 0) & (self._data <= 2)):
            raise ValueError("Ternary values must be in {0, 1, 2}")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Intercept NumPy ufuncs and dispatch to ternary ops."""

        if method != '__call__':
            return NotImplemented

        # Map NumPy ufuncs to ternary operations
        ufunc_map = {
            np.add: tse.tadd,
            np.subtract: tse.tadd,  # Use tnot + tadd
            np.multiply: tse.tmul,
            np.minimum: tse.tmin,
            np.maximum: tse.tmax,
            np.invert: tse.tnot,
        }

        if ufunc in ufunc_map:
            return self._apply_ternary_op(ufunc_map[ufunc], inputs)

        return NotImplemented

    def _apply_ternary_op(self, op, inputs):
        """Apply ternary SIMD operation."""
        arrays = [self._to_bytes(x) for x in inputs]
        result = op(*arrays)
        return TernaryArray(np.frombuffer(result, dtype=np.uint8))

    # Arithmetic operators
    def __add__(self, other):
        return self.__array_ufunc__(np.add, '__call__', self, other)

    def __mul__(self, other):
        return self.__array_ufunc__(np.multiply, '__call__', self, other)

    def __invert__(self):
        return self.__array_ufunc__(np.invert, '__call__', self)

    # Properties
    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"TernaryArray({self._data})"
```

#### 3.2 NumPy ufunc Registration

```python
# src/engine/numpy/ufuncs.py
import numpy as np

# Create custom ufuncs for ternary operations
tadd_ufunc = np.frompyfunc(lambda a, b: _tadd_scalar(a, b), 2, 1)
tmul_ufunc = np.frompyfunc(lambda a, b: _tmul_scalar(a, b), 2, 1)
tnot_ufunc = np.frompyfunc(lambda a: _tnot_scalar(a), 1, 1)

# Register as NumPy generalized ufuncs
np.add.register(TernaryArray, tadd_ufunc)
np.multiply.register(TernaryArray, tmul_ufunc)
np.invert.register(TernaryArray, tnot_ufunc)
```

### Implementation Priority

**High Priority** - NumPy integration provides immediate value with moderate effort

---

## 4. HuggingFace Integration

### Why This Matters

HuggingFace Transformers is the standard for pretrained LLMs. Integration enables access to thousands of models.

### What's Missing

```python
# DESIRED: HuggingFace integration
from transformers import AutoModelForCausalLM
from ternary_engine.huggingface import TernaryConfig, ternary_quantize

# Load and quantize model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
ternary_model = ternary_quantize(model, config=TernaryConfig(threshold=0.5))

# Save/load ternary format
ternary_model.save_pretrained("./llama-2-7b-ternary")
loaded = AutoModelForCausalLM.from_pretrained("./llama-2-7b-ternary")

# Push to Hub
ternary_model.push_to_hub("user/llama-2-7b-ternary")
```

### Required Components

#### 4.1 Quantization Config

```python
# src/engine/huggingface/config.py
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class TernaryConfig:
    """Configuration for ternary quantization."""

    threshold: float = 0.5
    skip_layers: List[str] = None
    per_channel: bool = True
    calibration_samples: int = 128

    # Layers to keep in higher precision
    sensitive_layers: List[str] = None

    def to_dict(self):
        return {
            "quant_method": "ternary",
            "threshold": self.threshold,
            "skip_layers": self.skip_layers or [],
            "per_channel": self.per_channel,
        }
```

#### 4.2 Model Quantizer

```python
# src/engine/huggingface/quantizer.py
import torch
from transformers import PreTrainedModel
from typing import Optional

def ternary_quantize(
    model: PreTrainedModel,
    config: TernaryConfig,
    calibration_data: Optional[torch.Tensor] = None
) -> PreTrainedModel:
    """
    Quantize HuggingFace model to ternary weights.

    Args:
        model: HuggingFace model
        config: Ternary quantization configuration
        calibration_data: Optional data for threshold calibration

    Returns:
        Quantized model with ternary weights
    """
    # Calibrate thresholds if data provided
    if calibration_data is not None:
        config.threshold = _calibrate_threshold(model, calibration_data)

    # Quantize each layer
    for name, module in model.named_modules():
        if name in (config.skip_layers or []):
            continue

        if isinstance(module, torch.nn.Linear):
            _quantize_linear(module, config)
        elif isinstance(module, torch.nn.Conv2d):
            _quantize_conv2d(module, config)

    # Add ternary config to model
    model.config.quantization_config = config.to_dict()

    return model
```

#### 4.3 Custom Model Classes

```python
# src/engine/huggingface/modeling.py
from transformers import PreTrainedModel

class TernaryPreTrainedModel(PreTrainedModel):
    """Base class for ternary models."""

    def _init_weights(self, module):
        """Initialize ternary weights."""
        if isinstance(module, TernaryLinear):
            # Ternary-aware initialization
            torch.nn.init.uniform_(module.weight, -1.0, 1.0)

    def save_pretrained(self, save_directory, **kwargs):
        """Save with ternary-specific format."""
        super().save_pretrained(save_directory, **kwargs)

        # Save packed ternary weights
        self._save_ternary_weights(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load with ternary weight support."""
        model = super().from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        # Load packed ternary weights if available
        model._load_ternary_weights(pretrained_model_name_or_path)

        return model
```

### Implementation Priority

**High Priority** - Access to model ecosystem is critical for adoption

---

## 5. ONNX Runtime Integration

### Why This Matters

ONNX is the standard for model interchange. Support enables deployment across platforms.

### What's Missing

```python
# DESIRED: ONNX export/import
from ternary_engine.onnx import export_ternary, TernaryExecutionProvider

# Export PyTorch model to ONNX with ternary ops
model = TernaryMLP()
export_ternary(model, "model.onnx", input_shape=(1, 784))

# Run inference with custom execution provider
import onnxruntime as ort
sess = ort.InferenceSession(
    "model.onnx",
    providers=[TernaryExecutionProvider()]
)
output = sess.run(None, {"input": data})
```

### Required Components

#### 5.1 Custom ONNX Operators

```cpp
// src/engine/onnx/ternary_ops.cpp
#include <onnxruntime/core/graph/graph.h>

// Register ternary domain
static const char* TERNARY_DOMAIN = "com.ternary";

// Custom operator for ternary matmul
struct TernaryMatmul {
    static OrtStatus* Compute(OrtKernelContext* context) {
        // Get inputs
        const OrtValue* input = ...;
        const OrtValue* weight = ...;

        // Use SIMD kernel
        ternary_matmul_avx2(input_data, weight_data, output_data, ...);

        return nullptr;
    }
};

// Register operator
ORT_CUSTOM_OP_DOMAIN(TernaryDomain, TERNARY_DOMAIN, 1,
    OrtCustomOp<TernaryMatmul>,
    OrtCustomOp<TernaryAdd>,
    OrtCustomOp<TernaryMul>
);
```

#### 5.2 Export Utilities

```python
# src/engine/onnx/export.py
import torch
import onnx

def export_ternary(
    model: torch.nn.Module,
    output_path: str,
    input_shape: tuple,
    opset_version: int = 14
):
    """
    Export PyTorch ternary model to ONNX format.

    Uses custom ternary ops domain for optimized inference.
    """
    # Register custom symbolic functions
    _register_ternary_symbolics()

    # Create dummy input
    dummy_input = torch.randn(input_shape)

    # Export with custom domain
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        custom_opsets={"com.ternary": 1},
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}}
    )
```

### Implementation Priority

**Medium Priority** - Important for production deployment but requires ONNX expertise

---

## 6. Scikit-learn Integration

### Why This Matters

Scikit-learn is widely used for traditional ML. Integration enables ternary support for non-deep learning use cases.

### What's Missing

```python
# DESIRED: Scikit-learn estimators
from sklearn.base import BaseEstimator, ClassifierMixin
from ternary_engine.sklearn import TernaryLogisticRegression

# Train ternary classifier
clf = TernaryLogisticRegression(threshold=0.5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# Model is 8× smaller than standard sklearn
print(f"Model size: {clf.model_size_bytes} bytes")
```

### Implementation Priority

**Low Priority** - Niche use case, implement after core integrations

---

## 7. GPU/CUDA Support

### Why This Matters

GPU acceleration is essential for competitive training and inference performance.

### What's Missing

```python
# DESIRED: GPU-accelerated operations
import torch
from ternary_engine.cuda import ternary_matmul_cuda

# Move to GPU
a = torch.randn(1024, 1024).cuda()
w = ternary_quantize(torch.randn(1024, 1024)).cuda()

# CUDA-accelerated ternary matmul
result = ternary_matmul_cuda(a, w)  # 10-100× faster than CPU
```

### Required Components

#### 7.1 CUDA Kernels

```cuda
// src/engine/cuda/ternary_kernels.cu
#include <cuda_runtime.h>

__global__ void ternary_matmul_kernel(
    const float* __restrict__ input,
    const int8_t* __restrict__ weight,  // Packed ternary
    float* __restrict__ output,
    int M, int N, int K
) {
    // Shared memory for weight tile
    __shared__ int8_t weight_tile[TILE_K][TILE_N];

    // Each thread computes one element
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    float sum = 0.0f;

    for (int k = 0; k < K; k += TILE_K) {
        // Load weight tile
        weight_tile[threadIdx.y][threadIdx.x] =
            weight[(k + threadIdx.y) * N + col];
        __syncthreads();

        // Compute partial sum
        for (int kk = 0; kk < TILE_K; kk++) {
            int8_t w = weight_tile[kk][threadIdx.x];
            float a = input[row * K + k + kk];

            // Ternary multiply: w * a where w ∈ {-1, 0, +1}
            sum += (w == 1) ? a : ((w == -1) ? -a : 0.0f);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        output[row * N + col] = sum;
    }
}
```

#### 7.2 PyTorch CUDA Extension

```python
# src/engine/cuda/extension.py
from torch.utils.cpp_extension import load

ternary_cuda = load(
    name='ternary_cuda',
    sources=[
        'src/engine/cuda/ternary_kernels.cu',
        'src/engine/cuda/ternary_ops.cpp'
    ],
    extra_cuda_cflags=['-O3', '-use_fast_math']
)

def ternary_matmul_cuda(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """CUDA-accelerated ternary matrix multiplication."""
    assert input.is_cuda and weight.is_cuda
    return ternary_cuda.ternary_matmul(input, weight)
```

### Implementation Priority

**Medium Priority** - High impact but high effort

---

## Implementation Recommendations

### Phase 1: Foundation (2-4 weeks)

| Task | Priority | Dependencies |
|------|----------|--------------|
| NumPy TernaryArray | HIGH | None |
| NumPy ufunc dispatch | HIGH | TernaryArray |
| PyTorch TernaryLinear | CRITICAL | None |
| PyTorch STE gradients | CRITICAL | TernaryLinear |

### Phase 2: Core ML (4-8 weeks)

| Task | Priority | Dependencies |
|------|----------|--------------|
| PyTorch Conv2d | HIGH | TernaryLinear |
| PyTorch quantize_model | HIGH | All layers |
| HuggingFace quantizer | HIGH | PyTorch integration |
| HuggingFace save/load | MEDIUM | Quantizer |

### Phase 3: Production (8-12 weeks)

| Task | Priority | Dependencies |
|------|----------|--------------|
| ONNX export | MEDIUM | PyTorch layers |
| ONNX runtime EP | MEDIUM | ONNX export |
| TensorFlow layers | MEDIUM | Pattern from PyTorch |
| CUDA kernels | MEDIUM | PyTorch integration |

### Phase 4: Scale (12+ weeks)

| Task | Priority | Dependencies |
|------|----------|--------------|
| CUDA optimization | LOW | CUDA kernels |
| JAX support | LOW | Pattern from TF |
| Scikit-learn | LOW | NumPy integration |
| Multi-GPU | LOW | CUDA optimized |

---

## Integration Testing

### Test Matrix

| Integration | Unit Tests | Integration Tests | E2E Tests |
|-------------|------------|-------------------|-----------|
| NumPy | Array ops | ufunc dispatch | Mixed ops |
| PyTorch | Layer forward | Training loop | MNIST |
| HuggingFace | Config | Model quantize | LLM inference |
| ONNX | Op export | Runtime | Cross-platform |

### Benchmark Targets

| Integration | Metric | Target |
|-------------|--------|--------|
| NumPy | Overhead vs raw | <5% |
| PyTorch | Training throughput | >80% of FP32 |
| HuggingFace | Quantization time | <5 min for 7B |
| ONNX | Inference speed | 2× FP32 |
| CUDA | TFLOPS | >10 TFLOPS |

---

**Document Version:** 1.0
**Generated:** 2025-12-09
**Author:** Claude Code Analysis
