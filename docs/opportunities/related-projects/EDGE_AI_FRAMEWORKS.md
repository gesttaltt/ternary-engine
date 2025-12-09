# Edge AI & Framework Integration

**Doc-Type:** Ecosystem Analysis · Version 1.0 · Generated 2025-12-09

This document analyzes edge AI frameworks and deployment platforms that represent integration targets for Ternary Engine.

---

## Table of Contents

1. [Overview](#overview)
2. [LiteRT (TensorFlow Lite)](#1-litert-tensorflow-lite)
3. [ONNX Runtime](#2-onnx-runtime)
4. [AI Edge Torch](#3-ai-edge-torch)
5. [Framework Integration Patterns](#framework-integration-patterns)
6. [Edge Deployment Strategy](#edge-deployment-strategy)
7. [Lessons for Ternary Engine](#lessons-for-ternary-engine)

---

## Overview

Edge AI is a natural fit for Ternary Engine because:
1. **Memory constraints** - 8× compression matters on edge devices
2. **Power efficiency** - Simpler operations = less energy
3. **No GPU required** - CPU SIMD is the target

| Framework | Platform Focus | Quantization | Integration Path |
|-----------|----------------|--------------|------------------|
| LiteRT | Mobile, IoT | INT8, INT16 | Custom delegate |
| ONNX Runtime | Cross-platform | INT8 | Execution provider |
| AI Edge Torch | Mobile | INT8 | Export pipeline |

---

## 1. LiteRT (TensorFlow Lite)

### Repository Information

- **URL:** https://github.com/google-ai-edge/LiteRT
- **Previous Name:** TensorFlow Lite
- **Stars:** 2,000+
- **Language:** C++, Python
- **License:** Apache 2.0
- **Status:** Very active (Google-maintained)

### What It Is

LiteRT (formerly TFLite) is Google's edge inference framework:
- Mobile: Android, iOS
- Embedded: Raspberry Pi, microcontrollers
- Web: WASM

### Architecture

```
LiteRT Architecture:

┌─────────────────────────────────────────────────────────────┐
│                    Model Format (.tflite)                    │
│         FlatBuffers-based, includes ops and weights          │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    Interpreter                               │
│              Graph execution orchestration                   │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    Delegates (Hardware Acceleration)         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ GPU         │  │ NNAPI       │  │ Custom Delegate     │  │
│  │ Delegate    │  │ (Android)   │  │ (Ternary Engine?)   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    Kernel Library                            │
│            Optimized implementations per op                  │
└─────────────────────────────────────────────────────────────┘
```

### Quantization Support

```python
# LiteRT quantization options
import tensorflow as tf

# Post-training quantization
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)

# Dynamic range quantization (weights only)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Full integer quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
converter.representative_dataset = representative_data_gen

# Float16 quantization
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()
```

### Custom Delegate Interface

```cpp
// Creating a custom delegate for Ternary Engine
#include "tensorflow/lite/delegates/delegate.h"

class TernaryDelegate : public TfLiteDelegate {
public:
    TernaryDelegate() {
        // Set delegate flags
        flags = kTfLiteDelegateFlagsAllowDynamicTensors;

        // Provide callback functions
        Prepare = TernaryPrepare;
        CopyFromBufferHandle = TernaryCopyFromBuffer;
        CopyToBufferHandle = TernaryCopyToBuffer;
        FreeBufferHandle = TernaryFreeBuffer;
    }

    // Called to check if this delegate can handle an op
    static TfLiteStatus Prepare(
        TfLiteContext* context,
        TfLiteDelegate* delegate
    ) {
        // Identify nodes this delegate can handle
        TfLiteIntArray* nodes_to_replace = TfLiteIntArrayCreate(0);

        for (int i = 0; i < execution_plan->size; ++i) {
            TfLiteNode* node = &nodes[execution_plan->data[i]];
            TfLiteRegistration* registration = &registrations[node->builtin_code];

            if (CanHandleOp(registration)) {
                // Add to list of ops we'll handle
                nodes_to_replace = TfLiteIntArrayAppend(nodes_to_replace, i);
            }
        }

        // Replace nodes with delegate kernels
        context->ReplaceNodeSubsetsWithDelegateKernels(
            context, TernaryKernelFactory(), nodes_to_replace, delegate
        );

        return kTfLiteOk;
    }
};

// Register delegate
TfLiteDelegate* CreateTernaryDelegate() {
    return new TernaryDelegate();
}
```

### Integration Opportunity

```python
# Goal: Use Ternary Engine as a TFLite delegate
import tensorflow as tf
from ternary_engine.tflite import TernaryDelegate

# Load model
interpreter = tf.lite.Interpreter(model_path="model.tflite")

# Add Ternary delegate
ternary_delegate = TernaryDelegate()
interpreter.modify_graph_with_delegate(ternary_delegate)

# Run inference - ternary ops use SIMD acceleration
interpreter.allocate_tensors()
interpreter.invoke()
```

---

## 2. ONNX Runtime

### Repository Information

- **URL:** https://github.com/microsoft/onnxruntime
- **Stars:** 15,000+
- **Language:** C++, Python
- **License:** MIT
- **Status:** Very active (Microsoft-maintained)

### What It Is

ONNX Runtime is a cross-platform inference engine for ONNX models:
- Supports multiple execution providers (CPU, CUDA, TensorRT, etc.)
- Used in production at Microsoft, Azure, Windows

### Architecture

```
ONNX Runtime Architecture:

┌─────────────────────────────────────────────────────────────┐
│                    ONNX Model (.onnx)                        │
│         Protocol buffers format, standardized ops            │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    Graph Optimization                        │
│       Constant folding, fusion, layout transformation        │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    Session (Execution)                       │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                Execution Providers                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ CPU         │  │ CUDA        │  │ Custom EP           │  │
│  │ (Default)   │  │ (NVIDIA)    │  │ (Ternary Engine?)   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Execution Provider Interface

```cpp
// Creating a Ternary Execution Provider
#include "onnxruntime/core/providers/shared_library/provider_api.h"

class TernaryExecutionProvider : public IExecutionProvider {
public:
    TernaryExecutionProvider() : IExecutionProvider("TernaryExecutionProvider") {}

    // Which ops can this provider handle?
    std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
        const onnxruntime::GraphViewer& graph,
        const IKernelLookup& /*kernel_lookup*/
    ) const override {
        std::vector<std::unique_ptr<ComputeCapability>> capabilities;

        for (const auto& node : graph.Nodes()) {
            // Check if this is an op we can accelerate
            if (CanHandle(node)) {
                std::unique_ptr<IndexedSubGraph> sub_graph =
                    std::make_unique<IndexedSubGraph>();
                sub_graph->nodes.push_back(node.Index());

                capabilities.push_back(
                    std::make_unique<ComputeCapability>(std::move(sub_graph))
                );
            }
        }

        return capabilities;
    }

    // Create kernel for a node
    common::Status CreateKernel(
        const KernelCreateInfo& info,
        const KernelRegistryManager& /*kernel_registry_mgr*/,
        std::unique_ptr<OpKernel>& out
    ) const override {
        // Create appropriate kernel based on op type
        if (info.kernel_def->OpName() == "MatMul") {
            out = std::make_unique<TernaryMatMulKernel>(info);
        } else if (info.kernel_def->OpName() == "Conv") {
            out = std::make_unique<TernaryConvKernel>(info);
        }
        return common::Status::OK();
    }
};

// Register the provider
void RegisterTernaryProvider() {
    OrtSessionOptions* options = OrtGetApiBase()->GetApi(ORT_API_VERSION)->CreateSessionOptions();
    OrtSessionOptionsAppendExecutionProvider(
        options, "TernaryExecutionProvider", nullptr, nullptr, 0
    );
}
```

### Custom ONNX Operator

```cpp
// Define custom ternary operators
#include "onnxruntime/core/framework/custom_ops_author.h"

// Custom operator: TernaryMatMul
struct TernaryMatMulOp : Ort::CustomOpBase<TernaryMatMulOp, TernaryMatMulKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new TernaryMatMulKernel(api, info);
    }

    const char* GetName() const { return "TernaryMatMul"; }
    const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }

    size_t GetInputTypeCount() const { return 2; }
    ONNXTensorElementDataType GetInputType(size_t index) const {
        return index == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT :
                           ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;  // Packed ternary
    }

    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
};

// Kernel implementation
struct TernaryMatMulKernel {
    TernaryMatMulKernel(const OrtApi& api, const OrtKernelInfo* info) {}

    void Compute(OrtKernelContext* context) {
        // Get inputs
        const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
        const OrtValue* input_W = ort_.KernelContext_GetInput(context, 1);

        // Get data pointers
        const float* X = ort_.GetTensorData<float>(input_X);
        const int8_t* W = ort_.GetTensorData<int8_t>(input_W);

        // Call Ternary Engine SIMD kernel
        ternary_matmul_avx2(X, W, output, M, N, K);
    }
};
```

### Integration Opportunity

```python
# Goal: Use Ternary Engine with ONNX Runtime
import onnxruntime as ort
from ternary_engine.onnx import TernaryExecutionProvider

# Create session with Ternary EP
sess_options = ort.SessionOptions()
sess = ort.InferenceSession(
    "model.onnx",
    sess_options,
    providers=["TernaryExecutionProvider", "CPUExecutionProvider"]
)

# Run inference - ternary layers use SIMD acceleration
result = sess.run(None, {"input": data})
```

---

## 3. AI Edge Torch

### Repository Information

- **URL:** https://github.com/google-ai-edge/ai-edge-torch
- **Stars:** 500+
- **Language:** Python
- **License:** Apache 2.0
- **Status:** Active (Google-maintained)

### What It Is

AI Edge Torch converts PyTorch models to TFLite format:

```python
import torch
import ai_edge_torch

# Define PyTorch model
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)

# Convert to TFLite
model = MyModel().eval()
sample_input = torch.randn(1, 784)

tflite_model = ai_edge_torch.convert(model, (sample_input,))
tflite_model.export("model.tflite")
```

### Integration Opportunity

```python
# Goal: Convert Ternary PyTorch models to TFLite
import ai_edge_torch
from ternary_engine.torch import TernaryLinear, TernaryConv2d

class TernaryModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = TernaryLinear(784, 256)
        self.fc2 = TernaryLinear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Register custom converter for TernaryLinear
@ai_edge_torch.register_converter(TernaryLinear)
def convert_ternary_linear(converter, node, inputs):
    # Convert to TFLite custom op or decompose
    pass

# Convert model
model = TernaryModel().eval()
tflite_model = ai_edge_torch.convert(model, (sample_input,))
```

---

## Framework Integration Patterns

### Pattern 1: Export Quantized Weights

```python
# Export ternary weights for use in any framework
def export_ternary_weights(model, path):
    weights = {}

    for name, param in model.named_parameters():
        if hasattr(param, 'ternary_data'):
            # Export packed ternary format
            weights[name] = {
                'packed': param.ternary_data.numpy(),  # Packed 2-bit
                'scale': param.scale.numpy(),
                'shape': param.shape
            }
        else:
            weights[name] = param.numpy()

    np.savez(path, **weights)

# Load in any framework
def load_ternary_weights(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}
```

### Pattern 2: ONNX Custom Domain

```python
# Define ternary operations in custom ONNX domain
import onnx
from onnx import helper, TensorProto

# Create custom op
ternary_matmul = helper.make_node(
    'TernaryMatMul',              # Op type
    inputs=['X', 'W', 'scale'],   # Input names
    outputs=['Y'],                 # Output names
    domain='com.ternary'          # Custom domain
)

# Create graph with custom op
graph_def = helper.make_graph(
    nodes=[ternary_matmul],
    name='TernaryModel',
    inputs=[
        helper.make_tensor_value_info('X', TensorProto.FLOAT, [None, 784]),
        helper.make_tensor_value_info('W', TensorProto.INT8, [784, 256]),
        helper.make_tensor_value_info('scale', TensorProto.FLOAT, [256])
    ],
    outputs=[
        helper.make_tensor_value_info('Y', TensorProto.FLOAT, [None, 256])
    ]
)

# Create model with custom opset
model_def = helper.make_model(
    graph_def,
    opset_imports=[
        helper.make_opsetid('', 14),           # Standard ONNX
        helper.make_opsetid('com.ternary', 1)  # Custom ternary ops
    ]
)

onnx.save(model_def, 'model_with_ternary.onnx')
```

### Pattern 3: Framework-Agnostic C Library

```c
// ternary_inference.h - Framework-agnostic C API
#ifndef TERNARY_INFERENCE_H
#define TERNARY_INFERENCE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to ternary model
typedef struct TernaryModel TernaryModel;

// Load model from file
TernaryModel* ternary_load_model(const char* path);

// Run inference
int ternary_inference(
    TernaryModel* model,
    const float* input,
    size_t input_size,
    float* output,
    size_t output_size
);

// Free model
void ternary_free_model(TernaryModel* model);

#ifdef __cplusplus
}
#endif

#endif // TERNARY_INFERENCE_H
```

```c
// Usage in any language via FFI
// Python: ctypes
// Java: JNI
// C#: P/Invoke
// Rust: bindgen
```

---

## Edge Deployment Strategy

### Target Platforms

| Platform | Framework | Integration |
|----------|-----------|-------------|
| Android | LiteRT | Custom delegate |
| iOS | Core ML | Custom layer |
| Raspberry Pi | ONNX Runtime | Execution provider |
| Browser | ONNX Web | Custom backend |
| Microcontroller | Custom | Direct C library |

### Memory Requirements

```
Model Size Comparison (7B parameters):

FP32:  28 GB    ████████████████████████████████████████
FP16:  14 GB    ████████████████████
INT8:   7 GB    ██████████
INT4:  3.5 GB   █████
Ternary: 1.75 GB  ██  ← Fits in mobile RAM!

Edge Device RAM:
- Raspberry Pi 4: 2-8 GB
- Android phone:  4-12 GB
- iPhone:         4-8 GB
```

### Deployment Pipeline

```
Deployment Pipeline:

┌─────────────────────────────────────────────────────────────┐
│                    1. Training (PyTorch)                     │
│    Train with TernaryLinear layers or quantize post-hoc      │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    2. Export                                 │
│    a) PyTorch → ONNX with custom ops                        │
│    b) PyTorch → TFLite via ai_edge_torch                    │
│    c) Custom .ternary format                                 │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    3. Optimize                               │
│    Fuse ops, optimize memory layout, profile                 │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    4. Deploy                                 │
│  ┌─────────────────┬─────────────────┬─────────────────┐    │
│  │ Mobile          │ Desktop         │ Embedded        │    │
│  │ (LiteRT/ONNX)   │ (ONNX Runtime)  │ (C library)     │    │
│  └─────────────────┴─────────────────┴─────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Lessons for Ternary Engine

### Integration Priorities

1. **ONNX Runtime (High Priority)**
   - Cross-platform support
   - Growing ecosystem
   - Clear execution provider API

2. **LiteRT/TFLite (Medium Priority)**
   - Mobile deployment
   - Google ecosystem
   - Custom delegate API

3. **AI Edge Torch (Lower Priority)**
   - Depends on PyTorch integration first
   - Simplifies mobile deployment

### Implementation Roadmap

```
Phase 1: Core Integration
├── ONNX custom ops (TernaryMatMul, TernaryConv)
├── ONNX Runtime execution provider
└── Python wrapper for easy use

Phase 2: Mobile Support
├── LiteRT custom delegate
├── Android integration guide
└── iOS Core ML converter

Phase 3: Full Ecosystem
├── AI Edge Torch converter
├── ONNX Web backend
└── Microcontroller runtime
```

### Key Technical Decisions

1. **Model format:** Use ONNX as interchange format
2. **Weight storage:** 2-bit packed with separate scales
3. **Runtime:** C library with language bindings
4. **Optimization:** Platform-specific SIMD kernels

---

## Action Items

### Immediate

1. [ ] Define ONNX custom op schema for ternary operations
2. [ ] Implement ONNX export from PyTorch ternary layers
3. [ ] Create basic ONNX Runtime execution provider

### Short-term

1. [ ] Complete ONNX Runtime integration
2. [ ] Add graph optimization passes (op fusion)
3. [ ] Benchmark on desktop and Raspberry Pi

### Medium-term

1. [ ] Implement LiteRT delegate
2. [ ] Create mobile demo apps
3. [ ] Document deployment guides

---

**Document Version:** 1.0
**Generated:** 2025-12-09
**Author:** Claude Code Analysis
