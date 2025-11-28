# TritNet Integration Roadmap

**Doc-Type:** Implementation Roadmap · Version 1.0 · Created 2025-11-23

---

## Executive Summary

**TritNet** is a tiny, fully ternary-parameter neural network that learns exact dense243 arithmetic operations by training on complete truth tables, then distilling to pure ternary weights {-1, 0, +1} using the BitNet b1.58 pipeline.

### Vision

Replace lookup table (LUT) operations with learned matmul operations, enabling:
- **Potential hardware acceleration** via matmul accelerators (TPU, GPU, tensor cores)
- **Learned generalization** beyond hand-coded truth tables
- **Weight compression** - single weight matrix vs 243-entry LUTs per operation
- **Fuzzy ternary logic** - potential for approximate/probabilistic operations

---

## Architecture Overview

### Current Implementation (LUT-Based)

```
Input: 2 dense243 bytes (10 trits total)
  ↓
Unpack: byte → 5 trits each (10 trits {-1, 0, +1})
  ↓
Lookup: TADD_LUT[(a << 2) | b] for each trit pair
  ↓
Pack: 5 result trits → 1 dense243 byte
Output: Result byte
```

**Characteristics:**
- **Memory access:** 2 loads + 5 LUT lookups + 1 store
- **Compute:** Minimal (bit shifts and indexing)
- **Scalability:** Poor (memory-bound for large arrays)

### TritNet Implementation (Matmul-Based)

```
Input: 2 dense243 bytes (10 trits total)
  ↓
Unpack: byte → 5 trits each (10 trits {-1, 0, +1})
  ↓
TritNet Inference:
  Layer 1: [10 → 16] matmul with ternary weights W1 {-1, 0, +1}
  Activation: sign() → ternary outputs
  Layer 2: [16 → 5] matmul with ternary weights W2 {-1, 0, +1}
  ↓
Pack: 5 result trits → 1 dense243 byte
Output: Result byte
```

**Characteristics:**
- **Memory access:** 2 loads + 1 store + weight matrix loads
- **Compute:** 2 matmuls (10×16 + 16×5 = 160 + 80 = 240 ternary multiplies)
- **Scalability:** Excellent (compute-bound, batches well, accelerator-friendly)

---

## Phase 1: Truth Table Generation

### Goal
Generate complete arithmetic truth tables for dense243 operations.

### Operations to Train

**Binary operations** (2 inputs → 1 output):
- **tadd:** Ternary addition with saturation
- **tmul:** Ternary multiplication
- **tmin:** Element-wise minimum
- **tmax:** Element-wise maximum

**Unary operations** (1 input → 1 output):
- **tnot:** Ternary negation

### Dataset Structure

**Binary operation dataset:**
```
243² = 59,049 samples per operation
4 operations × 59,049 = 236,196 total samples

Sample format:
{
  "a": [t0, t1, t2, t3, t4],  # 5 trits from byte A
  "b": [t0, t1, t2, t3, t4],  # 5 trits from byte B
  "result": [r0, r1, r2, r3, r4],  # 5 result trits
  "operation": "tadd"
}
```

**Unary operation dataset:**
```
243 samples per operation
1 operation × 243 = 243 total samples

Sample format:
{
  "a": [t0, t1, t2, t3, t4],
  "result": [r0, r1, r2, r3, r4],
  "operation": "tnot"
}
```

### Implementation

```python
# models/tritnet/src/generate_truth_tables.py

import numpy as np
import json
from ternary_dense243_module import unpack, pack

def generate_binary_truth_table(operation, op_func):
    """Generate complete truth table for binary operation"""
    dataset = []

    for byte_a in range(243):  # All valid dense243 values
        for byte_b in range(243):
            # Unpack to 5 trits each
            trits_a = unpack(np.array([byte_a], dtype=np.uint8), num_trits=5)
            trits_b = unpack(np.array([byte_b], dtype=np.uint8), num_trits=5)

            # Compute result using LUT
            result_trits = []
            for i in range(5):
                r = op_func(trits_a[i], trits_b[i])
                result_trits.append(r)

            dataset.append({
                "a": trits_a.tolist(),
                "b": trits_b.tolist(),
                "result": result_trits,
                "operation": operation,
            })

    return dataset

def generate_unary_truth_table(operation, op_func):
    """Generate complete truth table for unary operation"""
    dataset = []

    for byte_a in range(243):
        trits_a = unpack(np.array([byte_a], dtype=np.uint8), num_trits=5)

        result_trits = []
        for i in range(5):
            r = op_func(trits_a[i])
            result_trits.append(r)

        dataset.append({
            "a": trits_a.tolist(),
            "result": result_trits,
            "operation": operation,
        })

    return dataset

# Generate all datasets
datasets = {
    "tadd": generate_binary_truth_table("tadd", lambda a, b: tadd_lut(a, b)),
    "tmul": generate_binary_truth_table("tmul", lambda a, b: tmul_lut(a, b)),
    "tmin": generate_binary_truth_table("tmin", lambda a, b: tmin_lut(a, b)),
    "tmax": generate_binary_truth_table("tmax", lambda a, b: tmax_lut(a, b)),
    "tnot": generate_unary_truth_table("tnot", lambda a: tnot_lut(a)),
}

# Save to JSON
for op, data in datasets.items():
    with open(f"datasets/tritnet_{op}_truth_table.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {len(data)} samples for {op}")
```

---

## Phase 2: BitNet Training

### Goal
Train tiny BitNet model on exact arithmetic using bitnet.cpp framework.

### Model Architecture

```
TritNet-Add (example for tadd operation):

Input Layer: 10 trits (5 from A + 5 from B)
  ↓
Hidden Layer 1: 16 ternary neurons {-1, 0, +1}
  Weights: W1[10×16] ∈ {-1, 0, +1}
  Activation: sign(x) → {-1, 0, +1}
  ↓
Hidden Layer 2: 16 ternary neurons
  Weights: W2[16×16] ∈ {-1, 0, +1}
  Activation: sign(x) → {-1, 0, +1}
  ↓
Output Layer: 5 ternary outputs
  Weights: W3[16×5] ∈ {-1, 0, +1}
  Activation: sign(x) → {-1, 0, +1}
```

**Total parameters:**
- W1: 10 × 16 = 160 ternary weights
- W2: 16 × 16 = 256 ternary weights
- W3: 16 × 5 = 80 ternary weights
- **Total: 496 ternary weights** (vs 243-entry LUT = 243 bytes)

### Training Script (using bitnet.cpp)

```python
# models/tritnet/src/train_bitnet.py

import torch
import bitnet  # Assuming bitnet.cpp Python bindings

def load_truth_table(operation):
    with open(f"datasets/tritnet_{operation}_truth_table.json") as f:
        data = json.load(f)

    # Convert to tensors
    X = []
    Y = []

    for sample in data:
        # Concatenate inputs: [a_trits, b_trits] for binary ops
        if "b" in sample:
            x = sample["a"] + sample["b"]  # 10 trits
        else:
            x = sample["a"]  # 5 trits (unary)

        y = sample["result"]  # 5 trits

        X.append(x)
        Y.append(y)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# Define TritNet model using BitNet layers
class TritNetAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = bitnet.TernaryLinear(10, 16)  # 10 inputs → 16 hidden
        self.layer2 = bitnet.TernaryLinear(16, 16)  # 16 → 16
        self.layer3 = bitnet.TernaryLinear(16, 5)   # 16 → 5 outputs
        self.activation = bitnet.SignActivation()   # Ternary activation

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        return x

# Train
model = TritNetAdd()
X_train, Y_train = load_truth_table("tadd")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        accuracy = ((outputs.round() == Y_train).float().mean().item())
        print(f"Epoch {epoch}: Loss {loss.item():.6f}, Accuracy {accuracy*100:.2f}%")

# Target: 100% accuracy (exact arithmetic)
```

### Distillation to Ternary Weights

```python
# Extract and quantize weights to {-1, 0, +1}
def distill_to_ternary(model):
    ternary_weights = {}

    for name, param in model.named_parameters():
        # BitNet b1.58 quantization: sign(W) with threshold
        W = param.data.cpu().numpy()
        W_ternary = np.sign(W)  # Maps to {-1, 0, +1}

        ternary_weights[name] = W_ternary

    return ternary_weights

weights = distill_to_ternary(model)

# Save as .tritnet format
np.savez("models/tritnet_tadd.tritnet",
         W1=weights['layer1.weight'],
         W2=weights['layer2.weight'],
         W3=weights['layer3.weight'])
```

---

## Phase 3: C++ TritNet Inference Backend

### Goal
Implement TritNet inference in C++ for integration with dense243 module.

### Implementation

```cpp
// src/engine/experimental/tritnet/tritnet_inference.h

#ifndef TRITNET_INFERENCE_H
#define TRITNET_INFERENCE_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// Ternary weight matrix
template<size_t Rows, size_t Cols>
struct TernaryWeights {
    std::array<int8_t, Rows * Cols> data;  // {-1, 0, +1}

    int8_t at(size_t row, size_t col) const {
        return data[row * Cols + col];
    }
};

// TritNet model for a single operation
struct TritNetModel {
    TernaryWeights<10, 16> W1;  // Input → Hidden1
    TernaryWeights<16, 16> W2;  // Hidden1 → Hidden2
    TernaryWeights<16, 5>  W3;  // Hidden2 → Output

    // Load from .tritnet file
    static TritNetModel load(const std::string& filepath);

    // Inference: 10 input trits → 5 output trits
    std::array<int8_t, 5> forward(const std::array<int8_t, 10>& inputs) const;
};

// Ternary matmul: C = sign(A × B)
template<size_t N, size_t M, size_t K>
std::array<int8_t, N * K> ternary_matmul(
    const std::array<int8_t, N * M>& A,
    const TernaryWeights<M, K>& B
) {
    std::array<int8_t, N * K> C;

    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < K; ++k) {
            int32_t sum = 0;
            for (size_t j = 0; j < M; ++j) {
                sum += A[i * M + j] * B.at(j, k);
            }
            // Ternary activation: sign(sum)
            C[i * K + k] = (sum > 0) ? 1 : (sum < 0) ? -1 : 0;
        }
    }

    return C;
}

// TritNet inference implementation
std::array<int8_t, 5> TritNetModel::forward(const std::array<int8_t, 10>& inputs) const {
    // Layer 1: [10] → [16]
    auto h1 = ternary_matmul<1, 10, 16>(inputs, W1);

    // Layer 2: [16] → [16]
    auto h2 = ternary_matmul<1, 16, 16>(h1, W2);

    // Layer 3: [16] → [5]
    auto output = ternary_matmul<1, 16, 5>(h2, W3);

    return output;
}

#endif // TRITNET_INFERENCE_H
```

### Integration with Dense243 Module

```cpp
// src/engine/ternary_dense243_module.cpp (updated)

// Global TritNet models (loaded at module init)
static std::unique_ptr<TritNetModel> g_tritnet_tadd;
static std::unique_ptr<TritNetModel> g_tritnet_tmul;
// ... other operations

// Backend selection implementation
void set_operation_backend(const std::string& backend) {
    if (backend == "lut") {
        g_backend = OperationBackend::LUT;
    } else if (backend == "tritnet") {
        // Load TritNet models
        g_tritnet_tadd = std::make_unique<TritNetModel>(
            TritNetModel::load("models/tritnet_tadd.tritnet")
        );
        // ... load other models

        g_backend = OperationBackend::TRITNET;
    }
}

// Updated binary operation template
template<typename ScalarOp>
py::array_t<uint8_t> binary_op_dense243(...) {
    for (ssize_t i = 0; i < n; ++i) {
        Dense243Unpacked a = dense243_unpack(a_data[i]);
        Dense243Unpacked b = dense243_unpack(b_data[i]);

        if (g_backend == OperationBackend::TRITNET) {
            // TritNet inference path
            std::array<int8_t, 10> inputs = {
                trit_to_int(a.t0), trit_to_int(a.t1), ...,
                trit_to_int(b.t0), trit_to_int(b.t1), ...
            };

            auto result = g_tritnet_tadd->forward(inputs);

            r_data[i] = dense243_pack(
                int_to_trit(result[0]),
                int_to_trit(result[1]),
                ...
            );
        } else {
            // LUT path (current)
            uint8_t r0 = scalar_op(a.t0, b.t0);
            // ...
        }
    }
}
```

---

## Phase 4: Benchmarking & Optimization

### Metrics to Track

**Correctness:**
- Exact match vs LUT results (100% accuracy required)
- Round-trip validation (pack → unpack → operate → pack)

**Performance:**
- Single operation latency (ns/operation)
- Throughput (Mops/sec for large arrays)
- Memory bandwidth utilization

**Model Quality:**
- Training accuracy (must reach 100%)
- Weight sparsity (how many {-1, 0, +1} vs zeros)
- Model size vs LUT size

### Expected Results

**LUT Baseline:**
- Latency: ~2 ns/operation (single lookup)
- Throughput: ~500 Mops/sec (memory-bound)
- Size: 243 bytes per operation

**TritNet (CPU):**
- Latency: ~50 ns/operation (240 ternary multiplies)
- Throughput: ~20 Mops/sec (compute-bound, single-threaded)
- Size: 496 ternary weights (~500 bytes)

**TritNet (GPU/TPU):**
- Latency: ~5 ns/operation (batched, accelerated)
- Throughput: ~2000 Mops/sec (batched matmul)
- Size: 496 ternary weights

**Conclusion:** TritNet likely slower on CPU for small batches, but potentially faster on accelerators with batching.

---

## Timeline & Milestones

### Week 1-2: Truth Table Generation
- [ ] Implement truth table generator
- [ ] Generate all 236,439 training samples
- [ ] Validate dataset correctness (manual spot checks)
- [ ] Split train/test (90/10)

### Week 3-4: BitNet Training
- [ ] Set up bitnet.cpp environment
- [ ] Implement TritNet model architecture
- [ ] Train on tadd operation (target: 100% accuracy)
- [ ] Validate on test set
- [ ] Train remaining operations (tmul, tmin, tmax, tnot)

### Week 5-6: Distillation & Export
- [ ] Implement ternary weight quantization
- [ ] Export models to .tritnet format
- [ ] Verify model files load correctly

### Week 7-8: C++ Inference Backend
- [ ] Implement ternary matmul operations
- [ ] Implement TritNet model loading
- [ ] Integrate with dense243 module
- [ ] Add backend selection API

### Week 9-10: Benchmarking
- [ ] Correctness validation (100% match vs LUT)
- [ ] Performance benchmarks (CPU single-threaded)
- [ ] Performance benchmarks (CPU batched)
- [ ] Performance benchmarks (GPU if available)

### Week 11-12: Optimization & Documentation
- [ ] Optimize critical paths (vectorize matmul)
- [ ] Write TritNet usage guide
- [ ] Create example workflows
- [ ] Update documentation

---

## Success Criteria

**Must Have:**
- ✅ 100% accuracy on all operations vs LUT
- ✅ Models successfully load and infer
- ✅ API allows runtime backend switching
- ✅ No performance regression for LUT backend

**Nice to Have:**
- ✅ TritNet faster than LUT on GPU/TPU
- ✅ Batched operations show >10× speedup
- ✅ Model size <50% of LUT size
- ✅ Generalization to fuzzy ternary logic

---

## Future Extensions

### Phase 5: Hybrid Operations
- Use TritNet for some operations, LUT for others
- Automatic backend selection based on array size
- Mixed-precision: LUT for hot paths, TritNet for cold storage

### Phase 6: Advanced TritNet
- Multi-operation models (single network handles all ops)
- Recurrent TritNet (sequential operation chains)
- Attention-based TritNet (learned operation fusion)

### Phase 7: Hardware Acceleration
- CUDA kernels for batched TritNet inference
- TPU optimization via XLA compilation
- FPGA implementation of ternary matmul

---

## References

- **BitNet:** https://github.com/microsoft/BitNet
- **bitnet.cpp:** https://github.com/microsoft/BitBLAS
- **Dense243 spec:** `docs/t5-dense243-spec.md`
- **Module README:** `src/engine/experimental/dense243/README.md`

---

**Version:** 1.0 · **Date:** 2025-11-23 · **Status:** Planning Phase
**Next Step:** Generate truth tables for all operations
