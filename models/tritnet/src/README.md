# TritNet Training Scripts

**Purpose:** Train neural networks with ternary weights {-1, 0, +1} to learn exact balanced ternary arithmetic operations.

**Status:** Phase 2A - Proof-of-Concept (tnot operation)

---

## Quick Start

```bash
# 1. Generate truth tables (236,439 training samples)
python models/tritnet/src/generate_truth_tables.py

# 2. Train tnot model (proof-of-concept)
python models/tritnet/src/train_tritnet.py --operation tnot

# 3. Or train all operations
python models/tritnet/src/train_tritnet.py --all
```

---

## What is TritNet?

**TritNet** is a tiny neural network with pure ternary weights {-1, 0, +1} that learns exact balanced ternary arithmetic by training on complete truth tables.

### Research Question

Can gradient descent discover exact arithmetic functions using only ternary parameters?

**If yes:** Enables hardware-accelerated ternary computing via matmul instead of lookup tables

**If no:** Documents fundamental limits of learned arithmetic (publishable negative result)

### Architecture

```
Input: 10 trits {-1, 0, +1} (5 from A, 5 from B)
  ↓
Hidden Layer 1: 16 neurons, ternary weights W1 ∈ {-1, 0, +1}
  Activation: sign(x)
  ↓
Hidden Layer 2: 16 neurons, ternary weights W2 ∈ {-1, 0, +1}
  Activation: sign(x)
  ↓
Output Layer: 5 neurons, ternary weights W3 ∈ {-1, 0, +1}
  Activation: sign(x)
  ↓
Output: 5 trits {-1, 0, +1}
```

**Total parameters:** 496 ternary weights (vs 243-entry LUT)

---

## Scripts

### 1. generate_truth_tables.py

**Purpose:** Generate complete truth tables for all ternary operations

**Usage:**
```bash
python models/tritnet/src/generate_truth_tables.py [--output-dir DIR]
```

**Outputs:**
- `models/datasets/tritnet/tadd_truth_table.json` - Addition (59,049 samples)
- `models/datasets/tritnet/tmul_truth_table.json` - Multiplication (59,049 samples)
- `models/datasets/tritnet/tmin_truth_table.json` - Minimum (59,049 samples)
- `models/datasets/tritnet/tmax_truth_table.json` - Maximum (59,049 samples)
- `models/datasets/tritnet/tnot_truth_table.json` - Negation (243 samples)
- `models/datasets/tritnet/generation_summary.json` - Statistics

**Total:** 236,439 samples, 78.33 MB

**Coverage:** 100% of all valid dense243 state combinations

**Validation:** 100 random samples verified per operation before generation

### 2. train_tritnet.py

**Purpose:** Train TritNet models on ternary operations

**Usage:**
```bash
# Train single operation
python models/tritnet/src/train_tritnet.py --operation tnot --hidden-size 8

# Train all operations
python models/tritnet/src/train_tritnet.py --all

# Custom hyperparameters
python models/tritnet/src/train_tritnet.py \
  --operation tadd \
  --hidden-size 16 \
  --learning-rate 0.001 \
  --max-epochs 2000
```

**Arguments:**
- `--operation` - Operation to train (tnot, tadd, tmul, tmin, tmax)
- `--all` - Train all operations
- `--hidden-size` - Hidden layer neurons (default: 8 for unary, 16 for binary)
- `--learning-rate` - Adam learning rate (default: 0.001)
- `--max-epochs` - Maximum epochs (default: 2000)
- `--threshold` - Ternary quantization threshold (default: 0.5)
- `--seed` - Random seed for reproducibility (default: 42)

**Outputs:**
- `models/tritnet/tritnet_<op>.tritnet` - Trained model
- `models/tritnet/tritnet_<op>_history.json` - Training metrics

**Success criteria:**
- 100% exact match accuracy on all samples
- Convergence within 1000 epochs
- Reproducible results (deterministic seed)

### 3. ternary_layers.py

**Purpose:** PyTorch layers with ternary weight quantization

**Key components:**
- `TernaryLinear` - Linear layer with ternary weights
- `TernaryActivation` - Sign activation for ternary outputs
- `StraightThroughEstimator` - Gradient flow through quantization

**Usage:**
```python
from ternary_layers import TernaryLinear, ternary_sign

layer = TernaryLinear(10, 16, threshold=0.5)
x = torch.randn(batch_size, 10)
y = ternary_sign(layer(x))  # Ternary outputs {-1, 0, +1}
```

**Quantization method:**
```python
w_ternary = sign(w) if |w| > threshold else 0
```

**Gradient flow:** Straight-through estimator (STE) passes gradients to full-precision weights

### 4. tritnet_model.py

**Purpose:** TritNet model definitions and serialization

**Models:**
- `TritNetUnary` - For tnot (5 inputs → 5 outputs)
- `TritNetBinary` - For tadd, tmul, tmin, tmax (10 inputs → 5 outputs)

**Usage:**
```python
from tritnet_model import TritNetUnary, save_tritnet_model, load_tritnet_model

# Create and train
model = TritNetUnary(hidden_size=8)
# ... training loop ...

# Save model
save_tritnet_model(model, "models/tritnet/my_model.tritnet", metadata={...})

# Load model
loaded_model, metadata = load_tritnet_model("models/tritnet/my_model.tritnet")
```

---

## Training Strategy

### Phase 2A: Proof-of-Concept (tnot)

**Goal:** Prove exact arithmetic is learnable

**Why tnot first?**
- Smallest dataset (243 samples vs 59,049)
- Simplest operation (unary, no interaction)
- Fastest training (<5 minutes)
- Clear Go/No-Go decision

**Expected outcome:**
- ✅ 100% accuracy → Proceed to binary operations
- ❌ <99% accuracy → Investigate architecture, pivot if needed

### Phase 2B: Scale to All Operations

**Goal:** Train remaining operations (tadd, tmul, tmin, tmax)

**Success criteria:**
- ≥3/5 operations achieve >99% accuracy
- ≥1/5 operations achieve 100% accuracy
- Convergence within 1000 epochs per operation

**Go/No-Go decision:**
- **GO:** Criteria met → Proceed to Phase 3 (C++ integration)
- **NO-GO:** Criteria not met → Publish research on limits of learned arithmetic
- **PIVOT:** Partial success → Explore approximate arithmetic research

---

## Framework & Implementation

### Custom PyTorch Quantization (Not BitNet)

**Decision:** Use custom ternary quantization instead of BitNet framework

**Rationale:**
1. Full control over ternary semantics {-1, 0, +1}
2. No external dependencies beyond PyTorch
3. Standard straight-through estimator (STE)
4. Easy to verify and debug
5. Proven approach in quantized NN literature

**BitNet considered but rejected:**
- Designed for 1-bit {-1, +1} or 1.58-bit, not pure ternary
- Inference-only (no training support in bitnet.cpp)
- More complex integration
- Potential version compatibility issues

### Ternary Quantization Method

**Forward pass:**
```python
w_ternary = sign(w) * (abs(w) > threshold).float()
```

**Backward pass (STE):**
```python
grad_w = grad_w_ternary  # Pass gradients straight through
```

**Training loop:**
1. Weights stored as full-precision floats
2. Quantized to {-1, 0, +1} during forward pass
3. Gradients flow to full-precision weights via STE
4. Standard optimizers (Adam, SGD) update full-precision weights

---

## Hyperparameters

### Network Architecture

**Unary operations (tnot):**
- Input size: 5 trits
- Hidden layer 1: 8 neurons
- Hidden layer 2: 8 neurons
- Output size: 5 trits
- Total parameters: 144 ternary weights

**Binary operations (tadd, tmul, tmin, tmax):**
- Input size: 10 trits (5+5)
- Hidden layer 1: 16 neurons
- Hidden layer 2: 16 neurons
- Output size: 5 trits
- Total parameters: 496 ternary weights

### Training Configuration

**Optimizer:** Adam
- Learning rate: 0.001
- Betas: (0.9, 0.999)
- Weight decay: 0 (no L2 regularization initially)

**Loss function:** MSE (Mean Squared Error)
- Alternative: Cross-entropy (per-trit classification)
- Monitoring: Exact match accuracy (all 5 trits correct)

**Batch size:** Full batch (243 or 59,049 samples)
- Small enough to fit in memory
- No mini-batching needed

**Early stopping:** 100% accuracy reached or 2000 epochs

**Initialization:** Normal(0, 0.1)
- Small values to encourage ternary quantization

---

## Expected Results

### Performance Predictions

**LUT Baseline:**
- Latency: ~2 ns/operation
- Throughput: ~500 Mops/sec (memory-bound)
- Size: 243 bytes per operation

**TritNet (CPU, Phase 2-3):**
- Latency: ~50 ns/operation (240 ternary multiplies)
- Throughput: ~20 Mops/sec (compute-bound)
- Size: ~500 bytes (496 ternary weights)

**TritNet (GPU, future):**
- Latency: ~5 ns/operation (batched)
- Throughput: ~2000 Mops/sec (batched matmul)
- Size: ~500 bytes

**Conclusion:** CPU likely slower, GPU potentially 4× faster with batching

### Success Metrics

**Must achieve (Phase 2A):**
- ✅ 100% accuracy on tnot (all 243 samples)
- ✅ Convergence within 1000 epochs
- ✅ Training completes in <5 minutes
- ✅ Reproducible with deterministic seed

**Nice to have:**
- ✅ Weight sparsity >30% (many zeros)
- ✅ Model size <150 parameters
- ✅ Interpretable weight patterns
- ✅ Converges in <500 epochs

---

## Outputs

### Trained Models

**Format:** `.tritnet` PyTorch checkpoint

**Contents:**
- Full model state dict (full-precision weights)
- Quantized ternary weights (for inspection)
- Model configuration (architecture, hyperparameters)
- Training metadata (accuracy, epochs, time)

**Size:** ~10 KB per model (uncompressed)

**Location:** `models/tritnet/`

### Training History

**Format:** JSON file

**Contents:**
```json
{
  "metadata": {
    "operation": "tnot",
    "final_accuracy": 1.0,
    "best_accuracy": 1.0,
    "epochs_trained": 487,
    "training_time_seconds": 143.2,
    "hidden_size": 8,
    "ternary_counts": {
      "minus_one": 45,
      "zero": 54,
      "plus_one": 45
    }
  },
  "history": [
    {"epoch": 0, "loss": 0.842, "accuracy": 0.12},
    {"epoch": 1, "loss": 0.735, "accuracy": 0.23},
    ...
  ]
}
```

**Location:** `models/tritnet/tritnet_<op>_history.json`

---

## Dependencies

### Required

```bash
pip install torch numpy
```

**Versions:**
- Python: 3.7+
- PyTorch: 2.0+ (or 1.13+)
- NumPy: 1.19+

### Optional (for analysis)

```bash
pip install matplotlib seaborn pandas scikit-learn
```

**No CUDA required** - CPU-only training sufficient for Phase 2

---

## Next Steps

### Immediate (Phase 2A)

1. ✅ Framework selected (custom PyTorch quantization)
2. ✅ Architecture designed (2 hidden layers)
3. ✅ Scripts implemented (ternary_layers, tritnet_model, train_tritnet)
4. 🔄 Train tnot model
5. ⏳ Validate 100% accuracy
6. ⏳ Make Go/No-Go decision

### If Go (Phase 2B)

1. Train remaining operations (tadd, tmul, tmin, tmax)
2. Analyze results across operations
3. Document learned weight patterns
4. Prepare for Phase 3 (C++ integration)

### If No-Go

1. Investigate why NNs cannot learn exact arithmetic
2. Test approximate arithmetic (95-99% accuracy)
3. Analyze failure modes and misclassifications
4. Publish research on limits of learned arithmetic

---

## Troubleshooting

### Training doesn't converge

**Solutions:**
1. Increase network size (16 → 32 hidden neurons)
2. Add more layers (2 → 3 hidden layers)
3. Try different initialization (Xavier, Kaiming)
4. Reduce learning rate (0.001 → 0.0001)
5. Use SGD instead of Adam

### Accuracy plateaus at 95-99%

**Analysis:**
1. Plot training curves (loss, accuracy over time)
2. Examine misclassified samples (edge cases?)
3. Check weight distribution (stuck in local minimum?)
4. Try curriculum learning (easy → hard samples)

### Training is slow

**Solutions:**
1. Reduce network size (if accuracy permits)
2. Increase learning rate
3. Use learning rate scheduler
4. Implement early stopping

**Note:** With only 243-59K samples, training should be <5 min per model

---

## Related Documentation

- **Vision & Goals:** `docs/TRITNET_VISION.md`
- **Implementation Roadmap:** `docs/TRITNET_ROADMAP.md`
- **Dataset Documentation:** `models/datasets/tritnet/README.md`
- **Dense243 Module:** `src/engine/experimental/dense243/README.md`
- **Scripts Overview:** `scripts/README.md`

---

**Version:** 1.0 · **Updated:** 2025-11-23 · **Phase:** 2A (Proof-of-Concept)
**Next:** Train tnot model and validate 100% accuracy
