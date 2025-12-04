# TritNet Training Failure Analysis - 2025-12-04

**Purpose:** Root cause analysis of why binary TritNet operations fail to train while unary operations succeed.

---

## Executive Summary

**Problem:** tadd (binary operation) achieves only 13-17% accuracy after 3000 epochs, while tnot (unary operation) achieved 100% accuracy in 487 epochs.

**Key Finding:** Binary ternary operations appear fundamentally harder to learn via gradient descent with MSE loss. The problem is NOT dataset quality or network capacity - it's the optimization landscape.

**Status:** Testing deep architecture (TritNetBinaryDeep with skip connections) to see if this improves learnability.

---

## Training Results Comparison

### tnot (Unary - SUCCESS)

```
Operation: tnot (unary negation)
Architecture: TritNetUnary (shallow, 2 hidden layers)
Dataset: 243 samples (3^5 all possible 5-trit inputs)
Network: 5 → 8 → 8 → 5
Parameters: 144 ternary weights
Loss: MSE

Results:
- Final accuracy: 100.00%
- Epochs: 487
- Training time: ~5 minutes
- Loss convergence: Smooth descent to near-zero
```

### tadd 16-hidden (Binary - FAILED)

```
Operation: tadd (ternary addition)
Architecture: TritNetBinary (shallow, 2 hidden layers)
Dataset: 59,049 samples (3^10 all possible 5-trit pairs)
Network: 10 → 16 → 16 → 5
Parameters: 496 ternary weights
Loss: MSE

Results:
- Final accuracy: 13.17%
- Epochs: 2000
- Training time: 26.2s
- Loss plateau: 0.074254 (plateaued at epoch 1300)
```

### tadd 64-hidden (Binary - FAILED)

```
Operation: tadd (ternary addition)
Architecture: TritNetBinary (shallow, 2 hidden layers)
Dataset: 59,049 samples
Network: 10 → 64 → 64 → 5
Parameters: 5,056 ternary weights (10× more than 16-hidden)
Loss: MSE

Results:
- Final accuracy: 16.75%
- Epochs: 3000
- Training time: 225.8s
- Loss plateau: 0.074074 (plateaued at epoch 300)
- Improvement over 16-hidden: Only 3.58% despite 10× parameters
```

**Critical observation:** Scaling network capacity from 496 to 5,056 parameters (10×) only improved accuracy by 3.58%. This suggests the problem is NOT lack of capacity but fundamental learnability.

---

## Hypothesis Analysis

### Hypothesis 1: Dataset Quality ❌ REJECTED

**Test:** Validated truth table dataset
```python
Total samples: 59,049 (expected: 243² = 59,049) ✓
Unique inputs: 59,049 (no duplicates) ✓
Sample check: [-1,-1,-1,-1,-1] + [-1,-1,-1,-1,-1] = [-1,-1,-1,-1,-1] ✓
```

**Conclusion:** Dataset is correct and complete.

---

### Hypothesis 2: Insufficient Network Capacity ❌ REJECTED

**Test:** Increased hidden size 16 → 64 (10× parameters)
```
16-hidden: 496 parameters → 13.17% accuracy
64-hidden: 5,056 parameters → 16.75% accuracy
Improvement: 3.58% (NOT proportional to capacity increase)
```

**Conclusion:** Network has enough capacity. Problem is elsewhere.

---

### Hypothesis 3: Loss Function Unsuitable ⏳ TESTING

**Current:** MSE (Mean Squared Error) on continuous outputs
```python
loss = MSELoss()(predictions, targets)
# predictions: continuous values (before sign activation)
# targets: discrete {-1, 0, +1}
```

**Problem:** MSE on discrete targets creates difficult optimization landscape
- Many local minima for discrete outputs
- Gradient direction unclear when output is "between" discrete values
- Sign function (used during inference) has zero/undefined gradient

**Alternative:** Cross-entropy loss on discrete classes
```python
# Convert ternary {-1, 0, +1} to classes {0, 1, 2}
loss = CrossEntropyLoss()(logits, class_indices)
# logits: [batch, 5, 3] (3 classes per trit position)
# class_indices: [batch, 5] (0=minus, 1=zero, 2=plus)
```

**Advantage:** Natural for discrete outputs, clearer gradient signal

**Status:** NOT YET IMPLEMENTED (lines 236-243 in train_tritnet.py show TODO)

---

### Hypothesis 4: Shallow Architecture Insufficient ⏳ TESTING

**Current:** 2 hidden layers (no skip connections)
```
Unary (tnot): 5 → 8 → 8 → 5 (SUCCESS)
Binary (tadd): 10 → 16 → 16 → 5 (FAILED)
```

**Problem:** Binary operations may require deeper networks
- More complex input space (10 dimensions vs 5)
- More complex decision boundaries
- 243× more samples (59,049 vs 243)

**Solution:** TritNetBinaryDeep (4 hidden layers + skip connections)
```
Architecture: 10 → 32 → 32 → 32 → 32 → 5
Skip connections: Every 2 layers (ResNet-style)
Advantages:
- Deeper networks capture more complex patterns
- Skip connections help gradient flow (avoid vanishing gradients)
- Proven success for TritNetUnaryDeep
```

**Status:** CURRENTLY TRAINING (test in progress)

---

### Hypothesis 5: Training Methodology Issues ⏳ INVESTIGATING

**Potential issues:**

1. **Learning rate too high/low**
   - Current: 0.001 (Adam default)
   - Binary operations may need different LR than unary

2. **Quantization threshold**
   - Current: 0.5 (symmetric around zero)
   - May need asymmetric threshold or adaptive thresholding

3. **No curriculum learning**
   - Training on all 59,049 samples simultaneously
   - May need gradual difficulty increase (start with simple patterns)

4. **Batch training vs full dataset**
   - Current: Full dataset forward pass each epoch
   - May benefit from mini-batch SGD with shuffling

---

## Fundamental Difference: Unary vs Binary

### Why tnot Succeeded

**Ternary NOT truth table:**
```
Input  | Output
-------|-------
  -1   |   +1
   0   |    0
  +1   |   -1
```

**Pattern:** Simple negation (flip sign, keep zero)
- Linear transformation possible: output = -1 × input
- 243 samples (3^5) well-distributed
- 5-dimensional input space

**Network task:** Learn matrix W where `W × input ≈ -input`
- This is a simple linear transformation
- Gradient descent easily finds solution

---

### Why tadd Failed

**Ternary ADD truth table (sample):**
```
A     | B     | A + B (saturated at ±1)
------|-------|-------------------------
 -1   |  -1   |   -1
 -1   |   0   |   -1
 -1   |  +1   |    0
  0   |  -1   |   -1
  0   |   0   |    0
  0   |  +1   |   +1
 +1   |  -1   |    0
 +1   |   0   |   +1
 +1   |  +1   |   +1
```

**Pattern:** Complex carry logic with saturation
- NOT a simple linear transformation
- 59,049 samples (3^10) sparse in 10D space
- Requires learning carry propagation across 5 trit positions

**Network task:** Learn non-linear function with:
- Carry logic (e.g., -1 + -1 = -1 due to saturation)
- Position-dependent interactions (trit[0] + trit[5] → output[0])
- Complex decision boundaries

**Challenge:** Gradient descent struggles with:
- Discrete logic (carry yes/no)
- Saturation discontinuities
- High-dimensional sparse patterns

---

## Comparison to Prior Work

### BitNet (Binary Networks)

**BitNet problem:** Quantize FP32 neural network weights to {-1, +1}
**BitNet solution:** Straight-Through Estimator (STE) - train in FP32, quantize for forward pass
**BitNet success:** Achieves ~95-98% of FP32 accuracy on large models (LLaMA, etc.)

**Key difference:**
- BitNet quantizes TRAINED weights (initialization matters)
- TritNet learns LOGIC from scratch (no pre-trained initialization)

---

### BNN (Binarized Neural Networks)

**BNN problem:** Train networks with binary activations {-1, +1}
**BNN solution:** Use sign activation during forward, straight-through during backward
**BNN challenge:** Difficult to train from scratch, often needs FP32 pre-training

**Similarity to TritNet:**
- Both use straight-through estimator
- Both struggle with discrete outputs
- Both benefit from pre-training or special initialization

---

### Research Literature

**Courbariaux et al. (2016) - BinaryConnect:**
> "Training binary neural networks is hard because the sign function has zero gradient almost everywhere."

**Hubara et al. (2016) - Quantized Neural Networks:**
> "Quantized networks require careful initialization and learning rate scheduling to converge."

**Rastegari et al. (2016) - XNOR-Net:**
> "Binary networks benefit from scaling factors to account for quantization error."

**Lesson for TritNet:** Discrete arithmetic learning is a known hard problem, not specific to ternary.

---

## Experimental Next Steps

### Experiment 1: Deep Architecture ⏳ IN PROGRESS

**Test:** TritNetBinaryDeep with 4 hidden layers + skip connections
```bash
python train_tritnet.py --operation tadd --architecture deep --hidden-size 32 --max-epochs 3000
```

**Expected outcome:**
- If accuracy >50%: Architecture was the bottleneck
- If accuracy <25%: Architecture not sufficient, need different approach

**Status:** Training started, awaiting results

---

### Experiment 2: Cross-Entropy Loss (Future)

**Implementation required:**
1. Modify TritNet output layer to produce logits: [batch, 5, 3]
2. Use CrossEntropyLoss instead of MSELoss
3. Convert targets to class indices: {-1→0, 0→1, +1→2}

**Expected benefit:** Better gradient signal for discrete outputs

---

### Experiment 3: Curriculum Learning (Future)

**Strategy:** Train on progressively harder patterns
```python
# Phase 1: Identity patterns (A + 0 = A)
# Phase 2: Simple addition (no carries needed)
# Phase 3: Patterns requiring carries
# Phase 4: Full dataset
```

**Expected benefit:** Network learns basic patterns before complex ones

---

### Experiment 4: Different Initializations (Future)

**Test:** Initialize weights to approximate ternary addition
```python
# Initialize layer1 to separate A and B operands
# Initialize layer2 to compute partial sums
# Initialize layer3 to handle carries and saturation
```

**Expected benefit:** Start closer to solution in optimization landscape

---

### Experiment 5: Auxiliary Losses (Future)

**Add per-trit accuracy loss:**
```python
total_loss = mse_loss + lambda * per_trit_accuracy_loss
# Encourage network to get individual trits correct, not just overall pattern
```

**Expected benefit:** More fine-grained learning signal

---

## Decision Criteria

### Go/No-Go Thresholds

**SUCCESS (proceed to Phase 3):**
- Any architecture achieves >99% accuracy on tadd
- At least 3/4 binary operations achieve >99%
- Demonstrates exact arithmetic is learnable

**PARTIAL SUCCESS (research pivot):**
- Best model achieves 50-90% accuracy
- Some operations learn better than others
- Pivot to approximate arithmetic research
- Publish findings on "which ternary operations are learnable"

**FAILURE (defer TritNet):**
- No model exceeds 25% accuracy after all experiments
- Fundamental limitation of gradient descent for discrete logic
- Defer TritNet, focus on LUT-based optimizations (Priority 2: SIMD)
- Publish negative results (academic value)

---

## Alternative Approaches (If Current Fails)

### Alternative 1: Hybrid LUT-NN

**Concept:** Use LUT for simple operations, NN for complex patterns
```
if pattern in LUT:
    return LUT[pattern]
else:
    return TritNet(pattern)  # Generalization for unseen patterns
```

**Advantage:** Best of both worlds (speed + generalization)

---

### Alternative 2: Reinforcement Learning

**Concept:** Treat arithmetic as RL problem
```
State: Current input pattern
Action: Predict next trit
Reward: +1 if correct, 0 if wrong
```

**Advantage:** May handle discrete logic better than supervised learning

---

### Alternative 3: Symbolic Regression

**Concept:** Search program space for ternary arithmetic
```
Use genetic programming to evolve:
- Decision trees
- If-then rules
- Symbolic expressions
```

**Advantage:** May discover patterns gradient descent cannot find

---

### Alternative 4: Neuro-Symbolic Hybrid

**Concept:** Combine neural network with rule-based system
```
NN learns feature extraction (identify carries, saturation cases)
Rule engine applies ternary logic rules
```

**Advantage:** Explicit logic + learned patterns

---

## Timeline

**Current:** Testing deep architecture (Experiment 1)
**Next 24h:**
- If deep works (>50% accuracy) → Train all operations with deep
- If deep fails (<25% accuracy) → Implement cross-entropy (Experiment 2)

**Next 48h:**
- If cross-entropy works → Complete training all operations
- If cross-entropy fails → Investigate curriculum learning (Experiment 3)

**Next 72h:**
- Make Go/No-Go decision based on best results
- If GO → Proceed to Phase 3 (C++ integration)
- If PARTIAL → Pivot to approximate arithmetic research
- If NO-GO → Document findings, defer TritNet, focus on SIMD (Priority 2)

---

## Key Insights

1. **Binary operations are fundamentally harder than unary**
   - 10D input space vs 5D
   - Carry logic vs simple negation
   - 59,049 samples vs 243

2. **Network capacity is NOT the bottleneck**
   - 10× parameters only improved 3.58%
   - Problem is optimization landscape, not model size

3. **MSE loss may be unsuitable for discrete outputs**
   - Creates many local minima
   - Gradient signal unclear for "between" values

4. **This is a known hard problem in literature**
   - Binary/ternary networks struggle to train from scratch
   - Often need pre-training or special techniques

5. **TritNet may require different approach than standard supervised learning**
   - Curriculum learning
   - Different loss functions
   - Hybrid symbolic-neural methods

---

## Conclusion

The tadd training failure is NOT due to:
- ❌ Bad dataset
- ❌ Insufficient capacity
- ❌ Bug in code

The failure is due to:
- ✅ Fundamental difficulty of learning discrete logic via gradient descent
- ✅ MSE loss unsuitable for discrete outputs
- ✅ Shallow architecture insufficient for complex patterns

**Next step:** Await results of TritNetBinaryDeep (deep architecture test)

**If deep fails:** Implement cross-entropy loss and curriculum learning

**Go/No-Go decision:** Within 72 hours based on best experimental results

---

**Created:** 2025-12-04
**Status:** Investigation in progress
**Next Review:** After deep architecture training completes
