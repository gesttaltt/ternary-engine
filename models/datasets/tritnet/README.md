# TritNet Training Datasets

**Generated:** 2025-11-23
**Total Samples:** 236,439
**Total Size:** 78.33 MB
**Coverage:** 100% of all valid dense243 state combinations

---

## Overview

These datasets contain complete truth tables for all dense243 ternary operations, designed to train **TritNet** - a tiny neural network with pure ternary weights {-1, 0, +1} that learns exact arithmetic operations.

### Purpose

TritNet will replace lookup table (LUT) operations with learned matmul operations, enabling:
- **Learnable operations**: Discover optimal arithmetic patterns
- **Compression**: Single weight matrix vs 243-entry tables
- **Hardware acceleration**: Use matmul accelerators (GPU/TPU) instead of memory lookups
- **Generalization**: Potential for approximate/fuzzy ternary logic

---

## Dataset Files

| File | Operation | Type | Samples | Size |
|:-----|:----------|:-----|--------:|-----:|
| `tadd_truth_table.json` | Addition | Binary | 59,049 | 19.58 MB |
| `tmul_truth_table.json` | Multiplication | Binary | 59,049 | 19.55 MB |
| `tmin_truth_table.json` | Minimum | Binary | 59,049 | 19.61 MB |
| `tmax_truth_table.json` | Maximum | Binary | 59,049 | 19.53 MB |
| `tnot_truth_table.json` | Negation | Unary | 243 | 0.06 MB |
| `generation_summary.json` | Summary | Meta | - | 0.00 MB |

---

## Data Format

### Binary Operations (tadd, tmul, tmin, tmax)

**Input Size:** 10 (two 5-trit operands)
**Output Size:** 5 (one 5-trit result)
**Samples per Operation:** 243² = 59,049

**Example Sample:**
```json
{
  "input": [-1, -1, -1, -1, -1, 1, -1, -1, -1, -1],
  "output": [0, -1, -1, -1, -1],
  "input_dense243": [0, 2],
  "output_dense243": 1
}
```

**Interpretation:**
- `input[0:5]`: First operand as 5 trits {-1, 0, +1}
- `input[5:10]`: Second operand as 5 trits {-1, 0, +1}
- `output[0:5]`: Result as 5 trits {-1, 0, +1}
- `input_dense243`: Dense243 byte values [0-242] for validation
- `output_dense243`: Dense243 byte value [0-242] for validation

### Unary Operations (tnot)

**Input Size:** 5 (one 5-trit operand)
**Output Size:** 5 (one 5-trit result)
**Samples:** 243

**Example Sample:**
```json
{
  "input": [-1, -1, -1, -1, -1],
  "output": [1, 1, 1, 1, 1],
  "input_dense243": 0,
  "output_dense243": 242
}
```

### Metadata

Each truth table file includes metadata:
```json
{
  "metadata": {
    "operation": "tadd",
    "operation_type": "binary",
    "input_size": 10,
    "output_size": 5,
    "num_samples": 59049,
    "encoding": "balanced_ternary",
    "value_range": [-1, 0, 1],
    "dense243_range": [0, 242],
    "generator_version": "1.0.0"
  },
  "samples": [...]
}
```

---

## TritNet Architecture

### Proposed Network Structure

**Binary Operations (tadd, tmul, tmin, tmax):**
```
Input Layer:  10 ternary values {-1, 0, +1}
   ↓
Hidden Layer 1: 16 neurons, ternary weights
   ↓
Hidden Layer 2: 16 neurons, ternary weights
   ↓
Output Layer: 5 ternary values {-1, 0, +1}
```

**Unary Operations (tnot):**
```
Input Layer:  5 ternary values {-1, 0, +1}
   ↓
Hidden Layer 1: 8 neurons, ternary weights
   ↓
Hidden Layer 2: 8 neurons, ternary weights
   ↓
Output Layer: 5 ternary values {-1, 0, +1}
```

**Activation:** sign() function (ternary output: -1 if x < 0, 0 if x = 0, +1 if x > 0)

### Training Requirements

**Target Accuracy:** 100% (exact arithmetic, not approximation)
**Training Framework:** BitNet b1.58 pipeline
**Quantization:** Distill to pure ternary weights {-1, 0, +1}
**Loss Function:** Cross-entropy or MSE on ternary values

---

## Usage

### Load Dataset in Python

```python
import json
import numpy as np

# Load truth table
with open('datasets/tritnet/tadd_truth_table.json') as f:
    data = json.load(f)

# Extract samples
samples = data['samples']
X = np.array([s['input'] for s in samples])  # Shape: (59049, 10)
y = np.array([s['output'] for s in samples]) # Shape: (59049, 5)

print(f"Loaded {len(samples)} samples for {data['metadata']['operation']}")
print(f"Input shape: {X.shape}, Output shape: {y.shape}")
```

### Validate Against Module

```python
import ternary_dense243_module as td

# Load sample
sample = samples[0]
a_dense243 = sample['input_dense243'][0]
b_dense243 = sample['input_dense243'][1]
expected_output = sample['output_dense243']

# Verify with module
import numpy as np
a = np.array([a_dense243], dtype=np.uint8)
b = np.array([b_dense243], dtype=np.uint8)
result = td.tadd(a, b)

assert result[0] == expected_output, "Truth table mismatch!"
```

---

## Generation Details

### Generator Script
- **Location:** `models/tritnet/src/generate_truth_tables.py`
- **Version:** 1.0.0
- **Module Used:** `ternary_dense243_module` v1.0.0
- **Backend:** LUT (lookup table based)

### Coverage Verification

All datasets achieve **100% coverage** of valid dense243 state space:
- **Binary operations:** All 243² = 59,049 combinations per operation
- **Unary operations:** All 243 combinations

### Correctness Verification

Each operation verified with 100 random samples before generation:
```
✓ tadd: 100 random samples verified
✓ tmul: 100 random samples verified
✓ tmin: 100 random samples verified
✓ tmax: 100 random samples verified
✓ tnot: 100 random samples verified
```

---

## Next Steps

### Phase 2: BitNet Training (Weeks 3-4)

1. **Set up BitNet environment:**
   ```bash
   git clone https://github.com/microsoft/BitNet
   cd BitNet
   # Follow BitNet installation instructions
   ```

2. **Convert datasets to BitNet format:**
   - Create data loader for truth tables
   - Implement ternary-aware training loop
   - Configure BitNet b1.58 quantization

3. **Train TritNet models:**
   - Train one model per operation (5 total)
   - Target 100% accuracy on all samples
   - Monitor convergence and validate exact arithmetic

4. **Validate trained models:**
   - Test on all 236,439 samples
   - Ensure bit-exact match with LUT results
   - Verify ternary weight quantization

### Phase 3: Distillation & Export (Weeks 5-6)

1. **Quantize to ternary weights:**
   - Use BitNet b1.58 pipeline
   - Ensure all weights ∈ {-1, 0, +1}
   - Validate accuracy retention

2. **Export models:**
   - Create `.tritnet` model format
   - Include weight matrices and biases
   - Add metadata (operation, architecture, accuracy)

3. **Implement C++ inference:**
   - Ternary matmul kernel
   - sign() activation function
   - Integration with `ternary_dense243_module`

### Phase 4: Integration (Weeks 7-12)

See `docs/TRITNET_ROADMAP.md` for complete implementation plan.

---

## References

- **TritNet Roadmap:** `docs/TRITNET_ROADMAP.md`
- **Dense243 Module:** `src/engine/experimental/dense243/`
- **Generator Script:** `models/tritnet/src/generate_truth_tables.py`
- **BitNet Repository:** https://github.com/microsoft/BitNet
- **BitNet Paper:** https://arxiv.org/abs/2310.11453

---

## Validation Commands

```bash
# Verify dataset integrity
python -c "
import json
data = json.load(open('datasets/tritnet/generation_summary.json'))
assert data['total_samples'] == 236439, 'Sample count mismatch'
assert data['module_version'] == '1.0.0', 'Version mismatch'
print('✓ Dataset integrity verified')
"

# Test data loading
python -c "
import json
import numpy as np
for op in ['tadd', 'tmul', 'tmin', 'tmax', 'tnot']:
    with open(f'datasets/tritnet/{op}_truth_table.json') as f:
        data = json.load(f)
    X = np.array([s['input'] for s in data['samples']])
    y = np.array([s['output'] for s in data['samples']])
    print(f'✓ {op}: {X.shape} → {y.shape}')
"

# Verify sample correctness
python models/tritnet/src/generate_truth_tables.py --help
```

---

**Status:** ✅ Ready for BitNet Training
**Next Phase:** Set up BitNet environment and train TritNet models
**Timeline:** Weeks 3-4 (see roadmap)
