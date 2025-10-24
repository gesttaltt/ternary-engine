# Macro Benchmarks: Real-World Application Performance

**Purpose:** Measure actual end-to-end speedup in realistic workloads, not isolated micro-kernels.

---

## Why Macro Benchmarks?

**Micro benchmarks** (in `benchmarks/micro/`) measure isolated operation speedup:
- `fused_tnot_tadd` vs `tnot(tadd)` on fresh arrays
- Result: **1.75-1.78× speedup** (validated)

**But real applications don't work this way:**
- Multiple operations in sequence
- Python overhead (loops, function calls)
- Memory allocation/deallocation
- Non-ternary operations (I/O, preprocessing)
- Mixed operation sizes

**Macro benchmarks measure what users actually care about:** End-to-end application speedup.

---

## Benchmark Scenarios

### 1. Image Processing Pipeline (`bench_image_pipeline.py`)

**Scenario:** Ternary image convolution + filters

```python
def process_image(img):
    # Convolution (many tmul operations)
    conv = convolve_ternary(img, kernel)

    # Bias add
    biased = ternary.tadd(conv, bias)

    # Activation (tnot)
    activated = ternary.tnot(biased)  # ← Fusion opportunity!

    # More filters...
    return result
```

**Fusion coverage:** ~20-30% of operations (tadd+tnot step)

**Expected macro speedup:** 1.10-1.15× (10-15% improvement)

---

### 2. Neural Network Layer (`bench_neural_layer.py`)

**Scenario:** Ternary matrix multiply + activation

```python
def ternary_layer_forward(x, W, b):
    # Matrix multiply (many operations)
    z = ternary_matmul(x, W)

    # Add bias + activation
    z = ternary.tadd(z, b)
    a = ternary.tnot(z)  # ← Fusion opportunity!

    return a
```

**Fusion coverage:** ~15-25% of operations

**Expected macro speedup:** 1.08-1.12× (8-12% improvement)

---

### 3. Batch Data Processing (`bench_batch_processing.py`)

**Scenario:** Large dataset transformations

```python
def process_batch(data_a, data_b):
    # Multiple ternary operations
    temp1 = ternary.tmul(data_a, data_b)
    temp2 = ternary.tmax(temp1, data_a)
    result = ternary.tnot(temp2)  # ← Could fuse tnot+tmax if we had it
    return result
```

**Fusion coverage:** Depends on operation patterns

**Expected macro speedup:** 1.05-1.20× (5-20% improvement)

---

### 4. Mixed Workload (`bench_mixed_workload.py`)

**Scenario:** Realistic combination with Python overhead

```python
for batch in dataloader:  # I/O overhead
    x = quantize_to_ternary(batch)  # Conversion

    # Ternary operations (some fusible)
    y = ternary_pipeline(x)

    # Post-processing
    result = dequantize(y)
    save(result)  # I/O
```

**Fusion coverage:** ~10-20% of total runtime

**Expected macro speedup:** 1.02-1.08× (2-8% improvement)

---

## Validation Criteria

### Success Metrics

**If macro speedup ≥ 1.10× (10% improvement):**
- ✓ Fusion is worthwhile
- → Deploy Phase 4.1 to production
- → Consider Phase 4.2/4.3 (more fusion patterns)

**If macro speedup = 1.05-1.10× (5-10% improvement):**
- ⚠ Modest gains
- → Deploy Phase 4.1 (still useful)
- → Phase 4.2/4.3 likely not worth engineering cost

**If macro speedup < 1.05× (< 5% improvement):**
- ✗ Fusion benefits too small
- → Re-evaluate: Maybe Python overhead dominates?
- → Phase 4.2/4.3 definitely not worth it

---

## Honest Expectations (Based on micro-vs-macro analysis)

**Micro speedup:** 1.77× (validated)

**Realistic macro estimates:**

| Workload Type | Fusible % | Expected Macro | Measured |
|---------------|-----------|----------------|----------|
| Image pipeline | 20-30% | 1.10-1.15× | TBD |
| Neural layer | 15-25% | 1.08-1.12× | TBD |
| Batch processing | 10-30% | 1.05-1.20× | TBD |
| Mixed workload | 10-20% | 1.02-1.08× | TBD |

**If measured < expected:** Python overhead or I/O dominates (fusion can't help)

**If measured ≈ expected:** Theory validated, fusion works as predicted

**If measured > expected:** Bonus! Cache effects or other synergies

---

## Decision Framework

```
IF avg_macro_speedup >= 1.10:
    VERDICT: "Deploy Phase 4.1, consider expanding"
    NEXT: Phase 4.2/4.3 investigation

ELIF avg_macro_speedup >= 1.05:
    VERDICT: "Deploy Phase 4.1, STOP fusion expansion"
    NEXT: Focus on other optimizations (lower abstractions, JIT)

ELSE:
    VERDICT: "Fusion ROI too low for deployment"
    NEXT: Investigate why (profiling, Python overhead analysis)
```

---

## Running Macro Benchmarks

```bash
# Run all macro benchmarks
python benchmarks/macro/bench_image_pipeline.py
python benchmarks/macro/bench_neural_layer.py
python benchmarks/macro/bench_batch_processing.py
python benchmarks/macro/bench_mixed_workload.py

# Or run the suite
python benchmarks/macro/run_all_macro_benchmarks.py
```

**Each benchmark reports:**
- Baseline time (unfused)
- Fused time (Phase 4.1 operations)
- Macro speedup (end-to-end)
- Comparison to micro predictions

---

## Truth-First Commitment

**We will:**
- ✓ Measure real workloads (not synthetic micro-kernels)
- ✓ Report actual speedups (with variance)
- ✓ Compare to theoretical predictions
- ✓ Make honest go/no-go decision on Phase 4.2/4.3

**We will NOT:**
- ❌ Cherry-pick favorable workloads
- ❌ Claim micro speedups are macro speedups
- ❌ Build more fusion if macro gains < 5%

**Goal:** Data-driven decision on whether to continue fusion expansion or deploy Phase 4.1 and move on.

---

**Micro benchmarks tell us what's possible. Macro benchmarks tell us what's real.** 🔬
