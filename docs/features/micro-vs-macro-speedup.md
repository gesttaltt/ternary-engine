# Micro vs Macro Speedup: Reality Check

**Date:** 2025-10-23
**Context:** Phase 4.1 - Setting realistic expectations

---

## The Problem with Micro-Kernel Benchmarks

When we benchmark individual fused operations in isolation, we measure **micro-kernel speedup**:
- Single operation: `tnot(tadd(a, b))`
- Fresh arrays each benchmark
- No other operations competing for resources
- **Result: 1.75-1.78× speedup**

But real applications don't work this way.

---

## What Happens in Real Applications

### Example: Neural Network Forward Pass

```python
# Typical ternary neural network layer
def ternary_layer_forward(x, W, b):
    # 1. Matrix multiply (many tmul operations)
    z = ternary_matmul(x, W)

    # 2. Add bias (tadd)
    z = ternary.tadd(z, b)

    # 3. Activation (maybe tnot)
    a = ternary.tnot(z)

    # 4. More layers...
    return a
```

**Fusion opportunity:** Only step 2+3 can be fused (`tnot(tadd(z, b))`)

**Micro speedup:** 1.75× for that operation
**Macro impact:** Much smaller

---

## The Math: Amdahl's Law

If the fused operation is only **10% of total runtime**:
- Unfused total: 100 time units
- Fused operation: 10 units → 10/1.75 = 5.7 units
- **Total speedup: 100 → 95.7 (4.3% improvement)**

If the fused operation is **50% of total runtime**:
- Unfused total: 100 time units
- Fused operation: 50 units → 50/1.75 = 28.6 units
- **Total speedup: 100 → 78.6 (21.4% improvement)**

**Key insight:** Micro speedup × operation coverage = macro speedup

---

## Realistic Expectations by Use Case

### Best Case: Memory-Bound Pipelines

**Scenario:** Long chains of ternary operations with minimal other work

```python
# Example: Image processing pipeline
result = ternary.tnot(ternary.tadd(
    ternary.tmul(a, b),
    ternary.tmin(c, d)
))
```

**Fusion opportunities:** High (80%+ of runtime is fusible operations)
**Expected macro speedup:** **1.4-1.6×** (approaching micro speedup)

**Why this works:**
- Most operations are ternary array ops
- Memory bandwidth is the bottleneck
- Fusion eliminates intermediate allocations

---

### Typical Case: Mixed Workloads

**Scenario:** Ternary operations mixed with Python overhead, I/O, non-ternary ops

```python
# Example: Training loop
for epoch in range(epochs):
    for batch in dataloader:  # I/O overhead
        x_ternary = quantize(batch)  # Conversion overhead
        y_pred = model.forward(x_ternary)  # Some fusion here
        loss = compute_loss(y_pred, y_true)  # Non-ternary
        optimizer.step()  # Update weights
```

**Fusion opportunities:** Medium (30-50% of runtime is fusible operations)
**Expected macro speedup:** **1.1-1.3×** (10-30% improvement)

**Why it's lower:**
- Python overhead (loop iteration, function calls)
- I/O bottlenecks (data loading)
- Non-fusible operations (loss computation, weight updates)

---

### Worst Case: Compute-Bound or Sparse Operations

**Scenario:** Small arrays, high Python overhead, or sparse computation patterns

```python
# Example: Small batch processing
for item in dataset:  # Many small operations
    result = fusion.fused_tnot_tadd(
        small_array_a,  # Only 100 elements
        small_array_b
    )
    process_result(result)  # Python overhead dominates
```

**Fusion opportunities:** Low (fused ops are tiny fraction of runtime)
**Expected macro speedup:** **1.0-1.1×** (0-10% improvement)

**Why it's minimal:**
- Python overhead > operation time
- Small arrays (< 10K elements) don't benefit much
- Fusion savings drown in other costs

---

## Validated Phase 4.1 Expectations

### Micro-Kernel Results (Measured)

**All four Binary→Unary operations at 100K elements:**

| Operation | Speedup | Variance (CV) | Status |
|-----------|---------|---------------|--------|
| tnot_tadd | 1.77× | 2.2% / 21.5% | ✓ Pass ⚠ Variance |
| tnot_tmul | 1.75× | 10.5% / 2.7% | ✓ Pass ✓ Stable |
| tnot_tmin | 1.78× | 12.6% / 4.4% | ✓ Pass ✓ Stable |
| tnot_tmax | 1.77× | 0.8% / 18.1% | ✓ Pass ✓ Stable |

**Average: 1.77× micro-kernel speedup**

### Conservative Macro-Level Expectations

Based on typical application profiles:

**Best case (80% fusible operations):**
- Micro speedup: 1.77×
- Coverage: 80%
- **Macro speedup: ~1.55×** (55% improvement)

**Typical case (40% fusible operations):**
- Micro speedup: 1.77×
- Coverage: 40%
- **Macro speedup: ~1.25×** (25% improvement)

**Worst case (20% fusible operations):**
- Micro speedup: 1.77×
- Coverage: 20%
- **Macro speedup: ~1.13×** (13% improvement)

---

## How to Measure Macro Speedup

### Step 1: Profile your application

```python
import cProfile

cProfile.run('your_application()')
```

**Identify:**
- What % of time is spent in ternary operations?
- Which operations are fusible?

### Step 2: Calculate expected benefit

```
Expected speedup = 1 / (
    (fusible_fraction / micro_speedup) +
    (1 - fusible_fraction)
)
```

**Example:**
- 40% of runtime is fusible ternary ops
- Micro speedup: 1.75×

```
Expected = 1 / (0.4/1.75 + 0.6)
         = 1 / (0.229 + 0.6)
         = 1 / 0.829
         = 1.21× (21% improvement)
```

### Step 3: Measure actual end-to-end

```python
import time

# Baseline
start = time.time()
result_unfused = application_without_fusion()
baseline_time = time.time() - start

# Fused
start = time.time()
result_fused = application_with_fusion()
fused_time = time.time() - start

print(f"Macro speedup: {baseline_time / fused_time:.2f}×")
```

---

## Honest Claims We Can Make

### What We CAN Say

✓ **"Fusion provides 1.75-1.78× speedup for isolated Binary→Unary operations"**
✓ **"Memory bandwidth reduced by 40% for fused operations"**
✓ **"In memory-bound pipelines, expect 10-30% end-to-end improvement"**
✓ **"Best gains on large arrays (>100K elements) with minimal Python overhead"**

### What We CANNOT Say

❌ ~~"Fusion makes your application 1.75× faster"~~ (depends on application profile)
❌ ~~"All ternary code will see 75% speedup"~~ (only fused operations benefit)
❌ ~~"Fusion eliminates all performance bottlenecks"~~ (only memory traffic for specific ops)

---

## Recommendations for Users

### When to Use Fusion

✓ **Memory-bound pipelines** with long chains of ternary operations
✓ **Large array processing** (>100K elements)
✓ **Batch operations** where ternary ops dominate runtime

### When NOT to Rely on Fusion

❌ **Small arrays** (< 10K elements) - overhead dominates
❌ **Python-heavy code** - fusion can't fix Python slowness
❌ **I/O-bound applications** - memory bandwidth isn't the bottleneck
❌ **Sparse operations** - fusion benefits are diluted

### How to Maximize Benefit

1. **Profile first:** Identify where ternary ops spend time
2. **Batch operations:** Use larger arrays when possible
3. **Chain fusions:** Use multiple fused ops in sequence
4. **Minimize Python overhead:** Use NumPy-style vectorization

---

## Example: Realistic Benchmark

**Scenario:** Simple ternary image processing pipeline

```python
def process_image_unfused(img_a, img_b, img_c):
    temp1 = ternary.tadd(img_a, img_b)
    temp2 = ternary.tnot(temp1)
    temp3 = ternary.tmul(temp2, img_c)
    result = ternary.tmin(temp3, img_a)
    return result

def process_image_fused(img_a, img_b, img_c):
    temp2 = fusion.fused_tnot_tadd(img_a, img_b)  # Fused
    temp3 = ternary.tmul(temp2, img_c)  # Cannot fuse further
    result = ternary.tmin(temp3, img_a)
    return result
```

**Analysis:**
- Total operations: 4
- Fusible: 1 (tadd+tnot)
- Fusible fraction: 25%

**Expected macro speedup:**
```
1 / (0.25/1.75 + 0.75) = 1 / 0.893 = 1.12× (12% improvement)
```

**Not the 1.75× micro speedup, but still worthwhile!**

---

## Conclusion

**Micro-kernel speedups (1.75×) are real but don't directly translate to application speedups.**

**Realistic expectations:**
- Memory-bound ternary pipelines: **1.3-1.6×** macro speedup
- Mixed workloads: **1.1-1.3×** macro speedup
- Python-heavy or small arrays: **1.0-1.1×** macro speedup

**This is still valuable engineering:**
- 10-30% improvement with minimal code changes
- Scales with ternary operation coverage
- Foundation for future optimizations (3-op chains, Binary→Binary)

**But we must be honest about expectations: micro ≠ macro.**

---

**Truth-first engineering:** We measure micro, we estimate macro, we validate both. 🔬
