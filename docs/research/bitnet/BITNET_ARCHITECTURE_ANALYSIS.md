# BitNet Architecture Analysis

**Date:** 2025-11-23
**Analyzed Version:** bitnet.cpp v1.0 (Microsoft)
**Purpose:** Understand BitNet internals to design integration with our ternary engine

---

## Executive Summary

BitNet.cpp uses **Lookup Table (LUT) kernels** to accelerate ternary matrix multiplication. The approach:
- Precomputes weighted sums of activation combinations into LUTs
- Uses table indexing instead of direct multiply-accumulate
- Achieves 2-6× speedup on CPUs via reduced memory bandwidth

**Our opportunity:** Replace their generic LUT approach with our specialized Dense243-packed ternary operations for an additional 2-3× speedup.

---

## Repository Structure

```
BitNet/
├── src/
│   ├── ggml-bitnet-lut.cpp      # LUT kernel implementation (166 lines)
│   ├── ggml-bitnet-mad.cpp      # Multiply-add kernel (362 lines)
│   └── ggml.c                    # Based on llama.cpp GGML library
├── include/
│   └── ggml-bitnet.h             # Public API for BitNet kernels
├── preset_kernels/
│   ├── bitnet_b1_58-3B/
│   │   ├── bitnet-lut-kernels-tl1.h  # ARM NEON kernels
│   │   └── bitnet-lut-kernels-tl2.h  # x86 AVX2 kernels
│   └── Llama3-8B-1.58-100B-tokens/
│       └── ...
├── gpu/
│   ├── model.py                  # Python model definition
│   └── bitnet_kernels/          # CUDA kernels
└── utils/
    ├── codegen_tl1.py            # Generate ARM TL1 kernels
    └── codegen_tl2.py            # Generate x86 TL2 kernels
```

**Key insight:** Most kernel logic is CODE GENERATED, not hand-written. They generate specialized LUT kernels for each model size.

---

## Kernel Types

BitNet supports three kernel types:

### 1. I2_S (Integer 2-bit Standard)
- **Format:** 2-bit quantized weights {-1, 0, +1} packed into INT8
- **Method:** Standard quantized matmul with INT8 operations
- **Platform:** x86 and ARM
- **Performance:** Baseline (1× speedup)

### 2. TL1 (Table Lookup 1)
- **Format:** 2-bit ternary weights + lookup tables
- **Method:** Precompute weighted sums, use table indexing
- **Platform:** ARM NEON (optimized for mobile)
- **Performance:** 1.37-5× speedup on ARM

### 3. TL2 (Table Lookup 2)
- **Format:** 2-bit ternary weights + two-level LUTs
- **Method:** Hierarchical table lookup with AVX2
- **Platform:** x86 AVX2 (optimized for desktop/server)
- **Performance:** 2.37-6.17× speedup on x86

**Observation:** TL1/TL2 are where the magic happens. This is what we need to replace.

---

## Core Algorithm: LUT-Based Matmul

### Problem Statement

Compute: **C = A × B** where:
- A: Activations (float32, M×K)
- B: Weights (ternary {-1, 0, +1}, K×N)
- C: Output (float32, M×N)

### Naive Approach (What We'd Normally Do)

```cpp
for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            int8_t w = B[k][n];  // Ternary weight {-1, 0, 1}
            if (w == 1)
                acc += A[m][k];
            else if (w == -1)
                acc -= A[m][k];
            // if w == 0, skip
        }
        C[m][n] = acc;
    }
}
```

**Problem:** K is large (e.g., 4096), so inner loop runs 4096 times per output element. Memory bandwidth bottleneck.

### LUT Approach (What BitNet Does)

**Key insight:** Instead of accumulating one activation at a time, accumulate GROUPS of activations by indexing into precomputed tables.

```cpp
// Precompute LUT: For every 4-bit pattern, store weighted sum
// Example: pattern 0b0101 means weights [+1, 0, -1, +1]
// LUT[0b0101] = a[0] + 0*a[1] - a[2] + a[3]

for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
        float acc = 0.0f;
        for (int k = 0; k < K; k += 4) {  // Process 4 weights at once
            uint8_t pattern = pack_4_ternary_weights(B[k:k+4][n]);
            // pattern is 0-255 (4 weights × 2 bits each = 8 bits)

            float* act_group = &A[m][k];  // 4 consecutive activations
            acc += LUT[pattern](act_group);  // Table lookup + sum
        }
        C[m][n] = acc;
    }
}
```

**Benefit:** Inner loop now runs K/4 times instead of K times. 4× fewer iterations.

### How LUT is Constructed

**For each possible ternary pattern (256 entries for 4 weights):**

```cpp
// LUT entry for pattern 0b00011011 (weights: [+1, +1, 0, -1])
LUT[0b00011011](a) = a[0]*1 + a[1]*1 + a[2]*0 + a[3]*(-1)
                   = a[0] + a[1] - a[3]
```

**In practice (SIMD-optimized):**
- Load 4 activations into SIMD register
- Multiply by precomputed mask {+1, +1, 0, -1}
- Horizontal sum to get result
- Store in accumulator

---

## BitNet Implementation Details

### File: `include/ggml-bitnet.h`

**Key data structure:**

```cpp
struct bitnet_tensor_extra {
    int lut_scales_size;   // Number of scale factors
    int BK;                // Block size for K dimension
    int n_tile_num;        // Number of tiles for tiling optimization
    uint8_t* qweights;     // Quantized ternary weights (packed)
    bitnet_float_type* scales;  // Per-block scale factors
};
```

**Key functions:**

1. **`ggml_bitnet_mul_mat_task_init`**
   - Preprocesses activations into LUT-friendly format
   - Quantizes per-tensor: `scale = 127 / max(|activations|)`
   - Constructs lookup tables

2. **`ggml_qgemm_lut`** (TL1/TL2 versions)
   - Main matrix multiply using LUT
   - ARM TL1: Uses NEON instructions
   - x86 TL2: Uses AVX2 instructions

3. **`ggml_preprocessor`**
   - Converts weights from model format to LUT format
   - Packs ternary weights into compact representation
   - Computes scale factors

### File: `src/ggml-bitnet-lut.cpp`

**Key operations:**

1. **Per-tensor quantization** (scales activations to [-127, 127]):

```cpp
void per_tensor_quant(int k, void* lut_scales_, void* b_) {
    // Find max absolute value using SIMD
    #ifdef __ARM_NEON
        float32x4_t temp_max = vdupq_n_f32(0);
        for (int i=0; i < k / 4; i++) {
            float32x4_t vec_bs = vld1q_f32(b + 4 * i);
            float32x4_t abssum = vabsq_f32(vec_bs);
            temp_max = vmaxq_f32(abssum, temp_max);
        }
        float32_t scales = 127 / vmaxvq_f32(temp_max);
    #elif defined __AVX2__
        __m256 max_vec = _mm256_set1_ps(0.f);
        for (int i = 0; i < k / 8; i++) {
            __m256 vec_b = _mm256_loadu_ps(b + i * 8);
            __m256 vec_babs = _mm256_andnot_ps(vec_sign, vec_b);
            max_vec = _mm256_max_ps(vec_babs, max_vec);
        }
        // ... horizontal max reduction
        float scales = 127 / max_value;
    #endif
    *lut_scales = scales;
}
```

2. **LUT construction** (template for different K sizes):

```cpp
template<int act_k>
inline void lut_ctor(int8_t* qlut, bitnet_float_type* b, bitnet_float_type* lut_scales) {
    // For every possible ternary pattern (2^(2*act_k) entries)
    // Compute: weighted_sum = sum(b[i] * pattern[i]) for i in 0..act_k

    // SIMD optimized version processes multiple patterns at once
    // Stores results in qlut table for fast lookup during matmul
}
```

### Preset Kernels (Code Generated)

**File:** `preset_kernels/bitnet_b1_58-3B/bitnet-lut-kernels-tl1.h`

**Structure:** ~50,000 lines of generated SIMD code for specific model sizes.

Example snippet (simplified):

```cpp
void ggml_qgemm_lut(int m, int k, void* A, void* LUT, void* Scales, void* LUT_Scales, void* C) {
    // Outer loops over M (batch) and K (accumulation dimension)
    for (int m_idx = 0; m_idx < m; m_idx++) {
        int32x4_t acc[8];  // NEON accumulator registers

        // Inner loop: lookup and accumulate
        for (int k_idx = 0; k_idx < k / 4; k_idx++) {
            uint8_t pattern = get_pattern(LUT, k_idx);

            // Load 4 activations
            int32x4_t act = vld1q_s32(&A[m_idx * k + k_idx * 4]);

            // Lookup precomputed weights for this pattern
            int32x4_t weights = vld1q_s32(&LUT[pattern * 4]);

            // Multiply-accumulate
            acc[0] = vmlaq_s32(acc[0], act, weights);
        }

        // Reduce accumulators and store result
        C[m_idx] = horizontal_sum(acc) * Scales[m_idx];
    }
}
```

**Why code generation?** Kernels are specialized for:
- Model hidden size (e.g., 2048, 4096)
- Block size (BK parameter)
- Number of LUT entries (depends on grouping factor)

---

## Weight Packing Format

### Original BitNet Format

Ternary weights stored as:
- **INT8 array:** Each weight is -1, 0, or +1
- **Per-block scales:** FP32 scale factor for each 128-weight block
- **GGUF format:** Standard llama.cpp model format

### LUT-Packed Format (After Preprocessing)

Weights are reorganized into:
1. **Grouped patterns:** Every 4 weights packed into 8-bit pattern (2 bits × 4 = 8 bits)
2. **Reordered for cache:** Weights reordered to improve memory access locality
3. **Lookup tables:** Precomputed for all 256 patterns

**Example:**

```
Original: [-1, +1, 0, -1, +1, +1, 0, 0, ...]  (8 weights, 8 bytes)

Packed:   [0b11010011, 0b01010000, ...]        (2 bytes)
          pattern1     pattern2

Encoding: -1 → 00, 0 → 01, +1 → 10
```

---

## Performance Characteristics

### Memory Bandwidth Analysis

**Naive matmul:**
- Read A: M×K×4 bytes (float32)
- Read B: K×N×1 bytes (int8 ternary)
- Write C: M×N×4 bytes (float32)
- **Total:** 4MK + KN + 4MN bytes

**LUT matmul:**
- Read A: M×K×4 bytes (same)
- Read LUT: (K/4)×N×1 bytes (pattern indices)
- Read LUT table: 256×4 bytes (reused)
- Write C: M×N×4 bytes (same)
- **Total:** 4MK + KN/4 + 1024 + 4MN bytes

**Bandwidth reduction:** ~75% less weight data transferred (KN → KN/4).

### Computation Analysis

**Naive matmul:**
- Inner loop iterations: M×N×K
- Operations per iteration: 1 multiply-add (if weight ≠ 0)
- **Total:** ~M×N×K ops (sparse due to zeros)

**LUT matmul:**
- Inner loop iterations: M×N×(K/4)
- Operations per iteration: 1 table lookup + 4-8 SIMD adds
- **Total:** ~M×N×K/2 effective ops (SIMD parallelism)

**Speedup sources:**
1. 4× fewer iterations (K/4 instead of K)
2. SIMD parallelism (process 4-8 elements at once)
3. Better cache locality (LUT fits in L1)
4. No branch mispredictions (table lookup instead of if-else)

### Measured Performance (from README)

| Model | Platform | Kernel | Speedup | Energy Savings |
|:------|:---------|:-------|--------:|---------------:|
| 3B | ARM M2 | TL1 | 1.37-5.07× | 55.4-70.0% |
| 3B | x86 i7 | TL2 | 2.37-6.17× | 71.9-82.2% |
| 100B | x86 i7 | TL2 | 5-7 tok/sec | ~80% |

**Observation:** Larger models benefit more (better SIMD utilization, amortized LUT overhead).

---

## Integration Opportunities

### What BitNet Does Well ✅

1. **Proven architecture:** 2B-100B parameter models validated
2. **Ecosystem:** Hugging Face models, llama.cpp compatibility
3. **Multi-platform:** ARM NEON and x86 AVX2 support
4. **Production-ready:** Stable inference, server deployment

### What BitNet Doesn't Have ❌

1. **Dense243 packing:** Uses 2 bits per trit (8 trits/byte) vs our 5 trits/byte
2. **Balanced ternary operations:** Generic LUT vs our specialized tadd/tmul
3. **Operation fusion:** No tnot+tadd fusion like our 1.94× speedup
4. **Direct GEMM:** Uses LUT indirection vs direct ternary multiply-accumulate

### Our Improvement Strategy 🎯

**Replace TL1/TL2 kernels with our Direct Ternary GEMM:**

| Component | BitNet (Current) | TritNet (Improved) | Benefit |
|:----------|:-----------------|:-------------------|:--------|
| Weight packing | 2-bit, 8 trits/byte | Dense243, 5 trits/byte | 60% memory reduction |
| Matmul method | LUT indirection | Direct ternary ops | Eliminate table lookup overhead |
| Inner loop | K/4 iterations | K/5 iterations | 25% fewer iterations |
| Operations | Table lookup + SIMD add | Fused ternary_fma | 2× fewer instructions |
| Cache usage | LUT (256×4 = 1KB) | No LUT | Better L1 utilization |

**Expected performance gain:** 2-3× on top of BitNet's existing 2-6× = **4-18× total speedup over baseline**.

---

## Integration Architecture

### Proposed Stack

```
┌─────────────────────────────────────────┐
│  Python API (Hugging Face / llama.cpp)  │
│  - Model loading, tokenization          │
│  - Unchanged from BitNet                │
├─────────────────────────────────────────┤
│  GGML Backend Adapter                   │
│  - Detect TL1/TL2 operations            │
│  - Route to TritNet GEMM instead        │
├─────────────────────────────────────────┤
│  TritNet GEMM Kernel (NEW)              │
│  - Dense243 weight unpacking            │
│  - Direct ternary multiply-accumulate   │
│  - Fused ternary_fma operations         │
│  - AVX2/NEON optimized                  │
├─────────────────────────────────────────┤
│  Our Ternary Engine (Existing)          │
│  - tadd, tmul, ternary_fma primitives   │
│  - Dense243 packing/unpacking           │
│  - 35 Gops/s validated                  │
└─────────────────────────────────────────┘
```

### Integration Points

**1. Replace `ggml_qgemm_lut` (TL1/TL2):**

```cpp
// Current: BitNet LUT approach
void ggml_qgemm_lut(...) {
    // Lookup table matmul
}

// New: Our direct ternary approach
void tritnet_qgemm_direct(
    int M, int N, int K,
    const float* A,           // Activations [M, K]
    const uint8_t* B_packed,  // Dense243-packed weights [K/5, N]
    float* C                  // Output [M, N]
) {
    // Use our ternary_fma and Dense243 unpacking
    // No LUT overhead, direct computation
}
```

**2. Weight conversion (one-time during model load):**

```cpp
// Convert BitNet 2-bit format to our Dense243 format
void convert_bitnet_to_dense243(
    const uint8_t* bitnet_weights,  // 2-bit packed [K/4, N]
    uint8_t* dense243_weights,      // Dense243 packed [K/5, N]
    int K, int N
) {
    // Unpack BitNet 2-bit → ternary {-1, 0, +1}
    // Repack using Dense243 encoding
    // 40% memory savings!
}
```

**3. CMake integration:**

```cmake
# Add our ternary engine as submodule
add_subdirectory(ternary_engine)

# Link BitNet with TritNet GEMM
target_link_libraries(bitnet_main PRIVATE ternary_engine)
target_compile_definitions(bitnet_main PRIVATE USE_TRITNET_GEMM)
```

---

## Next Steps

### Week 1-2: Design & Prototype

**Tasks:**
1. ✅ Clone and study BitNet (DONE)
2. ✅ Analyze kernel architecture (DONE - this document)
3. Design TritNet GEMM API matching ggml_qgemm_lut signature
4. Implement naive Direct Ternary GEMM (no SIMD)
5. Test correctness on small matrices

**Deliverables:**
- `include/tritnet_gemm.h` - API specification
- `src/tritnet_gemm_naive.cpp` - Naive implementation
- `tests/test_tritnet_gemm.cpp` - Unit tests
- Benchmark: correctness validation

### Week 3-4: SIMD Optimization

**Tasks:**
1. Add AVX2 vectorization for x86
2. Add NEON vectorization for ARM
3. Implement Dense243 batch unpacking
4. Add loop tiling for cache efficiency
5. Benchmark on realistic sizes (4096×4096)

**Deliverables:**
- `src/tritnet_gemm_avx2.cpp`
- `src/tritnet_gemm_neon.cpp`
- Micro-benchmarks showing 2-3× improvement over BitNet TL2/TL1

### Week 5-6: Integration & Release

**Tasks:**
1. Fork bitnet.cpp to tritnet.cpp
2. Replace TL1/TL2 with our GEMM
3. Add weight conversion (BitNet → Dense243)
4. Test on BitNet 2B model
5. Write documentation and examples

**Deliverables:**
- `tritnet.cpp` repository
- Comparative benchmarks (stock vs TritNet)
- README with installation instructions
- PR to llama.cpp ecosystem

---

## Challenges & Mitigations

### Challenge 1: GGML API Compatibility

**Issue:** BitNet uses GGML tensor format from llama.cpp. We need to integrate without breaking compatibility.

**Mitigation:**
- Keep GGML tensor interface unchanged
- Add `tritnet_tensor_extra` structure (like BitNet's `bitnet_tensor_extra`)
- Weight conversion happens transparently during model load

### Challenge 2: Code Generation Complexity

**Issue:** BitNet generates specialized kernels for each model size. We may need similar specialization.

**Mitigation:**
- Start with template-based approach (compile-time specialization)
- Use C++ templates for different K/N sizes
- If needed, add code generation later

### Challenge 3: Multiple Platforms

**Issue:** Need to support both x86 AVX2 and ARM NEON.

**Mitigation:**
- Develop AVX2 version first (primary development platform)
- Port to NEON using our existing SIMD experience
- Reuse our existing build system's SIMD detection

### Challenge 4: Numerical Accuracy

**Issue:** Must match BitNet's output bit-exact for validation.

**Mitigation:**
- Extensive testing on known-good inputs
- Compare against BitNet TL2 output element-wise
- Tolerance: 1e-6 for FP32 accumulation

---

## Success Metrics

### Phase 1: Correctness
- ✅ Naive GEMM passes all unit tests
- ✅ Output matches BitNet TL2 within 1e-6 tolerance
- ✅ Works on matrices up to 8192×8192

### Phase 2: Performance
- ✅ 2× speedup over BitNet TL2 on AVX2 (x86)
- ✅ 2× speedup over BitNet TL1 on NEON (ARM)
- ✅ 40% memory reduction (Dense243 vs 2-bit)

### Phase 3: Integration
- ✅ BitNet 2B model inference works
- ✅ 3-5× total speedup on token generation
- ✅ Bit-exact output validation

### Long-term: Adoption
- ✅ PR merged into llama.cpp or BitNet
- ✅ Used in at least 3 downstream projects
- ✅ Citations in research papers

---

## Conclusion

BitNet.cpp provides a proven ternary LLM architecture with impressive performance (2-6× speedup, 55-82% energy savings). However, it uses generic LUT-based matmul that doesn't leverage balanced ternary nuances.

**Our opportunity:** Replace their LUT kernels with our Direct Ternary GEMM powered by:
- Dense243 packing (60% memory savings)
- Specialized ternary operations (tadd, tmul, ternary_fma)
- Operation fusion (1.94× demonstrated speedup)

**Expected result:** 2-3× additional speedup = **4-18× total over baseline**, making ternary LLMs viable for edge deployment.

**Next action:** Design TritNet GEMM API specification and implement naive prototype.

---

**Status:** Analysis Complete - Ready for Design Phase
**Timeline:** Weeks 1-6 starting immediately
**Success criteria:** 3× speedup on BitNet 2B inference

**Last updated:** 2025-11-23
