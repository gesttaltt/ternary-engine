# High-Performance Inference Engines

**Doc-Type:** Ecosystem Analysis · Version 1.0 · Generated 2025-12-09

This document analyzes high-performance inference engines that represent both deployment targets and sources of optimization patterns for Ternary Engine.

---

## Table of Contents

1. [Overview](#overview)
2. [llama.cpp](#1-llamacpp)
3. [whisper.cpp / GGML](#2-whispercpp--ggml)
4. [vLLM](#3-vllm)
5. [TensorRT-LLM](#4-tensorrt-llm)
6. [SGLang](#5-sglang)
7. [RTen](#6-rten)
8. [Comparative Analysis](#comparative-analysis)
9. [Integration Opportunities](#integration-opportunities)
10. [Lessons for Ternary Engine](#lessons-for-ternary-engine)

---

## Overview

Modern inference engines have solved many problems Ternary Engine faces:
- Memory-efficient execution
- SIMD optimization
- Multi-platform support
- Model format standardization

| Engine | Focus | Platform | Language | Stars |
|--------|-------|----------|----------|-------|
| llama.cpp | LLM CPU inference | Multi | C/C++ | 75k+ |
| whisper.cpp | Audio models | Multi | C/C++ | 40k+ |
| vLLM | LLM GPU serving | CUDA | Python/C++ | 35k+ |
| TensorRT-LLM | LLM GPU (NVIDIA) | CUDA | Python/C++ | 10k+ |
| SGLang | LLM serving | CUDA | Python | 8k+ |
| RTen | ONNX inference | Multi | Rust | 500+ |

---

## 1. llama.cpp

### Repository Information

- **URL:** https://github.com/ggml-org/llama.cpp
- **Stars:** 75,000+
- **Language:** C/C++
- **License:** MIT
- **Status:** Very actively maintained

### Why It Matters Most

llama.cpp is the single most important reference for Ternary Engine because it:
1. Proves pure C/C++ can compete with frameworks
2. Has best-in-class SIMD optimization
3. Supports extreme quantization (1.5-8 bit)
4. Shows how to build a complete ecosystem

### Architecture

```
llama.cpp Architecture:

┌─────────────────────────────────────────────────────────────┐
│                      User Interface                          │
│  CLI (main), Server (server), Python (llama-cpp-python)     │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    Model Loading (GGUF)                      │
│  Tokenizer, Weights, Config, Quantization metadata          │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    Inference Engine                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ KV Cache    │  │ Attention   │  │ Feed Forward        │  │
│  │ Management  │  │ Computation │  │ + Quantized MatMul  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    GGML Tensor Library                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Backend Selection: CPU, CUDA, Metal, Vulkan, etc.   │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌───────────┬───────────┬───────────┬─────────────────┐    │
│  │ AVX/AVX2  │ AVX-512   │ ARM NEON  │ CUDA/Metal      │    │
│  │ Kernels   │ Kernels   │ Kernels   │ Kernels         │    │
│  └───────────┴───────────┴───────────┴─────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Quantization Support

llama.cpp supports multiple quantization formats:

| Format | Bits | Description | Use Case |
|--------|------|-------------|----------|
| Q2_K | 2.5 | 2-bit with K-quant | Maximum compression |
| Q3_K_S/M/L | 3.0-3.5 | 3-bit variants | Balance |
| Q4_0 | 4.0 | Simple 4-bit | Fast, good quality |
| Q4_K_S/M | 4.5 | K-quant 4-bit | Better quality |
| Q5_0 | 5.0 | 5-bit | Higher quality |
| Q5_K_S/M | 5.5 | K-quant 5-bit | Near FP16 quality |
| Q6_K | 6.5 | 6-bit | Best quality |
| Q8_0 | 8.0 | 8-bit | Baseline |

### Key SIMD Patterns

```cpp
// From ggml-quants.c - Q4_0 dequantization and dot product
void ggml_vec_dot_q4_0_q8_0(
    int n, float * restrict s,
    const void * restrict vx, const void * restrict vy
) {
    const int nb = n / QK4_0;
    const block_q4_0 * restrict x = vx;
    const block_q8_0 * restrict y = vy;

#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        // Load quantized blocks
        const __m256i qx = bytes_from_nibbles_32(x[i].qs);
        const __m256i qy = _mm256_loadu_si256((const __m256i *)y[i].qs);

        // Subtract 8 to center around zero (Q4_0 stores 0-15)
        const __m256i qx_centered = _mm256_sub_epi8(qx, _mm256_set1_epi8(8));

        // Multiply and accumulate (using maddubs trick)
        const __m256i dot = _mm256_maddubs_epi16(qx_centered, qy);
        const __m256i sum = _mm256_madd_epi16(dot, _mm256_set1_epi16(1));

        // Scale by dequantization factors
        const __m256 scale = _mm256_set1_ps(
            GGML_FP16_TO_FP32(x[i].d) * GGML_FP16_TO_FP32(y[i].d)
        );
        acc = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(sum), acc);
    }

    *s = hsum_float_8(acc);
#else
    // Scalar fallback
    // ...
#endif
}
```

### Memory Layout

```cpp
// Q4_0 block structure (32 values)
typedef struct {
    ggml_fp16_t d;       // Scale factor (2 bytes)
    uint8_t qs[QK4_0/2]; // Quantized values, 4 bits each (16 bytes)
} block_q4_0;            // Total: 18 bytes for 32 values = 4.5 bits/value

// Ternary equivalent (potential)
typedef struct {
    ggml_fp16_t d;       // Scale factor (2 bytes)
    uint8_t qs[QK/4];    // Ternary values, 2 bits each
} block_ternary;         // e.g., 32 values = 8 bytes + 2 = 10 bytes = 2.5 bits/value
```

### Why It Matters for Ternary Engine

**Learn:**
- SIMD kernel patterns for quantized operations
- Memory layout for efficient cache utilization
- Multi-backend architecture
- Model format (GGUF) design

**Potential Integration:**
- Add ternary quantization format to GGUF
- Implement ternary-specific kernels in GGML
- Enable ternary models in llama.cpp ecosystem

---

## 2. whisper.cpp / GGML

### Repository Information

- **URL:** https://github.com/ggml-org/whisper.cpp
- **GGML:** Embedded in whisper.cpp and llama.cpp
- **Stars:** 40,000+
- **Language:** C
- **License:** MIT

### GGML Tensor Library

GGML (Georgi Gerganov ML) is the foundation for both llama.cpp and whisper.cpp:

```c
// GGML tensor creation and operations
struct ggml_context * ctx = ggml_init(params);

// Create tensors
struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 512);
struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 512);

// Build computation graph
struct ggml_tensor * c = ggml_mul_mat(ctx, a, b);  // Matrix multiply

// Execute
struct ggml_cgraph * graph = ggml_build_forward(c);
ggml_graph_compute(ctx, graph);
```

### Type System

```c
// GGML type definitions
enum ggml_type {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    // ... many quantized types
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    // ...
    GGML_TYPE_COUNT,
};

// Each type has associated functions
struct ggml_type_traits {
    const char * type_name;
    int          blck_size;    // Block size for quantized types
    size_t       type_size;    // Bytes per block
    bool         is_quantized;
    ggml_to_float_t to_float;  // Dequantization function
    ggml_from_float_t from_float;  // Quantization function
    // ...
};
```

### Why It Matters for Ternary Engine

**GGML is a model for kernel design:**
- Clean C implementation
- Minimal dependencies
- Excellent SIMD coverage
- Proven at scale

**Adding ternary to GGML:**
```c
// Hypothetical ternary type
GGML_TYPE_TERNARY = XX,

// With associated traits
{
    .type_name = "ternary",
    .blck_size = 32,         // 32 ternary values per block
    .type_size = sizeof(block_ternary),
    .is_quantized = true,
    .to_float = ternary_to_float,
    .from_float = ternary_from_float,
}
```

---

## 3. vLLM

### Repository Information

- **URL:** https://github.com/vllm-project/vllm
- **Stars:** 35,000+
- **Language:** Python, C++, CUDA
- **License:** Apache 2.0
- **Status:** Very active

### Core Innovation: PagedAttention

vLLM's key innovation is treating KV cache like virtual memory:

```
Traditional KV Cache:
┌─────────────────────────────────────────┐
│ Request 1: [████████████░░░░░░░░░░░░░░] │  Wasted space
│ Request 2: [██████████████░░░░░░░░░░░░] │  (pre-allocated)
│ Request 3: [██████░░░░░░░░░░░░░░░░░░░░] │
└─────────────────────────────────────────┘

PagedAttention:
┌─────────────────────────────────────────┐
│ Page Pool: [█1][█1][█2][█3][█1][█2][█3] │  No waste
│ Request 1: Pages [0,1,4]                 │  (paged)
│ Request 2: Pages [2,5]                   │
│ Request 3: Pages [3,6]                   │
└─────────────────────────────────────────┘
```

### Architecture

```python
# vLLM high-level API
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=1,
    quantization="awq",  # Quantization method
)

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

outputs = llm.generate(["Hello, how are you?"], sampling_params)
```

### Quantization Support

```python
# vLLM quantization options
quantization_methods = [
    "awq",           # 4-bit AWQ
    "gptq",          # 4-bit GPTQ
    "squeezellm",    # Mixed precision
    "fp8",           # FP8
    "bitsandbytes",  # 4/8-bit
    # "ternary",     # <-- Future opportunity
]
```

### Why It Matters for Ternary Engine

**Deployment target:**
- vLLM is the de facto standard for LLM serving
- Adding ternary support would reach many users

**Integration approach:**
```python
# Goal: Ternary support in vLLM
llm = LLM(
    model="user/llama-2-7b-ternary",
    quantization="ternary",  # New option
)
```

---

## 4. TensorRT-LLM

### Repository Information

- **URL:** https://github.com/NVIDIA/TensorRT-LLM
- **Stars:** 10,000+
- **Language:** Python, C++, CUDA
- **License:** Apache 2.0
- **Status:** Active (NVIDIA-maintained)

### Architecture

```
TensorRT-LLM Stack:

┌─────────────────────────────────────────────────────────────┐
│                    Python High-Level API                     │
│            (LLM class, model builders, quantization)         │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    TensorRT Engine Builder                   │
│         (Graph optimization, kernel selection, fusion)       │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    CUDA Kernels                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ FP8 MatMul  │  │ INT8 MatMul │  │ INT4 MatMul         │  │
│  │ (FP8 Tensor │  │ (INT8 Tensor│  │ (Specialized        │  │
│  │  Cores)     │  │  Cores)     │  │  kernels)           │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Quantization Support

```python
from tensorrt_llm.quantization import QuantMode

quant_modes = [
    QuantMode.INT8_WEIGHT_ONLY,    # W8A16
    QuantMode.INT4_WEIGHT_ONLY,    # W4A16
    QuantMode.FP8,                 # W8A8 FP8
    QuantMode.INT8_KV_CACHE,       # KV cache quantization
    # Custom modes possible via plugin
]
```

### Why It Matters for Ternary Engine

**GPU optimization patterns:**
- Tensor Core utilization
- Kernel fusion strategies
- Memory coalescing

**Potential integration:**
- TensorRT plugin for ternary operations
- Ternary Tensor Cores (future hardware?)

---

## 5. SGLang

### Repository Information

- **URL:** https://github.com/sgl-project/sglang
- **Stars:** 8,000+
- **Language:** Python, C++
- **License:** Apache 2.0
- **Status:** Very active

### Innovation: RadixAttention

SGLang introduces RadixAttention for efficient prefix caching:

```
Prefix Caching with RadixAttention:

Request 1: "What is the capital of France?"
Request 2: "What is the capital of Germany?"
Request 3: "What is the population of France?"

Radix Tree:
           "What is the"
           /            \
    "capital of"    "population of"
       /    \              \
  "France?" "Germany?"   "France?"

Shared prefix computation saves ~50% compute for related queries.
```

### Why It Matters

SGLang represents cutting-edge serving optimization. Future ternary serving could leverage similar techniques.

---

## 6. RTen

### Repository Information

- **URL:** https://github.com/robertknight/rten
- **Stars:** 500+
- **Language:** Rust
- **License:** MIT
- **Status:** Active

### Why It's Relevant

RTen is a lightweight ONNX runtime with excellent SIMD support:

```rust
// RTen SIMD abstraction
pub trait SimdFloat {
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn reduce_sum(self) -> f32;
}

// Implementations for different ISAs
impl SimdFloat for __m256 { /* AVX2 */ }
impl SimdFloat for float32x4_t { /* NEON */ }
impl SimdFloat for v128 { /* WASM SIMD */ }
```

### Cross-Platform SIMD

```rust
// RTen's approach to multi-ISA support
#[cfg(target_arch = "x86_64")]
mod avx2 {
    pub fn matmul_avx2(...) { /* AVX2 implementation */ }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    pub fn matmul_neon(...) { /* NEON implementation */ }
}

// Runtime dispatch
fn matmul(...) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return avx2::matmul_avx2(...);
    }

    #[cfg(target_arch = "aarch64")]
    return neon::matmul_neon(...);

    // Scalar fallback
    scalar::matmul(...)
}
```

### Why It Matters for Ternary Engine

**Clean cross-platform example:**
- Shows how to abstract SIMD across ISAs
- Rust patterns applicable to C++ design
- ONNX runtime integration patterns

---

## Comparative Analysis

### Performance Characteristics

| Engine | Tokens/sec (7B) | Memory (7B) | Platforms |
|--------|-----------------|-------------|-----------|
| llama.cpp Q4_0 | ~20-40 (CPU) | ~4 GB | All |
| vLLM FP16 | ~100-200 (GPU) | ~14 GB | CUDA |
| vLLM AWQ | ~150-250 (GPU) | ~4 GB | CUDA |
| TensorRT-LLM | ~200-300 (GPU) | ~4 GB | CUDA |
| **Ternary (target)** | ~30-50 (CPU) | ~2 GB | All |

### Quantization Support

| Engine | FP16 | INT8 | INT4 | INT2 | Ternary |
|--------|------|------|------|------|---------|
| llama.cpp | Yes | Yes | Yes | Yes* | No |
| vLLM | Yes | Yes | Yes | No | No |
| TensorRT-LLM | Yes | Yes | Yes | No | No |
| SGLang | Yes | Yes | Yes | No | No |
| RTen | Yes | Yes | No | No | No |
| **Target** | - | - | - | - | **Yes** |

*llama.cpp's Q2_K is 2.5-bit average, not true INT2

---

## Integration Opportunities

### Priority 1: llama.cpp/GGML

**Why:** Largest ecosystem, best fit for CPU inference

**How:**
```c
// Add to ggml-quants.c
void quantize_row_ternary(const float * x, void * y, int k);
void dequantize_row_ternary(const void * x, float * y, int k);
void ggml_vec_dot_ternary_q8_0(int n, float * s, const void * vx, const void * vy);
```

### Priority 2: vLLM

**Why:** Standard for GPU serving, access to production deployments

**How:**
```python
# vllm/model_executor/layers/quantization/ternary.py
class TernaryLinearMethod(LinearMethodBase):
    def create_weights(self, ...):
        # Ternary weight storage
        pass

    def apply_weights(self, ...):
        # Ternary matmul kernel
        pass
```

### Priority 3: ONNX Custom Op

**Why:** Framework-agnostic deployment

**How:**
```cpp
// Register ternary ONNX operators
ONNX_OPERATOR_KERNEL_EX(
    TernaryMatMul,           // Op name
    kTernaryDomain,          // Domain
    1,                       // Version
    kCpuExecutionProvider,   // Provider
    KernelDefBuilder(),
    TernaryMatMulKernel
);
```

---

## Lessons for Ternary Engine

### From llama.cpp

1. **Memory layout is critical:**
   ```cpp
   // Block-based quantization for cache efficiency
   struct block_ternary {
       float scale;      // Dequantization scale
       uint8_t qs[16];   // 64 ternary values (2 bits each)
   };
   ```

2. **SIMD kernel structure:**
   ```cpp
   // Pattern: load, process, reduce
   void ternary_dot(const block_ternary* x, const block_q8_0* y, float* out) {
       __m256 acc = _mm256_setzero_ps();

       for (int i = 0; i < nblocks; i++) {
           // 1. Load and unpack ternary values
           __m256i tx = unpack_ternary(x[i].qs);

           // 2. Load q8 values
           __m256i ty = _mm256_loadu_si256(&y[i].qs);

           // 3. Ternary multiply-add
           // tx ∈ {-1, 0, +1}, ty ∈ [-127, 127]
           __m256i prod = ternary_mul(tx, ty);

           // 4. Horizontal sum and accumulate
           float partial = hsum_i32(prod);
           acc += x[i].scale * y[i].scale * partial;
       }

       *out = hsum_f32(acc);
   }
   ```

3. **Runtime dispatch:**
   ```cpp
   // Check CPU features and select best kernel
   if (ggml_cpu_has_avx512()) {
       ternary_dot_avx512(x, y, out);
   } else if (ggml_cpu_has_avx2()) {
       ternary_dot_avx2(x, y, out);
   } else {
       ternary_dot_scalar(x, y, out);
   }
   ```

### From vLLM

1. **Memory efficiency patterns:**
   - Paged memory allocation
   - Lazy tensor creation
   - Memory pooling

2. **Serving infrastructure:**
   - Request batching
   - Priority scheduling
   - Async execution

### From TensorRT-LLM

1. **Kernel fusion:**
   - Fuse quantize + matmul + dequantize
   - Fuse attention components
   - Minimize memory round-trips

2. **Plugin system:**
   - Clean interfaces for custom ops
   - Integration with graph optimization

---

## Action Items for Ternary Engine

### Phase 1: CPU Excellence

1. [ ] Study llama.cpp Q4_0/Q2_K kernel implementations
2. [ ] Implement ternary block format compatible with GGML
3. [ ] Create AVX2 ternary dot product kernel
4. [ ] Benchmark against llama.cpp Q2_K

### Phase 2: Integration

1. [ ] Create GGUF ternary quantization type
2. [ ] Implement llama.cpp model loader for ternary
3. [ ] Add ternary option to llama-cpp-python

### Phase 3: Serving

1. [ ] Implement vLLM ternary quantization method
2. [ ] Create ONNX custom ops for ternary
3. [ ] Benchmark serving throughput

### Phase 4: GPU

1. [ ] Study TensorRT-LLM INT4 kernels
2. [ ] Implement CUDA ternary matmul
3. [ ] Integrate with TensorRT plugin system

---

**Document Version:** 1.0
**Generated:** 2025-12-09
**Author:** Claude Code Analysis
