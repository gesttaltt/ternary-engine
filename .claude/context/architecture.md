# Ternary Engine Architecture Guide

System architecture and design principles for the Ternary Engine.

## Core Innovation

**Traditional approach (memory-bound):**
```
Input → Lookup Table (LUT) → Output
         ↑
    Memory access (slow)
```

**TritNet approach (compute-bound):**
```
Input → Neural Network (matmul) → Output
         ↑
    GPU/TPU tensor cores (fast, batched)
```

## Architectural Layers

### Layer 0: Compile-Time LUT Generation

**File:** `ternary_core/algebra/ternary_lut_gen.h`
**Purpose:** Generate lookup tables at compile time
**Key concept:** Algorithm-as-documentation

```cpp
// Mathematical rule defined once
constexpr uint8_t compute_tadd(int8_t a, int8_t b) {
    int8_t sum = a + b;
    return clamp(sum, -1, 1);  // Saturate to ternary range
}

// Compile-time LUT generation
constexpr auto TADD_LUT = generate_lut<compute_tadd>();
```

**Benefits:**
- Single source of truth (mathematical rule)
- Zero runtime cost
- Type-safe at compile time
- Self-documenting code

### Layer 1: Scalar Operations

**File:** `ternary_core/algebra/ternary_algebra.h`
**Purpose:** Branch-free scalar operations using LUTs
**Encoding:** 2-bit per trit (0b00=-1, 0b01=0, 0b10=+1)

```cpp
// Force-inlined for zero overhead
FORCE_INLINE uint8_t tadd_scalar(uint8_t a, uint8_t b) {
    uint8_t index = (a << 2) | b;  // Combine for LUT index
    return TADD_LUT[index];        // Single memory access
}
```

**Performance:** ~2 CPU cycles per operation (0.5 ns @ 3 GHz)

### Layer 2: SIMD Vectorization

**File:** `ternary_core/simd/ternary_simd_kernels.h`
**Purpose:** Process 32 trits in parallel via AVX2
**Instruction:** `_mm256_shuffle_epi8` (LUT within 256-bit register)

```cpp
// 32 parallel operations in single instruction
__m256i tadd_simd_32(__m256i a, __m256i b) {
    __m256i indices = blend_operands(a, b);  // (a << 2) | b for 32 trits
    return _mm256_shuffle_epi8(LUT_REG, indices);  // Parallel lookup
}
```

**Performance:** ~0.23 CPU cycles per element (32× parallelism)
**Throughput:** 35,042 Mops/s peak (validated)

### Layer 3: Operation Fusion

**File:** `ternary_core/simd/ternary_fusion.h`
**Purpose:** Combine multiple operations to reduce memory traffic

```cpp
// Fused: result = tadd(tnot(a), b)
// Traditional: 2 passes (tnot → memory → tadd)
// Fused: 1 pass (tnot + tadd in registers)

result = tadd_simd(tnot_simd(a), b);  // No intermediate memory write
```

**Performance:** 1.6× to 15.5× speedup (validated Phase 4.0)
**Key benefit:** Cache-friendly, reduces memory bandwidth

### Layer 4: Python Bindings

**File:** `ternary_simd_engine.cpp`
**Purpose:** Zero-copy NumPy integration via pybind11

```cpp
py::array_t<uint8_t> tadd(py::array_t<uint8_t> a, py::array_t<uint8_t> b) {
    // Three execution paths:
    // 1. OpenMP parallel (arrays ≥100K)
    // 2. Serial SIMD (medium arrays)
    // 3. Scalar tail (remainder elements)
    return process_binary_array<true>(a, b, tadd_scalar, tadd_simd_32);
}
```

**Template unification:** Single `process_binary_array<Sanitize>()` handles:
- Sanitized vs unsanitized paths
- All binary operations (tadd, tmul, tmin, tmax)
- Automatic path selection based on size and alignment

**Code reduction:** 73% vs original implementation

### Layer 5: Runtime Safety

**File:** `ternary_core/simd/ternary_cpu_detect.h`
**Purpose:** Graceful degradation and validation

**CPU Detection:**
```cpp
if (supports_avx2()) {
    use_simd_path();
} else {
    throw std::runtime_error("AVX2 required");
}
```

**Alignment Validation:**
```cpp
if (is_aligned(ptr, 32)) {
    use_streaming_stores();  // Non-temporal writes
} else {
    use_regular_stores();    // Standard writes
}
```

**Hardware Concurrency Clamping:**
```cpp
// Prevent VM crashes from over-threading
max_threads = std::min(omp_get_max_threads(), hardware_concurrency);
```

## Kernel vs Engine Separation

### Production Kernel (ternary_core/)

**Deployment status:** Production-ready (Windows x64 validated)
**Criteria:**
- 100% test coverage
- Performance validated with benchmarks
- Mathematically stable
- Clear documentation with validation dates

**Files:**
- ternary_core/algebra/ - Core operations
- ternary_core/simd/ - SIMD kernels
- ternary_core/ffi/ - C FFI layer
- ternary_core/core_api.h - Unified entry point

### Experimental Engine (ternary_engine/)

**Deployment status:** Pending validation
**Criteria:**
- Implementation complete
- Must prove >10% performance gain
- Awaiting rigorous benchmarking
- Clear phase tracking

**Files:**
- experimental/dense243/ - High-density encoding (validated, ready)
- experimental/fusion/ - Operation fusion (Phase 4.0 validated, 4.1 pending)

## Design Principles

### YAGNI (You Aren't Gonna Need It)

**Rule:** No speculative code, only proven optimizations

**Example:**
- ✅ Dense243 encoding: Proven 80% space savings
- ❌ Dense729 encoding: Theoretical, not implemented (no proven use case)

### Phase Coherence

**Rule:** Only add complexity if >10% performance gain measured

**Example:**
- Phase 4.0 fusion: 1.6-15.5× speedup → ACCEPTED
- Hypothetical Phase 4.2: 3% improvement → REJECTED (not worth complexity)

### Single Source of Truth

**Rule:** Define mathematical rules once, generate everything else

**Example:**
```cpp
// GOOD: Single definition
constexpr auto compute_tadd(a, b) { return clamp(a + b, -1, 1); }
constexpr auto TADD_LUT = generate_lut<compute_tadd>();

// BAD: Hardcoded LUT (duplicate definition)
constexpr uint8_t TADD_LUT[16] = {0, 1, 2, /* ... */};  // Error-prone
```

### Template-Based Unification

**Rule:** Single template > multiple code paths

**Before (duplicated):**
```cpp
tadd_sanitized(a, b);
tadd_fast(a, b);
tmul_sanitized(a, b);
tmul_fast(a, b);
// ... 6 separate implementations
```

**After (unified):**
```cpp
template<bool Sanitize, typename ScalarOp, typename SimdOp>
process_binary_array(a, b, scalar_op, simd_op);
// Single implementation, compile-time specialization
```

**Result:** 73% code reduction, zero runtime cost

## Data Flow

### Standard Operations

```
Python NumPy array
    ↓
pybind11 (zero-copy)
    ↓
Input validation (if Sanitize=true)
    ↓
Size check → Path selection
    ↓
┌─────────────┬──────────────┬──────────────┐
│ Large array │ Medium array │ Small array  │
│ (≥100K)     │ (≥32)        │ (<32)        │
├─────────────┼──────────────┼──────────────┤
│ OpenMP      │ Serial SIMD  │ Scalar       │
│ parallel    │ (32-wide)    │ (fallback)   │
└─────────────┴──────────────┴──────────────┘
    ↓
Result array (NumPy)
    ↓
Return to Python
```

### Fused Operations

```
Python: fused_tnot_tadd(a, b)
    ↓
C++: Single pass, no intermediate storage
    ↓
For each SIMD chunk (32 elements):
    temp = tnot_simd(a)      // In register
    result = tadd_simd(temp, b)  // No memory write
    ↓
Return to Python
```

**Benefit:** 1.6-15.5× speedup by avoiding memory roundtrip

## TritNet Architecture

### Vision

Replace memory-bound LUTs with compute-bound neural networks to enable:
- GPU/TPU hardware acceleration
- Batch inference parallelization
- Learned patterns beyond hand-coded arithmetic

### Model Architecture

**TritNetUnary** (for tnot):
```
Input: [batch, 5] trits {-1, 0, +1}
    ↓
TernaryLinear: [5 → 8] weights {-1, 0, +1}
    ↓
TernaryLinear: [8 → 8] weights {-1, 0, +1}
    ↓
TernaryLinear: [8 → 5] weights {-1, 0, +1}
    ↓
sign() activation → Output: [batch, 5] trits
```

**TritNetBinary** (for tadd, tmul, tmin, tmax):
```
Input: [batch, 10] trits (5 from A, 5 from B)
    ↓
TernaryLinear: [10 → 16] weights {-1, 0, +1}
    ↓
TernaryLinear: [16 → 16] weights {-1, 0, +1}
    ↓
TernaryLinear: [16 → 5] weights {-1, 0, +1}
    ↓
sign() activation → Output: [batch, 5] trits
```

### Training Strategy

**Dataset:** Complete truth tables
- Unary: 243 samples (3^5 input states)
- Binary: 59,049 samples (3^10 input state combinations)

**Optimizer:** Adam (default PyTorch settings)
**Loss:** MSE (Mean Squared Error)
**Target:** 100% accuracy (99%+ acceptable)

**Straight-Through Estimator (STE):**
```python
# Forward: Quantize to ternary
weights_ternary = sign(weights) * (|weights| > threshold)

# Backward: Pass gradients straight through
grad_weights = grad_output  # No quantization in gradient
```

### Future Integration (Phase 3+)

**C++ inference engine:**
```cpp
// Export ternary weights from PyTorch
W1 = load_ternary_weights("W1.npy");  // int8 array {-1, 0, +1}
W2 = load_ternary_weights("W2.npy");
W3 = load_ternary_weights("W3.npy");

// Inference using ternary matmul
result = tritnet_forward(input, W1, W2, W3);
```

**GPU acceleration (Phase 4):**
```cpp
// Batch inference on GPU
batch = load_batch(inputs, batch_size=1024);
results = tritnet_forward_gpu(batch, W1, W2, W3);
```

## Performance Characteristics

### Scaling Behavior

**Small arrays (32 elements):**
- Throughput: 23-30 Mops/s
- Speedup: 135-141× vs Python
- Overhead-dominated (function call, setup)

**Medium arrays (1K-10K elements):**
- Throughput: 664-883 Mops/s
- Speedup: 2,569-3,995× vs Python
- SIMD benefits visible

**Large arrays (100K-1M elements):**
- Throughput: 11,059-35,042 Mops/s
- Peak performance zone
- Full SIMD + cache utilization

**Huge arrays (10M+ elements):**
- Throughput: 4,574-5,196 Mops/s
- Memory bandwidth limited
- Cache thrashing, DRAM bottleneck

**Optimal size:** 1M elements (peak 35,042 Mops/s)

### Latency Breakdown

**Per-element latency:**
- Python baseline: ~10 ns
- C++ scalar LUT: ~0.5 ns (5 CPU cycles @ 3 GHz)
- C++ SIMD (amortized): ~0.077 ns (0.23 CPU cycles)
- C++ fused (best case): ~0.040 ns (0.12 CPU cycles)

**SIMD advantage:** 12.9× latency reduction vs scalar
**Fusion advantage:** 1.9× latency reduction vs non-fused SIMD

## Critical Implementation Details

### 2-bit Encoding

```
Trit value | Encoding | Bit pattern
-----------+----------+------------
    -1     |   0b00   |    00
     0     |   0b01   |    01
    +1     |   0b10   |    10
 reserved  |   0b11   |    11 (undefined, triggers error)
```

### LUT Index Calculation

```cpp
// Binary operations (e.g., tadd)
index = (a << 2) | b;  // 4 bits: aaab

// Example: tadd(-1, +1)
a = 0b00, b = 0b10
index = (0b00 << 2) | 0b10 = 0b0010 = 2
result = TADD_LUT[2] = 0b01 = 0
```

### Alignment Requirements

**Streaming stores (non-temporal writes):**
- Require 32-byte alignment
- Used for arrays >1M elements
- Bypass cache, direct to memory

**Validation:**
```cpp
if (reinterpret_cast<uintptr_t>(ptr) % 32 == 0) {
    _mm256_stream_si256((__m256i*)ptr, data);  // OK
} else {
    _mm256_storeu_si256((__m256i*)ptr, data);  // Fallback
}
```

## Future Directions

### Multi-Platform SIMD

**AVX-512** (Intel Skylake-X+, AMD Zen 4+):
- 64 parallel operations (vs 32 for AVX2)
- Expected 1.8-2× throughput improvement

**ARM NEON** (ARM Cortex-A series):
- 16 parallel operations
- Mobile/edge deployment

**ARM SVE** (ARM Neoverse):
- Scalable vector length (128-2048 bits)
- Future server deployment

### GPU/TPU Acceleration

**TritNet on GPU** (Phase 4):
- Batch inference for massive parallelism
- Tensor core utilization
- Expected 10-100× throughput vs CPU LUT

**Custom ternary hardware** (Long-term):
- FPGA/ASIC designs optimized for ternary arithmetic
- Learned patterns inform hardware architecture

### Learned Generalization (Phase 5)

**Beyond exact arithmetic:**
- Approximate ternary operations for ML
- Novel learned operations not in truth tables
- Discover patterns humans haven't coded

## Summary

**Ternary Engine architecture = Hybrid approach:**
1. **LUT path** (current): Memory-bound, 35 Gops/s peak, CPU-only
2. **TritNet path** (future): Compute-bound, GPU/TPU accelerated, batched

**Key insight:** Moving from memory access to matrix multiplication unlocks $100B+ investment in ML hardware (GPUs, TPUs, tensor cores) for ternary computing.

**Phase progression:**
- Phase 1-4.0: Optimize LUT path (COMPLETE)
- Phase 1-2 TritNet: Train networks on truth tables (IN PROGRESS)
- Phase 3-4 TritNet: C++ integration, GPU acceleration (PLANNED)
- Phase 5 TritNet: Learned generalization (RESEARCH)
