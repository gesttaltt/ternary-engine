# Memory Efficiency Benchmark Suite

**Doc-Type:** Benchmark Documentation · Version 1.0 · Updated 2025-12-03

---

## Overview

This benchmark measures the **TRUE value proposition** of ternary encoding: **MEMORY EFFICIENCY**, not raw throughput.

For AI model inference, memory bandwidth is often the bottleneck. Ternary encoding provides:
- **4x compression** vs INT8 (2-bit vs 8-bit)
- **8x compression** vs FP16 (2-bit vs 16-bit)
- **16x compression** vs FP32 (2-bit vs 32-bit)

---

## Files

| File | Purpose |
|:-----|:--------|
| `bench_memory_efficiency.cpp` | Main benchmark implementation |
| `include/bench_memory.h` | C-compatible memory efficiency API |
| `build_memory_bench.bat` | Windows build script |

---

## Key Metrics

| Metric | Description |
|:-------|:------------|
| **Bits/Element** | Storage bits per logical element |
| **Gops/s** | Throughput in billions of ops/second |
| **GB/s** | Memory bandwidth achieved |
| **Ops/Byte** | Computational density (higher = better) |
| **Compression** | Ratio vs FP32 (32 bits/element) |

---

## Compilation

### Windows (Developer Command Prompt)

```cmd
cd benchmarks\cpp-native-kernels
build_memory_bench.bat
```

Or manually:
```cmd
cl /O2 /arch:AVX2 /std:c++17 /EHsc /I..\..\src /I.\include ^
   bench_memory_efficiency.cpp /Fe:bin\bench_memory.exe
```

### Linux/macOS

```bash
g++ -O3 -march=native -mavx2 -std=c++17 -I../../src -I./include \
    bench_memory_efficiency.cpp -o bin/bench_memory
```

---

## Usage

```bash
./bench_memory              # Full benchmark
./bench_memory --quiet      # Minimal output
./bench_memory --no-models  # Skip AI model comparison
```

---

## What This Benchmark Measures

### 1. Cache-Level Performance

Tests at different memory hierarchy levels:
- **L1 Cache (8K elements)**: ~32 KB data
- **L2 Cache (64K elements)**: ~256 KB data
- **L2 Boundary (256K)**: ~1 MB data
- **L3 Cache (1M elements)**: ~4 MB data
- **Main Memory (16M elements)**: ~64 MB data

### 2. Format Comparison

For each cache level, compares:
- **Ternary-2bit**: Our implementation (2 bits/element)
- **INT8**: Standard 8-bit integers
- **FP32**: Standard 32-bit floats

### 3. AI Model Size Impact

Shows memory requirements for popular models:

| Model | FP32 | FP16 | INT8 | INT4 | Ternary |
|:------|:-----|:-----|:-----|:-----|:--------|
| TinyLlama-1.1B | 4.4 GB | 2.2 GB | 1.1 GB | 0.55 GB | **0.28 GB** |
| Phi-2 (2.7B) | 10.8 GB | 5.4 GB | 2.7 GB | 1.35 GB | **0.68 GB** |
| LLaMA-7B | 28 GB | 14 GB | 7 GB | 3.5 GB | **1.75 GB** |
| LLaMA-13B | 52 GB | 26 GB | 13 GB | 6.5 GB | **3.25 GB** |
| LLaMA-70B | 280 GB | 140 GB | 70 GB | 35 GB | **17.5 GB** |

---

## Why Memory Efficiency Matters

### The Memory Wall

Modern CPUs/GPUs are compute-bound only for small data:
- L1 cache: Compute-bound (full throughput)
- L2 cache: Transitioning
- L3 cache: Often bandwidth-bound
- Main memory: **Always bandwidth-bound**

### Ternary Advantage

When memory bandwidth is the bottleneck:
```
Theoretical throughput at 50 GB/s bandwidth:
- Ternary: 66.7 Gops/s (2 bits × 3 arrays / 8 = 0.75 bytes/op)
- INT8:    16.7 Gops/s (8 bits × 3 arrays / 8 = 3 bytes/op)
- FP32:     4.2 Gops/s (32 bits × 3 arrays / 8 = 12 bytes/op)
```

Ternary can process **4x more operations** than INT8 at the same bandwidth.

---

## Computational Density

**Ops per byte** measures how much useful work is done per byte transferred:

| Format | Bytes/Element | Bytes/Op (binary) | Ops/Byte |
|:-------|:--------------|:------------------|:---------|
| Ternary | 0.25 | 0.75 | **1.33** |
| INT8 | 1.0 | 3.0 | 0.33 |
| FP32 | 4.0 | 12.0 | 0.083 |

Ternary is **4x more compute-dense** than INT8.

---

## Practical Applications

### Edge Deployment
- Mobile devices with 4-8 GB RAM
- IoT devices with < 1 GB RAM
- Ternary enables large models on small devices

### GPU VRAM Optimization
- Consumer GPUs: 8-16 GB VRAM
- Ternary enables 70B models on 16 GB GPU
- Run larger models without A100/H100

### Inference Latency
- Bandwidth-bound inference benefits from compression
- Fewer bytes to transfer = lower latency
- Important for real-time applications

---

## Trade-offs

### Ternary Advantages
- 4x memory compression
- 4x higher computational density
- Enables larger models on constrained hardware

### Ternary Disadvantages
- Lower raw throughput than optimized INT8/FP32
- Requires quantization (potential accuracy loss)
- More complex operations (LUT-based)

### When to Use Ternary
- Memory-constrained deployment
- Bandwidth-limited inference
- Edge/mobile devices
- Large model deployment

### When NOT to Use Ternary
- Training (need FP32 gradients)
- Accuracy-critical applications
- Compute-bound workloads with plenty of memory

---

## Integration with Other Benchmarks

This suite complements:
- `bench_gops_comparative.cpp` - Raw throughput comparison
- `bench_kernels.cpp` - SIMD vs scalar performance
- Python benchmarks - Higher-level integration

---

## Version History

| Date | Version | Description |
|:-----|:--------|:------------|
| 2025-12-03 | 1.0 | Initial memory efficiency benchmark suite |
