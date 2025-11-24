# Ternary Engine Roadmap

**Last Updated:** 2025-11-24
**Current Version:** v1.1.0 "ktr"
**Target Version:** v1.2.0 → v2.0 → v3.0+

---

## Vision

Build a **universal ternary computing platform** with:
- **Portable scalar core** (reference implementation for all platforms)
- **Platform-specific backends** (AVX2, AVX-512, ARM SVE, RISC-V Vector, GPU)
- **Dense encoding layers** (Sixtet, Octet, Dense243)
- **Stable 25-50 Gops/s** on consumer CPUs (v1.2.0)
- **100-150 Gops/s** on AVX-512 (v2.0)
- **1-5 TOps** on GPU with TritNet (v3.0+)

---

## Architecture Principles

### **Principle 1: Separation of Concerns**
- **Mathematical core** (scalar, portable, truth-table-based)
- **Encoding layers** (Sixtet/Octet/Dense243 for I/O and cache optimization)
- **Compute backends** (SIMD-specific optimizations as plugins)

### **Principle 2: Platform Agnostic**
- Scalar reference runs everywhere (C99)
- Backends are optional performance layers
- No ISA lock-in (support x86, ARM, RISC-V, FPGA, ASIC, GPU)

### **Principle 3: Future-Proof**
- Layered design supports new backends
- TritNet enables neural network-based arithmetic
- Ready for custom silicon (FPGA/ASIC)

---

## Current Status (v1.1.0 "ktr")

### **Production-Ready** ✅
- Scalar ternary algebra (16 tests passing)
- AVX2 SIMD kernels (28.6-35.0 Gops/s)
- Dense243 encoding (5 trits/byte, validated)
- Operation fusion Phase 4.0 (1.59× - 21.65× speedup)
- Windows x64 platform (fully validated)

### **Validated & Ready** ✅
- TriadSextet encoding (6 trits in 2 bytes)
- TritNet GEMM integration (AVX2 matmul)
- Competitive benchmarks (vs NumPy INT8)

### **Pending Validation** ⚠️
- Linux/macOS builds (untested)
- Multi-platform CI (disabled for OpenMP)
- TritNet Phase 2A (tnot learning)

---

## Roadmap

### **v1.2.0 "Encoding-Aware Pipeline"** (Target: Next Release)

**Goal:** Add Sixtet/Octet layers for cache optimization and I/O efficiency without compromising portability.

**Major Features:**
1. **Encoding Layer (TEL)** - Ternary Encoding Layer
   - Sixtet pack/unpack (3 trits → 6 bits)
   - Octet pack/unpack (2 trits → 3 bits)
   - Dense243 integration
   - Portable scalar implementation

2. **Backend Interface (TCBI)** - Ternary Compute Backend Interface
   - Clean separation between scalar core and SIMD backends
   - Backend registration system
   - Runtime backend selection

3. **Safe SIMD Optimizations**
   - Canonical index LUT (removes shift/OR arithmetic)
   - LUT-256B (256-byte expanded lookup tables)
   - Dual-shuffle XOR (parallel execution on separate ports)
   - Selective interleaving (portable subset)

**Performance Targets:**
- **Sustained:** 20-35 Gops/s (stable under load)
- **Peak:** 45 Gops/s (ideal conditions)
- **Cache improvements:** +15-25% from Sixtet strip-mining

**Platform Support:**
- Windows x64 (primary)
- Linux x64 (build validation)
- macOS ARM64 (build validation)

**Breaking Changes:** None (external API unchanged)

---

### **v2.0 "SIMD Kernel v2.0"** (Future)

**Goal:** Maximum AVX2 performance through microarchitectural optimization.

**Major Features:**
1. **Index Arithmetic Elimination**
   - Remove all `(a << 2) | b` operations
   - Pure LUT-based index generation
   - Zero integer ALU pressure

2. **Pipeline Port Saturation**
   - Permute + Shuffle + XOR interleaving
   - Utilize 3 execution ports simultaneously
   - Reduce thermal variance

3. **Multi-LUT Fusion**
   - 2-op fusion (1.9× - 3.4× speedup)
   - 3-op fusion (2.8× - 5.5× speedup)
   - Fused LUTs for common patterns

**Performance Targets:**
- **Sustained:** 25-50 Gops/s (stable)
- **Peak:** 55-70 Gops/s (burst)
- **Variance:** <2× (vs 7× in v1.0)

**Platform:** x86-64 AVX2 optimized (Intel Skylake+, AMD Zen2+)

---

### **v2.5 "Multi-Platform Backends"** (Future)

**Goal:** Add backends for AVX-512, ARM NEON/SVE, RISC-V Vector.

**Backends:**
1. **AVX-512** (Intel Ice Lake+)
   - 64-element vectors
   - 100-150 Gops/s sustained

2. **ARM NEON** (Apple Silicon, ARM Cortex-A)
   - 16-element vectors
   - Mobile/embedded deployment

3. **ARM SVE/SVE2** (ARM servers, Fujitsu A64FX)
   - Variable-length vectors (128-2048 bits)
   - 200-400 Gops/s sustained

4. **RISC-V Vector** (SiFive, Alibaba T-Head)
   - Variable-length vectors
   - Future-proof open ISA

---

### **v3.0 "TritNet GPU Acceleration"** (Future)

**Goal:** Replace memory-bound LUTs with compute-bound neural network operations.

**Major Features:**
1. **TritNet Training**
   - Neural networks learn ternary operations
   - Ternary weights (1.58 bits/weight)
   - 100% accuracy requirement

2. **GPU Inference**
   - CUDA/ROCm kernels
   - Tensor core acceleration
   - Batch processing

3. **BitNet Integration**
   - Hybrid binary/ternary matmul
   - Popcount + XOR tricks
   - 10× faster than NumPy matmul

**Performance Targets:**
- **GPU:** 1-5 TOps (trillion ops/sec)
- **Batch efficiency:** 30-50× speedup vs scalar
- **Model quantization:** TinyLlama, Phi-2, Gemma

---

### **v4.0+ "Hardware Acceleration"** (Long-term)

**Goal:** Custom silicon and FPGA implementations.

**Platforms:**
1. **FPGA** (Xilinx/Altera)
   - 100-300 Gops/s
   - HDL generation from scalar core
   - Reconfigurable logic

2. **ASIC** (Custom silicon)
   - 2-10 TOps
   - Ternary ALU units
   - In-memory LUT arrays

3. **NPU/TPU** Integration
   - Google Edge TPU
   - Qualcomm Hexagon
   - Apple Neural Engine

---

## Detailed v1.2.0 Implementation Plan

### **Phase 1: Encoding Layer**

**Sixtet Implementation**
- 3 trits → 6 bits packing
- LUT-based pack/unpack (64-entry tables)
- Strip-mining for L1 cache optimization
- Branchless encoding/decoding

**Octet Implementation**
- 2 trits → 3 bits with canonical mapping
- 7 valid states + 1 sentinel
- Byte-aligned for DMA/GPU transfers
- Error detection support

**Dense243 Integration**
- Existing implementation (5 trits/byte)
- Positional base-3 encoding
- Storage/network transport optimized

**Files:**
- `src/core/packing/sixtet_pack.h`
- `src/core/packing/octet_pack.h`
- `src/core/packing/pack.h`
- `src/core/packing/unpack.h`

---

### **Phase 2: Backend Interface**

**Ternary Compute Backend Interface (TCBI)**
```cpp
struct TernaryBackend {
    const char* name;
    bool (*detect)(void);
    void (*tadd)(const uint8_t*, const uint8_t*, uint8_t*, size_t);
    void (*tmul)(const uint8_t*, const uint8_t*, uint8_t*, size_t);
    void (*tmin)(const uint8_t*, const uint8_t*, uint8_t*, size_t);
    void (*tmax)(const uint8_t*, const uint8_t*, uint8_t*, size_t);
    void (*tnot)(const uint8_t*, uint8_t*, size_t);
};
```

**Backend Registration**
- Runtime detection (CPUID for AVX2/AVX-512)
- Backend priority system
- Fallback to scalar reference

**Backends:**
- `scalar` (C99 reference, always available)
- `avx2_v1` (existing SIMD, portable subset)
- `avx2_v2` (future optimized, x86-specific)

**Files:**
- `src/core/backend/backend_interface.h`
- `src/core/backend/backend_scalar.c`
- `src/core/backend/backend_avx2_v1.cpp`
- `src/core/backend/backend_registry.c`

---

### **Phase 3: Safe SIMD Optimizations**

**Canonical Index LUT**
- Pre-computed index mapping
- Eliminates `(a << 2) | b` at runtime
- 16-byte or 256-byte LUT options

**LUT-256B Expansion**
- 256-byte lookup tables (4 cache lines)
- Direct byte indexing (no bit manipulation)
- Fits in L1 data cache

**Dual-Shuffle XOR**
```cpp
__m256i lo = _mm256_shuffle_epi8(LUT_LO, a);
__m256i hi = _mm256_shuffle_epi8(LUT_HI, b);
__m256i out = _mm256_xor_si256(lo, hi);
```
- Two parallel shuffles (separate execution ports)
- Zero-latency XOR fusion
- 1.5-1.7× speedup on Zen CPUs

**Files:**
- `src/core/simd/ternary_simd_kernels_v2.h`
- `src/core/algebra/ternary_lut_256.h`
- `src/core/simd/ternary_canonical_index.h`

---

### **Phase 4: Testing & Validation**

**Correctness Tests**
- Sixtet pack/unpack round-trip
- Octet encoding validation
- Backend equivalence tests
- Cross-platform consistency

**Performance Benchmarks**
- Cache pressure reduction (Sixtet)
- Backend selection overhead
- Canonical index speedup
- Dual-shuffle XOR gains

**Platform Validation**
- Windows x64 (primary)
- Linux x64 (Docker/CI)
- macOS ARM64 (CI)

**Files:**
- `tests/cpp/test_sixtet.cpp`
- `tests/cpp/test_octet.cpp`
- `tests/python/test_backends.py`
- `benchmarks/bench_encoding.py`

---

### **Phase 5: Documentation**

**Technical Documentation**
- Encoding layer design document
- Backend interface specification
- SIMD optimization guide
- Platform support matrix

**API Documentation**
- Sixtet/Octet usage examples
- Backend selection API
- Performance tuning guide
- Migration from v1.1.0

**Files:**
- `docs/architecture/encoding-layer.md`
- `docs/architecture/backend-interface.md`
- `docs/performance/simd-optimizations.md`
- `docs/migration/v1.1-to-v1.2.md`

---

## Performance Projections

### **AVX2 Theoretical Limits**

| Version | Ops/Cycle | Clock | Theoretical | Measured Stable | Measured Peak |
|:--------|----------:|------:|------------:|----------------:|--------------:|
| v1.0    | 32        | 3.5 GHz | 112 Gops/s | 12-28 Gops/s | 35 Gops/s |
| v1.1    | 32        | 3.8 GHz | 122 Gops/s | 28 Gops/s | 35 Gops/s |
| v1.2    | 48        | 3.8 GHz | 182 Gops/s | 30 Gops/s | 45 Gops/s |
| v2.0    | 64        | 4.0 GHz | 256 Gops/s | 40 Gops/s | 65 Gops/s |

### **Platform Projections**

| Platform | Version | Sustained | Peak | Efficiency |
|:---------|:--------|----------:|-----:|-----------:|
| AVX2 (Zen2) | v1.2 | 30 Gops/s | 45 Gops/s | 25% |
| AVX2 (Zen3) | v2.0 | 45 Gops/s | 65 Gops/s | 35% |
| AVX-512 (Ice Lake) | v2.5 | 120 Gops/s | 180 Gops/s | 60% |
| ARM SVE (A64FX) | v2.5 | 250 Gops/s | 400 Gops/s | 50% |
| GPU (RTX 3050) | v3.0 | 2 TOps | 5 TOps | ~40% |

---

## Dependencies & Requirements

### **Core Requirements**
- C++17 compiler (MSVC, GCC 9+, Clang 10+)
- Python 3.7+ (for bindings)
- pybind11 2.6+
- NumPy 1.19+

### **Platform-Specific**
- **AVX2:** Intel Haswell (2013+), AMD Excavator (2015+)
- **AVX-512:** Intel Ice Lake (2019+), AMD Zen4 (2022+)
- **ARM NEON:** ARMv7-A+, Apple M1+
- **ARM SVE:** ARMv8.2-A+, Fujitsu A64FX

### **Optional**
- PyTorch 2.0+ (for TritNet)
- CUDA 11.0+ (for GPU backend)
- OpenMP (for multi-threading)

---

## Success Metrics

### **v1.2.0 Success Criteria**
- ✅ Sixtet/Octet encoding implementations complete
- ✅ Backend interface working (scalar + AVX2)
- ✅ No performance regression from v1.1.0
- ✅ +15% sustained throughput from cache optimization
- ✅ Builds on Windows/Linux/macOS
- ✅ All tests passing
- ✅ Documentation complete

### **v2.0 Success Criteria**
- ✅ 40+ Gops/s sustained on Zen3/Skylake
- ✅ <2× performance variance (vs 7× in v1.0)
- ✅ Multi-LUT fusion operational
- ✅ Thermal stability improvements
- ✅ Comprehensive benchmarks

### **v3.0 Success Criteria**
- ✅ TritNet training reaches 100% accuracy
- ✅ GPU backend operational
- ✅ 1+ TOps sustained on consumer GPU
- ✅ Model quantization (TinyLlama, Phi-2)

---

## Timeline & Milestones

**v1.2.0 Development:**
- Phase 1 (Encoding): 6 weeks
- Phase 2 (Backend Interface): 4 weeks
- Phase 3 (SIMD Optimizations): 8 weeks
- Phase 4 (Testing): 4 weeks
- Phase 5 (Documentation): 2 weeks
- **Total:** ~24 weeks (~6 months)

**v2.0 Development:**
- Advanced SIMD: 12 weeks
- Multi-LUT fusion: 8 weeks
- Thermal optimization: 4 weeks
- **Total:** ~24 weeks (~6 months)

**v3.0 Development:**
- TritNet training: 12 weeks
- GPU kernels: 16 weeks
- Model integration: 8 weeks
- **Total:** ~36 weeks (~9 months)

---

## Open Questions

1. **Sixtet vs TriadSextet:** Which to prioritize in v1.2.0?
   - Sixtet: 3 trits → 6 bits (canonical, simpler)
   - TriadSextet: 6 trits → 16 bits (higher density, more complex)

2. **Backend dispatch overhead:** Acceptable cost for runtime selection?
   - Virtual function calls: ~2-5 cycles
   - Function pointers: ~1-3 cycles
   - Static dispatch: 0 cycles (compile-time only)

3. **Multi-platform CI:** Re-enable OpenMP tests after fixes?
   - OpenMP crashes on CI (root cause fixed in v1.0)
   - Needs validation across platforms

4. **TritNet accuracy requirement:** 100% or 99%+ acceptable?
   - 100%: Perfect arithmetic (required for exact computation)
   - 99%+: Approximate arithmetic (sufficient for ML/AI)

---

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines.

**Roadmap discussions:** [GitHub Discussions](https://github.com/gesttaltt/ternary-engine/discussions)

**Issue tracking:** [GitHub Issues](https://github.com/gesttaltt/ternary-engine/issues)

---

## References

### **Internal Documentation**
- [TRITNET_ROADMAP.md](./TRITNET_ROADMAP.md) - TritNet neural network learning
- [architecture/optimization-roadmap.md](./architecture/optimization-roadmap.md) - SIMD optimizations
- [TECHNICAL_DEBT_CATALOG.md](./TECHNICAL_DEBT_CATALOG.md) - Known issues
- [BITNET_INTEGRATION_STRATEGY.md](./BITNET_INTEGRATION_STRATEGY.md) - BitNet hybrid approach

### **Local Reports** (not in git)
- `local-reports/opt.md` - Comprehensive SIMD optimization guide v2.0
- `local-reports/tpo.md` - v1.2.0 architecture proposal
- `local-reports/2025-11-24/PERFORMANCE_INVESTIGATION.md` - 35 Gops/s validation

### **External Resources**
- [Balanced Ternary](https://en.wikipedia.org/wiki/Balanced_ternary) - Wikipedia
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/) - AVX2/AVX-512
- [ARM SVE Programming](https://developer.arm.com/documentation/102476/latest/) - Scalable Vector Extension
- [RISC-V Vector Extension](https://github.com/riscv/riscv-v-spec) - RVV specification

---

**Last Updated:** 2025-11-24
**Maintainers:** Jonathan Verdun, Ternary Engine Contributors
**License:** Apache 2.0
