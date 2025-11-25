# Hybrid Architecture Roadmap - Ternary Engine v3.0

**Type:** Strategic Roadmap
**Date:** 2025-11-25
**Status:** Planning Phase
**Goal:** Adaptive semantic-cold hybrid architecture for 70+ Gops/s sustained

---

## Executive Summary

**Current Achievement (v1.3.0):**
- 45.3 Gops/s effective throughput (90% of 50 Gops/s target)
- Canonical indexing successfully integrated from archived backends
- Direct kernels + geometric optimization = best of both worlds

**Next Phase (v3.0):**
- Adaptive hybrid architecture with runtime path selection
- Comprehensive invariant measurement suite
- Self-optimizing kernel based on input characteristics
- Target: 70+ Gops/s sustained, 100+ Gops/s peak in geometric regions

---

## Architectural Philosophy

### The Dual Invariant Discovery

**Invariant A - Cold Arithmetic Path (Current Direct Kernels):**
- Minimal branching, minimal dispatch
- High IPC, low jitter
- Optimal for stable, deterministic kernels
- Predictable performance across random inputs

**Invariant B - Semantic Encoding Geometry (Canonical Indexing):**
- Dual-shuffle creates "geometric collapses" in ternary space
- Massive speedups in patterns with strong trit correlation
- Reduces dependency chains through semantic compression
- Performance scales with input structure

**Key Insight:** These are NOT competing approaches - they optimize DIFFERENT dimensions of the ternary computation space.

**v1.3.0 Achievement:** Successfully combined both invariants in a single unified kernel
- Cold path: Direct AVX2 implementation (minimal dispatch)
- Geometric path: Canonical indexing with dual-shuffle
- Result: 74-1,100% improvements

**v3.0 Vision:** Adaptive selection between multiple optimized paths based on runtime input characteristics

---

## What Works Now (v1.3.0 Baseline)

### ✅ Integrated Optimizations

1. **Canonical Indexing (Dual-Shuffle)** - ACTIVE since 2025-11-25
   - Dual-shuffle + ADD indexing pattern
   - Integrated from ternary_backend_avx2_v2.cpp
   - 74-1,100% measured improvements

2. **Direct Cold Path** - ACTIVE
   - Zero-dispatch AVX2 kernels
   - Single compilation unit
   - Simple, maintainable architecture

3. **Operation Fusion** - ACTIVE
   - 4 fusion patterns (Binary→Unary)
   - 15.93× peak speedup (fused_tnot_tadd)
   - 45.3 Gops/s effective throughput

4. **OpenMP Parallelization** - ACTIVE
   - 12 threads auto-configured
   - Fixed during canonical indexing integration
   - Scales properly for n ≥ 100K elements

5. **Three-Path Architecture** - ACTIVE
   - OpenMP path: n ≥ 100K
   - SIMD path: 1K ≤ n < 100K
   - Scalar path: n < 1K

### 📦 Available But Not Active

6. **Dense243 Encoding**
   - Module exists, validated
   - 5 trits/byte (95.3% density)
   - Not integrated into core operations

7. **PGO (Profile-Guided Optimization)**
   - Scripts exist (build_pgo_unified.py)
   - Not used in recent builds
   - Expected 5-15% additional gain

8. **Backend System (Archived)**
   - Multi-ISA dispatch infrastructure
   - Preserved in src/core/simd/
   - Useful for future FPGA/ASIC/PLC deployment

---

## v3.0 Roadmap - Three-Phase Plan

### Phase 1: Invariant Measurement Suite (1-2 weeks)

**Goal:** Build comprehensive benchmark suite that measures WHY performance varies

**Components to Measure:**

#### 1.1 Cold Invariants (Hardware Metrics)
- **IPC (Instructions Per Cycle)** - Using performance counters
- **L1/L2 cache miss ratio** - Memory access patterns
- **Register pressure** - AVX2 register usage
- **µop fusion** - Instruction-level parallelism
- **Dependency depth** - Critical path length
- **Shuffle throughput** - Port 5 saturation on Intel/AMD

#### 1.2 Geometric Invariants (Semantic Metrics)
- **Local entropy** - Shannon entropy of trit sequences
- **Inter-trit correlation** - Autocorrelation analysis
- **Repetitiveness score** - Pattern compression ratio
- **Fractal self-similarity** - Hausdorff dimension estimate
- **±1/0 triad distribution** - Statistical trit balance
- **Semantic distance compression** - LUT locality measure

#### 1.3 Meta-Analysis (Pattern Discovery)
- **Clustering** - KMeans/DBSCAN on performance profiles
- **Dimensionality reduction** - PCA/UMAP projection
- **Geometry scoring** - Ternary manifold regions
- **Anomaly detection** - Identify "breakpoint" patterns
- **Architecture preference map** - Which inputs favor which path

**Deliverables:**
```
benchmarks/bench_invariants.py        # Main measurement suite
benchmarks/utils/hardware_metrics.py  # CPU counters (via perf/VTune)
benchmarks/utils/geometric_metrics.py # Entropy, correlation, fractals
benchmarks/analysis/cluster_analysis.py # Pattern discovery
reports/invariant_analysis_YYYY-MM-DD.md # Findings report
```

**Validation Criteria:**
- Identify ≥3 distinct performance regions with statistical significance
- Measure variance in each invariant across 1000+ random inputs
- Discover correlation between geometric metrics and speedup

---

### Phase 2: Dynamic Route Selector (2-3 weeks)

**Goal:** Implement adaptive path selection based on measured invariants

#### 2.1 Architecture Design

**Hybrid Kernel Structure:**
```cpp
// Two optimized paths in unified kernel
__m256i hybrid_binary_op(__m256i a, __m256i b, __m256i lut,
                         PathSelector selector) {
    if (selector.use_geometric_path()) {
        // High correlation detected → canonical indexing path
        return geometric_optimized_path(a, b, lut);
    } else {
        // Low correlation → direct cold path
        return cold_optimized_path(a, b, lut);
    }
}
```

**Path Selection Heuristics:**
```cpp
struct PathSelector {
    // Lightweight runtime checks
    bool use_geometric_path() {
        return (local_entropy < THRESHOLD) &&
               (correlation_score > MIN_CORRELATION);
    }

    // Compile-time optimization via templates
    template<PathType P>
    static constexpr bool select_at_compile_time();
};
```

#### 2.2 Implementation Strategy

**Option A: Runtime Adaptive (Dynamic)**
- Measure input characteristics in first N elements
- Select path for rest of array
- Overhead: ~1-5% for measurement
- Benefit: Optimal path for each input

**Option B: Compile-Time Specialization (Static)**
- PGO-driven path selection
- Zero runtime overhead
- Benefit: Maximum performance for known workloads

**Option C: Hybrid (Recommended)**
- PGO provides default path per operation type
- Runtime check for anomalous inputs
- Best of both worlds

#### 2.3 Path Variants to Implement

1. **Cold Path (Current Default)**
   - Direct AVX2 kernels
   - Zero dispatch
   - Best for: Random inputs, high entropy

2. **Geometric Path (Canonical Indexing)**
   - Dual-shuffle + ADD indexing
   - Best for: Correlated inputs, low entropy

3. **Dense243 Path (New)**
   - 5 trits/byte packed encoding
   - Best for: Large arrays, memory-bound workloads

4. **Fusion Path (Current)**
   - Multi-operation chains
   - Best for: Expression sequences

**Deliverables:**
```
src/core/simd/path_selector.h           # Selection logic
src/core/simd/ternary_hybrid_kernels.h  # Unified hybrid implementation
src/engine/bindings_hybrid_ops.cpp      # Python bindings
tests/test_hybrid_selector.py           # Validation tests
benchmarks/bench_hybrid.py              # Performance validation
```

**Validation Criteria:**
- Hybrid kernel matches or exceeds best single-path performance
- Overhead of path selection < 5%
- Geometric path triggers correctly on synthetic low-entropy inputs
- Cold path triggers correctly on random inputs

---

### Phase 3: Advanced Optimizations (3-4 weeks)

**Goal:** Push beyond 70 Gops/s with advanced techniques

#### 3.1 Profile-Guided Optimization Integration

**Action:** Enable PGO for automatic path optimization
```bash
python build/build_pgo_unified.py --clang --profile-workload=geometric
```

**Expected Gain:** 5-15% additional throughput
**Result:** 45.3 × 1.10 = 49.8 Gops/s → crosses 50 Gops/s milestone

#### 3.2 Dense243 Integration

**Action:** Integrate Dense243 encoding into main operations
- Add pack/unpack in Python layer
- Optimize for memory-bound workloads (10M+ elements)
- Enable runtime format selection

**Expected Gain:** 20-30% on large arrays (memory bandwidth reduction)
**Result:** 45.3 × 1.25 = 56.6 Gops/s

#### 3.3 Advanced Fusion Patterns

**Action:** Implement 3-operation fusion chains
- tadd(tmul(a, b), c) - multiply-add fusion
- tnot(tadd(tmul(a, b), c)) - 3-op fusion
- Pattern auto-detection in Python layer

**Expected Gain:** 2-3× effective throughput on fusion workloads
**Result:** 45.3 × 2 = 90.6 Gops/s effective

#### 3.4 Cache Blocking for Large Arrays

**Action:** Implement cache-aware blocking for 10M+ element arrays
- Tile operations to L2/L3 cache size
- Reduces memory bandwidth saturation
- Improves sustained throughput

**Expected Gain:** 30-50% on very large arrays (10M+)

**Deliverables:**
```
src/core/simd/ternary_fusion_advanced.h  # 3-op fusion patterns
src/engine/lib/dense243_integration/     # Dense243 integration layer
build/build_pgo_geometric.py             # PGO with geometric profiling
reports/v3.0_performance_final.md        # Final validation report
```

**Validation Criteria:**
- Peak throughput: ≥70 Gops/s element-wise
- Effective throughput: ≥90 Gops/s with advanced fusion
- Sustained throughput: ≥60 Gops/s on 10M+ elements
- Memory efficiency: 5× improvement with Dense243

---

## Detailed Action Plan - Execution Order

### Week 1-2: Invariant Measurement

**Day 1-2: Hardware Metrics Infrastructure**
- Integrate CPU performance counters (Intel PCM or Linux perf)
- Implement IPC, cache miss, µop measurements
- Validate on synthetic workloads

**Day 3-5: Geometric Metrics Implementation**
- Implement entropy calculation (Shannon entropy on trit sequences)
- Implement autocorrelation for inter-trit correlation
- Implement fractal dimension estimation (box-counting)
- Validate on synthetic datasets (low/medium/high entropy)

**Day 6-8: Benchmark Suite Development**
- Create synthetic datasets covering entropy spectrum
- Run full measurement suite on 1000+ random inputs
- Collect baseline performance profiles

**Day 9-10: Meta-Analysis**
- Implement clustering (KMeans on performance profiles)
- Generate PCA/UMAP visualizations
- Identify performance regions and breakpoints
- Write `reports/invariant_analysis_YYYY-MM-DD.md`

**Deliverables:**
- [ ] `benchmarks/bench_invariants.py` - Comprehensive measurement suite
- [ ] `benchmarks/utils/hardware_metrics.py` - CPU counters integration
- [ ] `benchmarks/utils/geometric_metrics.py` - Entropy, correlation, fractals
- [ ] `benchmarks/analysis/cluster_analysis.py` - Pattern discovery tools
- [ ] `reports/invariant_analysis_YYYY-MM-DD.md` - Findings and insights
- [ ] Synthetic datasets in `benchmarks/datasets/synthetic/`

---

### Week 3-4: Dynamic Route Selector

**Day 11-13: Path Selector Design**
- Design PathSelector API (runtime + compile-time variants)
- Implement lightweight entropy estimator for runtime checks
- Define threshold values based on Week 1-2 analysis

**Day 14-16: Hybrid Kernel Implementation**
- Refactor ternary_simd_kernels.h for two-path architecture
- Implement geometric_optimized_path() (canonical indexing)
- Implement cold_optimized_path() (direct kernels)
- Add template-based compile-time selection

**Day 17-18: Python Bindings Integration**
- Update bindings_core_ops.cpp for hybrid operations
- Add Python API for path selection control
- Expose selector statistics for debugging

**Day 19-20: Validation and Benchmarking**
- Write comprehensive tests for hybrid selector
- Validate path selection correctness on synthetic datasets
- Benchmark overhead of path selection (<5% requirement)
- Measure performance on real workloads

**Deliverables:**
- [ ] `src/core/simd/path_selector.h` - Selection logic and heuristics
- [ ] `src/core/simd/ternary_hybrid_kernels.h` - Unified implementation
- [ ] `src/engine/bindings_hybrid_ops.cpp` - Python bindings
- [ ] `tests/test_hybrid_selector.py` - Correctness validation
- [ ] `benchmarks/bench_hybrid.py` - Performance validation
- [ ] `docs/hybrid_architecture.md` - Architecture documentation

---

### Week 5-6: PGO and Dense243 Integration

**Day 21-23: Profile-Guided Optimization**
- Set up PGO workflow for geometric workloads
- Profile with representative low/medium/high entropy datasets
- Rebuild with PGO enabled
- Validate 5-15% improvement

**Day 24-26: Dense243 Integration**
- Refactor Dense243 module for core integration
- Implement pack/unpack in Python bindings layer
- Add runtime format selection (2-bit vs Dense243)
- Benchmark memory bandwidth reduction

**Day 27-28: Advanced Fusion Patterns**
- Implement 3-operation fusion chains
- Add pattern detection in Python layer
- Benchmark effective throughput (target: 90+ Gops/s)

**Day 29-30: Cache Blocking**
- Implement cache-aware tiling for large arrays
- Tune block sizes for L2/L3 cache
- Validate sustained throughput on 10M+ elements

**Deliverables:**
- [ ] `build/build_pgo_geometric.py` - PGO build for geometric workloads
- [ ] `src/engine/lib/dense243_integration/` - Integration layer
- [ ] `src/core/simd/ternary_fusion_advanced.h` - 3-op fusion
- [ ] `benchmarks/bench_cache_blocking.py` - Cache blocking validation
- [ ] `reports/v3.0_performance_final.md` - Final achievement report

---

## Expected Performance Milestones

### Milestone 1: Invariant Measurement Complete (Week 2)
- **Deliverable:** Comprehensive understanding of performance regions
- **Metrics:** Statistical clustering of 1000+ inputs into ≥3 distinct regions
- **Decision Point:** Confirm correlation between geometric metrics and speedup

### Milestone 2: Hybrid Selector Validated (Week 4)
- **Deliverable:** Self-optimizing kernel with path selection
- **Metrics:** Hybrid matches or beats best single-path performance
- **Performance Target:** Maintain 45.3 Gops/s baseline, achieve 50+ Gops/s on geometric inputs

### Milestone 3: PGO Integration (Week 5)
- **Deliverable:** PGO-optimized build with geometric profiling
- **Performance Target:** 50 Gops/s milestone crossed (45.3 × 1.10 = 49.8)

### Milestone 4: Dense243 Integration (Week 5-6)
- **Deliverable:** Memory-efficient encoding for large arrays
- **Performance Target:** 56+ Gops/s on memory-bound workloads (45.3 × 1.25)

### Milestone 5: Advanced Fusion (Week 6)
- **Deliverable:** 3-operation fusion chains
- **Performance Target:** 90+ Gops/s effective throughput (45.3 × 2)

### Final Target: v3.0 Release
- **Element-wise peak:** 70+ Gops/s (vs current 39.1)
- **Effective throughput:** 90+ Gops/s with advanced fusion (vs current 45.3)
- **Sustained throughput:** 60+ Gops/s on large arrays (vs current 35-39)
- **Memory efficiency:** 5× improvement with Dense243
- **Adaptive architecture:** Self-optimizing based on input characteristics

---

## Technical Components

### 1. Invariant Measurement Suite

**File:** `benchmarks/bench_invariants.py`

**Measurements:**
```python
class InvariantMeasurement:
    # Cold invariants (hardware)
    ipc: float                    # Instructions per cycle
    l1_miss_rate: float          # L1 cache miss ratio
    l2_miss_rate: float          # L2 cache miss ratio
    register_pressure: int       # AVX2 register spills
    dependency_depth: int        # Critical path length
    shuffle_throughput: float    # Port 5 utilization

    # Geometric invariants (semantic)
    entropy: float               # Shannon entropy of trit sequence
    correlation: float           # Autocorrelation coefficient
    repetitiveness: float        # Compression ratio
    fractal_dimension: float     # Hausdorff dimension
    triad_balance: dict          # Distribution of {-1, 0, +1}

    # Meta-analysis
    cluster_id: int              # Performance region
    anomaly_score: float         # Statistical outlier measure
```

**Synthetic Datasets:**
```python
def generate_dataset(entropy_level='low'|'medium'|'high'):
    """
    Generate synthetic ternary arrays with controlled characteristics:
    - low: repetitive patterns, high correlation
    - medium: pseudo-Markov with local structure
    - high: cryptographic random
    """
```

---

### 2. Hybrid Architecture

**File:** `src/core/simd/ternary_hybrid_kernels.h`

**Design:**
```cpp
// Path selection based on runtime or compile-time heuristics
enum class OptimizationPath {
    COLD_PATH,       // Direct AVX2, zero dispatch
    GEOMETRIC_PATH,  // Canonical indexing, dual-shuffle
    DENSE243_PATH,   // Packed 5-trits/byte encoding
    FUSION_PATH      // Multi-operation chains
};

// Lightweight path selector
struct PathSelector {
    static OptimizationPath select(const uint8_t* data, size_t n) {
        // Quick entropy estimate on first N elements
        float entropy = estimate_entropy(data, std::min(n, 1024));

        if (entropy < LOW_ENTROPY_THRESHOLD) {
            return OptimizationPath::GEOMETRIC_PATH;  // High correlation
        } else if (n > LARGE_ARRAY_THRESHOLD) {
            return OptimizationPath::DENSE243_PATH;   // Memory-bound
        } else {
            return OptimizationPath::COLD_PATH;       // Default
        }
    }
};

// Unified hybrid kernel
template <bool Sanitize = true>
static inline __m256i hybrid_binary_op(__m256i a, __m256i b, __m256i lut,
                                       OptimizationPath path) {
    switch (path) {
        case OptimizationPath::GEOMETRIC_PATH:
            return geometric_binary_op<Sanitize>(a, b, lut);
        case OptimizationPath::COLD_PATH:
        default:
            return cold_binary_op<Sanitize>(a, b, lut);
    }
}
```

---

### 3. Advanced Fusion Patterns

**File:** `src/core/simd/ternary_fusion_advanced.h`

**Patterns:**
```cpp
// 3-operation fusion: tadd(tmul(a, b), c)
// Equivalent to: temp = tmul(a, b); result = tadd(temp, c)
// Fused: Single pass, no intermediate allocation
__m256i fused_mul_add(__m256i a, __m256i b, __m256i c,
                      __m256i lut_mul, __m256i lut_add) {
    // Fuse: Load a, b, c once → compute mul → add → write once
    __m256i temp = binary_simd_op<false>(a, b, lut_mul);
    return binary_simd_op<false>(temp, c, lut_add);
}

// 4-operation fusion: tnot(tadd(tmul(a, b), c))
__m256i fused_not_mul_add(__m256i a, __m256i b, __m256i c,
                          __m256i lut_mul, __m256i lut_add,
                          __m256i lut_not) {
    __m256i temp1 = binary_simd_op<false>(a, b, lut_mul);
    __m256i temp2 = binary_simd_op<false>(temp1, c, lut_add);
    return unary_simd_op<false>(temp2, lut_not);
}
```

**Expected Speedup:**
- 2-op fusion: 1.5-16× (validated)
- 3-op fusion: 2-3× (estimated)
- 4-op fusion: 3-4× (estimated)

---

## Risk Assessment and Mitigation

### Risk 1: Path Selection Overhead
**Risk:** Runtime path selection adds 5-10% overhead
**Mitigation:**
- Implement compile-time path selection via PGO
- Amortize cost over large arrays (select once, process thousands)
- Cache selector state between operations

### Risk 2: Increased Code Complexity
**Risk:** Hybrid architecture harder to maintain than direct kernels
**Mitigation:**
- Template-based unification (single source, multiple paths)
- Comprehensive test coverage for all paths
- Clear architectural documentation

### Risk 3: PGO May Not Help
**Risk:** PGO profiling with geometric workloads may not generalize
**Mitigation:**
- Profile with diverse workload mix
- Fall back to runtime selection if PGO ineffective
- Validate on real TritNet/VAE workloads

### Risk 4: Dense243 Integration Complexity
**Risk:** Pack/unpack overhead negates memory bandwidth savings
**Mitigation:**
- Only use Dense243 for large arrays (10M+ elements)
- Benchmark pack/unpack cost vs memory bandwidth gain
- Implement streaming pack/unpack with SIMD

---

## Success Metrics

### Performance Metrics
- [ ] Element-wise peak: ≥70 Gops/s (vs 39.1 current)
- [ ] Effective throughput: ≥90 Gops/s (vs 45.3 current)
- [ ] Sustained throughput: ≥60 Gops/s on 10M+ (vs 35-39 current)
- [ ] Path selection overhead: <5%
- [ ] PGO improvement: 5-15%
- [ ] Dense243 memory reduction: 5×

### Engineering Metrics
- [ ] Test coverage: 100% for hybrid paths
- [ ] Documentation: Complete API reference
- [ ] Benchmark reproducibility: <3% variance
- [ ] Code maintainability: Single-source hybrid implementation

### Research Metrics
- [ ] Invariant discovery: ≥3 distinct performance regions
- [ ] Statistical significance: p < 0.05 for region clustering
- [ ] Geometric correlation: R² > 0.7 for entropy vs speedup

---

## Long-Term Vision (Post-v3.0)

### Hardware Acceleration
- **GPU/TPU deployment:** TritNet inference on GPU via CUDA/OpenCL
- **FPGA implementation:** Use archived backend system for HDL generation
- **ASIC design:** Ternary compute units with geometric optimizers
- **ARM NEON/SVE:** Mobile/edge deployment with hybrid architecture

### Learned Generalization
- **Approximate arithmetic:** Trade precision for speed in neural networks
- **Novel operations:** Discover new ternary operations via TritNet
- **Auto-tuning:** ML-based path selection learned from profiling

### Ecosystem Integration
- **PyTorch integration:** Ternary tensor backend
- **TensorFlow integration:** Custom op kernels
- **ONNX export:** Ternary quantization format
- **Edge deployment:** Mobile/IoT runtimes

---

## Conclusion

**Current State (v1.3.0):** 45.3 Gops/s effective, 39.1 Gops/s peak
- Successfully unified cold path (direct kernels) + geometric path (canonical indexing)
- 90% of 50 Gops/s target achieved

**Next Phase (v3.0):** Hybrid adaptive architecture
- Adaptive path selection based on input characteristics
- Comprehensive invariant measurement for understanding WHY performance varies
- Advanced optimizations: PGO, Dense243, 3-op fusion, cache blocking
- Target: 70+ Gops/s sustained, 100+ Gops/s peak in geometric regions

**Engineering Philosophy:**
- Preserve simplicity of v1.3.0 architecture
- Add complexity only when measured gains >10%
- Validate every optimization with comprehensive benchmarks
- Document architectural decisions with clear rationale

**Timeline:** 6 weeks to v3.0 release
**Risk Level:** Low (building on proven v1.3.0 baseline)
**Expected ROI:** 1.5-2× performance improvement with maintained code quality

---

**Status:** Planning complete, ready for execution approval
**Next Action:** Review this roadmap, approve execution order, begin Week 1 (Invariant Measurement)
**Decision Point:** After Week 2, validate invariant findings before proceeding to hybrid selector

---

**Document Version:** 1.0
**Last Updated:** 2025-11-25
**Next Review:** After each milestone (Weeks 2, 4, 5, 6)
