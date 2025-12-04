# Critical Path Prioritization - 2025-12-04

**Purpose:** Separate optimal paths (build on proven foundation) from exploratory paths (research-heavy) to focus development on immediate value delivery.

---

## Current State Summary

| Component | Status | Performance |
|-----------|--------|-------------|
| **Core SIMD Engine** | ✅ Production | 19.57 Gops/s peak |
| **Fusion Operations** | ✅ Validated | 1.53-11.26× speedup |
| **Bridge Layer** | ✅ Validated | 44.58× vs NumPy |
| **TritNet tnot (PoC)** | ✅ Complete | 100% accuracy (487 epochs) |
| **Dense243** | ❌ Headers only | Not compiled |
| **TritNet GEMM** | ❌ Headers only | Not compiled |
| **Binary TritNet (tadd/tmul)** | ❌ Not trained | Infrastructure ready |

---

## Path Classification

### OPTIMAL PATHS (Build Phase)
**Characteristics:** Low-medium risk, builds on existing infrastructure, immediate value

### EXPLORATORY PATHS (Research Phase)
**Characteristics:** High risk, theoretical foundation, speculative value

---

## OPTIMAL PATHS (Recommended)

### Priority 1: Complete TritNet Stack (Immediate - 2-3 days)

**What:** Finish the ~40% remaining infrastructure that's already designed

**Tasks:**
1. **Train remaining binary operations** (~1 day)
   - tadd, tmul, tmin, tmax (tnot already at 100% accuracy)
   - Infrastructure exists, just need training runs
   - Expected: 30 minutes training per operation

2. **Build Dense243 module** (~4 hours)
   - Headers complete: `src/engine/lib/dense243/`
   - Build script exists: `build/build_dense243.py`
   - Just needs compilation

3. **Build TritNet GEMM module** (~4 hours)
   - C++ implementation exists: `src/engine/bindings_tritnet_gemm.cpp`
   - Just needs compilation

4. **Validate full stack** (~2 hours)
   - End-to-end test: Python → TritNet → Dense243 → result
   - Benchmark vs LUT baseline

**Rationale:**
- 60% of work already done (headers, bindings, infrastructure)
- Low risk - tnot PoC proves concept works
- Unlocks Path 2 (AI applications)

**Dependencies:** None - can start immediately

**Risk:** LOW
- tnot achieved 100% accuracy
- Infrastructure validated
- No new architecture needed

---

### Priority 2: SIMD Incremental Optimizations (1-2 weeks)

**What:** Phase 3 production refinements from optimization-roadmap.md

**Tasks (from docs/architecture/optimization-roadmap.md Phase 3):**

1. **Adaptive threading** (30 min)
   - Currently: Static `OMP_THRESHOLD = 100000`
   - Target: `OMP_THRESHOLD = 32768 * hardware_concurrency()`
   - Expected: 5-10% gain on multi-core systems

2. **Prefetch tuning** (2 days)
   - Currently: Static `_mm_prefetch(... + 256)`
   - Target: Tunable `PREFETCH_DIST = 512` per CPU family
   - Expected: 2-5% gain

3. **C API for FFI** (2-3 days)
   - File: `ternary_c_api.h` (proposed)
   - Enables Rust, Zig, C# integration
   - No performance gain, expands ecosystem

4. **C++ native benchmarks** (1 hour)
   - Currently: Only Python benchmarks (includes NumPy overhead)
   - Target: Direct C++ kernel benchmarks
   - Honest GOPS measurements for commercial claims

**Rationale:**
- Builds on proven AVX2 foundation
- Low risk - extending existing architecture
- Clear incremental gains

**Dependencies:** None

**Risk:** LOW
- All optimizations are refinements of existing code
- No architectural changes

---

### Priority 3: Real Model Validation (2-4 weeks)

**What:** Prove commercial viability with actual AI models

**Tasks:**

1. **TinyLlama quantization** (1 week)
   - Quantize TinyLlama-1.1B to ternary
   - Measure memory: 4× reduction vs INT8 (claim validation)
   - Measure accuracy: target <5% loss
   - Measure inference latency vs FP16/INT8

2. **Computer vision demo** (1 week)
   - Edge detection with ternary convolutions
   - Visual proof of concept
   - Benchmark vs OpenCV

3. **Power consumption** (1 week)
   - Requires hardware instrumentation (Intel RAPL / nvidia-smi)
   - Target: 2-4× better energy efficiency
   - Validates edge/mobile deployment claims

**Rationale:**
- Most convincing proof of commercial value
- Clear success/failure metrics
- Opens partnership opportunities

**Dependencies:**
- Priority 1 (TritNet stack) must complete first
- Requires GPU/TPU for competitive benchmarking

**Risk:** MEDIUM
- Accuracy requirements may fail on large models
- Depends on external ecosystems (HuggingFace, PyTorch)

---

## EXPLORATORY PATHS (Reserve for Later)

### Path A: Hexatic Self-Organizing Architecture

**From:** roadmap/automaton_integration.md, roadmap/prototype_roadmap.md

**What:** Category-theoretic meta-backend with self-modifying dispatch

**Status:** 100% conceptual - no implementation exists

**Risk:** VERY HIGH
- Unproven theoretical foundation
- 6+ months research before any value
- May not outperform hand-tuned kernels

**Recommendation:** **DEFER**
- Only pursue after core product stabilizes
- Treat as experimental research branch
- Do not block production roadmap

**Value Proposition:** Maximum disruptive potential IF successful

---

### Path B: Multi-Platform Production Hardening

**From:** docs/planning/ROADMAP.md (v2.5)

**What:** Linux, macOS, ARM validation and CI

**Status:**
- Only Windows x64 validated
- Linux/macOS builds exist but untested
- CI disabled for OpenMP

**Risk:** LOW-MEDIUM
- Engineering overhead, not technical risk
- Each platform: 2-4 weeks validation

**Recommendation:** **DEFER**
- Windows x64 already covers most dev workloads
- Delay until Windows product is feature-complete
- Linux validation can happen in parallel with low effort

**Value Proposition:** Expands market, but not critical for initial commercial viability

---

### Path C: AVX-512 / ARM NEON Backends

**From:** docs/architecture/optimization-roadmap.md Phase 2

**What:** 64-trit AVX-512 backend, 16-trit ARM NEON backend

**Status:**
- AVX-512: Designed but not implemented
- ARM NEON: Conceptual only

**Risk:** MEDIUM
- Engineering overhead
- Multi-platform validation burden
- AVX-512 limited CPU availability

**Recommendation:** **DEFER**
- Current AVX2 performance (19.57 Gops/s) already excellent
- AVX-512 only 2× theoretical gain
- Focus on algorithmic improvements first

**Value Proposition:** Platform portability, but not critical path

---

## Decision Matrix: Optimal vs Exploratory

| Path | Risk | Time | Value | Type | Priority |
|------|------|------|-------|------|----------|
| **Complete TritNet Stack** | LOW | 2-3 days | High | Build | **1** |
| **SIMD Incremental Opts** | LOW | 1-2 weeks | Medium | Build | **2** |
| **Real Model Validation** | MEDIUM | 2-4 weeks | Very High | Build | **3** |
| Hexatic Architecture | VERY HIGH | 6+ months | Speculative | Research | DEFER |
| Multi-Platform Hardening | LOW-MEDIUM | 2-4 weeks/platform | Medium | Build | DEFER |
| AVX-512 / ARM Backends | MEDIUM | 4-6 weeks | Low-Medium | Build | DEFER |

---

## Recommended Execution Plan

### Week 1: Complete TritNet (Priority 1)

**Day 1-2:** Train binary operations
- Run `models/tritnet/run_tritnet.py` for tadd, tmul, tmin, tmax
- Validate 100% accuracy (or ≥99%)
- Document learned weights

**Day 3:** Build Dense243 + TritNet GEMM
- `python build/build_dense243.py`
- `python build/build_tritnet_gemm.py` (needs creation)
- Run integration tests

**Day 4:** End-to-end validation
- Python API test: dense pack → TritNet inference → unpack
- Benchmark vs LUT baseline
- Document performance

**Day 5:** C++ native benchmarks (Priority 2 quick win)
- Compile `benchmarks/cpp/bench_kernels.cpp` (from roadmap)
- Run honest GOPS measurements
- Update README with validated claims

**Deliverable:** Fully functional TritNet stack + validated performance claims

---

### Week 2-3: SIMD Refinements (Priority 2)

**Week 2:**
- Adaptive threading (30 min)
- Prefetch tuning (2 days)
- C API for FFI (2-3 days)

**Week 3:**
- Integration testing
- Performance regression validation
- Documentation updates

**Deliverable:** Production-ready AVX2 engine with ecosystem expansion (C API)

---

### Week 4-7: Real Model Validation (Priority 3)

**Week 4-5:** TinyLlama quantization
- Quantize model to ternary
- Measure memory, accuracy, latency
- Document results

**Week 6:** Computer vision demo
- Edge detection demo
- Benchmark vs OpenCV
- Visual proof of concept

**Week 7:** Power consumption analysis
- Hardware instrumentation setup
- Energy efficiency measurements
- Validate edge/mobile deployment claims

**Deliverable:** Proof of commercial viability with real AI workloads

---

## Success Metrics

### Immediate (Week 1)
- ✅ All 5 TritNet operations trained (100% accuracy)
- ✅ Dense243 module compiles and passes tests
- ✅ TritNet GEMM module compiles and passes tests
- ✅ C++ benchmarks show honest GOPS numbers

### Medium-term (Week 3)
- ✅ Adaptive threading shows 5-10% gain on multi-core
- ✅ C API enables Rust/Zig/C# integration
- ✅ No performance regressions from refactoring

### Long-term (Week 7)
- ✅ TinyLlama demonstrates 4× memory reduction
- ✅ Accuracy loss <5% on real model
- ✅ Energy efficiency 2-4× better than FP16/INT8
- ✅ Production deployment guide complete

---

## Why This Prioritization?

### Focus on BUILDING, not RESEARCHING

**Current State:** ~60% of designed features unbuilt
- TritNet infrastructure exists but not compiled
- Dense243 headers complete but not used
- Optimization roadmap has clear tasks but not executed

**Problem:** Adding exploratory paths (Hexatic, AVX-512, Multi-platform) before finishing existing work creates:
1. **Context switching overhead** - Research vs build require different mindsets
2. **Incomplete foundation** - Can't evaluate Hexatic value without TritNet baseline
3. **Diluted focus** - Spreading effort across 6 paths vs concentrating on 3

### Build vs Research Trade-offs

| Approach | Pros | Cons |
|----------|------|------|
| **Optimal Path (Build)** | Immediate value, low risk, completes existing work | Less "exciting", incremental |
| **Exploratory Path (Research)** | Maximum disruption potential, IP value | High risk, long time to value, may fail |

**Recommendation:** Build first, research later
- Complete TritNet (60% done → 100%)
- Validate with real models (prove commercial value)
- THEN evaluate Hexatic (from position of strength)

### Resource Constraints Matter

**Single Developer Context:**
- Building 3 optimal paths: 7-8 weeks
- Building 3 optimal + 3 exploratory: 6+ months
- Risk of never finishing anything

**Team Context:**
- Can parallelize optimal + exploratory
- But still risky to research before foundation stable

---

## Questions Before Proceeding

1. **Target Market Clarity:**
   - Edge AI (needs ARM, power efficiency)?
   - Cloud AI (needs GPU/TPU)?
   - Developer Tools (current Windows x64)?

2. **Resource Reality:**
   - Solo developer → Focus on optimal paths only
   - Small team → Optimal paths + 1 exploratory stream
   - Well-funded → Parallel streams possible

3. **Risk Tolerance:**
   - Conservative → Optimal paths only
   - Moderate → Optimal + Multi-platform
   - Aggressive → Optimal + Hexatic research

4. **Success Definition:**
   - Commercial viability (TinyLlama working) → Priority 3 critical
   - Raw performance (40-70 Gops/s) → Priority 2 focus
   - Academic novelty (Hexatic) → Exploratory Path A

---

## Recommendation Summary

**OPTIMAL PATH:**
1. Week 1: Complete TritNet stack (2-3 days) ← **START HERE**
2. Weeks 2-3: SIMD incremental optimizations (1-2 weeks)
3. Weeks 4-7: Real model validation (2-4 weeks)

**EXPLORATORY PATHS (DEFER):**
- Hexatic: Only after TritNet validates commercial value
- Multi-Platform: Only after Windows x64 feature-complete
- AVX-512/ARM: Only if TritNet shows need for more raw performance

**Decision Point (Week 7):**
After real model validation:
- If TinyLlama successful → GPU/TPU acceleration (Path 1 advanced)
- If accuracy insufficient → Pivot to algorithmic improvements or Hexatic research
- If commercial interest strong → Multi-platform hardening

---

**Document Status:** Strategic prioritization based on roadmap analysis
**Created:** 2025-12-04
**Next Review:** After Week 1 completion (TritNet stack complete)
**Author:** Strategic analysis of existing roadmap documents
