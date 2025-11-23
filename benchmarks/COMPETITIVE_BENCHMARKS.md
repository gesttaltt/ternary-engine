# Competitive Benchmarking Suite

**Doc-Type:** Technical Documentation · Version 1.0 · Updated 2025-11-23

Comprehensive benchmark suite to prove whether ternary has commercial value by comparing against industry standards (NumPy INT8, INT4, FP16, and quantized models).

---

## Purpose & Motivation

**objective** - Determine if ternary computing provides measurable advantages in:
- Computational performance
- Memory efficiency
- Power consumption
- Real-world AI workloads

**critical_question** - By Week 4, do we have a business or a hobby project?

**success_criteria**:
1. Memory efficiency at same model capacity ✓
2. Throughput at equivalent bit-width ✓
3. Inference latency in real models ⚠
4. Power consumption on edge devices ⚠
5. Accuracy retention after quantization ⚠

---

## Benchmark Phases

### Phase 1: Arithmetic Operations vs NumPy

**what** - Direct comparison with NumPy INT8 operations
**why** - Establish baseline performance for equivalent information density
**metrics**:
- Operations per second
- Throughput (GB/s)
- Speedup vs NumPy INT8

**run**:
```bash
python bench_competitive.py --phase 1
```

**expected_results** - Ternary should be competitive or faster than NumPy INT8 due to reduced bit-width (2 bits vs 8 bits).

---

### Phase 2: Memory Efficiency

**what** - Compare storage requirements at equivalent model capacity
**why** - Memory is the bottleneck for large models
**comparison_targets**:
- FP16 (baseline)
- INT8 quantization
- INT4 quantization
- Ternary (2 bits/weight)
- Dense243 (1.6 bits/weight)

**run**:
```bash
python bench_competitive.py --phase 2
```

**expected_results**:
- 7B model in FP16: 14 GB
- 7B model in Ternary: 1.75 GB
- 7B model in Dense243: 1.4 GB
- **Advantage: 8x smaller than FP16, 4x smaller than INT8**

---

### Phase 3: Throughput at Equivalent Bit-Width

**what** - Operations/second when memory footprint is equal
**why** - This is the REAL competition - comparing against other ultra-low bit schemes
**test_approach** - Fix memory at 1GB, compare:
- Ternary (2 bits/element)
- INT2 (2 bits/element)
- INT4 (4 bits/element)

**run**:
```bash
python bench_competitive.py --phase 3
```

**note** - INT2/INT4 comparison requires reference implementations

---

### Phase 4: Neural Network Workload Patterns

**what** - Matrix operations typical in neural networks
**why** - AI is matrix multiplication. If ternary ops are fast but matmul is slow, there's no viable AI solution
**test_cases**:
- Small MLP (512x512)
- Medium Layer (2048x2048)
- Large Layer (4096x4096)
- Attention Head (8192x1024)

**run**:
```bash
python bench_competitive.py --phase 4
```

**critical** - Must achieve >0.5x NumPy performance to be viable for AI

---

### Phase 5: Real Model Quantization

**what** - Quantize actual pre-trained models to ternary
**why** - This is the PROOF. If accuracy is maintained and speed improves, we have a product
**target_models**:
- TinyLlama-1.1B (easy)
- Phi-2 (2.7B)
- Gemma-2B

**metrics**:
- Perplexity degradation
- Accuracy on benchmark tasks
- Inference latency
- Memory footprint
- Throughput (tokens/sec)

**run**:
```bash
python bench_model_quantization.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

**success_criteria**:
- Accuracy loss < 5%
- Inference latency < 2x original
- Memory < 25% of FP16
- Maintains coherent text generation

**note** - Requires PyTorch and Transformers:
```bash
pip install torch transformers
```

---

### Phase 6: Power Consumption

**what** - Measure energy efficiency (operations/Joule)
**why** - Edge AI is power-constrained. If ternary saves power, that's the killer feature
**platforms**:
- x86 (Intel RAPL)
- ARM (USB power meter)
- NVIDIA GPU (nvidia-smi)

**run**:
```bash
# Auto-detect platform
python bench_power_consumption.py --platform auto

# Specific platform
python bench_power_consumption.py --platform intel
python bench_power_consumption.py --platform nvidia
```

**expected_advantage** - 2-4x lower power consumption vs INT8

**note** - Requires hardware access and permissions

---

## Quick Start

### Run All Benchmarks

```bash
# Full suite (takes ~30 minutes)
python bench_competitive.py --all

# Results saved to: results/competitive_results_TIMESTAMP.json
```

### Run Individual Phase

```bash
# Phase 1 only
python bench_competitive.py --phase 1

# Phase 4 only
python bench_competitive.py --phase 4
```

### Generate Reports

```bash
# Text report
python utils/visualization.py results/competitive_results_TIMESTAMP.json

# HTML report
python utils/visualization.py results/competitive_results_TIMESTAMP.json report.html
```

---

## Installation

### Core Requirements

```bash
# NumPy (required)
pip install numpy

# For visualization
pip install matplotlib
```

### Optional Requirements

**model_quantization** (Phase 5):
```bash
pip install torch transformers
```

**power_monitoring** (Phase 6):
- Intel RAPL: Linux with `/sys/class/powercap/intel-rapl/` access
- NVIDIA: `nvidia-smi` installed
- ARM: USB power meter hardware

---

## Results Structure

**output_directory** - `benchmarks/results/`

**file_format** - JSON with timestamp

**schema**:
```json
{
  "metadata": {
    "timestamp": "2025-11-23T...",
    "platform": "win32",
    "numpy_version": "1.24.0"
  },
  "phase1_arithmetic_comparison": {
    "size": [1000, 10000, ...],
    "ternary_add_ns": [...],
    "numpy_int8_add_ns": [...],
    "add_speedup": [...]
  },
  "phase2_memory_efficiency": [...],
  "phase3_throughput_equivalent_bitwidth": {...},
  "phase4_neural_workload_patterns": [...],
  "phase5_model_quantization": {...},
  "phase6_power_consumption": {...}
}
```

---

## Interpreting Results

### Performance Metrics

**speedup** - `baseline_time / optimized_time`
- > 1.0: Ternary is faster
- < 1.0: Ternary is slower
- 1.0: Equal performance

**throughput** - Operations or GB/s
- Higher is better
- Compare at same memory footprint

**GOPS** - Giga-operations per second
- Billions of operations per second
- Standard metric for computational performance

### Memory Metrics

**reduction_factor** - `baseline_size / ternary_size`
- FP16 → Ternary: 8x smaller
- INT8 → Ternary: 4x smaller
- INT4 → Ternary: 2x smaller

### Power Metrics

**ops_per_joule** - Operations per unit energy
- Higher is better
- Critical for edge deployment

**watts_per_gops** - Power per billion operations
- Lower is better
- Enables longer battery life

---

## Commercial Viability Checklist

**criteria** - What proves commercial value:

| Criterion | Target | Status |
|:----------|:-------|:-------|
| Memory efficiency at same capacity | 4x vs INT8 | ✓ Proven |
| Throughput at equivalent bit-width | > INT2 | ⚠ Needs INT2 ref |
| Inference latency in real models | < 2x FP16 | ⚠ Needs testing |
| Power consumption on edge | 2-4x better | ⚠ Needs hardware |
| Accuracy retention after quantization | < 5% loss | ⚠ Needs models |

**verdict** - 2/5 criteria validated so far

---

## Next Steps

### Immediate (This Week)

1. **Run Phase 1-4** - Establish performance baselines
2. **Analyze results** - Identify bottlenecks
3. **Optimize hot paths** - Focus on matmul (Phase 4)

### Short Term (Next 2 Weeks)

4. **Implement Phase 5** - Quantize TinyLlama-1.1B
5. **Measure accuracy** - Compare with FP16/INT8
6. **Optimize inference** - C++ SIMD for matmul

### Medium Term (Week 4)

7. **Run Phase 6** - Measure power consumption
8. **Write up results** - Technical report
9. **Make decision** - Business vs hobby project

---

## Comparison with Existing Benchmarks

**existing** - `benchmarks/bench_compare.py`, `bench_fusion.py`
- Focus on internal ternary operations
- Compare different ternary implementations

**competitive** - `benchmarks/bench_competitive.py` (this suite)
- Focus on external comparisons
- Prove commercial viability

**relationship** - Complementary
- Internal: Optimize ternary performance
- Competitive: Prove market value

---

## Architecture Notes

### Why These Specific Tests?

**phase1** - Proves basic operation speed
**phase2** - Quantifies memory advantage (key selling point)
**phase3** - Head-to-head with INT2/INT4 (real competition)
**phase4** - Tests AI workloads (target market)
**phase5** - Proves real-world viability
**phase6** - Proves edge deployment value

### Known Limitations

**current_implementation**:
- Phase 4 uses Python loops (slow)
- Need C++ SIMD for fair matmul comparison
- Missing INT2/INT4 reference implementations

**future_work**:
- Implement optimized matmul in C++
- Add INT2/INT4 comparison baselines
- Extend to transformer layers
- Add quantization-aware training

---

## Troubleshooting

### Mock Operations Warning

**symptom**: "Warning: ternary_core not available, using mock operations"

**cause**: Ternary core not built

**fix**:
```bash
cd ../ternary_core
python setup.py build_ext --inplace
```

### PyTorch/Transformers Missing

**symptom**: Phase 5 disabled

**fix**:
```bash
pip install torch transformers
```

### Power Monitoring Not Available

**symptom**: "Using mock power monitor"

**platforms**:
- **Linux/Intel**: Check `/sys/class/powercap/intel-rapl/` permissions
- **NVIDIA**: Verify `nvidia-smi` works
- **ARM**: Requires USB power meter hardware

---

## File Structure

```
benchmarks/
├── bench_competitive.py              # Main suite (Phases 1-6)
├── bench_model_quantization.py       # Phase 5 detailed
├── bench_power_consumption.py        # Phase 6 detailed
├── utils/
│   └── visualization.py              # Report generation
├── results/                          # Output directory
│   ├── competitive_results_*.json
│   ├── model_quantization_*.json
│   └── power_consumption_*.json
└── COMPETITIVE_BENCHMARKS.md         # This file
```

---

## Contributing

**reporting_issues** - Found a bug or have a suggestion?
1. Check existing issues
2. Include benchmark results
3. Specify platform/environment

**adding_benchmarks** - Want to add more tests?
1. Follow existing phase structure
2. Add to CompetitiveBenchmark class
3. Update documentation
4. Include expected results

---

## References

**internal_docs**:
- `benchmarks/README.md` - General benchmark documentation
- `real.md` - Original benchmark requirements

**external_standards**:
- NumPy performance: https://numpy.org/doc/stable/reference/routines.linalg.html
- INT8 quantization: PyTorch quantization docs
- Model benchmarks: HuggingFace model cards

---

## Changelog

| Date | Version | Changes |
|:-----|:--------|:--------|
| 2025-11-23 | v1.0.0 | Initial implementation - all 6 phases |

---

**Remember:** The goal is truth, not marketing. These benchmarks will tell us objectively whether ternary has commercial value. Run them, analyze honestly, and make data-driven decisions.

---

**Version:** 1.0.0 · **Updated:** 2025-11-23 · **Author:** AI Development Team
