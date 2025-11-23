Run performance benchmarks for the Ternary Engine.

**Standard performance benchmarks** (core operations):
```bash
python benchmarks/bench_phase0.py
```

This measures:
- Throughput (Mops/s) for tadd, tmul, tmin, tmax, tnot
- Scaling behavior across array sizes (32 to 10M elements)
- Speedup vs pure Python baseline
- Latency per element

**Competitive benchmarking suite** (6 phases):
```bash
# Run all phases
python benchmarks/bench_competitive.py --all

# Run specific phase
python benchmarks/bench_competitive.py --phase 1  # vs NumPy INT8
python benchmarks/bench_competitive.py --phase 2  # Memory efficiency
python benchmarks/bench_competitive.py --phase 3  # Throughput at equivalent bit-width
python benchmarks/bench_competitive.py --phase 4  # Neural workload patterns
python benchmarks/bench_competitive.py --phase 5  # Model quantization
python benchmarks/bench_competitive.py --phase 6  # Power consumption
```

**Generate visualization report**:
```bash
python benchmarks/utils/visualization.py results/competitive_results_*.json
```

Results are saved to:
- reports/YYYY-MM-DD/ directory
- JSON format for programmatic analysis
- Markdown reports for human review

Expected performance (Windows x64, 12 cores):
- Peak throughput: 35,042 Mops/s
- Average speedup: 8,234× vs Python
