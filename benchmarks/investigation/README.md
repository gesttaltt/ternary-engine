# Investigation Scripts

**Status:** One-time analysis - Not regular benchmarks

These scripts were created for specific investigations during Phase 1 development. They may be useful for future debugging but are not part of the regular benchmark suite.

## Files

| File | Investigation Purpose |
|------|----------------------|
| `investigate_repetitive_performance.py` | Root cause analysis for 40x slowdown with repetitive patterns |
| `analyze_phase1_datasets.py` | Analysis of Phase 1 synthetic dataset characteristics |
| `test_0121_pattern.py` | Deep dive into pathological [0,1,2,1] pattern behavior |

## Key Findings

### Repetitive Pattern Slowdown
- Pattern length correlates with cache line conflicts
- All operations affected equally (memory issue, not algorithmic)
- Short repetitive patterns cause worst performance

### [0,1,2,1] Pattern
- Perfect negative autocorrelation at lag 2
- Creates specific memory access patterns causing cache conflicts
- 37x slowdown compared to random data

## When to Use

Run these scripts when:
- Debugging unexpected performance regressions
- Investigating cache-related issues
- Analyzing input data characteristics

## Dependencies

Requires:
- `ternary_simd_engine` module
- `benchmarks/utils/geometric_metrics.py`
- `benchmarks/utils/hardware_metrics.py`
