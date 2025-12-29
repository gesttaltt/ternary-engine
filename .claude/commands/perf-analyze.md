# Performance Analysis Command

Analyze the performance of recent benchmark runs and compare against documented baselines.

## Task

1. **Find Recent Results**
   - Search for benchmark results in `reports/performance/`
   - Look for JSON files with benchmark data
   - Identify the most recent benchmark run

2. **Compare Against Baselines**
   Reference these documented baselines from CLAUDE.md:
   - Peak throughput (fusion): 45.3 Gops/s
   - Peak throughput (element-wise): 39.1 Gops/s
   - Average speedup vs Python: 8,234x
   - Speedup vs NumPy INT8: 2.75-11.60x

3. **Identify Regressions**
   - Flag any metrics that are >5% slower than baseline
   - Note the severity (minor: 5-10%, moderate: 10-20%, severe: >20%)

4. **Generate Report**
   Create a markdown table with:
   | Metric | Baseline | Current | Delta | Status |

   Include:
   - Recommendations for optimization
   - Areas that improved
   - Next investigation steps

## Output Format

```markdown
## Performance Analysis Report
**Date:** [current date]
**Platform:** Windows x64

### Summary
[1-2 sentence summary]

### Metrics Comparison
| Metric | Baseline | Current | Delta | Status |
|--------|----------|---------|-------|--------|
| ...    | ...      | ...     | ...   | OK/WARN/FAIL |

### Regressions
[List any regressions with severity]

### Recommendations
[Actionable next steps]
```
