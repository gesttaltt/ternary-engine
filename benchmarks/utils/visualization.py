"""
Visualization utilities for benchmark results

Generates charts and reports from benchmark JSON output:
- Performance comparison charts
- Memory efficiency graphs
- Power consumption analysis
- Comprehensive HTML reports

Usage:
    from utils.visualization import BenchmarkVisualizer

    viz = BenchmarkVisualizer()
    viz.load_results("results/competitive_results_20250123_120000.json")
    viz.generate_report("report.html")
"""

import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime


class BenchmarkVisualizer:
    """
    Generate visualizations from benchmark results

    Supports both text-based terminal output and HTML reports
    """

    def __init__(self):
        self.results = None
        self.report_lines = []

    def load_results(self, filename: str):
        """Load benchmark results from JSON file"""
        with open(filename, 'r') as f:
            self.results = json.load(f)
        print(f"Loaded results from {filename}")

    def generate_text_report(self) -> str:
        """Generate text-based report suitable for terminal"""
        if not self.results:
            return "No results loaded"

        lines = []
        lines.append("=" * 80)
        lines.append("TERNARY ENGINE COMPETITIVE BENCHMARK REPORT")
        lines.append("=" * 80)

        # Metadata
        if 'metadata' in self.results:
            meta = self.results['metadata']
            lines.append(f"\nTimestamp: {meta.get('timestamp', 'N/A')}")
            lines.append(f"Platform:  {meta.get('platform', 'N/A')}")
            lines.append(f"NumPy:     {meta.get('numpy_version', 'N/A')}")

        # Phase 1: Arithmetic comparison
        if 'phase1_arithmetic_comparison' in self.results:
            lines.append("\n" + "=" * 80)
            lines.append("PHASE 1: ARITHMETIC OPERATIONS")
            lines.append("=" * 80)
            lines.extend(self._format_phase1(self.results['phase1_arithmetic_comparison']))

        # Phase 2: Memory efficiency
        if 'phase2_memory_efficiency' in self.results:
            lines.append("\n" + "=" * 80)
            lines.append("PHASE 2: MEMORY EFFICIENCY")
            lines.append("=" * 80)
            lines.extend(self._format_phase2(self.results['phase2_memory_efficiency']))

        # Phase 3: Throughput
        if 'phase3_throughput_equivalent_bitwidth' in self.results:
            lines.append("\n" + "=" * 80)
            lines.append("PHASE 3: THROUGHPUT AT EQUIVALENT BIT-WIDTH")
            lines.append("=" * 80)
            lines.extend(self._format_phase3(self.results['phase3_throughput_equivalent_bitwidth']))

        # Phase 4: Neural network patterns
        if 'phase4_neural_workload_patterns' in self.results:
            lines.append("\n" + "=" * 80)
            lines.append("PHASE 4: NEURAL NETWORK WORKLOADS")
            lines.append("=" * 80)
            lines.extend(self._format_phase4(self.results['phase4_neural_workload_patterns']))

        return "\n".join(lines)

    def _format_phase1(self, data: Dict) -> List[str]:
        """Format Phase 1 results"""
        lines = []

        if not data or 'size' not in data:
            lines.append("No data available")
            return lines

        lines.append("\nPerformance vs NumPy INT8:")
        lines.append(f"{'Size':>12} | {'Add Speedup':>12} | {'Mul Speedup':>12} | {'Throughput':>15}")
        lines.append("-" * 60)

        for i, size in enumerate(data['size']):
            add_speedup = data['add_speedup'][i]
            mul_speedup = data['mul_speedup'][i]
            throughput = data['ternary_throughput_gbps'][i]

            lines.append(
                f"{size:12,} | {add_speedup:>12.2f}x | {mul_speedup:>12.2f}x | "
                f"{throughput:>12.2f} GB/s"
            )

        # Summary
        avg_add = sum(data['add_speedup']) / len(data['add_speedup'])
        avg_mul = sum(data['mul_speedup']) / len(data['mul_speedup'])

        lines.append("-" * 60)
        lines.append(f"Average addition speedup:       {avg_add:.2f}x")
        lines.append(f"Average multiplication speedup: {avg_mul:.2f}x")

        return lines

    def _format_phase2(self, data: List[Dict]) -> List[str]:
        """Format Phase 2 results"""
        lines = []

        if not data:
            lines.append("No data available")
            return lines

        lines.append("\nMemory Footprint Comparison:")
        lines.append(f"{'Model':>20} | {'FP16':>10} | {'INT8':>10} | {'Ternary':>10} | {'Dense243':>10}")
        lines.append("-" * 70)

        for model in data:
            lines.append(
                f"{model['name']:>20} | "
                f"{model['fp16_gb']:>8.2f}GB | "
                f"{model['int8_gb']:>8.2f}GB | "
                f"{model['ternary_gb']:>8.2f}GB | "
                f"{model['dense243_gb']:>8.2f}GB"
            )

        lines.append("-" * 70)
        lines.append("Ternary advantage: 4.0x smaller than INT8, 2.0x smaller than INT4")

        return lines

    def _format_phase3(self, data: Dict) -> List[str]:
        """Format Phase 3 results"""
        lines = []

        if 'ternary_gops' in data:
            lines.append(f"\nTernary throughput: {data['ternary_gops']:.2f} GOPS")
            lines.append(f"Elements tested:    {data['ternary_elements']:,}")
            lines.append(f"Memory footprint:   {data['target_bytes'] / 1e9:.1f} GB")
        else:
            lines.append("No data available")

        return lines

    def _format_phase4(self, data: List[Dict]) -> List[str]:
        """Format Phase 4 results"""
        lines = []

        if not data:
            lines.append("No data available")
            return lines

        lines.append("\nMatrix Multiplication Performance:")
        lines.append(f"{'Layer':>20} | {'Shape':>15} | {'Speedup':>10} | {'Ternary GOPS':>15}")
        lines.append("-" * 70)

        for result in data:
            shape_str = f"{result['shape'][0]}x{result['shape'][1]}"
            lines.append(
                f"{result['name']:>20} | "
                f"{shape_str:>15} | "
                f"{result['speedup']:>10.2f}x | "
                f"{result['ternary_gops']:>15.2f}"
            )

        avg_speedup = sum(r['speedup'] for r in data) / len(data)
        lines.append("-" * 70)
        lines.append(f"Average matmul speedup: {avg_speedup:.2f}x")

        return lines

    def generate_html_report(self, output_file: str = "report.html"):
        """Generate comprehensive HTML report"""
        if not self.results:
            print("No results loaded")
            return

        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append("<title>Ternary Engine Benchmark Report</title>")
        html.append("<style>")
        html.append(self._get_css())
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")

        html.append("<h1>Ternary Engine Competitive Benchmark Report</h1>")

        # Metadata
        if 'metadata' in self.results:
            meta = self.results['metadata']
            html.append("<div class='metadata'>")
            html.append(f"<p><strong>Timestamp:</strong> {meta.get('timestamp', 'N/A')}</p>")
            html.append(f"<p><strong>Platform:</strong> {meta.get('platform', 'N/A')}</p>")
            html.append(f"<p><strong>NumPy Version:</strong> {meta.get('numpy_version', 'N/A')}</p>")
            html.append("</div>")

        # Phase summaries
        html.append(self._generate_phase1_html())
        html.append(self._generate_phase2_html())
        html.append(self._generate_phase4_html())

        html.append("</body>")
        html.append("</html>")

        with open(output_file, 'w') as f:
            f.write("\n".join(html))

        print(f"HTML report generated: {output_file}")

    def _get_css(self) -> str:
        """CSS styles for HTML report"""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #007acc;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
            border-left: 5px solid #007acc;
            padding-left: 10px;
        }
        .metadata {
            background-color: #fff;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: #fff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th, td {
            padding: 12px;
            text-align: right;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #007acc;
            color: white;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f1f1f1;
        }
        .summary {
            background-color: #e7f3ff;
            padding: 15px;
            border-left: 5px solid #007acc;
            margin: 20px 0;
        }
        .highlight {
            color: #007acc;
            font-weight: bold;
        }
        """

    def _generate_phase1_html(self) -> str:
        """Generate HTML for Phase 1"""
        if 'phase1_arithmetic_comparison' not in self.results:
            return ""

        data = self.results['phase1_arithmetic_comparison']
        if not data or 'size' not in data:
            return ""

        html = []
        html.append("<h2>Phase 1: Arithmetic Operations</h2>")
        html.append("<table>")
        html.append("<tr>")
        html.append("<th>Size</th>")
        html.append("<th>Add Speedup</th>")
        html.append("<th>Mul Speedup</th>")
        html.append("<th>Throughput (GB/s)</th>")
        html.append("</tr>")

        for i, size in enumerate(data['size']):
            html.append("<tr>")
            html.append(f"<td>{size:,}</td>")
            html.append(f"<td>{data['add_speedup'][i]:.2f}x</td>")
            html.append(f"<td>{data['mul_speedup'][i]:.2f}x</td>")
            html.append(f"<td>{data['ternary_throughput_gbps'][i]:.2f}</td>")
            html.append("</tr>")

        html.append("</table>")

        avg_add = sum(data['add_speedup']) / len(data['add_speedup'])
        html.append("<div class='summary'>")
        html.append(f"<p><strong>Average speedup:</strong> <span class='highlight'>{avg_add:.2f}x</span></p>")
        html.append("</div>")

        return "\n".join(html)

    def _generate_phase2_html(self) -> str:
        """Generate HTML for Phase 2"""
        if 'phase2_memory_efficiency' not in self.results:
            return ""

        data = self.results['phase2_memory_efficiency']
        if not data:
            return ""

        html = []
        html.append("<h2>Phase 2: Memory Efficiency</h2>")
        html.append("<table>")
        html.append("<tr>")
        html.append("<th>Model</th>")
        html.append("<th>FP16 (GB)</th>")
        html.append("<th>INT8 (GB)</th>")
        html.append("<th>Ternary (GB)</th>")
        html.append("<th>Dense243 (GB)</th>")
        html.append("</tr>")

        for model in data:
            html.append("<tr>")
            html.append(f"<td style='text-align: left;'>{model['name']}</td>")
            html.append(f"<td>{model['fp16_gb']:.2f}</td>")
            html.append(f"<td>{model['int8_gb']:.2f}</td>")
            html.append(f"<td>{model['ternary_gb']:.2f}</td>")
            html.append(f"<td>{model['dense243_gb']:.2f}</td>")
            html.append("</tr>")

        html.append("</table>")

        html.append("<div class='summary'>")
        html.append("<p><strong>Ternary advantage:</strong> 4.0x smaller than INT8, 2.0x smaller than INT4</p>")
        html.append("</div>")

        return "\n".join(html)

    def _generate_phase4_html(self) -> str:
        """Generate HTML for Phase 4"""
        if 'phase4_neural_workload_patterns' not in self.results:
            return ""

        data = self.results['phase4_neural_workload_patterns']
        if not data:
            return ""

        html = []
        html.append("<h2>Phase 4: Neural Network Workloads</h2>")
        html.append("<table>")
        html.append("<tr>")
        html.append("<th>Layer</th>")
        html.append("<th>Shape</th>")
        html.append("<th>Speedup</th>")
        html.append("<th>Ternary GOPS</th>")
        html.append("</tr>")

        for result in data:
            shape_str = f"{result['shape'][0]}x{result['shape'][1]}"
            html.append("<tr>")
            html.append(f"<td style='text-align: left;'>{result['name']}</td>")
            html.append(f"<td>{shape_str}</td>")
            html.append(f"<td>{result['speedup']:.2f}x</td>")
            html.append(f"<td>{result['ternary_gops']:.2f}</td>")
            html.append("</tr>")

        html.append("</table>")

        avg_speedup = sum(r['speedup'] for r in data) / len(data)
        html.append("<div class='summary'>")
        html.append(f"<p><strong>Average matmul speedup:</strong> <span class='highlight'>{avg_speedup:.2f}x</span></p>")
        html.append("</div>")

        return "\n".join(html)

    def save_text_report(self, output_file: str = "report.txt"):
        """Save text report to file"""
        report = self.generate_text_report()

        with open(output_file, 'w') as f:
            f.write(report)

        print(f"Text report saved: {output_file}")


def main():
    """Example usage"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualization.py <results_file.json> [output.html]")
        sys.exit(1)

    results_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "report.html"

    viz = BenchmarkVisualizer()
    viz.load_results(results_file)

    # Generate both text and HTML reports
    print(viz.generate_text_report())
    viz.save_text_report("report.txt")
    viz.generate_html_report(output_file)


if __name__ == "__main__":
    main()
