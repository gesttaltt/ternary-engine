#!/bin/bash

# Competitive Benchmark Runner
# Runs all competitive benchmarks and generates reports

set -e

echo "=================================="
echo "Ternary Competitive Benchmark Suite"
echo "=================================="
echo ""

# Create results directory structure
mkdir -p results/competitive
mkdir -p results/quantization
mkdir -p results/power
mkdir -p results/reports

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null || echo "Warning: Could not activate venv"

# Install dependencies
echo "Checking dependencies..."
pip install numpy --quiet

# Run benchmarks
echo ""
echo "=================================="
echo "Running Competitive Benchmarks"
echo "=================================="
echo ""

# Run main competitive suite
echo "[1/3] Running main competitive suite (Phases 1-6)..."
python bench_competitive.py --all

# Get latest results file
LATEST_RESULTS=$(ls -t results/competitive/competitive_results_*.json 2>/dev/null | head -1)

if [ -z "$LATEST_RESULTS" ]; then
    echo "Error: No results file found"
    exit 1
fi

echo ""
echo "Results saved to: $LATEST_RESULTS"

# Generate reports
echo ""
echo "[2/3] Generating text report..."
python utils/visualization.py "$LATEST_RESULTS" results/reports/report.txt

echo ""
echo "[3/3] Generating HTML report..."
python utils/visualization.py "$LATEST_RESULTS" results/reports/report.html

echo ""
echo "=================================="
echo "Benchmark Complete!"
echo "=================================="
echo ""
echo "Results:"
echo "  JSON:  $LATEST_RESULTS"
echo "  Text:  results/reports/report.txt"
echo "  HTML:  results/reports/report.html"
echo ""
echo "Next steps:"
echo "  1. Review results/report.html in browser"
echo "  2. Check competitive viability checklist"
echo "  3. Run model quantization (Phase 5) if PyTorch available"
echo "  4. Run power consumption (Phase 6) if on supported hardware"
echo ""
