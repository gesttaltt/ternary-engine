#!/usr/bin/env python3
# build_benchmark.py — Benchmark-ready build with metadata
#
# Copyright 2025 Ternary Core Contributors
# Licensed under the Apache License, Version 2.0
#
# Usage:
#   python build/scripts/build_benchmark.py
#
# This script wraps build_standard.py and adds benchmark-specific metadata

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "benchmarks" / "results"
BENCH_DIR.mkdir(parents=True, exist_ok=True)

print("Building optimized engine for benchmarks...")

# Run standard build
build_script = ROOT / "build" / "scripts" / "build_standard.py"
res = subprocess.check_output([sys.executable, str(build_script)])
meta = json.loads(res)

# Add benchmark-specific metadata
meta["benchmark_ready"] = True

# Get git commit hash if available
try:
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        stderr=subprocess.DEVNULL,
        cwd=str(ROOT)
    ).decode().strip()
    meta["commit_hash"] = commit
except:
    meta["commit_hash"] = "unknown"

# Add compiler info
if sys.platform == "win32":
    meta["compiler"] = "MSVC /O2 AVX2"
else:
    meta["compiler"] = "clang++ -O3"

# Write metadata to benchmarks/results
meta_file = BENCH_DIR / f"build_meta_{meta['timestamp']}.json"
meta_file.write_text(json.dumps(meta, indent=2))

print(f"✅ Benchmark build complete: {meta['timestamp']}")
print(f"   Commit: {meta['commit_hash']}")
print(f"   Metadata: {meta_file}")
print(json.dumps(meta, indent=2))
