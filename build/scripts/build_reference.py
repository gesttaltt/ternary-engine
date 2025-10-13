#!/usr/bin/env python3
# build_reference.py — Reference baseline build (minimal optimization)
#
# Copyright 2025 Ternary Core Contributors
# Licensed under the Apache License, Version 2.0
#
# Usage:
#   python build/scripts/build_reference.py
#
# Note: Requires benchmarks/reference_cpp.cpp to exist
#       This is a low-optimization baseline for performance comparison

from pathlib import Path
import sys
import platform
import json

# Add templates directory to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build" / "scripts"))

from templates.ext_build import build_module

# Output directory
ART = ROOT / "build" / "artifacts" / "reference"

# Platform-specific compiler flags (minimal optimization)
if platform.system() == "Windows":
    flags = ["/O1", "/std:c++17", "/EHsc"]
else:
    flags = ["-O1", "-std=c++17"]

# Build the module
source = ROOT / "benchmarks" / "reference_cpp.cpp"

if not source.exists():
    print(f"ERROR: Reference implementation not found at {source}", file=sys.stderr)
    print("Create benchmarks/reference_cpp.cpp to build reference baseline", file=sys.stderr)
    sys.exit(1)

meta = build_module("reference_cpp", str(source), flags, ART)
meta["type"] = "reference"

# Output metadata
print(json.dumps(meta, indent=2))
