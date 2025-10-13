#!/usr/bin/env python3
# build_standard.py — Standard production build for ternary_simd_engine
#
# Copyright 2025 Ternary Core Contributors
# Licensed under the Apache License, Version 2.0
#
# Usage:
#   python build/scripts/build_standard.py

from pathlib import Path
import sys
import platform
import json

# Add templates directory to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build" / "scripts"))

from templates.ext_build import build_module

# Output directory
ART = ROOT / "build" / "artifacts" / "standard"

# Platform-specific compiler flags
if platform.system() == "Windows":
    flags = ["/O2", "/GL", "/arch:AVX2", "/openmp", "/std:c++17", "/EHsc"]
    link = ["/LTCG"]
else:
    flags = ["-O3", "-march=native", "-fopenmp", "-std=c++17"]
    link = ["-flto"]

# Build the module
source = ROOT / "ternary_simd_engine.cpp"
meta = build_module("ternary_simd_engine", str(source), flags, ART, link)
meta["type"] = "standard"

# Output metadata
print(json.dumps(meta, indent=2))
