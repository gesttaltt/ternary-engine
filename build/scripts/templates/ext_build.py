# templates/ext_build.py — Reusable setuptools builder
#
# Copyright 2025 Ternary Core Contributors
# Licensed under the Apache License, Version 2.0

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime


def build_module(name, source, flags, outdir, extra_link_args=None):
    """
    Build a Python C++ extension module using setuptools.

    Args:
        name: Module name
        source: Path to .cpp source file
        flags: List of compiler flags
        outdir: Output directory for artifacts
        extra_link_args: Optional list of linker flags

    Returns:
        dict: Build metadata (name, timestamp, output path)
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    build_dir = outdir / ts
    temp = build_dir / "temp"
    out = build_dir / "output"
    latest = outdir / "latest"

    # Create directories
    for d in (temp, out):
        d.mkdir(parents=True, exist_ok=True)

    # Generate setup.py content
    setup_code = f"""
from setuptools import setup, Extension
import pybind11
import os

ext_modules = [
    Extension(
        '{name}',
        [r'{source}'],
        include_dirs=[pybind11.get_include(), pybind11.get_include(user=True)],
        language='c++',
        extra_compile_args={flags},
        extra_link_args={extra_link_args or []},
    )
]

setup(
    name='{name}',
    version='0.1.0',
    ext_modules=ext_modules,
    zip_safe=False
)
"""

    # Write temporary setup.py
    setup_temp = temp / "setup_temp.py"
    setup_temp.write_text(setup_code)

    # Run setuptools build
    subprocess.run([
        sys.executable, str(setup_temp), "build_ext",
        "--build-temp", str(temp),
        "--build-lib", str(out)
    ], check=True)

    # Update "latest" symlink (or copy on Windows)
    if latest.exists():
        import shutil
        shutil.rmtree(latest)
    import shutil
    shutil.copytree(build_dir, latest)

    return {"name": name, "timestamp": ts, "output": str(out)}
