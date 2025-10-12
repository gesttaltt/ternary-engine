"""
setup_reference.py - Build unoptimized C++ reference for benchmarking

Copyright 2025 Ternary Core Contributors
Licensed under the Apache License, Version 2.0

This builds a baseline C++ implementation WITHOUT optimizations:
- No LUTs (uses conversion-based operations)
- No SIMD
- No force inline
- Minimal compiler optimizations (/O1 instead of /O2)

PURPOSE: Fair performance comparison to measure actual optimization impact,
not Python vs C++ differences.
"""

from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'reference_cpp',
        ['benchmarks/reference_cpp.cpp'],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            '.'
        ],
        language='c++',
        extra_compile_args=[
            '/O1',           # MSVC: Basic optimization only (NOT /O2)
            '/std:c++17',    # C++17 standard
            '/EHsc',         # Exception handling
            # NO /GL, /arch:AVX2, /openmp - minimal optimizations
        ],
        # NO /LTCG - no link-time optimization
    ),
]

setup(
    name='reference_cpp',
    version='0.1.0',
    author='Ternary Core Team',
    description='Unoptimized C++ reference for fair benchmarking',
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires='>=3.7',
)
