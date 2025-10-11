"""
setup.py - Build script for ternary_core_simd_full module

Copyright 2025 Ternary Core Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This provides a proper build system for the Phase 0 optimized module.
"""

from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'ternary_core_simd_full',
        ['ternary_core_simd_full.cpp'],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            '.'  # For ternary_core.h
        ],
        language='c++',
        extra_compile_args=[
            '/O2',           # MSVC: Maximum optimization
            '/GL',           # MSVC: Whole program optimization
            '/arch:AVX2',    # MSVC: Enable AVX2
            '/std:c++17',    # C++17 standard
            '/EHsc',         # Exception handling
        ],
        extra_link_args=[
            '/LTCG',         # Link-time code generation
        ],
    ),
]

setup(
    name='ternary_core_simd_full',
    version='0.1.0',
    author='Ternary Core Team',
    description='AVX2-optimized ternary logic operations with Phase 0 LUT optimizations',
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires='>=3.7',
)
