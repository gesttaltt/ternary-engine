from setuptools import setup, Extension
import sys

# Build test executable as a Python extension (we'll just run it from C++)
test_module = Extension(
    'test_simd_native',
    sources=['test_simd_correctness.cpp'],
    include_dirs=['..'],
    extra_compile_args=[
        '/std:c++17' if sys.platform == 'win32' else '-std=c++17',
        '/O2' if sys.platform == 'win32' else '-O2',
        '/arch:AVX2' if sys.platform == 'win32' else '-mavx2',
        '/EHsc' if sys.platform == 'win32' else '',
    ],
    language='c++'
)

setup(
    name='test_simd_native',
    ext_modules=[test_module],
)
