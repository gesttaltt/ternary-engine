Build the Ternary Engine standard module with optimized compiler flags.

Run the build script:
```bash
python build/build.py
```

This will:
- Compile ternary_simd_engine.cpp with C++17 and AVX2 optimizations
- Create timestamped build artifacts in build/artifacts/standard/<timestamp>/
- Copy the compiled module to project root for easy import
- Perform platform-specific optimization (MSVC on Windows, GCC/Clang on Linux/macOS)

After building, verify the module:
```bash
python -c "import ternary_simd_engine; print('Build successful!')"
```

Note: Only Windows x64 builds are production-validated. Linux/macOS builds are experimental.
