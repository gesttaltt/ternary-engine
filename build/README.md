# Build System

This directory contains the build infrastructure for the ternary-kernel-python-c library.

## Structure

```
build/
├── scripts/           # Build automation scripts
│   ├── setup.py              # Standard optimized build
│   ├── setup_pgo.py          # Profile-Guided Optimization build
│   ├── setup_reference.py    # Reference baseline build
│   ├── build_standard.py     # Standard build script
│   ├── build_reference.py    # Reference build script
│   ├── build_benchmark.py    # Benchmark build script
│   └── templates/            # Build templates
│       ├── __init__.py
│       └── ext_build.py
│
└── artifacts/         # Build outputs (timestamped)
    ├── final/                # Latest production builds
    ├── reference/            # Reference builds for comparison
    ├── manifest.txt          # Build manifest
    ├── archive_artifacts.sh  # Artifact archival script
    └── *.pyd                 # Compiled Python extensions

```

## Quick Start

### Standard Build (Recommended)

From the project root:

```bash
python build/scripts/setup.py
```

This produces an optimized build with:
- `/O2` (MSVC) or `-O3` (GCC/Clang) optimization
- AVX2 SIMD vectorization
- OpenMP parallelization
- Link-time optimization (LTO/LTCG)

### Profile-Guided Optimization (Maximum Performance)

For 5-15% additional performance:

```bash
python build/scripts/setup_pgo.py
```

See `docs/PGO_README.md` for detailed PGO documentation.

### Reference Build (Baseline Comparison)

For benchmarking and regression testing:

```bash
python build/scripts/setup_reference.py
```

## Build Outputs

### Timestamped Artifacts

All builds create timestamped artifacts in `build/artifacts/`:

- **final/** - Latest production-ready builds
- **reference/** - Baseline builds for performance comparison
- **manifest.txt** - Build metadata and timestamps

### Python Extensions

Built extensions (`.pyd` on Windows, `.so` on Linux/macOS):

- `ternary_core_simd_full.*.pyd` - Main optimized library
- `reference_cpp.*.pyd` - Reference implementation for benchmarks

## Build Flags

### MSVC (Windows)

```
/O2              Maximum speed optimization
/GL              Whole program optimization
/arch:AVX2       Enable AVX2 instructions
/openmp          Enable OpenMP parallelization
/LTCG            Link-time code generation
/std:c++17       C++17 standard
```

### GCC/Clang (Linux/macOS)

```
-O3              Maximum optimization
-march=native    Native CPU architecture
-mavx2           Enable AVX2 instructions
-fopenmp         Enable OpenMP parallelization
-flto            Link-time optimization
-std=c++17       C++17 standard
-fPIC            Position-independent code
```

## Manual Compilation

### Linux/macOS

```bash
c++ -O3 -march=native -mavx2 -fopenmp -flto -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    ternary_simd_engine.cpp \
    -o ternary_simd_engine$(python3-config --extension-suffix)
```

### Windows (MSVC)

```bash
cl /O2 /GL /arch:AVX2 /openmp /std:c++17 /EHsc /LD ^
   /I"%PYTHON_INCLUDE%" /I"%PYBIND11_INCLUDE%" ^
   ternary_simd_engine.cpp ^
   /link /LTCG /OUT:ternary_simd_engine.pyd
```

## Troubleshooting

### Windows Path Length Issues

If you encounter "filename or extension is too long" errors on Windows:

1. Enable long paths via Group Policy or registry
2. Use shorter directory names
3. Build from a directory closer to root (e.g., `C:\dev\`)

### Missing Dependencies

Ensure you have installed:

```bash
pip install pybind11 numpy
```

### AVX2 Not Supported

Check CPU compatibility:

```bash
# Linux
grep avx2 /proc/cpuinfo

# Windows PowerShell
Get-WmiObject -Class Win32_Processor | Select-Object -Property Name
```

Required: Intel Haswell (2013+) or AMD Excavator (2015+)

## Build Scripts Documentation

### setup.py

Standard optimized build with production flags. Outputs to `build/artifacts/final/`.

**Usage**: `python build/scripts/setup.py`

### setup_pgo.py

Two-phase Profile-Guided Optimization build:

1. **Instrumentation phase**: Builds with profiling hooks
2. **Training phase**: Runs representative workload
3. **Optimization phase**: Rebuilds with profile data

**Usage**: `python build/scripts/setup_pgo.py`

See `docs/PGO_README.md` for details.

### setup_reference.py

Baseline build for benchmarking. Uses conservative optimizations for fair comparison.

**Usage**: `python build/scripts/setup_reference.py`

## Related Documentation

- **[../docs/build-system/README.md](../docs/build-system/README.md)** - Detailed build system documentation
- **[../docs/PGO_README.md](../docs/PGO_README.md)** - Profile-Guided Optimization guide
- **[../docs/build-system/artifact-organization.md](../docs/build-system/artifact-organization.md)** - Artifact management
- **[../docs/build-system/setup-standard.md](../docs/build-system/setup-standard.md)** - Standard build details
- **[../docs/build-system/setup-pgo.md](../docs/build-system/setup-pgo.md)** - PGO build details
- **[../docs/build-system/setup-reference.md](../docs/build-system/setup-reference.md)** - Reference build details

## Artifact Management

### Archiving Old Builds

The `archive_artifacts.sh` script manages old build artifacts:

```bash
cd build/artifacts
./archive_artifacts.sh
```

This creates timestamped archives of old builds.

### Cleaning Artifacts

To remove all build artifacts (clean build):

```bash
# Windows
rmdir /s /q build\artifacts

# Linux/macOS
rm -rf build/artifacts/*
```

Then rebuild using your preferred build script.

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Test

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install pybind11 numpy
      - name: Build library
        run: python build/scripts/setup.py
      - name: Run tests
        run: python tests/test_phase0.py
```

## Performance Notes

### Build Time

- Standard build: ~30 seconds (MSVC), ~20 seconds (GCC)
- PGO build: ~2 minutes (includes training phase)

### Binary Size

- Optimized build: ~150 KB (.pyd/.so)
- Debug build: ~500 KB (with symbols)

## Version History

- **Phase 0**: Initial LUT optimization
- **Phase 1**: Multi-path SIMD optimization
- **Phase 2**: Complexity compression (current)
- **Phase 3**: Production refinements (2025-10-13)

See `CHANGELOG.md` for detailed version history.

---

**Last Updated**: 2025-10-13
**Maintained by**: Ternary Core Contributors
