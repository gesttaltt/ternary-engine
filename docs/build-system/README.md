# Build System Documentation

## Overview

This directory contains comprehensive documentation for the ternary-engine build system. The build system provides three distinct build configurations, each serving a specific purpose:

| Build Type | Script | Purpose | Performance | Build Time |
|------------|--------|---------|-------------|------------|
| **Standard** | `build.py` | Production deployments | 10-50x baseline | 30-60s |
| **PGO** | `build_pgo.py` | Performance-critical | 15-60x baseline | 8-10min |
| **Reference** | `build_reference.py` | Benchmarking baseline | 1x baseline | 25-45s |

All build scripts are located in `build/` and generate timestamped artifacts in `build/artifacts/`.

## Documentation Index

### Core Documentation

1. **[Cleanup System (clean_all.py)](./cleanup-system.md)** ⭐ NEW
   - Comprehensive build artifact cleanup
   - Automated cleanup strategies
   - Disk space management
   - Integration with benchmarking

2. **[Artifact Organization](./artifact-organization.md)**
   - Directory structure and naming conventions
   - Artifact types and their purposes
   - Disk space management
   - Retention policies

3. **[Standard Build (build.py)](./setup-standard.md)**
   - Production-ready optimized build
   - AVX2 SIMD + OpenMP + LUT optimizations
   - Quick start and usage guide
   - Technical implementation details

4. **[PGO Build (build_pgo.py)](./setup-pgo.md)**
   - Profile-Guided Optimization (3-phase build)
   - 5-15% additional performance gain
   - Detailed phase-by-phase walkthrough
   - Custom profiling workflows

5. **[Reference Build (build_reference.py)](./setup-reference.md)**
   - Unoptimized baseline for benchmarking
   - Fair performance comparisons
   - Measuring optimization impact
   - Academic/research use cases

## Quick Start Guide

### Choose Your Build Type

```bash
# Most users: Production-ready standard build
python build/build.py

# Performance-critical applications: PGO build
python build/build_pgo.py full

# Benchmarking/research: Reference baseline
python build/build_reference.py
```

### First-Time Setup

```bash
# 1. Install dependencies
pip install setuptools pybind11 numpy

# 2. Verify compiler (Windows: MSVC, Linux: GCC/Clang)
cl.exe /?      # Windows
gcc --version  # Linux

# 3. Build standard version
python build/build.py

# 4. Test
python -c "import ternary_simd_engine; print('Build successful!')"
```

### Building All Versions

```bash
# Build all three versions for comparison
python build/build_reference.py
python build/build.py
python build/build_pgo.py full

# Run benchmarks
python benchmarks/bench_phase0.py
```

## Build System Architecture

### Directory Structure

```
ternary-engine/
├── build/
│   └── artifacts/              # All build outputs
│       ├── standard/           # Standard optimized builds
│       │   ├── YYYYMMDD_HHMMSS/
│       │   │   ├── temp/       # Intermediate (.obj, .lib, .exp)
│       │   │   └── output/     # Final (.pyd)
│       │   └── latest/         # Most recent build
│       ├── pgo/                # Profile-guided builds
│       │   ├── instrumented/
│       │   ├── optimized/
│       │   ├── pgo_data/
│       │   └── latest/
│       └── reference/          # Baseline builds
│           ├── YYYYMMDD_HHMMSS/
│           └── latest/
│
├── scripts/
│   └── build/                  # Build scripts
│       ├── build.py            # Standard build
│       ├── build_pgo.py        # PGO build (3-phase)
│       └── build_reference.py  # Reference build
│
├── tests/                      # Test suite
│   └── run_tests.py            # Test runner
│
├── docs/
│   └── build-system/           # Build documentation (this directory)
│       ├── README.md           # This file
│       ├── artifact-organization.md
│       ├── setup-standard.md
│       ├── setup-pgo.md
│       └── setup-reference.md
│
├── benchmarks/                 # Benchmarking suite
│   ├── bench_phase0.py        # Main benchmark runner
│   └── reference_cpp.cpp      # Reference implementation source
│
├── ternary_simd_engine.cpp # Optimized implementation source
└── ternary_algebra.h              # Core header with LUTs
```

### Build Workflow

```
                    ┌─────────────────────┐
                    │   Developer runs    │
                    │   build script      │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Generate timestamp │
                    │  (YYYYMMDD_HHMMSS)  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Create directories  │
                    │  temp/ + output/    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Invoke setuptools  │
                    │  with MSVC/GCC      │
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
    ┌───────▼───────┐  ┌───────▼───────┐  ┌──────▼──────┐
    │ Preprocessing │  │  Compilation  │  │   Linking   │
    └───────┬───────┘  └───────┬───────┘  └──────┬──────┘
            │                  │                  │
            └──────────────────┼──────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Build artifacts   │
                    │   .obj, .lib, .pyd  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Copy to latest/    │
                    │  Copy to root       │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Build complete    │
                    └─────────────────────┘
```

## Build Comparison Matrix

### Feature Comparison

| Feature | Standard | PGO | Reference |
|---------|----------|-----|-----------|
| **SIMD (AVX2)** | ✅ Yes | ✅ Yes | ❌ No |
| **OpenMP Threading** | ✅ Yes | ✅ Yes | ❌ No |
| **LUT Operations** | ✅ Yes | ✅ Yes | ❌ No |
| **Compiler Opts** | `/O2 /GL /LTCG` | `/O2 /GL /LTCG:PGO` | `/O1` only |
| **Profile-Guided** | ❌ No | ✅ Yes | ❌ No |
| **Build Time** | 30-60s | 8-10min | 25-45s |
| **Binary Size** | ~145 KB | ~145-155 KB | ~125 KB |
| **Performance** | 10-50x | 15-60x | 1x (baseline) |

### Performance Characteristics

| Array Size | Reference | Standard | PGO | Notes |
|------------|-----------|----------|-----|-------|
| 100 elements | 250 ns | 50 ns (5x) | 48 ns (5.2x) | LUT benefit |
| 10K elements | 25 µs | 5 µs (5x) | 4.75 µs (5.3x) | + SIMD benefit |
| 1M elements | 2.5 ms | 100 µs (25x) | 86 µs (29x) | + OpenMP benefit |

**Key insights:**
- Small arrays: Dominated by LUT vs conversion (~5x)
- Medium arrays: SIMD vectorization adds another ~5x
- Large arrays: OpenMP parallelization adds 2-4x more
- PGO: Adds 5-15% on top of standard optimizations

### Use Case Recommendations

| Scenario | Recommended Build | Reason |
|----------|-------------------|--------|
| **Development/iteration** | Standard | Fast builds, full performance |
| **Production deployment** | Standard or PGO | Depends on perf requirements |
| **Performance-critical** | PGO | Extra 5-15% matters |
| **Benchmarking** | Reference | Fair baseline |
| **Research/papers** | Reference | Measure actual optimizations |
| **CI/CD testing** | Standard | Balance of speed and coverage |

## Common Tasks

### Building for Production

```bash
# Standard build (recommended for most cases)
python build/build.py

# Or PGO if performance critical
python build/build_pgo.py full
```

### Measuring Performance Impact

```bash
# Build baseline
python build/build_reference.py

# Build optimized
python build/build.py

# Compare
python benchmarks/bench_phase0.py
```

### Testing a New Optimization

```bash
# Before changes
python build/build.py
python benchmarks/bench_phase0.py > before.txt

# Make changes to ternary_simd_engine.cpp...

# After changes
python build/build.py
python benchmarks/bench_phase0.py > after.txt

# Compare
diff before.txt after.txt
```

### Cleaning Build Artifacts

```bash
# Clean all build artifacts (recommended)
python build/clean_all.py

# Preview what would be deleted
python build/clean_all.py --dry-run

# Keep latest builds
python build/clean_all.py --keep-latest

# Keep recent benchmark results
python build/clean_all.py --keep-results 5

# See full documentation
# docs/build-system/cleanup-system.md
```

## Environment Requirements

### Required Software

| Component | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.7+ | Build system and runtime |
| **setuptools** | Latest | Python extension building |
| **pybind11** | Latest | C++/Python bindings |
| **MSVC** | 2019+ (Windows) | C++ compiler |
| **GCC/Clang** | 9+ (Linux) | C++ compiler |
| **AVX2 CPU** | - | Target CPU requirement |

### Installing Dependencies

**Windows:**
```powershell
# Python packages
pip install setuptools pybind11 numpy

# Visual Studio 2019+ with C++ workload
# Download from: https://visualstudio.microsoft.com/
```

**Linux:**
```bash
# Python packages
pip install setuptools pybind11 numpy

# GCC/Clang
sudo apt-get install build-essential  # Debian/Ubuntu
sudo yum install gcc-c++               # CentOS/RHEL
```

### Verifying Environment

```bash
# Check Python
python --version

# Check packages
python -c "import setuptools, pybind11; print('Packages OK')"

# Check compiler
cl.exe /?           # Windows
gcc --version       # Linux

# Check AVX2 support
python -c "import platform; print(platform.processor())"
```

## Troubleshooting

### Build Failures

**Symptom:** Build script fails with errors

**Common causes:**
1. Missing dependencies (setuptools, pybind11)
2. Compiler not found (MSVC, GCC)
3. Incorrect working directory
4. Permission issues

**Solutions:**
```bash
# Install dependencies
pip install setuptools pybind11

# Check working directory
pwd  # Should be project root

# Check permissions
ls -la build/
chmod -R u+w build/
```

### Runtime Errors

**Symptom:** Import fails or crashes

**Common causes:**
1. Python version mismatch
2. Missing Visual C++ Redistributable (Windows)
3. CPU doesn't support AVX2

**Solutions:**
```bash
# Check Python version
python --version  # Must match .pyd (e.g., cp312 = 3.12)

# Install VC++ Redistributable (Windows)
# Download: https://aka.ms/vs/17/release/vc_redist.x64.exe

# Test without AVX2 (build reference instead)
python build_reference.py
```

### Performance Issues

**Symptom:** Optimized build not faster than expected

**Possible causes:**
1. Small arrays (overhead dominates)
2. Old CPU (no AVX2)
3. Debug build accidentally used
4. System under load

**Diagnosis:**
```bash
# Verify build type
python -c "import ternary_simd_engine; print(ternary_simd_engine.__file__)"
# Should point to standard/latest/output/ or pgo/latest/output/

# Check CPU features
lscpu | grep avx2  # Linux
wmic cpu get caption  # Windows

# Run benchmarks with profiling
python -m cProfile benchmarks/bench_phase0.py
```

## Advanced Topics

### Custom Build Flags

To modify compiler flags, edit the respective `build*.py` script:

```python
# In build.py
extra_compile_args=[
    '/O2',           # Keep maximum optimization
    '/GL',           # Keep whole program optimization
    '/arch:AVX2',    # Or change to AVX512: '/arch:AVX512'
    '/openmp',       # Keep OpenMP
    '/std:c++17',    # Or upgrade to C++20: '/std:c++20'
    '/EHsc',         # Keep exception handling
    '/favor:AMD64',  # Add CPU-specific tuning
]
```

### Cross-Compilation

```bash
# Windows ARM64 (from x64 host)
# Modify build.py:
extra_compile_args=[
    '/O2', '/GL', '/std:c++17', '/EHsc',
    # No /arch:AVX2 (ARM doesn't support)
    # Add ARM NEON flags instead
]

# Linux ARM (from x86 host)
CC=aarch64-linux-gnu-gcc python build.py
```

### Continuous Integration

See individual build documentation for CI/CD examples:
- [Standard Build CI](./setup-standard.md#integration-examples)
- [PGO Build CI](./setup-pgo.md#integration-with-cicd)
- [Reference Build CI](./setup-reference.md#integration-with-ci)

## Contributing

When contributing changes to the build system:

1. **Test all three builds:**
   ```bash
   python build/build_reference.py
   python build/build.py
   python build/build_pgo.py full
   ```

2. **Verify performance:**
   ```bash
   python benchmarks/bench_phase0.py
   ```

3. **Update documentation:**
   - Modify relevant `.md` file in `docs/build/`
   - Document any new flags or features

4. **Test on multiple platforms:**
   - Windows (MSVC)
   - Linux (GCC/Clang)
   - macOS (Clang)

5. **Update build paths**: Use `build/build.py` (not root-level `build.py`)

## FAQ

### Q: Which build should I use for production?

**A:** Start with **standard build** (`build.py`). It's fast, reliable, and provides 10-50x speedup over baseline. Only use PGO if you need that extra 5-15% and can afford 8-10 minute builds.

---

### Q: How often should I rebuild?

**A:** Rebuild whenever:
- Source code changes (`.cpp`, `.h` files)
- Compiler/toolchain updates
- Python version changes
- You want to test new optimizations

You don't need to rebuild if only Python code or docs change.

---

### Q: Can I distribute the `.pyd` file?

**A:** Yes, but consider:
- ✅ Same Python version required (e.g., 3.12.x)
- ✅ Same platform (Windows x64, Linux x64, etc.)
- ✅ Target CPU must support AVX2 (for standard/PGO builds)
- ✅ Visual C++ Redistributable required (Windows)

For wider compatibility, build without AVX2 or distribute multiple versions.

---

### Q: Why is my PGO build not faster?

**A:** Common reasons:
1. Profiling workload not representative
2. Code doesn't have hot paths (all paths equally important)
3. Already bottlenecked by memory bandwidth
4. Measurement noise (run multiple times)

Try profiling with your actual production workload.

---

### Q: Can I use both standard and PGO builds?

**A:** Yes! Build both and use appropriate one:
```python
import sys

if performance_critical:
    sys.path.insert(0, 'build/artifacts/pgo/latest/output')
else:
    sys.path.insert(0, 'build/artifacts/standard/latest/output')

import ternary_simd_engine
```

---

### Q: How do I clean up old builds?

**A:** Use the comprehensive cleanup utility:
```bash
# Clean all build artifacts
python build/clean_all.py

# Keep latest builds to avoid rebuilding
python build/clean_all.py --keep-latest

# Preview before cleaning
python build/clean_all.py --dry-run
```

See [Cleanup System Documentation](./cleanup-system.md) for full details.

## Additional Resources

### External Documentation

- [MSVC Compiler Reference](https://docs.microsoft.com/en-us/cpp/build/reference/compiler-options)
- [Intel AVX2 Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [OpenMP Specification](https://www.openmp.org/specifications/)
- [pybind11 Documentation](https://pybind11.readthedocs.io/)

### Project Documentation

- [Architecture Overview](../architecture.md)
- [Optimization Rationale](../optimization-complexity-rationale.md)
- [PGO Technical Details](../pgo/README.md)

### Performance Analysis Tools

- [Intel VTune Profiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html)
- [perf (Linux)](https://perf.wiki.kernel.org/)
- [Windows Performance Analyzer](https://docs.microsoft.com/en-us/windows-hardware/test/wpt/)

## Support

For build system issues:

1. Check this documentation first
2. Review individual build script documentation
3. Search existing issues: https://github.com/gesttaltt/ternary-engine/issues
4. Open new issue with:
   - OS and Python version
   - Compiler version
   - Full error output
   - Steps to reproduce

---

**Last updated:** 2025-10-12

**Documentation version:** 1.0.0
