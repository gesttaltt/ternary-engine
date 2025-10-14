# Standard Build Script (build.py)

## Overview

`build.py` is the primary build script for creating production-ready, fully optimized builds of the `ternary_simd_engine` module. It produces AVX2-accelerated binaries with OpenMP multi-threading support.

**Location:** `build.py` (project root)

**Module produced:** `ternary_simd_engine.cp312-win_amd64.pyd`

**Typical build time:** 30-60 seconds

## Quick Start

```bash
# From project root
python build.py
```

That's it! The script handles everything automatically:
1. Generates timestamp
2. Creates build directories
3. Compiles with full optimizations
4. Copies artifacts to `latest/` and project root

## Usage

### Basic Build

```bash
# Standard usage (recommended)
python build/scripts/setup.py
```

### Build Output

```
======================================================================
  STANDARD OPTIMIZED BUILD
  Timestamp: 20251012_143022
======================================================================

Created build directories:
  Temp:   H:\...\build\artifacts\standard\20251012_143022\temp
  Output: H:\...\build\artifacts\standard\20251012_143022\output

Building ternary_simd_engine module...

[Compiler output...]

Copying to latest directory...
  ✓ ternary_simd_engine.cp312-win_amd64.pyd → H:\...\ternary_simd_engine.cp312-win_amd64.pyd

======================================================================
  ✅ BUILD COMPLETE
======================================================================

Build artifacts:
  Timestamped: H:\...\build\artifacts\standard\20251012_143022
  Latest:      H:\...\build\artifacts\standard\latest

Generated modules:
  - ternary_simd_engine.cp312-win_amd64.pyd (145.4 KB)
```

## Technical Details

### Build Process

The script executes the following workflow:

```
1. Timestamp Generation
   ↓
2. Directory Setup
   ├── standard/YYYYMMDD_HHMMSS/temp/
   └── standard/YYYYMMDD_HHMMSS/output/
   ↓
3. Dynamic Setup Generation
   └── Creates temporary setup_temp.py with embedded paths
   ↓
4. Compilation (setuptools + MSVC)
   ├── Preprocessing
   ├── Compilation → .obj files
   └── Linking → .pyd file
   ↓
5. Post-Build Actions
   ├── Copy to latest/
   └── Copy to project root
   ↓
6. Cleanup
   └── Remove setup_temp.py
```

### Compiler Flags

#### Compile Flags

| Flag | Purpose | Impact |
|------|---------|--------|
| `/O2` | Maximum optimization | Speed over size, aggressive inlining |
| `/GL` | Whole program optimization | Cross-module optimizations |
| `/arch:AVX2` | Enable AVX2 instructions | 4-8x SIMD speedup |
| `/openmp` | OpenMP multi-threading | Parallel array processing (>100K elements) |
| `/std:c++17` | C++17 standard | Modern C++ features |
| `/EHsc` | Exception handling | C++ exceptions enabled |

#### Link Flags

| Flag | Purpose | Impact |
|------|---------|--------|
| `/LTCG` | Link-time code generation | Whole-program optimization at link stage |

### Optimization Impact

| Optimization | Expected Speedup | Applies To |
|--------------|------------------|------------|
| AVX2 SIMD | 4-8x | Element-wise operations |
| OpenMP | 2-4x (4 cores) | Arrays >100K elements |
| LUT-based ops | 2-3x | Ternary operations |
| `/O2` + `/GL` | 1.5-2x | Overall |
| **Combined** | **10-50x** | Baseline vs optimized |

## Source Files

### Compiled Sources

| File | Description | Lines | Purpose |
|------|-------------|-------|---------|
| `ternary_simd_engine.cpp` | Main SIMD implementation | ~297 | AVX2 vectorized operations |
| `ternary_algebra.h` | Core header | ~111 | Scalar operations, LUTs |

### Build Script Structure

```python
# build/scripts/setup.py (simplified)

# 1. Path resolution
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "build" / "artifacts"

# 2. Timestamp generation
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# 3. Directory structure
BUILD_TYPE_DIR = ARTIFACTS_DIR / "standard"
BUILD_TIMESTAMP_DIR = BUILD_TYPE_DIR / TIMESTAMP
BUILD_TEMP_DIR = BUILD_TIMESTAMP_DIR / "temp"
BUILD_OUTPUT_DIR = BUILD_TIMESTAMP_DIR / "output"
BUILD_LATEST_DIR = BUILD_TYPE_DIR / "latest"

# 4. Build workflow
def main():
    print_header()          # Display build info
    setup_directories()     # Create directory structure
    build_module()          # Invoke setuptools/MSVC
    copy_to_latest()        # Update latest/ and project root
    print_summary()         # Show results
```

## Directory Layout

### Before Build

```
build/
├── artifacts/
│   └── standard/
│       └── (empty or previous builds)
└── scripts/
    └── setup.py
```

### After Build

```
build/
├── artifacts/
│   └── standard/
│       ├── 20251012_143022/          # New timestamped build
│       │   ├── temp/
│       │   │   ├── Release/
│       │   │   │   ├── ternary_simd_engine.obj          (8.2 MB)
│       │   │   │   ├── ternary_simd_engine.*.exp        (899 bytes)
│       │   │   │   └── ternary_simd_engine.*.lib        (2.3 KB)
│       │   │   └── (setup_temp.py deleted)
│       │   └── output/
│       │       └── ternary_simd_engine.cp312-win_amd64.pyd  (145 KB)
│       └── latest/                   # Copy of 20251012_143022/
│           ├── temp/
│           └── output/
└── scripts/
    └── setup.py
```

### Project Root

```
ternary-engine/
├── ternary_simd_engine.cp312-win_amd64.pyd    # Copied for convenience
└── (other files...)
```

## Advanced Usage

### Custom Build Location

The script uses automatic path resolution, but you can modify the build location by editing:

```python
# In setup.py, modify these lines:
ARTIFACTS_DIR = PROJECT_ROOT / "build" / "artifacts"  # Change to custom path
BUILD_TYPE_DIR = ARTIFACTS_DIR / "standard"          # Or rename "standard"
```

### Verbose Compilation

To see detailed compiler output:

```bash
# Redirect stderr to see all compiler messages
python build/scripts/setup.py 2>&1 | tee build.log
```

### Build Without Copying to Root

```python
# In setup.py, comment out the copy-to-root section in copy_to_latest():
def copy_to_latest():
    # ... existing code ...

    # # Also copy .pyd to project root for convenience
    # for pyd_file in BUILD_OUTPUT_DIR.glob("*.pyd"):
    #     dest = PROJECT_ROOT / pyd_file.name
    #     shutil.copy2(pyd_file, dest)
    #     print(f"  ✓ {pyd_file.name} → {dest}")
```

## Dependencies

### Required

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | 3.7+ | Build system |
| setuptools | Latest | Python extension build |
| pybind11 | Latest | C++/Python binding |
| MSVC | 2019+ | C++ compiler (Windows) |
| AVX2 support | - | Target CPU requirement |

### Installing Dependencies

```bash
# Python packages
pip install setuptools pybind11

# MSVC (Visual Studio 2019 or later)
# Install from: https://visualstudio.microsoft.com/downloads/
# Select "Desktop development with C++" workload
```

### Verifying Dependencies

```bash
# Check Python
python --version

# Check setuptools
python -c "import setuptools; print(setuptools.__version__)"

# Check pybind11
python -c "import pybind11; print(pybind11.__version__)"

# Check MSVC (Windows)
cl.exe /?
```

## Performance Characteristics

### Build Time

| System | Build Time | Notes |
|--------|------------|-------|
| Modern desktop (8-core) | 30-45 sec | Typical |
| Laptop (4-core) | 45-60 sec | Typical |
| Low-end (2-core) | 60-90 sec | Acceptable |

**Bottlenecks:**
1. **Compilation phase** (~70% of time): Single-threaded per-file compilation
2. **Linking phase** (~20% of time): Link-time code generation with `/LTCG`
3. **File I/O** (~10% of time): Writing large `.obj` files

### Artifact Sizes

| Artifact | Size | Notes |
|----------|------|-------|
| `.obj` (object file) | 8.2 MB | Debug symbols included |
| `.pyd` (final module) | 145 KB | Stripped, optimized |
| `.lib` (import library) | 2.3 KB | Symbol table |
| `.exp` (export file) | 899 bytes | Export table |
| **Total per build** | ~8.4 MB | Can be reduced (see below) |

### Size Optimization

**To reduce `.obj` size:**

```python
# Add to extra_compile_args in setup.py:
extra_compile_args=[
    # ... existing flags ...
    '/Zi-',     # Disable debug info (reduces .obj by ~80%)
]
```

**To reduce `.pyd` size:**

```python
# Add to extra_link_args:
extra_link_args=[
    # ... existing flags ...
    '/OPT:REF',     # Remove unreferenced code
    '/OPT:ICF',     # Identical COMDAT folding
]
```

## Troubleshooting

### Build Failures

#### Error: "MSVC not found"

**Symptom:**
```
error: Microsoft Visual C++ 14.0 or greater is required
```

**Solution:**
1. Install Visual Studio 2019 or later
2. Select "Desktop development with C++" workload
3. Restart terminal/IDE

---

#### Error: "pybind11 not found"

**Symptom:**
```
ModuleNotFoundError: No module named 'pybind11'
```

**Solution:**
```bash
pip install pybind11
```

---

#### Error: "AVX2 not supported"

**Symptom:**
```
Illegal instruction (core dumped)  # On runtime, not build
```

**Solution:**
- Remove `/arch:AVX2` flag from `extra_compile_args`
- Module will run on older CPUs but slower (no SIMD)

---

#### Error: "Permission denied writing to build/artifacts/"

**Symptom:**
```
PermissionError: [Errno 13] Permission denied: 'build/artifacts/...'
```

**Solution:**
```bash
# Check permissions
ls -la build/artifacts/

# Fix permissions
chmod -R u+w build/artifacts/

# Or run as admin (Windows)
# Right-click terminal → "Run as administrator"
```

### Runtime Issues

#### Error: "DLL load failed"

**Symptom:**
```python
>>> import ternary_simd_engine
ImportError: DLL load failed while importing ternary_simd_engine
```

**Causes:**
1. Missing Visual C++ Redistributable
2. Python version mismatch
3. Corrupted build

**Solution:**
```bash
# 1. Install VC++ Redistributable
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

# 2. Verify Python version matches
python --version  # Must match .pyd suffix (e.g., cp312 = Python 3.12)

# 3. Rebuild
python build/scripts/setup.py
```

---

#### Error: "Module has no attribute 'tadd'"

**Symptom:**
```python
>>> import ternary_simd_engine as tc
>>> tc.tadd([1], [2])
AttributeError: module 'ternary_simd_engine' has no attribute 'tadd'
```

**Cause:** Wrong module version loaded (old build)

**Solution:**
```bash
# Force reload
python -c "import sys; sys.path.insert(0, 'build/artifacts/standard/latest/output'); import ternary_simd_engine; print(dir(ternary_simd_engine))"

# Or clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
```

## Integration Examples

### Python Script

```python
#!/usr/bin/env python
import sys
import numpy as np

# Use latest build
sys.path.insert(0, 'build/artifacts/standard/latest/output')
import ternary_simd_engine as tc

# Create test arrays
a = np.array([0, 1, 2, 0, 1], dtype=np.uint8)
b = np.array([1, 1, 1, 2, 2], dtype=np.uint8)

# Perform ternary addition
result = tc.tadd(a, b)
print(f"tadd({a}, {b}) = {result}")
```

### pytest Configuration

```python
# conftest.py
import sys
from pathlib import Path

def pytest_configure(config):
    # Add latest build to path
    artifacts = Path(__file__).parent / "build" / "artifacts"
    latest = artifacts / "standard" / "latest" / "output"
    sys.path.insert(0, str(latest))
```

### Docker Build

```dockerfile
# Dockerfile
FROM python:3.12-windowsservercore

# Install dependencies
RUN pip install setuptools pybind11

# Copy source
COPY . /app
WORKDIR /app

# Build
RUN python build/scripts/setup.py

# Test
RUN python -c "import ternary_simd_engine; print('Build OK')"
```

## Comparison with Other Builds

| Feature | Standard | PGO | Reference |
|---------|----------|-----|-----------|
| AVX2 SIMD | ✅ Yes | ✅ Yes | ❌ No |
| OpenMP | ✅ Yes | ✅ Yes | ❌ No |
| LUTs | ✅ Yes | ✅ Yes | ❌ No |
| Profile-guided | ❌ No | ✅ Yes | ❌ No |
| Build time | 30-60s | 8-10 min | 25-45s |
| Expected speedup | 10-50x | 15-60x | 1x (baseline) |
| Use case | Production | Critical perf | Benchmarking |

## See Also

- [Artifact Organization](./artifact-organization.md) - Build output structure
- [PGO Build](./setup-pgo.md) - Profile-guided optimization
- [Reference Build](./setup-reference.md) - Unoptimized baseline
- [Build System Overview](./README.md) - Complete build system documentation
