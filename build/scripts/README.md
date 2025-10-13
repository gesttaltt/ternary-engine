# Build Scripts

Modular build system for ternary SIMD engine with deterministic, timestamped artifacts.

## Structure

```
build/scripts/
├── templates/
│   ├── __init__.py
│   └── ext_build.py         # Reusable setuptools builder
├── build_standard.py        # Production build (/O2, AVX2, LTO)
├── build_reference.py       # Baseline build (/O1, no SIMD)
└── build_benchmark.py       # Wraps standard + metadata
```

## Usage

### Standard Production Build

```bash
python build/scripts/build_standard.py
```

**Output:**
- Builds `ternary_simd_engine` module with maximum optimizations
- Platform-specific flags:
  - **Windows**: `/O2 /GL /arch:AVX2 /openmp /LTCG`
  - **Linux/macOS**: `-O3 -march=native -fopenmp -flto`
- Artifacts: `build/artifacts/standard/{timestamp}/`
- Symlink: `build/artifacts/standard/latest/`

### Reference Baseline Build

```bash
python build/scripts/build_reference.py
```

**Output:**
- Builds `reference_cpp` module with minimal optimizations
- No SIMD, no LTO
- Platform-specific flags:
  - **Windows**: `/O1`
  - **Linux/macOS**: `-O1`
- Artifacts: `build/artifacts/reference/{timestamp}/`

**Note:** Requires `benchmarks/reference_cpp.cpp` to exist.

### Benchmark Build (with metadata)

```bash
python build/scripts/build_benchmark.py
```

**Output:**
- Wraps `build_standard.py`
- Adds benchmark metadata:
  - Git commit hash
  - Compiler version
  - Timestamp
  - Build flags
- Writes: `benchmarks/results/build_meta_{timestamp}.json`

**Example metadata:**
```json
{
  "name": "ternary_simd_engine",
  "timestamp": "20251013_143052",
  "output": "build/artifacts/standard/20251013_143052/output",
  "type": "standard",
  "benchmark_ready": true,
  "commit_hash": "3f17247",
  "compiler": "clang++ -O3"
}
```

## Design Principles

### 1. Zero Duplication
All build logic is centralized in `templates/ext_build.py`. Individual build scripts only specify:
- Source file
- Compiler flags
- Output directory

### 2. Timestamped Artifacts
Each build creates a timestamped directory:
```
build/artifacts/standard/
├── 20251013_120000/
├── 20251013_130000/
└── latest/           # symlink/copy to most recent
```

This enables:
- Rollback to previous builds
- A/B performance testing
- CI artifact archiving

### 3. Cross-Platform
Automatic platform detection (`platform.system()`):
- Windows: MSVC flags
- Linux/macOS: GCC/Clang flags

### 4. CI-Ready
- Scripts output JSON to stdout
- Exit codes: 0 (success), 1 (failure)
- No interactive prompts

## Adding New Build Configurations

Example: PGO (Profile-Guided Optimization) build

```python
# build_pgo.py
from pathlib import Path
import sys
import platform
import json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build" / "scripts"))

from templates.ext_build import build_module

ART = ROOT / "build" / "artifacts" / "pgo"

if platform.system() == "Windows":
    flags = ["/O2", "/GL", "/arch:AVX2", "/openmp", "/LTCG:PGI"]
    link = ["/LTCG:PGI"]
else:
    flags = ["-O3", "-march=native", "-fopenmp", "-fprofile-generate"]
    link = ["-fprofile-generate"]

source = ROOT / "ternary_simd_engine.cpp"
meta = build_module("ternary_simd_engine", str(source), flags, ART, link)
meta["type"] = "pgo_instrumented"

print(json.dumps(meta, indent=2))
```

## Troubleshooting

### Path Length Issues (Windows)

Windows has a 260-character path limit. If builds fail:

1. Enable long paths:
   ```
   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
   ```

2. Use shorter workspace paths

3. Build at root level:
   ```
   C:\ternary\
   ```

### Missing Dependencies

Ensure you have:
- **pybind11**: `pip install pybind11`
- **setuptools**: `pip install setuptools`
- **Compiler**:
  - Windows: Visual Studio 2019+ (or Build Tools)
  - Linux: GCC 7+ or Clang 10+
  - macOS: Xcode command-line tools

### Import Errors

If `from templates.ext_build import build_module` fails:

```bash
# Ensure templates/__init__.py exists
touch build/scripts/templates/__init__.py

# Or use absolute import
export PYTHONPATH="${PYTHONPATH}:$(pwd)/build/scripts"
```

## See Also

- `../local-reports/build.md` - Full build system design
- `../local-reports/benchmark.md` - Benchmark suite integration
