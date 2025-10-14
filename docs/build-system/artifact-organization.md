# Build Artifact Organization

## Overview

The build system organizes all compilation artifacts in a structured, timestamp-based directory hierarchy under `build/artifacts/`. This approach provides:

- **Clean source tree**: No build artifacts pollute the project root or source directories
- **Historical tracking**: Each build is timestamped, allowing comparison and rollback
- **Build type separation**: Standard, PGO, and reference builds are kept separate
- **Easy access**: Latest builds are always available via `latest/` symlinks

## Directory Structure

```
build/
├── artifacts/
│   ├── standard/              # Standard optimized builds (AVX2 + OpenMP)
│   │   ├── YYYYMMDD_HHMMSS/   # Timestamped build directory
│   │   │   ├── temp/          # Intermediate build files (.obj, .exp, .lib)
│   │   │   └── output/        # Final compiled modules (.pyd)
│   │   └── latest/            # Symlink/copy to most recent build
│   │
│   ├── pgo/                   # Profile-Guided Optimization builds
│   │   ├── instrumented/      # Phase 1: Instrumented builds
│   │   │   └── YYYYMMDD_HHMMSS/
│   │   │       ├── temp/
│   │   │       └── output/
│   │   ├── optimized/         # Phase 3: Optimized builds
│   │   │   └── YYYYMMDD_HHMMSS/
│   │   │       ├── temp/
│   │   │       └── output/
│   │   ├── pgo_data/          # Profile data (.pgd, .pgc files)
│   │   └── latest/            # Latest optimized build
│   │
│   └── reference/             # Unoptimized reference builds
│       ├── YYYYMMDD_HHMMSS/
│       │   ├── temp/
│       │   └── output/
│       └── latest/
│
└── scripts/                   # Build scripts
    ├── setup.py               # Standard build
    ├── setup_pgo.py           # PGO build (3-phase)
    └── setup_reference.py     # Reference build
```

## Timestamp Format

All builds use the timestamp format: `YYYYMMDD_HHMMSS`

**Examples:**
- `20251012_143022` → October 12, 2025 at 14:30:22
- `20251225_090000` → December 25, 2025 at 09:00:00

This format ensures:
- Lexicographic sorting matches chronological order
- Human-readable without parsing
- Filesystem-safe (no special characters)
- Timezone-independent (uses local time)

## Artifact Types

### Output Files

Located in `{build-type}/{timestamp}/output/`:

| Extension | Description | Distributable | Size (typical) |
|-----------|-------------|---------------|----------------|
| `.pyd` | Python extension module (Windows DLL) | ✅ Yes | 125-150 KB |
| `.so` | Python extension module (Linux) | ✅ Yes | 100-130 KB |

### Intermediate Files

Located in `{build-type}/{timestamp}/temp/`:

| Extension | Description | Purpose | Size (typical) |
|-----------|-------------|---------|----------------|
| `.obj` | MSVC compiled object file | Linking | 7-8 MB |
| `.lib` | Import library | Linking | 2-3 KB |
| `.exp` | Export table | Linking | 800-900 bytes |
| `.o` | GCC/Clang object file | Linking | 5-6 MB |

### Profile Data (PGO Only)

Located in `build/artifacts/pgo/pgo_data/`:

| Extension | Description | Phase | Size (typical) |
|-----------|-------------|-------|----------------|
| `.pgd` | Profile-guided database | 1, 2, 3 | 100-200 KB |
| `.pgc` | Profile counter file | 2 | 10-50 KB |

## Build Type Comparison

### Standard Build (`standard/`)

**Characteristics:**
- Full AVX2 SIMD optimizations
- OpenMP multi-threading
- Whole program optimization (`/GL`, `/LTCG`)
- Maximum optimization level (`/O2`)

**Use case:** Production deployments

**Module:** `ternary_simd_engine.cp312-win_amd64.pyd`

**Typical size:** 145-150 KB

---

### PGO Build (`pgo/`)

**Characteristics:**
- All standard optimizations PLUS profile-guided optimization
- Uses runtime profiling data to optimize hot paths
- Three-phase build process
- 5-15% performance improvement expected

**Use case:** Performance-critical deployments

**Module:** `ternary_simd_engine.cp312-win_amd64.pyd`

**Typical size:** 145-155 KB

**Subdirectories:**
- `instrumented/` - Phase 1 builds with profiling instrumentation
- `optimized/` - Phase 3 builds using profile data
- `pgo_data/` - Profiling databases and counters

---

### Reference Build (`reference/`)

**Characteristics:**
- Minimal optimizations (`/O1`)
- No SIMD (scalar operations only)
- No whole program optimization
- Uses conversion-based operations (no LUTs)

**Use case:** Performance baseline for benchmarking

**Module:** `reference_cpp.cp312-win_amd64.pyd`

**Typical size:** 125-130 KB

## Latest Build Access

Each build type maintains a `latest/` directory containing a complete copy of the most recent build.

**Contents of `latest/`:**
```
latest/
├── temp/              # Complete intermediate build artifacts
│   ├── *.obj
│   ├── *.lib
│   ├── *.exp
│   └── setup_temp.py  (deleted after build)
└── output/
    └── *.pyd          # Final compiled module
```

**Access pattern:**
```python
# Always use the latest standard build
import sys
sys.path.insert(0, 'build/artifacts/standard/latest/output')
import ternary_simd_engine
```

**Note:** The `.pyd` file is also copied to the project root for convenience during development.

## Artifact Lifecycle

### Creation

1. **Build script invoked** → Timestamp generated
2. **Directories created** → `temp/` and `output/` under timestamped path
3. **Compilation** → Intermediate files written to `temp/`
4. **Linking** → Final `.pyd` written to `output/`
5. **Post-build** → Copy to `latest/` and project root

### Retention

- **All timestamped builds are kept** by default
- **Manual cleanup required** if disk space is a concern
- **Use `setup_pgo.py clean`** to remove all PGO artifacts

### Cleanup Recommendations

**Keep:**
- Last 5 builds of each type
- Any builds used for benchmarking
- Any builds deployed to production

**Remove:**
- Experimental/test builds
- Builds older than 30 days
- Failed/incomplete builds

**Automated cleanup script** (future enhancement):
```bash
# Keep only last 5 builds
find build/artifacts/standard -maxdepth 1 -type d -name "202*" |
  sort -r | tail -n +6 | xargs rm -rf
```

## Disk Space Analysis

### Typical Space Usage Per Build

| Build Type | Temp Files | Output Files | Total |
|------------|------------|--------------|-------|
| Standard | ~8 MB | ~150 KB | ~8.2 MB |
| PGO (full) | ~16 MB | ~150 KB | ~16.2 MB |
| Reference | ~8 MB | ~125 KB | ~8.1 MB |

### Space Optimization

**Option 1: Keep only output files**
- Delete `temp/` directory after successful build
- Saves ~8 MB per build
- Cannot relink without full rebuild

**Option 2: Compress old builds**
- Compress builds older than 7 days
- Expected compression ratio: 10:1 for `.obj` files
- Reduces ~8 MB to ~800 KB

**Option 3: External artifact storage**
- Move builds to external archive after 30 days
- Keep only `latest/` and last 5 builds locally

## Artifact Portability

### Same Machine

✅ **Direct use:** `.pyd` files can be used directly

### Different Windows Machine

⚠️ **Compatibility check required:**
- Python version must match exactly (e.g., 3.12.x)
- CPU must support AVX2 (standard/PGO builds)
- Visual C++ Redistributable must be installed

### Cross-Platform

❌ **Not portable:**
- `.pyd` files are Windows-specific
- Must rebuild on target platform
- Use same compiler and optimization flags for consistency

## Integration with CI/CD

### Artifact Archiving

```yaml
# Example GitHub Actions
- name: Archive build artifacts
  uses: actions/upload-artifact@v3
  with:
    name: ternary-engine-${{ github.sha }}
    path: |
      build/artifacts/*/latest/output/*.pyd
      build/artifacts/*/latest/output/*.so
```

### Build Comparison

```bash
# Compare two builds by size
ls -lh build/artifacts/standard/20251012_143022/output/*.pyd
ls -lh build/artifacts/standard/20251012_150000/output/*.pyd

# Compare by performance (requires benchmarks)
python benchmarks/bench_phase0.py --build build/artifacts/standard/20251012_143022/output
python benchmarks/bench_phase0.py --build build/artifacts/standard/20251012_150000/output
```

## Troubleshooting

### Build artifacts missing

**Symptom:** Expected files not in `output/` directory

**Causes:**
1. Build failed (check console output)
2. Permission issues (check write permissions)
3. Antivirus quarantine (whitelist project directory)

**Solution:**
```bash
# Check build logs
python build/scripts/setup.py 2>&1 | tee build.log

# Verify permissions
ls -la build/artifacts/

# Retry with verbose output
python build/scripts/setup.py --verbose
```

### Timestamped directory empty

**Symptom:** Directory created but no files inside

**Cause:** Build failed during compilation phase

**Solution:**
1. Check `temp/` directory for partial artifacts
2. Review compiler error messages
3. Verify all dependencies installed (pybind11, MSVC, etc.)

### Latest symlink broken

**Symptom:** `latest/` directory missing or pointing to deleted build

**Solution:**
```bash
# Recreate by re-running latest build type
python build/scripts/setup.py
```

## Future Enhancements

1. **Automatic cleanup** - Configurable retention policy
2. **Build manifest** - JSON metadata for each build (git commit, compiler version, flags)
3. **Size tracking** - Historical size analysis and regression detection
4. **Checksum verification** - SHA256 hashes for artifact integrity
5. **Remote artifact cache** - S3/Azure Blob storage integration for team sharing
