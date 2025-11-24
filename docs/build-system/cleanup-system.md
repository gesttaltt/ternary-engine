# Build Cleanup System

## Overview

The Ternary Engine build cleanup system provides comprehensive, automated cleanup of all build artifacts, temporary files, cached data, and benchmark results. This ensures fresh builds, accurate benchmarks, and prevents disk space accumulation.

**Location:** `build/clean_all.py`

**Key Features:**
- Removes all build artifacts across all build types
- Cleans setuptools temporary directories (prevents MAX_PATH issues on Windows)
- Consolidates PGO data from multiple historical locations
- Removes compiled modules from project root
- Cleans old benchmark results with configurable retention
- Supports dry-run mode for safe preview
- Optional deep cleanup of external temp directories

## Quick Start

### Basic Cleanup

```bash
# Clean everything (most common use case)
python build/clean_all.py

# Preview what would be deleted without actually deleting
python build/clean_all.py --dry-run

# Keep the latest build for each build type
python build/clean_all.py --keep-latest

# Keep the 5 most recent benchmark result files
python build/clean_all.py --keep-results 5

# Include external temp directories (Windows)
python build/clean_all.py --deep
```

### Integration with Benchmarks

The cleanup system is integrated into the benchmarking workflow:

```bash
# Clean before running benchmarks
python benchmarks/run_all_benchmarks.py --clean

# Equivalent to:
python build/clean_all.py
python benchmarks/run_all_benchmarks.py
```

## What Gets Cleaned

### 1. Build Artifacts (`build/artifacts/`)

The cleanup system removes all timestamped build directories while optionally preserving the latest build.

**Cleaned directories:**
- `build/artifacts/standard/YYYYMMDD_HHMMSS/` - Standard optimized builds
- `build/artifacts/pgo/instrumented/YYYYMMDD_HHMMSS/` - PGO instrumentation builds
- `build/artifacts/pgo/optimized/YYYYMMDD_HHMMSS/` - PGO optimized builds
- `build/artifacts/reference/YYYYMMDD_HHMMSS/` - Reference baseline builds
- `build/artifacts/dense243/YYYYMMDD_HHMMSS/` - Dense243 module builds
- `build/artifacts/*/latest/` - Latest build symlinks (unless --keep-latest)

**Example output:**
```
Cleaning Build Artifacts
======================================================================

  Build type: standard
  [DELETED] standard/20251122_213416 (158.0 KB)
  [DELETED] standard/20251122_214604 (158.0 KB)
  [DELETED] standard/20251122_215004 (158.0 KB)
  [DELETED] standard/latest (162.5 KB)
```

### 2. Setuptools Temporary Directories

Deep nested temporary directories created by setuptools during builds. These can cause MAX_PATH issues on Windows and accumulate unnecessary disk space.

**Cleaned directories:**
- `build/temp.win-amd64-cpython-312/` - Windows build temps
- `build/temp.linux-x86_64-cpython-312/` - Linux build temps
- `build/lib.win-amd64-cpython-312/` - Windows library output
- `build/lib.linux-x86_64-cpython-312/` - Linux library output

**Contains:**
- `.obj` files (compiled object files)
- `.lib` files (static libraries)
- `.exp` files (export files)
- Intermediate build artifacts with very deep paths

**Example:**
```
Cleaning Setuptools Directories
======================================================================
  [DELETED] build/temp.win-amd64-cpython-312 (8.2 MB)
  [DELETED] build/lib.win-amd64-cpython-312 (489.0 KB)
```

### 3. PGO Profile Data

Profile-Guided Optimization data from all historical locations. The cleanup system consolidates and removes PGO data to ensure fresh profiling runs.

**Cleaned locations:**

1. **Build-integrated location** (current standard):
   - `build/artifacts/pgo_data/` - Unified PGO data location

2. **Legacy location** (build_pgo.py):
   - `build/artifacts/pgo/pgo_data/` - Old MSVC PGO location

3. **Root location** (old build_pgo_unified.py):
   - `pgo_data/` - Deprecated project root location

**Contains:**
- `.profraw` files - Clang raw profile data
- `.profdata` files - Clang merged profile database
- `.pgc` files - MSVC profile counter files
- `.pgd` files - MSVC profile database

**Example:**
```
Cleaning PGO Data
======================================================================
  [DELETED] build/artifacts/pgo_data (1.2 MB)
```

### 4. Root Compiled Modules

Python extension modules built with `--inplace` flag, which places them in the project root for easy import during development.

**Cleaned files:**
- `*.pyd` - Windows Python extension modules
- `*.so` - Linux/macOS Python extension modules

**Modules cleaned:**
- `ternary_simd_engine.cp312-win_amd64.pyd`
- `ternary_dense243_module.cp312-win_amd64.pyd`
- `ternary_fusion_engine.cp312-win_amd64.pyd`
- `reference_cpp.cp312-win_amd64.pyd`

**Example:**
```
Cleaning Root Module Files
======================================================================
  [DELETED] ternary_simd_engine.cp312-win_amd64.pyd (162.5 KB)
  [DELETED] ternary_dense243_module.cp312-win_amd64.pyd (169.0 KB)
  [DELETED] reference_cpp.cp312-win_amd64.pyd (128.0 KB)
```

### 5. Benchmark Results

Old benchmark result files with configurable retention policy. By default, all results are removed, but you can keep the N most recent.

**Cleaned files:**
- `benchmarks/results/bench_results_*.json` - Detailed benchmark data
- `benchmarks/results/bench_results_*.csv` - CSV exports
- Empty subdirectories in `benchmarks/results/`

**Retention:**
```bash
# Remove all benchmark results
python build/clean_all.py --keep-results 0  # default

# Keep last 5 result files (both JSON and CSV)
python build/clean_all.py --keep-results 5

# Keep last 10 result files
python build/clean_all.py --keep-results 10
```

**Example:**
```
Cleaning Benchmark Results (keeping last 2)
======================================================================
  [DELETED] benchmarks/results/bench_results_20251122_215148.json (13.8 KB)
  [DELETED] benchmarks/results/bench_results_20251122_214950.json (7.8 KB)
  [DELETED] benchmarks/results/bench_results_20251122_215148.csv (1.4 KB)
  [DELETED] benchmarks/results/bench_results_20251122_214950.csv (823.0 B)

  [KEPT] 2 most recent result file(s)
```

### 6. External Temp Directories (--deep)

Windows-specific external temporary directories used to avoid MAX_PATH issues. Only cleaned when `--deep` flag is used.

**Cleaned locations:**
- `C:/Temp/ternary_build/` - Standard build temps
- `C:/Temp/ternary_pgo_inst/` - PGO instrumentation temps
- `C:/Temp/ternary_pgo_opt/` - PGO optimization temps

**Example:**
```bash
python build/clean_all.py --deep
```

```
Cleaning External Temp Directories
======================================================================
  [DELETED] C:/Temp/ternary_build (4.5 MB)
  [DELETED] C:/Temp/ternary_pgo_inst (3.2 MB)
```

## Usage Examples

### Development Workflow

```bash
# Start of day: clean everything for fresh builds
python build/clean_all.py

# Build standard version
python build/build.py

# Test
python -c "import ternary_simd_engine; print('OK')"
```

### Before Benchmarking

```bash
# Clean everything including old results
python build/clean_all.py

# Build fresh
python build/build.py

# Run benchmarks
python benchmarks/bench_phase0.py
```

### CI/CD Pipeline

```bash
# Always start with clean slate
python build/clean_all.py --quiet

# Build
python build/build.py

# Test
python -m pytest tests/

# Benchmark
python benchmarks/bench_phase0.py --quick
```

### Disk Space Management

```bash
# Preview what would be deleted
python build/clean_all.py --dry-run

# Clean but keep latest builds (saves rebuild time)
python build/clean_all.py --keep-latest

# Keep recent benchmark data for comparison
python build/clean_all.py --keep-results 10
```

### Complete System Reset

```bash
# Nuclear option: clean everything including external temps
python build/clean_all.py --deep

# Verify clean state
ls *.pyd 2>/dev/null || echo "Root clean"
ls build/artifacts/
```

## Command-Line Options

### `--keep-latest`

Preserves the most recent timestamped build for each build type, removing all older builds.

**Usage:**
```bash
python build/clean_all.py --keep-latest
```

**Effect:**
- Keeps: `build/artifacts/standard/20251123_015113/`
- Removes: All earlier `build/artifacts/standard/YYYYMMDD_HHMMSS/` directories
- Preserves: `build/artifacts/standard/latest/` directory

**Use case:** You want to clean old builds but avoid rebuilding the latest version.

### `--keep-results N`

Retains the N most recent benchmark result files (both JSON and CSV), removing all older results.

**Usage:**
```bash
python build/clean_all.py --keep-results 5
```

**Effect:**
- Keeps: 5 most recent `.json` files
- Keeps: 5 most recent `.csv` files
- Removes: All older result files
- Removes: Empty subdirectories

**Use case:** Maintain historical benchmark data for trend analysis.

### `--deep`

Includes external temporary directories (Windows-specific) in the cleanup.

**Usage:**
```bash
python build/clean_all.py --deep
```

**Effect:**
- Cleans: `C:/Temp/ternary_build/`
- Cleans: `C:/Temp/ternary_pgo_inst/`
- Cleans: `C:/Temp/ternary_pgo_opt/`

**Use case:** Complete cleanup when experiencing disk space issues or MAX_PATH errors.

### `--dry-run`

Previews what would be deleted without actually deleting anything. Shows size of each item.

**Usage:**
```bash
python build/clean_all.py --dry-run
```

**Output:**
```
[DRY RUN] Would delete: standard/20251122_213416 (158.0 KB)
[DRY RUN] Would delete: standard/20251122_214604 (158.0 KB)
...
======================================================================
  CLEANUP SUMMARY
======================================================================

  [DRY RUN] Would delete 41 items
  [DRY RUN] Would free ~53.2 MB
```

**Use case:** Verify what will be cleaned before executing.

### `--quiet`

Suppresses detailed output, showing only errors and final summary.

**Usage:**
```bash
python build/clean_all.py --quiet
```

**Use case:** Scripted/automated environments where minimal output is desired.

### Combining Options

Options can be combined for fine-grained control:

```bash
# Preview deep cleanup while keeping latest builds
python build/clean_all.py --dry-run --deep --keep-latest

# Quiet cleanup keeping last 3 results
python build/clean_all.py --quiet --keep-results 3

# Complete cleanup except latest builds and 5 recent results
python build/clean_all.py --keep-latest --keep-results 5
```

## Integration Points

### Benchmark Orchestrator

The cleanup system is integrated into `benchmarks/run_all_benchmarks.py`:

```bash
# Automatic cleanup before benchmarking
python benchmarks/run_all_benchmarks.py --clean

# Equivalent to:
python build/clean_all.py
python benchmarks/run_all_benchmarks.py
```

**Implementation:**
```python
# In run_all_benchmarks.py
CLEAN_SCRIPT = PROJECT_ROOT / "scripts" / "build" / "clean_all.py"

def clean_builds():
    """Clean all build artifacts using comprehensive cleanup utility"""
    cmd = [sys.executable, str(CLEAN_SCRIPT)]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
```

### Build Scripts

Build scripts automatically use `--inplace` which places modules in the project root. The cleanup system removes these after benchmarking or when starting fresh builds.

**Standard build workflow:**
```python
# build.py builds to project root
# ternary_simd_engine.cp312-win_amd64.pyd created

# cleanup system removes it
# python build/clean_all.py
```

### PGO Workflow

The cleanup system consolidates PGO data from historical locations:

```python
# Old location (deprecated): pgo_data/
# Old location (build_pgo.py): build/artifacts/pgo/pgo_data/
# Current location: build/artifacts/pgo_data/

# Cleanup removes all three to ensure fresh profiling
python build/clean_all.py
python build/build_pgo_unified.py
```

## Disk Space Impact

### Typical Cleanup Results

Based on a project with multiple builds over several days:

| Category | Items | Size | Notes |
|----------|-------|------|-------|
| Build artifacts | 15-25 | 10-30 MB | Depends on build frequency |
| Setuptools temps | 2-4 | 5-15 MB | Deep nested paths |
| Root modules | 3-5 | 500-800 KB | Active development artifacts |
| Benchmark results | 10-20 | 100-300 KB | Accumulates over time |
| PGO data | 1-3 | 0-2 MB | If PGO builds were used |
| **Total** | **30-50** | **15-50 MB** | Per week of active development |

### Storage Growth Prevention

Without regular cleanup:
- **Weekly growth:** 15-50 MB
- **Monthly growth:** 60-200 MB
- **Yearly growth:** 700 MB - 2.4 GB

With automated cleanup (via `--clean` flag):
- **Steady state:** ~5-10 MB
- **Growth rate:** Near zero

## Troubleshooting

### MAX_PATH Errors (Windows)

**Symptom:**
```
OSError: [WinError 206] The filename or extension is too long
```

**Solution:**
```bash
# Clean setuptools deep temp directories
python build/clean_all.py

# If that fails, use deep cleanup
python build/clean_all.py --deep
```

**Prevention:**
- Run cleanup regularly
- Use `--clean` flag in benchmark orchestrator
- Build scripts already use short paths (C:/Temp/*) to minimize this

### Permission Errors

**Symptom:**
```
[ERROR] Failed to delete standard/20251122_213416: Permission denied
```

**Possible causes:**
1. Module still imported in Python session
2. File locked by antivirus
3. File in use by another process

**Solutions:**
```bash
# Close all Python sessions
exit()  # or close all terminals

# Kill Python processes (Windows)
taskkill /F /IM python.exe

# Kill Python processes (Linux)
pkill python

# Retry cleanup
python build/clean_all.py
```

### Dry-Run Shows 0 Items

**Symptom:**
```
[DRY RUN] Would delete 0 items
[DRY RUN] Would free ~0.0 B
```

**Meaning:** Nothing to clean - project is already in clean state.

**Verification:**
```bash
# Check for any artifacts
ls *.pyd 2>/dev/null || echo "No .pyd files"
find build/artifacts -name "*.pyd" | wc -l
ls benchmarks/results/*.json 2>/dev/null | wc -l
```

### Cleanup Too Aggressive

**Symptom:** Want to keep some artifacts for comparison

**Solution:** Use retention flags:
```bash
# Keep latest builds
python build/clean_all.py --keep-latest

# Keep recent benchmark data
python build/clean_all.py --keep-results 10

# Both
python build/clean_all.py --keep-latest --keep-results 10
```

## Best Practices

### Regular Cleanup Schedule

**For active development:**
```bash
# Start of each work session
python build/clean_all.py --keep-latest --keep-results 5
```

**For benchmarking:**
```bash
# Always use clean flag
python benchmarks/run_all_benchmarks.py --clean
```

**For CI/CD:**
```bash
# Always start fresh
python build/clean_all.py --quiet
```

### Disk Space Monitoring

```bash
# Check total build artifacts size
du -sh build/

# Check what would be cleaned
python build/clean_all.py --dry-run | tail -5

# Clean if over threshold
SIZE=$(du -sm build/ | cut -f1)
if [ $SIZE -gt 100 ]; then
    python build/clean_all.py --keep-latest
fi
```

### Selective Retention

Different retention strategies for different scenarios:

```bash
# Development: Keep latest for quick iteration
python build/clean_all.py --keep-latest

# Performance testing: Keep recent results for comparison
python build/clean_all.py --keep-results 5

# Release preparation: Complete cleanup
python build/clean_all.py --deep

# Continuous benchmarking: Balance between history and space
python build/clean_all.py --keep-latest --keep-results 10
```

### Automation Examples

**Git pre-commit hook:**
```bash
#!/bin/bash
# .git/hooks/pre-commit
python build/clean_all.py --quiet --keep-latest
```

**Scheduled task (Windows):**
```powershell
# Daily cleanup at midnight
schtasks /create /tn "TernaryCleanup" /tr "python C:\path\to\ternary-engine\build\clean_all.py --keep-latest" /sc daily /st 00:00
```

**Cron job (Linux):**
```bash
# Daily cleanup at midnight
0 0 * * * cd /path/to/ternary-engine && python build/clean_all.py --keep-latest
```

## Implementation Details

### Architecture

The cleanup system is implemented as a single Python script (`clean_all.py`) with a class-based design:

```python
class CleanupManager:
    def __init__(self, dry_run=False, verbose=True):
        self.dry_run = dry_run
        self.verbose = verbose
        self.deleted_count = 0
        self.deleted_size = 0

    def clean_build_artifacts(self, keep_latest=False): ...
    def clean_setuptools_dirs(): ...
    def clean_pgo_data(): ...
    def clean_root_modules(): ...
    def clean_benchmark_results(keep_count=0): ...
    def clean_external_temp_dirs(): ...
```

### Safety Features

1. **Dry-run mode:** Preview before deletion
2. **Counting and sizing:** Track what's being deleted
3. **Error handling:** Continue on errors, report failures
4. **Verbose output:** Clear feedback on every operation
5. **Path validation:** Only cleans known safe locations
6. **No wildcards on root:** Never uses `rm -rf /` patterns

### Performance

**Typical execution time:**
- Small cleanup (few artifacts): 0.5-1 seconds
- Medium cleanup (20-30 items): 1-2 seconds
- Large cleanup (50+ items): 2-5 seconds
- Deep cleanup (including external temps): 5-10 seconds

**I/O operations:**
- File counting and sizing: Read-only
- Deletion: Sequential (not parallelized for safety)
- Directory traversal: Uses Python pathlib (cross-platform)

## Related Documentation

- [Build System Overview](./README.md)
- [Artifact Organization](./artifact-organization.md)
- [PGO Build System](./setup-pgo.md)
- [Standard Build](./setup-standard.md)
- [Benchmark Suite](../../benchmarks/README.md)

## FAQ

### Q: Will cleanup delete my source code?

**A:** No. The cleanup system only targets build artifacts, not source code. Locations cleaned:
- `build/artifacts/` (build outputs only)
- `build/temp.*`, `build/lib.*` (build temps)
- `*.pyd`, `*.so` in project root (compiled modules)
- `benchmarks/results/` (benchmark data)
- `pgo_data/`, `build/artifacts/pgo_data/` (profiling data)

Source directories (`src/engine/`, `src/core/`, etc.) are never touched.

---

### Q: Can I run cleanup while a build is in progress?

**A:** Not recommended. This could cause:
- Permission errors
- Build failures
- Corrupted artifacts

Best practice: Wait for builds to complete before cleaning.

---

### Q: How often should I run cleanup?

**A:** Depends on your workflow:
- **Active development:** Daily or before each benchmarking session
- **Automated CI/CD:** Every build
- **Occasional development:** Weekly or when disk space low
- **Release preparation:** Before tagging/releasing

---

### Q: What's the difference between `--keep-latest` and not using it?

**A:**

Without `--keep-latest`:
```bash
python build/clean_all.py
# Removes: all timestamped builds + latest/ directory
# Result: build/artifacts/* completely empty
```

With `--keep-latest`:
```bash
python build/clean_all.py --keep-latest
# Removes: all but most recent timestamped build
# Keeps: newest timestamped build + latest/ directory
# Result: Can import module without rebuilding
```

---

### Q: Does cleanup affect different Python versions differently?

**A:** Yes. Cleanup is version-agnostic:
- Cleans all `.pyd`/`.so` files regardless of Python version
- If you have modules for multiple Python versions (e.g., cp310, cp312), all are removed
- After cleanup, rebuild for your current Python version

---

### Q: Can I customize which directories get cleaned?

**A:** Currently no built-in customization. Options:
1. **Use flags:** `--keep-latest`, `--keep-results N` for retention
2. **Modify script:** Edit `clean_all.py` to add/remove cleanup targets
3. **Manual cleanup:** Use standard shell commands for specific targets

If you need custom cleanup, consider contributing a PR with additional flags.

---

### Q: What happens if cleanup fails partway through?

**A:** The cleanup system continues on errors:
- Failed deletions are reported
- Remaining items still cleaned
- Exit code indicates partial failure
- Summary shows what was actually deleted

Example:
```
[DELETED] standard/20251122_213416 (158.0 KB)
[ERROR] Failed to delete standard/20251122_214604: Permission denied
[DELETED] standard/20251122_215004 (158.0 KB)
...
Summary: Deleted 15 items (some failures)
```

Re-run cleanup after resolving errors.

---

## Support

For cleanup system issues:

1. Check this documentation
2. Try `--dry-run` to diagnose
3. Verify permissions
4. Search existing issues
5. Open new issue with:
   - OS and Python version
   - Command run
   - Full output
   - Error message

---

**Last updated:** 2025-11-23
**Documentation version:** 1.0.0
**Script location:** `build/clean_all.py`
