# Clang Installation Guide for PGO Builds

**Doc-Type:** Installation Guide · Version 1.0 · Date 2025-11-23 · Ternary Engine

---

## Why Clang for PGO?

**problem_with_msvc** - MSVC PGO requires DLL unload to write .pgc files, Python extensions don't trigger this reliably
**clang_advantage** - Writes .profraw files immediately during execution, no DLL lifecycle dependency
**result** - Clang PGO works perfectly with Python extensions, MSVC PGO has fundamental limitations

See [PGO_LIMITATIONS.md](PGO_LIMITATIONS.md) for technical details on MSVC issues.

---

## Windows Installation

### Option 1: Official LLVM Release (Recommended)

**download** - https://releases.llvm.org/download.html

**steps**:
1. Download latest LLVM for Windows (e.g., LLVM-18.1.0-win64.exe)
2. Run installer
3. **IMPORTANT**: Check "Add LLVM to the system PATH for all users"
4. Complete installation
5. Restart terminal/PowerShell

**verify**:
```cmd
clang-cl --version
llvm-profdata --version
```

**expected_output**:
```
clang version 18.1.0
Target: x86_64-pc-windows-msvc
```

### Option 2: Visual Studio Integration

**download** - Visual Studio Installer

**steps**:
1. Open Visual Studio Installer
2. Modify existing installation
3. Under "Individual components", search "Clang"
4. Install "C++ Clang Compiler for Windows"
5. Restart VS and terminal

**location** - Usually in `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\bin`

### Option 3: Chocolatey

```cmd
choco install llvm
```

**verify**:
```cmd
refreshenv
clang-cl --version
```

---

## Linux Installation

### Ubuntu/Debian

```bash
sudo apt update
sudo apt install clang llvm
```

**verify**:
```bash
clang --version
llvm-profdata --version
```

### Fedora/RHEL

```bash
sudo dnf install clang llvm
```

### Arch Linux

```bash
sudo pacman -S clang llvm
```

---

## macOS Installation

### Homebrew (Recommended)

```bash
brew install llvm
```

**add_to_path** - Add to ~/.zshrc or ~/.bash_profile:
```bash
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
```

**verify**:
```bash
clang --version
llvm-profdata --version
```

### Xcode Command Line Tools

```bash
xcode-select --install
```

**note** - macOS Xcode clang includes llvm-profdata

---

## Verification Test

After installation, test Clang PGO works:

```bash
# Windows
clang-cl /O2 /std:c++17 test.cpp /Fe:test.exe

# Linux/macOS
clang++ -O3 -std=c++17 test.cpp -o test
```

**test_file** (test.cpp):
```cpp
#include <iostream>
int main() {
    std::cout << "Clang works!" << std::endl;
    return 0;
}
```

---

## Using Unified PGO Build

Once Clang is installed:

```bash
# Auto-detect (prefers Clang if available)
python build/build_pgo_unified.py

# Force Clang
python build/build_pgo_unified.py --clang

# Force MSVC fallback (not recommended)
python build/build_pgo_unified.py --msvc
```

---

## Clang PGO Workflow

### Phase 1: Instrumentation Build

```bash
python build/build_pgo_unified.py --clang --clean
```

**what_happens**:
- Builds with `-fprofile-generate=pgo_data/profiles`
- Instruments code to collect runtime data
- Creates `ternary_simd_engine` with profiling hooks

### Phase 2: Profile Collection

**automatic** - Script runs `benchmarks/bench_phase0.py --quick`

**manual_alternative**:
```bash
python benchmarks/bench_phase0.py
python benchmarks/bench_fusion.py
```

**output** - `.profraw` files in `pgo_data/profiles/`

### Phase 3: Profile Merging

**automatic** - Script runs `llvm-profdata merge`

**manual**:
```bash
llvm-profdata merge -output=pgo_data/merged.profdata pgo_data/profiles/*.profraw
```

### Phase 4: Optimized Build

**automatic** - Script rebuilds with `-fprofile-use=pgo_data/merged.profdata`

**result** - PGO-optimized `ternary_simd_engine.pyd`

---

## Performance Expectations

### Clang PGO Benefits

**typical_gains** - 5-15% performance improvement
**best_case** - 20-30% for branch-heavy code
**measured** - Test with `python benchmarks/bench_phase0.py`

### Comparison with MSVC

| Aspect | Clang PGO | MSVC PGO |
|--------|-----------|----------|
| Python Extension Support | ✅ Works | ❌ Broken (DLL issue) |
| Profile Collection | ✅ Immediate | ❌ Requires DLL unload |
| Cross-Platform | ✅ Yes | ❌ Windows only |
| Ease of Use | ✅ Simple | ⚠️  Complex |
| Performance Gain | 5-15% | N/A (no profile data) |

---

## Troubleshooting

### Issue: "clang-cl: command not found"

**solution**:
1. Verify LLVM installed: Check `C:\Program Files\LLVM\bin`
2. Add to PATH manually:
   - Windows: System Properties > Environment Variables > Path
   - Add `C:\Program Files\LLVM\bin`
3. Restart terminal

### Issue: "llvm-profdata: command not found"

**solution** - llvm-profdata is part of LLVM, reinstall LLVM package

### Issue: No .profraw files generated

**possible_causes**:
- Profile directory doesn't exist
- Insufficient permissions
- Benchmark didn't run

**fix**:
```bash
# Check profile directory
ls pgo_data/profiles/

# Run benchmark manually
python benchmarks/bench_phase0.py --quick

# Check for .profraw files
find pgo_data -name "*.profraw"
```

### Issue: Profile merge fails

**error**: `Malformed instrumentation profile data`

**solution**:
- Delete `pgo_data/` directory
- Rebuild from Phase 1
- Ensure same Clang version for all phases

---

## Advanced Usage

### Custom Profiling Workload

```bash
# Phase 1: Build instrumented
python build/build_pgo_unified.py --clang

# Phase 2: Run custom workload (instead of automatic)
python my_custom_benchmark.py
python another_workload.py

# Phase 3 & 4: Merge and rebuild
llvm-profdata merge -output=pgo_data/merged.profdata pgo_data/profiles/*.profraw
CPPFLAGS="-fprofile-use=pgo_data/merged.profdata" python build/build.py
```

### Analyze Profile Data

```bash
# View profile summary
llvm-profdata show --all-functions pgo_data/merged.profdata

# Check coverage
llvm-profdata show --detailed-summary pgo_data/merged.profdata
```

---

## Comparison with Standard Build

After PGO build, compare performance:

```bash
# Backup PGO build
cp ternary_simd_engine.cp312-win_amd64.pyd ternary_simd_engine_pgo.pyd

# Standard build
python build/build.py

# Benchmark standard
python benchmarks/bench_phase0.py > results_standard.txt

# Restore PGO build
cp ternary_simd_engine_pgo.pyd ternary_simd_engine.cp312-win_amd64.pyd

# Benchmark PGO
python benchmarks/bench_phase0.py > results_pgo.txt

# Compare
diff results_standard.txt results_pgo.txt
```

---

## References

**LLVM PGO:** https://clang.llvm.org/docs/UsersManual.html#profile-guided-optimization
**llvm-profdata:** https://llvm.org/docs/CommandGuide/llvm-profdata.html
**Clang Downloads:** https://releases.llvm.org/download.html
**Clang on Windows:** https://clang.llvm.org/docs/MSVCCompatibility.html

---

**Version:** 1.0 · **Date:** 2025-11-23 · **Status:** Production-Ready · **Recommended:** Clang PGO over MSVC
