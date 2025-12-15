# Environment Setup Documentation

**Doc-Type:** Index · Version 1.0 · Updated 2025-12-15

Documentation for setting up Ternary Engine development environments on various platforms.

---

## Available Guides

| Platform | Hardware | Status | Guide |
|:---------|:---------|:-------|:------|
| Windows x64 | Intel/AMD | Production | See main `README.md` |
| Linux AMD | Zen 3+ | Experimental | [AMD_LINUX_SETUP_GUIDE.md](AMD_LINUX_SETUP_GUIDE.md) |
| Linux Intel | Haswell+ | Planned | - |
| macOS Intel | Haswell+ | Planned | - |
| macOS ARM64 | Apple Silicon | Planned | - |

---

## Platform Support Matrix

| Feature | Windows x64 | Linux AMD | Linux Intel | macOS |
|:--------|:------------|:----------|:------------|:------|
| AVX2 SIMD | Validated | Experimental | Experimental | Experimental |
| OpenMP | Validated | Experimental | Experimental | Not Supported |
| PGO Build | Validated | Untested | Untested | Untested |
| All 65 Tests | Passing | TBD | TBD | TBD |
| Peak 45 Gops/s | Validated | TBD | TBD | TBD |

---

## Quick Start (Any Platform)

```bash
# 1. Verify AVX2 support
# Windows (PowerShell):
#   Get-WmiObject Win32_Processor | Select-Object -ExpandProperty Caption
# Linux:
#   lscpu | grep avx2

# 2. Clone repository
git clone https://github.com/gesttaltt/ternary-engine.git
cd ternary-engine

# 3. Create virtual environment
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

# 4. Install dependencies
pip install numpy pybind11

# 5. Build
python build/build.py

# 6. Test
python run_tests.py
```

---

## Contributing Platform Guides

When validating on a new platform, please document:

1. **Hardware specifications** (CPU model, RAM, storage type)
2. **OS/kernel version**
3. **Compiler versions** (GCC/Clang/MSVC)
4. **Build modifications** (if any)
5. **Test results** (all 65 tests passing?)
6. **Performance numbers** (ops/s, compared to Windows baseline)
7. **Known issues** and workarounds

Submit results via GitHub issues or pull requests.

---

## Related Documentation

- Main README: `../README.md`
- Build system: `../build-system/README.md`
- Testing guide: `../../TESTING.md`
- Architecture: `../architecture/`
