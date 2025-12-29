# Ternary Engine - AMD Linux Development Environment Setup Guide

**Doc-Type:** Environment Configuration · Version 1.0 · Updated 2025-12-15 · Author Ternary Engine Team

Configuration guide for setting up the Ternary Engine development environment on AMD Ryzen (Zen 3+) systems running Linux.

---

## Target Hardware Profile

| Component | Specification |
|:----------|:--------------|
| **Model** | Dell Inspiron 15 3535 |
| **CPU** | AMD Ryzen U-series Zen 3+ (Family 23 Model 160), 4C/8T |
| **Base Clock** | ~2.8 GHz (dynamic boost) |
| **GPU** | Integrated AMD RDNA2 iGPU |
| **RAM** | 8 GB DDR5 (7.38 GB usable) |
| **Storage** | NVMe SSD (PCIe) |
| **Architecture** | x86_64 (AMD64) |
| **BIOS** | Dell UEFI 1.23.0 (2025-06-02) |

---

## Hardware Compatibility Assessment

### CPU Feature Validation

**SIMD Support:**
- AVX2: **Fully Supported** (Zen 2+)
- AVX-512: **Not Available** (consumer Zen 3+)
- AES-NI: Supported
- PCLMULQDQ: Supported

**Verification command:**
```bash
cat /proc/cpuinfo | grep -E "avx2|avx|sse4"
# Or
lscpu | grep -i avx
```

**Expected output should include:** `avx2`

### Memory Considerations

| Factor | Value | Impact |
|:-------|:------|:-------|
| Total RAM | 8 GB | Moderate constraint |
| Usable | 7.38 GB | OS overhead ~620 MB |
| Swap recommended | 8-16 GB | For large benchmark runs |
| Max array size | ~500M elements | Before swap pressure |

**Recommendation:** Configure 8-16 GB swap for TritNet training and large benchmarks.

### AMD-Specific Optimizations Available

The Ternary Engine includes AMD-tuned optimizations in `src/core/simd/opt_dual_shuffle_xor.h`:

- **Zen 3+ shuffle optimization:** Uses dual xor+shuffle pattern
- **Expected speedup:** 1.5-1.7x vs Intel on shuffle-heavy operations
- **Port utilization:** Tuned for Zen's single shuffle port
- **Prefetch stride:** 512 bytes (optimal for Zen 2/3/4)

---

## Environment Setup

### 1. System Prerequisites

**Distribution:** Ubuntu 22.04 LTS / Fedora 38+ / Debian 12+ recommended

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y \
    build-essential \
    gcc-12 g++-12 \
    clang-15 \
    python3.12 python3.12-dev python3.12-venv \
    python3-pip \
    git \
    cmake \
    libomp-dev

# Fedora
sudo dnf install -y \
    gcc gcc-c++ \
    clang \
    python3.12 python3.12-devel \
    python3-pip \
    git \
    cmake \
    libomp-devel

# Set GCC 12 as default (if multiple versions)
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100
```

### 2. Compiler Verification

```bash
# Verify GCC version (9+ required, 12+ recommended)
gcc --version
# Expected: gcc (Ubuntu 12.x.x) 12.x.x or higher

# Verify C++17 support
echo '#include <optional>' | g++ -std=c++17 -x c++ - -c -o /dev/null && echo "C++17 OK"

# Verify AVX2 intrinsics compilation
echo '#include <immintrin.h>
int main() { __m256i a = _mm256_setzero_si256(); return 0; }' | \
g++ -mavx2 -x c++ - -o /dev/null && echo "AVX2 OK"

# Verify OpenMP
echo '#include <omp.h>
int main() { return omp_get_max_threads(); }' | \
g++ -fopenmp -x c++ - -o /dev/null && echo "OpenMP OK"
```

### 3. Python Environment Setup

```bash
# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install core dependencies
pip install numpy pybind11

# Install development dependencies
pip install pytest pytest-benchmark matplotlib

# Install TritNet dependencies (optional)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4. Clone Repository

```bash
git clone https://github.com/gesttaltt/ternary-engine.git
cd ternary-engine
```

---

## Build Configuration

### Linux Build Flags

The build system auto-detects Linux and uses appropriate flags. Key settings in `build/build.py`:

```python
# Linux/AMD optimized flags
LINUX_FLAGS = [
    '-O3',                    # Maximum optimization
    '-march=haswell',         # AVX2 baseline (safe for all AVX2 CPUs)
    '-mavx2',                 # Explicit AVX2 enable
    '-fopenmp',               # OpenMP threading
    '-std=c++17',             # C++17 standard
    '-flto',                  # Link-time optimization
    '-fPIC',                  # Position-independent code
    '-DNDEBUG',               # Disable assertions
]
```

**AMD-Specific Alternative (for maximum performance):**
```bash
# In build.py, change -march=haswell to:
'-march=znver3',              # Zen 3 optimizations
# Or for auto-detection:
'-march=native',              # Uses all available CPU features
```

### Standard Build

```bash
# Activate environment
source venv/bin/activate

# Build standard module
python build/build.py

# Expected output:
# [BUILD] Compiling src/engine/bindings_core_ops.cpp...
# [BUILD] Linking ternary_simd_engine.cpython-312-x86_64-linux-gnu.so...
# [BUILD] Success!
```

### Verify Build

```bash
# Quick test
python -c "import ternary_simd_engine as engine; print('Module loaded:', engine.__name__)"

# Run test suite
python run_tests.py

# Run benchmarks
python benchmarks/bench_phase0.py
```

---

## Resource-Constrained Configuration

### Memory Optimization (8GB System)

**1. Swap Configuration:**
```bash
# Check current swap
swapon --show

# Add 8GB swap file (if needed)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

**2. OpenMP Thread Limiting:**
```bash
# Limit threads to reduce memory pressure (4C/8T system)
export OMP_NUM_THREADS=4    # Half of logical cores
export OMP_STACKSIZE=4M     # Reduce per-thread stack
```

**3. Test Array Size Limits:**
```python
# In benchmarks, use smaller arrays for memory-constrained testing
# Modify bench_phase0.py:
SIZES = [1000, 10000, 100000, 500000]  # Instead of 1M+
```

### Thermal Management

Zen 3+ U-series mobile chips are thermally constrained. For sustained benchmarks:

```bash
# Monitor CPU temperature
watch -n 1 'sensors | grep -i temp'

# Limit frequency if thermal throttling observed
# (requires root)
echo 2000000 | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq
```

---

## Platform Validation Status

### Current Status: EXPERIMENTAL

| Component | Windows x64 | Linux AMD |
|:----------|:------------|:----------|
| Build System | Production | Experimental |
| AVX2 Operations | Validated | Needs Testing |
| OpenMP | Validated | Needs Testing |
| TritNet Training | Validated | Needs Testing |
| All 65 Tests | Passing | Pending |
| Performance Claims | 45.3 Gops/s | TBD |

### Validation Checklist

Run these after setup to validate the environment:

```bash
# 1. CPU capabilities
python tests/python/test_capabilities.py

# 2. Basic correctness
python tests/python/test_phase0.py

# 3. SIMD validation
python tests/python/test_simd_validation.py

# 4. OpenMP scaling
python tests/python/test_omp.py

# 5. Full test suite
python run_tests.py

# 6. Performance benchmark
python benchmarks/bench_phase0.py
```

**Record baseline results in:** `reports/linux-amd-validation/`

---

## Expected Performance

### Theoretical Analysis

Based on AMD Zen 3+ architecture:

| Metric | Estimate | Notes |
|:-------|:---------|:------|
| AVX2 throughput | 256-bit/cycle | Same as Intel |
| Shuffle ports | 1 (vs Intel 2) | Our code is optimized for this |
| L1 cache | 32KB/core | Standard |
| L2 cache | 512KB/core | Larger than Intel equivalent |
| Peak ops/s | ~15-25 Gops/s | Conservative estimate |

### Memory Bandwidth Impact

8GB DDR5 provides ~38.4 GB/s theoretical bandwidth:
- Single-channel penalty: ~50% vs dual-channel
- Expected effective: ~15-20 GB/s
- Impact on large arrays: Moderate (memory-bound at >1M elements)

### Scaling Expectations

```
# Expected scaling (4C/8T)
1 thread:  1.0x baseline
2 threads: ~1.8x
4 threads: ~3.2x
8 threads: ~4.0x (hyperthreading limited)
```

---

## Known Issues & Workarounds

### Issue 1: OpenMP Crashes

**Symptoms:** Segfault on multi-threaded operations
**Cause:** Stack size or memory alignment issues
**Workaround:**
```bash
export OMP_STACKSIZE=8M
ulimit -s unlimited
```

### Issue 2: Build Fails with Clang

**Symptoms:** Missing OpenMP headers
**Cause:** libomp not installed or not in path
**Fix:**
```bash
# Ubuntu
sudo apt install libomp-dev
# Fedora
sudo dnf install libomp-devel
```

### Issue 3: Module Import Fails

**Symptoms:** `ModuleNotFoundError: No module named 'ternary_simd_engine'`
**Cause:** Module not in Python path
**Fix:**
```bash
# Option 1: Set PYTHONPATH
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Option 2: Install in development mode
pip install -e .
```

### Issue 4: Low Performance Numbers

**Symptoms:** Benchmark shows <10 Gops/s
**Possible causes:**
1. CPU thermal throttling (check `sensors`)
2. Background processes consuming resources
3. Memory swap pressure (check `free -h`)
4. Non-optimal compiler flags

**Diagnostics:**
```bash
# Check CPU frequency
cat /proc/cpuinfo | grep MHz

# Check memory pressure
free -h

# Check for throttling
dmesg | grep -i thermal
```

---

## Development Workflow

### Daily Development

```bash
# 1. Activate environment
cd ternary-engine
source venv/bin/activate

# 2. Pull latest changes
git pull origin main

# 3. Rebuild if C++ changed
python build/build.py

# 4. Run tests
python run_tests.py

# 5. Run targeted tests
python -m pytest tests/python/test_phase0.py -v
```

### Performance Testing

```bash
# Quick performance check
python benchmarks/bench_phase0.py --quick

# Full benchmark suite
python benchmarks/bench_phase0.py

# Competitive benchmarks
python benchmarks/bench_competitive.py --phase 1
```

### TritNet Development

```bash
# Generate truth tables (if not present)
python models/tritnet/run_tritnet.py generate

# Train models
python models/tritnet/run_tritnet.py train

# Note: CPU-only training on this hardware
# Expected training time: ~5-10 min per operation
```

---

## Recommended IDE Setup

### VS Code

```bash
# Install VS Code
sudo snap install code --classic

# Install extensions
code --install-extension ms-python.python
code --install-extension ms-vscode.cpptools
code --install-extension ms-vscode.cmake-tools
```

**settings.json:**
```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "C_Cpp.default.cppStandard": "c++17",
    "C_Cpp.default.compilerPath": "/usr/bin/g++-12",
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.so": false
    }
}
```

### PyCharm

```bash
# Install PyCharm Community
sudo snap install pycharm-community --classic
```

Configure interpreter to use `venv/bin/python`.

---

## Git Configuration

```bash
# Configure git
git config --local user.name "Your Name"
git config --local user.email "your.email@example.com"

# Enable credential caching (1 hour)
git config --global credential.helper 'cache --timeout=3600'
```

---

## Quick Start Checklist

- [ ] Verify AVX2 support: `lscpu | grep avx2`
- [ ] Install GCC 12+ and dependencies
- [ ] Create Python 3.12 virtual environment
- [ ] Install pip dependencies
- [ ] Clone repository
- [ ] Run `python build/build.py`
- [ ] Run `python run_tests.py`
- [ ] Run `python benchmarks/bench_phase0.py`
- [ ] Record baseline performance in `reports/`

---

## Support & Troubleshooting

**Documentation:**
- Build system: `docs/build-system/README.md`
- Testing: `TESTING.md`
- Architecture: `docs/architecture/`

**Issues:**
- GitHub: https://github.com/gesttaltt/ternary-engine/issues
- Tag Linux-specific issues with `[LINUX]` prefix

**Performance Reports:**
- Save to: `reports/linux-amd-YYYYMMDD/`
- Include: CPU model, kernel version, GCC version, test results

---

## Changelog

| Date | Version | Description |
|:-----|:--------|:------------|
| 2025-12-15 | v1.0.0 | Initial AMD Linux setup guide for Dell Inspiron 15 3535 |

---

**Remember:**
- Linux support is EXPERIMENTAL until validation complete
- Document any issues found during testing
- Performance claims are for Windows x64 only until Linux validation
- Memory-constrained systems need swap configured
