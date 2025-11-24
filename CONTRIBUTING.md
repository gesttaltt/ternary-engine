# Contributing to Ternary Engine

Thank you for your interest in contributing to the Ternary Engine library! This document provides guidelines for contributing code, documentation, and other improvements.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Performance Guidelines](#performance-guidelines)
- [Adding New Operations](#adding-new-operations)

## Code of Conduct

This project follows the principle of **technical accuracy and professional objectivity**:

- Focus on facts and problem-solving
- Provide constructive, objective feedback
- Respect diverse approaches and perspectives
- Prioritize code quality and maintainability

## Getting Started

### Prerequisites

1. **Development Environment**:
   - C++17-capable compiler (GCC 7+, Clang 5+, MSVC 2017+)
   - Python 3.7+
   - AVX2-capable CPU (Intel Haswell 2013+ or AMD Excavator 2015+)

2. **Dependencies**:
   ```bash
   pip install pybind11 numpy
   ```

3. **Build the Library**:
   ```bash
   python build.py
   ```

4. **Run Tests**:
   ```bash
   python tests/test_phase0.py
   python tests/test_omp.py
   ```

### Repository Structure

```
ternary-engine/
├── src/                # Centralized source code
│   ├── core/           # Production kernel (algebra, SIMD, FFI)
│   └── engine/         # Python bindings and libraries
├── docs/               # Documentation (organized by category)
├── tests/              # Test suite
├── benchmarks/         # Performance benchmarks
├── build/              # Build system and artifacts
├── models/             # LLM/Neural Network integration (TritNet, etc.)
└── local-reports/      # Development notes (not in git)
```

## Development Workflow

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/ternary-engine.git
cd ternary-engine
```

### 2. Create a Branch

```bash
git checkout -b feature/my-new-feature
# or
git checkout -b bugfix/issue-123
```

### 3. Make Changes

- **Source code**: Edit `.h` or `.cpp` files at root level
- **Documentation**: Update files in `docs/` directory
- **Tests**: Add or modify tests in `tests/`
- **Build scripts**: Modify files in `build/scripts/`

### 4. Test Your Changes

```bash
# Build
python build.py

# Test correctness
python tests/test_phase0.py

# Test performance
python benchmarks/bench_phase0.py
```

### 5. Commit

```bash
git add .
git commit -m "Brief description of changes

Detailed explanation if needed:
- What was changed
- Why it was changed
- Performance impact (if applicable)"
```

### 6. Push and Create PR

```bash
git push origin feature/my-new-feature
```

Then create a Pull Request on GitHub.

## Coding Standards

### C++ Style

#### File Organization

- **Source files**: Keep all `.h` and `.cpp` files at root level (no nesting)
- **Header guards**: Use `#ifndef HEADER_NAME_H` format
- **Includes**: Group system headers, then library headers, then local headers

#### Naming Conventions

```cpp
// Types: CamelCase
struct TernaryVector { ... };
enum class SIMDLevel { ... };

// Functions: snake_case
trit tadd(trit a, trit b);
__m256i tadd_simd(const uint8_t* a, const uint8_t* b);

// Constants: UPPER_CASE
constexpr int OMP_THRESHOLD = 100000;
constexpr int PREFETCH_DIST = 512;

// Variables: snake_case
size_t array_size = 1000;
__m256i vec_result;
```

#### Optimization Tags

Use `OPT-XXX` tags to track optimizations:

```cpp
// OPT-PHASE3-01: Adaptive OMP threshold
static const ssize_t OMP_THRESHOLD = 32768 * std::thread::hardware_concurrency();

// OPT-AUTO-LUT: Constexpr compile-time LUT generation
constexpr auto TADD_LUT = make_binary_lut(tadd_logic);
```

#### Comments

```cpp
// Good: Explains WHY
// Use adaptive threshold to scale with CPU core count
static const ssize_t OMP_THRESHOLD = 32768 * std::thread::hardware_concurrency();

// Bad: Explains WHAT (obvious from code)
// Set threshold to 32768 times hardware concurrency
static const ssize_t OMP_THRESHOLD = 32768 * std::thread::hardware_concurrency();
```

### Python Style

Follow PEP 8 with these additions:

```python
# Imports: standard, third-party, local
import sys
import numpy as np
import ternary_simd_engine as tc

# Type hints (Python 3.7+)
def process_array(data: np.ndarray, size: int) -> np.ndarray:
    ...

# Docstrings
def int_to_trit(value: int) -> int:
    """Convert integer (-1, 0, +1) to trit encoding (0b00, 0b01, 0b10).

    Args:
        value: Integer in range [-1, 0, 1]

    Returns:
        Trit encoding (0b00, 0b01, or 0b10)
    """
```

### Import Path Convention

All Python scripts that import project modules should use this standard pattern:

```python
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
```

**Why this pattern:**
- Consistent across the entire codebase
- Readable and maintainable
- Platform-independent (works on Windows, Linux, macOS)
- Explicit about what's being added to sys.path
- Uses `.resolve()` to get absolute paths

**Anti-patterns to avoid:**
```python
# DON'T: Chained os.path calls (hard to read)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# DON'T: Relative string paths (fragile)
sys.path.insert(0, '..')

# DON'T: Multiple sys.path additions (causes import confusion)
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "models" / "tritnet" / "src"))
```

**For files at different depths:**
- **Depth 2** (e.g., `build/build.py`): `PROJECT_ROOT = Path(__file__).parent.parent.resolve()`
- **Depth 3** (e.g., `tests/python/test_phase0.py`): `PROJECT_ROOT = Path(__file__).parent.parent.resolve()`
- **Depth 4** (e.g., `models/tritnet/src/train_tritnet.py`): `PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()`

This ensures all scripts add the project root to `sys.path`, enabling imports like:
```python
import ternary_simd_engine as tc
from models.tritnet.src.ternary_layers import TernaryLinear
```

## Testing Requirements

### Correctness Tests (Required)

All code changes must pass existing tests:

```bash
python tests/test_phase0.py  # Must pass
python tests/test_omp.py     # Must pass
```

### New Operation Tests

When adding a new operation, add tests to `tests/test_phase0.py`:

```python
def test_new_operation():
    """Test new operation correctness."""
    # Test all edge cases
    a = np.array([0b00, 0b01, 0b10], dtype=np.uint8)
    b = np.array([0b00, 0b01, 0b10], dtype=np.uint8)
    result = tc.new_op(a, b)
    expected = np.array([...], dtype=np.uint8)
    assert np.array_equal(result, expected), "new_op failed"
    print("✓ new_op passed")
```

### Performance Benchmarks

For optimization changes, provide before/after benchmarks:

```bash
# Before optimization
python benchmarks/bench_phase0.py > before.txt

# Apply changes and rebuild

# After optimization
python benchmarks/bench_phase0.py > after.txt

# Include diff in PR description
```

## Documentation

### Source Code Documentation

When modifying source files, update corresponding documentation:

- `ternary_lut_gen.h` → No docs (self-documenting)
- `ternary_algebra.h` → `docs/api-reference/ternary-core-header.md`
- `ternary_errors.h` → `docs/api-reference/error-handling.md`
- `ternary_simd_engine.cpp` → `docs/api-reference/ternary-core-simd.md`

### Documentation Standards

```markdown
# Use ATX-style headers (not Setext)
## Section Header

# Code blocks: specify language
​```cpp
__m256i result = _mm256_shuffle_epi8(lut, indices);
​```

# Link to related sections
See [Performance Analysis](#performance-analysis) for details.

# Include examples
Example usage:
​```python
result = tc.tadd(a, b)
​```
```

### README Updates

Update `README.md` if changes affect:
- API surface (new operations)
- Installation process
- Performance characteristics
- System requirements

## Pull Request Process

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Build system change

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Benchmarks show no regression (or improvement)

## Performance Impact
Before: X operations/sec
After: Y operations/sec
Speedup: Z%

## Documentation
- [ ] Code comments updated
- [ ] API documentation updated
- [ ] README updated (if needed)

## Checklist
- [ ] Code follows project style guidelines
- [ ] Commit messages are clear and descriptive
- [ ] No warnings from compiler
- [ ] Changes are backward compatible (or migration guide provided)
```

### Review Process

1. **Automated checks**: CI runs tests automatically
2. **Code review**: Maintainers review for correctness and style
3. **Performance review**: Benchmark results reviewed
4. **Documentation review**: Docs checked for accuracy
5. **Merge**: Approved PRs merged to main branch

## Performance Guidelines

### Phase Coherence Principle

Only add complexity if it provides **>10% performance gain**:

```cpp
// Good: 15% speedup justified
#pragma omp parallel for if(n >= OMP_THRESHOLD)

// Bad: 2% speedup, adds complexity
if (is_aligned(ptr)) {
    // Aligned load path
} else {
    // Unaligned load path
}
```

### Benchmarking Standards

Use consistent methodology:

```python
# Fixed seed for reproducibility
np.random.seed(42)

# Warmup iterations (not measured)
for _ in range(100):
    result = tc.tadd(a, b)

# Measured iterations
start = time.perf_counter()
for _ in range(1000):
    result = tc.tadd(a, b)
elapsed = time.perf_counter() - start
```

### Optimization Priorities

1. **Correctness** - Never sacrifice correctness for speed
2. **Maintainability** - Simple code is better than complex optimizations
3. **Measurable gains** - Profile before optimizing
4. **Documentation** - Explain WHY optimizations work

## Adding New Operations

### Step-by-Step Guide

#### 1. Define Logic (ternary_lut_gen.h)

```cpp
// Add operation logic function
constexpr trit new_op_logic(trit a, trit b) noexcept {
    // Implement logic using -1, 0, +1 values
    if (a == -1 && b == -1) return -1;
    // ... define all 9 cases
    return 0;
}
```

#### 2. Generate LUT (ternary_algebra.h)

```cpp
// Add constexpr LUT generation
constexpr auto NEW_OP_LUT = make_binary_lut(new_op_logic);

// Add scalar operation
static FORCE_INLINE trit new_op(trit a, trit b) {
    return NEW_OP_LUT[(a << 2) | b];
}
```

#### 3. Add SIMD Operation (ternary_simd_engine.cpp)

```cpp
// Add SIMD template specialization
template <bool Sanitize>
__m256i new_op_simd(__m256i va, __m256i vb) {
    // Mask if sanitizing
    if constexpr (Sanitize) {
        const __m256i mask = _mm256_set1_epi8(0b11);
        va = _mm256_and_si256(va, mask);
        vb = _mm256_and_si256(vb, mask);
    }

    // Build indices
    __m256i hi = _mm256_slli_epi16(va, 2);
    __m256i indices = _mm256_or_si256(hi, vb);

    // Broadcast LUT
    __m128i lut_128 = _mm_loadu_si128((const __m128i*)NEW_OP_LUT.data());
    __m256i lut_256 = _mm256_broadcastsi128_si256(lut_128);

    // Lookup
    return _mm256_shuffle_epi8(lut_256, indices);
}
```

#### 4. Add Wrapper (ternary_simd_engine.cpp)

```cpp
py::array_t<uint8_t> new_op_array(py::array_t<uint8_t> A, py::array_t<uint8_t> B) {
    return process_binary_array<SANITIZE>(A, B, new_op_simd<SANITIZE>, new_op);
}
```

#### 5. Add Python Binding (ternary_simd_engine.cpp)

```cpp
PYBIND11_MODULE(ternary_simd_engine, m) {
    // ... existing bindings
    m.def("new_op", &new_op_array, "New operation",
          py::arg("A"), py::arg("B"));
}
```

#### 6. Add Tests (tests/test_phase0.py)

```python
def test_new_op():
    # Test all cases
    a = np.array([0b00, 0b01, 0b10], dtype=np.uint8)
    b = np.array([0b00, 0b01, 0b10], dtype=np.uint8)
    result = tc.new_op(a, b)
    expected = compute_expected_new_op(a, b)
    assert np.array_equal(result, expected)
```

#### 7. Update Documentation

- Update `docs/api-reference/ternary-core-header.md`
- Update `docs/api-reference/ternary-core-simd.md`
- Update `README.md` operations table

## Questions?

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers (see NOTICE file)

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

---

**Last Updated**: 2025-10-13
**Maintained by**: Jonathan Verdun (Ternary Engine Project)
