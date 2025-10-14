# Error Handling in Ternary Engine

## Overview

Ternary Engine uses domain-specific exception types for clear error semantics and Python-friendly error propagation. All exceptions inherit from `TernaryError`, which extends `std::runtime_error`, ensuring compatibility with pybind11's automatic exception translation.

**File**: `ternary_errors.h`
**Design Principle**: YAGNI (You Aren't Gonna Need It) - Only defines actually-needed exceptions, expands only when real use cases emerge.

---

## Exception Hierarchy

```
std::runtime_error
    └── TernaryError (base)
        ├── ArraySizeMismatchError
        ├── InvalidTritError
        └── AllocationError
```

---

## Exception Types

### 1. TernaryError (Base Class)

**Purpose**: Base exception for all ternary operations.

**Constructor**:
```cpp
explicit TernaryError(const std::string& message)
```

**Usage**: Generally not thrown directly; serves as base for domain-specific exceptions.

**Python Mapping**: Caught by pybind11 and converted to `RuntimeError`.

---

### 2. ArraySizeMismatchError

**Purpose**: Thrown when binary operation inputs have different sizes.

**Constructor**:
```cpp
ArraySizeMismatchError(size_t size_a, size_t size_b)
```

**Methods**:
```cpp
size_t size_a() const;  // Size of first array
size_t size_b() const;  // Size of second array
```

**Error Message Format**:
```
Array size mismatch: array A has 1000 elements, array B has 500 elements.
Binary operations require equal-sized arrays.
```

**When Thrown**:
- In `process_binary_array()` when `A.size() != B.size()`
- Applies to: `tadd`, `tmul`, `tmin`, `tmax`

**C++ Example**:
```cpp
try {
    auto result = process_binary_array(A, B, tadd_simd<true>, tadd);
} catch (const ArraySizeMismatchError& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    std::cerr << "Size A: " << e.size_a() << ", Size B: " << e.size_b() << std::endl;
}
```

**Python Example**:
```python
import numpy as np
import ternary_simd_engine as tc

a = np.array([0, 1, 2], dtype=np.uint8)
b = np.array([0, 1], dtype=np.uint8)  # Different size!

try:
    result = tc.tadd(a, b)
except RuntimeError as e:
    print(f"Error: {e}")
    # Output: Array size mismatch: array A has 3 elements, array B has 2 elements.
    #         Binary operations require equal-sized arrays.
```

---

### 3. InvalidTritError

**Purpose**: Thrown when a trit value is outside the valid range (0b00, 0b01, 0b10).

**Constructor**:
```cpp
explicit InvalidTritError(uint8_t invalid_value)
```

**Methods**:
```cpp
uint8_t invalid_value() const;  // The invalid trit value
```

**Error Message Format**:
```
Invalid trit value: 0b11 (decimal 3).
Valid trit values are: 0b00 (-1), 0b01 (0), 0b10 (+1).
```

**When Thrown**:
- **Rarely in practice**: Input sanitization (OPT-HASWELL-02) masks invalid values by default
- Provided for explicit validation scenarios where `Sanitize=false`
- Currently not thrown in production code (reserved for future use)

**Note**: The default `Sanitize=true` mode automatically masks inputs to valid range using `maybe_mask()`, preventing invalid trit errors at runtime. This exception is available for scenarios requiring strict validation.

**C++ Example** (hypothetical strict validation):
```cpp
// If strict validation were enabled
try {
    uint8_t invalid_trit = 0b11;  // Invalid!
    if ((invalid_trit & 0b11) == 0b11) {
        throw InvalidTritError(invalid_trit);
    }
} catch (const InvalidTritError& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    std::cerr << "Invalid value: 0x" << std::hex << (int)e.invalid_value() << std::endl;
}
```

---

### 4. AllocationError

**Purpose**: Thrown when output array allocation fails.

**Constructor**:
```cpp
explicit AllocationError(size_t requested_size)
```

**Methods**:
```cpp
size_t requested_size() const;  // Requested allocation size in bytes
```

**Error Message Format**:
```
Memory allocation failed: could not allocate 10000000 bytes for output array.
Consider processing data in smaller chunks.
```

**When Thrown**:
- During `py::array_t<uint8_t> out(n)` allocation
- **Rare in practice**: Only with extremely large arrays or low memory conditions
- Currently not explicitly thrown (relies on pybind11's allocation failure handling)

**Future Use**: May be explicitly thrown with custom allocators or chunked processing.

**Python Example** (hypothetical):
```python
import numpy as np
import ternary_simd_engine as tc

# Very large array (may fail on constrained systems)
try:
    a = np.zeros(10**9, dtype=np.uint8)  # 1 GB
    b = np.zeros(10**9, dtype=np.uint8)
    result = tc.tadd(a, b)
except MemoryError as e:
    print(f"Out of memory: {e}")
```

---

## Design Rationale

### Minimal Exception Set

Following YAGNI principle, we define only exceptions that are:
1. **Actually thrown** in production code (`ArraySizeMismatchError`)
2. **Likely needed** for explicit validation (`InvalidTritError`)
3. **Documented edge cases** (`AllocationError`)

**Not included** (unless real use cases emerge):
- `IndexOutOfBoundsError` - Prevented by bounds checking in pybind11
- `NullPointerError` - Prevented by C++ type system
- `ConcurrencyError` - OpenMP handles thread safety
- `InvalidOperationError` - Static typing prevents invalid operations

### Typed Exceptions vs Generic Errors

**Before** (generic):
```cpp
if (n != B.size()) throw std::runtime_error("Arrays must match");
```

**After** (typed):
```cpp
if (n != B.size()) throw ArraySizeMismatchError(n, B.size());
```

**Benefits**:
- **Semantic clarity**: Exception type conveys intent
- **Structured data**: Access to `size_a()`, `size_b()` for debugging
- **Consistent messages**: Formatted automatically
- **Catchable by type**: Can catch specific exceptions in C++

---

## Python Exception Mapping

Pybind11 automatically translates C++ exceptions to Python:

| C++ Exception              | Python Exception | Message Format              |
|----------------------------|------------------|-----------------------------|
| `TernaryError`             | `RuntimeError`   | Custom message              |
| `ArraySizeMismatchError`   | `RuntimeError`   | "Array size mismatch: ..."  |
| `InvalidTritError`         | `RuntimeError`   | "Invalid trit value: ..."   |
| `AllocationError`          | `MemoryError`    | "Memory allocation failed..."|

**Note**: Python users see descriptive error messages without needing to know internal exception types.

---

## Error Handling Best Practices

### For C++ Developers

1. **Use typed exceptions** instead of generic `std::runtime_error`
2. **Provide context** via exception constructor parameters
3. **Catch specific types** when handling errors
4. **Document when exceptions are thrown** in function comments

**Good**:
```cpp
if (n != B.size()) throw ArraySizeMismatchError(n, B.size());
```

**Bad**:
```cpp
if (n != B.size()) throw std::runtime_error("Size mismatch");
```

### For Python Developers

1. **Catch `RuntimeError`** for ternary operation errors
2. **Parse error messages** for diagnostic information
3. **Validate inputs** before calling ternary operations
4. **Handle edge cases** (empty arrays, mismatched sizes)

**Example** (defensive Python code):
```python
import numpy as np
import ternary_simd_engine as tc

def safe_tadd(a, b):
    """Safely add two ternary arrays with validation."""
    if len(a) != len(b):
        raise ValueError(f"Array size mismatch: {len(a)} vs {len(b)}")

    if len(a) == 0:
        return np.array([], dtype=np.uint8)

    try:
        return tc.tadd(a, b)
    except RuntimeError as e:
        print(f"Ternary operation failed: {e}")
        raise
```

---

## Future Expansion

### Potential Additions (YAGNI - add only if needed)

1. **Platform-Specific Errors**:
   ```cpp
   class AVX2NotSupportedError : public TernaryError;  // CPU lacks AVX2
   ```

2. **Validation Errors**:
   ```cpp
   class InvalidDimensionError : public TernaryError;  // Multi-dimensional array
   ```

3. **Concurrency Errors**:
   ```cpp
   class ThreadPoolError : public TernaryError;  // OpenMP initialization failure
   ```

4. **I/O Errors**:
   ```cpp
   class LUTLoadError : public TernaryError;  // LUT loading failure (if loading from disk)
   ```

**Decision Criteria**:
- Does the error occur in production code?
- Does the error provide actionable information?
- Can users recover from the error?

If **all three** are "yes", add the exception. Otherwise, use generic `TernaryError`.

---

## Testing Error Handling

### C++ Tests

```cpp
// tests/test_errors.cpp
#include "../ternary_errors.h"
#include <cassert>

void test_array_size_mismatch() {
    try {
        throw ArraySizeMismatchError(100, 50);
        assert(false);  // Should not reach here
    } catch (const ArraySizeMismatchError& e) {
        assert(e.size_a() == 100);
        assert(e.size_b() == 50);
        assert(std::string(e.what()).find("mismatch") != std::string::npos);
    }
}
```

### Python Tests

```python
# tests/test_errors.py
import pytest
import numpy as np
import ternary_simd_engine as tc

def test_array_size_mismatch():
    a = np.array([0, 1, 2], dtype=np.uint8)
    b = np.array([0, 1], dtype=np.uint8)

    with pytest.raises(RuntimeError, match="Array size mismatch"):
        tc.tadd(a, b)

def test_valid_operations_no_error():
    a = np.array([0, 1, 2], dtype=np.uint8)
    b = np.array([2, 1, 0], dtype=np.uint8)

    result = tc.tadd(a, b)  # Should not raise
    assert len(result) == 3
```

---

## Summary

- **Centralized error handling** in `ternary_errors.h`
- **Three exception types**: `ArraySizeMismatchError` (active), `InvalidTritError` (reserved), `AllocationError` (future)
- **YAGNI principle**: Minimal set, expand only when needed
- **Python-friendly**: Automatic translation via pybind11
- **Typed exceptions**: Better semantics than generic `std::runtime_error`
- **Auditability**: Structured error data for debugging

**Current Status**: Production-ready with minimal exception set, ready for future expansion as use cases emerge.
