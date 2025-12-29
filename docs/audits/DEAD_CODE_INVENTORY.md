# Dead Code Inventory

**Doc-Type:** Technical Audit · Version 1.0 · Generated 2025-12-09

This document provides a comprehensive inventory of dead, unused, and deprecated code in the Ternary Engine repository.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Unimplemented API Slots](#1-unimplemented-api-slots)
3. [Deprecated Benchmarks](#2-deprecated-benchmarks)
4. [Unused Function Definitions](#3-unused-function-definitions)
5. [Orphaned Files](#4-orphaned-files)
6. [Commented-Out Code](#5-commented-out-code)
7. [Cleanup Recommendations](#cleanup-recommendations)

---

## Executive Summary

| Category | File Count | Lines of Code | Disk Space | Recommendation |
|----------|------------|---------------|------------|----------------|
| Unimplemented APIs | 3 files | ~20 lines | Minimal | Implement or Remove |
| Deprecated Benchmarks | 10 files | ~3,000 lines | 108 KB | Archive or Delete |
| Unused Functions | 6 files | ~200 lines | Minimal | Remove |
| Orphaned Files | 2 files | ~100 lines | 4 KB | Delete |
| Commented Code | 4 files | ~50 lines | Minimal | Remove |

**Total Dead Code:** ~3,400 lines across 25 files

---

## 1. Unimplemented API Slots

### 1.1 tand and tor Operations

**Impact:** Dead API surface area, confusing for API consumers

#### Location

These operations are defined in all three backend implementations but set to NULL:

```cpp
// backend_avx2_v1_baseline.cpp:139-140
static const TernaryBackend g_avx2_v1_backend = {
    .name = "avx2_v1_baseline",
    .version = "1.0.0",
    // ... other operations ...
    .tand = NULL,  // <-- Unimplemented
    .tor = NULL    // <-- Unimplemented
};

// backend_avx2_v2_optimized.cpp:641-642
static const TernaryBackend g_avx2_v2_backend = {
    .name = "avx2_v2_optimized",
    .version = "2.0.0",
    // ... other operations ...
    .tand = NULL,  // <-- Unimplemented
    .tor = NULL    // <-- Unimplemented
};

// backend_scalar_impl.cpp:126-127
static const TernaryBackend g_scalar_backend = {
    .name = "scalar",
    .version = "1.0.0",
    // ... other operations ...
    .tand = NULL,  // <-- Unimplemented
    .tor = NULL    // <-- Unimplemented
};
```

#### API Definition

```cpp
// backend_plugin_api.h
struct TernaryBackend {
    // ... other fields ...

    // Binary operations
    void (*tand)(const uint8_t*, const uint8_t*, uint8_t*, size_t);
    void (*tor)(const uint8_t*, const uint8_t*, uint8_t*, size_t);
};
```

#### Analysis

**What are tand and tor?**

These would be ternary AND and OR operations, analogous to binary AND/OR but for balanced ternary values. In balanced ternary:

```
Ternary AND (tand):
  -1 AND -1 = -1
  -1 AND  0 = -1
  -1 AND +1 = -1
   0 AND -1 = -1
   0 AND  0 =  0
   0 AND +1 =  0
  +1 AND -1 = -1
  +1 AND  0 =  0
  +1 AND +1 = +1

Ternary OR (tor):
  Similar truth table with OR semantics
```

**Why Unimplemented?**

1. `tmin` and `tmax` may serve similar purposes (min = AND-like, max = OR-like)
2. Semantic definition of tand/tor in balanced ternary is non-obvious
3. May have been planned but never prioritized

#### Recommendation

**Option A: Implement**

```cpp
// Implementation for tand (min-like semantics)
static void tand_scalar(const uint8_t* a, const uint8_t* b,
                        uint8_t* r, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        // tand = min in balanced ternary interpretation
        r[i] = (a[i] < b[i]) ? a[i] : b[i];
    }
}

// Or if tand means "both positive"
static void tand_positive(const uint8_t* a, const uint8_t* b,
                          uint8_t* r, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        // +1 only if both are +1
        int8_t va = (int8_t)a[i] - 1;  // Convert 0,1,2 to -1,0,+1
        int8_t vb = (int8_t)b[i] - 1;
        int8_t vr = (va > 0 && vb > 0) ? 1 : ((va < 0 || vb < 0) ? -1 : 0);
        r[i] = (uint8_t)(vr + 1);
    }
}
```

**Option B: Remove from API**

```cpp
// backend_plugin_api.h - Remove unused slots
struct TernaryBackend {
    // Remove: void (*tand)(...);
    // Remove: void (*tor)(...);
};
```

**Recommended Action:** Remove from API (Option B) unless there's a clear use case

---

## 2. Deprecated Benchmarks

### Location

**Directory:** `benchmarks/deprecated/`

### File Inventory

| File | Size | Lines | Purpose | Superseded By |
|------|------|-------|---------|---------------|
| `bench_backend_fusion.py` | 10 KB | ~300 | Backend fusion testing | `bench_simd_fusion_ops.py` |
| `bench_backends.py` | 12 KB | ~350 | Backend comparison | Integrated into main suite |
| `bench_backends_improved.py` | 20 KB | ~550 | Improved backend tests | `bench_simd_core_ops.py` |
| `bench_fusion_phase41.py` | 10 KB | ~280 | Phase 4.1 fusion | `bench_simd_fusion_ops.py` |
| `bench_fusion_poc.py` | 15 KB | ~420 | Proof of concept | `test_fusion.py` |
| `bench_fusion_rigorous.py` | 10 KB | ~290 | Rigorous testing | `bench_simd_fusion_ops.py` |
| `bench_fusion_simple.py` | 4 KB | ~120 | Simple fusion tests | Merged into main |
| `bench_fusion_validation.py` | 8 KB | ~230 | Validation tests | `test_fusion.py` |
| `bench_with_load_context.py` | 17 KB | ~480 | Load-aware benchmarks | Experimental |
| `README.md` | 1 KB | ~30 | Documentation | - |

**Total:** 10 files, ~3,050 lines, 108 KB

### Example Deprecated Code

```python
# bench_backends.py (deprecated)
"""
Backend comparison benchmarks.

DEPRECATED: This file has been superseded by the unified benchmark suite.
See benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py

Keeping for historical reference only.
"""

# ... 350 lines of outdated benchmark code ...
```

### Why These Exist

The benchmark suite evolved through several iterations:

1. **Initial:** Individual benchmark files per feature
2. **Phase 4.1:** Fusion-specific benchmarks added
3. **Consolidation:** Unified into `python-with-interpreter-overhead/`
4. **Current:** Old files moved to `deprecated/` but not deleted

### Recommendation

**Option A: Delete Entirely**

```bash
# Remove deprecated benchmarks
rm -rf benchmarks/deprecated/
```

**Option B: Archive to Separate Branch**

```bash
git checkout -b archive/deprecated-benchmarks
git add benchmarks/deprecated/
git commit -m "Archive deprecated benchmarks before removal"
git checkout main
rm -rf benchmarks/deprecated/
git commit -m "Remove deprecated benchmarks (archived in archive/deprecated-benchmarks)"
```

**Option C: Move to docs/historical/**

```bash
mkdir -p docs/historical/benchmarks/
mv benchmarks/deprecated/* docs/historical/benchmarks/
```

**Recommended Action:** Delete entirely (Option A) - these add confusion without value

---

## 3. Unused Function Definitions

### 3.1 CPU Detection Functions

**Location:** `src/core/simd/cpu_simd_capability.h`

```cpp
// Defined but never called:

bool has_avx512f() {
    // Detects AVX-512 Foundation
    // No AVX-512 backend exists
}

bool has_avx512bw() {
    // Detects AVX-512 Byte/Word
    // No AVX-512 backend exists
}

bool has_avx512vl() {
    // Detects AVX-512 Vector Length
    // No AVX-512 backend exists
}

bool has_neon() {
    // Detects ARM NEON
    // Project is x86-64 only
}

bool has_sve() {
    // Detects ARM SVE
    // Project is x86-64 only
}

bool has_sse41() {
    // Detects SSE 4.1
    // No SSE4.1-specific backend
}
```

**Evidence of Non-Use:**

```bash
$ grep -r "has_avx512" src/
# Only definition in cpu_simd_capability.h, no calls

$ grep -r "has_neon" src/
# Only definition in cpu_simd_capability.h, no calls

$ grep -r "has_sve" src/
# Only definition in cpu_simd_capability.h, no calls
```

### 3.2 simd_level_name Function

**Location:** `src/core/simd/cpu_simd_capability.h`

```cpp
const char* simd_level_name(SIMDLevel level) {
    switch (level) {
        case SIMD_NONE:    return "None";
        case SIMD_SSE2:    return "SSE2";
        case SIMD_SSE41:   return "SSE4.1";
        case SIMD_AVX2:    return "AVX2";
        case SIMD_AVX512:  return "AVX-512";
        default:           return "Unknown";
    }
}
```

**Usage:** Exposed to Python but never tested or used in any script

### Recommendation

**For AVX-512/NEON/SVE detection:**

```cpp
// Option A: Remove entirely
// Delete has_avx512f(), has_avx512bw(), has_avx512vl(), has_neon(), has_sve()

// Option B: Keep with clear documentation
#ifdef TERNARY_ENABLE_FUTURE_ISA
    // These functions are placeholders for future backends
    bool has_avx512f();
    bool has_neon();
    // ...
#endif
```

---

## 4. Orphaned Files

### 4.1 nul File in Project Root

**Location:** `c:\Users\Alejandro\Documents\Ivan\Work\ternary-engine\nul`

**Content:** 235 bytes, likely Windows NUL device artifact

**Recommendation:** Delete immediately

```bash
rm nul
```

### 4.2 Investigation Benchmarks

**Location:** `benchmarks/investigation/`

These files are exploratory code that was never cleaned up:

| File | Purpose | Status |
|------|---------|--------|
| `investigate_*.py` | Performance investigations | Results integrated elsewhere |

**Recommendation:** Review, document findings, then archive or delete

---

## 5. Commented-Out Code

### 5.1 Dual-Shuffle Initialization

**Location:** `src/core/simd/backend_avx2_v2_optimized.cpp:61`

```cpp
void avx2_v2_init() {
    init_canonical_luts();
    // init_dual_shuffle_luts();  // TODO: Enable for additional performance
    g_avx2_v2_initialized = true;
}
```

**Status:** This is intentionally disabled pending validation (see DISABLED_OPTIMIZATIONS.md)

**Recommendation:** Either enable or add clear documentation why disabled

### 5.2 Debug Print Statements

Various files contain commented-out debug prints:

```cpp
// Example from various files:
// printf("Debug: value = %d\n", value);
// std::cout << "Entering function X" << std::endl;
```

**Recommendation:** Remove all commented debug statements

### 5.3 Old Implementation Remnants

```cpp
// Some files contain old implementations:

// OLD: Using single shuffle
// __m256i result = _mm256_shuffle_epi8(lut, idx);

// NEW: Using canonical indexing
__m256i result = _mm256_shuffle_epi8(canonical_lut, canonical_idx);
```

**Recommendation:** Remove commented-out old implementations

---

## Cleanup Recommendations

### Immediate Actions (< 1 hour)

| # | Action | Command | Impact |
|---|--------|---------|--------|
| 1 | Delete `nul` file | `rm nul` | Remove artifact |
| 2 | Delete deprecated benchmarks | `rm -rf benchmarks/deprecated/` | -108 KB |
| 3 | Remove debug comments | Manual cleanup | Cleaner code |

### Short-Term Actions (1-4 hours)

| # | Action | Files | Impact |
|---|--------|-------|--------|
| 4 | Remove tand/tor from API | 4 files | Cleaner API |
| 5 | Remove unused CPU detection | 1 file | ~50 lines |
| 6 | Clean investigation/ | 3 files | ~200 lines |

### Validation Script

Create a script to detect dead code:

```python
# scripts/find_dead_code.py
"""
Scan for potentially dead code in the repository.
"""
import re
from pathlib import Path

def find_unused_functions(directory):
    """Find functions defined but never called."""
    definitions = {}
    calls = set()

    for file in Path(directory).rglob("*.cpp"):
        content = file.read_text()

        # Find function definitions
        for match in re.finditer(r"^\w+\s+(\w+)\s*\([^)]*\)\s*{", content, re.MULTILINE):
            func_name = match.group(1)
            definitions[func_name] = str(file)

        # Find function calls
        for match in re.finditer(r"(\w+)\s*\(", content):
            calls.add(match.group(1))

    unused = set(definitions.keys()) - calls
    return {name: definitions[name] for name in unused}

def find_commented_code(directory):
    """Find blocks of commented-out code."""
    issues = []

    for file in Path(directory).rglob("*.cpp"):
        content = file.read_text()
        lines = content.split("\n")

        comment_block = []
        for i, line in enumerate(lines):
            if line.strip().startswith("//") and not line.strip().startswith("///"):
                # Check if it looks like code (contains operators, semicolons)
                if re.search(r"[=;{}()\[\]]", line):
                    comment_block.append((i + 1, line))
            else:
                if len(comment_block) >= 3:
                    issues.append({
                        "file": str(file),
                        "start_line": comment_block[0][0],
                        "end_line": comment_block[-1][0],
                        "lines": len(comment_block)
                    })
                comment_block = []

    return issues

if __name__ == "__main__":
    print("Scanning for dead code...")

    unused = find_unused_functions("src/")
    if unused:
        print(f"\nPotentially unused functions ({len(unused)}):")
        for name, file in unused.items():
            print(f"  {name} in {file}")

    commented = find_commented_code("src/")
    if commented:
        print(f"\nCommented-out code blocks ({len(commented)}):")
        for issue in commented:
            print(f"  {issue['file']}:{issue['start_line']}-{issue['end_line']} ({issue['lines']} lines)")
```

---

## Dead Code Impact Summary

### By Category

```
Dead Code Distribution:

Deprecated Benchmarks  ████████████████████████████████████████ 88%
Unused Functions       ████                                      6%
Unimplemented APIs     ██                                        3%
Commented Code         █                                         2%
Orphaned Files         █                                         1%
```

### Cleanup Priority

1. **Delete `benchmarks/deprecated/`** - Largest impact, no risk
2. **Remove `nul` file** - Artifact cleanup
3. **Remove unused CPU detection** - Clean code
4. **Remove/implement tand/tor** - API consistency
5. **Clean commented code** - Maintainability

### Post-Cleanup Benefits

- **-3,400 lines** of code to maintain
- **-108 KB** of repository size
- **Cleaner API** surface
- **Reduced confusion** for contributors
- **Faster CI** (fewer files to process)

---

**Document Version:** 1.0
**Last Updated:** 2025-12-09
**Author:** Claude Code Audit
