# Backend System Reference

**Files:** `backend_plugin_api.h`, `backend_registry_dispatch.cpp`, `backend_*.cpp`
**Status:** Production-ready

---

## Overview

The backend system provides a pluggable architecture for ternary operations. Multiple implementations can coexist (Scalar, AVX2 v1, AVX2 v2, future AVX-512, ARM NEON), with runtime selection of the best available backend.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        APPLICATION CODE                                │
│                                                                        │
│   ternary_dispatch_tadd(dst, a, b, n);  // High-level API             │
│                                                                        │
└────────────────────────────────────────┬───────────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       DISPATCH LAYER                                   │
│                                                                        │
│   g_active_backend->tadd(dst, a, b, n);  // Route to active backend   │
│                                                                        │
└────────────────────────────────────────┬───────────────────────────────┘
                                         │
          ┌──────────────────────────────┼──────────────────────────────┐
          │                              │                              │
          ▼                              ▼                              ▼
┌──────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  Scalar Backend  │     │    AVX2 v1 Backend   │     │   AVX2 v2 Backend   │
│                  │     │                      │     │                     │
│  Portable        │     │  Current production  │     │  v1.2.0 opts        │
│  ~100 ME/s       │     │  ~3,500 ME/s         │     │  ~4,000 ME/s (TBD)  │
│                  │     │                      │     │                     │
└──────────────────┘     └──────────────────────┘     └─────────────────────┘
```

---

## Backend Interface

### C API Structure

```c
typedef struct {
    // Metadata
    TernaryBackendInfo info;

    // Core unary operations
    TernaryUnaryOp tnot;

    // Core binary operations
    TernaryBinaryOp tadd;
    TernaryBinaryOp tmul;
    TernaryBinaryOp tmax;
    TernaryBinaryOp tmin;

    // Fusion operations (Phase 4.1)
    TernaryBinaryOp fused_tnot_tadd;
    TernaryBinaryOp fused_tnot_tmul;
    TernaryBinaryOp fused_tnot_tmin;
    TernaryBinaryOp fused_tnot_tmax;

    // Advanced operations (optional)
    TernaryBinaryOp tand;
    TernaryBinaryOp tor;
} TernaryBackend;
```

### Function Signatures

```c
// Unary: dst[i] = op(src[i]) for i in [0, n)
typedef void (*TernaryUnaryOp)(uint8_t* dst, const uint8_t* src, size_t n);

// Binary: dst[i] = op(a[i], b[i]) for i in [0, n)
typedef void (*TernaryBinaryOp)(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n);
```

---

## Backend Capabilities

```c
typedef enum {
    TERNARY_CAP_SCALAR       = 0x0001,  // Scalar operations
    TERNARY_CAP_SIMD_128     = 0x0002,  // 128-bit SIMD (SSE/NEON)
    TERNARY_CAP_SIMD_256     = 0x0004,  // 256-bit SIMD (AVX2)
    TERNARY_CAP_SIMD_512     = 0x0008,  // 512-bit SIMD (AVX-512)
    TERNARY_CAP_OPENMP       = 0x0010,  // OpenMP multi-threading
    TERNARY_CAP_FUSION       = 0x0020,  // Operation fusion
    TERNARY_CAP_CANONICAL    = 0x0040,  // Canonical indexing
    TERNARY_CAP_DUAL_SHUFFLE = 0x0080,  // Dual-shuffle XOR
    TERNARY_CAP_LUT_256B     = 0x0100   // 256-byte LUTs
} TernaryBackendCapabilities;
```

---

## Backend Selection

### Automatic Selection

```c
// At program startup
ternary_backend_init();  // Auto-detects and selects best backend

// Uses scoring based on capabilities:
// - SIMD-512: +1000 points
// - SIMD-256: +500 points
// - SIMD-128: +250 points
// - Dual-shuffle: +100 points
// - Canonical: +50 points
// - LUT-256B: +50 points
// - Fusion: +25 points
// - OpenMP: +25 points
```

### Manual Selection

```c
// Find specific backend
const TernaryBackend* avx2_v2 = ternary_backend_find("AVX2_v2");
if (avx2_v2) {
    ternary_backend_set_active(avx2_v2);
}

// Re-enable automatic selection
ternary_backend_set_active(NULL);
```

---

## Registered Backends

### Scalar Backend

**Name:** `Scalar`
**Capabilities:** `TERNARY_CAP_SCALAR`
**Throughput:** ~100 ME/s
**Use case:** Baseline, verification, fallback on non-x86

```c
// Portable, works everywhere
static void scalar_tadd(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = tadd(a[i], b[i]);
    }
}
```

### AVX2 v1 Backend

**Name:** `AVX2_v1`
**Capabilities:** `SIMD_256 | FUSION`
**Throughput:** ~3,500 ME/s
**Use case:** Current production

```c
// Processes 32 trits per iteration
static void avx2_v1_tadd(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i vr = tadd_simd<true>(va, vb);
        _mm256_storeu_si256((__m256i*)(dst + i), vr);
    }
    // Scalar tail...
}
```

### AVX2 v2 Backend (Experimental)

**Name:** `AVX2_v2`
**Capabilities:** `SIMD_256 | FUSION | CANONICAL | DUAL_SHUFFLE`
**Throughput:** ~4,000 ME/s (target)
**Use case:** v1.2.0 optimizations

Adds canonical indexing and dual-shuffle XOR for ~15% improvement.

---

## Dispatch Functions

High-level API that automatically uses the best backend:

```c
// Core operations
void ternary_dispatch_tnot(uint8_t* dst, const uint8_t* src, size_t n);
void ternary_dispatch_tadd(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n);
void ternary_dispatch_tmul(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n);
void ternary_dispatch_tmax(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n);
void ternary_dispatch_tmin(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n);

// Fusion operations
void ternary_dispatch_fused_tnot_tadd(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n);
void ternary_dispatch_fused_tnot_tmul(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n);
void ternary_dispatch_fused_tnot_tmin(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n);
void ternary_dispatch_fused_tnot_tmax(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n);
```

---

## Usage Example

```c
#include "core/simd/backend_plugin_api.h"

int main() {
    // Initialize backend system
    if (!ternary_backend_init()) {
        fprintf(stderr, "Failed to initialize backends\n");
        return 1;
    }

    // Print available backends
    ternary_backend_print_all();

    // Allocate arrays
    uint8_t a[1000], b[1000], result[1000];
    // ... initialize a, b ...

    // Use dispatch API (auto-selects best backend)
    ternary_dispatch_tadd(result, a, b, 1000);

    // Query active backend
    const TernaryBackend* active = ternary_backend_get_active();
    printf("Using backend: %s\n", active->info.name);

    // Cleanup
    ternary_backend_shutdown();
    return 0;
}
```

---

## Adding a New Backend

### 1. Implement the Backend

```c
// backend_my_impl.cpp

#include "backend_plugin_api.h"

static bool my_backend_available(void) {
    // Check if this backend can run on current CPU
    return has_my_feature();
}

static void my_tadd(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t n) {
    // Implementation
}

// ... other operations ...

static const TernaryBackend MY_BACKEND = {
    .info = {
        .name = "MyBackend",
        .description = "My custom backend",
        .version = TERNARY_VERSION(1, 0, 0),
        .capabilities = TERNARY_CAP_SCALAR | TERNARY_CAP_MY_FEATURE,
        .preferred_batch_size = 64,
        .is_available = my_backend_available
    },
    .tnot = my_tnot,
    .tadd = my_tadd,
    .tmul = my_tmul,
    .tmax = my_tmax,
    .tmin = my_tmin,
    .fused_tnot_tadd = my_fused_tnot_tadd,
    // ...
};

void ternary_register_my_backend(void) {
    ternary_backend_register(&MY_BACKEND);
}
```

### 2. Register in Dispatch

```c
// In backend_registry_dispatch.cpp

extern void ternary_register_my_backend(void);

bool ternary_backend_init(void) {
    // ...
    ternary_register_my_backend();  // Add registration
    // ...
}
```

---

## Debugging

### Print All Backends

```c
ternary_backend_print_all();

// Output:
// ===========================================
// Registered Ternary Backends (3)
// ===========================================
//
// [0] Scalar
//     Description: Portable scalar implementation
//     Version: 1.0.0
//     Capabilities: 0x00000001
//     Batch size: 1
//     Active: no
//
// [1] AVX2_v1
//     Description: AVX2 SIMD implementation
//     Version: 1.0.0
//     Capabilities: 0x00000024
//     Batch size: 32
//     Active: YES
```

### Capabilities to String

```c
char buffer[256];
ternary_backend_capabilities_to_string(0x00000024, buffer, sizeof(buffer));
// buffer = "SIMD-256, Fusion"
```

---

**See also:** [SIMD Kernels](SIMD_KERNELS.md), [CPU Detection](CPU_DETECTION.md)
