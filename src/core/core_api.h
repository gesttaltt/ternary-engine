// core_api.h — Unified API for Ternary Core Kernel
//
// Copyright (c) 2025 Jonathan Verdun (Ternary Engine Project)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// =============================================================================
// DESIGN RATIONALE
// =============================================================================
//
// This header provides a single entry point to the ternary_core kernel.
// It includes all validated, production-ready components:
//
// - Core algebra: LUT-based ternary operations
// - SIMD kernels: AVX2-accelerated operations
// - CPU detection: Runtime ISA detection
// - FFI layer: Pure C API for cross-language integration
//
// ARCHITECTURE:
// - ternary_core/ contains mathematically stable, validated components
// - ternary_engine/ contains experimental optimizations (not included here)
// - This file establishes the boundary between kernel and engine
//
// USAGE:
//   #include "ternary_core/core_api.h"
//
// =============================================================================

#ifndef TERNARY_CORE_API_H
#define TERNARY_CORE_API_H

// Core algebra and LUT generation
#include "algebra/ternary_lut_gen.h"
#include "algebra/ternary_algebra.h"

// SIMD kernels and CPU detection
#include "simd/cpu_simd_capability.h"
#include "simd/simd_avx2_32trit_ops.h"
#include "simd/fused_binary_unary_ops.h"

// FFI layer for cross-language integration
#include "ffi/ternary_c_api.h"

// =============================================================================
// VERSION INFORMATION
// =============================================================================

#define TERNARY_CORE_VERSION_MAJOR 1
#define TERNARY_CORE_VERSION_MINOR 0
#define TERNARY_CORE_VERSION_PATCH 0

#define TERNARY_CORE_VERSION "1.0.0"

// =============================================================================
// FEATURE FLAGS
// =============================================================================

// Validated features (safe for production use)
#define TERNARY_CORE_HAS_ALGEBRA       1  // Core ternary algebra
#define TERNARY_CORE_HAS_SIMD_AVX2     1  // AVX2 SIMD kernels
#define TERNARY_CORE_HAS_CPU_DETECT    1  // Runtime CPU detection
#define TERNARY_CORE_HAS_FFI           1  // C FFI layer
#define TERNARY_CORE_HAS_FUSION_POC    1  // Phase 4.0 fusion (tnot_tadd)

// Validated features (awaiting integration)
#define TERNARY_CORE_HAS_FUSION_SUITE  1  // ✓ VALIDATED 2025-10-29: Phase 4.1 (avg 2.80× speedup, min 1.53×)

// Experimental features (not included in core API)
#define TERNARY_CORE_HAS_DENSE243      0  // Dense243 in ternary_engine/lib/, not ternary_core/
#define TERNARY_CORE_HAS_OPENMP        0  // OpenMP threading (root cause fixed, needs CI validation)
#define TERNARY_CORE_HAS_STREAMING     0  // Non-temporal stores (experimental)

#endif // TERNARY_CORE_API_H
