// ternary_profiler.h — Optional profiler annotations (VTune, NVTX, Perfetto)
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
// IMPLEMENTATION STATUS
// =============================================================================
//
// **INTEGRATED - PRODUCTION READY**
//
// This header provides cross-platform profiler integration for performance analysis.
// Status:
//   - VTune (ITT API): Fully integrated, tested with Intel VTune Profiler
//   - NVTX (CUDA/GPU): Framework ready, awaiting GPU port
//   - Perfetto: Stub placeholder for future web-based tracing
//   - Default (no-op): Zero overhead when profiling disabled
//
// =============================================================================
// DESIGN RATIONALE
// =============================================================================
//
// Profiler annotations enable:
// - Visualization of SIMD loop timing in VTune/Perfetto
// - GPU profiling correlation with NVTX (for future CUDA support)
// - Zero overhead when profiling is disabled (compile-time no-ops)
// - Integration with existing profiling workflows
//
// PROFILER TARGETS:
//   1. Intel VTune (ITT API) - CPU profiling [INTEGRATED]
//   2. NVIDIA Nsight (NVTX) - GPU profiling [Framework ready]
//   3. Chrome Tracing (Perfetto) - Web timeline [Stub only]
//
// CROSS-PLATFORM COMPATIBILITY:
//   - GCC/Clang (Linux/macOS): Full support
//   - MSVC (Windows): Full support
//   - Any C++11+ compiler: No-op stubs compile cleanly
//
// COMPILATION:
//   # Enable VTune annotations
//   g++ -DTERNARY_ENABLE_VTUNE -O3 ... -littnotify
//
//   # Enable NVTX annotations
//   g++ -DTERNARY_ENABLE_NVTX -O3 ... -lnvToolsExt
//
//   # Default: profiling disabled (zero overhead)
//   g++ -O3 ...
//
// USAGE EXAMPLE:
//   #include "ternary_core/profiling/ternary_profiler.h"
//
//   TERNARY_PROFILE_DOMAIN(g_domain, "TernaryCore");
//   TERNARY_PROFILE_TASK_NAME(g_simd_loop, "SIMD_Loop");
//
//   void process_array() {
//       TERNARY_PROFILE_TASK_BEGIN(g_domain, g_simd_loop);
//       // ... hot loop code ...
//       TERNARY_PROFILE_TASK_END(g_domain);
//   }
//
// =============================================================================

#ifndef TERNARY_PROFILER_H
#define TERNARY_PROFILER_H

// =============================================================================
// Intel VTune (ITT API) Support
// =============================================================================

#ifdef TERNARY_ENABLE_VTUNE

#include <ittnotify.h>

// Domain creation (call once at startup)
#define TERNARY_PROFILE_DOMAIN(var_name, domain_name) \
    __itt_domain* var_name = __itt_domain_create(domain_name)

// String handle creation (reusable task names)
#define TERNARY_PROFILE_TASK_NAME(var_name, task_name) \
    __itt_string_handle* var_name = __itt_string_handle_create(task_name)

// Task begin/end (marks execution regions)
#define TERNARY_PROFILE_TASK_BEGIN(domain, handle) \
    __itt_task_begin(domain, __itt_null, __itt_null, handle)

#define TERNARY_PROFILE_TASK_END(domain) \
    __itt_task_end(domain)

// Frame markers (for iterative algorithms)
#define TERNARY_PROFILE_FRAME_BEGIN(domain) \
    __itt_frame_begin_v3(domain, nullptr)

#define TERNARY_PROFILE_FRAME_END(domain) \
    __itt_frame_end_v3(domain, nullptr)

// =============================================================================
// NVIDIA NVTX (Nsight) Support
// =============================================================================

#elif defined(TERNARY_ENABLE_NVTX)

#include <nvToolsExt.h>

// Domain creation (NVTX domains for logical grouping)
#define TERNARY_PROFILE_DOMAIN(var_name, domain_name) \
    nvtxDomainHandle_t var_name = nvtxDomainCreateA(domain_name)

// String handle (NVTX uses message IDs for efficiency)
#define TERNARY_PROFILE_TASK_NAME(var_name, task_name) \
    const char* var_name = task_name

// Task begin/end (push/pop on NVTX stack)
#define TERNARY_PROFILE_TASK_BEGIN(domain, handle) \
    do { \
        nvtxEventAttributes_t eventAttrib = {0}; \
        eventAttrib.version = NVTX_VERSION; \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
        eventAttrib.message.ascii = handle; \
        nvtxDomainRangePushEx(domain, &eventAttrib); \
    } while(0)

#define TERNARY_PROFILE_TASK_END(domain) \
    nvtxDomainRangePop(domain)

// Frame markers (less common in NVTX, use push/pop)
#define TERNARY_PROFILE_FRAME_BEGIN(domain) \
    nvtxDomainRangePushA(domain, "Frame")

#define TERNARY_PROFILE_FRAME_END(domain) \
    nvtxDomainRangePop(domain)

// =============================================================================
// Chrome Tracing / Perfetto Support (ROADMAP)
// =============================================================================

#elif defined(TERNARY_ENABLE_PERFETTO)

// ROADMAP: Perfetto SDK integration planned for future release
// Current: Stub implementation (compiles but no output)

#define TERNARY_PROFILE_DOMAIN(var_name, domain_name) \
    int var_name = 0

#define TERNARY_PROFILE_TASK_NAME(var_name, task_name) \
    const char* var_name = task_name

#define TERNARY_PROFILE_TASK_BEGIN(domain, handle) \
    ((void)0)

#define TERNARY_PROFILE_TASK_END(domain) \
    ((void)0)

#define TERNARY_PROFILE_FRAME_BEGIN(domain) \
    ((void)0)

#define TERNARY_PROFILE_FRAME_END(domain) \
    ((void)0)

// =============================================================================
// No-Op Stubs (default, zero overhead)
// =============================================================================

#else

// When profiling is disabled, all macros compile to no-ops
#define TERNARY_PROFILE_DOMAIN(var_name, domain_name) \
    int var_name = 0

#define TERNARY_PROFILE_TASK_NAME(var_name, task_name) \
    const char* var_name = nullptr

#define TERNARY_PROFILE_TASK_BEGIN(domain, handle) \
    ((void)0)

#define TERNARY_PROFILE_TASK_END(domain) \
    ((void)0)

#define TERNARY_PROFILE_FRAME_BEGIN(domain) \
    ((void)0)

#define TERNARY_PROFILE_FRAME_END(domain) \
    ((void)0)

#endif

// =============================================================================
// Convenience: RAII-style profiling scope (C++11+)
// =============================================================================

#ifdef __cplusplus

#if defined(TERNARY_ENABLE_VTUNE) || defined(TERNARY_ENABLE_NVTX)

// RAII helper for automatic task begin/end
template <typename DomainT>
struct TernaryProfileScope {
    DomainT domain;

    TernaryProfileScope(DomainT d, const char* task_name) : domain(d) {
#ifdef TERNARY_ENABLE_VTUNE
        __itt_string_handle* handle = __itt_string_handle_create(task_name);
        __itt_task_begin(domain, __itt_null, __itt_null, handle);
#elif defined(TERNARY_ENABLE_NVTX)
        nvtxEventAttributes_t eventAttrib = {0};
        eventAttrib.version = NVTX_VERSION;
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        eventAttrib.message.ascii = task_name;
        nvtxDomainRangePushEx(domain, &eventAttrib);
#endif
    }

    ~TernaryProfileScope() {
#ifdef TERNARY_ENABLE_VTUNE
        __itt_task_end(domain);
#elif defined(TERNARY_ENABLE_NVTX)
        nvtxDomainRangePop(domain);
#endif
    }
};

#define TERNARY_PROFILE_SCOPE(domain, task_name) \
    TernaryProfileScope<decltype(domain)> __profile_scope_##__LINE__(domain, task_name)

#else

// No-op RAII scope when profiling is disabled
struct TernaryProfileScope {
    template <typename... Args>
    TernaryProfileScope(Args&&...) {}
};

#define TERNARY_PROFILE_SCOPE(domain, task_name) \
    ((void)0)

#endif

#endif // __cplusplus

// =============================================================================
// Usage Example (for documentation)
// =============================================================================

/*
// 1. Create domain at startup (global or static)
TERNARY_PROFILE_DOMAIN(g_domain, "TernaryCore");

// 2. Create task name handles (optional, for efficiency)
TERNARY_PROFILE_TASK_NAME(g_simd_loop, "SIMD_Loop");
TERNARY_PROFILE_TASK_NAME(g_scalar_tail, "Scalar_Tail");

// 3. Annotate hot loops
void process_array() {
    TERNARY_PROFILE_TASK_BEGIN(g_domain, g_simd_loop);
    #pragma omp parallel for
    for (...) {
        // SIMD processing
    }
    TERNARY_PROFILE_TASK_END(g_domain);

    TERNARY_PROFILE_TASK_BEGIN(g_domain, g_scalar_tail);
    for (...) {
        // Scalar tail processing
    }
    TERNARY_PROFILE_TASK_END(g_domain);
}

// 4. RAII-style (C++ only, automatic cleanup)
void process_array_raii() {
    {
        TERNARY_PROFILE_SCOPE(g_domain, "SIMD_Loop");
        // SIMD processing
    }  // Automatic TERNARY_PROFILE_TASK_END

    {
        TERNARY_PROFILE_SCOPE(g_domain, "Scalar_Tail");
        // Scalar tail
    }
}
*/

#endif // TERNARY_PROFILER_H
