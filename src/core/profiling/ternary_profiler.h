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
// **CALL SITES INTEGRATED; PERFETTO VERIFIED, VTUNE/NVTX STILL UNVERIFIED**
//
// This header provides cross-platform profiler integration for performance analysis.
// Status (corrected 2026-08-12 — this section previously overclaimed, see
// CLAUDE.md "Critical Gaps" for the reconciled status; corrected again
// 2026-08-25 in the opposite direction — Perfetto below had been left
// describing itself as a stub for a full session after the real backend
// was implemented further down in this same file):
//   - Call sites: genuinely wired into the hot path — see
//     TERNARY_PROFILE_TASK_BEGIN/END usage in bindings_core_ops.cpp
//   - VTune (ITT API): macros implemented, but no build script in build/
//     ever defines TERNARY_ENABLE_VTUNE — "tested with Intel VTune
//     Profiler" was not something this repo's own build system can
//     substantiate; treat as unverified until someone actually builds
//     with -DTERNARY_ENABLE_VTUNE -littnotify and confirms
//   - NVTX (CUDA/GPU): Framework ready, awaiting GPU port
//   - Perfetto: REAL backend, built and verified 2026-08-25 -- vendored
//     SDK (third_party/perfetto/), real TRACE_EVENT_BEGIN/END/TRACE_EVENT
//     calls, wired into both a standalone native demo
//     (benchmarks/cpp-native-kernels/bench_perfetto_trace.cpp) and the
//     main ternary_simd_engine module (build/build.py --enable-perfetto).
//     Verified against actual trace contents (trace_processor_shell),
//     not just "it compiled" -- see
//     reports/2026-08-25/PERFETTO_PROFILER_INTEGRATION.md
//   - Default (no-op, what every current build actually uses): Zero
//     overhead when profiling disabled
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
//   1. Intel VTune (ITT API) - CPU profiling [Macros implemented, unbuilt/unverified]
//   2. NVIDIA Nsight (NVTX) - GPU profiling [Framework ready, awaiting GPU port]
//   3. Perfetto - Tracing (ui.perfetto.dev) [REAL, built and verified 2026-08-25]
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
//   #include "core/profiling/ternary_profiler.h"
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
// Chrome Tracing / Perfetto Support
// =============================================================================
//
// Real integration, added 2026-08-25 -- see third_party/perfetto/README.md
// for the vendored SDK's provenance, and ternary_profiler_perfetto.cc for
// the one translation unit that owns PERFETTO_TRACK_EVENT_STATIC_STORAGE()
// and the start/stop helpers below. Unlike VTune/NVTX, Perfetto needs no
// proprietary tool or GPU to build and verify against -- it's an
// open-source SDK, which is why this backend (and not those two) is the
// one this project could actually finish and verify. See
// reports/2026-08-25/PERFETTO_PROFILER_INTEGRATION.md for the validated
// end-to-end trace.
//
// Perfetto's TRACE_EVENT_BEGIN/END take a compile-time CATEGORY string,
// not a runtime domain handle like ITT/NVTX -- this project only ever
// uses one category ("ternary_core"), so TERNARY_PROFILE_DOMAIN is kept
// as an unused placeholder for macro-signature compatibility with the
// other two backends, and the category name is hardcoded here rather
// than threaded through from the call site.

#elif defined(TERNARY_ENABLE_PERFETTO)

// gen_amalgamated expanded: #include "third_party/perfetto/perfetto.h"
#include "third_party/perfetto/perfetto.h"

PERFETTO_DEFINE_CATEGORIES(perfetto::Category("ternary_core"));

#define TERNARY_PROFILE_DOMAIN(var_name, domain_name) \
    int var_name = 0

#define TERNARY_PROFILE_TASK_NAME(var_name, task_name) \
    const char* var_name = task_name

#define TERNARY_PROFILE_TASK_BEGIN(domain, handle) \
    TRACE_EVENT_BEGIN("ternary_core", perfetto::DynamicString(handle))

#define TERNARY_PROFILE_TASK_END(domain) \
    TRACE_EVENT_END("ternary_core")

#define TERNARY_PROFILE_FRAME_BEGIN(domain) \
    TRACE_EVENT_BEGIN("ternary_core", "Frame")

#define TERNARY_PROFILE_FRAME_END(domain) \
    TRACE_EVENT_END("ternary_core")

// Start/stop helpers: implemented in ternary_profiler_perfetto.cc (the one
// translation unit linking perfetto.cc). Not part of the VTune/NVTX
// backends' contract (those attach to an already-running external
// profiler instead) -- Perfetto's in-process backend needs the traced
// program itself to start a session and write the resulting trace file.
extern "C" {
// Initializes the in-process backend, registers track events, and starts
// a session writing directly to trace_file_path. Returns true on success.
// Call once, before any TERNARY_PROFILE_* macro use.
bool ternary_profiler_perfetto_start(const char* trace_file_path);
// Stops the session (flushing all pending events to the file opened by
// ternary_profiler_perfetto_start) and closes the file.
void ternary_profiler_perfetto_stop();
}

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

#elif defined(TERNARY_ENABLE_PERFETTO)

// Perfetto's TRACE_EVENT (unscoped, one-argument form -- not
// TRACE_EVENT_BEGIN/END) is documented as "Begin a slice which gets
// automatically closed when going out of scope," so it's a direct RAII
// fit with no wrapper struct needed. Found 2026-08-25: this branch was
// missing entirely when the real Perfetto backend was added a few lines
// above (TERNARY_PROFILE_TASK_BEGIN/END became real, but this guard was
// never extended past VTUNE/NVTX) -- silently fell through to the no-op
// stub below, the same silent-degrade shape this project has repeatedly
// hunted and fixed elsewhere.
#define TERNARY_PROFILE_SCOPE(domain, task_name) \
    TRACE_EVENT("ternary_core", perfetto::DynamicString(task_name))

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
