// ternary_profiler_perfetto.cc — Perfetto backend implementation
//
// Copyright (c) 2025 Jonathan Verdun (Ternary Engine Project)
// Licensed under the Apache License, Version 2.0
//
// This is the one translation unit that:
//   (a) defines PERFETTO_TRACK_EVENT_STATIC_STORAGE(), as required by
//       exactly one .cc file per the Perfetto SDK's own quickstart guide
//       (see third_party/perfetto/perfetto.h's "Quickstart guide" comment
//       block), and
//   (b) implements the ternary_profiler_perfetto_start()/_stop() helpers
//       declared in ternary_profiler.h.
//
// Only compiled when TERNARY_ENABLE_PERFETTO is defined -- this file must
// be added to any build target that also compiles third_party/perfetto/
// perfetto.cc and links whatever bindings_core_ops.cpp / a native
// benchmark uses ternary_profiler.h's TERNARY_PROFILE_* macros.

#include "ternary_profiler.h"

#ifdef TERNARY_ENABLE_PERFETTO

#include <cstdio>
#include <memory>

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

namespace {
std::unique_ptr<perfetto::TracingSession> g_session;
FILE* g_trace_file = nullptr;
}  // namespace

extern "C" bool ternary_profiler_perfetto_start(const char* trace_file_path) {
    if (g_session) {
        // Already started -- Perfetto only supports one in-process
        // Tracing::Initialize() call per process; refuse a second start
        // rather than silently ignoring it or corrupting the first session.
        return false;
    }

    if (!perfetto::Tracing::IsInitialized()) {
        perfetto::TracingInitArgs args;
        args.backends = perfetto::kInProcessBackend;
        perfetto::Tracing::Initialize(args);
        perfetto::TrackEvent::Register();
    }

    g_trace_file = fopen(trace_file_path, "wb");
    if (!g_trace_file) return false;

    perfetto::TraceConfig cfg;
    cfg.add_buffers()->set_size_kb(4096);
    auto* ds_cfg = cfg.add_data_sources()->mutable_config();
    ds_cfg->set_name("track_event");

    g_session = perfetto::Tracing::NewTrace();
    g_session->Setup(cfg, fileno(g_trace_file));
    g_session->StartBlocking();
    return true;
}

extern "C" void ternary_profiler_perfetto_stop() {
    if (!g_session) return;
    g_session->StopBlocking();
    g_session.reset();
    if (g_trace_file) {
        fclose(g_trace_file);
        g_trace_file = nullptr;
    }
}

#endif  // TERNARY_ENABLE_PERFETTO
