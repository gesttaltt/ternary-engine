# Perfetto Profiler Integration — 2026-08-25

**Scope:** Closes part of Critical Gap #10 (profiler integration). User
recommendation, after dropping an unproductive GPU-access angle (two
Remote Control peer sessions checked multiple times, both stayed
offline): re-examine gap #10, since CLAUDE.md lumped VTune/NVTX/Perfetto
together as "unbuilt" without distinguishing that they have different
blockers.

## 1. Why Perfetto specifically

- **VTune (ITT API)** — macros already implemented, but needs Intel's
  proprietary VTune Profiler application to build against meaningfully
  and to open/verify the resulting trace. Not installed here, and
  installing it wasn't attempted (license/tooling weight, not a code
  problem).
- **NVTX** — needs a CUDA GPU. This sandbox has none (confirmed the same
  session that dropped the GPU-access angle: no `nvidia-smi`, `torch.cuda
  .is_available() == False`).
- **Perfetto** — a genuinely open-source SDK with no special hardware or
  license requirement. Verified concretely before committing to this:
  `google/perfetto`'s GitHub releases (`v58.2`) publish a self-contained
  `perfetto-cpp-sdk-src.zip` (~1.2MB), and this sandbox has working
  network access to fetch it. This is the one backend that could
  actually be built **and verified** here, not just written.

## 2. What was vendored and built

- `third_party/perfetto/perfetto.h` + `perfetto.cc` — Google's official
  amalgamated (single-header + single-source) SDK build, fetched
  directly from the `v58.2` release, Apache 2.0 licensed (same license
  as this project). See `third_party/perfetto/README.md` for full
  provenance and update instructions. Vendored rather than fetched at
  build time, matching this project's preference for builds that don't
  depend on network access.
- `src/core/profiling/ternary_profiler.h` — the previous Perfetto branch
  was a stub ("ROADMAP: Perfetto SDK integration planned for future
  release... compiles but no output"). Replaced with a real
  implementation: `TERNARY_PROFILE_TASK_BEGIN`/`END` map to Perfetto's
  `TRACE_EVENT_BEGIN`/`END` macros against one compile-time category
  (`"ternary_core"`) — this project only ever needs one, so the
  existing `TERNARY_PROFILE_DOMAIN` parameter is kept only for
  macro-signature compatibility with the VTune/NVTX branches, unused at
  the Perfetto call site. Runtime task names go through
  `perfetto::DynamicString` (verified this class exists in the vendored
  SDK version before using it) since they're `const char*` variables,
  not compile-time literals.
- `src/core/profiling/ternary_profiler_perfetto.cc` — the one
  translation unit Perfetto's own quickstart guide requires to define
  `PERFETTO_TRACK_EVENT_STATIC_STORAGE()`, plus
  `ternary_profiler_perfetto_start()`/`_stop()` helpers (Perfetto's
  in-process backend needs the traced program itself to start a session
  and point it at an output file — unlike VTune/NVTX, which attach to an
  already-running external profiler).
- `benchmarks/cpp-native-kernels/bench_perfetto_trace.cpp` — a
  pybind11-free native demo, per this project's `ffi_isolation`
  convention. Replicates `bindings_core_ops.cpp`'s *exact* profiling
  pattern (same domain, same three task names — `OpenMP_Parallel`,
  `Serial_SIMD`, `Scalar_Tail` — same OMP-threshold branching) around
  the real AVX2 `tadd_simd` kernel (`simd_avx2_32trit_ops.h`), so the
  resulting trace reflects genuine hot-path structure, not a synthetic
  sleep-loop stand-in.
- `build/build_perfetto_demo.py` — builds the demo with
  `-DTERNARY_ENABLE_PERFETTO` linking `perfetto.cc` +
  `ternary_profiler_perfetto.cc`, then runs it once, producing a real
  `.perfetto-trace` file. This is the actual "wire it into a build
  script" deliverable gap #10 was missing for all three backends.

## 3. Verifying the trace is real, not just "compiles"

Compiling with `TERNARY_ENABLE_PERFETTO` produced warnings only (GCC
static-analysis false positives from aggressive inlining through
Perfetto's category-registration path, and one harmless `#pragma
system_header` notice from including `perfetto.cc` as a translation
unit rather than a header) — no errors, clean link.

Running the demo produces a real trace file. To confirm it's genuinely
correct (not just non-empty), fetched Perfetto's own
`trace_processor_shell` (from the same `v58.2` release, not vendored —
it's a ~14MB standalone tool, only needed for this verification step,
not for building or using the integration) and queried the actual
slice data:

```
"name","n","total_ns"
"Serial_SIMD",10,45486
"Scalar_Tail",10,1836
"OpenMP_Parallel",10,1367332
```

This is internally consistent with the demo's own code, which is what
makes it a real verification and not just "a file exists":

- The demo runs 4 array sizes × 5 repetitions = 20 total calls to the
  traced function.
- Sizes 1,000 and 50,000 are below `OMP_THRESHOLD` (262,144) → take the
  `Serial_SIMD` path. **10 slices** (2 sizes × 5 reps) — matches exactly.
- Sizes 500,000 and 2,000,000 are above the threshold → take the
  `OpenMP_Parallel` path. **10 slices** — matches exactly.
- `Scalar_Tail` only fires when array length isn't a multiple of 32:
  1,000 % 32 ≠ 0 and 50,000 % 32 ≠ 0 (both trigger a tail), while
  500,000 % 32 = 0 and 2,000,000 % 32 = 0 (neither does). **10 tail
  slices, correctly correlated with the two below-threshold sizes only**
  — this is the detail that rules out a coincidental match: if the trace
  were fake or the categorization were wrong, this exact correlation
  wouldn't hold.
- Durations are in a sane, differentiated range: `OpenMP_Parallel`
  averages ~137µs/call (parallel work on the two largest arrays),
  `Serial_SIMD` ~4.5µs/call (small-array serial SIMD), `Scalar_Tail`
  ~184ns/call (a handful of leftover elements) — the relative ordering
  matches what the code actually does.

## 4. What this does and doesn't claim

- Does NOT claim VTune or NVTX are now viable — those remain blocked by
  the same hardware/license constraints as before; nothing about this
  session changes that.
- Does claim: this project now has its **first real, verified profiler
  trace ever** — not a stub, not "should work," an actual trace file
  whose contents were checked against the code that produced it and
  found to be exactly correct. `TERNARY_ENABLE_VTUNE`/`_NVTX` remain
  unbuilt (as documented); `TERNARY_ENABLE_PERFETTO` no longer is.

## 5. Same-day follow-up: wiring it into the main pybind11 module build

Direct follow-up (user: "wire it into the main module build"). Added
`build/build.py --enable-perfetto`: adds `-DTERNARY_ENABLE_PERFETTO` and
the two extra sources (`third_party/perfetto/perfetto.cc`,
`src/core/profiling/ternary_profiler_perfetto.cc`) to the
`ternary_simd_engine` extension when passed; default builds are
completely unaffected (verified: `has_perfetto` attribute is `False` and
`tests/run_tests.py` is 16/16 on the plain `python build/build.py`
build). Added `perfetto_start(trace_path)`/`perfetto_stop()` Python
bindings in `bindings_core_ops.cpp` so a user can actually drive a
tracing session around real `tadd`/`tmul`/etc. calls, plus a
`has_perfetto` capability flag matching the existing `has_avx2`
convention.

**A real, pre-existing latent bug surfaced immediately**: running the
default build followed immediately by `--enable-perfetto` silently
reused the *first* build's `.so`, unchanged — `has_perfetto` stayed
`False`. Root cause, confirmed by inspecting `build/temp.linux-x86_64-
cpython-312/` directly: distutils' `build_ext --inplace` doesn't just
skip individually-stale `.o` files, it skips the **entire** extension
rebuild whenever the existing in-place `.so` already looks newer than
every one of its declared source files — it has no way to know that the
sources *list* or compiler *flags* changed between two invocations, only
file mtimes. Since the newly-added `perfetto.cc`/
`ternary_profiler_perfetto.cc` files' mtimes predated the `.so` that had
just been built moments earlier, distutils considered the whole
extension "up to date" and skipped compiling anything at all -- no
`perfetto.o` or `ternary_profiler_perfetto.o` were ever produced.

This isn't specific to the Perfetto flag -- it was a latent bug in
`build.py` from before this session, just never visible because compiler
flags and the sources list essentially never changed between two
consecutive runs of this script without also editing
`bindings_core_ops.cpp` (which *does* update its own mtime and force a
partial recompile, just not necessarily a full relink-from-scratch of
every source). Fixed by adding `--force` to the `build_ext` invocation --
this script always intends a full rebuild, never incremental reuse, so
forcing that to be literally true regardless of timestamps is the
correct fix for both the Perfetto case and the general one.

**Verified after the fix**, clean `build/temp.../` each time:

- `python build/build.py --enable-perfetto`: `perfetto.o` and
  `ternary_profiler_perfetto.o` both genuinely compiled; resulting `.so`
  jumps from 3,714.9 KB (default) to 55,714.8 KB (unstripped, full
  Perfetto/protobuf code linked in -- expected for a debug-style build,
  not optimized for distribution size). `tc.has_perfetto == True`.
- Drove a real tracing session **through the actual Python module**
  (not just the native demo): `tc.perfetto_start(path)`, several
  `tc.tadd`/`tc.tmul` calls at sizes crossing the OMP threshold and the
  32-element tail boundary, `tc.perfetto_stop()`. Queried the resulting
  trace with `trace_processor_shell` the same way as the native demo:
  5 `OpenMP_Parallel` slices (the 5 `tadd` calls at n=2,000,000, above
  threshold), 5 `Serial_SIMD` + 5 `Scalar_Tail` slices (the 5 `tmul`
  calls at n=1,000, below threshold and not a multiple of 32) -- exactly
  matching the call pattern, the same style of internally-consistent
  verification as the native demo, now proven through the real
  pybind11-facing module.
- `tests/run_tests.py` 16/16 with the Perfetto-enabled `.so` active --
  confirms enabling profiling doesn't change any operation's correctness.
- Rebuilt the plain default `.so` afterward and re-confirmed
  `has_perfetto == False` and 16/16 again, restoring the repo's normal
  build state.

## Files changed (this session, both parts)

- `third_party/perfetto/{perfetto.h,perfetto.cc,README.md}` (new,
  vendored)
- `src/core/profiling/ternary_profiler.h` (real Perfetto backend,
  replacing the stub)
- `src/core/profiling/ternary_profiler_perfetto.cc` (new)
- `benchmarks/cpp-native-kernels/bench_perfetto_trace.cpp` (new)
- `build/build_perfetto_demo.py` (new)
- `src/engine/bindings_core_ops.cpp` (§5: `perfetto_start`/`_stop`/
  `has_perfetto` bindings)
- `build/build.py` (§5: `--enable-perfetto` flag; `--force` fix for the
  stale-rebuild bug found while adding it)

`tests/run_tests.py`: 16/16 throughout (none of these files touch the
core kernel's default no-op build path — verified separately by
rebuilding the standard module via `build/build.py` with no profiler
macro defined, exit 0, all 16 suites still pass).
