# Code Review — This Session's New Code, 2026-08-25

**Scope:** User request ("review this session's own new code") after a
long run of changes (GEMM `MB=8` tuning, Perfetto profiler integration,
TinyLlama quantization pipeline) had accumulated without an independent
review pass — unlike almost everything else in this project's history.
Ran `/code-review` at high effort against `d97099d~1..HEAD` (the full
12-commit range for this session). All 5 findings verified by direct
reproduction before fixing anything, per this project's own discipline.

## Findings

1. **CONFIRMED, fixed** — `TERNARY_PROFILE_SCOPE`'s real RAII
   implementation was gated on `TERNARY_ENABLE_VTUNE || TERNARY_ENABLE_NVTX`
   only; the guard was never extended when the real Perfetto backend was
   added a few lines above in the same file. Code built with
   `-DTERNARY_ENABLE_PERFETTO` using `TERNARY_PROFILE_SCOPE` (documented
   in this same header's own usage example) silently fell through to the
   no-op stub and recorded zero events — the same silent-degrade shape
   this project has repeatedly hunted and fixed elsewhere. Fixed with a
   dedicated `#elif defined(TERNARY_ENABLE_PERFETTO)` branch using
   Perfetto's `TRACE_EVENT` macro, which is itself documented as
   "automatically closed when going out of scope" — a direct RAII fit,
   no wrapper struct needed. **Verified, not just recompiled**: wrote a
   small standalone program using `TERNARY_PROFILE_SCOPE` around a
   5ms-sleep block, called it 3 times, queried the resulting trace —
   3 slices, avg duration 5.15ms, exactly matching the sleep.

2. **CONFIRMED, fixed** — `bench_perfetto_trace.cpp`'s own documented
   manual compile command used `-I../../third_party` instead of
   `-I../..` (repo root). Reproduced verbatim: running the command
   exactly as written fails with `fatal error: third_party/perfetto/
   perfetto.h: No such file or directory` (the header includes that path
   relative to repo root, not to `third_party/` itself). Fixed the
   documented command; re-verified it now compiles clean.

3. **CONFIRMED, fixed** — doc comments in both `ternary_gemm_dense.h`
   and the mirrored section header in `ternary_gemm_dense.cpp` still
   said the AVX2 kernel is "M-blocked in groups of 4" / "(4 rows/tile)",
   left over from before commit `33385bd` changed
   `TERNARY_GEMM_DENSE_MB` to 8 to fix the batch=128 regression. A
   maintainer trusting the stale doc could reintroduce that exact
   regression by "restoring" `MB=4`. Fixed both comments to reference
   the macro rather than hardcode a number, so this can't drift again
   the same way.

4. **CONFIRMED, fixed** — `ternary_profiler.h`'s own top-of-file
   "IMPLEMENTATION STATUS" banner still described Perfetto as "Stub
   placeholder for future web-based tracing" / "[Stub only]",
   unchanged from before this session even though the real backend was
   implemented later in the same file. This file's own history already
   shows one correction (2026-08-12) for *overclaiming* VTune's status;
   this is the same class of bug in the opposite direction —
   *underclaiming* what's now real. Fixed the banner and the
   "PROFILER TARGETS" list to reflect the real, verified state (and
   tightened VTune's own claim from "[INTEGRATED]" to "[Macros
   implemented, unbuilt/unverified]" while in there, matching what the
   surrounding paragraph already says more precisely).

5. **PLAUSIBLE, no change needed** — adding `--force` to `build.py`'s
   `build_ext` invocation (the fix for the stale-rebuild bug found while
   adding `--enable-perfetto`) means every invocation of the script now
   always fully rebuilds, not just when flags/sources actually changed.
   Considered a more surgical fix (tracking the previous build's flags
   via a marker file) and decided against it: the always-force approach
   is the *safer* direction — the alternative risks a stale
   Perfetto-enabled object silently linking into a nominally-default
   build if the flag is toggled off between runs, a worse failure mode
   than a slightly slower rebuild. The actual cost is small: `perfetto.cc`
   is only ever added to `sources` when `--enable-perfetto` is passed, so
   default builds never pay to recompile the large vendored SDK — the
   only added cost is always recompiling the one ~30KB
   `bindings_core_ops.cpp` file, a few seconds, not a meaningful burden
   given this project's own "correctness over speculative optimization"
   principle (YAGNI: don't add complexity without a measured need).

## Outcome

4 of 5 findings fixed and re-verified by actual execution (not just
recompilation) where applicable; 1 reviewed and consciously left as-is
with the reasoning documented. `tests/run_tests.py`: 16/16 throughout.
Default build re-confirmed unaffected (`has_perfetto == False`).
