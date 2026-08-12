# Code Review Session Report — 2026-08-12

**Scope:** Full-codebase bug-hunting pass, requested as an open-ended "keep reviewing
the codebase for bugs" session following an earlier `.claude/` config cleanup in the
same session. Platform: Linux x64 (this repo's only formally-validated Linux
environment; Windows x64 remains the production platform per CLAUDE.md).

**Method:** Alternated between background `code-review` subagents (full-file audits,
not diffs — the repo had no pending diff at any point this session) and manual
review/verification. Every finding below was reproduced or directly verified against
running code before being called a bug — not taken on the reviewer's word. Several
background subagent launches failed mid-run against the session's API rate limit
(reset ~11:20am America/Asuncion) and were continued manually rather than retried.

**Net result:** 19 commits, `a527da0`..`32eada2`, all on `main`, all pushed. Every
commit rebuilt the affected module(s) and re-ran `tests/run_tests.py` (13/13 suites,
up from 7/13 at session start — see "Test suite expansion" below) before landing.

---

## What was reviewed and fixed, by area

### `.claude/` config (session start, before the bug-hunt proper)
Traced ~285 foreign files (business/medtech agent personas, a generic "skill
marketplace" dump, another project's `settings.local.json` complete with that
project's env vars and a hardcoded wrong `SessionStart` banner) to one bulk-import
commit and removed them; `.claude/` now matches exactly what its own `README.md`
documents. Commits `a527da0`, `35a6622`.

### `src/core/` + `src/engine/` (C++ core and Python bindings) — commits `6455a73`
through `6339400`, `b0d98a4`
Read every file in both trees. Highlights:

- **OpenMP fence bug** in `bindings_core_ops.cpp`: a prior "fix" for a memory-visibility
  bug had only patched one of four hot-path functions, and even there placed
  `_mm_sfence()` inside the per-block loop (O(n/32) fences instead of O(nthreads)).
  Fixed all four consistently.
- **Zero-skip GEMM** (`ternary_gemm_zero_skip.cpp`): ~20 unchecked `malloc`/`calloc`
  calls (the caller already expected `nullptr`-on-failure but the callee never
  delivered it), an `int` overflow in nnz-counting for `K*N > INT_MAX`, a serial
  transpose loop parallelized, and a 3x-copy-pasted SAXPY inner loop deduplicated.
- **Duplicate canonical-LUT header**: `ternary_canonical_lut.h` and `ternary_lut_gen.h`
  independently defined the same-named `make_canonical_binary_lut`/`make_canonical_unary_lut`
  with *different* padding semantics — a hard redefinition error whenever both got
  included (only happened in one file, which is why it went unnoticed). Traced both
  to their consumers, confirmed the 9 valid entries were mathematically identical,
  repointed the one real consumer at the tested version, deleted the dead duplicate.
- **Same file's fix surfaced a second bug**: dispatching to a backend that doesn't
  implement an op (by design, e.g. `AVX2_v1` has no fused ops) silently returned
  *uninitialized memory* instead of raising — found in both the 4 fused-op dispatch
  functions and, on a second manual pass, the 5 core dispatch functions too. Both
  fixed the same way: dispatcher returns `bool`, Python binding raises on `false`.
- **Dense243 crash-on-import risk**: a global static object ran real AVX2 intrinsics
  unconditionally at shared-library load time (before the module's own `has_avx2()`
  check could run) — `import ternary_dense243_module` would SIGILL on non-AVX2
  hardware. Fixed with lazy initialization.
- **Broken SIMD trit-extraction path** in the same file: misuse of
  `_mm256_shuffle_epi8` (can only address 16 entries/lane; Dense243 bytes are
  genuine 0–242 values) — confirmed dead code via full call-graph search, marked
  "DO NOT CALL" rather than risk an unverified redesign.
- **`core_api.h` didn't compile standalone**, contradicting its own documented
  purpose ("a single entry point... for direct `#include`") — root cause was a
  missing `<cstddef>` in `ternary_lut_gen.h`, silently relying on pybind11 to
  provide `size_t` transitively in every real build.
- **7 stale include-guard closing comments** across the tree (mechanical, from past
  file renames never propagated to the trailing `#endif // NAME` comment).
- **`optimization_config.h`**: documented a `#define TERNARY_OMP_THRESHOLD ...`
  override mechanism that the code never actually checked for — implemented it.
- **`ternary_profiler.h` vs. CLAUDE.md**: directly contradicted each other on
  whether VTune profiling is "integrated." Reconciled: call sites are real, but no
  build script ever defines `TERNARY_ENABLE_VTUNE`, so it's never been exercised.

### `models/tritnet/` — commits `2a78690`, `6761858`
- **The single most consequential finding of the session**: `tritnet_validate_gemm()`
  (exposed as `ternary_tritnet_gemm.validate()`) compared the naive GEMM kernel
  against *itself*, always returning `0.0` ("perfect correctness") while validating
  nothing. Fixing it to do a real comparison against the AVX2 kernel (which compiles
  into every build but was **never actually called** — `tritnet_gemm_f32_avx2` was
  never declared anywhere `bindings_tritnet_gemm.cpp` could see it) immediately
  surfaced a genuine bug in that AVX2 kernel: a row-stride mistake that silently read
  weights from the wrong output column for any N > 1. Measured error: 4.20 → 0.00
  after the fix. Also wired the AVX2 kernel into the actual `gemm()` Python API
  (previously always ran the ~10-15x slower naive path regardless of what was built)
  and replaced a hardcoded `return 0.0` placeholder in the benchmark function with
  real `std::chrono` timing.
- **The TritNet orchestrator (`run_tritnet.py`) was structurally disconnected from
  the pipeline that actually works**: `--all` only ever called the old MSE-based
  `src/train_tritnet.py`, never `train_phase2a.py`/`train_phase2b.py` (the QAT
  pipeline that achieved the documented Phase 2B GO). Its dataset-generation check
  trusted a stale summary file without checking the actual truth-table files existed
  (they didn't — traced to `generate_truth_tables.py`'s own `PROJECT_ROOT` being one
  `.parent` short, silently writing to `models/models/datasets/tritnet/`). Its
  validation read only legacy history data and would report **NO-GO**, directly
  contradicting CLAUDE.md's documented "Phase 2B GO" — now reads the real
  `phase2b/*/result.json` results and correctly reports GO. Added `--phase2b` so the
  real pipeline is actually reachable from the documented entry point.
- Smaller, all individually verified: Phase 1 checkpoint resume didn't check
  convergence before treating a checkpoint as "done" (mirrored the Phase 2 fix
  already in place, verified with an isolated smoke test); `--hidden-size 0` and
  `threshold=0.0` both silently treated as "unset" (Python falsy-zero bugs);
  `--loss crossentropy` recorded the wrong value in saved metadata; weight export
  silently dropped 3 of 6 layers for Deep-architecture models.
- **Not fixed** (documented, real, out of scope): checkpoint format incompatibility
  between `load_tritnet_model()` and `train_phase2b.py`'s saves — will block Phase 3
  C++ weight export whenever that starts. Duplicated QAT training code between
  `train_phase2a.py`/`train_phase2b.py` (structural refactor, not a bug fix).

### `benchmarks/cpp-native-kernels/` — commit `4ee8608`
This entire directory only ships Windows `.bat` build scripts — no Linux/macOS
equivalent — so it had **never been compiled outside MSVC**. Compiling every `.cpp`
file directly with g++ found: a `clock_t` naming collision with the C standard
library's own `::clock_t` (two separate files), several missing standard-library
includes relying on transitive luck, and a hand-rolled JSON builder whose `Value`
class had no default constructor but was used through `std::map::operator[]` (which
needs one). Fixed all of it; fully linked and ran `benchmark_main.cpp` end-to-end
for the first time, producing valid JSON with 60 real benchmark runs.

### `benchmarks/python-with-interpreter-overhead/` — commits `82504c8`, `f10c99b`, `32eada2`
- `bench_dense243.py` called `td243.encode()`/`.decode()` — methods that don't exist
  (real API: `pack`/`unpack`) — crashed immediately past the first phase. Also a
  hardcoded "2.5x" in the summary verdict directly contradicting the "5.00x" actually
  measured and printed two lines above it in the same output.
- `bench_regression_detect.py`: `--threshold` parsed but never used (hardcoded ±5%
  regardless); division by zero crashed the whole comparison run on a degenerate
  baseline entry.
- `bench_fair_baseline.py`: `--repeats 1` crashed every cell (`statistics.stdev`
  needs ≥2 samples, no minimum enforced).
- `benchmark_framework.py`: unguarded division could crash on legitimately-possible
  0ns timing measurements; a naive Gaussian confidence interval could go negative on
  strictly-positive timing data (not clamped, printed/exported as-is).
- **A systemic off-by-one path bug in 12 files**, found while chasing why
  `bench_simd_fusion_ops.py` couldn't even print `--help`: `PROJECT_ROOT` computed
  one directory level short of the real repo root (same bug class as the
  `generate_truth_tables.py` fix above, found via a systematic repo-wide scan rather
  than one file at a time). This was more dangerous than a simple crash because most
  of these files catch `ImportError` and silently fall back to "mock"/"unavailable"
  mode instead of failing loud — **`bench_competitive.py`, the script behind
  CLAUDE.md's documented competitive-benchmark numbers, was one of the 12** and was
  silently computing with `(a+b)%3` mock arithmetic instead of the real engine under
  a clean environment.

---

## Test suite expansion (commit `73f280c`, part of the `src/`/`src/engine` pass)

`tests/run_tests.py` (and therefore CI) only ever wired 7 of ~17 real test files
under `tests/python/` — including `quality_gates.py`, whose own header says it "must
pass before any release," and `test_backend_integration.py`, the largest Python test
file. Triaged every un-wired file:

- **6 wired in** (all now pass): `test_canonical_lut.py`, `test_simd_validation.py`,
  `test_fusion_correctness.py`, `test_backend_integration.py`, `test_fused_op_bug.py`,
  `quality_gates.py`.
- **2 removed as genuinely dead**: `test_simd_python.py` (truncated mid-docstring,
  never valid Python), `test_path_fixes.py` (broken path math, referenced a
  `build_fusion.py`/`ternary_fusion_engine` that no longer exist).
- **1 correctly left un-wired**: `test_dual_shuffle_validation.py` fails because it
  tests a feature its own file labels "future enhancement," not a bug.

`tests/run_tests.py` now runs **13/13 suites**, up from 7.

---

## Important methodological note for future sessions: the `PYTHONPATH` trap

A leftover `PYTHONPATH=.:./api` (contamination from an unrelated project, the same
root cause already cleaned out of `.claude/settings.local.json` earlier this
session) was still set in the dev shell used for most of this session's manual
testing. It happened to put the repo root on `sys.path` via a side channel, **masking
the exact class of bug this session spent the most time finding** — several "verified
working" claims made earlier in the session (before the masking was discovered) had
to be re-checked with `env -u PYTHONPATH` once the pattern became clear, and at least
one (`bench_dense243.py`) turned out to have been masked too, though its actual fix
still held up under a clean re-check.

**For the next session**: run bare `python3` (not through this specific dev shell) or
explicitly `env -u PYTHONPATH` when verifying any benchmark/tooling script, unless
`PYTHONPATH` is intentionally set for a specific reason. Don't trust a script "working"
in this shell as proof its own path-resolution code is correct.

---

## What's pending for the next session

Explicitly **not yet reviewed** (ran out of session time/budget, not because
anything looked suspicious going in):

- `benchmarks/utils/` — 4 of 6 files unreviewed (`benchmark_validator.py` — appears
  completely unused, worth checking if it's dead code; `geometric_metrics.py`,
  `visualization.py`, `windows_power.py`). `hardware_metrics.py` and
  `system_load_monitor.py` were spot-checked (former has an honestly-placeholder
  `_estimate_ipc()` that always returns a hardcoded `3.0`, correctly documented as
  such — not fixed, matches the project's other honest-placeholder patterns).
- `benchmarks/macro/` (2 files + README + RESULTS.md) — not reviewed at all.
- `research/` and `opentimestamps/` — not reviewed at all this session.
- The **path-computation bug hunt is very likely not exhaustive**. The scan covered
  `PROJECT_ROOT = Path(__file__).parent...` and `os.path.dirname(os.path.dirname(...))`
  chains specifically; other idioms (e.g. `Path(__file__).parents[N]`, manual string
  path joins, `os.chdir`-relative assumptions) were spot-checked but not
  systematically swept. Worth a second pass, especially over `research/` and
  `opentimestamps/` which haven't been touched yet.

**Known, documented, deliberately-not-fixed** (real issues, out of scope for a
bug-fixing pass, need a design decision or larger refactor):

- `tritnet_gemm_f32_avx2_tiled` (in `models/tritnet/gemm/tritnet_gemm_avx2.cpp`) has
  the same row-stride bug class just fixed in the reachable AVX2 kernel, but is
  itself unreachable (not declared anywhere, not called from anywhere in this repo)
  — left with a detailed warning comment rather than an unverifiable fix.
- Checkpoint format incompatibility between `tritnet_model.py`'s `load_tritnet_model()`
  and `train_phase2b.py`'s raw `state_dict()` saves — will block Phase 3 C++ weight
  export whenever that work starts.
- Duplicated QAT training code between `train_phase2a.py`/`train_phase2b.py` (no
  shared module) — real technical debt, demonstrated concretely by needing to
  re-apply the same checkpoint-resume fix in two places this session.
- `BenchmarkRunner` (`benchmark_framework.py`) is imported nowhere except itself and
  a deprecated file — `bench_fair_baseline.py`, `bench_simd_core_ops.py`, and
  `bench_simd_fusion_ops.py` each hand-roll their own statistics instead of reusing
  it, with inconsistent rigor between them.
- **Whether any already-recorded benchmark report** (e.g. the numbers in
  `reports/2026-08-11/LINUX_VALIDATION_REPORT.md` or CLAUDE.md's competitive-benchmark
  claims) was produced under the mock-fallback bug described above is **unknown** —
  this session found and fixed the bug but has no way to audit the environment a past
  run used. Given the honest-claims policy this project holds itself to, the
  competitive suite is worth a fresh re-run now that the path bug is fixed, to
  confirm the existing numbers reproduce under a known-clean environment.

**`benchmarks/deprecated/`** — intentionally out of scope throughout this session
(5 files there have the identical path bug; not fixed, matches this session's
consistent policy of not investing in archived/deprecated code).

---

## Verification discipline used throughout

Every fix in this session was checked one of these ways before being committed,
not just read and assumed correct:
- Rebuilt the affected compiled module and re-ran `tests/run_tests.py`
- Reproduced the original bug first (crash, wrong output, or wrong exit code), then
  confirmed the fix resolves that exact reproduction
- For numeric/algorithmic fixes: compared against a NumPy or hand-computed reference
  across multiple input shapes, not just the one case that happened to be reported
- For anything involving `PROJECT_ROOT`/path resolution: re-verified with
  `env -u PYTHONPATH` once that masking pattern was understood
