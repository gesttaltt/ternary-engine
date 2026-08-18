# Gap #6 scoped: code duplication in src/engine/

**Date:** 2026-08-18
**Request:** "scope gap #6 for code duplication between engines" -- CLAUDE.md's gap #6 has read
"Code duplication - Between engines, needs refactoring" with no further detail since the doc's
initial 2025-11-23 version. This report replaces that one-liner with concrete evidence.
**Status:** Scoping only -- nothing changed. Three real, independently-verifiable clusters found,
all in `src/engine/` (the pybind11 binding layer, i.e. the "engines" -- `ternary_simd_engine`,
`ternary_backend`, `ternary_zero_skip_gemm`, `ternary_tritnet_gemm`, `ternary_dense243_module`).
`src/core/` (the kernel) was not part of this pass -- it was already reviewed for correctness bugs
in the 2026-08-14 session; this is a duplication-specific pass over the binding layer only.

## Method

Read every `.cpp` file in `src/engine/` (6 files, 2,265 lines total) and compared function
signatures, validation logic shape, and dispatch-wrapper shape across and within files. Two files
turned out to already correctly follow this project's own `template_unification` convention
(`bindings_dense243.cpp`'s `binary_op_dense243<ScalarOp>`/`unary_op_dense243<ScalarOp>`, and
`bindings_tritnet_inference.cpp`'s shared `validate_chunks()`/`make_output()` helpers) -- confirmed
clean, not part of the findings below. Three files were not.

## Cluster A: uint8_t/int8_t template duplication (`bindings_core_ops.cpp`, within one file)

`process_binary_array<Sanitize>` (lines 141-234, 94 lines) and `process_binary_array_int8<Sanitize>`
(lines 385-465, 81 lines) are near-byte-identical: same OMP-threshold branch, same guided
scheduling, same prefetch distance, same streaming-store alignment check, same per-thread sfence
fix, same scalar tail -- differing **only** in element type (`uint8_t` vs `int8_t`). Same story for
`process_unary_array`/`process_unary_array_int8` (83 and 74 lines). Total: **332 lines**, of which
roughly **150 lines are eliminable** by making the type a template parameter
(`template<typename T, bool Sanitize>`) instead of hand-copying the whole function body for a
second element type.

This is the exact "Bad" pattern CLAUDE.md's own C++ Style section warns against by name
(`process_binary_array_sanitized()`/`process_binary_array_fast()` as the "Bad: Code duplication"
example) -- except the actual duplication that shipped is dtype-keyed rather than
sanitize-flag-keyed, which the doc's own example doesn't cover, likely why it wasn't caught by the
convention it's demonstrating.

**Fix shape:** `template<typename T, bool Sanitize, typename SimdOp, typename ScalarOp>
py::array_t<T> process_binary_array(...)`, instantiated as `process_binary_array<uint8_t, SANITIZE>`
and `process_binary_array<int8_t, SANITIZE>`. The SIMD load/store intrinsics
(`_mm256_loadu_si256` etc.) are type-agnostic at the byte level already (both are 8-bit element
types packed 32-wide), so this should be a mechanical template-parameterization, not a rewrite --
but needs a real benchmark before/after per this doc's own "Modifying Hot Paths" rule, since this
*is* one of the three files that rule explicitly names.

## Cluster B: hand-copied dispatch wrappers (`bindings_backend_api.cpp`, within one file)

9 functions (`dispatch_tnot`, `dispatch_tadd`, `dispatch_tmul`, `dispatch_tmax`, `dispatch_tmin`,
`dispatch_fused_tnot_tadd/tmul/tmin/tmax`, lines 142-333, ~190 lines) share one skeleton: unchecked
array access -> size-match check -> allocate output -> call a `ternary_dispatch_*` function pointer
-> throw a runtime_error naming the op if it returns false -> return. The only things that differ
between any two of these functions are which dispatch function to call and the op name embedded in
two strings.

This is the same shape `bindings_core_ops.cpp`'s `process_binary_array<Sanitize>(A, B, simd_op,
scalar_op)` and `bindings_dense243.cpp`'s `binary_op_dense243<ScalarOp>(A, B, scalar_op, op_name)`
already solve correctly elsewhere in this same directory -- `bindings_backend_api.cpp` (added
2026-08-12, per CLAUDE.md's own history, as the newer pluggable-backend module) didn't reuse or
follow that established pattern when it was written. Roughly **140 of the ~190 lines are
eliminable**.

**Fix shape:** two small templates,
`dispatch_binary(name, dispatch_fn, A, B)` / `dispatch_unary(name, dispatch_fn, A)`, taking the
`ternary_dispatch_*` function pointer and op name as parameters; each of the 9 current functions
becomes a 1-line wrapper, mirroring `bindings_dense243.cpp`'s `tadd_dense243()`-style one-liners.

## Cluster C: matrix-validation boilerplate (`bindings_zero_skip_gemm.cpp` +
`bindings_tritnet_gemm.cpp`, both within and across files)

"2-D, correct dtype, C-contiguous" checks for GEMM input matrices are hand-written at **5 separate
call sites** in `bindings_zero_skip_gemm.cpp` alone (`ZeroSkipWeights` constructor, `gemm()`,
`gemm_tiled()`, `py_gemm()`, `py_sparsity_info()`) and again, independently, at 2 more sites in
`bindings_tritnet_gemm.cpp` (`py_gemm()`, `py_gemm_scaled()`). None of the 7 sites call a shared
helper; each hand-rolls `buf.ndim != 2`, a dtype/shape check, and
`.attr("flags").attr("c_contiguous").cast<bool>()` from scratch.

Concrete, already-visible cost of the duplication: the error message wording has already drifted
between copies of the *same* check -- `bindings_zero_skip_gemm.cpp`'s class methods say `"A must be
2-D [M, K] float32 array"` while its own `py_gemm()` (30 lines below) says `"A must be 2-D [M, K]
float32"` (no trailing "array") for the identical condition. Small on its own, but it's the same
"copy-paste-without-a-shared-function lets independent instances quietly diverge" pattern this
project has already found and fixed repeatedly in Python scripts across this review chain (the
`sys.path` off-by-one class, the inverted-valuation bug, etc.) -- this is that same failure shape
showing up in C++ input validation instead. Roughly **60-80 lines** of near-identical validation
code across the two files.

**Fix shape:** a small shared header (e.g. `src/engine/lib/py_array_validate.h`, matching this
project's existing `src/engine/lib/dense243/` convention for shared binding-layer code) with
something like `validate_2d_contiguous(py::buffer_info, const char* name, ssize_t expect_dim0,
ssize_t expect_dim1)` -- or, given each file has its own error-message vocabulary, at minimum a
single per-file helper so a future validation fix only needs to happen once per file, not at 5 or 7
sites. Cross-file sharing (`zero_skip_gemm` + `tritnet_gemm`) is a smaller win than the within-file
consolidation and worth doing only if it doesn't blur the two engines' otherwise-independent APIs.

## What was checked and found clean

- `bindings_dense243.cpp`: `binary_op_dense243`/`unary_op_dense243` templates correctly used by all
  5 ops, no duplication.
- `bindings_tritnet_inference.cpp`: `validate_chunks()`/`make_output()` shared correctly across
  `tnot`/`tadd`/`tmul`/`tmin`/`tmax`.
- `has_avx2()` itself: single implementation in `core/simd/cpu_simd_capability.h`, correctly
  reused by every file that needs it (`bindings_tritnet_inference.cpp`,
  `bindings_tritnet_gemm.cpp`, `bindings_core_ops.cpp`) -- not a duplication instance.
- No duplication found *between* `bindings_zero_skip_gemm.cpp` and `bindings_tritnet_gemm.cpp`'s
  actual GEMM kernels or dispatch logic -- they're genuinely different algorithms (zero-skip CSC/CSR
  sparse vs. Dense243-packed dense), only their input-validation boilerplate overlaps (Cluster C).

## Recommended priority, if the fixes are wanted

1. **Cluster B** (backend dispatch wrappers) -- highest value-to-risk ratio: doesn't touch any hot
   SIMD path, straightforward mechanical refactor, ~140 lines removed, and it's the one place this
   project's own established `process_binary_array`-style pattern was available and simply wasn't
   reused.
2. **Cluster C** (GEMM validation) -- low risk (pure validation code, not hot-path compute), fixes
   a real (if minor) UX inconsistency (drifted error wording) as a side effect.
3. **Cluster A** (uint8/int8 templates) -- highest potential value (~150 lines, and it's literally
   the file CLAUDE.md's own docs use as the positive template-unification example) but also the
   only one touching a file this doc's "Modifying Hot Paths" rule explicitly names
   (`src/engine/bindings_core_ops.cpp`) -- needs a real before/after benchmark, not just a
   recompile-and-test-pass, before landing.

None of these three clusters were fixed in this session -- this was a scoping pass only, per the
request.

## Update (same day): Cluster B fixed

Per a same-day follow-up request ("go ahead with cluster B"), Cluster B was implemented.

`bindings_backend_api.cpp`'s 9 `dispatch_*` functions were replaced with two shared helpers,
`dispatch_unary(op_name, fn, fail_msg, A)` and `dispatch_binary(op_name, fn, fail_msg, A, B)`,
taking the `ternary_dispatch_*` function pointer and the failure message as parameters. The fused
ops' distinct failure message ("no active backend, or active backend does not implement this
fused op", vs. the core ops' "no active backend (call init()/set_backend() first)") is preserved
exactly via two separate `static const char*` constants passed in at each call site, rather than
being merged into one generic message -- the whole point of this fix is removing accidental
duplication, not removing an intentional distinction.

**Result:** `bindings_backend_api.cpp` dropped from 428 to 328 lines (net -100; the dispatch
section itself: -146 duplicated lines replaced by +46 shared-helper lines, matching this report's
original ~140-line estimate almost exactly).

**Verification performed:**
- Rebuilt via `build/build_backend.py` -- succeeds, no new compiler warnings introduced by this
  change (the build's existing warnings are pre-existing, in unrelated files).
- Ran all 9 dispatch functions against known trit-encoding semantics, cross-checked against NumPy
  (`tadd`/`tmul`/`tmax`/`tmin`/`tnot` + all 4 fused ops).
- Verified `fused_tnot_X(a,b) == tnot(X(a,b))` for all 4 fused ops (the ops must still compose the
  same way after the refactor, not just individually match some expected value).
- Verified both error paths byte-for-byte: `ValueError` size-mismatch messages (e.g. `"tadd: array
  size mismatch"`, `"fused_tnot_tmax: array size mismatch"`) and `RuntimeError` no-active-backend
  messages for both a core op and a fused op, confirming the two different failure-message strings
  are each still attached to the right op family.
- `tests/run_tests.py`: 15/15 (includes `test_backend_integration.py`, this module's dedicated
  suite).

Cluster A remains open, and needs a real before/after benchmark before landing, per this doc's own
"Modifying Hot Paths" rule (it's a hot SIMD path).

## Update (same day): Cluster C fixed

Per a further same-day follow-up ("go ahead with cluster C"), Cluster C was implemented.

New shared header `src/engine/py_array_validate.h` factors out `validate_2d_contiguous<T>()` and
`validate_1d_contiguous<T>()` -- the part genuinely identical across both GEMM engines
(dimensionality + C-contiguity checks). Each call site keeps its own domain-specific piece that
was never actually duplicated: `bindings_tritnet_gemm.cpp`'s exact-shape check against
caller-supplied M/N/K (factored into a local `check_exact_shape()` helper within that file, since
`bindings_zero_skip_gemm.cpp` has no equivalent -- it derives M/N/K from the array shape instead).

**Wording harmonized, not just deduplicated**: the drifted "float32 array" vs. "float32" wording
noted in the original scoping is now one consistent style project-wide ("`<name>` must be
`<shape>`", no redundant trailing "array"/"uint8" word repetition). `bindings_tritnet_gemm.cpp`'s
combined ndim+exact-shape checks (previously one generic message covering both failure modes) now
report specific, more actionable messages for each failure independently.

**A real latent bug found and fixed along the way**: `bindings_zero_skip_gemm.cpp`'s
`py_sparsity_info()` was the one call site of the 5 in that file that never checked C-contiguity,
despite indexing its buffer with flat linear arithmetic (`data[i]` for `i` in `[0, K*N)`) that
silently assumes row-major layout. A non-contiguous input (e.g. a transposed view) would have
silently produced wrong sparsity statistics instead of raising a clear error -- not merely "less
validated than its siblings," a genuine correctness gap the consolidation surfaced by making every
site go through the same helper.

**Test coverage added**: neither `test_zero_skip_gemm.py` nor `test_tritnet_gemm_integration.py`
exercised any validation-failure path before this change -- both files' only coverage was the
happy path. Added `test_input_validation()` to each (6 and 7 malformed-input cases respectively:
wrong ndim/shape, non-contiguous via `np.asfortranarray`/strided slicing, and for zero-skip-gemm
specifically the `sparsity_info()` fix itself), wired into each file's existing test list.

**Result:** `bindings_zero_skip_gemm.cpp` + `bindings_tritnet_gemm.cpp` combined: -82/+53 lines
(net -29), plus the new 96-line shared header (mostly documentation -- the two template functions
themselves are ~20 lines of actual logic).

**Verification performed:**
- Rebuilt both modules (`build/build_zero_skip_gemm.py`, `build/build_tritnet_gemm.py`) -- both
  succeed, no new compiler warnings; each build script's own built-in correctness check
  (AVX2/scalar/tiled vs. reference for zero-skip-gemm; max-error-vs-naive for tritnet-gemm) still
  passes with the same near-zero error as before.
- `test_zero_skip_gemm.py` and `test_tritnet_gemm_integration.py` run directly: all pre-existing
  correctness tests still pass (unaffected -- happy-path behavior unchanged), plus the new
  validation tests (6/6 and 7/7 malformed-input cases correctly rejected).
- `tests/run_tests.py`: 15/15.

Cluster A remains the only open item -- see above for why it's intentionally deferred.
