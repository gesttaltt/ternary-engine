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
