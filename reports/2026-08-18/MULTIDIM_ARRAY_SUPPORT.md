# Multi-dimensional array support (bindings_core_ops.cpp)

**Date:** 2026-08-18
**Platform:** Linux x64, AMD Ryzen 5 7520U, g++ 13.3.0, `-O3 -march=haswell -mavx2 -fopenmp -flto`
**File:** `src/engine/bindings_core_ops.cpp` (a hot-path file this doc's own "Modifying Hot Paths"
rule names explicitly, requiring a dated before/after benchmark before any change lands -- the
third time this specific file was touched in one day, after the two Cluster A sessions earlier)
**Closes:** the "Nice to Have" item "Multi-dimensional arrays - Currently 1D only"

## What changed

`process_binary_array`/`process_unary_array` -- the two templates every `tadd`/`tmul`/`tmin`/
`tmax`/`tnot`/fused op (both uint8 and int8 encodings, 18 functions total) is built on -- required
exactly 1-D input via `A.unchecked<1>()`, which threw a `ValueError` for anything else. The
SIMD/OMP compute paths never actually needed 1-D specifically: they always operated on a flat
pointer + element count, with shape only ever mattering to that one entry-point check. Generalized
to accept any shape, with the output array constructed in the *same* shape as the input rather than
flattened.

**Validation added**, matching CLAUDE.md's own established conventions from earlier the same day
(gap #6 Cluster C's GEMM-binding validation):
- Both inputs must be **C-contiguous** (`arr.flags() & py::array::c_style`) -- required for the
  flat-pointer SIMD/OMP paths to be correct; a transposed view or Fortran-order array now raises a
  clear `ValueError` instead of what would otherwise be a silent wrong-answer risk.
- For binary ops, A and B must have **identical shapes** (no broadcasting). A new
  `ArrayShapeMismatchError` (`src/core/common/ternary_errors.h`) covers the case where two arrays
  have the same total element count but different shapes (e.g. `(3,4)` vs `(4,3)`) -- a condition
  that couldn't previously arise (1-D arrays: same size implies same shape). The pre-existing
  `ArraySizeMismatchError` still covers different element counts, with byte-for-byte unchanged
  wording for every existing 1-D caller.

## Correctness

A snapshot script SHA-256'd the output of all 18 array functions across 13 sizes chosen to cross
every SIMD-block/tail boundary. **234 calls, identical digest before and after this entire change**
(`bcdae62746788ff687e90c864d206ba2148ee1de2bcef93220d193edffca0e05`) -- the 1-D path is untouched
in behavior, not just "should be" but verified bit-for-bit. A new dedicated test file,
`tests/python/test_multidim_arrays.py` (8 test groups, wired into `tests/run_tests.py`), covers:
2-D/3-D/4-D correctness against a flattened-1-D reference, fused ops, the int8 bridge, the
OMP-parallel path exercised with a genuinely multi-dimensional array (not inferred from the 1-D
case), 1-D behavior/error-message preservation, shape-vs-size mismatch error discrimination, and
non-contiguous-input rejection. `tests/run_tests.py`: 16/16 (was 15/15; this is the +1).

## Performance: three iterations to get it right

This is the part worth documenting in detail, because the first two attempts were real
regressions, caught only because this file's own "Modifying Hot Paths" discipline requires
measuring before claiming success -- exactly the failure mode this project has spent this entire
session (and many before it) hunting in other people's code, this time in its own.

### Attempt 1: `py::buffer_info` (A.request()/B.request())

Straightforward implementation: validate via `py::buffer_info`, construct output via
`py::array_t<T>(shape_vector)`. Correctness was perfect (digest matched immediately). Performance
was not: a tight interleaved before/after (git-stash the change, rebuild, measure; restore,
rebuild, measure again, both within seconds of each other to control for this machine's known
`cpufreq` noise -- see `reports/2026-08-18/CV_SPIKE_ROOT_CAUSE.md`) showed a **reproducible ~2x
slowdown at small sizes (32, 1,000 elements)** and a smaller but real hit at 100,000.

### Attempt 2: switch `.attr("flags").attr("c_contiguous")` to `arr.flags() & py::array::c_style`

Suspected the Python-attribute-lookup contiguity check (already flagged as slow, in principle, for
`py_array_validate.h`'s GEMM-binding helpers, though harmless there since those are called once per
matrix multiply). Fixed it project-wide (including in the GEMM bindings, a genuine improvement
there too). Re-measured: **no change**. Same ~2x gap. Wrong hypothesis -- or at least, not the
dominant one.

### Root cause: `py::buffer_info` allocates two `std::vector`s per call

Isolated by testing in stages. Reverting *only* the output constructor to a fast 1-D-specific path
(`py::array_t<T>(n)` instead of `py::array_t<T>(shape_vector)`) recovered most of the loss
immediately, pointing at construction, not validation, as the larger factor. `py::buffer_info`
(what `.request()` returns) stores `std::vector<ssize_t> shape` and `std::vector<ssize_t> strides`
-- two heap allocations, times two calls (`A.request()` and `B.request()`) = **4 heap allocations
per binary op call**, on a codepath that used to do zero. At 100,000 elements the whole SIMD/OMP
compute takes on the order of 5-6 microseconds; a few hundred nanoseconds of allocator overhead
against that is a genuinely measurable, non-noise percentage -- consistent with the ~6-7%
still-present gap even after fixing the constructor alone.

Fix: replaced all `py::buffer_info`/`.request()` usage in the hot path with `py::array_t<T>`'s
lightweight direct accessors -- `.flags()`, `.ndim()`, `.shape()` (raw pointer), `.data()` -- which
dereference the numpy array's internal C struct directly, zero allocation. `std::vector` is now
only ever constructed on the (already-throwing, already-slow) shape-mismatch error path, where cost
doesn't matter.

### The last ~1.5%: `.shape(dim)`'s bounds check

Even with buffer_info gone, a tight interleaved measurement still showed a small (~1-1.5%, right at
the edge of this machine's noise floor) gap. Traced to `array_t::shape(ssize_t dim)`, which
internally re-derives `.ndim()` and calls a `fail_dim_check()` helper capable of throwing --
plausibly enough to perturb the compiler's code layout around the hot loop even when the throw path
is never taken. Fixed by giving the overwhelmingly common 1-D case its own branch, structured to
match the original code's shape almost exactly (a single `n != B.size()` comparison, nothing else),
and only falling into a general shape-array walk (using the raw `.shape()` pointer directly, not
the bounds-checked `.shape(dim)`) for genuine multi-dimensional inputs.

### Final verification (tight interleaved, low CV both sides)

| Size | Before | After | Delta |
|---|---|---|---|
| 32 | 36.39/44.06/43.02/42.79 Mops/s | 36.22/36.72/42.76/43.81 Mops/s | within noise |
| 1,000 | 1236.65/1264.97/1220.06/1246.96 | 1257.73/1231.38/1220.88/1260.94 | within noise |
| 100,000 | 18965/18928/18911/18852 | 18766/18766/18734/18772 | <1.5% |

All within this doc's 5% regression threshold, most within outright measurement noise. Correctness
digest re-verified identical after every iteration, not just the final one.

## Follow-up ("examine it further"): a real pre-existing bug found, plus a test-coverage gap

A user-requested deeper look after the initial ship turned up two real things, not just
reassurance:

**1. A genuine, real, pre-existing latent bug the C-contiguity check silently fixed as a side
effect.** The commit message already noted the check "rejects transposed/Fortran-order views that
would otherwise silently read wrong data through the flat-pointer path" as a property of the new
N-D code path -- but the pre-multi-dim 1-D-only code had *never* validated contiguity for 1-D
arrays either (only the entry-point ndim check existed). Verified directly, not assumed: checked
out the actual pre-multi-dim commit's `bindings_core_ops.cpp` (temporarily, then restored),
rebuilt it, and ran `tc.tadd(a[::2], b)` (a step-sliced, genuinely non-contiguous 1-D array)
against it:

```
a_strided contiguous? False size: 50
OLD CODE: NO exception -- computed a result: [2, 2, 0, 1, 2, 0, 2, 1, 1, 2]
Matches correct strided computation? False
CONFIRMED: old code silently computed WRONG results for non-contiguous 1-D input
```

This is a real bug that shipped in production before today (not introduced by this session's
work) -- every prior release's `tadd`/`tmul`/`tmin`/`tmax`/`tnot`/fused/int8-bridge function would
silently return wrong values for any 1-D input that wasn't C-contiguous (a strided slice like
`a[::2]`, a reversed view `a[::-1]`, or any other non-owning strided numpy view), with no error, no
warning -- exactly the "silent wrong result" failure class this project's review culture has spent
this entire session (and many prior ones) hunting in other files. It happened not to be caught
until now because nothing had previously exercised a non-contiguous 1-D array against this specific
file. Fixed as an incidental consequence of the multi-dimensional-array C-contiguity check, which
applies unconditionally (both the 1-D fast path and the general N-D path check `.flags() &
py::array::c_style` before doing anything else) -- confirmed the current (already-shipped) code
correctly rejects it: `ValueError: A must be C-contiguous (row-major)`.

Added `test_noncontiguous_1d_regression()` to lock this in (step-sliced, reversed, and unary
variants), since without an explicit regression test this exact bug class could silently return in
some future refactor that special-cases the 1-D path again for performance (as this file's own
history shows is a real, recurring temptation for this exact code).

**2. A test-coverage gap in the original 8-group test file**: `fused_tnot_tadd_int8`/
`fused_tnot_tmul_int8`/`fused_tnot_tmin_int8`/`fused_tnot_tmax_int8` -- the intersection of "int8
bridge" and "fused" -- were covered by neither `test_fused_ops()` (uint8 only) nor
`test_int8_bridge()` (core ops only). All 4 were re-verified correct once actually tested
(`test_fused_int8_ops()`, added). Also added `test_shape_edge_cases()` covering boundary shapes
explored but not previously locked into the suite: 0-d arrays (numpy scalars, correctly preserve
`shape=()`), empty N-D arrays (`(0,5)`), singleton dimensions (`(1,5,1)`), and 6-D arrays -- all
already correct, now regression-tested rather than just spot-checked once.

`tests/python/test_multidim_arrays.py`: 8 groups -> 11 groups. `tests/run_tests.py`: still 16/16
(same suite, more assertions inside it).

## Lesson

"Modifying Hot Paths" isn't a one-time gate -- it caught two consecutive real mistakes on the way
to a correct answer, in the same session that already used the same discipline successfully once
today (Cluster A's uint8/int8 unification, which needed zero iteration because it changed nothing
about the entry/exit machinery, only a type parameter). The difference here: this change touched
per-call validation and allocation, exactly the class of thing that's cheap to get wrong and easy
to not notice without measuring, since correctness was never in doubt at any point.

Second lesson, from the follow-up: "ship it, it works" and "examine it further" found different
things because they were looking for different failure modes. Shipping verified the feature does
what it's supposed to (N-D input, right output, right shape) and doesn't regress what already
worked (1-D digest, benchmark). Examining further went looking for what the feature's own
boundaries implied but the initial pass didn't check -- the full function list (18, not just a
representative sample), and the specific claim already written into the commit message
("rejects... that would otherwise silently read wrong data") that had never actually been checked
against the *old* code to see if it was already true. It was, and had been for a long time.
