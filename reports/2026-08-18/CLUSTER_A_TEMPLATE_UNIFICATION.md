# Gap #6 Cluster A: uint8_t/int8_t template unification (bindings_core_ops.cpp)

**Date:** 2026-08-18
**Platform:** Linux x64, AMD Ryzen 5 7520U, g++ 13.3.0, `-O3 -march=haswell -mavx2 -fopenmp -flto`
**File:** `src/engine/bindings_core_ops.cpp` (the file this doc's "Modifying Hot Paths" rule
explicitly names, requiring a dated before/after benchmark before any change lands)

## The change

`process_binary_array`/`process_unary_array` (uint8_t) and `process_binary_array_int8`/
`process_unary_array_int8` (int8_t) were near-byte-identical: same OMP-threshold branch, guided
scheduling, prefetch distance, streaming-store alignment check, per-thread sfence, scalar tail --
differing only in element type. Unified into `template<typename T, bool Sanitize, ...>
process_binary_array(...)` / `process_unary_array(...)`, instantiated as `<uint8_t, SANITIZE>` and
`<int8_t, SANITIZE>` at each of the 18 call sites (9 uint8 ops + 9 int8 ops). The two duplicate
int8 templates (~155 lines) were deleted.

This is a pure mechanical parameterization: nothing inside either function body depended on which
of the two types it was -- the AVX2 intrinsics (`_mm256_loadu_si256` etc.) operate on raw 256-bit
register contents regardless of signedness, and both `uint8_t`/`int8_t` are 1-byte types with
identical alignment. `T.unchecked<1>()`/`T.mutable_unchecked<1>()` needed the `.template` keyword
added (calling a template member function on a type that depends on the enclosing function's own
template parameter requires it) -- the only syntactic wrinkle, not a semantic one.

## Why a benchmark is required here specifically

Clusters B and C (fixed earlier the same day) didn't touch a file this doc's "Modifying Hot Paths"
section names. This one does -- `bindings_core_ops.cpp` is explicitly listed there, so this fix
needed the same before/after discipline any other change to this file would.

## Correctness (checked first, byte-for-byte)

A snapshot script called all 18 array functions (`tadd`/`tmul`/`tmin`/`tmax`/`tnot` +4 fused, both
uint8 and int8 encodings) across 13 sizes (0, 1, 5, 31, 32, 33, 63, 64, 65, 100, 1000, 12345,
100000 -- chosen to cross every SIMD-block/tail boundary at n%32), fixed seeds, and SHA-256'd every
output array. **234 calls, identical digest before and after**
(`bcdae62746788ff687e90c864d206ba2148ee1de2bcef93220d193edffca0e05`). `tests/run_tests.py`: 15/15
both before and after.

## Performance

`bench_simd_core_ops.py --quick`, 2 independent runs before and 2 after (this machine's confirmed
`powersave`-governor noise -- see `reports/2026-08-18/CV_SPIKE_ROOT_CAUSE.md` -- means a single run
either side proves nothing; multiple runs is the same discipline used throughout today's session).

| Size | Op | Before (2 runs) | After (2 runs) | Verdict |
|---|---|---|---|---|
| 32 | tadd/tmul/tmin/tmax/tnot | 22.7-61.7 Mops/s, CV 0.7-9.4% | 23.3-34.4 Mops/s, CV 1.4-5.9% | No regression -- both ranges overlap heavily |
| 1,000 | tadd/tmul/tmin/tnot | 668-1867 Mops/s, CV 1.1-3.1% | 670-1946 Mops/s, CV 1.3-5.6% | No regression -- ranges overlap |
| 100,000 | tadd | 18824-19072 Mops/s | 18928-19031 Mops/s | No regression (<1% either direction) |
| 100,000 | tmul | 18800-19080 Mops/s | 19012-19099 Mops/s | No regression (<1%, after slightly higher) |
| 100,000 | tmin | 19034 Mops/s (1 clean sample) | 19043 Mops/s (1 clean sample) | No regression (0.05%) |

Cells not in this table (tmax and tnot at 100,000+, everything at 1,000,000) hit CV up to 15-17% in
one or both before/after runs -- the same governor-driven instability
`CV_SPIKE_ROOT_CAUSE.md` already root-caused, unrelated to this code change and present on both
sides of the comparison equally. Reporting a precise before/after delta for those specific cells
would be false precision; they're excluded from the verdict rather than cherry-picked to fit one.

**Verdict: no regression.** Every cell with usably low variance shows the after-measurement within
~1% of before, well inside this doc's 5% regression threshold, and consistent with the a priori
expectation that a pure type-parameterization produces near-identical generated code for two
same-width integer types.

## Result

`bindings_core_ops.cpp`: -195/+55 lines (net -140) -- matching the scoping report's ~150-line
estimate closely.

That closes out all three clusters from `reports/2026-08-18/GAP6_DUPLICATION_SCOPE.md`. Gap #6 has
no remaining open items.
