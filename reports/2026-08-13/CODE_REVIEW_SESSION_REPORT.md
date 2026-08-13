# Code Review Session Report — 2026-08-13 (continuation)

**Scope:** Closes out the pending scope left by
[reports/2026-08-12/CODE_REVIEW_SESSION_REPORT.md](../2026-08-12/CODE_REVIEW_SESSION_REPORT.md):
`benchmarks/utils/` (4 of 6 files not yet reviewed), `benchmarks/macro/`,
`research/`, `opentimestamps/`, and a broader path-resolution sweep beyond the two
idioms the prior session covered. Platform: Linux x64, same environment as the
2026-08-12 session.

**Method:** Same as the prior session — background `code-review` subagents
(multiple parallel "finder angle" passes over `research/scripts/falsify.py`, the
largest single file in scope at 2738 lines) plus manual review, with every finding
reproduced or directly verified against running code before being called a bug.

**Net result:** 2 commits, `babff9b` and `fb954f2`, both on `main`, both pushed.
Both rebuilt/re-ran `tests/run_tests.py` (13/13 suites) before landing.

---

## What was reviewed and fixed, by area

### `benchmarks/utils/geometric_metrics.py`

`autocorrelation(lag=0)` crashed with a broadcast `ValueError`. Root cause:
`x[:-lag]` with `lag=0` evaluates as `x[:-0] == x[:0]` (Python's negative-zero
slicing gotcha) instead of the full array, so the lag-0 case — which should
trivially return an autocorrelation of 1.0 — instead tried to broadcast an empty
array against the full one. Fixed with an explicit `lag > 0` branch. Reproduced the
crash first, then verified the fix across 4 cases (lag=0/lag=1 on random data,
lag=0/lag=5 on constant data).

### `benchmarks/utils/benchmark_validator.py`

Two related bugs, both traced to `extract_performance()` returning `0.0` for two
different situations that need different handling: "genuinely zero throughput" and
"operation/size not found in any of the 3 hardcoded JSON schemas" (this repo's real
`fair_baseline_*.json` output, keyed under `'cells'`, matches neither of the 3).

1. `compare_performance()` treated the second case as if it were the first,
   reporting a false -100% regression instead of "no data available" whenever a
   JSON's schema didn't match.
2. `generate_report()` had no way to distinguish "`load_data()` never succeeded"
   from "it succeeded and found zero regressions" — both rendered identically as
   "Total Benchmarks: 0, Failed: 0, Status: ✅ PASS, Action: Proceed with merge to
   main branch", even when nothing was ever compared because the input files were
   missing or malformed.

Fixed: `extract_performance()`'s return type changed to `Optional[float]` (`None` =
not found in this schema); `compare_performance()` now emits an explicit `NO_DATA`
status distinct from `PASS`/`FAIL`, plumbed through both the console table and the
markdown report; a `data_loaded` flag set by `load_data()` gates `generate_report()`
into an explicit failure report instead of a false pass.

Verified with synthetic inputs:
- Missing baseline/current files → exit code 1, report correctly says "Could not
  load benchmark data" (previously would have said 0/0 PASS).
- Baseline JSON with an unrecognized schema + current JSON with only `tadd` data →
  1 real `PASS` comparison + 4 correctly-labeled `NO_DATA` rows (previously: 4 false
  `FAIL` regressions at -100%).

### `benchmarks/utils/visualization.py`

Three bugs plus a missing header, all confirmed against the real
`bench_competitive.py` output schemas (read directly from
`benchmarks/python-with-interpreter-overhead/bench_competitive.py`, not assumed):

1. `_format_phase4`/`_generate_phase4_html` crashed with `TypeError` when phase4
   data was the `{'error': 'ternary_zero_skip_gemm not available'}` dict
   `bench_competitive.py` emits when that module isn't built. A non-empty dict is
   truthy, so the existing `if not data:` guard didn't catch it; execution fell
   through to `for result in data:`, which iterates the dict's string keys (yielding
   `'error'`), then crashed on `result['shape']`.
2. `_format_phase3` checked for a `'ternary_gops'` key that no longer exists in
   Phase 3's output schema since `bench_competitive.py`'s 2026-08-11 fix (CLAUDE.md
   gap #3) — current schema is `{'target_bytes', 'representations': [...], 'note',
   'verdict'}`. This always fell through to "No data available" even with real data
   present.
3. `generate_html_report()` never rendered Phase 3 at all — only
   `_generate_phase1_html`/`_generate_phase2_html`/`_generate_phase4_html` were ever
   called; no `_generate_phase3_html` method existed to call.
4. Missing the mandatory Apache 2.0 copyright header (present on its siblings in the
   same directory).

All fixed; verified together against a synthetic JSON matching the real schemas —
Phase 3 now renders correctly in both text and HTML output, Phase 4's error-dict
case no longer crashes either path.

### `benchmarks/utils/windows_power.py`

`WindowsPowerBenchmark.benchmark_operation()` called the blocking PowerShell-based
`monitor.sample()` inside the same timed `while` loop used to measure operation
throughput. Each sample can take up to its own 5-10s subprocess timeout, so
`ops_per_sec` (computed as `iterations / elapsed`) was deflated by however much
wall-clock time happened to go to sampling calls landing mid-benchmark. Fixed by
tracking operation-only elapsed time separately and computing `ops_per_sec` from
that instead. Verified functionally on Linux (PowerShell calls fail fast via
`FileNotFoundError`, caught) — runs without crashing, `ops_per_sec` now reflects
only `operation_fn()` time.

Also confirmed and documented (not merged, per YAGNI/no-speculative-refactor
policy): this entire module is unimported anywhere in the repo —
`bench_power_efficiency.py`, its only conceivable caller, reimplements an
independent `WindowsPowerMonitor(PowerMonitor)` from scratch instead. Same
duplication shape already logged for TritNet Phase 2a/2b (CLAUDE.md gap #7). Added
the missing copyright header.

### `benchmarks/macro/` (`bench_image_pipeline.py`, `bench_layer_forward.py`)

Reviewed both files and ran each end-to-end. Both correct: path resolution
(`Path(__file__).parent.parent.parent`) checks out for their actual depth,
correctness validation passes before the timed benchmark runs, and both complete
with sensible speedup numbers and exit codes. No bugs found.

### `opentimestamps/{timestamp_create,timestamp_verify}.py`

Reviewed both. `timestamp_create.py` has real, hard-to-reverse side effects (can
submit a hash to the public Bitcoin blockchain via the external `ots` CLI — see its
own header warning, "runs immediately on invocation"), so it was read-reviewed only,
never executed. No correctness bugs found in its logic. `timestamp_verify.py` is
read-only, so it was actually run against a real manifest from an earlier session
(`manifest_20260811_135314.json`) — correctly flagged the genuine file
modifications/deletions/removals that happened earlier in this multi-day review
effort, serving as an incidental smoke test of both the tool and this session's own
cumulative footprint.

Both files had `datetime.utcnow()` deprecation warnings (removed in Python 3.12+).
Fixed in both (3 sites in `timestamp_create.py`, 2 in `timestamp_verify.py`) by
switching to `datetime.now(timezone.utc)`, preserving the exact prior output string
format via `.replace("+00:00", "Z")` on `.isoformat()` call sites. Re-verified clean
with `-W error::DeprecationWarning` (would crash on any remaining warning).

Incidental cleanup: running `timestamp_verify.py` twice for testing created 2 real
`verification_report_*.json` files in `opentimestamps/logs/` — a directory the
project's own `.gitignore` says to keep for audit-trail purposes. These were testing
side effects, not deliberate audit records, so deleted rather than left to clutter
the real trail.

### `research/scripts/falsify.py` (2738 lines) — the highest-value findings

Reviewed via multiple parallel background `code-review` "finder angle" subagents
(8 angles dispatched; this file's size warranted the parallel-angle approach used
for other large files in the 2026-08-12 session). Every finding below was
reproduced against the real pipeline (not just read and assumed) before and after
the fix, using `env -u PYTHONPATH python3 research/scripts/falsify.py --hypothesis
<H>` and a `--all` run for the full regression check.

**`ComponentLoader.build_corpus()`'s inverted valuation array** (the most
consequential finding): computed `valuations[i] = v3(i)` on the **raw** corpus
encoding index (`idx = Σ(trit_i+1)·3^i`, range `[0, 19682]`) instead of
`v3(i - idx_offset)` on the **decoded** balanced-ternary value (`idx_offset = 9841`,
the index whose 9 trits are all zero). This silently swapped which indices looked
"near zero" for every consumer that looks the array up pointwise by index: the true
ternary zero (raw index 9841) got valuation 0 (since `9841 % 3 == 1`), while the
most negative representable value (raw index 0) got the maximal valuation 999
(since `v3(0) = 999` by definition). Affected H3's valuation-radius-correlation test
and H24's associativity-vs-valuation bucketing. Aggregate histograms (e.g. H13's
level counts) were unaffected — invariant under a constant shift over a complete
residue system of length 3⁹. Reproduced standalone with the actual
`compute_3adic_valuation` logic before touching the real file; verified fixed both
standalone and against the real pipeline (the corpus's own printed valuation
distribution now shows exactly one index at valuation 999, matching the single true
zero).

**`test_H9_information`'s zero-check used the same inverted index directly**:
`zero_valuation = v3(zero_idx)` where `zero_idx = 9841` — the raw index, not the
decoded value 0. Since `9841 % 3 == 1`, this returned valuation 0 and the "zero
should have very high valuation" assertion (`>= 8`) always failed, regardless of any
real ternary structure. Fixed to `v3(zero_idx - idx_offset)`. Verified: H9 now
scores 100% with `zero_is_special: True` in its details (was unconditionally
`False` before, confirmed via direct inspection of the result object, not just the
truncated console summary).

**`test_H1_padic`'s skewed sample**: `all_results[:10000]` sliced only the head of
`np.concatenate([d['results'] for d in luts.values()])` — since `luts` has ~4
operation keys each contributing ~50,000 samples, this drew entirely (or almost
entirely) from whichever operation's dict-insertion order came first, not a
representative cross-operation mix, corrupting the valuation-distribution test that
follows. Fixed with a fixed-seed (`42`) uniform random sample of the same size.
Reproduced with synthetic tagged data confirming the old slice contained only one
"operation ID" while the new sample contains a roughly even mix of all 4; verified
against the real pipeline (H1 now scores 100%, valuation distribution shows the
expected `~69%/18%/7%/3%/...` decreasing pattern).

**`main()`'s incomplete pre-flight guard**: checked only
`status['data']`/`status['corpus']` before running any hypothesis test. Since
corpus-building doesn't depend on LUTs or the hyperbolic module (`build_corpus()`
only needs `v3`/`index_to_trits`), LUT-loading or hyperbolic-module failure could
pass this guard silently. `test_H1_padic`/`test_H9_information` (unconditional
`self.c['luts']`) and `test_H3_hyperbolic` (unconditional `self.c['hyperbolic']`)
would then raise a bare `KeyError`, caught by `run_hypothesis`'s generic
`except Exception` and reported as an opaque `grade='E'` with `error="'luts'"` — no
indication this was a component-loading gap rather than a genuine test failure (and
this misleading grade would get persisted to the saved JSON results). Added explicit
guards to all three tests (a descriptive `RuntimeError` naming the missing
component and pointing at the loader's own `[ERROR]`/`[WARN]` console output) and an
upfront `[WARN]` in `main()` when either component fails to load, so the gap is
visible before any test even starts.

**`test_H24_sui_generis`'s dead code**: the right-distributivity block computed
`rhs_r` from `ab`/`bc2` (`ab` is `tmul(a,b)`, not one of the two terms
right-distributivity actually needs), immediately discarded that result, then
recomputed correctly from freshly-derived `ac2`/`bc3` three lines later — both of
which duplicate values already available (`ac` from the left-distributivity test
just above; `bc2`/`bc3` are the identical call run twice). A leftover
`# Wait: need tmul(a,c) and tmul(b,c) separately` comment marks where this
in-place bug-fix happened. Cleaned up to reuse `ac` and compute `bc_mul` only once;
verified byte-for-byte identical output against the real engine before and after the
change (not just "looks equivalent" — actually diffed the numpy arrays).

**Also fixed** (CLAUDE.md convention gaps, found by a dedicated review angle):
docstring header referenced a `--tier` flag that was never wired into `argparse`
(only `--hypothesis`/`--all` exist) and had no `OUTPUT:` line; both fixed. Added a
`-> int` return-type hint to `main()`, matching its actual `return 0`/`return 1`
paths and every other function in the file already being typed.

**Documented but intentionally not fixed** — structural/methodology findings, not
wrong-answer bugs, matching this project's established policy of deferring
non-correctness duplication/performance work (see CLAUDE.md gaps #7/#8 for the same
pattern already applied to the TritNet training pipelines and `BenchmarkRunner`):

- `compute_3adic_valuation` (the `while n % 3 == 0: n //= 3; v += 1` loop) is
  independently reimplemented in **10 places** across the repo
  (`models/3-vae-gemm-v1/{data,model,hyperbolic_ops}.py` — the latter twice — and
  five files under `models/company-flagships/`, plus this one), each with a
  different max-valuation clamp (9, `num_trits`, hardcoded 10, or — this file's
  version — no clamp at all). The `ebm/ultrametric_energy.py` module the comment
  here cites as the reason for inlining ("math needed to be self-contained") no
  longer exists anywhere in the tree, so there's no canonical version left for
  future code to converge on.
- All 14 `test_H*()` methods hand-roll identical `passed`/`tested`/`anomalies`/
  `details` bookkeeping (74 raw init occurrences) and an identical ~10-line
  `FalsificationResult(...)` construction (16 call sites, including 2 inside
  `TestRunner`'s own error-handling path) instead of sharing a
  `TestContext`/decorator. Only `_score_to_grade()` is actually centralized.
- `H4`/`H10`/`H11`/`H23` each independently call `np.random.seed(42)` then draw
  same-shaped arrays with the same calls in the same order — running `--all`
  makes their samples bit-for-bit identical instead of independent (does not
  affect any single hypothesis run in isolation, only cross-hypothesis
  independence when batched).
- `H8`'s categorical tests call `simd_op` on single-element arrays ~1300 times
  instead of batching like every other hypothesis test in the file, ~185x more
  calls than the equivalent batched pattern would need.
- `tnot_batch()` is duplicated verbatim in two places and reimplements a
  per-element decode/negate/encode loop, when the file's own documented index
  encoding (`idx = Σ(trit_i+1)·3^i`) admits a closed-form vectorized negation:
  `neg_idx = (num_values - 1) - idx` (provable since `Σ2·3^i = num_values - 1`).
- Three independent Shannon-entropy implementations exist across this file and
  `geometric_metrics.py`.

### Broader path-resolution sweep

The 2026-08-12 session's path-bug hunt covered two specific idioms
(`Path(__file__).parent...` chains and `os.path.dirname(os.path.dirname(...))`
chains) and flagged a second, more general pass as worthwhile. This session
grepped for every `ROOT`/`PROJECT_ROOT`-style assignment built from `Path(__file__)`
across all 48 files in the repo that manipulate `sys.path` (~50 such assignments,
outside `legacy/`/`deprecated/`), then programmatically checked each one's parent-hop
count against that file's actual directory depth from repo root.

Found 2 more instances of the exact bug class already fixed in 12 files by commit
`32eada2`: `tests/python/compile_test.py` and `tests/python/run_simd_harness.py`
both live at `tests/python/` (2 directories deep) but computed their project root
with only 2 `.parent` calls instead of 3, landing on `tests/` instead of the repo
root. This doubled the `"tests"` path segment downstream
(`PROJECT_ROOT / "tests" / "test_simd_correctness.cpp"` became
`tests/tests/test_simd_correctness.cpp`), and in `run_simd_harness.py` also put
`BUILD_DIR`/`RESULTS_DIR` under `tests/` instead of the real repo root. Independently
of that bug, both scripts' target file — `test_simd_correctness.cpp` — has since
moved to `tests/cpp/`, not `tests/` directly, so the pre-bug intended path was
already stale too; in practice neither bug made a difference before this fix (the
file resolved under neither path), but the path math is now correct and resolves to
the real file. Both scripts remain Windows/MSVC-only dev utilities not wired into
`tests/run_tests.py`, unchanged from the 2026-08-12 session's classification (gap
#1) — this only fixes their path math, not their (unexercisable-on-Linux) MSVC
compilation logic.

Two other candidates flagged during the initial grep
(`benchmarks/cpp-native-kernels/build_gops_bench.py` and `build_kernels.py`) were
checked and found to already be correct — both compute
`script_dir = Path(__file__).parent` then `project_root = script_dir.parent.parent`
(3 parent-hops total from `__file__`, not the 2 an incomplete grep-based tally
initially suggested); confirmed by resolving the actual paths and checking
`src_dir.exists()`. `opentimestamps/{timestamp_create,timestamp_verify}.py` were
also re-confirmed correct, consistent with having already been exercised earlier in
this session.

No other instance of the bug was found in the ~50 checked. `benchmarks/deprecated/`
(5 files, identical bug) remains intentionally unfixed, matching the 2026-08-12
session's consistent policy on archived/deprecated code.

---

## Verification discipline used throughout

Same standard as the 2026-08-12 session:
- Reproduced the original bug first (crash, wrong output, or wrong exit code), then
  confirmed the fix resolves that exact reproduction — for `falsify.py`'s bugs, this
  meant reproducing the buggy math standalone (with a bare-bones reimplementation of
  `compute_3adic_valuation`) before touching the real file, then re-verifying against
  the actual pipeline afterward.
- For the H24 dead-code cleanup: diffed the actual numpy output arrays before and
  after the change to confirm it was behavior-preserving, not just "looks
  equivalent."
- For anything involving `PROJECT_ROOT`/path resolution: verified with
  `env -u PYTHONPATH`, and cross-checked hand-tallied parent-hop counts by actually
  resolving the paths in Python rather than trusting a manual count (this caught 2 of
  the audit's own transcription errors before they became false positives).
- `tests/run_tests.py` (13/13 suites) re-run after every commit in this session.
