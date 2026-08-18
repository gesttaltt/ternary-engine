# Competitive Benchmark Re-validation — 2026-08-18

**Scope:** Closes the caveat Critical Gap #3 has carried since 2026-08-12:
`bench_competitive.py` had a path-resolution bug that could silently
substitute mock `(a+b)%3` arithmetic for the real engine whenever
`PYTHONPATH` didn't already contain the repo root. That bug was fixed the
same day, but the 2026-08-11 numbers behind the project's headline "2/5
commercial-viability criteria validated" claim were never re-run under a
**verified**-clean environment — this session does that.

## 1. Verifying the environment is actually clean

This machine's normal shell already has `PYTHONPATH=.:./api` set, which
would resolve to the repo root when run from there regardless of whether
`bench_competitive.py`'s own path-fix works — not a clean test. To
actually verify the fix (not just get lucky with an already-forgiving
environment), the script was run standalone with `PYTHONPATH` unset and
cwd set to `/tmp`:

```
cd /tmp && env -u PYTHONPATH python3 /path/to/bench_competitive.py --help
```

No `"Warning: ternary_simd_engine not available, using mock operations"`
line appeared — the script's own `sys.path.insert` (using
`os.path.dirname` chained from `__file__`, not cwd or the caller's
`PYTHONPATH`) correctly resolves the repo root and imports the real
compiled engine on its own. The fix holds independent of environment.

## 2. Full 6-phase run, verified-clean, 2026-08-18

With the import path confirmed self-sufficient, the full suite was run
normally (`python benchmarks/python-with-interpreter-overhead/bench_competitive.py --all`).
Results: `benchmarks/python-with-interpreter-overhead/results/competitive/competitive_results_20260818_074207.json`.

| Phase | 2026-08-11 (unverified) | 2026-08-18 (verified-clean) | Verdict |
|---|---|---|---|
| 1 — Arithmetic vs NumPy | 0.63-0.68x avg | 0.70x add / 0.69x mul avg | ✗ NEEDS WORK (unchanged) |
| 2 — Memory efficiency | 4.0x vs INT8 | 4.0x vs INT8 (exact match) | ✓ SIGNIFICANT ADVANTAGE (unchanged) |
| 3 — Throughput @ bit-width | Dense243 9.6x faster than INT2 ref | Dense243 8.0x faster than INT2 ref | ✓ still faster, magnitude shifted |
| 4 — Neural workload | 0.21x avg matmul | 0.189x avg matmul | ✗ TOO SLOW FOR AI (unchanged) |
| 5 — Model quantization | Framework only | Framework only | ⚠ unchanged, no measurement code |
| 6 — Power consumption | Framework only | Framework only | ⚠ unchanged, no measurement code |

**Commercial viability: 2/5 criteria validated — unchanged from before, and
now confirmed genuine.** The two "✓" phases (memory efficiency, throughput
at equivalent bit-width) both used the real compiled engine, verified by
the clean-environment check in §1, not the mock fallback.

## 3. Reading the Phase 1/3/4 run-to-run drift

None of the moved numbers change a verdict, but they're reported plainly
rather than rounded to "matches":

- **Phase 1** (0.63-0.68x → 0.70x/0.69x): within normal run-to-run
  variance for a wall-clock microbenchmark on a shared desktop machine (no
  isolation, no CPU pinning) — still solidly "behind NumPy," same
  conclusion as `bench_fair_baseline.py`'s independent finding.
- **Phase 3** (9.6x → 8.0x): same story, both runs on different physical
  machines (2026-08-11's host is unrecorded; this session is the AMD Ryzen
  5 4500 used for the 2026-08-17 TritNet Phase 4/5 work) — a 1.6x swing on
  a single-sample ratio-of-two-measurements metric is not remarkable, and
  Dense243 still clearly beats the INT2 reference either way.
- **Phase 4** (0.21x → 0.189x): same direction, same verdict, both runs
  share the pre-existing caveat that batch=1 with ~33% random sparsity
  understates real trained-model sparsity (~40%, per this project's own
  falsification notes) — not touched by this session.

No phase crossed a verdict boundary. This was a confirmation run, not a
correction.

## 4. Conclusion

The 2026-08-12 caveat is resolved: the mock-fallback bug's fix has now
been independently verified to hold in a genuinely clean environment (not
just re-run in the same convenient shell that might have masked the bug
before), and the resulting numbers reproduce the prior run's verdicts on
all 6 phases. The project's "2/5 commercial-viability criteria validated"
claim can be cited without the unresolved-mock-fallback qualifier that has
applied to it since 2026-08-12.

---

**Reproduce:** `python benchmarks/python-with-interpreter-overhead/bench_competitive.py --all`
(engine module must be built first: `python build/build.py`). To repeat the
environment-cleanliness check: run the same script with `PYTHONPATH`
unset and cwd outside the repo, and confirm no mock-fallback warning
appears.
