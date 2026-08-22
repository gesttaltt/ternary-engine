# Phase 6 Power Consumption: Wiring Real Measurement Code — 2026-08-22

**Scope:** Direct follow-up to the 2026-08-20 GEMM optimization session ("push
it please, and then continue"). Picked Phase 6 (power consumption) from the
earlier recommendation menu — Phase 5 (real model quantization) deferred for
now, heavier and lower-priority than closing a gap that already had a working
implementation sitting unused.

## 1. The gap wasn't "no code" — it was "unwired code"

`bench_competitive.py`'s `phase6_power_consumption()` only ever printed a
static "framework" description ending in the literal note `"Requires actual
hardware power monitoring"`. But `benchmarks/python-with-interpreter-overhead/
bench_power_efficiency.py` already contained a complete, standalone
`PowerConsumptionBenchmark` class with working `IntelRAPLMonitor`,
`NVIDIAPowerMonitor`, `WindowsPowerMonitor`, and `MockPowerMonitor`
implementations, auto-detection, and a real energy/ops-per-Joule measurement
loop. Nothing in Phase 6 ever called it.

## 2. Verifying this sandbox's actual RAPL access before trusting anything

Before wiring anything up, checked directly whether RAPL is genuinely usable
here (not just "the directory exists"):

```
$ ls -la /sys/class/powercap/intel-rapl/intel-rapl:0/
-r-------- 1 root root 4096 energy_uj
$ cat /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
cat: Permiso denegado
$ perf stat -e power/energy-pkg/ sleep 1
Error: perf_event_paranoid setting is 4 ... needs CAP_PERFMON
$ sudo -n true
[blocked by harness permission policy]
```

Directory present, counter file root-only, `perf` fallback blocked, no sudo.
This sandbox has no path to a genuine hardware power reading — the same
class of hardware/permission blocker already documented for gap #8's
ARM/NEON deferral, not a code problem to solve here.

## 3. A real bug found in the unwired code, not sandbox-specific

`IntelRAPLMonitor.is_available()`:

```python
def is_available(self) -> bool:
    return os.path.exists(self.rapl_path)   # the DIRECTORY, not energy_uj
```

This checks the directory, which exists regardless of `energy_uj`'s
permissions. On any machine where the calling user lacks read access to
`energy_uj` — the common case for an unprivileged user on a stock Linux
install, not a sandbox quirk — this would report `True`, and
`_read_energy()`'s own `PermissionError` handler would then silently return
`0.0` for both `start_energy` and `end_energy`, producing `energy_uj = 0.0`
Joules for every real operation measured. The only trace: one easy-to-miss
`Warning: Cannot read RAPL` print per call. Same silent-degrade shape already
found and fixed repeatedly elsewhere in this project's `benchmarks/` and
`models/` (see `.claude/CLAUDE.md` Critical Gaps).

**Fix:** `is_available()` now attempts an actual read of `energy_uj` and
returns `False` on `PermissionError`/`IOError`, letting the auto-detect
cascade in `PowerConsumptionBenchmark._create_monitor()` correctly fall
through to `NVIDIAPowerMonitor` then `MockPowerMonitor`. Verified directly:

```
is_available (should be False, permission-denied file): False
Using power monitor: MockPowerMonitor
Selected monitor: MockPowerMonitor
```

## 4. A second, independent bug: RAPL counter wraparound

`energy_uj` is a wrapping counter (`max_energy_range_uj` ≈ 65.5kJ on this
hardware, confirmed by reading the real file, not assumed). `get_energy_joules()`
computed `end_energy - start_energy` with no wraparound handling — a long
enough or high-enough-power measurement window can wrap mid-benchmark,
producing a negative Joules reading. Not reachable at this module's default
`duration_sec=10.0` on typical desktop/laptop power draw (~20+ minutes to
wrap), but cheap and correct to guard unconditionally. Fixed and verified
with a synthetic before/after-wrap test:

```
Wraparound-corrected energy: 1.500 J (expect ~1.5 J)
Normal energy: 2.500 J (expect 2.5 J)
```

## 5. Wiring Phase 6 to the (now-fixed) real code

Replaced the static description with a real 2×3s ternary-vs-NumPy comparison
via `PowerConsumptionBenchmark`, reporting whichever monitor is genuinely
active. Verified end-to-end on this machine — correctly lands on
`MockPowerMonitor`, prints an unmissable warning, and records
`is_real_hardware_measurement: false` in the saved JSON:

```
Using power monitor: MockPowerMonitor
...
[WARN] No real hardware power monitor available on this machine
The numbers above are SIMULATED (MockPowerMonitor, fixed ~50W draw), NOT a real measurement.
Status: ⚠ NO HARDWARE MONITOR -- results are simulated, not citable
```

No downstream JSON consumer references `phase6` (`grep -rl phase6
benchmarks/` finds only `bench_competitive.py` itself), so this schema
change is safe.

## 6. What this does and doesn't claim

- Does NOT produce a real power number from this session — no hardware
  access exists in this sandbox to produce one.
- Does claim: Phase 6 is now real, working measurement code rather than a
  permanently-unwired stub, and — critically — it is now honest about
  whether a given run's numbers are real or simulated, instead of the
  pre-existing (unwired, but latent) bug that would have silently reported
  a genuine-looking zero-energy "success" the first time someone tried to
  actually use it on a normal, unprivileged Linux account.
- "2/5 commercial-viability criteria" is unchanged — Phase 6 still needs a
  run with real hardware access before it produces a citable number.

## Files changed

- `benchmarks/python-with-interpreter-overhead/bench_power_efficiency.py`
  (`IntelRAPLMonitor.is_available()` fix; wraparound fix in
  `get_energy_joules()`)
- `benchmarks/python-with-interpreter-overhead/bench_competitive.py`
  (`phase6_power_consumption()` rewired to call real measurement code)

`tests/run_tests.py`: 16/16 throughout (neither file is wired into that
suite — both are benchmarks, verified individually via direct execution).
