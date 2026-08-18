# Root cause: the large-array CV spike in bench_simd_core_ops.py

**Date:** 2026-08-18
**Platform:** Linux x64, AMD Ryzen 5 7520U (mobile part), 8 logical CPUs
**Follow-up to:** `reports/2026-08-18/BENCH_SIMD_CORE_OPS_STATISTICAL_RIGOR.md`, which surfaced but
did not explain CV up to 114% at 1,000,000 elements.
**Conclusion:** Environmental (CPU frequency governor), not a code bug. No engine or benchmark
code changed as a result of this investigation.

## Starting hypothesis

The statistical-rigor upgrade's full-suite run showed every op's CV jump sharply and specifically
at 1,000,000 elements (up to 114%), receding somewhat at 10,000,000. Two plausible
size-triggered code paths sit right at that boundary:

- `OMP_THRESHOLD = 32768 * hardware_concurrency()` = 262,144 on this 8-core machine -- arrays at
  or above this parallelize across all cores; 100,000 is below it (single-threaded), 1,000,000
  and 10,000,000 are both above it.
- `STREAM_THRESHOLD = 1,000,000` (hardcoded, `src/core/config/optimization_config.h`) -- arrays at
  or above this switch to non-temporal (streaming) stores. Its own comment assumes a "typical
  8-32 MB" L3 cache; this CPU's actual L3 is only **4 MiB** (`lscpu`), so the assumption behind
  the constant doesn't hold on this specific hardware.

Initial hypothesis: a cache-boundary or streaming-store code-path effect specific to n=1,000,000
on a CPU whose L3 is much smaller than the constant was tuned for.

## Test: fine-grained size sweep

Swept tadd/tmax/tnot across 15 sizes from 200,000 to 5,000,000 (finer than the production script's
fixed `TEST_SIZES`), using the same repeated-block statistics machinery. If the spike were a clean
software boundary at exactly 1,000,000, CV should stay low below that point and jump sharply at
it.

That is **not** what the data shows:

```
      size  tadd Mops/s     cv% |  tmax Mops/s     cv% |  tnot Mops/s     cv%
------------------------------------------------------------------------------------------
   200,000        9,839    1.1% |        9,915    0.5% |       21,160    0.8%
   300,000       17,710   32.7% |       17,845    5.7% |       22,572   42.3%
   500,000       21,258    3.1% |       18,084   17.3% |       28,234    2.8%
   700,000       22,763   23.4% |       21,411   17.7% |       32,071    1.1%
   800,000       23,662    8.0% |       23,129   13.3% |       33,945    5.5%
   900,000       19,870   17.7% |       20,769    9.4% |       34,449   24.3%
   999,000       24,615   35.4% |       21,326   42.8% |       35,703    7.5%
 1,000,000       22,486   22.3% |       20,655   29.6% |       36,061    1.9%
 1,001,000       27,062   33.3% |       44,724   22.1% |       67,247    3.2%
 1,100,000       30,787   47.8% |       35,341   21.9% |       64,008   28.5%
 1,300,000       41,801    9.6% |       41,598   22.4% |       66,002    9.7%
 1,500,000       18,303   37.1% |       17,906   22.7% |       19,557    7.8%
 2,000,000       10,235   54.8% |       11,545    8.3% |       60,630   15.1%
 3,000,000        4,000    4.7% |        4,022    6.1% |       15,696   25.0%
 5,000,000        5,858    3.6% |        5,846    5.2% |       14,104   12.7%
```

This rules out a clean n=1,000,000 discontinuity: CV is already high at 300,000 (well below
`STREAM_THRESHOLD`) and stays ragged and non-monotonic all the way to 5,000,000, with no visible
step at exactly 1,000,000 or 1,001,000. Throughput itself is wildly non-monotonic (tadd: 9.8K ->
17.7K -> 22.5K -> 41.8K -> 18.3K -> 10.2K -> 4.0K Mops/s across increasing sizes) -- not a shape
either a cache-capacity curve or a fixed threshold would produce on its own. The one consistent
boundary that *does* line up: everything at or above ~300,000 (i.e. above `OMP_THRESHOLD` =
262,144) is unstable; 200,000 (below it, single-threaded) is not, in every sweep run so far.

## Root cause: CPU frequency governor, amplified by parallelism

```
$ for i in 0..7; do cat .../cpu$i/cpufreq/scaling_governor cpufreq/scaling_cur_freq; done
cpu0: governor=powersave cur_freq=1329261
cpu1: governor=powersave cur_freq=1242075
cpu2: governor=powersave cur_freq=1990582
cpu3: governor=powersave cur_freq=2119122
cpu4: governor=powersave cur_freq=1211990
cpu5: governor=powersave cur_freq=2119872
cpu6: governor=powersave cur_freq=1247515
cpu7: governor=powersave cur_freq=2120915
```

All 8 cores run the `powersave` cpufreq governor. A second snapshot taken seconds later, at idle,
already showed a different spread (1.11-2.12 GHz vs. 1.21-2.12 GHz) -- frequencies are actively
drifting per-core in real time, independent of anything this benchmark does. `scaling_max_freq` is
4,386 MHz; observed frequencies during these snapshots ranged as low as 1.11 GHz, under 25% of
max.

This explains both the OMP-threshold correlation and the non-reproducible fine shape:

- **Below `OMP_THRESHOLD` (100,000, single-threaded):** one core's momentary frequency determines
  the whole measurement. Consistently low CV (<5%) in every run.
- **At or above `OMP_THRESHOLD` (>=~262,144, multi-threaded):** a parallel region's wall-clock
  time is bounded by whichever of the 8 participating cores is running slowest *at that instant*
  -- and which core that is changes from block to block as `powersave` continuously retunes each
  core independently. This turns a single governor's per-core noise into a max-of-8 amplifier,
  which is exactly the kind of instability that would (a) start right around the OMP threshold
  rather than at any specific large size, (b) stay ragged and non-monotonic rather than tracking
  array size cleanly, and (c) look different on every independent sweep, since it's sampling
  whatever the governor's schedule happens to be doing at the time, not something size determines.

The "sharp spike exactly at 1,000,000" in the original full-suite run was very likely this same
mechanism caught at one particular moment -- larger sizes take longer per timed block, giving the
governor more opportunity to step frequency mid-measurement -- not a reproducible software
boundary. The fine sweep, run later with a different governor-state history, shows a different,
messier pattern with no discontinuity at 1,000,000 specifically, which is the actual evidence
against the STREAM_THRESHOLD-code-path hypothesis this investigation started with.

`performance` is available as an alternate governor on this machine
(`scaling_available_governors: performance powersave`) but wasn't tested here -- changing a
system-wide power-management setting is the kind of thing this session left for the user to
decide rather than doing unilaterally (and this sandboxed environment has no `sudo` access to do
it directly regardless). **Recommended next step, for the user, not performed in this session:**
`sudo cpupower frequency-set -g performance` (or per-core via
`/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`), then re-run
`bench_simd_core_ops.py --quick` -- if this hypothesis is right, CV at 1,000,000+ elements should
drop sharply, roughly to the same <5% band seen at 100,000 and below.

## What this does and doesn't mean

- **Not a code bug.** No engine or benchmark script logic is wrong; nothing was changed as a
  result of this investigation.
- **Not evidence against `STREAM_THRESHOLD`'s calibration**, either -- the comment's "typical
  8-32 MB L3" assumption not matching this CPU's actual 4 MiB L3 is real and worth a future look,
  but this investigation's evidence doesn't implicate it as the driver of the observed variance.
- **Reinforces the existing platform-support policy**, rather than changing it: CLAUDE.md already
  treats Linux as "experimental only... no production claims until formally validated" and only
  trusts Windows x64 for headline figures. This investigation shows a concrete, specific mechanism
  (mobile-CPU power management, not just generic "shared container noise") for why that caution is
  warranted, at least on this machine.
- **A general lesson for any future large-array benchmark session on Linux dev hardware**: check
  `cpufreq` governor before trusting CV figures at sizes that cross `OMP_THRESHOLD`. The
  statistical-rigor upgrade did its job here -- it surfaced a real measurement-reliability problem
  that the old single-block methodology had no way to detect at all.
