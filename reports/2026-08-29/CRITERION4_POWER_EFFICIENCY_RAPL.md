# Criterion 4: power efficiency — passes, and only because the baseline is right

**Date:** 2026-08-29 · **Platform:** Linux x64, AMD Ryzen 5 4500 (Zen 2, 6C/12T),
Intel-RAPL powercap (`package-0`, `core`), idle floor 4.25 W ·
**Author:** Ternary Engine Team

---

## 1. Unblocked

Criterion 4 had never been measured. The roadmap recorded it as needing "root
or other hardware", which was corrected earlier today to the accurate cause:
`/sys/class/powercap/intel-rapl:0/energy_uj` is mode `0400`, the post-PLATYPUS
mitigation. A `sudo chmod a+r` on both domains (`package-0` and `core`) made
it readable — hardware was never the issue.

(`sudo python3 …` fails on this machine for an unrelated reason: NumPy lives
in the user's `~/.local`, invisible to root. With the chmod applied, no sudo
is needed.)

---

## 2. Three bugs stood between "run it" and "trust it"

Measuring this honestly required fixing the project's own power benchmark
first. All three would have produced a confident, wrong verdict.

**(a) Silent fabrication.** `MockPowerMonitor.is_available()` returned `True`
unconditionally and sat in the auto-detect chain, so on any root-only-RAPL
machine `--platform auto` would report a fixed 50 W, derive a real-looking
ops/joule ratio, save it with no marker, and exit 0.

**(b) Wrong device, and it was live.** `NVIDIAPowerMonitor` sat ahead of the
mock and returns True whenever `nvidia-smi` exists. `--platform auto` was
therefore measuring **GPU** power for a **CPU** workload — reporting ~11.3 W
of idle GPU draw as the energy cost of AVX2 ternary ops, with
`is_simulated: false`. Nothing fabricated; real readings of the wrong thing,
which no simulation marker could ever catch.

**(c) The verdict-deciding one: unlike operations.** The benchmark timed
`tc.tadd` — a **saturating** ternary add — against `np.add(a, b)`, a **raw
wrapping** int8 add. Different operations. That mismatch, plus uncontrolled
threading (see §3), is what produced its original
**"0.54×, ✗ NO POWER ADVANTAGE"**.

(a) and (b) were fixed in commit `cf46366`; (c) is fixed here.

---

## 3. Two fairness controls, both of which change the answer

**Threading.** The engine uses OpenMP above `OMP_THRESHOLD`; NumPy's
elementwise ops are single-threaded. Measured directly: `tc.tadd` consumes
**11.89 cores**, `np.add` **1.02**. An uncontrolled run compares ~12 cores
against 1 and charges ternary for all of them. Results below are reported
both pinned (`OMP_NUM_THREADS=1`) and unrestricted.

**Semantic equivalence.** `tadd_int8` was verified against
`np.clip(a+b, -1, 1)` over the **complete 9-entry truth table** before being
used as a baseline, and the engine's uint8 encoding was confirmed as
`v = trit + 1`. This is asserted in the benchmark now, not assumed, so a
future kernel change that breaks the equivalence fails loudly rather than
silently restoring the old mismatch.

That verification caught two of my own errors mid-session: I first fed raw
int8 `{-1,0,1}` to the uint8 `tadd` entry point (producing garbage outputs
like 52 and 64, and a meaningless 4.8 Gelem/s), and then wrote the encoded
baseline as `a+b-1` when encoded trits sum as `a+b-2`. The assertion caught
the second one before it reached a number.

---

## 4. Result

1,000,000-element arrays (fits in this CPU's 8 MB L3, so this is
cache-bandwidth-bound, not DRAM-bound). RAPL `package-0`. Ratios are
ternary ÷ NumPy-equivalent; **> 1.0 means ternary is more efficient**.

| Configuration | raw ratio | idle-subtracted ratio |
|---|---|---|
| int8 API, 1 thread | 5.56× | **3.92×** |
| uint8 API, 1 thread | 7.96× | 5.54× |
| int8 API, all cores | 5.27× | **3.05×** |
| uint8 API, all cores | 7.61× | 4.29× |

Underlying single-thread figures:

| operation | pkg W | Melem/s | Melem/J |
|---|---|---|---|
| engine `tadd_int8` | 15.63 | 16,038 | 1,026 |
| NumPy equivalent `clip(a+b,-1,1)` | 8.73 | 1,611 | 185 |
| engine `tadd` (uint8) | 14.58 | 17,408 | 1,194 |
| NumPy equivalent (uint8 encoded) | 8.39 | 1,258 | 150 |
| NumPy raw `a+b` *(not equivalent)* | 13.49 | 30,061 | 2,228 |

**Criterion 4 target is "2–4× better". The conservative reading — int8 API,
idle-subtracted — is 3.05–3.92×, squarely inside that band. The raw
package-energy reading is 5.3–8.0×, above it. Either way the criterion is
met.**

Both framings are reported because both are defensible: raw package energy
is what the socket actually draws, while idle-subtracted isolates the energy
attributable to the work. The idle floor (4.25 W of a ~9–16 W total) is large
enough that the choice moves the number materially, so quoting only one would
be a framing choice disguised as a measurement.

---

## 5. The result depends entirely on the baseline, and that must travel with it

Against raw `np.add` — a strictly **simpler** operation, no saturation —
ternary is **2.2× worse** on energy (1,026 vs 2,228 Melem/J).

The advantage is real but it is *specifically* the saturating semantics
coming for free in the LUT, which is exactly the position this project
already documents (`SKEPTICAL_METRICS.md`: the real wins are
"saturation-for-free (tadd 1.7–3.5×)"). Quoting the 5.6× without the
baseline definition would be the same selective framing retired with the
"8,234× vs Python" headline. The corrected benchmark therefore times the raw
add too and prints it, labelled non-equivalent and excluded from the verdict.

---

## 6. Honest limits

- **One operation, one size.** `tadd` at 1M elements. Not tmul/tmin/tmax, not
  a full workload, and at a size that fits in L3 — DRAM-resident arrays would
  shift both sides.
- **A disagreement I am not hiding.** After the baseline fix, the project's
  own benchmark reports **21.76×** where the controlled measurement here says
  7.96× raw. The energy figures agree between the two (≈14.7 W vs ≈15.6 W
  ternary, ≈7.3 W vs ≈8.7 W NumPy); the **iteration counts** do not, and that
  script also still displays "Avg power: 0.00 W" (its `stop_monitoring()`
  returns 0.0 by design and the display never recomputes from energy). Its
  absolute ratio should not be cited until that accounting is audited. The
  numbers in §4 come from the direct, self-contained measurement.
- **Package domain.** `package-0` includes uncore; the `core` sub-domain reads
  far lower and was not used for the headline.
- **Not end-to-end.** This is kernel-level energy, like criterion 3's
  kernel-level latency.

---

## 7. Status

**4/5 commercial-viability criteria now validated.** The remaining one is
criterion 5 (accuracy), which fails across five measured quantization
techniques and whose honest reframing — running natively-ternary models
rather than manufacturing them — is covered in
`BITNET_NATIVELY_TERNARY_ON_ENGINE.md`.

## 8. Reproduction

```bash
# Persistent (survives reboot) -- installs a udev rule, 0440 root:adm
sudo python3 scripts/setup/install_rapl_udev_rule.py
python3 scripts/setup/install_rapl_udev_rule.py --check   # verify, no root

# Or one-shot for this boot only
sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj \
               /sys/class/powercap/intel-rapl:0:0/energy_uj

OMP_NUM_THREADS=1 python3 benchmarks/python-with-interpreter-overhead/bench_power_efficiency.py --platform intel
```

A bare `chmod` does **not** persist: the powercap devices are recreated with
default 0400 permissions on every boot. The udev rule reapplies the change on
each device-add. It grants group read to `adm` rather than world read, to keep
the PLATYPUS side-channel exposure to accounts that are already privileged --
see `scripts/setup/99-ternary-rapl-readable.rules` for the full trade-off, and
do not install it on a shared or multi-tenant host.
