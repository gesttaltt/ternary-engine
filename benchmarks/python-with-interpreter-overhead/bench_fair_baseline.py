#!/usr/bin/env python3
"""
bench_fair_baseline.py - Fair headline-number benchmark vs NumPy

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Replaces the "8,234x vs pure Python" strawman (see
benchmarks/SKEPTICAL_METRICS.md, "Anti-Patrones") with a defensible
comparison: every engine operation is measured against the FASTEST NumPy
implementation of the SAME ternary semantics on int8 values {-1, 0, +1}:

    tadd -> np.clip(a + b, -1, 1)   (saturated add; clip is the cost of
                                     saturation, which NumPy cannot fuse)
    tmul -> a * b
    tmin -> np.minimum(a, b)
    tmax -> np.maximum(a, b)
    tnot -> np.negative(a)

Context row (different semantics, same information content): raw
np.add(int8) without saturation.

Methodology (per project standards):
    - Fixed seed, warmup, N timed repeats per (op, size)
    - Median, std, coefficient of variation reported per cell
    - Geometric mean across ops/sizes as the honest "average speedup"
    - Engine operates on its native 2-bit encoding (uint8 {0,1,2});
      NumPy on int8 {-1,0,+1}. Both stay in their steady-state formats;
      pack/unpack conversion is not included (elementwise pipelines keep
      data encoded between ops).

USAGE:  python benchmarks/python-with-interpreter-overhead/bench_fair_baseline.py
        [--sizes 1000 100000 ...] [--repeats 30] [--quick]
OUTPUT: benchmarks/results/fair_baseline_<timestamp>.json + stdout table
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import ternary_simd_engine as eng  # noqa: E402

# Shared statistics engine (median/mean/stdev/CV/CI), single-sourced with
# BenchmarkRunner and bench_simd_fusion_ops.py. Extracted 2026-08-18,
# CLAUDE.md gap #8. compute_timing_statistics() already guards the
# len==1 statistics.stdev() ValueError this file's own safe_stdev() used to
# handle locally (--repeats 1 is a legitimate quick smoke-test value --
# argparse has no minimum on it -- found 2026-08-12).
sys.path.insert(0, str(Path(__file__).parent))
from benchmark_framework import compute_timing_statistics  # noqa: E402

SEED = 42
DEFAULT_SIZES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
DEFAULT_REPEATS = 30


# NumPy implementations of the SAME ternary semantics, written the
# fastest reasonable way (preallocated output, out= to avoid allocation).
def np_tadd(a, b, out, tmp):
    np.add(a, b, out=tmp)
    return np.clip(tmp, -1, 1, out=out)

def np_tmul(a, b, out, tmp):
    return np.multiply(a, b, out=out)

def np_tmin(a, b, out, tmp):
    return np.minimum(a, b, out=out)

def np_tmax(a, b, out, tmp):
    return np.maximum(a, b, out=out)

def np_tnot(a, b, out, tmp):
    return np.negative(a, out=out)

def np_raw_add(a, b, out, tmp):
    return np.add(a, b, out=out)


# Fused ops: engine computes tnot(op(a, b)) in ONE pass; NumPy needs 2-3
# ufunc passes over memory. This is where fusion's advantage lives.
def np_tnot_tadd(a, b, out, tmp):
    np.add(a, b, out=tmp)
    np.clip(tmp, -1, 1, out=tmp)
    return np.negative(tmp, out=out)

def np_tnot_tmul(a, b, out, tmp):
    np.multiply(a, b, out=tmp)
    return np.negative(tmp, out=out)

def np_tnot_tmin(a, b, out, tmp):
    np.minimum(a, b, out=tmp)
    return np.negative(tmp, out=out)

def np_tnot_tmax(a, b, out, tmp):
    np.maximum(a, b, out=tmp)
    return np.negative(tmp, out=out)


OPS = {
    # name: (engine_fn(a[,b]), numpy_fn, is_binary)
    'tadd': (eng.tadd, np_tadd, True),
    'tmul': (eng.tmul, np_tmul, True),
    'tmin': (eng.tmin, np_tmin, True),
    'tmax': (eng.tmax, np_tmax, True),
    'tnot': (eng.tnot, np_tnot, False),
}

FUSED_OPS = {
    # name: engine one-pass fused op vs NumPy multi-pass equivalent of
    # the same semantics: fused_tnot_X(a, b) == tnot(X(a, b)).
    'tnot(tadd)': (eng.fused_tnot_tadd, np_tnot_tadd, True),
    'tnot(tmul)': (eng.fused_tnot_tmul, np_tnot_tmul, True),
    'tnot(tmin)': (eng.fused_tnot_tmin, np_tnot_tmin, True),
    'tnot(tmax)': (eng.fused_tnot_tmax, np_tnot_tmax, True),
}

CONTEXT_OPS = {
    # Different semantics (no saturation) — context only, excluded from
    # the headline geometric mean.
    'raw_int8_add': (eng.tadd, np_raw_add, True),
}


def time_call(fn, repeats):
    """Return list of per-call wall times (seconds) over `repeats` calls."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def bench_cell(op_name, engine_fn, numpy_fn, is_binary, size, repeats, rng):
    # Engine inputs: native 2-bit encoding {0,1,2} in uint8
    a_e = rng.integers(0, 3, size, dtype=np.uint8)
    b_e = rng.integers(0, 3, size, dtype=np.uint8)
    # NumPy inputs: int8 values {-1,0,+1}
    a_n = (a_e.astype(np.int8) - 1)
    b_n = (b_e.astype(np.int8) - 1)
    out = np.empty(size, dtype=np.int8)
    tmp = np.empty(size, dtype=np.int8)

    if is_binary:
        run_e = lambda: engine_fn(a_e, b_e)
        run_n = lambda: numpy_fn(a_n, b_n, out, tmp)
    else:
        run_e = lambda: engine_fn(a_e)
        run_n = lambda: numpy_fn(a_n, None, out, tmp)

    # Warmup
    for _ in range(5):
        run_e()
        run_n()

    t_e = time_call(run_e, repeats)
    t_n = time_call(run_n, repeats)

    stats_e = compute_timing_statistics(t_e)
    stats_n = compute_timing_statistics(t_n)
    med_e = stats_e['median']
    med_n = stats_n['median']
    return {
        'op': op_name,
        'size': size,
        'engine_median_us': med_e * 1e6,
        'engine_std_us': stats_e['stdev'] * 1e6,
        # compute_timing_statistics() reports 'cv' as a percentage (0-100);
        # this script's own convention (and the 0.15 threshold below) is a
        # 0-1 fraction, so convert here rather than changing either
        # convention's meaning.
        'engine_cv': stats_e['cv'] / 100,
        'numpy_median_us': med_n * 1e6,
        'numpy_std_us': stats_n['stdev'] * 1e6,
        'numpy_cv': stats_n['cv'] / 100,
        'engine_mops_s': size / med_e / 1e6,
        'numpy_mops_s': size / med_n / 1e6,
        'speedup': med_n / med_e,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--sizes', type=int, nargs='+', default=DEFAULT_SIZES)
    ap.add_argument('--repeats', type=int, default=DEFAULT_REPEATS)
    ap.add_argument('--quick', action='store_true',
                    help='Smoke test: small sizes, few repeats')
    args = ap.parse_args()

    sizes = [1_000, 100_000] if args.quick else args.sizes
    repeats = 5 if args.quick else args.repeats

    rng = np.random.default_rng(SEED)
    print("=" * 78)
    print("Fair Baseline Benchmark: ternary_simd_engine vs NumPy (same semantics)")
    print(f"Platform: {platform.platform()}  Python {platform.python_version()}  "
          f"NumPy {np.__version__}")
    print(f"Sizes: {sizes}  Repeats: {repeats}  Seed: {SEED}")
    print("=" * 78)

    cells = []
    fused_cells = []
    context_cells = []
    for group, target in ((OPS, cells), (FUSED_OPS, fused_cells),
                          (CONTEXT_OPS, context_cells)):
        for op_name, (efn, nfn, is_bin) in group.items():
            for size in sizes:
                c = bench_cell(op_name, efn, nfn, is_bin, size, repeats, rng)
                target.append(c)

    hdr = (f"{'op':<14} {'size':>10} {'engine Mops/s':>14} {'numpy Mops/s':>13} "
           f"{'speedup':>8} {'cv_e':>6} {'cv_n':>6}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for c in cells + fused_cells + context_cells:
        print(f"{c['op']:<14} {c['size']:>10,} {c['engine_mops_s']:>14,.0f} "
              f"{c['numpy_mops_s']:>13,.0f} {c['speedup']:>7.2f}x "
              f"{c['engine_cv']:>6.2f} {c['numpy_cv']:>6.2f}")

    speedups = [c['speedup'] for c in cells]
    geo = float(np.exp(np.mean(np.log(speedups))))
    mn, mx = min(speedups), max(speedups)
    fused_speedups = [c['speedup'] for c in fused_cells]
    geo_fused = float(np.exp(np.mean(np.log(fused_speedups))))
    print("-" * len(hdr))
    print(f"HEADLINE (same-semantics single ops, geometric mean): {geo:.2f}x "
          f"(range {mn:.2f}x – {mx:.2f}x)")
    print(f"FUSED (one-pass engine vs multi-pass NumPy, geometric mean): "
          f"{geo_fused:.2f}x (range {min(fused_speedups):.2f}x – "
          f"{max(fused_speedups):.2f}x)")

    high_cv = [c for c in cells + fused_cells
               if c['engine_cv'] > 0.15 or c['numpy_cv'] > 0.15]
    if high_cv:
        print(f"[WARN] {len(high_cv)} cells have CV > 15% — rerun on an idle "
              f"machine before publishing these numbers")

    out = {
        'timestamp': datetime.now().isoformat(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'seed': SEED,
        'sizes': sizes,
        'repeats': repeats,
        'quick': args.quick,
        'cells': cells,
        'fused_cells': fused_cells,
        'context_cells': context_cells,
        'headline_geomean_speedup': geo,
        'speedup_range': [mn, mx],
        'fused_geomean_speedup': geo_fused,
    }
    results_dir = PROJECT_ROOT / 'benchmarks' / 'results'
    results_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = results_dir / f"fair_baseline_{stamp}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nResults saved: {path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
