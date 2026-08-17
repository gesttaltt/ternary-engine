#!/usr/bin/env python3
"""
phase5_error_characterization.py - TritNet Phase 5: is TritNet's imperfection
"approximate arithmetic" or just noise?

Three of the five Phase 2B/2A checkpoints are not 100% exact: tmul 99.4919%,
tmin 99.8882%, tmax 99.8510% (tadd and tnot are both 100%, and serve as a
zero-error control here). CLAUDE.md's Phase 5 goal is "explore approximate
arithmetic" -- this script asks the falsifiable question directly: are those
~0.1-0.5% of wrong samples structured (clustering near ternary-native
hierarchy boundaries, or near-tied decision margins -- "graceful" approximate
arithmetic) or statistically indistinguishable from noise scattered uniformly
across the input space?

Per .claude/CLAUDE.md's mandatory "CORRECT Ternary-Native Metrics" table,
this uses valuation depth and sparsity, NOT a binary/Euclidean notion of
"how wrong" a prediction is:

  - valuation depth v3(x): reuses the exact convention already established in
    research/scripts/falsify.py's load_ultrametric() (v3(n) = largest k such
    that 3^k | n, v3(0) = 999/"infinite"), applied to each 5-trit operand
    (A, or B) read as a single balanced-ternary integer sum(t_i * 3^i),
    i = 0..4 (trit index 0 = 3^0, i.e. LSB). For a 5-trit vector this means
    v3 in {0,1,2,3,4}, with a 5th bin ("all-zero") standing in for the
    v3=999 sentinel since 5 trits can't represent a larger finite depth.
    v3(operand) = k means the operand's lowest k trits are all zero -- k is
    literally "how many levels deep in the p-adic tree this operand's leaf
    sits below the root," matching CLAUDE.md's "Ultrametric transition cost"
    framing directly.
  - sparsity: fraction of the 10 input trits (5 for tnot) equal to 0.

Also checks logit margins: for every WRONG output-trit position, is the
model's own confidence near the decision boundary (a "near miss") or
confidently, arbitrarily wrong?

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

USAGE: python models/tritnet/phase5_error_characterization.py
OUTPUT: Per-op error-rate-by-bin tables, chi-square tests for non-uniform
        clustering, and a margin-distribution summary for wrong positions.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent.parent
EXPORT_DIR = PROJECT_ROOT / "models" / "tritnet" / "phase2b_export"
CKPT_DIRS = {
    'tnot': PROJECT_ROOT / "models" / "tritnet" / "phase2a" / "tnot",
    'tadd': PROJECT_ROOT / "models" / "tritnet" / "phase2b" / "tadd",
    'tmul': PROJECT_ROOT / "models" / "tritnet" / "phase2b" / "tmul",
    'tmin': PROJECT_ROOT / "models" / "tritnet" / "phase2b" / "tmin",
    'tmax': PROJECT_ROOT / "models" / "tritnet" / "phase2b" / "tmax",
}

_TRITS = (-1, 0, 1)
_POWERS5 = np.array([3 ** i for i in range(5)], dtype=np.int64)  # [1,3,9,27,81]


def _all_trit_vectors(n: int):
    if n == 0:
        yield []
        return
    for t in _TRITS:
        for rest in _all_trit_vectors(n - 1):
            yield [t] + rest


SCALAR_OPS = {
    'tnot': lambda a: -a,
    'tadd': lambda a, b: max(-1, min(1, a + b)),
    'tmul': lambda a, b: a * b,
    'tmin': min,
    'tmax': max,
}


def make_full_dataset(op_name: str):
    op = SCALAR_OPS[op_name]
    rows_x, rows_y = [], []
    if op_name == 'tnot':
        for vec_a in _all_trit_vectors(5):
            rows_x.append(vec_a)
            rows_y.append([op(t) for t in vec_a])
    else:
        for vec_a in _all_trit_vectors(5):
            for vec_b in _all_trit_vectors(5):
                rows_x.append(vec_a + vec_b)
                rows_y.append([op(a, b) for a, b in zip(vec_a, vec_b)])
    return np.array(rows_x, dtype=np.float32), np.array(rows_y, dtype=np.float32)


def forward(X: np.ndarray, out_dir: Path):
    """Same recipe as export_weights.py / test_tritnet_export.py's forward(),
    but also returns raw logits (needed for the margin analysis)."""
    W1 = np.load(out_dir / "W1.npy"); b1 = np.load(out_dir / "b1.npy")
    W2 = np.load(out_dir / "W2.npy"); b2 = np.load(out_dir / "b2.npy")
    W3 = np.load(out_dir / "W3.npy"); b3 = np.load(out_dir / "b3.npy")

    h1 = np.maximum(0.0, X @ W1 + b1)
    h2 = np.maximum(0.0, h1 @ W2 + b2)
    logits = h2 @ W3 + b3  # [N, n_out_trits * 3]

    n_out_trits = logits.shape[1] // 3
    logits = logits.reshape(-1, n_out_trits, 3)
    pred = logits.argmax(axis=2).astype(np.float32) - 1.0
    return pred, logits


# ---------------------------------------------------------------------------
# Ternary-native metrics: valuation depth (v3) and sparsity
# ---------------------------------------------------------------------------

_V3_LOOKUP = {}
for _n in range(-121, 122):
    _k = _n
    _v = 0
    if _k == 0:
        _V3_LOOKUP[_n] = 5  # sentinel: "all-zero" bin (deepest possible for 5 trits)
    else:
        while _k % 3 == 0:
            _k //= 3
            _v += 1
        _V3_LOOKUP[_n] = _v  # in {0,1,2,3,4}


def valuation_depth(trit_vecs: np.ndarray) -> np.ndarray:
    """v3 of each row, read as a balanced-ternary integer sum(t_i * 3^i).
    Same v3 definition as research/scripts/falsify.py's compute_3adic_valuation,
    reused via the "all-zero -> deepest bin" sentinel described in the module
    docstring instead of that function's v3(0)=999 (5 trits can't reach 999)."""
    values = (trit_vecs.astype(np.int64) * _POWERS5).sum(axis=1)
    return np.array([_V3_LOOKUP[v] for v in values], dtype=np.int64)


def sparsity(trit_vecs: np.ndarray) -> np.ndarray:
    """Fraction of trits equal to 0, per row."""
    return (trit_vecs == 0).mean(axis=1)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def chi_square_clustering(bin_ids: np.ndarray, wrong_mask: np.ndarray, n_bins: int, label: str):
    """Are wrong samples uniformly spread across bins (proportional to each
    bin's population), or clustered? Standard chi-square goodness-of-fit:
    expected wrong-count per bin = (bin's share of all samples) * total wrong."""
    total_wrong = wrong_mask.sum()
    if total_wrong == 0:
        print(f"    {label}: 0 errors -- nothing to test (exact op).")
        return

    bin_counts = np.bincount(bin_ids, minlength=n_bins).astype(np.float64)
    wrong_counts = np.bincount(bin_ids[wrong_mask], minlength=n_bins).astype(np.float64)
    expected = bin_counts * (total_wrong / bin_counts.sum())

    # Drop bins with 0 expected count (chi-square needs positive expectations)
    keep = expected > 0
    chi2, p = stats.chisquare(wrong_counts[keep], expected[keep])

    print(f"    {label}: chi2={chi2:.2f}  p={p:.4g}  "
          f"({'NON-UNIFORM (clustered)' if p < 0.01 else 'consistent with uniform'})")
    for b in range(n_bins):
        if bin_counts[b] == 0:
            continue
        rate = wrong_counts[b] / bin_counts[b]
        overall_rate = total_wrong / bin_counts.sum()
        print(f"      bin {b}: n={int(bin_counts[b]):>6}  wrong={int(wrong_counts[b]):>4}  "
              f"error_rate={rate*100:.4f}%  (overall={overall_rate*100:.4f}%, "
              f"{'x' + format(rate/overall_rate, '.2f') if overall_rate > 0 else 'n/a'})")


def analyze_op(op_name: str):
    out_dir = EXPORT_DIR / op_name
    if not out_dir.exists():
        print(f"{op_name}: SKIP (not exported)")
        return None

    X, Y = make_full_dataset(op_name)
    pred, logits = forward(X, out_dir)

    wrong_sample = (pred != Y).any(axis=1)  # sample-level: matches result.json's accuracy defn
    n_total = len(Y)
    n_wrong = int(wrong_sample.sum())
    print(f"\n=== {op_name} ===  n={n_total}  wrong={n_wrong}  "
          f"acc={100*(1 - n_wrong/n_total):.4f}%")

    if op_name == 'tnot':
        A = X
        depth = valuation_depth(A)
        spars = sparsity(A)
    else:
        A, B = X[:, :5], X[:, 5:]
        depth = np.minimum(valuation_depth(A), valuation_depth(B))  # "joint depth"
        spars = sparsity(X)  # over all 10 input trits

    print("  -- valuation depth (min(v3(A), v3(B)) for binary ops; v3(A) for tnot) --")
    chi_square_clustering(depth, wrong_sample, n_bins=6, label="depth bins 0-5")

    print("  -- sparsity (fraction of input trits == 0, binned into deciles) --")
    spars_bin = np.minimum((spars * 10).astype(np.int64), 9)  # 10 deciles, 0..9
    chi_square_clustering(spars_bin, wrong_sample, n_bins=10, label="sparsity deciles")

    # Margin analysis: for every WRONG output-trit position, how close was the
    # model's own logit ranking to getting it right?
    n_out = Y.shape[1]
    true_class = (Y + 1).astype(np.int64)  # {-1,0,1} -> {0,1,2}
    pos_wrong = (pred != Y)  # [N, n_out] per-position wrong mask

    if pos_wrong.any():
        idx_n, idx_pos = np.nonzero(pos_wrong)
        tc = true_class[idx_n, idx_pos]
        pos_logits = logits[idx_n, idx_pos, :]  # [n_wrong_positions, 3]
        true_logit = pos_logits[np.arange(len(tc)), tc]
        masked = pos_logits.copy()
        masked[np.arange(len(tc)), tc] = -np.inf
        best_other_logit = masked.max(axis=1)
        margin = true_logit - best_other_logit  # negative by construction (it's wrong)

        # Compare against margins on CORRECT positions for scale
        pos_right = ~pos_wrong
        idx_n2, idx_pos2 = np.nonzero(pos_right)
        tc2 = true_class[idx_n2, idx_pos2]
        pos_logits2 = logits[idx_n2, idx_pos2, :]
        true_logit2 = pos_logits2[np.arange(len(tc2)), tc2]
        masked2 = pos_logits2.copy()
        masked2[np.arange(len(tc2)), tc2] = -np.inf
        best_other_logit2 = masked2.max(axis=1)
        margin2 = true_logit2 - best_other_logit2  # positive by construction (it's right)

        print(f"  -- margin analysis ({len(margin)} wrong positions out of "
              f"{n_total * n_out} total) --")
        print(f"    wrong positions:   margin mean={margin.mean():.4f}  "
              f"median={np.median(margin):.4f}  min={margin.min():.4f}  max={margin.max():.4f}")
        print(f"    correct positions: margin mean={margin2.mean():.4f}  "
              f"median={np.median(margin2):.4f}  min={margin2.min():.4f}")
        near_miss_frac = (margin > -0.5).mean()
        print(f"    near-miss rate (margin > -0.5, i.e. a small logit swing "
              f"would have fixed it): {near_miss_frac*100:.2f}%")
    else:
        print("  -- margin analysis: no wrong positions (exact op) --")

    return {
        "n_total": n_total, "n_wrong": n_wrong,
        "acc": 1 - n_wrong / n_total,
    }


def main() -> int:
    if not EXPORT_DIR.exists():
        print(f"[SKIP] {EXPORT_DIR} does not exist -- run "
              f"'python models/tritnet/export_weights.py' first.")
        return 0

    print("=" * 78)
    print("TritNet Phase 5: Error characterization -- structured or noise?")
    print("=" * 78)

    results = {}
    for op_name in CKPT_DIRS:
        r = analyze_op(op_name)
        if r is not None:
            results[op_name] = r

    out_path = PROJECT_ROOT / "models" / "tritnet" / "phase5_error_characterization_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
