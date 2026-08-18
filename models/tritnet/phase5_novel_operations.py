#!/usr/bin/env python3
"""
phase5_novel_operations.py - TritNet Phase 5: "discover novel ternary operations"

The last unstarted bullet of Phase 5 (Learned Generalization). Phase 4 and
Phase 5's first two results already closed off "TritNet as a faster or more
capable *replacement* for tadd/tmul/tmin/tmax/tnot" -- both direct CPU LUTs
and direct GPU arithmetic beat it on throughput and exactness. Per this
project's own conclusion (reports/2026-08-17/TRITNET_PHASE5_SESSION_REPORT.md
Section 5): "If TritNet has a real niche, it's not as a replacement for exact
per-trit-chunk arithmetic -- it would need an operation without a cheap
closed form." This script asks that question directly and rigorously.

METHOD (four falsifiable stages, each independently checkable):

1. ENUMERATE the full space of single-trit binary ternary operations:
   {-1,0,+1} x {-1,0,+1} -> {-1,0,+1}, all 3^9 = 19,683 possible truth
   tables. (Unary operations are a much smaller space, 3^3 = 27, and tnot
   is already the only unary op this project defines; not the focus here.)

2. DEDUPLICATE by symmetry: an operation and its "obvious variants" (swap
   the arguments, negate either input, negate the output, or any
   composition of these -- a 16-element group) are the same underlying
   idea wearing different clothes, not 16 independent discoveries. Reduces
   19,683 raw truth tables to their equivalence classes, and excludes
   every class containing tadd, tmul, tmin, or tmax.

3. FILTER "cheap closed form" two ways, since neither alone is a complete
   proxy for "a hardware/software engineer could just write this by hand":
     a. Curated catalog: does the operation match (up to the same 16-
        element symmetry) any of a bounded set of short compositions of
        standard ternary primitives (saturating add/sub, multiply, min,
        max, comparison, negation)? This is the "would anyone just write
        this in a few lines of C" check.
     b. GF(3) algebraic degree: every function on a finite field has a
        polynomial representation (Lagrange interpolation); its degree is
        a rigorous, well-defined, computable complexity measure independent
        of any curated list. Low degree (0-2) means a short polynomial
        formula exists even if it's not in the curated catalog. NOTE:
        min/max are algebraically high-degree despite being cheap in
        hardware (comparison-based, not polynomial) -- degree alone isn't
        sufficient, which is exactly why check (a) exists too; an operation
        only counts as "no cheap closed form" if it fails BOTH checks.

4. SCORE the survivors with this project's own established ternary-native
   metrics (CLAUDE.md's own mandated table: valuation depth, sparsity,
   associativity/distributivity, information content -- the same convention
   research/scripts/falsify.py and phase5_error_characterization.py already
   use) and TRAIN the most interesting one(s) with the exact TritNet
   architecture/hyperparameters that produced the documented tadd/tmul/
   tmin/tmax GO checkpoints, to answer the concrete question: is a
   genuinely closed-form-resistant operation still learnable by a small
   ternary network? That's the shape any real TritNet niche would have to
   take.

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

USAGE: python models/tritnet/phase5_novel_operations.py
OUTPUT: models/tritnet/phase5_novel_operations_results.json
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "models" / "tritnet"))

from qat_common import (  # noqa: E402
    TritClassifier,
    TritClassifierFloat,
    exact_match_accuracy,
    rescale_weights_for_qat,
    targets_to_class_idx,
    weight_distribution,
)

TRITS = (-1, 0, 1)
PAIRS = tuple((a, b) for a in TRITS for b in TRITS)  # canonical 9-input order


# =============================================================================
# Stage 1-2: enumerate + deduplicate by symmetry
# =============================================================================

def all_truth_tables():
    """All 3^9 = 19,683 possible binary ternary truth tables, as 9-tuples of
    outputs in PAIRS order."""
    return itertools.product(TRITS, repeat=9)


def transform(tt_dict, swap, na, nb, no):
    """Apply one of the 16 symmetry-group elements to a truth table (given as
    a dict {(a,b): out}), returning the resulting truth table as a tuple."""
    out = []
    for a, b in PAIRS:
        aa, bb = (b, a) if swap else (a, b)
        aa = -aa if na else aa
        bb = -bb if nb else bb
        v = tt_dict[(aa, bb)]
        out.append(-v if no else v)
    return tuple(out)


def symmetry_orbit(tt: tuple) -> frozenset:
    """The 16-element orbit of `tt` under {swap, negate-a, negate-b,
    negate-output} and their compositions."""
    tt_dict = dict(zip(PAIRS, tt))
    orbit = set()
    for swap in (False, True):
        for na in (False, True):
            for nb in (False, True):
                for no in (False, True):
                    orbit.add(transform(tt_dict, swap, na, nb, no))
    return frozenset(orbit)


def canonicalize(tt: tuple) -> tuple:
    """A deterministic representative of tt's orbit (the lexicographically
    smallest member) -- used as a dict key so equivalent operations collapse
    to one entry regardless of which orbit member was generated first."""
    return min(symmetry_orbit(tt))


def _tadd(a, b): return max(-1, min(1, a + b))
def _tmul(a, b): return a * b
def _tmin(a, b): return min(a, b)
def _tmax(a, b): return max(a, b)

NAMED_OPS = {'tadd': _tadd, 'tmul': _tmul, 'tmin': _tmin, 'tmax': _tmax}


def named_op_truth_table(fn) -> tuple:
    return tuple(fn(a, b) for a, b in PAIRS)


# =============================================================================
# Stage 3a: curated "cheap closed form" catalog
# =============================================================================
# Short compositions of standard ternary primitives -- the "would a hardware
# engineer just write this by hand" check. Each is defined once; its orbit
# (not just its literal truth table) is excluded, so e.g. tsub(a,b)=tadd(a,-b)
# also rules out tadd(-a,b), tadd(b,-a), etc. automatically.

def _clip(x): return max(-1, min(1, x))

_UNARY = {'id': lambda x: x, 'neg': lambda x: -x}
_BINARY_COMBINATORS = {
    'add_sat': lambda x, y: _clip(x + y),
    'sub_sat': lambda x, y: _clip(x - y),
    'mul': lambda x, y: x * y,
    'min': min,
    'max': max,
    'add2_sat': lambda x, y: _clip(2 * x + y),   # weighted sum, still O(1)
    'avg_round': lambda x, y: int(np.sign(x + y)) if (x + y) != 0 else 0,
    'eq': lambda x, y: 1 if x == y else -1,       # comparison, O(1) in hardware
    'gt': lambda x, y: 1 if x > y else (-1 if x < y else 0),
}


def build_cheap_catalog() -> set:
    """Every truth table (as its canonical orbit representative) reachable by
    one binary combinator applied to (u1(a), u2(b)) for u1,u2 in {id, neg} --
    a depth-1 formula. Bounded, not exhaustive of all possible short
    formulas (that's an open-ended program-synthesis problem), but covers
    every standard ternary arithmetic/comparison primitive this project
    documents or a reasonable engineer would reach for first.
    """
    catalog = set()
    for combinator in _BINARY_COMBINATORS.values():
        for u1 in _UNARY.values():
            for u2 in _UNARY.values():
                tt = tuple(combinator(u1(a), u2(b)) for a, b in PAIRS)
                catalog.add(canonicalize(tt))
    return catalog


# =============================================================================
# Stage 3b: GF(3) algebraic degree (Lagrange interpolation)
# =============================================================================
# Every function on GF(3)^2 -> GF(3) has a unique polynomial representation
# of the form sum_{i,j in {0,1,2}} c_ij * a^i * b^j (since a^3 = a for all
# a in GF(3), by Fermat's little theorem, so degree in each variable is
# capped at 2). This computes that representation exactly via Lagrange
# interpolation over GF(3) and returns its total degree
# (max over nonzero-coefficient terms of i+j) -- 0 for constants, up to 4
# for a term needing a^2*b^2. Values are worked in GF(3) = {0,1,2}, not
# balanced-ternary {-1,0,1}; the two encodings are related by x -> x-1.

def _gf3_inv(x):
    # Every nonzero element of GF(3) is its own inverse under multiplication
    # except none needed here (only used for interpolation denominators,
    # which are always 1 or 2, and 2*2=4=1 mod 3, so 2 is self-inverse too).
    return {0: None, 1: 1, 2: 2}[x % 3]


def gf3_poly_degree(tt_balanced: tuple) -> int:
    """Algebraic degree over GF(3) of the function whose truth table (in
    PAIRS order, balanced-ternary {-1,0,1} values) is `tt_balanced`.

    Solves for the unique coefficients c_ij (i,j in {0,1,2}) of
    f(a,b) = sum c_ij * a^i * b^j (mod 3) by exact linear solve over GF(3)
    (9 equations -- one per input pair -- 9 unknowns -- one per monomial),
    using the Vandermonde-like structure of evaluating a^i*b^j at each of
    the 9 points. Returns max(i+j) over monomials with c_ij != 0, or -1 for
    the all-zero function (degree of the zero polynomial is conventionally
    undefined/-infinity; -1 here is just a sentinel below every real degree).
    """
    # GF(3) domain points, in the same order as PAIRS but shifted to {0,1,2}
    gf3_pairs = [((a + 1) % 3, (b + 1) % 3) for a, b in PAIRS]
    monomials = [(i, j) for i in range(3) for j in range(3)]  # 9 monomials

    # Build the 9x9 Vandermonde-style matrix M where M[k][m] = a_k^i * b_k^j
    # for point k and monomial m=(i,j), all arithmetic mod 3.
    M = np.zeros((9, 9), dtype=np.int64)
    for k, (a, b) in enumerate(gf3_pairs):
        for m, (i, j) in enumerate(monomials):
            M[k, m] = (pow(a, i, 3) * pow(b, j, 3)) % 3

    targets = np.array([(v + 1) % 3 for v in tt_balanced], dtype=np.int64)

    # Solve M @ c = targets over GF(3) via Gaussian elimination mod 3
    # (numpy's float solver isn't exact enough to trust over a finite field;
    # 9x9 is small enough to hand-roll exact integer elimination).
    A = np.concatenate([M.copy(), targets.reshape(-1, 1)], axis=1) % 3
    n = 9
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, n):
            if A[r, col] % 3 != 0:
                pivot = r
                break
        if pivot is None:
            continue
        A[[row, pivot]] = A[[pivot, row]]
        inv = _gf3_inv(int(A[row, col]) % 3)
        A[row] = (A[row] * inv) % 3
        for r in range(n):
            if r != row and A[r, col] % 3 != 0:
                factor = A[r, col] % 3
                A[r] = (A[r] - factor * A[row]) % 3
        row += 1
        if row == n:
            break

    # Back out coefficients: after elimination each row's leading 1 gives one
    # coefficient directly (the matrix is square and this construction is
    # always full rank -- Lagrange interpolation over a field always has a
    # unique solution -- so this recovers c_ij exactly).
    coeffs = np.zeros(9, dtype=np.int64)
    for r in range(n):
        nz = np.nonzero(A[r, :9] % 3)[0]
        if len(nz) == 1:
            coeffs[nz[0]] = A[r, 9] % 3

    degree = -1
    for m, (i, j) in enumerate(monomials):
        if coeffs[m] % 3 != 0:
            degree = max(degree, i + j)
    return degree


# =============================================================================
# Stage 4: ternary-native scoring (CLAUDE.md's mandated metrics)
# =============================================================================

def v3(n: int) -> int:
    """3-adic valuation, matching research/scripts/falsify.py's exact
    convention: largest k such that 3^k | n; v3(0) = 999 sentinel."""
    if n == 0:
        return 999
    n = abs(n)
    k = 0
    while n % 3 == 0:
        n //= 3
        k += 1
    return k


def score_operation(fn) -> dict:
    """Ternary-native metrics for a candidate binary op, sampling associativity
    and distributivity over a fixed grid (all 27 triples for associativity,
    all 27 triples x each named op for distributivity -- small enough to be
    exhaustive, not sampled)."""
    zero_count = sum(1 for a, b in PAIRS if fn(a, b) == 0)
    sparsity = zero_count / 9

    assoc_ok = 0
    for a, b, c in itertools.product(TRITS, repeat=3):
        if fn(fn(a, b), c) == fn(a, fn(b, c)):
            assoc_ok += 1
    associativity = assoc_ok / 27

    comm_ok = sum(1 for a, b in PAIRS if fn(a, b) == fn(b, a))
    commutativity = comm_ok / 9

    # Distributes over tadd? f(a, tadd(b,c)) == tadd(f(a,b), f(a,c))
    dist_ok = 0
    for a, b, c in itertools.product(TRITS, repeat=3):
        if fn(a, _tadd(b, c)) == _tadd(fn(a, b), fn(a, c)):
            dist_ok += 1
    distributes_over_tadd = dist_ok / 27

    # How many of the 3 possible output values actually appear, and how
    # balanced is their spread? A function that only ever outputs 2 of the 3
    # trit values (or is near-constant) is a degenerate case of "novel" --
    # it's really a 2-valued (Boolean-like) operation wearing a ternary
    # domain, not a genuine 3-valued ternary operation. n_output_values==3
    # plus a high balance is the non-degenerate case worth highlighting.
    outputs = [fn(a, b) for a, b in PAIRS]
    counts = {v: outputs.count(v) for v in set(outputs)}
    n_output_values = len(counts)
    balance = (min(counts.values()) / max(counts.values())) if counts else 0.0

    return {
        'sparsity': sparsity,
        'associativity': associativity,
        'commutativity': commutativity,
        'distributes_over_tadd': distributes_over_tadd,
        'n_output_values': n_output_values,
        'balance': balance,
    }


# =============================================================================
# Discovery pipeline
# =============================================================================

def discover_candidates(verbose=True):
    if verbose:
        print("Enumerating all 3^9 = 19,683 binary ternary truth tables...")
    t0 = time.time()

    named_orbits = set()
    for fn in NAMED_OPS.values():
        named_orbits |= symmetry_orbit(named_op_truth_table(fn))

    cheap_catalog = build_cheap_catalog()

    seen_classes = {}  # canonical rep -> representative truth table
    for tt in all_truth_tables():
        canon = canonicalize(tt)
        if canon not in seen_classes:
            seen_classes[canon] = tt

    if verbose:
        print(f"  {len(seen_classes):,} equivalence classes under the 16-element "
              f"symmetry group ({time.time()-t0:.1f}s)")

    candidates = []
    for canon, tt in seen_classes.items():
        if canon in named_orbits:
            continue  # equivalent to tadd/tmul/tmin/tmax
        if canon in cheap_catalog:
            continue  # matches a curated cheap-closed-form formula
        degree = gf3_poly_degree(tt)
        if degree <= 2:
            continue  # low GF(3) algebraic degree -- a short polynomial exists
        candidates.append((canon, tt, degree))

    if verbose:
        print(f"  {len(candidates):,} candidates survive both cheap-closed-form "
              f"filters (curated catalog + GF(3) degree <= 2)")

    scored = []
    for canon, tt, degree in candidates:
        fn = lambda a, b, _tt=dict(zip(PAIRS, tt)): _tt[(a, b)]
        metrics = score_operation(fn)
        scored.append({
            'truth_table': tt,
            'gf3_degree': degree,
            **metrics,
        })

    return scored, len(seen_classes), len(candidates)


# =============================================================================
# Training: is the top candidate actually learnable by TritNet?
# =============================================================================

def make_binary_dataset_for_tt(tt: tuple):
    """Generate all 3^10 = 59,049 5-trit-chunk samples for a candidate scalar
    op, exactly mirroring train_phase2b.py's make_binary_dataset -- the
    scalar op is applied elementwise across each 5-trit chunk, same
    convention as tadd/tmul/tmin/tmax."""
    tt_dict = dict(zip(PAIRS, tt))
    scalar_op = lambda a, b: tt_dict[(a, b)]

    def all_trit_vectors(n):
        if n == 0:
            yield []
            return
        for t in TRITS:
            for rest in all_trit_vectors(n - 1):
                yield [t] + rest

    rows_x, rows_y = [], []
    for vec_a in all_trit_vectors(5):
        for vec_b in all_trit_vectors(5):
            result = [scalar_op(a, b) for a, b in zip(vec_a, vec_b)]
            rows_x.append(vec_a + vec_b)
            rows_y.append(result)
    X = torch.tensor(rows_x, dtype=torch.float32)
    Y = torch.tensor(rows_y, dtype=torch.float32)
    return X, Y


def train_on_candidate(tt: tuple, seed: int = 42, hidden: int = 128,
                        threshold: float = 0.3, max_epochs_p1: int = 3000,
                        max_epochs_p2: int = 10000) -> dict:
    """Train a TritClassifier on a discovered operation, using the exact
    architecture/hyperparameters that produced the documented tadd/tmul/
    tmin/tmax GO checkpoints (train_phase2b.py), for a directly comparable
    accuracy figure."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, Y = make_binary_dataset_for_tt(tt)
    criterion = nn.CrossEntropyLoss()
    Y_idx = targets_to_class_idx(Y)

    float_model = TritClassifierFloat(in_features=10, hidden=hidden, n_out_trits=5)
    opt1 = optim.Adam(float_model.parameters(), lr=1e-3)
    sch1 = optim.lr_scheduler.ReduceLROnPlateau(opt1, patience=300, factor=0.5, min_lr=1e-5)

    t0 = time.time()
    p1_acc, p1_epochs = 0.0, 0
    for epoch in range(max_epochs_p1):
        float_model.train()
        opt1.zero_grad()
        logits = float_model(X)
        loss = criterion(logits.permute(0, 2, 1), Y_idx)
        loss.backward()
        opt1.step()
        sch1.step(loss.item())
        float_model.eval()
        with torch.no_grad():
            p1_acc = exact_match_accuracy(float_model(X), Y)
        p1_epochs = epoch + 1
        if p1_acc >= 1.0:
            break
    print(f"  Phase1 (float): acc={p1_acc*100:.2f}% in {p1_epochs} epochs "
          f"({time.time()-t0:.1f}s)")

    qat_model = TritClassifier(in_features=10, hidden=hidden, n_out_trits=5, threshold=threshold)
    rescale_weights_for_qat(float_model, qat_model, threshold)
    qat_model.eval()
    with torch.no_grad():
        acc_rescale = exact_match_accuracy(qat_model(X), Y)
    print(f"  Phase2 starting accuracy: {acc_rescale*100:.2f}%")

    opt2 = optim.Adam(qat_model.parameters(), lr=1e-4)
    sch2 = optim.lr_scheduler.ReduceLROnPlateau(opt2, patience=500, factor=0.5, min_lr=1e-6)
    best_acc, best_epoch = acc_rescale, 0
    t0 = time.time()
    for epoch in range(max_epochs_p2):
        qat_model.train()
        opt2.zero_grad()
        logits = qat_model(X)
        loss = criterion(logits.permute(0, 2, 1), Y_idx)
        loss.backward()
        opt2.step()
        qat_model.eval()
        with torch.no_grad():
            acc = exact_match_accuracy(qat_model(X), Y)
        sch2.step(loss.item())
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
        if epoch % 500 == 0:
            print(f"  P2 epoch={epoch:5d} loss={loss.item():.4f} exact={acc*100:.2f}% "
                  f"best={best_acc*100:.2f}%")
        if best_acc >= 1.0:
            break
    elapsed = time.time() - t0
    neg, zero, pos = weight_distribution(qat_model)
    print(f"  Phase2 done: best={best_acc*100:.2f}% at epoch {best_epoch} ({elapsed:.1f}s)")

    return {
        'p1_acc': p1_acc, 'p1_epochs': p1_epochs,
        'acc_after_rescale': acc_rescale,
        'best_acc': best_acc, 'best_epoch': best_epoch,
        'converged': best_acc >= 0.9999,
        'passed': best_acc >= 0.99,
        'weight_neg_pct': neg * 100, 'weight_zero_pct': zero * 100, 'weight_pos_pct': pos * 100,
        'elapsed_s': elapsed,
    }


def main():
    print("=" * 78)
    print("TritNet Phase 5 -- Discover Novel Ternary Operations")
    print("=" * 78)

    scored, n_classes, n_candidates = discover_candidates()

    if not scored:
        print("\nNo candidates survived both filters -- every equivalence class is "
              "either a named op or has a cheap closed form.")
        result = {'n_equivalence_classes': n_classes, 'n_candidates': 0,
                  'candidates': [], 'trained': None}
    else:
        # Primary ranking: associativity first. It's the single property most
        # practically valuable for an operation (enables parallel reduction
        # trees, associative-scan algorithms, etc.) and one tadd itself
        # notably lacks -- H24 (research/scripts/falsify.py) found tadd
        # non-associative for 79.6% of triplets, so full associativity in an
        # unnamed candidate is a genuinely distinguishing property, not a
        # given. Among fully-associative candidates, prefer non-degenerate
        # ones: a function using only 2 of the 3 trit values as output is
        # really a Boolean function wearing a ternary domain, not a
        # meaningfully "ternary" operation -- checked explicitly rather than
        # assumed, since an early exploratory pass during this session
        # wrongly concluded no 3-valued fully-associative survivor existed
        # before a careful from-scratch recount found 17 of them.
        scored.sort(key=lambda c: (
            -c['associativity'],
            -(c['n_output_values'] == 3),
            -c['balance'],
            -c['gf3_degree'],
        ))

        n_fully_assoc = sum(1 for c in scored if c['associativity'] == 1.0)
        n_fully_assoc_3val = sum(1 for c in scored
                                  if c['associativity'] == 1.0 and c['n_output_values'] == 3)
        print(f"\n{n_fully_assoc} of {n_candidates} novel candidates are fully associative "
              f"(compare: tadd itself is non-associative for 79.6% of triplets, H24)")
        print(f"{n_fully_assoc_3val} of those are non-degenerate (all 3 trit values appear "
              f"as outputs, not just 2)")

        print(f"\nTop 10 candidates (associativity, non-degeneracy, balance, GF(3) degree):")
        print(f"{'assoc':>6} {'nvals':>6} {'balance':>8} {'degree':>6} {'sparsity':>9} {'comm':>6}")
        for c in scored[:10]:
            print(f"{c['associativity']:>6.3f} {c['n_output_values']:>6} {c['balance']:>8.3f} "
                  f"{c['gf3_degree']:>6} {c['sparsity']:>9.3f} {c['commutativity']:>6.3f}")

        top = scored[0]
        print(f"\nTraining TritNet on the top candidate "
              f"(associativity={top['associativity']:.3f}, "
              f"n_output_values={top['n_output_values']}, balance={top['balance']:.3f}, "
              f"GF(3) degree={top['gf3_degree']})...")
        print(f"Truth table (PAIRS order {PAIRS}):")
        print(f"  {top['truth_table']}")

        train_result = train_on_candidate(top['truth_table'])

        result = {
            'n_equivalence_classes': n_classes,
            'n_candidates': n_candidates,
            'n_fully_associative': n_fully_assoc,
            'n_fully_associative_nondegenerate': n_fully_assoc_3val,
            'top_candidates': scored[:20],
            'trained_candidate': top,
            'training_result': train_result,
        }

    out_path = Path(__file__).parent / "phase5_novel_operations_results.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Equivalence classes (of 19,683 raw truth tables): {n_classes:,}")
    print(f"Survive both cheap-closed-form filters: {n_candidates:,} "
          f"({100*n_candidates/n_classes:.1f}% of classes)")
    if scored:
        t = result['training_result']
        print(f"Top candidate trained to: {t['best_acc']*100:.2f}% "
              f"({'PASS' if t['passed'] else 'FAIL'} >=99% threshold)")


if __name__ == "__main__":
    main()
