#!/usr/bin/env python3
"""
test_tritnet_export.py - Verify TritNet weight export fidelity

Reloads models/tritnet/phase2b_export/<op>/*.npy (produced by
models/tritnet/export_weights.py) and runs a pure-NumPy forward pass over the
FULL input space for each of the 5 real GO-decision operations (tnot, tadd,
tmul, tmin, tmax), independent of PyTorch/the original TritClassifier class.
Confirms the exported weights reproduce each operation's checkpoint accuracy
recorded in its result.json -- the fidelity check that must hold before any
C++ inference engine (Phase 3) can be trusted against these weights.

Exits 0 with a [SKIP] notice if the export hasn't been generated yet (it is
a generated artifact, produced by running export_weights.py); this test
does not itself require PyTorch.

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

USAGE: python tests/python/test_tritnet_export.py
OUTPUT: Pass/fail per operation; exit 0 if all exported ops reproduce their
        recorded checkpoint accuracy, exit 1 otherwise.
"""

import json
import sys
from pathlib import Path

import numpy as np

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


def make_dataset(op_name: str):
    """Full input space: 3^5=243 for unary, 3^10=59,049 for binary."""
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


def forward(X: np.ndarray, out_dir: Path) -> np.ndarray:
    """Pure-NumPy replay of TritClassifier.forward: 2x ReLU hidden layers,
    linear output reshaped to [N, n_out_trits, 3], argmax-decoded to a trit."""
    W1 = np.load(out_dir / "W1.npy"); b1 = np.load(out_dir / "b1.npy")
    W2 = np.load(out_dir / "W2.npy"); b2 = np.load(out_dir / "b2.npy")
    W3 = np.load(out_dir / "W3.npy"); b3 = np.load(out_dir / "b3.npy")

    h1 = np.maximum(0.0, X @ W1 + b1)
    h2 = np.maximum(0.0, h1 @ W2 + b2)
    logits = h2 @ W3 + b3  # [N, n_out_trits * 3]

    n_out_trits = logits.shape[1] // 3
    logits = logits.reshape(-1, n_out_trits, 3)
    return logits.argmax(axis=2).astype(np.float32) - 1.0


def check_op(op_name: str) -> bool:
    out_dir = EXPORT_DIR / op_name
    if not out_dir.exists():
        print(f"  {op_name}: SKIP (not exported, run export_weights.py first)")
        return True

    result_json = CKPT_DIRS[op_name] / "result.json"
    if not result_json.exists():
        print(f"  {op_name}: FAIL (exported but no result.json to verify against: {result_json})")
        return False
    recorded_acc = json.loads(result_json.read_text())['best_acc']

    X, Y = make_dataset(op_name)
    pred = forward(X, out_dir)

    exact_match = (pred == Y).all(axis=1)
    replay_acc = exact_match.mean()

    # Exact fidelity: same deterministic weights replayed outside PyTorch should
    # reproduce the recorded checkpoint accuracy to within float32 argmax-tie noise.
    ok = abs(replay_acc - recorded_acc) < 1e-6
    status = "PASS" if ok else "FAIL"
    print(f"  {op_name}: {status}  replay_acc={replay_acc*100:.4f}%  "
          f"recorded_acc={recorded_acc*100:.4f}%  (n={len(Y)})")
    return ok


def main() -> int:
    print("=" * 70)
    print("TritNet Export Fidelity Test")
    print("=" * 70)

    if not EXPORT_DIR.exists():
        print(f"[SKIP] {EXPORT_DIR} does not exist -- run "
              f"'python models/tritnet/export_weights.py' first (optional artifact).")
        return 0

    results = {op: check_op(op) for op in CKPT_DIRS}

    n_pass = sum(results.values())
    n_total = len(results)
    print(f"\n{n_pass}/{n_total} operations verified")

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
