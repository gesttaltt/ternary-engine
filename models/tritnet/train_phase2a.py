#!/usr/bin/env python3
"""
train_phase2a.py - TritNet Phase 2A: tnot go/no-go gate

Train a neural network to learn the tnot operation (element-wise trit negation)
to 100% accuracy. This is the Phase 2A proof-of-concept decision gate.

Architecture decisions vs prior attempt (which peaked at 21.8%):
  - ReLU activations between layers (prior: no nonlinearity = pure linear chain)
  - CrossEntropy loss per trit position (prior: MSE regression)
  - Output: [batch, 5, 3] logits → argmax per trit (prior: continuous regression)
  - Then validate the learned solution survives ternary weight quantization

GO criterion: 100% exact-match accuracy with standard weights,
              ≥95% with ternary-quantized weights.

The QAT model classes, metrics, and checkpoint I/O below are shared with
train_phase2b.py via qat_common.py (extracted 2026-08-18, CLAUDE.md gap #7)
-- only this file's own dataset generation and seed-sweep training loop are
specific to Phase 2A.

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0
"""

import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "models" / "tritnet" / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from qat_common import (  # noqa: E402
    TritClassifier,
    TritClassifierFloat,
    exact_match_accuracy,
    logits_to_ternary,
    rescale_weights_for_qat,
    targets_to_class_idx,
    weight_distribution,
)
from qat_common import ckpt_path as _ckpt_path  # noqa: E402
from qat_common import load_result as _load_result  # noqa: E402
from qat_common import save_result as _save_result  # noqa: E402

# Checkpoint dir, mirroring train_phase2b.py's phase2b/<op>/{best_qat.pt,result.json}
# convention. Found 2026-08-14: this script trained tnot to its documented GO
# decision but never called torch.save anywhere -- the model was discarded after
# printing the decision, so there was no on-disk artifact for Phase 3 weight
# export to consume. Added save/resume here; the checkpoint I/O itself now lives
# in qat_common.py, shared with train_phase2b.py (CLAUDE.md gap #7).
CKPT_DIR = Path(__file__).parent / "phase2a"
CKPT_DIR.mkdir(exist_ok=True)

ckpt_path = partial(_ckpt_path, CKPT_DIR)
load_result = partial(_load_result, CKPT_DIR)
save_result = partial(_save_result, CKPT_DIR)


# ---------------------------------------------------------------------------
# Truth table generation (inline — no file dependency)
# ---------------------------------------------------------------------------

def make_tnot_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    """Generate all 3^5 = 243 tnot samples.

    Returns X [243, 5] and Y [243, 5], both in {-1, 0, +1}.
    tnot negates each trit: -1→+1, 0→0, +1→-1.
    """
    trits = [-1, 0, 1]
    rows_x, rows_y = [], []
    for a in trits:
        for b in trits:
            for c in trits:
                for d in trits:
                    for e in trits:
                        inp = [a, b, c, d, e]
                        rows_x.append(inp)
                        rows_y.append([-t for t in inp])
    X = torch.tensor(rows_x, dtype=torch.float32)
    Y = torch.tensor(rows_y, dtype=torch.float32)
    return X, Y


def make_tadd_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    """Generate all 3^10 = 59049 tadd samples.

    Returns X [59049, 10] and Y [59049, 5], both in {-1, 0, +1}.
    """
    from ternary_layers import count_parameters  # noqa: just verify import works
    try:
        import ternary_simd_engine as se
        use_simd = True
    except ImportError:
        use_simd = False

    trits = [-1, 0, 1]

    def tadd_scalar(a: int, b: int) -> int:
        s = a + b
        return max(-1, min(1, s))

    rows_x, rows_y = [], []
    for combo_a in range(243):
        vec_a = []
        n = combo_a
        for _ in range(5):
            vec_a.append(n % 3 - 1)
            n //= 3
        for combo_b in range(243):
            vec_b = []
            n = combo_b
            for _ in range(5):
                vec_b.append(n % 3 - 1)
                n //= 3
            result = [tadd_scalar(va, vb) for va, vb in zip(vec_a, vec_b)]
            rows_x.append(vec_a + vec_b)
            rows_y.append(result)

    X = torch.tensor(rows_x, dtype=torch.float32)
    Y = torch.tensor(rows_y, dtype=torch.float32)
    return X, Y


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_seed(
    X: torch.Tensor,
    Y: torch.Tensor,
    seed: int,
    hidden: int = 64,
    lr: float = 1e-3,
    max_epochs_p1: int = 500,
    max_epochs_p2: int = 5000,
    threshold: float = 0.3,
    verbose: bool = True,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    in_features = X.shape[1]
    n_out_trits = Y.shape[1]
    criterion = nn.CrossEntropyLoss()
    Y_idx = targets_to_class_idx(Y)

    # ------------------------------------------------------------------
    # Phase 1: float weights → reach 100%
    # ------------------------------------------------------------------
    float_model = TritClassifierFloat(in_features=in_features, hidden=hidden, n_out_trits=n_out_trits)
    opt1 = optim.Adam(float_model.parameters(), lr=lr)

    t0 = time.time()
    p1_acc = 0.0
    p1_epochs = 0
    for epoch in range(max_epochs_p1):
        float_model.train()
        opt1.zero_grad()
        logits = float_model(X)
        loss = criterion(logits.permute(0, 2, 1), Y_idx)
        loss.backward()
        opt1.step()

        float_model.eval()
        with torch.no_grad():
            p1_acc = exact_match_accuracy(float_model(X), Y)
        p1_epochs = epoch + 1
        if p1_acc >= 1.0:
            break

    if verbose:
        print(f"  Phase1 (float): acc={p1_acc*100:.1f}% in {p1_epochs} epochs")

    # ------------------------------------------------------------------
    # Phase 2: QAT from warm-start (weights rescaled above threshold)
    # ------------------------------------------------------------------
    qat_model = TritClassifier(
        in_features=in_features, hidden=hidden,
        n_out_trits=n_out_trits, threshold=threshold,
    )
    rescale_weights_for_qat(float_model, qat_model, threshold)

    # Verify accuracy is preserved after rescaling + quantization
    qat_model.eval()
    with torch.no_grad():
        acc_after_rescale = exact_match_accuracy(qat_model(X), Y)
    if verbose:
        print(f"  After rescale+QAT: acc={acc_after_rescale*100:.1f}%")

    opt2 = optim.Adam(qat_model.parameters(), lr=lr * 0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt2, patience=300, factor=0.5, min_lr=1e-6
    )

    best_acc = acc_after_rescale
    best_epoch = 0

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
        scheduler.step(loss.item())

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch

        if verbose and (epoch % 500 == 0 or acc >= 1.0):
            print(f"  Phase2(QAT) epoch={epoch:5d}  loss={loss.item():.4f}  "
                  f"acc={acc*100:.1f}%  best={best_acc*100:.1f}%  "
                  f"lr={opt2.param_groups[0]['lr']:.2e}")

        if best_acc >= 1.0:
            break

    elapsed = time.time() - t0

    # Weight distribution
    neg_frac, zero_frac, pos_frac = weight_distribution(qat_model)

    return {
        'seed': seed,
        'best_accuracy': best_acc,
        'best_epoch': best_epoch,
        'final_accuracy': best_acc,
        'ternary_accuracy': best_acc,  # QAT: quantized weights during forward
        'p1_acc': p1_acc,
        'acc_after_rescale': acc_after_rescale,
        'epochs_p1': p1_epochs,
        'epochs_p2': epoch + 1,
        'elapsed_s': elapsed,
        'converged': best_acc >= 1.0,
        'ternary_ok': best_acc >= 0.95,
        'weight_neg_pct': 100 * neg_frac,
        'weight_zero_pct': 100 * zero_frac,
        'weight_pos_pct': 100 * pos_frac,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("TritNet Phase 2A — tnot go/no-go gate")
    print("=" * 70)
    print()

    # --- tnot ---
    print("Operation: tnot (element-wise trit negation: -1→+1, 0→0, +1→-1)")
    print(f"Dataset:   243 samples (all 3^5 combinations), generated inline")
    print()

    X, Y = make_tnot_dataset()
    print(f"X shape: {X.shape}  Y shape: {Y.shape}")
    print(f"X range: {X.min().item():.0f} to {X.max().item():.0f}")
    print(f"Y range: {Y.min().item():.0f} to {Y.max().item():.0f}")
    print()

    # Resume: skip the sweep entirely if a passing checkpoint is already saved.
    saved = load_result('tnot')
    if saved and saved.get('passed'):
        print(f"[tnot] Already completed (best_acc={saved['best_acc']*100:.1f}%), "
              f"skipping training. Checkpoint: {ckpt_path('tnot', 'best_qat')}")
        return 0 if saved['best_acc'] >= 1.0 else 1

    seeds = [42, 123, 7]
    results = []

    for seed in seeds:
        print(f"--- Seed {seed} ---")
        r = train_one_seed(
            X, Y, seed=seed, hidden=64, lr=1e-3,
            max_epochs_p1=500, max_epochs_p2=5000, threshold=0.3,
        )
        results.append(r)
        print(f"  Result: ternary_acc={r['ternary_accuracy']*100:.1f}%  "
              f"p1={r['p1_acc']*100:.1f}%  rescale={r['acc_after_rescale']*100:.1f}%  "
              f"p1_ep={r['epochs_p1']}  p2_ep={r['epochs_p2']}  "
              f"t={r['elapsed_s']:.1f}s")
        print(f"  Weights: -{r['weight_neg_pct']:.1f}%  0={r['weight_zero_pct']:.1f}%  "
              f"+{r['weight_pos_pct']:.1f}%")
        print()

    # --- Summary and decision ---
    print("=" * 70)
    print("Phase 2A Summary — tnot")
    print("=" * 70)

    n_converged = sum(1 for r in results if r['converged'])
    n_ternary_ok = sum(1 for r in results if r['ternary_ok'])
    avg_epochs = np.mean([r['epochs_p1'] + r['epochs_p2'] for r in results])

    print(f"Seeds tested:            {len(seeds)}")
    print(f"Reached 100% (float):    {n_converged}/{len(seeds)}")
    print(f"Reached ≥95% (ternary):  {n_ternary_ok}/{len(seeds)}")
    print(f"Avg epochs to converge:  {avg_epochs:.0f}")
    print()

    # GO/NO-GO decision
    go = n_converged >= 2 and n_ternary_ok >= 2
    partial_go = n_converged >= 1

    if go:
        print("DECISION: GO")
        print("  tnot is reliably learnable to 100% AND survives ternary quantization.")
        print("  Proceed to Phase 2B: scale to tadd, tmul, tmin, tmax.")
    elif partial_go:
        print("DECISION: CONDITIONAL GO")
        print("  tnot converges to 100% on some seeds but ternary quantization degrades it.")
        print("  Next step: tune quantization threshold or use larger hidden size.")
    else:
        print("DECISION: NO-GO — investigate architecture")
        print("  tnot did not converge to 100% with this architecture.")
        print("  Consider: larger hidden, lower LR, different loss, or longer training.")

    print()

    # Quick tnot sanity check — verify the truth table
    print("Ternary truth table spot-check (best seed):")
    best_r = max(results, key=lambda r: r['best_accuracy'])
    best_seed = best_r['seed']

    # Rebuild the best model (two-phase)
    torch.manual_seed(best_seed)
    np.random.seed(best_seed)
    float_model = TritClassifierFloat(in_features=5, hidden=64, n_out_trits=5)
    opt_f = optim.Adam(float_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    Y_idx = targets_to_class_idx(Y)

    for epoch in range(best_r['epochs_p1']):
        float_model.train()
        opt_f.zero_grad()
        logits = float_model(X)
        loss = criterion(logits.permute(0, 2, 1), Y_idx)
        loss.backward()
        opt_f.step()
        with torch.no_grad():
            acc = exact_match_accuracy(float_model(X), Y)
        if acc >= 1.0:
            break

    model = TritClassifier(in_features=5, hidden=64, n_out_trits=5, threshold=0.3)
    rescale_weights_for_qat(float_model, model, threshold=0.3)

    opt_q = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(best_r['epochs_p2']):
        model.train()
        opt_q.zero_grad()
        logits = model(X)
        loss = criterion(logits.permute(0, 2, 1), Y_idx)
        loss.backward()
        opt_q.step()
        with torch.no_grad():
            acc = exact_match_accuracy(model(X), Y)
        if acc >= 1.0:
            break

    model.eval()
    with torch.no_grad():
        # Show a few predictions
        test_cases = [
            ([1, 0, -1, 1, 0], [-1, 0, 1, -1, 0]),
            ([-1, -1, -1, -1, -1], [1, 1, 1, 1, 1]),
            ([0, 0, 0, 0, 0], [0, 0, 0, 0, 0]),
        ]
        print(f"  {'Input':<25} {'Expected':<25} {'Predicted':<25} OK?")
        for inp, expected in test_cases:
            x_t = torch.tensor([inp], dtype=torch.float32)
            logits = model(x_t)
            pred = logits_to_ternary(logits)[0].tolist()
            ok = pred == expected
            print(f"  {str(inp):<25} {str(expected):<25} {str([int(p) for p in pred]):<25} {'YES' if ok else 'NO'}")

        # Recompute accuracy of THIS rebuilt model directly (don't trust the
        # sweep's recorded number blindly -- the rebuild is a separate,
        # deterministic-but-independent training run from the same seed).
        rebuilt_acc = exact_match_accuracy(model(X), Y)

    neg_frac, zero_frac, pos_frac = weight_distribution(model)

    # Persist the GO checkpoint. Previously this script trained tnot to its
    # documented decision and discarded the model -- no torch.save existed
    # anywhere in the file, so Phase 3 weight export had no tnot artifact to
    # consume despite CLAUDE.md documenting tnot as GO since Phase 2A.
    torch.save(model.state_dict(), ckpt_path('tnot', 'best_qat'))
    result = {
        'op': 'tnot',
        'seed': best_seed,
        'best_acc': rebuilt_acc,
        'converged': rebuilt_acc >= 0.9999,
        'passed': rebuilt_acc >= 0.99,
        'weight_neg_pct': 100 * neg_frac,
        'weight_zero_pct': 100 * zero_frac,
        'weight_pos_pct': 100 * pos_frac,
        'hidden': 64,
        'in_features': 5,
        'threshold': 0.3,
    }
    save_result('tnot', result)
    print(f"\n  Saved checkpoint: {ckpt_path('tnot', 'best_qat')}  "
          f"(rebuilt_acc={rebuilt_acc*100:.1f}%)")

    print()
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
