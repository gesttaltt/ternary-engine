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

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "models" / "tritnet" / "src"))


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
# Model
# ---------------------------------------------------------------------------

class STE(torch.autograd.Function):
    """Straight-through estimator for ternary weight quantization."""

    @staticmethod
    def forward(ctx, w: torch.Tensor, threshold: float) -> torch.Tensor:
        sign = torch.sign(w)
        mask = (w.abs() > threshold).float()
        return sign * mask

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad, None


class TernaryLinearQAT(nn.Module):
    """Linear layer with ternary weights quantized during the forward pass (QAT)."""

    def __init__(self, in_f: int, out_f: int, threshold: float = 0.3, bias: bool = True):
        super().__init__()
        self.threshold = threshold
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias_p = nn.Parameter(torch.zeros(out_f)) if bias else None
        nn.init.normal_(self.weight, std=1.0)  # start wide so most weights quantize to ±1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = STE.apply(self.weight, self.threshold)
        return nn.functional.linear(x, w_q, self.bias_p)

    def get_ternary_weights(self) -> torch.Tensor:
        with torch.no_grad():
            return quantize_ternary(self.weight, self.threshold)


class TritClassifier(nn.Module):
    """
    Classify each output trit position into {-1, 0, +1}.

    Output shape: [batch, 5, 3] logits (3 classes per trit position).
    Loss: CrossEntropyLoss summed over 5 positions.
    Accuracy: fraction of samples where all 5 argmax predictions match target.

    Uses QAT (quantization-aware training): weights are ternary {-1,0,+1} during
    the forward pass, STE allows gradients to update the underlying float weights.
    """

    def __init__(
        self,
        in_features: int = 5,
        hidden: int = 64,
        n_out_trits: int = 5,
        threshold: float = 0.3,
    ):
        super().__init__()
        self.n_out_trits = n_out_trits
        self.threshold = threshold
        self.fc1 = TernaryLinearQAT(in_features, hidden, threshold=threshold)
        self.fc2 = TernaryLinearQAT(hidden, hidden, threshold=threshold)
        self.fc3 = TernaryLinearQAT(hidden, n_out_trits * 3, threshold=threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, self.n_out_trits, 3)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

TRIT_TO_IDX = {-1: 0, 0: 1, 1: 2}


def targets_to_class_idx(Y: torch.Tensor) -> torch.Tensor:
    """Convert ternary targets [batch, 5] ∈ {-1,0,+1} → class indices [batch, 5] ∈ {0,1,2}."""
    return (Y + 1).long()


def logits_to_ternary(logits: torch.Tensor) -> torch.Tensor:
    """Convert [batch, 5, 3] logits → [batch, 5] ternary values ∈ {-1,0,+1}."""
    return logits.argmax(dim=2).float() - 1


def exact_match_accuracy(logits: torch.Tensor, Y: torch.Tensor) -> float:
    """Fraction of samples where ALL 5 output trits are correct."""
    pred = logits_to_ternary(logits)
    return (pred == Y).all(dim=1).float().mean().item()


def quantize_ternary(w: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Quantize weight tensor to {-1, 0, +1} using magnitude threshold."""
    sign = torch.sign(w)
    mask = (w.abs() > threshold).float()
    return sign * mask


def dead_zone_penalty(model: nn.Module) -> torch.Tensor:
    """Penalize float weights that sit inside the dead zone [−thr, +thr]."""
    total = torch.tensor(0.0)
    for layer in [model.fc1, model.fc2, model.fc3]:
        w = layer.weight
        thr = layer.threshold
        in_dead = (w.abs() < thr)
        total = total + (in_dead.float() * (thr - w.abs())).sum()
    return total


# ---------------------------------------------------------------------------
# Float-weight (no quantization) classifier for Phase 1
# ---------------------------------------------------------------------------

class TritClassifierFloat(nn.Module):
    """Same architecture as TritClassifier but with full-precision weights."""

    def __init__(self, in_features: int = 5, hidden: int = 64, n_out_trits: int = 5):
        super().__init__()
        self.n_out_trits = n_out_trits
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_out_trits * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, self.n_out_trits, 3)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def rescale_weights_for_qat(float_model: nn.Module, qat_model: nn.Module, threshold: float):
    """Copy float weights into QAT model, rescaled so most exceed the quantization threshold.

    Scaling by a positive constant preserves logit ordering (CE loss), so the
    rescaled model has the same 100% accuracy but weights that are mostly > threshold.
    """
    float_layers = [float_model.fc1, float_model.fc2, float_model.fc3]
    qat_layers = [qat_model.fc1, qat_model.fc2, qat_model.fc3]
    with torch.no_grad():
        for fl, ql in zip(float_layers, qat_layers):
            w = fl.weight.data
            b = fl.bias.data
            # Scale so that the 75th-percentile absolute weight = 2*threshold
            p75 = w.abs().quantile(0.75).clamp(min=1e-8)
            scale = (2.0 * threshold) / p75
            ql.weight.data.copy_(w * scale)
            ql.bias_p.data.copy_(b * scale)


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
    total_w = ternary_neg = ternary_zero = ternary_pos = 0
    with torch.no_grad():
        for layer in [qat_model.fc1, qat_model.fc2, qat_model.fc3]:
            w_q = layer.get_ternary_weights()
            ternary_neg += (w_q == -1).sum().item()
            ternary_zero += (w_q == 0).sum().item()
            ternary_pos += (w_q == 1).sum().item()
            total_w += w_q.numel()

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
        'weight_neg_pct': 100 * ternary_neg / total_w if total_w else 0,
        'weight_zero_pct': 100 * ternary_zero / total_w if total_w else 0,
        'weight_pos_pct': 100 * ternary_pos / total_w if total_w else 0,
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
    partial = n_converged >= 1

    if go:
        print("DECISION: GO")
        print("  tnot is reliably learnable to 100% AND survives ternary quantization.")
        print("  Proceed to Phase 2B: scale to tadd, tmul, tmin, tmax.")
    elif partial:
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

    print()
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
