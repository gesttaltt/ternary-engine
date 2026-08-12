#!/usr/bin/env python3
"""
train_phase2b.py - TritNet Phase 2B: binary operations (tadd, tmul, tmin, tmax)

Extends Phase 2A (tnot GO decision) to all four binary operations.
Uses the same two-phase QAT approach that achieved 100% on tnot.

GO criterion (per CLAUDE.md): >=3 of 4 operations reach >=99% with ternary weights,
                               with >=1 at 100%.

Architecture:
  Input:  10 trits (5 from A || 5 from B)
  Hidden: 2 layers of 128 neurons + ReLU
  Output: 5 x 3 logits (3 classes per output trit position)

Training (same two-phase as Phase 2A):
  Phase 1 -- float weights, CrossEntropy loss, ReLU activations -> reach 100%
  Phase 2 -- QAT warm-start (rescale weights above threshold) -> 100% with ternary weights

Checkpoints saved to models/tritnet/phase2b/<op>/best_float.pt and best_qat.pt
Resume supported: completed operations are skipped on re-run.

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).parent.parent.parent
CKPT_DIR = Path(__file__).parent / "phase2b"
CKPT_DIR.mkdir(exist_ok=True)


def log(*args, **kwargs):
    kwargs['flush'] = True
    print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Scalar ternary operations
# ---------------------------------------------------------------------------

def _tadd(a: int, b: int) -> int:
    return max(-1, min(1, a + b))

def _tmul(a: int, b: int) -> int:
    return a * b

def _tmin(a: int, b: int) -> int:
    return min(a, b)

def _tmax(a: int, b: int) -> int:
    return max(a, b)

SCALAR_OPS = {
    'tadd': _tadd,
    'tmul': _tmul,
    'tmin': _tmin,
    'tmax': _tmax,
}


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

_TRITS = [-1, 0, 1]

def _all_trit_vectors(n: int):
    if n == 0:
        yield []
        return
    for t in _TRITS:
        for rest in _all_trit_vectors(n - 1):
            yield [t] + rest


def make_binary_dataset(op_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate all 3^10 = 59,049 samples for a binary 5-trit operation."""
    scalar_op = SCALAR_OPS[op_name]
    rows_x, rows_y = [], []
    for vec_a in _all_trit_vectors(5):
        for vec_b in _all_trit_vectors(5):
            result = [scalar_op(a, b) for a, b in zip(vec_a, vec_b)]
            rows_x.append(vec_a + vec_b)
            rows_y.append(result)
    X = torch.tensor(rows_x, dtype=torch.float32)
    Y = torch.tensor(rows_y, dtype=torch.float32)
    return X, Y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, threshold):
        sign = torch.sign(w)
        mask = (w.abs() > threshold).float()
        return sign * mask

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class TernaryLinearQAT(nn.Module):
    def __init__(self, in_f, out_f, threshold=0.3, bias=True):
        super().__init__()
        self.threshold = threshold
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias_p = nn.Parameter(torch.zeros(out_f)) if bias else None
        nn.init.normal_(self.weight, std=1.0)

    def forward(self, x):
        w_q = STE.apply(self.weight, self.threshold)
        return nn.functional.linear(x, w_q, self.bias_p)

    def get_ternary_weights(self):
        with torch.no_grad():
            sign = torch.sign(self.weight)
            mask = (self.weight.abs() > self.threshold).float()
            return sign * mask


class TritClassifier(nn.Module):
    def __init__(self, in_features=10, hidden=128, n_out_trits=5, threshold=0.3):
        super().__init__()
        self.n_out_trits = n_out_trits
        self.threshold = threshold
        self.fc1 = TernaryLinearQAT(in_features, hidden, threshold=threshold)
        self.fc2 = TernaryLinearQAT(hidden, hidden, threshold=threshold)
        self.fc3 = TernaryLinearQAT(hidden, n_out_trits * 3, threshold=threshold)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, self.n_out_trits, 3)

    @property
    def qat_layers(self):
        return [self.fc1, self.fc2, self.fc3]


class TritClassifierFloat(nn.Module):
    def __init__(self, in_features=10, hidden=128, n_out_trits=5):
        super().__init__()
        self.n_out_trits = n_out_trits
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_out_trits * 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, self.n_out_trits, 3)

    @property
    def float_layers(self):
        return [self.fc1, self.fc2, self.fc3]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def targets_to_class_idx(Y):
    return (Y + 1).long()

def logits_to_ternary(logits):
    return logits.argmax(dim=2).float() - 1

def exact_match_accuracy(logits, Y):
    pred = logits_to_ternary(logits)
    return (pred == Y).all(dim=1).float().mean().item()

def trit_accuracy(logits, Y):
    pred = logits_to_ternary(logits)
    return (pred == Y).float().mean().item()

def rescale_weights_for_qat(float_model, qat_model, threshold):
    for fl, ql in zip(float_model.float_layers, qat_model.qat_layers):
        with torch.no_grad():
            w = fl.weight.data
            b = fl.bias.data
            p75 = w.abs().quantile(0.75).clamp(min=1e-8)
            scale = (2.0 * threshold) / p75
            ql.weight.data.copy_(w * scale)
            ql.bias_p.data.copy_(b * scale)

def weight_distribution(model):
    neg = zero = pos = total = 0
    for layer in model.qat_layers:
        w = layer.get_ternary_weights()
        neg += (w == -1).sum().item()
        zero += (w == 0).sum().item()
        pos += (w == 1).sum().item()
        total += w.numel()
    return neg / total, zero / total, pos / total

def ckpt_path(op_name: str, kind: str) -> Path:
    d = CKPT_DIR / op_name
    d.mkdir(exist_ok=True)
    return d / f"{kind}.pt"

def result_path(op_name: str) -> Path:
    d = CKPT_DIR / op_name
    d.mkdir(exist_ok=True)
    return d / "result.json"

def load_result(op_name: str) -> dict | None:
    p = result_path(op_name)
    if p.exists():
        return json.loads(p.read_text())
    return None

def save_result(op_name: str, result: dict):
    result_path(op_name).write_text(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_operation(
    op_name: str,
    X: torch.Tensor,
    Y: torch.Tensor,
    seed: int = 42,
    hidden: int = 128,
    lr_p1: float = 1e-3,
    lr_p2: float = 1e-4,
    max_epochs_p1: int = 3000,
    max_epochs_p2: int = 10000,
    threshold: float = 0.3,
    target_acc: float = 1.0,
    log_every: int = 100,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    in_features = X.shape[1]
    n_out_trits = Y.shape[1]
    criterion = nn.CrossEntropyLoss()
    Y_idx = targets_to_class_idx(Y)

    float_ckpt = ckpt_path(op_name, 'best_float')

    # ---- Phase 1: float weights ----
    float_model = TritClassifierFloat(in_features=in_features, hidden=hidden, n_out_trits=n_out_trits)

    # Warm-resume, mirroring the Phase 2 QAT resume below: a saved
    # checkpoint only sets the starting weights, it does NOT mean Phase 1
    # is done. Found 2026-08-12: this used to load the checkpoint and
    # unconditionally treat it as finished (p1_epochs=-1, no further
    # training), unlike Phase 2's resume path (already fixed, commit
    # d5b9fc1) which continues training from the checkpoint. If the
    # process was killed during Phase 1 before target_acc, best_float.pt
    # holds whatever accuracy it had at the last improvement — re-running
    # would silently accept that undertrained checkpoint and proceed
    # straight into Phase 2 from a poor float baseline.
    if float_ckpt.exists():
        log(f"  [{op_name}] Resuming Phase 1 from checkpoint {float_ckpt}")
        float_model.load_state_dict(torch.load(float_ckpt, weights_only=True))

    float_model.eval()
    with torch.no_grad():
        p1_acc = exact_match_accuracy(float_model(X), Y)
    best_p1 = p1_acc
    p1_epochs = 0
    t0 = time.time()

    if p1_acc >= target_acc:
        log(f"  [{op_name}] Loaded float checkpoint already at target: exact={p1_acc*100:.1f}%")
    else:
        opt1 = optim.Adam(float_model.parameters(), lr=lr_p1)
        sch1 = optim.lr_scheduler.ReduceLROnPlateau(opt1, patience=300, factor=0.5, min_lr=1e-5)

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

            if p1_acc > best_p1:
                best_p1 = p1_acc
                torch.save(float_model.state_dict(), float_ckpt)

            if epoch % log_every == 0 or p1_acc >= target_acc:
                trit_acc = trit_accuracy(float_model(X), Y)
                log(f"  [{op_name}] P1 epoch={epoch:5d}  loss={loss.item():.4f}  "
                    f"exact={p1_acc*100:.1f}%  trit={trit_acc*100:.1f}%")

            if p1_acc >= target_acc:
                break

    log(f"  [{op_name}] Phase1 done: exact={p1_acc*100:.1f}% in {p1_epochs} epochs  "
        f"({time.time()-t0:.1f}s)")

    # ---- Phase 2: QAT warm-start ----
    qat_ckpt = ckpt_path(op_name, 'best_qat')

    qat_model = TritClassifier(in_features=in_features, hidden=hidden,
                                n_out_trits=n_out_trits, threshold=threshold)

    if qat_ckpt.exists():
        # Warm-resume: continue from the best QAT weights instead of restarting
        # from the float rescale, so an interrupted run keeps its progress and
        # a worse fresh run can never overwrite a better saved checkpoint.
        log(f"  [{op_name}] Resuming Phase 2 from checkpoint {qat_ckpt}")
        qat_model.load_state_dict(torch.load(qat_ckpt, weights_only=True))
    else:
        rescale_weights_for_qat(float_model, qat_model, threshold)

    qat_model.eval()
    with torch.no_grad():
        acc_rescale = exact_match_accuracy(qat_model(X), Y)
    log(f"  [{op_name}] Phase 2 starting accuracy: {acc_rescale*100:.1f}%")

    opt2 = optim.Adam(qat_model.parameters(), lr=lr_p2)
    sch2 = optim.lr_scheduler.ReduceLROnPlateau(opt2, patience=500, factor=0.5, min_lr=1e-6)

    best_acc = acc_rescale
    best_epoch = 0
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
            best_acc = acc
            best_epoch = epoch
            torch.save(qat_model.state_dict(), qat_ckpt)

        if epoch % log_every == 0 or best_acc >= target_acc:
            log(f"  [{op_name}] P2 epoch={epoch:5d}  loss={loss.item():.4f}  "
                f"exact={acc*100:.1f}%  best={best_acc*100:.1f}%  "
                f"lr={opt2.param_groups[0]['lr']:.2e}")

        if best_acc >= target_acc:
            break

    elapsed = time.time() - t0
    neg, zero, pos = weight_distribution(qat_model)

    log(f"  [{op_name}] Phase2 done: best={best_acc*100:.1f}% at epoch {best_epoch}  "
        f"({elapsed:.1f}s)")
    log(f"  [{op_name}] Weights: -{neg*100:.1f}%  0={zero*100:.1f}%  +{pos*100:.1f}%")

    result = {
        'op': op_name,
        'seed': seed,
        'p1_acc': p1_acc,
        'p1_epochs': p1_epochs,
        'acc_after_rescale': acc_rescale,
        'best_acc': best_acc,
        'p2_best_epoch': best_epoch,
        'p2_epochs': epoch + 1,
        'elapsed_s': elapsed,
        'converged': best_acc >= 0.9999,
        'passed': best_acc >= 0.99,
        'weight_neg_pct': neg * 100,
        'weight_zero_pct': zero * 100,
        'weight_pos_pct': pos * 100,
    }
    save_result(op_name, result)
    return result


# ---------------------------------------------------------------------------
# Spot-checks: load saved QAT checkpoint, no retraining
# ---------------------------------------------------------------------------

def run_spot_checks(ops: list[str], results: dict):
    test_pairs = [
        ([1, 0, -1, 1, 0],    [1, -1, 0, -1, 1]),
        ([-1, -1, -1, -1, -1], [1, 1, 1, 1, 1]),
        ([0, 0, 0, 0, 0],     [0, 0, 0, 0, 0]),
    ]

    log("\n" + "=" * 70)
    log("Spot-checks (3 input pairs per operation, loaded from saved checkpoints)")
    log("=" * 70)

    for op in ops:
        qat_ckpt = ckpt_path(op, 'best_qat')
        if not qat_ckpt.exists():
            log(f"\n{op}: checkpoint not found, skipping spot-check")
            continue

        qat_model = TritClassifier(in_features=10, hidden=128, n_out_trits=5, threshold=0.3)
        qat_model.load_state_dict(torch.load(qat_ckpt, weights_only=True))
        qat_model.eval()

        X_spot = torch.tensor(
            [va + vb for va, vb in test_pairs], dtype=torch.float32
        )
        with torch.no_grad():
            preds = logits_to_ternary(qat_model(X_spot))

        log(f"\n{op}:")
        log(f"  {'A':<25} {'B':<25} {'Expected':<25} {'Predicted':<25} OK?")
        for i, (va, vb) in enumerate(test_pairs):
            exp = [SCALAR_OPS[op](a, b) for a, b in zip(va, vb)]
            pred = [int(p) for p in preds[i].tolist()]
            ok = pred == exp
            log(f"  {str(va):<25} {str(vb):<25} {str(exp):<25} {str(pred):<25} {'YES' if ok else 'NO'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log("=" * 70)
    log("TritNet Phase 2B -- binary operations (tadd, tmul, tmin, tmax)")
    log(f"Checkpoints: {CKPT_DIR}")
    log("=" * 70)

    ops = ['tadd', 'tmul', 'tmin', 'tmax']
    results = {}

    for op_name in ops:
        # Resume: skip ops with a saved result where passed=True
        saved = load_result(op_name)
        if saved and saved.get('passed'):
            log(f"\n[{op_name}] Already completed (best={saved['best_acc']*100:.1f}%), skipping.")
            results[op_name] = saved
            continue

        log(f"\n{'='*70}")
        log(f"Operation: {op_name}")
        log(f"{'='*70}", flush=True)

        t_gen = time.time()
        X, Y = make_binary_dataset(op_name)
        log(f"Dataset: {X.shape[0]:,} samples  X={X.shape}  Y={Y.shape}  "
            f"({time.time()-t_gen:.1f}s to generate)")

        r = train_operation(
            op_name=op_name,
            X=X, Y=Y,
            seed=42,
            hidden=128,
            lr_p1=1e-3,
            lr_p2=1e-4,
            max_epochs_p1=3000,
            max_epochs_p2=10000,
            threshold=0.3,
            log_every=100,
        )
        results[op_name] = r

    # ---- Summary ----
    log("\n" + "=" * 70)
    log("Phase 2B Summary")
    log("=" * 70)
    log(f"\n{'Op':<8} {'P1 acc':>8} {'Rescale':>8} {'Best (QAT)':>12} {'Time':>6}  Weights")
    log("-" * 70)
    for op, r in results.items():
        log(f"{op:<8} {r['p1_acc']*100:>7.1f}%  {r['acc_after_rescale']*100:>7.1f}%  "
            f"{r['best_acc']*100:>11.1f}%  "
            f"{r['elapsed_s']:5.1f}s  "
            f"-{r['weight_neg_pct']:.0f}% 0={r['weight_zero_pct']:.0f}% +{r['weight_pos_pct']:.0f}%")

    n_converged = sum(1 for r in results.values() if r['converged'])
    n_passed    = sum(1 for r in results.values() if r['passed'])

    log(f"\nOperations at 100%:  {n_converged}/{len(ops)}")
    log(f"Operations at >=99%: {n_passed}/{len(ops)}")

    go      = n_passed >= 3 and n_converged >= 1
    partial = n_passed >= 2

    log()
    if go:
        log("DECISION: GO")
        log("  >=3 operations reached >=99% with ternary weights.")
        log("  Proceed to Phase 3: C++ inference engine with ternary weights.")
    elif partial:
        log("DECISION: CONDITIONAL GO")
        log("  Some operations learned but not all. Try larger hidden size or more epochs.")
    else:
        log("DECISION: NO-GO -- investigate harder operations individually.")

    run_spot_checks(ops, results)

    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
