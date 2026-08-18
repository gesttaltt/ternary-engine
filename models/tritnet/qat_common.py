#!/usr/bin/env python3
"""
qat_common.py - Shared QAT (quantization-aware training) infrastructure for TritNet

Extracted 2026-08-18 to close CLAUDE.md gap #7 ("TritNet training pipeline
duplication"): train_phase2a.py and train_phase2b.py independently
reimplemented the same QAT model classes, metric helpers, and checkpoint
I/O near-verbatim. The concrete cost this caused: a Phase-1 checkpoint
resume bug (the "warm-resume must continue training, not just load and
stop" fix, commit d5b9fc1) had to be found and fixed separately in
train_phase2b.py, with no shared code a single fix would have applied to
train_phase2a.py as well.

This module is the single source of truth for:
  - STE / TernaryLinearQAT / TritClassifier / TritClassifierFloat (the model)
  - targets_to_class_idx / logits_to_ternary / exact_match_accuracy /
    trit_accuracy (metrics)
  - rescale_weights_for_qat / weight_distribution (QAT-specific helpers)
  - ckpt_path / result_path / load_result / save_result (checkpoint I/O)

Deliberately NOT unified here (genuinely different, not duplicated):
  - Dataset generation (tnot's truth table vs. the 4 binary ops')
  - The training-loop control flow itself: train_phase2a.py sweeps
    multiple seeds for a single op's go/no-go decision; train_phase2b.py
    trains one seed per op with full Phase-1-and-Phase-2 checkpoint resume
    across 4 ops. Different enough shapes that forcing one function over
    both would obscure more than it would share.

quantize_ternary itself is reused from models/tritnet/src/ternary_layers.py
rather than redefined a third time here, per this project's own
single_source_of_truth convention -- it's a pure math utility (magnitude-
threshold quantization) with no dependency on that module's unrelated
TernaryLinear architecture.

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ternary_layers import quantize_ternary  # noqa: E402


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class STE(torch.autograd.Function):
    """Straight-through estimator for ternary weight quantization."""

    @staticmethod
    def forward(ctx, w: torch.Tensor, threshold: float) -> torch.Tensor:
        return quantize_ternary(w, threshold)

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
        nn.init.normal_(self.weight, std=1.0)  # start wide so most weights quantize to +-1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = STE.apply(self.weight, self.threshold)
        return nn.functional.linear(x, w_q, self.bias_p)

    def get_ternary_weights(self) -> torch.Tensor:
        with torch.no_grad():
            return quantize_ternary(self.weight, self.threshold)


class TritClassifier(nn.Module):
    """
    Classify each output trit position into {-1, 0, +1}.

    Output shape: [batch, n_out_trits, 3] logits (3 classes per trit position).
    Loss: CrossEntropyLoss summed over trit positions.
    Accuracy: fraction of samples where ALL trit-position argmax predictions match target.

    Uses QAT (quantization-aware training): weights are ternary {-1,0,+1} during
    the forward pass, STE allows gradients to update the underlying float weights.

    in_features/hidden have no defaults deliberately: tnot (Phase 2A) and the
    binary ops (Phase 2B) use different architectures (5/64 vs 10/128) and a
    silently-applied wrong default here would train against the wrong shape.
    """

    def __init__(self, in_features: int, hidden: int, n_out_trits: int = 5, threshold: float = 0.3):
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

    @property
    def qat_layers(self):
        return [self.fc1, self.fc2, self.fc3]


class TritClassifierFloat(nn.Module):
    """Same architecture as TritClassifier but with full-precision weights (Phase 1 warm-start)."""

    def __init__(self, in_features: int, hidden: int, n_out_trits: int = 5):
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

    @property
    def float_layers(self):
        return [self.fc1, self.fc2, self.fc3]


# ---------------------------------------------------------------------------
# Metrics / conversions
# ---------------------------------------------------------------------------

def targets_to_class_idx(Y: torch.Tensor) -> torch.Tensor:
    """Convert ternary targets [batch, N] in {-1,0,+1} -> class indices [batch, N] in {0,1,2}."""
    return (Y + 1).long()


def logits_to_ternary(logits: torch.Tensor) -> torch.Tensor:
    """Convert [batch, N, 3] logits -> [batch, N] ternary values in {-1,0,+1}."""
    return logits.argmax(dim=2).float() - 1


def exact_match_accuracy(logits: torch.Tensor, Y: torch.Tensor) -> float:
    """Fraction of samples where ALL output trits are correct."""
    pred = logits_to_ternary(logits)
    return (pred == Y).all(dim=1).float().mean().item()


def trit_accuracy(logits: torch.Tensor, Y: torch.Tensor) -> float:
    """Fraction of individual output trits that are correct (partial credit)."""
    pred = logits_to_ternary(logits)
    return (pred == Y).float().mean().item()


# ---------------------------------------------------------------------------
# QAT-specific helpers
# ---------------------------------------------------------------------------

def rescale_weights_for_qat(float_model: nn.Module, qat_model: nn.Module, threshold: float):
    """Copy float weights into QAT model, rescaled so most exceed the quantization threshold.

    Scaling by a positive constant preserves logit ordering (CE loss), so the
    rescaled model has the same accuracy but weights that are mostly > threshold.
    """
    with torch.no_grad():
        for fl, ql in zip(float_model.float_layers, qat_model.qat_layers):
            w = fl.weight.data
            b = fl.bias.data
            # Scale so that the 75th-percentile absolute weight = 2*threshold
            p75 = w.abs().quantile(0.75).clamp(min=1e-8)
            scale = (2.0 * threshold) / p75
            ql.weight.data.copy_(w * scale)
            ql.bias_p.data.copy_(b * scale)


def weight_distribution(model: nn.Module) -> tuple:
    """Fraction of ternary weights that are -1/0/+1 across a TritClassifier's qat_layers.

    Returns (neg_frac, zero_frac, pos_frac).
    """
    neg = zero = pos = total = 0
    for layer in model.qat_layers:
        w = layer.get_ternary_weights()
        neg += (w == -1).sum().item()
        zero += (w == 0).sum().item()
        pos += (w == 1).sum().item()
        total += w.numel()
    return neg / total, zero / total, pos / total


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
# Each caller has its own checkpoint directory (models/tritnet/phase2a/,
# models/tritnet/phase2b/), so ckpt_dir is an explicit first argument rather
# than a module-level constant here -- bind it with functools.partial at the
# call site (see train_phase2a.py / train_phase2b.py) to keep existing call
# shapes (`ckpt_path('tnot', 'best_qat')`) unchanged.

def ckpt_path(ckpt_dir: Path, op_name: str, kind: str) -> Path:
    d = ckpt_dir / op_name
    d.mkdir(exist_ok=True)
    return d / f"{kind}.pt"


def result_path(ckpt_dir: Path, op_name: str) -> Path:
    d = ckpt_dir / op_name
    d.mkdir(exist_ok=True)
    return d / "result.json"


def load_result(ckpt_dir: Path, op_name: str):
    p = result_path(ckpt_dir, op_name)
    if p.exists():
        return json.loads(p.read_text())
    return None


def save_result(ckpt_dir: Path, op_name: str, result: dict):
    result_path(ckpt_dir, op_name).write_text(json.dumps(result, indent=2))
