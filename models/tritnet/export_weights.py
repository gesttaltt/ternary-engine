#!/usr/bin/env python3
"""
export_weights.py - Export TritNet Phase 2A/2B ternary weights to NumPy for C++ integration

Exports the ACTUAL GO-decision checkpoints (train_phase2a.py's tnot, train_phase2b.py's
tadd/tmul/tmin/tmax -- the numbers documented in CLAUDE.md's "TritNet Development" section)
as quantized int8 weight + float32 bias arrays, ready for a C++ inference engine (Phase 3).

Why this file exists instead of reusing models/tritnet/src/tritnet_model.py's
save_tritnet_model/load_tritnet_model/export_weights_to_numpy:
That module's TritNetUnary/TritNetBinary (backed by ternary_layers.TernaryLinear, no bias,
direct-regression output) is a DIFFERENT, earlier architecture from the one that actually
produced the documented GO checkpoints. Confirmed 2026-08-14: models/tritnet/tritnet_tadd.tritnet
(that pipeline's only surviving tadd checkpoint) is only 15.8% accurate -- an abandoned MSE-
regression attempt, not the 100%-accurate tadd result CLAUDE.md documents. The real GO
checkpoints (models/tritnet/phase2a/tnot/, phase2b/{tadd,tmul,tmin,tmax}/best_qat.pt) were
trained by train_phase2a.py/train_phase2b.py's own local TritClassifier/TernaryLinearQAT
classes: bias included, CrossEntropy classification head (5x3 logits -> argmax per trit),
ReLU between hidden layers -- structurally different from tritnet_model.py's classes, so
those checkpoints cannot be loaded into TritNetUnary/TritNetBinary at all. This script targets
the real architecture directly.

Inference recipe for the C++ side (per operation, per exported layer i in {1,2,3}):
    h = ReLU(x @ Wi + bi)     for i in {1, 2}   (x is {-1,0,+1} input trits, float)
    logits = h @ W3 + b3                         (no activation on the output layer)
    logits reshaped to [n_out_trits, 3]
    trit[k] = argmax(logits[k, :]) - 1            (maps class idx {0,1,2} -> value {-1,0,+1})

Weight layout: Wi.npy is [in_features, out_features] (transposed from PyTorch's
[out_features, in_features], i.e. C++ row-major "x @ W" convention -- same transpose
tritnet_model.py's export_weights_to_numpy already applies). bi.npy is [out_features].

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

USAGE: python models/tritnet/export_weights.py
OUTPUT: models/tritnet/phase2b_export/<op>/{W1,b1,W2,b2,W3,b3}.npy + manifest.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from qat_common import TritClassifier  # noqa: E402 -- shared by train_phase2a.py/train_phase2b.py

ROOT = Path(__file__).parent
EXPORT_DIR = ROOT / "phase2b_export"

# Per-op architecture config. MUST match what actually trained each checkpoint --
# these are read off train_phase2a.py (tnot: in_features=5, hidden=64) and
# train_phase2b.py (binary ops: in_features=10, hidden=128); both use threshold=0.3.
OPS = {
    'tnot': dict(in_features=5,  hidden=64,  n_out_trits=5,
                 ckpt=ROOT / "phase2a" / "tnot" / "best_qat.pt"),
    'tadd': dict(in_features=10, hidden=128, n_out_trits=5,
                 ckpt=ROOT / "phase2b" / "tadd" / "best_qat.pt"),
    'tmul': dict(in_features=10, hidden=128, n_out_trits=5,
                 ckpt=ROOT / "phase2b" / "tmul" / "best_qat.pt"),
    'tmin': dict(in_features=10, hidden=128, n_out_trits=5,
                 ckpt=ROOT / "phase2b" / "tmin" / "best_qat.pt"),
    'tmax': dict(in_features=10, hidden=128, n_out_trits=5,
                 ckpt=ROOT / "phase2b" / "tmax" / "best_qat.pt"),
}
THRESHOLD = 0.3


def export_op(op_name: str, cfg: dict) -> dict:
    """Load one op's checkpoint and export its ternary weights + biases to .npy."""
    ckpt = cfg['ckpt']
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found at {ckpt}")

    model = TritClassifier(
        in_features=cfg['in_features'], hidden=cfg['hidden'],
        n_out_trits=cfg['n_out_trits'], threshold=THRESHOLD,
    )
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    model.eval()

    out_dir = EXPORT_DIR / op_name
    out_dir.mkdir(parents=True, exist_ok=True)

    shapes = {}
    with torch.no_grad():
        for i, layer in enumerate([model.fc1, model.fc2, model.fc3], start=1):
            w = layer.get_ternary_weights().cpu().numpy()  # [out_f, in_f], values in {-1,0,+1}
            w_t = w.T.astype(np.int8)                       # -> [in_f, out_f] for x @ W
            np.save(out_dir / f"W{i}.npy", w_t)
            shapes[f"W{i}"] = list(w_t.shape)

            b = layer.bias_p.cpu().numpy().astype(np.float32)
            np.save(out_dir / f"b{i}.npy", b)
            shapes[f"b{i}"] = list(b.shape)

    print(f"  Exported {op_name}: " + ", ".join(f"{k}={v}" for k, v in shapes.items()))
    return shapes


def main() -> int:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {'threshold': THRESHOLD, 'ops': {}}

    print("Exporting TritNet ternary weights for C++ integration")
    print(f"Source: models/tritnet/phase2a/, models/tritnet/phase2b/ (real GO checkpoints)")
    print(f"Target: {EXPORT_DIR}\n")

    n_ok = 0
    for op_name, cfg in OPS.items():
        try:
            shapes = export_op(op_name, cfg)
            manifest['ops'][op_name] = {
                'shapes': shapes,
                'in_features': cfg['in_features'],
                'hidden': cfg['hidden'],
                'n_out_trits': cfg['n_out_trits'],
                'inference': 'h1=ReLU(x@W1+b1); h2=ReLU(h1@W2+b2); '
                             'logits=(h2@W3+b3).reshape(n_out_trits,3); '
                             'trit[k]=argmax(logits[k])-1',
            }
            n_ok += 1
        except FileNotFoundError as e:
            print(f"  SKIPPED {op_name}: {e}")

    manifest_path = EXPORT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {manifest_path}")
    print(f"Exported {n_ok}/{len(OPS)} operations")

    return 0 if n_ok == len(OPS) else 1


if __name__ == "__main__":
    sys.exit(main())
