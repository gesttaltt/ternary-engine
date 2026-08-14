#!/usr/bin/env python3
"""
generate_weights_header.py - Generate tritnet_weights.h from exported .npy weights

Reads models/tritnet/phase2b_export/<op>/*.npy (produced by
models/tritnet/export_weights.py) and emits a single constexpr C++ header with
every op's quantized weights and biases embedded as compile-time arrays.

Follows the project's existing "algorithm as documentation, LUTs generated
from a single source of truth at compile/codegen time" convention (see
src/core/algebra/ternary_lut_gen.h for the same pattern applied to the
scalar tadd/tmul/tmin/tmax/tnot LUTs) -- weights live in one generated file
instead of being hand-copied into the inference engine, and this script is
the one place that has to be re-run if a checkpoint is retrained.

Deliberately NumPy-only (no PyTorch import): the C++-facing codegen step
should not need a training framework installed to regenerate the header
from already-exported weights.

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

USAGE: python models/tritnet/inference/generate_weights_header.py
OUTPUT: models/tritnet/inference/tritnet_weights.h
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent  # models/tritnet/
EXPORT_DIR = ROOT / "phase2b_export"
OUT_HEADER = Path(__file__).parent / "tritnet_weights.h"

OPS = ['tnot', 'tadd', 'tmul', 'tmin', 'tmax']


def emit_int8_array(name: str, arr: np.ndarray, indent: str = "    ") -> str:
    """Emit a 2D int8 array as `constexpr int8_t name[rows][cols] = {...};`."""
    rows, cols = arr.shape
    lines = [f"constexpr int8_t {name}[{rows}][{cols}] = {{"]
    for r in range(rows):
        row_vals = ", ".join(str(int(v)) for v in arr[r])
        lines.append(f"{indent}{{{row_vals}}},")
    lines.append("};")
    return "\n".join(lines)


def format_float_literal(v: float) -> str:
    """Format as a valid C++ float literal. `%.9g` on an exact integer (e.g.
    the 0.0 bias padding added by pad_to_multiple()) prints "0" with no '.'
    or 'e' -- `0f` is not a valid C++ literal (the `f` suffix requires a
    floating-point-literal base, i.e. a decimal point or exponent), so gcc
    rejects it with "unable to find numeric literal operator". Force one in.
    """
    s = f"{v:.9g}"
    if '.' not in s and 'e' not in s and 'inf' not in s and 'nan' not in s:
        s += '.0'
    return f"{s}f"


def emit_float_array(name: str, arr: np.ndarray) -> str:
    """Emit a 1D float array as `constexpr float name[n] = {...};`."""
    vals = ", ".join(format_float_literal(float(v)) for v in arr)
    return f"constexpr float {name}[{arr.shape[0]}] = {{{vals}}};"


def pad_to_multiple(arr: np.ndarray, multiple: int, axis: int) -> np.ndarray:
    """Zero-pad `arr` along `axis` up to the next multiple of `multiple`.

    Used to widen W3/B3's output dimension (15 = N_OUT_TRITS*3) to a
    multiple of 8 so the AVX2 forward pass can process it with full-width
    loads/stores, no scalar tail. Padding columns get weight 0 and bias 0,
    so the extra logit lanes they produce are always 0 and are never read
    by the argmax decode loop (which only looks at the first N_OUT_TRITS*3
    lanes) -- this is purely a memory-layout widening, not a behavior change,
    and the scalar forward pass uses the same padded arrays.
    """
    pad_width = [(0, 0)] * arr.ndim
    current = arr.shape[axis]
    target = ((current + multiple - 1) // multiple) * multiple
    pad_width[axis] = (0, target - current)
    return np.pad(arr, pad_width, mode='constant', constant_values=0)


def generate_op_block(op_name: str) -> str:
    op_dir = EXPORT_DIR / op_name
    if not op_dir.exists():
        raise FileNotFoundError(f"{op_dir} not found -- run export_weights.py first")

    W1 = np.load(op_dir / "W1.npy"); b1 = np.load(op_dir / "b1.npy")
    W2 = np.load(op_dir / "W2.npy"); b2 = np.load(op_dir / "b2.npy")
    W3 = np.load(op_dir / "W3.npy"); b3 = np.load(op_dir / "b3.npy")

    in_features, hidden = W1.shape
    n_out_trits = W3.shape[1] // 3

    # The AVX2 forward pass (tritnet_inference_avx2.h) processes HIDDEN in
    # 8-wide chunks with no scalar tail -- both current architectures (64,
    # 128) already satisfy this, but fail loudly instead of silently
    # miscomputing if a future retrain changes it.
    if hidden % 8 != 0:
        raise ValueError(f"{op_name}: hidden={hidden} is not a multiple of 8; "
                          f"tritnet_inference_avx2.h's layer loops assume no scalar tail")

    # Widen the output layer to a multiple of 8 (AVX2 lane width) -- see
    # pad_to_multiple()'s docstring for why this is safe.
    W3 = pad_to_multiple(W3, 8, axis=1)
    b3 = pad_to_multiple(b3, 8, axis=0)
    out_padded = W3.shape[1]

    parts = [
        f"namespace {op_name}_weights {{",
        f"constexpr int IN_FEATURES = {in_features};",
        f"constexpr int HIDDEN = {hidden};",
        f"constexpr int N_OUT_TRITS = {n_out_trits};",
        f"constexpr int OUT_PADDED = {out_padded};  // N_OUT_TRITS*3 rounded up to a multiple of 8",
        "",
        emit_int8_array("W1", W1),
        emit_float_array("B1", b1),
        emit_int8_array("W2", W2),
        emit_float_array("B2", b2),
        emit_int8_array("W3", W3),
        emit_float_array("B3", b3),
        f"}}  // namespace {op_name}_weights",
    ]
    return "\n".join(parts)


def main() -> int:
    print(f"Generating {OUT_HEADER} from {EXPORT_DIR}")

    blocks = []
    for op_name in OPS:
        blocks.append(generate_op_block(op_name))
        print(f"  {op_name}: OK")

    header = f"""// tritnet_weights.h -- GENERATED FILE, do not edit by hand.
//
// Copyright 2025 Ternary Engine Contributors
// Licensed under the Apache License, Version 2.0
//
// Generated by models/tritnet/inference/generate_weights_header.py from the
// real Phase 2A/2B GO-decision checkpoints (models/tritnet/phase2a/tnot/,
// models/tritnet/phase2b/{{tadd,tmul,tmin,tmax}}/best_qat.pt), NOT from
// models/tritnet/src/tritnet_model.py's stale/abandoned checkpoints -- see
// CLAUDE.md "TritNet Phase 3" note (2026-08-14) for why that distinction
// matters. Re-run this script after retraining any op.
//
// Inference recipe per op (see tritnet_inference.h for the C++ implementation):
//   h1 = ReLU(x @ W1 + b1)
//   h2 = ReLU(h1 @ W2 + b2)
//   logits = h2 @ W3 + b3, reshaped to [N_OUT_TRITS, 3]
//   trit[k] = argmax(logits[k]) - 1

#pragma once

#include <cstdint>

namespace tritnet {{
namespace weights {{

{chr(10).join(blocks)}

}}  // namespace weights
}}  // namespace tritnet
"""

    OUT_HEADER.write_text(header)
    print(f"\nWrote {OUT_HEADER} ({OUT_HEADER.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
