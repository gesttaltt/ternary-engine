"""
run_bitnet_on_engine.py - run a NATIVELY-TERNARY LLM on this project's engine

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

WHY THIS EXISTS. Every prior quantization experiment in this repo tried to
*manufacture* a ternary model by crushing an already-converged fp16
checkpoint (TinyLlama-1.1B), and all five techniques failed in the same
direction -- naive per-tensor +697,074%, naive per-channel +1,037,361%,
GPTQ +145,167%, GPTQ+mixed-precision +111,681%, block-local QAT +17,404%
(see reports/2026-08-29/QAT_VS_PTQ_TERNARY_HEAD_TO_HEAD.md).

The project's own recorded conclusion is that published ternary successes
(BitNet b1.58 and similar) train FROM SCRATCH with the quantizer in the
loop. That reframes the question this repo should be asking:

    The engine's job is not to invent a good ternary model.
    It is to run one correctly and quickly.

So this script takes a model that was genuinely *trained* ternary --
1bitLLM/bitnet_b1_58-large, the public reproduction of BitNet b1.58 -- and
asks two falsifiable questions about the engine itself:

  1. FAITHFULNESS. Does `ternary_zero_skip_gemm.DenseWeights` agree with a
     reference float matmul of the same ternary weights, to within what
     fp32 accumulation order can explain?

     NOTE, corrected after the first run: an earlier version of this file
     asserted BIT-EXACT agreement, reasoning that ternary x fp32 products
     are exactly representable. The products are -- but the SUM of 1,536 to
     4,096 fp32 terms is order-dependent, and the engine accumulates in a
     different order (N-vectorized, M-blocked) than the reference. 0/63
     cells were bit-exact while the worst relative deviation was 2.2e-06,
     which is the correct result, not a failure. The test is therefore a
     tolerance appropriate to fp32 accumulation over K terms.

  2. WEIGHT SPARSITY of a real trained ternary LLM, as an independent check
     on this project's long-standing "~40% of products are zero" claim,
     which until now rested on synthetic data.

  SPEED IS DELIBERATELY NOT MEASURED HERE. Two reasons, both learned the
  hard way on the first run: (a) timing through pybind11 violates this
  project's own ffi_isolation convention for absolute performance claims;
  (b) the numbers were visibly untrustworthy -- the same shape produced
  0.177 ms and 5.97 ms in different rows, because the engine's OpenMP pool
  and NumPy's OpenBLAS pool oversubscribe each other, and CV on this shared
  desktop runs 26-72%. Speed at this model's shapes belongs in, and is
  measured by, benchmarks/cpp-native-kernels/bench_inference_latency_fp16.cpp,
  which is native, single-threaded on both sides, and interleaved.

IMPORTANT, and verified rather than assumed: the weights in this checkpoint
are stored as **latent full-precision** values (25,824 distinct values in
layer 0's q_proj), NOT as ternary. BitNet's `BitLinear` quantizes them in
the forward pass. So the deployed ternary weights must be materialized by
applying BitNet's own quantizer -- per-tensor absmean scale, round, clamp to
{-1,0,+1}. That is the same functional form this project already uses; the
difference that matters is that this model was TRAINED under it, so
materializing it is not post-training quantization, it is just reading out
what the model already is.

USAGE: python benchmarks/model_quantization/run_bitnet_on_engine.py
       python benchmarks/model_quantization/run_bitnet_on_engine.py --layers 0 11 23
OUTPUT: benchmarks/results/model_quantization/bitnet_engine_<timestamp>.json
"""

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_ID = "1bitLLM/bitnet_b1_58-large"
PROJ_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj")


def bitnet_ternary(w: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """BitNet b1.58's weight quantizer: per-tensor absmean scale, round,
    clamp to {-1,0,+1}. Returns (ternary_values, scale)."""
    scale = w.abs().mean().clamp(min=1e-8)
    return (w / scale).round().clamp(-1, 1), float(scale)


def find_checkpoint() -> str:
    hits = glob.glob(
        f"{Path.home()}/.cache/huggingface/hub/models--1bitLLM--bitnet_b1_58-large"
        "/snapshots/*/model.safetensors")
    if not hits:
        raise SystemExit(
            f"{MODEL_ID} not found in the HuggingFace cache. Fetch it with:\n"
            "  python3 -c \"from huggingface_hub import snapshot_download; \"\n"
            f"  \"snapshot_download('{MODEL_ID}', allow_patterns=['*.json','*.model','*.safetensors'])\"")
    return hits[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 11, 23],
                    help="transformer layer indices to test")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32],
                    help="batch sizes (M) to check numerically")
    args = ap.parse_args()

    import ternary_zero_skip_gemm as tzs
    from safetensors import safe_open

    print("=" * 78)
    print(f" Natively-ternary LLM on the ternary engine -- {MODEL_ID}")
    print("=" * 78)
    print(f" engine: AVX2={tzs.has_avx2}  OpenMP={tzs.has_openmp}")
    print(" Weights are stored LATENT (full precision) in this checkpoint and")
    print(" quantized by BitLinear at forward time; they are materialized here")
    print(" with BitNet's own absmean quantizer. Because the model was TRAINED")
    print(" under that quantizer, this is a readout, not a post-hoc quantization.\n")

    ckpt = find_checkpoint()
    rng = np.random.default_rng(1234)

    rows = []
    n_exact = n_total = 0
    zero_num = zero_den = 0

    with safe_open(ckpt, framework="pt") as f:
        for li in args.layers:
            for nm in PROJ_NAMES:
                pref = "self_attn" if nm in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
                key = f"model.layers.{li}.{pref}.{nm}.weight"
                w = f.get_tensor(key).float()
                q, scale = bitnet_ternary(w)

                uniq = torch.unique(q).tolist()
                assert set(uniq).issubset({-1.0, 0.0, 1.0}), \
                    f"{key}: not ternary after quantization -> {uniq}"

                # nn.Linear stores [out, in]; the GEMM wants B as [K, N].
                B = q.t().contiguous().numpy().astype(np.int8)
                K, N = B.shape
                zero_num += int((B == 0).sum()); zero_den += B.size

                dw = tzs.DenseWeights(B)
                info = dw.info()

                for M in args.batches:
                    A = rng.standard_normal((M, K), dtype=np.float32)
                    C_engine = dw.gemm(A)
                    C_ref = A.astype(np.float32) @ B.astype(np.float32)

                    maxabs = float(np.max(np.abs(C_engine - C_ref)))
                    denom = float(np.max(np.abs(C_ref))) or 1.0
                    rel = maxabs / denom
                    # Tolerance for summing K fp32 terms in a different order.
                    # ~K * eps with headroom; a real kernel bug (wrong stride,
                    # wrong sign, dropped term) lands orders of magnitude above
                    # this, not just outside it.
                    tol = float(64.0 * K * np.finfo(np.float32).eps)
                    # bool() is deliberate: `rel <= tol` against a numpy
                    # scalar yields numpy.bool_, which json.dump rejects --
                    # the same leak this project already had to fix in
                    # test_falsification.py (CLAUDE.md v1.23.0).
                    ok = bool(rel <= tol)
                    n_total += 1; n_exact += int(ok)

                    rows.append(dict(layer=li, proj=nm, K=K, N=N, M=M,
                                     within_fp32_tol=ok, max_abs_diff=maxabs,
                                     rel=float(rel), tol=tol,
                                     packed_bytes=info["packed_bytes"]))
            print(f"  layer {li}: {len(PROJ_NAMES)} projections checked")

    print(f"\n{'layer.proj':<22}{'K':>6}{'N':>6}{'M':>5}{'ok':>5}"
          f"{'max|diff|':>12}{'relative':>12}{'tol':>12}")
    print("-" * 76)
    for r in rows:
        print(f"L{r['layer']}.{r['proj']:<18}{r['K']:>6}{r['N']:>6}{r['M']:>5}"
              f"{('OK' if r['within_fp32_tol'] else 'BAD'):>5}"
              f"{r['max_abs_diff']:>12.3g}{r['rel']:>12.3g}{r['tol']:>12.3g}")

    zf = zero_num / zero_den
    print("-" * 76)
    worst = max(r["rel"] for r in rows)
    print(f"\nFAITHFULNESS: {n_exact}/{n_total} GEMM cells agree with the float "
          f"reference within fp32 accumulation tolerance")
    print(f"  worst relative deviation {worst:.3g} -- consistent with summing "
          f"K terms in a different order, not a kernel defect. (Bit-exactness "
          f"is NOT expected here and asserting it was an error in this "
          f"script's first version; see the module docstring.)")
    print(f"WEIGHT SPARSITY: {zf*100:.1f}% zeros across {zero_den:,} weights "
          f"-- an independent check on this project's long-standing "
          f"'~40% of products are zero' claim, measured on a real trained "
          f"ternary LLM rather than synthetic data.")
    print("SPEED: deliberately not measured here -- see the module docstring. "
          "Use benchmarks/cpp-native-kernels/bench_inference_latency_fp16.cpp.")

    out_dir = PROJECT_ROOT / "benchmarks" / "results" / "model_quantization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"bitnet_engine_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w") as fh:
        json.dump({"metadata": {"model": MODEL_ID,
                                "timestamp": datetime.now().isoformat(),
                                "avx2": tzs.has_avx2,
                                "openmp": tzs.has_openmp,
                                "note": "weights stored latent; materialized "
                                        "with BitNet's absmean quantizer"},
                   "cells": rows,
                   "faithfulness": {"cells_within_fp32_tol": n_exact,
                                    "total": n_total,
                                    "worst_relative_deviation": worst},
                   "weight_zero_fraction": float(zf)}, fh, indent=2)
    print(f"\nResults saved to {out}")
    return 0 if n_exact == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
