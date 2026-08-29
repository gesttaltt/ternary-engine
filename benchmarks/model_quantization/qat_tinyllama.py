"""
qat_tinyllama.py - block-local quantization-aware training for ternary
TinyLlama-1.1B, as the head-to-head against post-training quantization

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

WHY THIS EXISTS. docs/planning/ROADMAP.md's accuracy-retention criterion
("<5% loss") has now defeated four post-training-quantization (PTQ)
techniques, each failing in the same direction rather than a better one:

    naive per-tensor absmean            12.780 ->    89,100.682  (+697,074%)
    naive per-channel absmean           12.780 ->   132,590.254  (+1,037,361%)
    GPTQ (Hessian error compensation)   12.780 ->    18,565.469  (+145,167%)
    GPTQ + mixed precision (3+3)        12.780 ->    14,285.862  (+111,681%)

The roadmap's stated decision rule is that once mixed-precision PTQ fails
at full-model scale -- it did, 2026-08-28 -- "the real fix is QAT or
training from scratch", matching the observation that every published
ternary/1.58-bit success (BitNet b1.58 and similar) trains WITH the
quantizer in the loop rather than post-hoc-quantizing a converged
checkpoint. This script tests that claim in the cheapest falsifiable form
available on a 6GB card.

THE HYPOTHESIS, stated so it can fail: if the PTQ failure mode is really
"3 discrete levels cannot represent what a converged checkpoint's weights
need" (rather than "insufficiently compensated rounding error"), then
letting gradients move the weights while the quantizer is in the forward
pass should recover substantially more accuracy than any amount of
post-hoc rounding at the SAME scope. If QAT lands in the same range as
GPTQ at matched scope, that hypothesis is wrong and ternary is simply not
representationally sufficient here.

DESIGN: block-local QAT distillation, deliberately mirroring the structure
of quantize_tinyllama_gptq.py so the comparison isolates one variable.

  - Same block-sequential harness: a Catcher captures block 0's real
    inputs once, each block is processed in order on cached activations,
    and each block's post-quantization output becomes the next block's
    input -- so quantization error compounds forward exactly as it does in
    the GPTQ script.
  - Same calibration and eval windows, same corpus, same disjointness.
  - Same quantizer: per-tensor absmean scale with round-and-clamp to
    {-1,0,+1}*scale, byte-for-byte the scheme quantize_tinyllama_gptq.py's
    GPTQLayerQuantizer uses. THIS IS THE POINT -- the only difference
    between the two scripts is whether the weights are *trained* under
    that quantizer (here) or *rounded* into it (GPTQ). Anything else
    differing would confound the result.
  - Per block, the fp16 teacher output is computed first, then the block's
    nn.Linear layers are swapped for ternary-QAT equivalents initialized
    from the same fp16 weights, and Adam minimizes MSE against the teacher
    output. Only one block is ever on the backward graph, which is what
    makes this fit alongside a resident 1.1B model in 6GB.

NOT REUSED, and why: models/tritnet/qat_common.py's TernaryLinearQAT and
models/tritnet/src/ternary_layers.py's quantize_ternary implement ternary
QAT already, but with an ABSOLUTE magnitude threshold (default 0.3) tuned
for TritNet's deliberately wide nn.init.normal_(std=1.0) weights. Real
transformer weights have std ~0.02, so that threshold zeroes essentially
every weight. The STE pattern is the same idea and is credited as prior
art here; the quantizer itself has to be the scale-aware absmean one to
match the PTQ baseline being compared against.

SCOPE NOTE: like its PTQ siblings this measures quantization ACCURACY, not
this project's own ternary kernel speed.

USAGE: python benchmarks/model_quantization/qat_tinyllama.py --layers 14
           (blocks 0-1 -- the scope GPTQ scored 1,336.567 on, for a direct
           head-to-head)
       python benchmarks/model_quantization/qat_tinyllama.py
           (full model)
       python benchmarks/model_quantization/qat_tinyllama.py --steps 400 --lr 1e-4
OUTPUT: benchmarks/results/model_quantization/tinyllama_qat_<timestamp>.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from quantize_tinyllama_gptq import (  # noqa: E402
    CALIB_SEQ_LEN,
    CALIB_SEQS,
    CALIB_TOKEN_OFFSET,
    DEFAULT_MAX_TOKENS,
    MODEL_NAME,
    RESULTS_DIR,
    SEQ_LEN,
    compute_perplexity,
    load_wikitext2_test,
)


def ternary_quantize(w: torch.Tensor) -> torch.Tensor:
    """Per-tensor absmean ternary quantization: {-1,0,+1} * scale.

    Identical in form to GPTQLayerQuantizer.quantize()'s scheme in
    quantize_tinyllama_gptq.py. Kept identical on purpose -- it is the
    controlled variable of the PTQ-vs-QAT comparison.
    """
    scale = w.abs().mean().clamp(min=1e-8)
    return torch.clamp(torch.round(w / scale), -1, 1) * scale


class _TernarySTE(torch.autograd.Function):
    """Straight-through estimator around ternary_quantize.

    Same idea as models/tritnet/qat_common.py's STE (credited as prior art);
    the quantizer differs because that one uses an absolute threshold suited
    to TritNet's std=1.0 weights, not to a real transformer's std~0.02.
    """

    @staticmethod
    def forward(ctx, w: torch.Tensor) -> torch.Tensor:
        return ternary_quantize(w)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad


class TernaryLinearQAT(nn.Module):
    """nn.Linear whose weight is ternarized in the forward pass, with the
    underlying float weight left trainable through the STE."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.data.detach().clone().float())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.detach().clone().float())
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, _TernarySTE.apply(self.weight), self.bias)

    def baked_weight(self) -> torch.Tensor:
        with torch.no_grad():
            return ternary_quantize(self.weight)


def swap_linears(block: nn.Module) -> Dict[str, TernaryLinearQAT]:
    """Replaces every nn.Linear inside `block` with a TernaryLinearQAT.
    Returns {qualified_name: module} for later baking/reporting."""
    swapped = {}
    for name, module in list(block.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        parent = block if "." not in name else block.get_submodule(name.rsplit(".", 1)[0])
        attr = name.rsplit(".", 1)[-1]
        qat = TernaryLinearQAT(module).to(module.weight.device)
        setattr(parent, attr, qat)
        swapped[name] = qat
    return swapped


def bake_block(block: nn.Module, swapped: Dict[str, TernaryLinearQAT],
               dtype: torch.dtype) -> Dict[str, dict]:
    """Replaces each TernaryLinearQAT with a plain nn.Linear holding the
    hard-quantized weights, so the block behaves exactly as a deployed
    ternary block would for every later measurement."""
    stats = {}
    for name, qat in swapped.items():
        w_q = qat.baked_weight()
        parent = block if "." not in name else block.get_submodule(name.rsplit(".", 1)[0])
        attr = name.rsplit(".", 1)[-1]
        lin = nn.Linear(w_q.shape[1], w_q.shape[0], bias=qat.bias is not None)
        lin.weight = nn.Parameter(w_q.to(dtype), requires_grad=False)
        if qat.bias is not None:
            lin.bias = nn.Parameter(qat.bias.data.to(dtype), requires_grad=False)
        lin = lin.to(w_q.device)
        setattr(parent, attr, lin)
        stats[name] = {
            "shape": list(w_q.shape),
            "zero_fraction": float((w_q == 0).sum().item()) / w_q.numel(),
            "scale": float(qat.weight.data.abs().mean().item()),
        }
    return stats


def train_block(block: nn.Module, swapped: Dict[str, TernaryLinearQAT],
                inps: List[torch.Tensor], kwargs_list: List[dict],
                teacher: List[torch.Tensor], steps: int, lr: float,
                batch: int) -> dict:
    """Distills the ternarized block toward its own fp16 teacher output."""
    params = [p for m in swapped.values() for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    n = len(inps)
    losses = []
    g = torch.Generator().manual_seed(0)   # reproducible minibatch order
    for step in range(steps):
        idx = torch.randperm(n, generator=g)[:batch].tolist()
        opt.zero_grad(set_to_none=True)
        loss_val = 0.0
        for i in idx:
            out = block(inps[i].float(), **kwargs_list[i])
            if isinstance(out, tuple):
                out = out[0]
            loss = nn.functional.mse_loss(out, teacher[i].float())
            loss.backward()
            loss_val += loss.item()
        opt.step()
        losses.append(loss_val / len(idx))
    return {
        "steps": steps, "lr": lr, "batch": batch,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_min": min(losses) if losses else None,
    }


def run_qat(model, tokenizer, calib_text: str, dtype: torch.dtype,
            n_layers_limit: Optional[int], protect_first: int, protect_last: int,
            steps: int, lr: float, batch: int) -> dict:
    """Block-sequential QAT, structurally mirroring
    quantize_tinyllama_gptq.py's collect_calibration_and_quantize()."""
    device = next(model.parameters()).device
    decoder_layers = model.model.layers
    n_total = sum(1 for n, m in model.named_modules()
                  if isinstance(m, nn.Linear) and "lm_head" not in n)
    limit = n_layers_limit if n_layers_limit is not None else n_total
    n_blocks = len(decoder_layers)
    protected = set(range(protect_first)) | set(range(n_blocks - protect_last, n_blocks))
    if protected:
        print(f"  Mixed precision: protecting {len(protected)} block(s) at "
              f"original precision (indices {sorted(protected)})")

    ids = tokenizer(calib_text, return_tensors="pt").input_ids[0]
    n_seqs = min(CALIB_SEQS, len(ids) // CALIB_SEQ_LEN)
    calib_batches = [ids[s * CALIB_SEQ_LEN:(s + 1) * CALIB_SEQ_LEN].unsqueeze(0).to(device)
                     for s in range(n_seqs)]
    print(f"  Calibration: {n_seqs} sequences x {CALIB_SEQ_LEN} tokens")

    class _Catcher(nn.Module):
        def __init__(self, module, sink):
            super().__init__()
            self.module, self.sink = module, sink

        def forward(self, hidden_states, **kw):
            self.sink["inps"].append(hidden_states.detach().clone())
            self.sink["kwargs_list"].append({
                "attention_mask": kw.get("attention_mask"),
                "position_ids": kw.get("position_ids"),
                "position_embeddings": kw.get("position_embeddings"),
                "past_key_values": None,
                "use_cache": False,
            })
            raise StopIteration()

    sink = {"inps": [], "kwargs_list": []}
    orig0 = decoder_layers[0]
    decoder_layers[0] = _Catcher(orig0, sink)
    with torch.no_grad():
        for b in calib_batches:
            try:
                model(input_ids=b)
            except StopIteration:
                pass
    decoder_layers[0] = orig0
    cur_inps, kwargs_list = sink["inps"], sink["kwargs_list"]
    print(f"  Captured block-0 inputs for {len(cur_inps)} sequences")

    per_layer: Dict[str, dict] = {}
    per_block: Dict[str, dict] = {}
    n_done = 0
    t_total = time.time()

    for bi in range(n_blocks):
        if n_done >= limit:
            break
        block = decoder_layers[bi]

        if bi in protected:
            print(f"  Block {bi + 1}/{n_blocks}: PROTECTED (left at original precision)")
            if bi == n_blocks - 1:
                break
            with torch.no_grad():
                cur_inps = [block(x, **kw).detach() for x, kw in zip(cur_inps, kwargs_list)]
            continue

        n_linears = sum(1 for _, m in block.named_modules() if isinstance(m, nn.Linear))

        # Teacher output FIRST, from the still-unquantized block. This is the
        # target QAT distills toward, and it must be captured before any
        # weight is touched.
        with torch.no_grad():
            teacher = [block(x, **kw).detach() for x, kw in zip(cur_inps, kwargs_list)]

        block.float()
        swapped = swap_linears(block)
        t0 = time.time()
        tr = train_block(block, swapped, cur_inps, kwargs_list, teacher,
                         steps=steps, lr=lr, batch=batch)
        train_s = time.time() - t0

        stats = bake_block(block, swapped, dtype)
        block.to(dtype)
        for name, st in stats.items():
            per_layer[f"model.layers.{bi}.{name}"] = {**st, "block_idx": bi}
            n_done += 1

        with torch.no_grad():
            after = [block(x, **kw).detach() for x, kw in zip(cur_inps, kwargs_list)]
            resid = sum(nn.functional.mse_loss(a.float(), t.float()).item()
                        for a, t in zip(after, teacher)) / len(after)
        per_block[str(bi)] = {**tr, "train_seconds": train_s,
                              "baked_output_mse_vs_teacher": resid,
                              "n_linears": n_linears}
        zf = sum(s["zero_fraction"] for s in stats.values()) / len(stats)
        print(f"  Block {bi + 1}/{n_blocks}: {n_linears} layers, "
              f"loss {tr['loss_first']:.5f} -> {tr['loss_last']:.5f}, "
              f"baked MSE {resid:.5f}, mean zero_frac {zf:.3f}, "
              f"{train_s:.1f}s  [{n_done}/{limit}]")

        if n_done >= limit or bi == n_blocks - 1:
            break
        cur_inps = after
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "n_layers_quantized": n_done,
        "per_layer": per_layer,
        "per_block": per_block,
        "protected_blocks": sorted(protected),
        "total_seconds": time.time() - t_total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--layers", type=int, default=None,
                        help="Only ternarize the first N nn.Linear layers "
                             "(14 == blocks 0-1, the scope GPTQ scored "
                             "1,336.567 on -- use it for a direct head-to-head)")
    parser.add_argument("--protect-first", type=int, default=0)
    parser.add_argument("--protect-last", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200,
                        help="Adam steps per block")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=4,
                        help="calibration sequences per step")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable. "
                         "Refusing to silently fall back to CPU.")
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 80)
    print(f"Block-local ternary QAT -- {MODEL_NAME}")
    print(f"Precision: {args.dtype} | device: {device}")
    if device.type == "cuda":
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"GPU: {torch.cuda.get_device_properties(0).name}, "
              f"{total_b / 1024**3:.2f}GB total / {free_b / 1024**3:.2f}GB free")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype).eval().to(device)

    text = load_wikitext2_test()
    full_ids = tokenizer(text, return_tensors="pt").input_ids[0]
    calib_start = CALIB_TOKEN_OFFSET % max(1, len(full_ids) - CALIB_SEQS * CALIB_SEQ_LEN)
    calib_end = calib_start + CALIB_SEQS * CALIB_SEQ_LEN
    assert calib_start >= args.max_tokens, (
        f"Calibration window [{calib_start},{calib_end}) overlaps eval "
        f"[0,{args.max_tokens}) -- would contaminate the comparison.")
    calib_text = tokenizer.decode(full_ids[calib_start:calib_end])
    print(f"  Calibration window [{calib_start}, {calib_end}), "
          f"eval [0, {args.max_tokens}), seq_len {args.seq_len}")

    print(f"\n--- Baseline ({args.dtype}) perplexity ---")
    base = compute_perplexity(model, tokenizer, text, args.seq_len, args.max_tokens)
    print(f"  Perplexity: {base['perplexity']:.3f}")

    print(f"\n--- Block-local ternary QAT "
          f"{'(FULL MODEL)' if args.layers is None else f'(first {args.layers} layers)'} ---")
    stats = run_qat(model, tokenizer, calib_text, dtype, args.layers,
                    args.protect_first, args.protect_last,
                    args.steps, args.lr, args.batch)

    print(f"\n--- Quantized (QAT) perplexity ---")
    quant = compute_perplexity(model, tokenizer, text, args.seq_len, args.max_tokens)
    print(f"  Perplexity: {quant['perplexity']:.3f}")

    deg = (quant["perplexity"] - base["perplexity"]) / base["perplexity"] * 100
    print("\n" + "-" * 80)
    print("Summary:")
    print(f"  Perplexity: {base['perplexity']:.3f} -> {quant['perplexity']:.3f} "
          f"[{deg:+.2f}%]  ({stats['n_layers_quantized']} layers ternarized)")
    print(f"  PTQ reference at matched scope (same quantizer, same windows, "
          f"weights ROUNDED not TRAINED):")
    print(f"    GPTQ, 14/154 layers (blocks 0-1) : 12.780 -> 1,336.567")
    print(f"    GPTQ, 154/154 layers             : 12.780 -> 18,565.469")
    print(f"    GPTQ, 112/154 (3+3 protected)    : 12.780 -> 14,285.862")
    print(f"  Success criterion (<5% loss): {'PASS' if abs(deg) < 5.0 else 'FAIL'}")

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_NAME, "dtype": args.dtype,
            "device": str(device),
            "device_name": (torch.cuda.get_device_properties(0).name
                            if device.type == "cuda" else None),
            "method": "block-local ternary QAT (STE) distilled against the "
                      "fp16 teacher block output; per-tensor absmean "
                      "quantizer identical to quantize_tinyllama_gptq.py's",
            "layers_limit": args.layers,
            "protect_first": args.protect_first, "protect_last": args.protect_last,
            "steps": args.steps, "lr": args.lr, "batch": args.batch,
            "seq_len": args.seq_len,
            "calib_window": [calib_start, calib_end],
            "eval_window": [0, args.max_tokens],
        },
        "baseline": base,
        "quantized_qat": quant,
        "perplexity_degradation_pct": deg,
        "quantization": stats,
        "ptq_reference": {
            "gptq_14_of_154": 1336.567,
            "gptq_154_of_154": 18565.469,
            "gptq_112_of_154_protect_3_3": 14285.862,
            "baseline": 12.780,
        },
        "success_criteria": {"accuracy_loss_under_5pct": abs(deg) < 5.0},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"tinyllama_qat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
