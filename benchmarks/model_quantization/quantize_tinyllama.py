"""
quantize_tinyllama.py - Real ternary quantization + perplexity measurement
for TinyLlama-1.1B-Chat-v1.0

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Closes Phase 5 of the competitive benchmark suite (real model quantization)
for one model. Scoped down from the framework's original three-model plan
(TinyLlama + Phi-2 + Gemma-2B) per user direction, 2026-08-22: on a
CPU-only sandbox, downloading and evaluating all three wasn't worth it for
a first real data point -- TinyLlama-1.1B is the smallest of the three and
proves the methodology.

Quantization: BitNet-style per-tensor absmean scale, applied to every
`nn.Linear` weight matrix inside the transformer's attention/MLP blocks
(q/k/v/o/gate/up/down projections). Embedding and lm_head layers are left
at full precision -- standard practice for this class of quantization
scheme, and consistent with this project's own framework text ("Simple
threshold-based" ternary mapping):

    scale = mean(|W|)
    W_ternary = round(clip(W / scale, -1, 1))   -- values in {-1, 0, +1}
    W_dequant = W_ternary * scale                -- dequantized back to
                                                     float32 so the SAME
                                                     unmodified HF model/
                                                     forward pass measures
                                                     accuracy degradation

IMPORTANT SCOPE NOTE: this measures the numerical accuracy cost of
ternary quantization, not this project's own ternary engine's kernel
speed. The forward pass here runs entirely through standard PyTorch
float32 ops on dequantized weights -- it does NOT route through
src/core/simd/ternary_gemm_dense.h or any other kernel in this repo.
Porting an HF transformer's forward pass onto this project's own kernels
would be a separate, much larger integration project; the kernel-speed
question has already been answered independently (see TritNet Phases 3-5
and reports/2026-08-20/GEMM_DENSE_PACKED_OPTIMIZATION.md) -- this script
answers a different, narrower question: "if you quantize a real model's
weights to ternary, how much accuracy do you lose?"

Perplexity corpus: WikiText-2 raw test split (Salesforce/wikitext,
config wikitext-2-raw-v1), the standard perplexity benchmark for this
class of measurement -- fetched directly via huggingface_hub + pyarrow
(no `datasets` package dependency) to keep this script's own dependency
footprint small.

USAGE: python benchmarks/model_quantization/quantize_tinyllama.py
       python benchmarks/model_quantization/quantize_tinyllama.py --max-tokens 4096
OUTPUT: benchmarks/results/model_quantization/tinyllama_<timestamp>.json
"""

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = SCRIPT_DIR.parent / "results" / "model_quantization"

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SEQ_LEN = 512          # tokens per evaluation block
DEFAULT_MAX_TOKENS = 8192  # total tokens evaluated (across blocks)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def load_wikitext2_test() -> str:
    """Fetch WikiText-2 raw test split directly (parquet, via
    huggingface_hub + pyarrow) without pulling in the full `datasets`
    package. Returns the concatenated raw text."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(
        repo_id="Salesforce/wikitext", repo_type="dataset",
        filename="wikitext-2-raw-v1/test-00000-of-00001.parquet",
    )
    table = pq.read_table(path)
    return "".join(table.column("text").to_pylist())


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_linear_weight(weight: torch.Tensor) -> tuple:
    """BitNet-style absmean ternary quantization of one weight tensor.

    Returns (dequantized_weight, scale, zero_fraction) so callers can
    both replace the weight in-place and accumulate sparsity/scale
    statistics across the whole model.
    """
    scale = weight.abs().mean().clamp(min=1e-8)
    ternary = torch.clamp(torch.round(weight / scale), -1, 1)
    dequant = ternary * scale
    zero_fraction = (ternary == 0).float().mean().item()
    return dequant, scale.item(), zero_fraction


def quantize_model_(model: nn.Module) -> dict:
    """Quantize every nn.Linear weight inside the model's transformer
    layers IN PLACE (embedding/lm_head excluded -- see module docstring).
    Returns per-layer stats for the report."""
    stats = {"layers": [], "total_quantized_params": 0, "total_params_all": 0}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # Exclude the LM head (often named lm_head, tied or untied) --
        # standard practice: keep the final projection at full precision.
        if "lm_head" in name:
            continue
        with torch.no_grad():
            dequant, scale, zero_frac = quantize_linear_weight(module.weight.data)
            module.weight.data.copy_(dequant)
        stats["layers"].append({
            "name": name,
            "shape": list(module.weight.shape),
            "scale": scale,
            "zero_fraction": zero_frac,
        })
        stats["total_quantized_params"] += module.weight.numel()

    stats["total_params_all"] = sum(p.numel() for p in model.parameters())
    return stats


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def compute_perplexity(model, tokenizer, text: str, seq_len: int, max_tokens: int) -> dict:
    """Perplexity over non-overlapping seq_len-token blocks, up to
    max_tokens total. Standard exp(mean per-token NLL) formulation."""
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    ids = ids[:max_tokens]
    n_blocks = len(ids) // seq_len
    if n_blocks == 0:
        raise ValueError(f"Not enough tokens ({len(ids)}) for one {seq_len}-token block")

    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()
    with torch.no_grad():
        for b in range(n_blocks):
            block = ids[b * seq_len:(b + 1) * seq_len].unsqueeze(0)
            out = model(input_ids=block, labels=block)
            # loss is mean NLL per token over (seq_len - 1) predicted positions
            n_pred = block.shape[1] - 1
            total_loss += out.loss.item() * n_pred
            total_tokens += n_pred
    elapsed = time.time() - t0

    mean_nll = total_loss / total_tokens
    ppl = math.exp(mean_nll)
    return {
        "perplexity": ppl,
        "mean_nll": mean_nll,
        "n_blocks": n_blocks,
        "tokens_evaluated": total_tokens,
        "eval_wall_seconds": elapsed,
        "seconds_per_block": elapsed / n_blocks,
    }


# ---------------------------------------------------------------------------
# Memory footprint (real parameter counts, not assumed)
# ---------------------------------------------------------------------------

def compute_memory_footprint(quant_stats: dict) -> dict:
    """Real memory footprint comparison using this run's actual measured
    parameter counts, not assumed model sizes."""
    n_quant = quant_stats["total_quantized_params"]
    n_total = quant_stats["total_params_all"]
    n_unquantized = n_total - n_quant  # embeddings, lm_head, norms, biases

    fp16_bytes_quant_layers = n_quant * 2
    fp16_bytes_unquantized = n_unquantized * 2
    fp16_bytes_total = fp16_bytes_quant_layers + fp16_bytes_unquantized

    # This project's own encodings, applied only to the quantized layers
    # (the unquantized layers stay fp16 in both scenarios, same cost):
    two_bit_bytes_quant_layers = n_quant * 2 / 8            # 2 bits/trit
    dense243_bytes_quant_layers = n_quant * 8 / 5 / 8        # 5 trits/byte, packed

    return {
        "quantized_params": n_quant,
        "unquantized_params": n_unquantized,
        "total_params": n_total,
        "fp16_total_mb": fp16_bytes_total / 1e6,
        "ternary_2bit_total_mb": (two_bit_bytes_quant_layers + fp16_bytes_unquantized) / 1e6,
        "ternary_dense243_total_mb": (dense243_bytes_quant_layers + fp16_bytes_unquantized) / 1e6,
        "reduction_vs_fp16_2bit": fp16_bytes_total / (two_bit_bytes_quant_layers + fp16_bytes_unquantized),
        "reduction_vs_fp16_dense243": fp16_bytes_total / (dense243_bytes_quant_layers + fp16_bytes_unquantized),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis(max_tokens: int = DEFAULT_MAX_TOKENS, seq_len: int = SEQ_LEN) -> dict:
    """Library entry point -- no argparse/sys.argv involved, so this can
    be called directly from another script (e.g. bench_competitive.py's
    Phase 5) without that caller's own CLI arguments (--all, --phase, ...)
    being mistaken for this module's flags."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 80)
    print(f"Phase 5: Real Model Quantization -- {MODEL_NAME}")
    print("=" * 80)

    print("\nLoading model and tokenizer...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("\nFetching WikiText-2 test corpus...")
    text = load_wikitext2_test()
    print(f"  {len(text):,} characters")

    print(f"\n--- Baseline (fp32) perplexity, {max_tokens} tokens, "
          f"{seq_len}-token blocks ---")
    baseline_ppl = compute_perplexity(model, tokenizer, text, seq_len, max_tokens)
    print(f"  Perplexity: {baseline_ppl['perplexity']:.3f}  "
          f"({baseline_ppl['n_blocks']} blocks, "
          f"{baseline_ppl['eval_wall_seconds']:.1f}s)")

    print("\n--- Quantizing all attention/MLP nn.Linear weights to ternary ---")
    t0 = time.time()
    quant_stats = quantize_model_(model)
    print(f"  Quantized {len(quant_stats['layers'])} layers "
          f"({quant_stats['total_quantized_params']:,} / "
          f"{quant_stats['total_params_all']:,} params) in {time.time() - t0:.1f}s")
    zero_fracs = [l["zero_fraction"] for l in quant_stats["layers"]]
    print(f"  Zero fraction across layers: mean={sum(zero_fracs)/len(zero_fracs):.3f}, "
          f"min={min(zero_fracs):.3f}, max={max(zero_fracs):.3f}")

    print(f"\n--- Quantized (ternary) perplexity, same {max_tokens} tokens ---")
    quant_ppl = compute_perplexity(model, tokenizer, text, seq_len, max_tokens)
    print(f"  Perplexity: {quant_ppl['perplexity']:.3f}  "
          f"({quant_ppl['n_blocks']} blocks, "
          f"{quant_ppl['eval_wall_seconds']:.1f}s)")

    ppl_degradation_pct = (
        (quant_ppl["perplexity"] - baseline_ppl["perplexity"]) / baseline_ppl["perplexity"] * 100
    )
    latency_ratio = quant_ppl["seconds_per_block"] / baseline_ppl["seconds_per_block"]

    memory = compute_memory_footprint(quant_stats)

    print("\n" + "-" * 80)
    print("Phase 5 Summary:")
    print(f"  Perplexity: {baseline_ppl['perplexity']:.3f} (fp32) -> "
          f"{quant_ppl['perplexity']:.3f} (ternary)  "
          f"[{ppl_degradation_pct:+.2f}%]")
    print(f"  Success criterion (<5% accuracy loss): "
          f"{'PASS' if abs(ppl_degradation_pct) < 5.0 else 'FAIL'}")
    print(f"  Forward-pass latency ratio (ternary-simulated / fp32, "
          f"SAME PyTorch ops -- NOT this project's ternary kernels): "
          f"{latency_ratio:.3f}x")
    print(f"  Memory: {memory['fp16_total_mb']:.1f} MB (fp16) -> "
          f"{memory['ternary_dense243_total_mb']:.1f} MB (Dense243, quantized layers only) "
          f"[{memory['reduction_vs_fp16_dense243']:.2f}x smaller]")
    print(f"  Success criterion (<25% of FP16 memory): "
          f"{'PASS' if memory['ternary_dense243_total_mb'] / memory['fp16_total_mb'] < 0.25 else 'FAIL'}")
    print()
    print("  IMPORTANT: the latency ratio above reflects standard PyTorch float32")
    print("  ops on dequantized ternary weights, NOT this project's own AVX2 ternary")
    print("  kernels (src/core/simd/ternary_gemm_dense.h) -- that integration is out")
    print("  of scope for this quantization-accuracy measurement. The kernel-speed")
    print("  question is answered separately: see reports/2026-08-20/")
    print("  GEMM_DENSE_PACKED_OPTIMIZATION.md and TritNet Phases 3-5.")

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "corpus": "Salesforce/wikitext, wikitext-2-raw-v1, test split",
            "quantization": "BitNet-style absmean ternary, nn.Linear in "
                             "attention/MLP blocks only (embedding/lm_head excluded)",
            "scope_note": "Latency figures are standard PyTorch fp32 ops on "
                           "dequantized weights, NOT this project's own ternary "
                           "engine kernels -- see module docstring.",
        },
        "baseline_fp32": baseline_ppl,
        "quantized_ternary": quant_ppl,
        "perplexity_degradation_pct": ppl_degradation_pct,
        "latency_ratio_ternary_over_fp32": latency_ratio,
        "quantization_stats": {
            "n_layers_quantized": len(quant_stats["layers"]),
            "total_quantized_params": quant_stats["total_quantized_params"],
            "total_params_all": quant_stats["total_params_all"],
            "zero_fraction_mean": sum(zero_fracs) / len(zero_fracs),
            "zero_fraction_min": min(zero_fracs),
            "zero_fraction_max": max(zero_fracs),
            "per_layer": quant_stats["layers"],
        },
        "memory_footprint": memory,
        "success_criteria": {
            "accuracy_loss_under_5pct": abs(ppl_degradation_pct) < 5.0,
            "memory_under_25pct_of_fp16": memory["ternary_dense243_total_mb"] / memory["fp16_total_mb"] < 0.25,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"tinyllama_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                         help=f"Total tokens to evaluate (default {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN,
                         help=f"Tokens per evaluation block (default {SEQ_LEN})")
    args = parser.parse_args()
    return run_analysis(max_tokens=args.max_tokens, seq_len=args.seq_len)


if __name__ == "__main__":
    main()
