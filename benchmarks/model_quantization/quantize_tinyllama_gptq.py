"""
quantize_tinyllama_gptq.py - GPTQ-style calibrated ternary quantization
for TinyLlama-1.1B-Chat-v1.0

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Direct follow-up to quantize_tinyllama.py's two negative results
(per-tensor: perplexity 12.78 -> 89,101; per-channel: -> 132,590 --
WORSE, falsifying "the scale is too coarse" as the explanation). Both
of those quantize each weight independently with no calibration data
and no error compensation. This script tests the actual next-cheapest
hypothesis: does real error compensation (not finer scale granularity)
help, or is naive post-training ternary quantization broken at a more
fundamental level regardless of technique?

Algorithm: a direct implementation of GPTQ's core recipe (Frantar et al.,
"GPTQ: Accurate Post-Training Quantization for Generative Pre-trained
Transformers"), adapted to a ternary {-1,0,+1} quantizer instead of
GPTQ's original int4/int8 uniform quantizer -- the calibration/Hessian/
error-propagation machinery is otherwise unchanged from the published
algorithm:

  1. Calibration: run real WikiText-2 text through the model once,
     capturing each target layer's input activations via forward hooks,
     accumulating the Hessian approximation H = (2/n) * X^T X per layer
     (accumulated incrementally, not by storing raw activations -- the
     activation tensor for one layer across the whole calibration set
     would be prohibitively large).
  2. Per layer, per column (processed left to right): quantize that
     column to ternary using this project's absmean scheme, then
     propagate the resulting rounding error to every NOT-yet-quantized
     column using the (Cholesky-decomposed) inverse Hessian -- this is
     the actual defining mechanism of GPTQ, and the reason it can
     recover accuracy per-tensor/per-channel rounding cannot: instead of
     discarding each column's rounding error, later columns are nudged
     to compensate for it, weighted by how strongly they're correlated
     with the column that was just quantized (via H).

IMPORTANT SCOPE NOTE: same as quantize_tinyllama.py -- this measures
quantization accuracy, not this project's own ternary engine's kernel
speed (see that file's docstring for the full explanation, unchanged
here).

Calibration and evaluation use DISJOINT slices of the WikiText-2 test
corpus (calibration: tokens [CALIB_START, CALIB_START+CALIB_TOKENS);
evaluation: the same [0, max_tokens) window quantize_tinyllama.py's
baseline/per-tensor/per-channel runs used) so the perplexity comparison
isn't contaminated by the quantizer having seen the eval text.

USAGE: python benchmarks/model_quantization/quantize_tinyllama_gptq.py
       python benchmarks/model_quantization/quantize_tinyllama_gptq.py --layers 7
           (process only the first N nn.Linear layers found -- for timing
           a partial run before committing to the full ~154-layer model)
       python benchmarks/model_quantization/quantize_tinyllama_gptq.py \
           --device cuda --protect-first 3 --protect-last 3
           (full-model mixed-precision run on GPU -- the step
           docs/planning/ROADMAP.md calls for; see --device below)

DEVICE (added 2026-08-28): every earlier run of this script was CPU-only.
That is the direct reason the full-model mixed-precision run the roadmap
asks for had never been executed -- a full 154-layer pass costs ~2.5-3.5h
on this machine's CPU, and it reboots often enough mid-run for that to be
a real obstacle (which is also why the checkpoint/resume machinery below
exists). --device cuda moves the resident model, the calibration forward
passes, the Hessian accumulation and the GPTQ column loop onto the GPU.
The mathematics is untouched; only where it executes changes.
OUTPUT: benchmarks/results/model_quantization/tinyllama_gptq_<timestamp>.json
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
RESULTS_DIR = SCRIPT_DIR.parent / "results" / "model_quantization"

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SEQ_LEN = 512
DEFAULT_MAX_TOKENS = 8192          # eval window, matches quantize_tinyllama.py
CALIB_TOKEN_OFFSET = 400_000       # calibration text starts well past the eval
                                    # window so the two never overlap (corpus
                                    # is ~332K tokens total per the earlier
                                    # run's tokenizer warning -- offset chosen
                                    # deliberately past that so calibration
                                    # wraps to a distinct region, verified below)
CALIB_SEQS = 32                    # number of calibration sequences
CALIB_SEQ_LEN = 512                # tokens per calibration sequence

DEFAULT_CHECKPOINT_DIR = SCRIPT_DIR.parent / "results" / "model_quantization" / "gptq_checkpoints"
# Repo-relative, NOT /tmp: this machine has been observed to reboot outright
# (not just restart the Claude Code session) multiple times during a single
# multi-hour run (confirmed via `uptime`/`last` mid-session, 2026-08-28),
# which wipes /tmp unconditionally. A full 154-layer run takes ~2.5-3.5h
# (see collect_calibration_and_quantize's docstring); checkpointing to a
# path that survives a reboot is what makes that tractable here at all --
# simply rerunning the same command resumes from the last completed block.


# ---------------------------------------------------------------------------
# Checkpoint / resume (see DEFAULT_CHECKPOINT_DIR comment above for why)
# ---------------------------------------------------------------------------

def _manifest_path(ckpt_dir: Path) -> Path:
    return ckpt_dir / "manifest.json"


def load_checkpoint_manifest(ckpt_dir: Path) -> list:
    """Returns the sorted list of block indices already fully quantized
    and saved to disk. Empty if no checkpoint exists yet."""
    path = _manifest_path(ckpt_dir)
    if not path.exists():
        return []
    with open(path) as f:
        return sorted(json.load(f)["completed_blocks"])


def save_block_checkpoint(ckpt_dir: Path, block_idx: int, layer_weights: dict,
                           completed_blocks: list) -> None:
    """Saves one block's quantized layer weights (name -> tensor, fp16,
    typically a few MB to ~50MB per block) to its own file, then updates
    the manifest. Per-block files (not one growing merged file) so this
    never has to rewrite already-written data as the run progresses."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(layer_weights, ckpt_dir / f"block_{block_idx:03d}.pt")
    with open(_manifest_path(ckpt_dir), "w") as f:
        json.dump({"completed_blocks": sorted(completed_blocks)}, f)


def apply_block_checkpoint(model, ckpt_dir: Path, block_idx: int) -> dict:
    """Loads one previously-completed block's quantized weights back into
    the model in place. Returns {layer_name: weight_tensor} for stat
    reconstruction (zero_fraction is legitimately recomputable from the
    final weight tensor alone; other per-layer stats like dead_cols are
    not, and are simply omitted for resumed layers rather than faked)."""
    layer_weights = torch.load(ckpt_dir / f"block_{block_idx:03d}.pt", map_location="cpu")
    for name, w in layer_weights.items():
        model.get_submodule(name).weight.data.copy_(w)
    return layer_weights


# ---------------------------------------------------------------------------
# Corpus (shared logic with quantize_tinyllama.py, duplicated minimally to
# keep this script runnable standalone without import-path gymnastics)
# ---------------------------------------------------------------------------

def load_wikitext2_test() -> str:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(
        repo_id="Salesforce/wikitext", repo_type="dataset",
        filename="wikitext-2-raw-v1/test-00000-of-00001.parquet",
    )
    table = pq.read_table(path)
    return "".join(table.column("text").to_pylist())


# ---------------------------------------------------------------------------
# GPTQ core
# ---------------------------------------------------------------------------

class GPTQLayerQuantizer:
    """Accumulates a Hessian approximation for one nn.Linear layer's input
    activations across calibration batches, then quantizes that layer's
    weight in place using GPTQ's sequential column update."""

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self.rows, self.columns = layer.weight.shape  # [out_features, in_features]
        # H lives on the same device as the layer it describes, so add_batch()
        # never forces a device transfer of the activation tensor (the hook
        # fires once per calibration sequence per layer -- a per-call H2D/D2H
        # round trip there would dominate the GPU path's runtime).
        self.device = layer.weight.device
        self.H = torch.zeros((self.columns, self.columns), dtype=torch.float32,
                             device=self.device)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor):
        """inp: the input activations this layer saw, shape [..., columns].
        Accumulates H = (2/n) * X^T X incrementally (running-average form,
        matching the published GPTQ implementation) so raw activations
        never need to be stored."""
        if inp.dim() > 2:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.float()
        n = inp.shape[0]
        if n == 0:
            return
        self.H *= self.nsamples / (self.nsamples + n)
        self.nsamples += n
        inp_t = inp.t() * math.sqrt(2.0 / self.nsamples)
        self.H += inp_t @ inp_t.t()

    def quantize(self, percdamp: float = 0.01) -> dict:
        """Quantizes self.layer.weight in place. Returns stats for the report."""
        W = self.layer.weight.data.clone().float()
        if self.H is None:
            raise RuntimeError(
                "quantize() called twice on the same GPTQLayerQuantizer -- the "
                "Hessian is consumed (not copied) by the first call."
            )
        # Ownership transfer, not a copy: self.H is never read again after
        # this point, and cloning it momentarily doubled the largest Hessian
        # (down_proj is 5,632 columns = ~127MB fp32) for no benefit. On a 6GB
        # card shared with a desktop session that headroom is worth having.
        H = self.H
        self.H = None

        # Columns with zero Hessian entries (never activated during
        # calibration) can't be meaningfully compensated -- zero them out
        # rather than divide by zero, matching the published GPTQ recipe.
        dead = torch.diag(H) == 0
        n_dead = int(dead.sum().item())
        if n_dead > 0:
            H[dead, dead] = 1
            W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag_idx = torch.arange(self.columns, device=H.device)
        H[diag_idx, diag_idx] += damp

        # Cholesky of H, invert via cholesky_inverse, then Cholesky of the
        # inverse (upper triangular) -- the specific numerically-stable
        # recipe the published GPTQ implementation uses, rather than a
        # direct H.inverse() call.
        H_chol = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(H_chol)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)

        # Single per-tensor scale, computed once up front -- matches
        # quantize_tinyllama.py's per-tensor scheme (the less-bad of the
        # two naive baselines), so this run isolates the effect of error
        # compensation specifically, not a second simultaneous change to
        # scale granularity.
        scale = W.abs().mean().clamp(min=1e-8)

        # Accumulated as a device-resident tensor rather than an int: the
        # original `int(...sum().item())` inside the loop forced a GPU->CPU
        # sync on EVERY column (5,632 of them for down_proj), which stalls the
        # pipeline and would have dominated the CUDA path's runtime. Summed
        # once at the end instead -- identical value, no per-column sync.
        zero_count_t = torch.zeros((), dtype=torch.int64, device=W.device)
        for i in range(self.columns):
            w_col = W[:, i]
            d = Hinv[i, i]
            q_col = torch.clamp(torch.round(w_col / scale), -1, 1) * scale
            zero_count_t += (q_col == 0).sum()
            err = (w_col - q_col) / d
            if i + 1 < self.columns:
                W[:, i + 1:] -= torch.outer(err, Hinv[i, i + 1:])
            W[:, i] = q_col
        zero_count = int(zero_count_t.item())

        self.layer.weight.data.copy_(W)

        # Free this layer's working set now rather than at block end: a block
        # holds up to ~227MB of Hessian across its 7 Linears, and the Cholesky
        # chain above allocated several more temporaries of that same size.
        del H, H_chol, Hinv, W

        return {
            "scale": scale.item(),
            "dead_columns": n_dead,
            "zero_fraction": zero_count / (self.rows * self.columns),
            "nsamples": self.nsamples,
        }


def collect_calibration_and_quantize(model, tokenizer, calib_text: str,
                                      n_layers_limit: int = None,
                                      checkpoint_dir: Path = None,
                                      fresh: bool = False,
                                      protect_first: int = 0,
                                      protect_last: int = 0) -> dict:
    """Block-sequential GPTQ calibration, with reboot-resilient checkpointing.

    An EARLIER version of this function hooked one nn.Linear at a time and
    reran the FULL model (all 22 decoder blocks + lm_head) through every
    calibration sequence just to capture that one layer's input -- 888s per
    layer, times 154 layers, ~38 hours for the whole model (measured
    2026-08-26, see reports/2026-08-26/ for the timing run that surfaced
    this). The redundancy: capturing layer i's input only requires
    computing blocks [0, block_of(i)], not the other ~21 blocks and
    lm_head past it.

    This version processes one TRANSFORMER BLOCK at a time instead of one
    Linear at a time:
      1. A "Catcher" wrapper around block 0 captures its real input
         hidden-states AND the kwargs (attention_mask, position_ids,
         position_embeddings) the real model forward computed for it, for
         every calibration sequence, then aborts the forward via
         StopIteration -- so only block 0 (not blocks 1-21 or lm_head) is
         ever actually computed during this one-time capture step.
      2. For each block in order: hook every nn.Linear inside just that
         block, run the block's forward directly on the current cached
         inputs (no full-model call), accumulating each sub-layer's
         Hessian in ONE pass through ONE block.
      3. Quantize that block's layers (sequentially, GPTQ's column-by-
         column error propagation, unchanged from before).
      4. Re-run the block once more (now with its quantized weights) on
         the same cached inputs to get the block's real output -- this
         becomes the next block's input, so quantization error compounds
         forward through the network exactly as GPTQ's algorithm requires
         (unchanged semantics from the original implementation, just
         computed far more cheaply).

    past_key_values/use_cache are deliberately forced to None/False for
    every block-only replay call (steps 2 and 4 both call the same block
    twice) rather than reusing the kwargs Catcher captured verbatim --
    avoids any dependence on a stateful KV-cache object being safely
    reusable across two separate forward calls to the same block, a
    behavior this implementation doesn't need and shouldn't have to trust
    across transformers versions.

    VERIFIED numerically identical to the original per-Linear-full-rerun
    approach before this rewrite replaced it (scratchpad verification,
    not committed -- Hessians matched the old method bit-for-bit
    (max abs diff 0.0) on real q_proj/model data, chained block outputs
    matched the real full-model hidden states bit-for-bit, and the
    forced-no-cache replay path was confirmed numerically identical to
    the captured use_cache=True path). This function's OUTPUT (which
    weights end up quantized to what values) is unchanged by this
    rewrite; only how cheaply it gets there changed.

    checkpoint_dir, if given, makes this resumable across a full machine
    reboot (see DEFAULT_CHECKPOINT_DIR's module-level comment for why that
    matters here specifically): each block's quantized weights are saved
    to their own file plus a manifest the instant that block finishes, so
    simply rerunning the same command after an interruption picks up from
    the next un-quantized block instead of starting over. Already-
    completed blocks' weights are loaded back into the model, then
    "fast-forwarded" through (plain forward calls, no hooks/quantization)
    to reconstruct the correct input for the first block still needing
    work -- the same block-only-forward mechanism already used for the
    normal post-quantization recompute step, just without redoing the
    Hessian/quantization work for blocks already on disk.

    protect_first/protect_last (mixed-precision PTQ): the strategic note
    added to docs/planning/ROADMAP.md 2026-08-28 -- every full-precision
    quantization attempt so far (naive per-tensor, naive per-channel, and
    now GPTQ block-sequential itself) has failed, and each more
    sophisticated attempt failed in the SAME direction, not a better one.
    Nearly every published low-bit technique keeps its most sensitive
    layers (embeddings, first/last transformer blocks) at higher
    precision rather than uniformly quantizing 100% of the model -- this
    is that experiment. Blocks with index < protect_first or
    >= (n_blocks - protect_last) are left at their original fp16 weights
    entirely (no hooks, no calibration, no quantization -- just a plain
    forward pass to propagate the correct hidden states to the next
    block, since even an untouched block still needs to hand off its
    real output)."""
    decoder_layers = model.model.layers
    n_total_layers = sum(
        1 for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and "lm_head" not in n
    )
    limit = n_layers_limit if n_layers_limit is not None else n_total_layers
    n_blocks = len(decoder_layers)
    protected_blocks = set(range(protect_first)) | set(range(n_blocks - protect_last, n_blocks))
    if protected_blocks:
        print(f"  Mixed precision: protecting {len(protected_blocks)} block(s) "
              f"at original fp16 (indices {sorted(protected_blocks)})")

    completed_blocks = []
    per_layer_stats = {}
    n_quantized = 0
    resume_from_block = 0
    if checkpoint_dir is not None:
        if fresh:
            import shutil
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
        else:
            completed_blocks = load_checkpoint_manifest(checkpoint_dir)
            if completed_blocks:
                print(f"  Resuming from checkpoint: {len(completed_blocks)} "
                      f"block(s) already done ({checkpoint_dir})")
                for b in completed_blocks:
                    layer_weights = apply_block_checkpoint(model, checkpoint_dir, b)
                    for name, w in layer_weights.items():
                        per_layer_stats[name] = {
                            "shape": list(w.shape),
                            "block_idx": b,
                            "zero_fraction": float((w == 0).sum().item()) / w.numel(),
                            "resumed_from_checkpoint": True,
                        }
                        n_quantized += 1
                resume_from_block = max(completed_blocks) + 1
                print(f"  {n_quantized} layer(s) restored; resuming at block "
                      f"{resume_from_block}/{len(decoder_layers)}")
    initially_completed_blocks = list(completed_blocks)

    print(f"  Quantizing up to {limit} of {n_total_layers} total Linear layers, "
          f"across {len(decoder_layers)} transformer blocks (block-sequential)")
    if n_quantized >= limit:
        print(f"  Checkpoint already covers the requested {limit}-layer limit "
              f"-- nothing left to do.")
        return {
            "n_layers_quantized": n_quantized,
            "per_layer": per_layer_stats,
            "total_seconds": 0.0,
            "resumed_from_checkpoint_blocks": completed_blocks,
            "blocks_quantized_this_run": [],
            "protected_blocks": sorted(protected_blocks),
        }

    device = next(model.parameters()).device
    ids = tokenizer(calib_text, return_tensors="pt").input_ids[0]
    n_seqs = min(CALIB_SEQS, len(ids) // CALIB_SEQ_LEN)
    calib_batches = [
        ids[s * CALIB_SEQ_LEN:(s + 1) * CALIB_SEQ_LEN].unsqueeze(0).to(device)
        for s in range(n_seqs)
    ]
    print(f"  Calibration: {n_seqs} sequences x {CALIB_SEQ_LEN} tokens")

    class _Catcher(nn.Module):
        """Wraps decoder block 0; records its input + the kwargs the real
        model forward built for it, then aborts the forward immediately so
        blocks 1-21 and lm_head are never computed during this capture."""
        def __init__(self, module, sink):
            super().__init__()
            self.module = module
            self.sink = sink

        def forward(self, hidden_states, **kwargs):
            self.sink["inps"].append(hidden_states.detach().clone())
            self.sink["kwargs_list"].append({
                "attention_mask": kwargs.get("attention_mask"),
                "position_ids": kwargs.get("position_ids"),
                "position_embeddings": kwargs.get("position_embeddings"),
                "past_key_values": None,   # forced -- see docstring
                "use_cache": False,        # forced -- see docstring
            })
            raise StopIteration()

    sink = {"inps": [], "kwargs_list": []}
    orig_block0 = decoder_layers[0]
    decoder_layers[0] = _Catcher(orig_block0, sink)
    t_cap = time.time()
    with torch.no_grad():
        for batch in calib_batches:
            try:
                model(input_ids=batch)
            except StopIteration:
                pass
    decoder_layers[0] = orig_block0
    print(f"  Captured block-0 inputs for {len(sink['inps'])} sequences "
          f"in {time.time() - t_cap:.2f}s")

    cur_inps = sink["inps"]
    kwargs_list = sink["kwargs_list"]

    if resume_from_block > 0:
        t0 = time.time()
        with torch.no_grad():
            for b in range(resume_from_block):
                cur_inps = [
                    decoder_layers[b](inps, **kwargs).detach()
                    for inps, kwargs in zip(cur_inps, kwargs_list)
                ]
        print(f"  Fast-forwarded through {resume_from_block} completed "
              f"block(s) in {time.time() - t0:.1f}s (weights already "
              f"quantized, no recompute needed)")

    t_total = time.time()

    for block_idx in range(resume_from_block, len(decoder_layers)):
        if n_quantized >= limit:
            break
        block = decoder_layers[block_idx]

        if block_idx in protected_blocks:
            print(f"  Block {block_idx + 1}/{len(decoder_layers)}: PROTECTED "
                  f"(kept at original fp16, not quantized)")
            if block_idx == len(decoder_layers) - 1:
                break
            t0 = time.time()
            with torch.no_grad():
                cur_inps = [
                    block(inps, **kwargs).detach()
                    for inps, kwargs in zip(cur_inps, kwargs_list)
                ]
            print(f"    (forward-only recompute for next block: {time.time() - t0:.1f}s)")
            continue

        block_targets = [
            (f"model.layers.{block_idx}.{name}", module)
            for name, module in block.named_modules()
            if isinstance(module, nn.Linear)
        ]

        quantizers = {name: GPTQLayerQuantizer(module) for name, module in block_targets}
        handles = []
        for name, module in block_targets:
            q = quantizers[name]

            def hook(mod, inp, output, q=q):
                q.add_batch(inp[0].detach())

            handles.append(module.register_forward_hook(hook))

        t0 = time.time()
        with torch.no_grad():
            for inps, kwargs in zip(cur_inps, kwargs_list):
                block(inps, **kwargs)
        for h in handles:
            h.remove()
        calib_time = time.time() - t0

        block_quant_time = 0.0
        n_quantized_at_block_start = n_quantized
        for name, module in block_targets:
            if n_quantized >= limit:
                break
            t0 = time.time()
            stats = quantizers[name].quantize()
            qt = time.time() - t0
            block_quant_time += qt
            stats["block_calib_seconds"] = calib_time
            stats["quant_seconds"] = qt
            stats["shape"] = list(module.weight.shape)
            stats["block_idx"] = block_idx
            per_layer_stats[name] = stats
            n_quantized += 1
            print(f"    [{n_quantized}/{limit}] {name}: shape={stats['shape']} "
                  f"quant={qt:.1f}s zero_frac={stats['zero_fraction']:.3f} "
                  f"dead_cols={stats['dead_columns']}")

        print(f"  Block {block_idx + 1}/{len(decoder_layers)} "
              f"(calib={calib_time:.1f}s, quant_total={block_quant_time:.1f}s): "
              f"{n_quantized}/{limit} layers quantized so far")

        # Drop this block's quantizers (and with them any Hessian not already
        # freed inside quantize(), e.g. layers skipped by a --layers limit)
        # before the next block allocates its own. On CUDA, also return the
        # freed blocks to the driver: this model's per-block Hessian working
        # set is a meaningful fraction of a 6GB card shared with a desktop
        # session, and PyTorch's caching allocator will otherwise hold them.
        quantizers.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Only checkpoint a block once EVERY one of its Linears has been
        # quantized -- a partial block (cut short by --layers, used for
        # timing tests only) would otherwise look "done" to a future
        # resume and get skipped with some of its layers still unquantized.
        full_block_done = (n_quantized - n_quantized_at_block_start) == len(block_targets)
        if checkpoint_dir is not None and full_block_done:
            layer_weights = {
                name: module.weight.data.clone().cpu()
                for name, module in block_targets
            }
            completed_blocks.append(block_idx)
            save_block_checkpoint(checkpoint_dir, block_idx, layer_weights, completed_blocks)
            print(f"    (checkpoint saved: block {block_idx} -> {checkpoint_dir})")

        if n_quantized >= limit or block_idx == len(decoder_layers) - 1:
            break

        # Re-run this block with its (now quantized) weights to get the
        # correct output -- becomes the next block's input.
        t0 = time.time()
        next_inps = []
        with torch.no_grad():
            for inps, kwargs in zip(cur_inps, kwargs_list):
                next_inps.append(block(inps, **kwargs).detach())
        recompute_time = time.time() - t0
        cur_inps = next_inps
        print(f"    (post-quant recompute for next block: {recompute_time:.1f}s)")

    total_time = time.time() - t_total
    print(f"  Total calibration+quantization time: {total_time:.1f}s "
          f"({total_time / 60:.1f} min)")

    return {
        "n_layers_quantized": n_quantized,
        "per_layer": per_layer_stats,
        "total_seconds": total_time,
        "resumed_from_checkpoint_blocks": initially_completed_blocks,
        "blocks_quantized_this_run": [b for b in completed_blocks if b not in initially_completed_blocks],
        "protected_blocks": sorted(protected_blocks),
    }


# ---------------------------------------------------------------------------
# Perplexity (identical formulation to quantize_tinyllama.py)
# ---------------------------------------------------------------------------

def compute_perplexity(model, tokenizer, text: str, seq_len: int, max_tokens: int) -> dict:
    device = next(model.parameters()).device
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    ids = ids[:max_tokens].to(device)
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
            n_pred = block.shape[1] - 1
            total_loss += out.loss.item() * n_pred
            total_tokens += n_pred
    elapsed = time.time() - t0

    mean_nll = total_loss / total_tokens
    return {
        "perplexity": math.exp(mean_nll),
        "mean_nll": mean_nll,
        "n_blocks": n_blocks,
        "tokens_evaluated": total_tokens,
        "eval_wall_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--layers", type=int, default=None,
                         help="Only quantize the first N nn.Linear layers found "
                              "(for timing/validation runs before committing to "
                              "the full model)")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16",
                         help="Model load/compute precision (default fp16). "
                              "Switched from quantize_tinyllama.py's fp32 default "
                              "2026-08-25: on this machine, a resident fp32 "
                              "TinyLlama (~4.4-4.9GB) sharing 7GB total RAM with "
                              "a live desktop session caused real swap thrashing "
                              "(confirmed via /proc/<pid>/status VmSwap, not "
                              "assumed) -- fp16 halves the model's memory "
                              "footprint. The quantization math itself always "
                              "upconverts to fp32 internally (see "
                              "GPTQLayerQuantizer.quantize()) regardless of this "
                              "flag, for numerical stability. NOTE: this means "
                              "the baseline perplexity here is a FRESH fp16 "
                              "number, not directly the same fp32 baseline "
                              "(12.780) quantize_tinyllama.py established -- "
                              "reported as its own baseline, not conflated.")
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR),
                         help="Directory for per-block checkpoints, enabling resume "
                              "across a full machine reboot (default: "
                              f"{DEFAULT_CHECKPOINT_DIR}). Rerunning the exact same "
                              "command after an interruption auto-resumes from the "
                              "last completed block. Pass --no-checkpoint to disable.")
    parser.add_argument("--no-checkpoint", action="store_true",
                         help="Disable checkpointing entirely (a full-model run "
                              "will not survive an interruption if you use this).")
    parser.add_argument("--fresh", action="store_true",
                         help="Ignore/delete any existing checkpoint and start over.")
    parser.add_argument("--protect-first", type=int, default=0,
                         help="Keep the first N transformer blocks at original "
                              "fp16 precision instead of quantizing them (mixed "
                              "precision -- most published low-bit techniques "
                              "protect embeddings/early layers rather than "
                              "quantizing 100%% of the model uniformly).")
    parser.add_argument("--protect-last", type=int, default=0,
                         help="Keep the last M transformer blocks at original "
                              "fp16 precision (see --protect-first).")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                         help="Compute device (default auto: CUDA if available, "
                              "else CPU). Added 2026-08-28: every prior run of "
                              "this script was CPU-only, which is why the "
                              "full-model mixed-precision run docs/planning/"
                              "ROADMAP.md calls for had never actually been "
                              "executed -- a full 154-layer pass costs ~2.5-3.5h "
                              "on CPU here, and this machine reboots often enough "
                              "mid-run for that to matter. The quantization math "
                              "is unchanged by this flag; only where it runs is. "
                              "'cuda' is an explicit request and FAILS LOUDLY if "
                              "unavailable rather than silently falling back to "
                              "CPU and reporting a CPU-speed result as if it were "
                              "a GPU one.")
    args = parser.parse_args()
    checkpoint_dir = None if args.no_checkpoint else Path(args.checkpoint_dir)
    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "--device cuda requested but torch.cuda.is_available() is False. "
                "Refusing to silently fall back to CPU (a CPU run mislabeled as "
                "GPU would produce a meaningless timing comparison). Use "
                "--device cpu or --device auto if a CPU run is what you want."
            )
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 80)
    print(f"GPTQ-style calibrated ternary quantization -- {MODEL_NAME}")
    print(f"Load/compute precision: {args.dtype}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"Device: cuda -- {props.name}, compute {props.major}.{props.minor}, "
              f"{total_b / 1024**3:.2f}GB total / {free_b / 1024**3:.2f}GB free")
    else:
        print("Device: cpu")
    print("=" * 80)

    print("\nLoading model and tokenizer...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch_dtype)
    model.eval()
    model.to(device)
    print(f"  Loaded in {time.time() - t0:.1f}s (device={device})")
    if device.type == "cuda":
        print(f"  Model resident on GPU: "
              f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB allocated")

    print("\nFetching WikiText-2 test corpus...")
    text = load_wikitext2_test()
    full_ids = tokenizer(text, return_tensors="pt").input_ids[0]
    print(f"  {len(text):,} characters, {len(full_ids):,} tokens")

    calib_start = CALIB_TOKEN_OFFSET % max(1, len(full_ids) - CALIB_SEQS * CALIB_SEQ_LEN)
    calib_end = calib_start + CALIB_SEQS * CALIB_SEQ_LEN
    assert calib_start >= args.max_tokens, (
        f"Calibration window [{calib_start},{calib_end}) overlaps the eval "
        f"window [0,{args.max_tokens}) -- would contaminate the perplexity "
        f"comparison. Adjust CALIB_TOKEN_OFFSET."
    )
    calib_text = tokenizer.decode(full_ids[calib_start:calib_end])
    print(f"  Calibration window: tokens [{calib_start}, {calib_end}) "
          f"(disjoint from eval window [0, {args.max_tokens}))")

    print(f"\n--- Baseline ({args.dtype}) perplexity, {args.max_tokens} tokens ---")
    baseline_ppl = compute_perplexity(model, tokenizer, text, args.seq_len, args.max_tokens)
    print(f"  Perplexity: {baseline_ppl['perplexity']:.3f} "
          f"({baseline_ppl['eval_wall_seconds']:.1f}s)")

    print(f"\n--- GPTQ-style calibrated quantization "
          f"{'(FULL MODEL)' if args.layers is None else f'(first {args.layers} layers only)'} ---")
    quant_stats = collect_calibration_and_quantize(
        model, tokenizer, calib_text, args.layers,
        checkpoint_dir=checkpoint_dir, fresh=args.fresh,
        protect_first=args.protect_first, protect_last=args.protect_last,
    )

    print(f"\n--- Quantized (GPTQ) perplexity, same {args.max_tokens} tokens ---")
    quant_ppl = compute_perplexity(model, tokenizer, text, args.seq_len, args.max_tokens)
    print(f"  Perplexity: {quant_ppl['perplexity']:.3f} "
          f"({quant_ppl['eval_wall_seconds']:.1f}s)")

    degradation_pct = (quant_ppl["perplexity"] - baseline_ppl["perplexity"]) / baseline_ppl["perplexity"] * 100

    print("\n" + "-" * 80)
    print("Summary:")
    print(f"  Perplexity: {baseline_ppl['perplexity']:.3f} ({args.dtype} baseline) -> "
          f"{quant_ppl['perplexity']:.3f} (GPTQ ternary)  [{degradation_pct:+.2f}%]")
    print(f"  For comparison (fp32 baseline 12.780, NOT directly comparable to the "
          f"{args.dtype} baseline above -- different precision):")
    print(f"    naive per-tensor:  12.780 -> 89,100.682  (+697,074%)")
    print(f"    naive per-channel: 12.780 -> 132,590.254 (+1,037,361%)")
    print(f"  Success criterion (<5% accuracy loss): "
          f"{'PASS' if abs(degradation_pct) < 5.0 else 'FAIL'}")
    if quant_stats.get("protected_blocks"):
        print(f"  Mixed precision: blocks {quant_stats['protected_blocks']} kept at "
              f"{args.dtype} (not quantized); {quant_stats['n_layers_quantized']} "
              f"of {sum(1 for n, m in model.named_modules() if isinstance(m, nn.Linear) and 'lm_head' not in n)} "
              f"total layers were ternarized")
    if args.layers is not None:
        print(f"\n  NOTE: only {args.layers} of "
              f"{sum(1 for n, m in model.named_modules() if isinstance(m, nn.Linear) and 'lm_head' not in n)} "
              f"layers were quantized (--layers flag) -- most of the model is "
              f"still at {args.dtype} precision, unquantized. This perplexity "
              f"number is NOT representative of the fully-quantized model; it "
              f"validates timing/correctness only.")

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "dtype": args.dtype,
            "dtype_note": "Switched from quantize_tinyllama.py's fp32 default "
                           "2026-08-25 due to real swap thrashing on this "
                           "machine (confirmed via /proc/<pid>/status VmSwap) "
                           "when a resident fp32 model shared 7GB total RAM "
                           "with a live desktop session. This baseline is NOT "
                           "directly comparable to the fp32 baseline (12.780) "
                           "quantize_tinyllama.py established.",
            "method": "GPTQ-style sequential column quantization with "
                      "Hessian-based error compensation (Frantar et al.), "
                      "ternary {-1,0,+1} quantizer, per-tensor scale",
            "layers_limit": args.layers,
            "protect_first": args.protect_first,
            "protect_last": args.protect_last,
            "device": str(device),
            "device_name": (torch.cuda.get_device_properties(0).name
                            if device.type == "cuda" else None),
            "calib_window": [calib_start, calib_end],
            "eval_window": [0, args.max_tokens],
        },
        "baseline": baseline_ppl,
        "quantized_gptq": quant_ppl,
        "perplexity_degradation_pct": degradation_pct,
        "quantization": quant_stats,
        "success_criteria": {
            "accuracy_loss_under_5pct": abs(degradation_pct) < 5.0,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"tinyllama_gptq_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
