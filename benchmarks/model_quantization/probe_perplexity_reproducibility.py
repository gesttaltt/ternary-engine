"""
probe_perplexity_reproducibility.py - why a recorded baseline perplexity
does or does not reproduce

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Built 2026-08-29 to settle a concrete discrepancy, and kept because the
discrepancy is the kind that recurs: CLAUDE.md v1.56.0 and
docs/planning/ROADMAP.md recorded an fp16 baseline perplexity of 7.172 for
`quantize_tinyllama_gptq.py`, but re-running that script with its documented
flags yields 12.78. Since every GPTQ degradation percentage in those
documents is computed against the baseline, a wrong baseline silently
invalidates the whole table -- so "which number is right, and why do they
differ" had to be answered rather than assumed.

This script isolates the three explanations worth testing before blaming
the environment, which is the easy and usually wrong answer:

  MODE `grid`      Is it an eval-window difference? Perplexity from this
                   formulation is strongly dependent on `seq_len` (longer
                   blocks give more context per prediction, so the number
                   drops) and mildly on which slice of corpus `max_tokens`
                   happens to cover. Sweeps a (seq_len x max_tokens) grid so
                   two runs can be compared at matched settings instead of
                   accidentally at different ones.

  MODE `hunt`      Is there ANY window setting reproducing a given target?
                   Scans a dense grid and reports the closest matches. A
                   target that no configuration comes near is not a windowing
                   difference.

  MODE `version`   Is it a library difference? Prints the transformers
                   version, which kwarg spelling `from_pretrained` accepted,
                   and -- importantly -- the dtype the weights ACTUALLY
                   landed in. transformers v5 renamed `torch_dtype=` to
                   `dtype=`; a script written for one and run on the other
                   either raises, or (worse, in versions that swallow unknown
                   kwargs) silently loads the model at config.json's dtype
                   while the script's own banner still prints the dtype the
                   user asked for.

To compare library versions WITHOUT disturbing the machine's environment,
install the other version into a throwaway directory and shadow it:

    pip install --target /tmp/tf4 "transformers==4.46.3"
    PYTHONPATH=/tmp/tf4 python3 benchmarks/model_quantization/probe_perplexity_reproducibility.py --mode version

Findings from the 2026-08-29 run (see
reports/2026-08-29/PERPLEXITY_BASELINE_REPRODUCIBILITY.md): the window
hypothesis and the version hypothesis are both excluded, and the 7.172
figure does not reproduce from the committed code under any of the 54
configurations tried.

USAGE: python benchmarks/model_quantization/probe_perplexity_reproducibility.py --mode grid
       python benchmarks/model_quantization/probe_perplexity_reproducibility.py --mode hunt --target 7.172
       python benchmarks/model_quantization/probe_perplexity_reproducibility.py --mode version
OUTPUT: a table to stdout; nothing is written to disk (this is a diagnostic,
        not a benchmark whose numbers belong in benchmarks/results/)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

# Set before torch initializes its allocator. This card is routinely shared
# with a desktop session and games (a real run of this script OOMed against
# a game holding 1.9GB of a 6GB card), and the reserved-but-unallocated
# fragmentation this avoids is exactly what tips a marginal run over.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from quantize_tinyllama_gptq import (  # noqa: E402
    MODEL_NAME,
    compute_perplexity,
    load_wikitext2_test,
)

GRID_SEQ_LENS = (512, 1024, 2048)
GRID_MAX_TOKENS = (2048, 4096, 8192, 16384, 32768, 131072)
HUNT_SEQ_LENS = (128, 256, 512, 768, 1024, 1536, 2048)
HUNT_MAX_TOKENS = (1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384)


def load_model(device: torch.device, dtype: torch.dtype = torch.float16) -> Tuple:
    """Loads the model, tolerating either transformers kwarg spelling.

    Returns (model, tokenizer, kwarg_used, actual_weight_dtype). The last two
    are returned rather than discarded precisely because a silently-ignored
    dtype kwarg is one of the failure modes this script exists to detect.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)
        kwarg_used = "dtype (transformers v5 spelling)"
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
        kwarg_used = "torch_dtype (transformers v4 spelling)"
    model = model.eval().to(device)
    return model, tokenizer, kwarg_used, next(model.parameters()).dtype


def print_environment(kwarg_used: str, actual_dtype: torch.dtype) -> None:
    import transformers

    print(f"transformers      : {transformers.__version__}")
    print(f"torch             : {torch.__version__}")
    print(f"load kwarg used   : {kwarg_used}")
    print(f"ACTUAL weight dtype: {actual_dtype}")
    if actual_dtype != torch.float16:
        print("  [WARN] requested float16 but the weights are NOT float16 -- the "
              "dtype kwarg was accepted and then ignored. Any perplexity measured "
              "below is not an fp16 number regardless of what was asked for.")


def measure(model, tokenizer, text: str, seq_len: int,
            max_tokens: int) -> Optional[dict]:
    """One (seq_len, max_tokens) measurement, tolerant of a busy GPU.

    Returns None if the configuration could not be measured because the GPU
    was out of memory. This card is shared with whatever else the machine is
    doing, so one oversized configuration must not abort a 54-point scan --
    a partial table with the failures named is far more useful than a
    traceback, and lets the conclusion be stated over the cells that DID
    run.
    """
    for attempt in (1, 2):
        try:
            return compute_perplexity(model, tokenizer, text, seq_len, max_tokens)
        except torch.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if attempt == 2:
                return None
    return None


def run_grid(model, tokenizer, text: str) -> None:
    print(f"\n{'seq_len':>8}{'max_tokens':>12}{'n_blocks':>10}"
          f"{'perplexity':>14}{'mean_nll':>12}")
    for seq_len in GRID_SEQ_LENS:
        for max_tokens in GRID_MAX_TOKENS:
            if max_tokens < seq_len:
                continue
            r = measure(model, tokenizer, text, seq_len, max_tokens)
            if r is None:
                print(f"{seq_len:>8}{max_tokens:>12}{'--':>10}"
                      f"{'OOM (skipped)':>14}{'--':>12}")
                continue
            print(f"{seq_len:>8}{max_tokens:>12}{r['n_blocks']:>10}"
                  f"{r['perplexity']:>14.3f}{r['mean_nll']:>12.4f}")
    print("\nNote: perplexity drops as seq_len rises (more context per "
          "prediction). Two runs of this project's quantization scripts are "
          "only comparable at identical seq_len AND max_tokens.")


def run_hunt(model, tokenizer, text: str, target: float, tolerance: float) -> int:
    """Returns 0 if some configuration reproduces `target`, 1 otherwise."""
    results = []
    skipped = 0
    for seq_len in HUNT_SEQ_LENS:
        for max_tokens in HUNT_MAX_TOKENS:
            if max_tokens < seq_len:
                continue
            r = measure(model, tokenizer, text, seq_len, max_tokens)
            if r is None:
                skipped += 1
                continue
            results.append((abs(r["perplexity"] - target), seq_len, max_tokens,
                            r["perplexity"]))
    if not results:
        print("\nEvery configuration OOMed -- no conclusion can be drawn. "
              "Free GPU memory (or use --device cpu) and re-run.")
        return 2
    results.sort()
    if skipped:
        print(f"\n[WARN] {skipped} configuration(s) skipped for GPU OOM; the "
              f"verdict below covers only the {len(results)} that ran.")
    print(f"\nTarget = {target}. Closest 10 of {len(results)} configurations:")
    print(f"{'seq_len':>8}{'max_tokens':>12}{'perplexity':>14}{'|diff|':>10}")
    for diff, seq_len, max_tokens, ppl in results[:10]:
        print(f"{seq_len:>8}{max_tokens:>12}{ppl:>14.3f}{diff:>10.3f}")
    if results[0][0] < tolerance:
        print(f"\nREPRODUCED: seq_len={results[0][1]}, "
              f"max_tokens={results[0][2]} gives {results[0][3]:.3f} "
              f"(within {tolerance}).")
        return 0
    print(f"\nNOT REPRODUCED: closest configuration is {results[0][0]:.3f} away "
          f"(tolerance {tolerance}). The target is not explained by the eval "
          f"window.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["grid", "hunt", "version"], default="grid")
    parser.add_argument("--target", type=float, default=7.172,
                        help="hunt mode: the recorded perplexity to search for "
                             "(default 7.172, the figure recorded in CLAUDE.md "
                             "v1.56.0 that prompted this script)")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="hunt mode: how close counts as reproduced")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable.")
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu"))

    model, tokenizer, kwarg_used, actual_dtype = load_model(device)
    print_environment(kwarg_used, actual_dtype)
    print(f"device            : {device}")
    if args.mode == "version":
        return 0

    text = load_wikitext2_test()
    if args.mode == "grid":
        run_grid(model, tokenizer, text)
        return 0
    return run_hunt(model, tokenizer, text, args.target, args.tolerance)


if __name__ == "__main__":
    sys.exit(main())
