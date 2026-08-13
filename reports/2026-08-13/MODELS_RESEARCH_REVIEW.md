# Code Review: models/ and research/ — 2026-08-13

**Scope:** User-requested follow-up ("review research/ and models/ for more
bugs") after the [2026-08-13 code review continuation](CODE_REVIEW_SESSION_REPORT.md)
closed out `research/scripts/falsify.py` and the rest of the previously-pending
scope. `research/` has no further Python beyond `falsify.py` (already reviewed) and
`research/configs/schema.yaml` (see below). `models/tritnet/` was already reviewed
2026-08-12. `models/bitnet/` is an empty placeholder directory with no files. This
pass covers the two remaining substantial code directories:
`models/3-vae-gemm-v1/` (~3,500 lines, 8 files) and `models/company-flagships/`
(~4,800 lines, 9 files).

**Method:** Background `code-review` subagents (one per directory, 8 parallel
"finder angle" passes each) plus manual verification. Every finding below was
reproduced or directly executed against real code before being called a bug or
fixed — including instantiating actual PyTorch modules and running real forward
passes, not just reading source and inferring behavior.

**Net result:** 3 commits, `e7572e7` and `d2c4396` (company-flagships) and
`c4426c1` (3-vae-gemm-v1), all on `main`, all pushed. `tests/run_tests.py` (13/13)
still passes — none of these files are wired into that suite, so each fix was
verified individually via direct execution/`py_compile`.

---

## The headline finding: the inverted-valuation bug, ten times over

`research/scripts/falsify.py`'s `build_corpus()` had a bug (fixed earlier the same
day, see the sibling report) where 3-adic valuations were computed on the *raw*
corpus encoding index instead of the *decoded* balanced-ternary value, inverting
which point looked "near zero." The same bug — same root cause, same fix — turned
up **independently reimplemented 9 more times** across `models/`: 4 in
`models/3-vae-gemm-v1/` and 5 in `models/company-flagships/` (the second batch
found during a systematic re-check of this project's other known
valuation-function duplicates, after the first 4 made the pattern obvious enough
to search for deliberately).

**`models/3-vae-gemm-v1/`** (commit `c4426c1`):
- `data.py` — `TripletDataset._valuation()` / `__init__`
- `model.py` — `VAEGemmV1._build_valuation_table()`
- `hyperbolic_ops.py` — `UltrametricAttractorField._init_attractors()` (this one
  also seeds the initial position of all 19,683 learnable attractor parameters)
- `hyperbolic_ops.py` — `HyperbolicOperationModel._build_valuation_table()`

**`models/company-flagships/`** (commit `d2c4396`):
- `validate_checkpoints.py` — the function that computes `hierarchy_A`/
  `hierarchy_B` = `spearmanr(valuations, radii)`, i.e. **the VRC metric CLAUDE.md
  documents as "-0.83" for the homeostasis checkpoint.** This is the single most
  consequential instance found in this whole pass — see caveat below.
- `create_embedding_lut.py` — the embedding-generation loop's own valuation calc
- `explore_gemm_space.py` — `SoftGEMMExplorer.__init__`
- `embedding_exactitude_score.py` (2 call sites) — `compute_valuation_metrics()`
  (its own VRC-style `radius_correlation`) and `compute_hierarchy_metrics()`'s
  `sample_vals` (the same function whose dead `dendrogram_correlation` branch was
  fixed earlier the same session — that fix vectorized the computation faithfully,
  but the *input* it was fed was still wrong until this second pass)
- `explore_gemm_extended.py` — `ExtendedGEMMExplorer.__init__`

All ten compute `v3(idx)` directly on the raw encoding index (`idx =
sum((trit_k+1)*3^k)`, range `[0, num_values-1]`), when the true zero (all-zero
trits) actually lives at `idx_offset = (num_values-1)//2 = 9841`, not at `idx=0`
(which decodes to the *most negative* representable value). The fix everywhere:
compute valuation on `idx - idx_offset` instead.

Verified end-to-end by actually instantiating the real classes with real objects
(not just reading the code) for every one of the ten:

```
TripletDataset.valuations[9841]  (true zero):        0 -> 9   (fixed)
VAEGemmV1.valuations[9841]:                            0 -> 9   (fixed)
UltrametricAttractorField radius at idx=9841:  ~0.9 -> ~0.1  (center, correct)
UltrametricAttractorField radius at idx=0:     ~0.1 -> ~0.9  (boundary, correct)
HyperbolicOperationModel.valuations[9841]:             0 -> 9   (fixed)
SoftGEMMExplorer.valuations[9841]:                     0 -> 10  (fixed)
ExtendedGEMMExplorer.valuations[9841]:                 0 -> 9   (fixed)
compute_valuation_metrics() / compute_valuation() / create_embedding_lut.py's
  loop: all verified standalone the same way (9841 -> max valuation,
  0 -> valuation 0, previously swapped)
```

### Caveat: the documented VRC = -0.83 claim needs re-verification

`models/company-flagships/validate_checkpoints.py` is the tool that computes
`hierarchy_A`/`hierarchy_B`, the exact metric name and definition
(`Spearman(valuation, radius)`) behind CLAUDE.md's documented claim for the
**homeostasis** checkpoint: "VRC target: -0.83, coverage: 100%". This script had
the inverted-valuation bug. The training script that originally produced
`models/company-flagships/v5_11_homeostasis/best.pt` is not present in this repo,
so it's not possible to determine from here whether the *original* -0.83 figure
was itself produced by this buggy validator, by a different (possibly correct)
script, or computed some other way. What can be said: if anyone re-runs
`validate_checkpoints.py` against that checkpoint to reproduce or re-check the
claim, the pre-fix version would have reproduced an inverted VRC value, not the
correct one. Per this project's honest-claims policy, the -0.83 figure should be
treated as **unverified pending a re-run against the fixed script** before being
cited further — the same posture already applied to the `bench_competitive.py`
mock-fallback bug from the 2026-08-12 session.

The `UltrametricAttractorField` check is the most concrete: before the fix, the
true-zero attractor sat near the *boundary* of the Poincaré ball and the
most-negative-value attractor sat near the *center* — backwards from the intended
p-adic structure the whole class exists to encode.

**Caveat on existing checkpoints:** any checkpoint already trained with this code
— per CLAUDE.md, `checkpoints_hyperbolic/best_model.pt` (itself documented as only
1 epoch, 0.11% val accuracy, i.e. barely trained) — was trained against this
inverted structure. Given how early-stage that checkpoint already is, practical
impact is likely small, but any future retraining should use the fixed code, and
that checkpoint's existing VRC/attractor numbers should not be cited as evidence of
correct p-adic structure without re-validation after retraining.

---

## Other fixes, by file

### `models/company-flagships/` (commit `e7572e7`)

- **`validate_checkpoints.py`** — `AlgebraicMetrics.tadd_associativity`/
  `tmul_associativity` were declared and documented but never computed;
  `test_algebraic_properties()`'s sampling loop only ever drew a second operand
  (for commutativity), never a third. Both metrics silently stayed at the
  dataclass default of `0.0` for every checkpoint ever validated — indistinguishable
  from "genuinely never associative." Implemented by sampling a third operand.
  Verified standalone that this file's own `ternary_add`/`ternary_mul` (carry-based
  addition, element-wise multiplication — genuinely different operations from the
  engine's saturating `tadd`) are in fact associative (both metrics correctly come
  out ~1.0), so this now gives a real, useful learned-model-fidelity signal instead
  of a placeholder zero.

- **`embedding_exactitude_score.py`** (×2) **+ `explore_gemm_extended.py`** (×1) —
  `np.random.seed(seed + hash(op_name) % N)` relied on Python's builtin `hash()`,
  which is randomized per-process (`PYTHONHASHSEED`) unless disabled. Verified
  `hash('add') % 10000` returns a different value on every fresh interpreter
  invocation. Replaced with a `zlib.crc32`-based `stable_str_hash()` helper in both
  files; verified identical output across 3 separate process runs after the fix.

- **`embedding_exactitude_score.py`** — `compute_hierarchy_metrics()`'s
  `dendrogram_correlation` was gated behind `if sample_size <= 500:`, but its only
  caller always passes `min(1000, N)` — for any real corpus this guard was always
  `False`, so the metric silently stayed `0.0` for every real run. Vectorized the
  O(n²) Python loop behind the guard with `scipy.spatial.distance.pdist` (verified
  byte-for-byte identical output to the old loop first) and removed the now-dead
  guard. Verified end-to-end: now returns a real nonzero correlation.

- **`explore_gemm_space.py`** — `generate_soft_gemm_map()`'s discrete-operation
  branch only handled `'add'`/`'mul'`; any other operation (e.g. `'min'`/`'max'`,
  already implemented identically in `operation_neighborhood()` in the same class)
  silently fell through to `result_trits = a_trits` (identity) instead of raising.
  Added `min`/`max` support and now raises `ValueError` for genuinely unrecognized
  operations. Verified all four ops resolve to distinct correct results.

- **`export_vae_to_gemm.py`** — the `fc_mu`/`fc_logvar` export loop indexed
  `state_dict` directly with no existence guard, unlike the encoder-layer loop
  immediately above it. Confirmed reachable: this directory's own
  `create_embedding_lut.py` defines an `Encoder` with only `fc_mu`, no `fc_logvar`.
  Added the same guard used one loop above.

- **`explore_gemm_space.py`** — a per-operation `save_results` dict (with
  `soft_gap_stats`: mean/std/min/max) was built every loop iteration and discarded
  — the actual `report` written to disk is built separately with less detail
  (mean/max only). Now merged into the saved report instead of thrown away.

- **`test_extended_exploration.py`** — zero assert statements; unconditionally
  printed "ALL PIPELINE TESTS PASSED" regardless of the actual values returned.
  Added range/finiteness sanity checks after each pipeline stage and a proper
  `main() -> int` / `sys.exit(main())` pattern.

- **`create_embedding_lut.py`, `export_vae_to_gemm.py`, `test_vae_gemm.py`**
  (identically, ×3) — hardcoded checkpoint path `v5_11_homeostasis/epoch_20.pt`,
  but CLAUDE.md documents this checkpoint's path as `v5_11_homeostasis/best.pt`.
  Fixed all three to prefer `best.pt` and fall back to `epoch_20.pt`.

- **`explore_gemm_extended.py`** — a comment claimed "First trit multiplication
  only (simplified)" on code that is a full element-wise product across all 9
  trits. Corrected the comment; code was already correct.

- **`explore_gemm_extended.py`** — `_get_merge_height()` reimplemented cophenetic
  distance with an O(n) dict scan per merge step per pair (~O(n⁴) total for
  `sample_size=500`, ~124,750 pairs), when `scipy.cluster.hierarchy.cophenet()`
  computes the identical quantity in one vectorized call — already used for this
  exact purpose in the sibling file `explore_gemm_space.py`. Verified `cophenet()`
  produces byte-for-byte identical output to the old loop on synthetic data before
  replacing it and deleting the now-dead function.

### `models/3-vae-gemm-v1/` (commit `c4426c1`)

- **`hyperbolic_ops.py`** — `HyperbolicOperationLoss.radial_alignment_loss()`
  divided by `valuations.max()` with no epsilon guard, unlike every other division
  in the file (all clamp via `.clamp(min=1e-5)`). Reproduced the exact failure: a
  mini-batch where every sampled valuation is 0 (the majority class, ~2/3 of all
  19,683 values) makes `max_val=0`, producing `0/0 = NaN` that would poison the
  optimizer step. Fixed with the same clamp pattern used elsewhere in the file;
  verified the reproduction scenario now returns a finite loss.

- **`train.py`** — `Trainer.unfreeze_all()` had no way to invalidate
  `self.frozen_embeddings` (cached once, used as `ORALoss`'s ranking negatives).
  `train_epoch()`'s only recompute trigger was `if self.frozen_embeddings is None`,
  which never fires again once set — so from the freeze/unfreeze boundary onward,
  every epoch ranked against embeddings frozen at that single moment while the
  (now-training) encoder's live embeddings drifted further away every epoch. Added
  an `encoder_frozen` flag and changed the guard to recompute every epoch whenever
  the encoder isn't actually frozen. Verified end-to-end with a real
  `Trainer`+`VAEGemmV1`: embeddings now visibly change after
  unfreeze+perturb+recompute, where they previously would have stayed frozen at the
  pre-unfreeze snapshot.

- **`test_hyperbolic.py`** — rewritten. Previously zero assert statements, wrapped
  entirely in a bare `try/except` that only printed a traceback and always exited
  0. Added structural assertions (finite values, correct shapes — deliberately not
  asserting specific outcome values, since the model under test is untrained) plus
  one deterministic, training-independent assertion that `v3(3^k) = k` exactly for
  the valuation table. Also fixed the script's own `sample_indices`, which had been
  powers of 3 used directly as raw indices (the same misconception behind the
  valuation bug above) — now correctly converts decoded powers of 3 to raw indices
  via `+ idx_offset`. Added the missing copyright header, `__main__` guard, and a
  fixed random seed. **Caught a bug in my own first fix attempt**: the first
  version of the deterministic assertion had an off-by-one (`expected =
  [0,1,...,9]` instead of `[9,0,1,...,8]`, forgetting that `v3(0) = num_trits` by
  this codebase's own convention) — running the test immediately surfaced this,
  which was then corrected and re-verified. Final version runs clean, exit code 0.

---

## Documented but not fixed

Design/architecture-level findings that need a judgment call beyond a mechanical
bug fix, or lower-confidence findings — matching this project's established policy
(see CLAUDE.md gaps #7/#8, and the falsify.py duplication findings from earlier the
same day) of documenting rather than unilaterally redesigning training pipelines:

- **`loss.py`'s `EESLoss` reconstruction/KL branches are permanently dead** —
  `train.py` never calls the model's VAE forward path (only `forward_operation()`),
  so the `Decoder` never trains despite being fully wired with configured loss
  weights (`recon_weight`, `kl_weight` in `VAEGemmV1Config`).
- **`__init__.py`'s exported public API is the discredited Euclidean model** —
  `VAEGemmV1`/`OperationHead`/`UltrametricLoss` compute operand midpoints via plain
  Euclidean average (`(emb_a + emb_b) / 2`) and distances via plain L2 norm — the
  exact approach CLAUDE.md's own Hyperbolic GEMM Research section documents as
  falsified and non-equidistant. The correct geodesic-midpoint implementation
  (`hyperbolic_ops.py`/`train_hyperbolic.py`) exists in the same directory but
  isn't exported from the package's front door.
- **`--resume` documented but doesn't exist** — CLAUDE.md documents `python
  train_hyperbolic.py --resume` as a supported workflow; no such flag exists in
  `train_hyperbolic.py`'s argparse, and `train.py`'s `Trainer` has no load/resume
  counterpart to its `save_checkpoint()`.
- **`predicted_idx`/`predicted_emb` mismatch** — `HyperbolicOperationFlow.forward`'s
  `predicted_idx` is derived from 3 additional geodesic-flow steps beyond
  `predicted_emb` (the value `trajectory_loss` actually optimizes), so accuracy
  metrics computed from `predicted_idx` don't necessarily reflect what the loss
  supervises.
- **`attractor_ultrametric_loss` is O(n³) unbatched** — a triple-nested pure Python
  loop (up to ~161,700 iterations) instead of the vectorized
  `_batch_hyperbolic_distance` already defined in the same file. Likely the
  dominant cost behind the ~89-min/epoch training time CLAUDE.md documents for this
  model.
- **Three independently-drifting VRC/radial-target formulas** — `loss.py:VRCLoss`,
  `hyperbolic_ops.py:radial_alignment_loss`, and
  `hyperbolic_ops.py:UltrametricAttractorField._init_attractors` each hardcode a
  different constant for what CLAUDE.md documents as one shared metric.
- **`research/configs/schema.yaml`** references the archived
  `models.gemm_discovery.ebm.ultrametric_energy` module and a hardcoded
  Windows-only `.pyd` path. Confirmed harmless: `falsify.py`'s `self.config`
  (loaded from this file) is never actually read anywhere after being assigned in
  `__init__` — the file is loaded but its content has zero behavioral effect.

---

## Verification discipline used throughout

Same standard as every prior session in this review effort:
- Reproduced the original bug first (crash, wrong value, dead branch), then
  confirmed the fix resolves that exact reproduction.
- For the valuation bug: verified not just numerically but geometrically —
  actually instantiated `UltrametricAttractorField` and checked the resulting
  attractor *radii* land where the p-adic structure says they should.
- For behavioral changes (frozen-embeddings staleness, associativity metrics):
  instantiated the real `Trainer`/`VAEGemmV1`/model classes and ran the real
  methods, not synthetic stand-ins.
- For the vectorization fixes (`dendrogram_correlation`, `_get_merge_height`):
  diffed old-vs-new output on synthetic data to confirm the rewrite was
  behavior-preserving before deleting the old code.
- For `test_hyperbolic.py`'s rewrite: actually ran the finished script end-to-end
  against the real, already-fixed model code — which caught an off-by-one in the
  first draft of my own new assertion, corrected in the same pass.
- For the valuation bug specifically: after finding and fixing all 10 known
  instances, ran a final `grep -rn "while n % 3 == 0"` across all of `models/` and
  `research/` (not just the two directories under active review) to confirm no
  further instances existed, including a specific check of `models/tritnet/`
  (already reviewed 2026-08-12, confirmed no valuation-related code exists there
  at all) and every remaining file in `models/company-flagships/` that wasn't
  otherwise touched this session.
