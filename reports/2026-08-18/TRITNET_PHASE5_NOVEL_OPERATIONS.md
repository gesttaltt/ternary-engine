# TritNet Phase 5: "discover novel ternary operations"

**Date:** 2026-08-18
**Platform:** Linux x64, AMD Ryzen 5 7520U (CPU-only training, same as Phase 2A/2B)
**Script:** `models/tritnet/phase5_novel_operations.py`
**Closes:** the last unstarted bullet of TritNet Phase 5 (Learned Generalization)

## The question

Phase 4 and Phase 5's first two results (error characterization, GPU application context) both
closed off "TritNet as a faster or more capable *replacement* for tadd/tmul/tmin/tmax/tnot" --
direct CPU LUTs and direct GPU arithmetic beat it on both throughput and exactness. This project's
own stated conclusion: *"If TritNet has a real niche, it's not as a replacement for exact
per-trit-chunk arithmetic -- it would need an operation without a cheap closed form."* This script
asks that question directly.

## Method (four falsifiable stages)

1. **Enumerate** the full space of single-trit binary ternary operations: {-1,0,+1} x {-1,0,+1} ->
   {-1,0,+1}, all 3^9 = 19,683 possible truth tables.
2. **Deduplicate by symmetry**: an operation and its "obvious variants" (swap the arguments,
   negate either input, negate the output, or any composition -- a 16-element group) are the same
   underlying idea, not independent discoveries. Reduces 19,683 raw truth tables to **1,444
   equivalence classes**, and excludes every class containing tadd, tmul, tmin, or tmax.
3. **Filter "cheap closed form" two independent ways**, since neither alone is a complete proxy:
   - *Curated catalog*: does the operation match (up to the same symmetry) a short composition of
     standard ternary primitives (saturating add/sub, multiply, min, max, comparison, negation)?
     The "would anyone just write this by hand" check.
   - *GF(3) algebraic degree*: every function on a finite field has a unique polynomial
     representation (via exact Gaussian elimination over GF(3), not floating-point); its degree is
     a rigorous, computable complexity measure independent of any curated list. Verified against
     known cases first: `tmul` (literally the monomial a*b) comes out degree 2 exactly; a constant
     function comes out degree 0; a single-variable projection comes out degree 1 -- all correct.
     Important asymmetry, checked explicitly rather than assumed: `tmin`/`tmax` are algebraically
     **high**-degree (4, the maximum possible) despite being cheap in hardware (comparison-based,
     not polynomial) -- degree alone is not sufficient, which is exactly why the curated-catalog
     check exists too. All 4 named ops are correctly caught by at least one of the two filters.
4. **Score survivors** with this project's own established ternary-native metrics (sparsity,
   associativity, commutativity, distributivity over tadd -- computed exhaustively over all 27
   input triples, not sampled) and **train** the most interesting one with the exact TritNet
   architecture/hyperparameters that produced the documented tadd/tmul/tmin/tmax checkpoints.

## Discovery result

- **1,365 of 1,444 classes (94.5%)** survive both cheap-closed-form filters. Not surprising on its
  own -- it's the well-established general pattern that most functions of a finite domain lack a
  short algebraic description (the same reason most Boolean functions need large circuits) -- but
  worth confirming rather than assuming for this specific domain, and it establishes the baseline
  the rest of the analysis is measured against.
- **28 of those 1,365 (2.0%) are fully associative** -- genuinely rare, and a meaningfully
  distinguishing property: `tadd` itself is *non*-associative for 79.6% of triplets (H24,
  `research/scripts/falsify.py`), so full associativity in an unnamed, closed-form-resistant
  candidate is a real find, not a given.
- **17 of those 28 are non-degenerate** (use all 3 trit values as outputs, not secretly a 2-valued
  Boolean function wearing a ternary domain). This required a correction mid-session: an early
  exploratory check wrongly concluded *no* 3-valued fully-associative survivor existed, based on
  sorting the wrong intermediate list; a careful from-scratch recount (implemented as the script's
  actual ranking logic, not left as an ad-hoc finding) found 17. Recorded here as a reminder that
  the "verify before claiming" discipline this whole session has been built on applies to a
  research script's own exploratory output too, not just to code review findings.

The top-ranked candidate (associativity first, then non-degeneracy, then output balance, then
GF(3) degree): truth table `(-1,-1,-1,-1,0,0,-1,1,1)` in the order
`((-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1))`. GF(3) degree 3, sparsity 0.222,
commutativity 0.778 (not fully commutative -- not required for the discovery, and notable that a
fully-associative non-commutative ternary operation is itself an uncommon combination).

## Training result

Same two-phase QAT recipe (float warm-start -> ternary quantization-aware training),
same architecture (hidden=128, threshold=0.3), same 59,049-sample full-input-space dataset
generation (the discovered scalar op applied elementwise across 5-trit chunks, identical
convention to `tadd`/`tmul`/`tmin`/`tmax`) as `train_phase2b.py`'s documented checkpoints.

| | Phase 1 (float) | Phase 2 (QAT) |
|---|---|---|
| Result | 100.00% in 212 epochs (61.3s) | best 99.52% at epoch 7258 (2956.2s) |

**99.52% -- passes this project's own >=99% GO threshold**, landing in the same range as the
existing imperfect checkpoints (tmul 99.49%, tmin 99.89%, tmax 99.85%). Weight distribution after
training: -20.2% / 0=41.4% / +38.4% -- consistent with the ~40%-zero sparsity pattern H14 already
documented for the 4 named ops, not an outlier.

**TritNet can learn a genuinely novel, closed-form-resistant, non-degenerate, fully-associative
ternary operation it was never designed around, to accuracy comparable with the hand-picked named
operations.** That's a real, positive, falsifiable result -- the discovery methodology and the
training both succeeded on their own terms.

## The honest caveat this result does NOT establish a commercial niche

This is the part that matters most, and it's a direct consequence of a fact this analysis surfaced
by trying to be precise about "closed form": **"no cheap arithmetic formula" is not the same as
"no cheap implementation."** Any function on a bounded, small-arity domain -- which every operation
in this analysis is, by construction (9-entry truth tables, 5-trit chunks, 3^10 = 59,049-entry
tables at the granularity TritNet actually trains on) -- trivially admits a lookup table. A LUT for
this discovered operation would be no larger or more expensive to build than the LUTs `tadd`/
`tmul`/`tmin`/`tmax` already ship with, and per Phase 3's own measurement
(`reports/2026-08-14/TRITNET_PHASE3_SESSION_REPORT.md`), a LUT beats even the AVX2-accelerated
TritNet inference path by **169x-195x**, on any of those operations. Nothing about *this*
operation's algebraic irreducibility changes that arithmetic -- a LUT doesn't care whether the
function it encodes has a closed form or not, only how large its domain is, and this operation's
domain is exactly as small as the other four.

So the roadmap's original framing -- "a real niche would need an operation without a cheap closed
form" -- turns out to have been imprecise in a way this analysis makes concrete: the relevant
"cheap alternative" competing with TritNet was never really the *arithmetic formula*, it was always
the *LUT*, and a LUT is available for literally any function definable on this project's actual
truth-table domains. An operation that would genuinely resist a LUT would need an unbounded or
much larger input domain than the 5-trit-chunk convention this entire codebase is built around --
a different, larger research question than "discover a novel 5-trit-chunk operation," and outside
this bullet's original scope.

## Verdict on Phase 5's "discover novel operations" bullet

**Answered, not open-ended.** A genuinely novel operation was found (not equivalent to any named
op, no cheap arithmetic formula by two independent rigorous checks, fully associative unlike
`tadd`), and TritNet learned it to 99.52% -- matching this project's own GO criterion. But because
a LUT is always available for any operation in this domain regardless of algebraic complexity, this
success doesn't open a commercial niche; it reinforces Phase 3/4's conclusion from a different
angle instead of contradicting it. Combined with Phase 5's first two results (structured-not-noise
errors; no GPU-application niche either), **all three Phase 5 bullets are now closed**, and the
honest overall TritNet verdict across Phases 3-5 stands: LUT wins by two orders of magnitude
regardless of operation choice, hardware target, or how the operation was discovered.
