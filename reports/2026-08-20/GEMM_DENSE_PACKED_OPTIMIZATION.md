# Dense-Packed Ternary GEMM: Investigating and Fixing the 0.189× Matmul Gap — 2026-08-20

**Scope:** User-directed follow-up to the project roadmap review (v1.47.0
README.md pass). Asked to recommend and pursue the next highest-value item;
picked matmul/GEMM optimization as the clearest remaining commercial-viability
gap (Critical Gap #3: "0.189× avg matmul... too slow for AI"). Linux x64,
this session's sandboxed dev container (no GPU, no AVX-512, no ARM — same
constraints as every prior session).

## 1. The existing kernel was not naive

Before touching anything, read `src/core/simd/ternary_gemm_zero_skip.cpp`
in full. It already has AVX2 FMA vectorization, OpenMP (two variants — a
j-parallel CSC kernel and a k-parallel CSR "tiled" kernel with a documented
cache-fit rationale), and a CSC/CSR sparse index that skips zero weights.
This is not a "missing basic optimizations" bug — the 0.189× figure needed
a real diagnosis, not a reflexive rewrite.

## 2. Diagnosis

### 2a. `bench_competitive.py`'s Phase 4 tests `batch=1`

Both zero-skip kernels vectorize their inner SAXPY loop over the **batch**
dimension (`M`, 8-wide AVX2 FMA). Phase 4 explicitly tests `batch=1`
("single-token inference, matches this phase's original intent"). At
`M=1`, the AVX2 loop condition `m + 8 <= M` is never true — every call
falls entirely to the scalar tail. Verified directly:

```
batch=   1  ternary=  6.8357ms  numpy=  0.9147ms  speedup=0.134x
batch=   8  ternary=  4.3448ms  numpy=  2.4055ms  speedup=0.554x
batch=  32  ternary=  6.0676ms  numpy=  3.2410ms  speedup=0.534x
```

Speedup roughly quadruples going from batch=1 to batch=8-32 — confirms
the SIMD-starvation diagnosis. (It gets *worse* again at batch=512 for the
tiled kernel specifically — see 2c.)

### 2b. CSC/CSR index storage is larger than the dense array it indexes

At ternary's actual non-zero density (~60-67%, i.e. ~33-40% zero — not a
"real" sparsity regime), a CSC/CSR entry costs 5 bytes (4-byte index +
1-byte sign) per non-zero. Dense int8 storage costs 1 byte per weight,
zeros included. Measured directly (native benchmark, see §4):

| Shape | CSC index bytes | Dense packed bytes | Ratio |
|---|---|---|---|
| Small MLP (512×512) | 875,242 | 262,144 | 3.34× |
| Medium (2048×2048) | 13,986,206 | 4,194,304 | 3.33× |
| Large (4096×4096) | 55,950,158 | 16,777,216 | 3.33× |
| Attention (8192×1024) | 27,994,042 | 8,388,608 | 3.34× |

Since these kernels are memory-bandwidth-bound (each output element is
touched once per non-zero, no register-level reuse in the original
design), moving 3.3× more bytes costs more than the ~33% of multiply-adds
skipped saves. "Zero-skip" was optimizing the wrong resource.

### 2c. The tiled kernel's own cache-fit assumption breaks at realistic shapes

`ternary_gemm_zero_skip_tiled`'s thread-private `CT_local[N×M]` buffers
are sized `nthreads × N × M × 4` bytes. Its own header comment's worked
example (N=256, M so that CT_local≈256KB) fits L2. At Medium Layer's
actual shape (N=2048) with batch=512, that buffer is `8 × 2048 × 512 × 4`
= 32MB — nowhere near L2, and likely spilling L3 too. Measured directly
(native benchmark): `skip_tiled` at Large Layer batch=128 takes
**782.77ms** (2.74 GOPS) vs `skip_avx2`'s 45.55ms (47.14 GOPS) for the
*same* work — a 17× regression from the kernel specifically designed to
avoid this class of problem, once M grows past what its own cache-fit
assumption anticipated.

## 3. Fix: `ternary_gemm_dense.h`/`.cpp`

New kernel, no CSR/CSC index at all:

1. **Pack once, like the CSC/CSR build** (`pack_ternary_dense`): reorders
   `B[K,N]` row-major into `[N/8][K][8]` contiguous blocks — turns the
   column-block-strided-by-N access a naive dense-N-vectorized kernel
   would have into fully sequential reads (a full 64B cache line covers 8
   consecutive `k` values × 8 output columns exactly).
2. **Vectorize over N** (8 output columns/AVX2 register) instead of M —
   this axis exists at any batch size, including `batch=1`.
3. **M-block by 4** (`TERNARY_GEMM_DENSE_MB`): each loaded/widened weight
   tile is reused across 4 batch rows before moving to the next `k`,
   giving batch>1 an arithmetic-intensity gain without needing an index.
4. Reads raw int8 weights (`_mm256_cvtepi8_epi32` → `cvtepi32_ps`, the
   same widening idiom already used by this project's TritNet AVX2 kernel)
   — 4× less memory traffic than NumPy's fp32 weight array.

Correctness: verified bit-exact-within-float-tolerance against the
existing (already-tested) `ternary_gemm_zero_skip_scalar` kernel across 9
shapes including non-multiple-of-8 N/K, M=1, and tiny sizes (all
`maxerr=0.000000`); against a float64 scalar reference in the native
benchmark (`maxerr` 3e-5 to 6e-4, consistent with float32 accumulation
error at these K); and via `tests/python/test_zero_skip_gemm.py`'s
`DenseWeights` test groups (6 shapes × 3 sparsities, extreme all-zero/
no-zero matrices, scalar-vs-AVX2 agreement, `info()` byte accounting,
input validation) — 10/10 groups pass.

## 4. Results — native, pybind-free (`bench_gemm_dense.cpp`)

Isolates the kernel from Python/pybind11 overhead per this project's
`ffi_isolation` convention. No native benchmark existed for either GEMM
kernel before this session.

```
g++ -O3 -march=haswell -mavx2 -mfma -fopenmp -std=c++17 \
    -I../../src/core/simd bench_gemm_dense.cpp \
    ../../src/core/simd/ternary_gemm_zero_skip.cpp \
    ../../src/core/simd/ternary_gemm_dense.cpp -o bench_gemm_dense
```

Speedup of `dense_avx2` vs the *better* of the two existing zero-skip
kernels, same shapes as `bench_competitive.py` Phase 4:

| Shape | batch=1 | batch=8 | batch=32 | batch=128 |
|---|---|---|---|---|
| Small MLP | **28.6×** | 9.8× | 2.6× | 0.45× (loses) |
| Medium Layer | **31.9×** | 7.9× | 3.1× | 2.0× |
| Large Layer | **19.0×** | 6.9× | 3.0× | 2.0× |
| Attention Head | **21.8×** | 7.6× | 2.9× | 1.9× |

At batch=1 — the exact case Phase 4 tests — this is a 19×-32× win, not a
narrowing of the old loss. The one honest caveat: at batch=128 for the
smallest shape (Small MLP), the fixed `MB=4` register-blocking caps
arithmetic-intensity growth while the old kernel's M-vectorized SIMD keeps
scaling with batch, and `skip_avx2` pulls back ahead (0.45×, i.e. the
dense kernel is 2.2× *slower* there). Not hidden — reported as found.
This project's stated use cases (edge AI, single-token/low-batch
inference) sit in the regime this kernel wins decisively; a
production system doing large-batch training-style matmul would want a
larger or adaptive `MB`, not attempted here (scope: fix the case this
project's own benchmark measures, not build a general BLAS replacement).

## 5. Results — `bench_competitive.py` Phase 4 (Python, pybind11 overhead included)

Switched Phase 4 to `DenseWeights` (from `ZeroSkipWeights.gemm_tiled`).
First run looked dramatic (0.19x → up to 2.2x per cell) but re-running
showed **run-to-run variance up to 50× for the identical shape/code**
(e.g. Small MLP: 0.044x, 0.055x, 1.871x, 2.159x across 4 fresh-process
runs) — investigated rather than averaged away, since this project's own
`interleaved_timing`/statistical-rigor rules exist for exactly this
situation.

**Root cause, reproduced directly:** the old benchmark warmed up with a
fixed 3 calls. That was adequate when the kernel took ~1-5ms/call (several
ms of warmup, enough for this machine's documented `powersave`-governor
DVFS to ramp up — see `reports/2026-08-18/CV_SPIKE_ROOT_CAUSE.md`). Now
that the kernel takes ~10-30µs/call, 3 calls warm the CPU for a fraction
of a millisecond — nowhere near enough. Reproduced: identical fresh-process
runs of the same shape gave call times of 7.6µs, 16.5µs, 257µs, 267µs.

**Fix:** replaced the fixed-count warmup with a fixed **wall-clock**
warmup budget (50ms) and switched from one no-variance timing block to
200 interleaved ternary/NumPy samples (this project's own
`interleaved_timing` convention: both sides see the same clock/thermal
drift), reporting median + a stability check via
`benchmark_framework.compute_timing_statistics()` (the gap #8 statistics
engine already shared by `bench_fair_baseline.py`/`bench_simd_fusion_ops.py`).

**Second-order finding:** mean/stdev-based CV was *also* the wrong
stability metric at this timescale — a single OS-scheduling outlier among
200 samples (one 3.17ms spike against an otherwise rock-steady 13.8µs
median, p90 13.99µs) inflated CV past 300% even though the median itself
was reproducible to a few percent across repeated runs. Replaced with a
p90/median ratio check (≥2× flags real spread, not rare outliers) —
verified against the raw sample distribution before trusting it.

**Stable result, 3 independent fresh-process runs after both fixes:**

| Run | Small MLP | Medium | Large | Attention | Average |
|---|---|---|---|---|---|
| 1 | 2.13x | 0.76x | 1.04x | 0.90x | 1.21x |
| 2 | 2.08x | 0.57x | 0.97x | 0.64x | 1.06x |
| 3 | 2.10x | 0.84x | 1.00x | 0.88x | 1.21x |

**Verdict flips from "✗ TOO SLOW FOR AI" (0.189x avg) to "✓ VIABLE FOR AI"
(~1.1-1.2x avg)** by the script's own >0.5× threshold, consistently across
runs. Some individual cells still trip the (now-honest) p90/median
instability flag on this shared, non-isolated sandbox — not hidden, and a
reason to treat ~1.1-1.2x as "verified real and reproducible in this
environment," not "clean production number," matching this project's
established caveat style for every other shared-sandbox measurement.

## 6. What this does and doesn't claim

- Does NOT claim the "< 2× FP16" commercial-viability criterion is now
  met — Phase 4 compares against NumPy fp32, not fp16; that criterion is
  still unmeasured. The "2/5 criteria validated" figure is unchanged by
  this session for that reason.
- Does claim: the specific, previously-cited 0.189× "too slow for AI"
  matmul figure is stale and superseded — the real number, on the same
  shapes and batch size, with a correctly-optimized kernel, is
  ~1.1-1.2× (average), a genuine reversal, not a tuning nudge.
- `ZeroSkipWeights` (CSC/CSR) is kept in the codebase for comparison and
  regression tracking, not removed — but is no longer the kernel this
  project's benchmarks or documentation should cite as "the" ternary GEMM
  path.

## Files changed

- `src/core/simd/ternary_gemm_dense.{h,cpp}` (new)
- `src/engine/bindings_zero_skip_gemm.cpp` (added `DenseWeights` class)
- `build/build_zero_skip_gemm.py` (added new source; also fixed a
  pre-existing `-march=native` SIGILL-risk bug in this same file, the
  same crash class already fixed elsewhere per CLAUDE.md Critical Gaps)
- `tests/python/test_zero_skip_gemm.py` (4 new test groups, 10/10 pass)
- `benchmarks/cpp-native-kernels/bench_gemm_dense.cpp` (new — no native
  GEMM benchmark existed before this session)
- `benchmarks/python-with-interpreter-overhead/bench_competitive.py`
  (Phase 4 switched to `DenseWeights`; warmup/timing methodology fixed)

`tests/run_tests.py`: 16/16 throughout.
