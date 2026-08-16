# Deprecated Benchmarks

**Status:** Deprecated - Do not use for new development

These scripts were written against an earlier, since-removed `ternary_backend`
module and originally deprecated when the project consolidated to
`ternary_simd_engine` (see "Why Deprecated" below).

**Correction (2026-08-15):** a *different*, unrelated `ternary_backend`
module has since been built from scratch (the v1.2.0 pluggable
Scalar/AVX2_v1/AVX2_v2 backend system — `src/engine/bindings_backend_api.cpp`,
`build/build_backend.py`, in CI since 2026-08-12; see `.claude/CLAUDE.md`).
Its API (`init()`, `list_backends()`, `set_backend()`, `tadd()`/`tnot()`/etc.)
turns out to be compatible with these old scripts by coincidence of naming,
not by design — all 9 were verified to import and run successfully against
the current `ternary_backend` today, once a separate `sys.path` bug (each
script was 1 `.parent` short of the true repo root, since this directory is
2 levels deep) was fixed. That does **not** mean their benchmark
methodology, API assumptions, or results are current or trustworthy — this
directory remains deprecated. Do not treat a successful run as validation;
follow the migration path below or use the active benchmarks under
`benchmarks/python-with-interpreter-overhead/` instead.

## Files

| File | Original Purpose |
|------|------------------|
| `bench_backends.py` | Backend comparison benchmarks |
| `bench_backends_improved.py` | Improved backend benchmarks |
| `bench_backend_fusion.py` | Backend fusion operation tests |
| `bench_fusion_validation.py` | Phase 4.1 fusion validation |
| `bench_with_load_context.py` | Load-aware benchmarking |
| `bench_fusion_phase41.py` | Phase 4.1 micro benchmarks |
| `bench_fusion_poc.py` | Fusion proof-of-concept |
| `bench_fusion_rigorous.py` | Rigorous fusion testing |
| `bench_fusion_simple.py` | Simple fusion benchmarks |

## Migration Path

To revive these scripts, replace:
```python
import ternary_backend as tb
```

With:
```python
import ternary_simd_engine as tse
```

And update API calls accordingly. The `ternary_simd_engine` module provides equivalent functionality through a unified interface.

## Why Deprecated

The `ternary_backend` module was an experimental multi-backend architecture that added complexity without significant benefits. The project consolidated to a single optimized `ternary_simd_engine` module.
