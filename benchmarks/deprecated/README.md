# Deprecated Benchmarks

**Status:** Deprecated - Do not use for new development

These scripts depend on the deprecated `ternary_backend` module which has been replaced by `ternary_simd_engine`.

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
