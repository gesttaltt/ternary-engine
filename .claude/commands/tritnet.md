Work with TritNet neural network-based ternary arithmetic.

**Generate truth tables** (required before training):
```bash
# Generate all operations
python scripts/tritnet/generate_truth_tables.py --all --output datasets/tritnet

# Generate specific operation
python scripts/tritnet/generate_truth_tables.py --operation tnot --output datasets/tritnet
```

**Train TritNet models**:
```bash
# Train tnot (proof-of-concept, Phase 2A)
python scripts/tritnet/train_tritnet.py --operation tnot --hidden-size 8

# Train all binary operations (Phase 2B)
python scripts/tritnet/train_tritnet.py --all --output-dir models/tritnet

# Train specific operation
python scripts/tritnet/train_tritnet.py --operation tadd --hidden-size 16
```

**Unified workflow** (orchestrated):
```bash
# Complete TritNet workflow
python scripts/run_tritnet.py --all
```

Training outputs:
- Trained models: models/tritnet/tritnet_<operation>.tritnet
- Training history: models/tritnet/tritnet_<operation>_history.json
- Metrics: Loss, accuracy, validation accuracy per epoch

**Current status:**
- Phase 1: Truth table generation ✅ COMPLETE
- Phase 2A: tnot training 🔄 IN PROGRESS (validate 100% accuracy)
- Phase 2B: All operations ⏳ PENDING

**Next steps:**
1. Validate tnot model achieves 100% accuracy
2. Make Go/No-Go decision for TritNet approach
3. If successful, train remaining operations (tadd, tmul, tmin, tmax)
4. Export weights to C++ for integration (Phase 3)
