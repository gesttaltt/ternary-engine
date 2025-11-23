# Scripts Directory

**Purpose:** Centralized location for all project scripts organized by function.

---

## Directory Structure

```
scripts/
├── build/          # C++ module compilation scripts
│   ├── build.py                # Standard optimized build
│   ├── build_dense243.py       # Dense243 module build
│   ├── build_pgo.py            # Profile-guided optimization
│   └── ...
│
├── tritnet/        # TritNet neural network training
│   ├── generate_truth_tables.py  # Generate training datasets
│   ├── train_tritnet.py          # Train TritNet models
│   ├── ternary_layers.py         # Ternary NN layers
│   ├── tritnet_model.py          # Model definitions
│   └── README.md                 # TritNet documentation
│
└── run_tritnet.py  # Orchestration script (unified TritNet workflow)
```

---

## Quick Start

### Build C++ Modules

```bash
# Standard optimized build (main engine)
python scripts/build/build.py

# Dense243 module (separate module)
python scripts/build/build_dense243.py

# Profile-guided optimization (max performance)
python scripts/build/build_pgo_unified.py
```

### TritNet Training (Neural Network Arithmetic)

```bash
# Full workflow: generate datasets → train all models
python scripts/run_tritnet.py --all

# Individual steps:
python scripts/run_tritnet.py --generate-datasets
python scripts/run_tritnet.py --train tnot
python scripts/run_tritnet.py --train-all
```

---

## Script Categories

### 1. Build Scripts (`scripts/build/`)

**Purpose:** Compile C++ Python extensions

**Key scripts:**
- `build.py` - Standard build for main engine
- `build_dense243.py` - Dense243 module (experimental, separate)
- `build_pgo.py` - Profile-guided optimization (advanced)
- `build_reference.py` - Reference baseline (for benchmarks)

**Usage:** `python scripts/build/<script>.py`

**Outputs:** Compiled `.pyd` (Windows) or `.so` (Linux/macOS) modules

**Dependencies:** pybind11, NumPy, C++17 compiler with AVX2 support

---

### 2. TritNet Training (`scripts/tritnet/`)

**Purpose:** Train neural networks to learn ternary arithmetic

**Key scripts:**
- `generate_truth_tables.py` - Generate 236K training samples
- `train_tritnet.py` - Train TritNet models on operations
- `ternary_layers.py` - PyTorch layers with ternary weights {-1, 0, +1}
- `tritnet_model.py` - Model architectures (unary/binary operations)

**Usage:** See `scripts/tritnet/README.md` for detailed documentation

**Outputs:**
- `datasets/tritnet/*.json` - Training datasets (truth tables)
- `models/tritnet/*.tritnet` - Trained models

**Dependencies:** PyTorch, NumPy

**Background:** TritNet learns exact ternary arithmetic using neural networks with pure ternary weights, enabling potential hardware acceleration via matmul instead of lookup tables.

---

### 3. Orchestration (`scripts/run_tritnet.py`)

**Purpose:** Unified workflow for TritNet experiments

**Features:**
- Generate truth tables for all operations
- Train individual or all operations
- Run validation and Go/No-Go analysis
- Coordinate multi-step workflows

**Usage:**
```bash
# Full pipeline
python scripts/run_tritnet.py --all

# Individual phases
python scripts/run_tritnet.py --phase datasets
python scripts/run_tritnet.py --phase training
python scripts/run_tritnet.py --phase validation

# Specific operations
python scripts/run_tritnet.py --train tnot --hidden-size 8
```

---

## Design Principles

### Separation of Concerns

**Build vs Training:**
- `scripts/build/` = Compiling C++ code → Python extensions
- `scripts/tritnet/` = Training PyTorch models → Neural networks

**Why separate?**
- Different languages (C++ vs Python)
- Different dependencies (compiler vs PyTorch)
- Different outputs (binary modules vs trained models)
- Different workflows (compilation vs training)

### Orchestration Layer

**Single entry point:** `scripts/run_tritnet.py` coordinates TritNet workflow

**Benefits:**
- Users don't need to know individual scripts
- Ensures correct execution order
- Validates prerequisites
- Provides unified CLI

### Documentation Structure

**Top-level:** `scripts/README.md` (this file) - Navigation and overview

**Category-level:**
- `scripts/build/` - See `build/README.md` (points to artifact docs)
- `scripts/tritnet/README.md` - Complete TritNet documentation

**Script-level:** Each script has `--help` and inline documentation

---

## Common Workflows

### Workflow 1: Build Main Engine

```bash
# Standard build (recommended)
python scripts/build/build.py

# Test it works
python -c "import ternary_simd_engine; print('Success')"
```

### Workflow 2: Build Dense243 Module

```bash
# Build dense243 module
python scripts/build/build_dense243.py

# Test it works
python -c "import ternary_dense243_module as td; print(td.__version__)"
```

### Workflow 3: TritNet Proof-of-Concept (Phase 2A)

```bash
# 1. Generate truth tables (if not already done)
python scripts/tritnet/generate_truth_tables.py

# 2. Train tnot (simplest operation, proof-of-concept)
python scripts/tritnet/train_tritnet.py --operation tnot --hidden-size 8

# 3. Check results
ls models/tritnet/tritnet_tnot.tritnet
```

### Workflow 4: Full TritNet Training (All Operations)

```bash
# Orchestrated workflow (recommended)
python scripts/run_tritnet.py --all

# Or manual steps:
python scripts/tritnet/generate_truth_tables.py
python scripts/tritnet/train_tritnet.py --all
```

---

## File Naming Conventions

### Build Scripts

**Pattern:** `build_<target>.py`

**Examples:**
- `build.py` - Default/standard build
- `build_dense243.py` - Specific module
- `build_pgo.py` - Build variant (PGO)
- `build_reference.py` - Build type (reference)

### TritNet Scripts

**Pattern:** `<verb>_tritnet.py` or `<noun>.py` for libraries

**Examples:**
- `generate_truth_tables.py` - Verb: action script
- `train_tritnet.py` - Verb: action script
- `ternary_layers.py` - Noun: library module
- `tritnet_model.py` - Noun: library module

### Orchestration

**Pattern:** `run_<workflow>.py`

**Examples:**
- `run_tritnet.py` - Run TritNet workflow

---

## Adding New Scripts

### For Build Scripts

1. Add to `scripts/build/`
2. Follow naming: `build_<target>.py`
3. Use pybind11 Extension pattern
4. Output to appropriate `build/artifacts/` subdirectory
5. Update `build/README.md` if needed

### For TritNet Scripts

1. Add to `scripts/tritnet/`
2. Follow naming conventions (verb or noun)
3. Import from `ternary_layers.py` / `tritnet_model.py` for consistency
4. Update `scripts/tritnet/README.md`
5. Integrate with `scripts/run_tritnet.py` if part of main workflow

### For New Workflows

1. Create orchestration script at `scripts/run_<workflow>.py`
2. Document in this README
3. Provide `--help` and clear CLI

---

## Dependencies

### Build Scripts

```bash
pip install pybind11 numpy
```

**System requirements:**
- C++17 compiler (MSVC, GCC, or Clang)
- AVX2-capable CPU (Intel Haswell 2013+, AMD Excavator 2015+)

### TritNet Scripts

```bash
pip install torch numpy matplotlib
```

**Optional for analysis:**
```bash
pip install seaborn pandas scikit-learn
```

**No CUDA required** (CPU-only training for Phase 2)

---

## Outputs

### Build Artifacts

**Location:** `build/artifacts/`

**Structure:**
```
build/artifacts/
├── standard/           # Standard optimized builds
│   └── latest/
├── pgo/                # Profile-guided optimization
│   └── latest/
└── reference/          # Reference baselines
    └── latest/
```

**Files:**
- `ternary_simd_engine.*.pyd/.so` - Main engine module
- `ternary_dense243_module.*.pyd/.so` - Dense243 module
- `manifest.txt` - Build metadata

### TritNet Artifacts

**Datasets:** `datasets/tritnet/`
```
datasets/tritnet/
├── tadd_truth_table.json     # 59,049 samples
├── tmul_truth_table.json     # 59,049 samples
├── tmin_truth_table.json     # 59,049 samples
├── tmax_truth_table.json     # 59,049 samples
├── tnot_truth_table.json     # 243 samples
├── generation_summary.json   # Metadata
└── README.md
```

**Models:** `models/tritnet/`
```
models/tritnet/
├── tritnet_tnot.tritnet           # Trained model
├── tritnet_tnot_history.json      # Training history
├── tritnet_tadd.tritnet
├── tritnet_tadd_history.json
└── ... (one per operation)
```

---

## Troubleshooting

### "Module not found" errors

**For build scripts:**
```bash
pip install pybind11 numpy
```

**For TritNet scripts:**
```bash
pip install torch numpy
```

### "Cannot import ternary_dense243_module"

Build it first:
```bash
python scripts/build/build_dense243.py
```

### "Truth tables not found"

Generate them first:
```bash
python scripts/tritnet/generate_truth_tables.py
```

### Circular imports

Add project root to path (scripts handle this automatically):
```python
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
```

---

## Related Documentation

- **Build System:** `build/README.md`
- **TritNet Training:** `scripts/tritnet/README.md`
- **TritNet Vision:** `docs/TRITNET_VISION.md`
- **TritNet Roadmap:** `docs/TRITNET_ROADMAP.md`
- **Dense243 Module:** `ternary_engine/experimental/dense243/README.md`

---

**Version:** 1.0 · **Updated:** 2025-11-23
**Maintained by:** Jonathan Verdun (Ternary Engine Project)
