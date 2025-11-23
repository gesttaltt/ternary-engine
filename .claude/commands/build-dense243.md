Build the Dense243 high-density encoding module.

**Build the Dense243 module**:
```bash
python build/build_dense243.py
```

This will:
- Compile ternary_dense243_module.cpp with C++17 and AVX2 optimizations
- Create 5 trits/byte encoding (95.3% state utilization vs 4 trits/byte standard)
- Generate build artifacts in build/artifacts/dense243/<timestamp>/
- Copy compiled module to project root

**Features:**
- Pack/unpack: 2-bit ↔ dense243 format
- Operations: tadd, tmul, tmin, tmax, tnot on dense243 data
- TritNet-ready: Backend selection for future NN integration
- 80% space savings vs standard encoding

**Performance:**
- Pack: 0.25 ns/element
- Unpack: 0.91 ns/element
- All 243 states validated

**Verify the module**:
```bash
python -c "import ternary_dense243_module as td; print(td.__doc__)"
```

**Usage example**:
```python
import numpy as np
import ternary_dense243_module as td

# Pack 5 trits into 1 byte (vs 5 bytes in standard encoding)
trits = np.array([0b00, 0b01, 0b10, 0b10, 0b01], dtype=np.uint8)
packed = td.pack(trits)  # 5 → 1 byte (80% space savings)
unpacked = td.unpack(packed)  # 1 → 5 bytes

# Future: Neural network-based operations
td.set_backend('tritnet')  # Switch from LUT to trained model
result = td.tadd(packed_a, packed_b)  # Uses matmul instead of lookup
```

**Status:** Validated & ready (pack/unpack work, integration pending)
