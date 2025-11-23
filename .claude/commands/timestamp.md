Create IP protection timestamp using OpenTimestamps.

**Create new timestamp snapshot**:
```bash
python scripts/timestamp_snapshot.py --create
```

This will:
- Generate SHA512 hash of all 88 tracked source files
- Submit hash to OpenTimestamps Bitcoin blockchain
- Create verifiable proof of existence at specific date/time
- Save timestamp to timestamps/ directory with .ots extension

**Verify existing timestamp**:
```bash
python scripts/timestamp_snapshot.py --verify timestamps/snapshot_YYYYMMDD_HHMMSS.ots
```

**When to create timestamps:**
- Before major releases
- After novel innovations (TritNet breakthroughs, new algorithms)
- Weekly snapshots during active development
- Before public disclosure or publication

**Purpose:**
Establishes provable date of invention for:
- Patent applications
- IP dispute resolution
- Prior art establishment
- Copyright protection

**How it works:**
1. Creates cryptographic hash (SHA512) of source code
2. Submits to Bitcoin blockchain via OpenTimestamps
3. Generates immutable, tamper-proof record
4. Provides verifiable proof of existence at specific date/time

**Existing snapshots:**
- 2025-11-23 (ce39331): Initial snapshot - 88 files including TritNet Phase 1

See timestamps/ directory for all .ots files and verification tools.
