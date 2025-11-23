# OpenTimestamps Quick Reference

**one-page guide for common operations**

---

## Create Timestamp

**windows**:
```batch
timestamp_now.bat
```

**linux_macos**:
```bash
python timestamp_create.py
```

**output**:
- `timestamps/manifest_*.json` - Your proof
- `timestamps/manifest_*.json.ots` - Blockchain proof
- Keep these files safe!

---

## Verify Timestamp

**list_all**:
```bash
python timestamp_verify.py
```

**verify_specific**:
```bash
python timestamp_verify.py manifest_20251123_150000.json
```

---

## Install OTS Client (Optional)

```bash
pip install opentimestamps-client
ots --version
```

Without this: SHA512 hashes still work, but no blockchain proof.

---

## What Gets Timestamped?

**included**:
- Source code (*.cpp, *.h)
- Python scripts (*.py)
- Documentation (*.md)
- Datasets and configs

**excluded**:
- Build artifacts (*.pyd, *.pyc)
- Temporary files
- Git metadata

Edit `config.json` to customize.

---

## Recommended Schedule

**before_major_release** - Always
**after_novel_innovation** - Yes
**weekly_snapshots** - Good practice
**every_commit** - No, excessive

---

## Files to Backup

**critical**:
- All `.ots` files
- All `manifest_*.json` files
- All `manifest_*.sha512` files

**backup_locations**:
- Git repository (primary)
- Cloud storage (secondary)
- External drive (tertiary)

---

## Verification Results

**all_verified** ✓ - Files unchanged since timestamp
**modified** ✗ - Files have been changed
**missing** ✗ - Files deleted
**pending** ⚠ - Waiting for Bitcoin confirmation (15 min - 24 hrs)

---

## Common Commands

**create**:
```bash
python timestamp_create.py
```

**verify_latest**:
```bash
python timestamp_verify.py $(ls -t timestamps/manifest_*.json | head -1)
```

**list_timestamps**:
```bash
ls -lh timestamps/
```

**check_ots_status**:
```bash
ots verify timestamps/manifest_*.json.ots
```

---

## Troubleshooting

**ots_not_found**:
```bash
pip install opentimestamps-client
```

**files_not_timestamped** - Check `config.json` patterns

**verification_fails** - Expected if files changed

**pending_confirmation** - Wait for Bitcoin block

---

## Emergency: Prove File Existed

1. Find timestamp: `python timestamp_verify.py`
2. Verify: `python timestamp_verify.py manifest_<date>.json`
3. Export report from `logs/verification_report_*.json`
4. Show `.ots` file and verification output
5. Anyone can independently verify on blockchain

---

## Security Notes

**sha512** - Cryptographically secure hash
**bitcoin** - Immutable public blockchain
**calendar_servers** - Distributed, no single point of failure
**verification** - Anyone can verify independently

**limitations**:
- Does not prevent copying
- Not a patent or copyright
- Does not keep code secret
- Consult attorney for legal protection

---

**For full documentation, see README.md**
