# Example Usage Scenarios

**practical examples of using the OpenTimestamps system**

---

## Scenario 1: First-Time Setup

**situation** - You've completed initial development and want to protect your IP

**steps**:
```bash
# 1. Navigate to opentimestamps folder
cd opentimestamps

# 2. (Optional) Install OTS client for blockchain proof
pip install opentimestamps-client

# 3. Review configuration
notepad config.json  # Windows
# or
cat config.json      # Linux/Mac

# 4. Create your first timestamp
python timestamp_create.py

# 5. Verify it worked
python timestamp_verify.py

# 6. Commit to git
cd ..
git add opentimestamps/
git commit -m "IP: Add OpenTimestamps protection system"
git tag v1.0-timestamp
```

**result** - Your code is now cryptographically timestamped

---

## Scenario 2: Pre-Release Protection

**situation** - About to release v2.0 publicly, want to establish prior art

**steps**:
```bash
# 1. Create timestamp before release
cd opentimestamps
python timestamp_create.py

# Output:
# Manifest saved to: timestamps/manifest_20251123_143022.json
# Timestamp created: timestamps/manifest_20251123_143022.json.ots

# 2. Commit timestamp files
cd ..
git add opentimestamps/timestamps/
git add opentimestamps/logs/
git commit -m "TIMESTAMP: Pre-release IP protection for v2.0"

# 3. Tag the release
git tag -a v2.0-timestamp -m "Timestamped before public release"

# 4. Push to repository
git push origin main --tags

# 5. Now release publicly
# Your IP is protected on the blockchain
```

**result** - Blockchain proof that code existed before public disclosure

---

## Scenario 3: Regular Weekly Snapshots

**situation** - Want continuous IP protection during development

**cron_job_linux**:
```bash
# Edit crontab
crontab -e

# Add weekly timestamp (Fridays at 5 PM)
0 17 * * 5 cd /path/to/ternary-engine/opentimestamps && python timestamp_create.py >> logs/weekly.log 2>&1
```

**windows_task_scheduler**:
1. Open Task Scheduler
2. Create Basic Task
3. Name: "Weekly Timestamp"
4. Trigger: Weekly, Friday, 5:00 PM
5. Action: Start a program
   - Program: `python`
   - Arguments: `timestamp_create.py`
   - Start in: `C:\path\to\ternary-engine\opentimestamps`

**manual_weekly**:
```bash
# Every Friday
cd opentimestamps
python timestamp_create.py
git add timestamps/ logs/
git commit -m "TIMESTAMP: Weekly snapshot $(date +%Y-%m-%d)"
git push
```

---

## Scenario 4: Verifying Old Timestamp

**situation** - Need to prove code existed at a specific time (e.g., for legal case)

**steps**:
```bash
# 1. List available timestamps
cd opentimestamps
python timestamp_verify.py

# Output shows:
# 1. manifest_20251101_120000.json
#    Date: 2025-11-01T12:00:00Z
#    Files: 157
#    OTS Proof: Yes

# 2. Verify specific timestamp
python timestamp_verify.py manifest_20251101_120000.json

# 3. Review verification report
cat logs/verification_report_*.json

# 4. Export for legal documentation
cp timestamps/manifest_20251101_120000.json /export/legal-docs/
cp timestamps/manifest_20251101_120000.json.ots /export/legal-docs/
cp logs/verification_report_*.json /export/legal-docs/
```

**independent_verification**:
```bash
# Anyone can verify using OTS client
ots verify manifest_20251101_120000.json.ots

# Shows Bitcoin block where timestamp was anchored
# Example output:
# Success! Bitcoin block 820571 attests existence as of 2025-11-01 12:34:56 UTC
```

---

## Scenario 5: After Major Innovation

**situation** - Implemented novel ternary SIMD algorithm, want immediate protection

**steps**:
```bash
# 1. Create timestamp right away
cd opentimestamps
python timestamp_create.py

# 2. Document the innovation
cd ..
cat > docs/innovations/ternary-simd-innovation.md << 'EOF'
# Ternary SIMD Innovation

**Date:** 2025-11-23
**Timestamp:** manifest_20251123_143022.json

## Innovation
Novel approach to ternary arithmetic using SIMD instructions...

## Prior Art
This innovation was timestamped on the Bitcoin blockchain before disclosure.

**Timestamp Proof:** opentimestamps/timestamps/manifest_20251123_143022.json.ots
EOF

# 3. Commit everything together
git add opentimestamps/
git add docs/innovations/
git commit -m "INNOVATION: Ternary SIMD with timestamp proof"
git tag innovation-ternary-simd

# 4. Push to private repository first
git push origin main --tags

# 5. Wait for blockchain confirmation (15 min - 24 hrs)
# Then can safely publish or discuss publicly
```

---

## Scenario 6: Detecting Unauthorized Changes

**situation** - Someone claims your code was modified, you need to prove original state

**steps**:
```bash
# 1. Find timestamp from that time period
cd opentimestamps
python timestamp_verify.py

# 2. Verify files against that timestamp
python timestamp_verify.py manifest_20251101_120000.json

# Output shows:
# ✓ ternary_core/simd_operations.cpp - VERIFIED
# ✓ ternary_core/ternary_core.cpp - VERIFIED
# ✗ MODIFIED: tests/test_simd.py
#   Expected: 3a2f8b...
#   Current:  9d4c1e...

# 3. Investigate what changed
git diff <commit-at-timestamp> HEAD tests/test_simd.py

# 4. Prove original state
# - Show manifest with SHA512 hash
# - Show .ots file proving blockchain timestamp
# - Show verification output
# - Anyone can independently verify on blockchain
```

---

## Scenario 7: Preparing for Patent Filing

**situation** - Working with patent attorney, need proof of invention date

**steps**:
```bash
# 1. Create comprehensive timestamp
cd opentimestamps
python timestamp_create.py

# 2. Generate verification report
python timestamp_verify.py manifest_20251123_143022.json

# 3. Collect all evidence
mkdir patent-evidence
cp timestamps/manifest_20251123_143022.json patent-evidence/
cp timestamps/manifest_20251123_143022.json.ots patent-evidence/
cp logs/timestamp_log_20251123_143022.txt patent-evidence/
cp logs/verification_report_*.json patent-evidence/

# 4. Add README for attorney
cat > patent-evidence/README.txt << 'EOF'
OpenTimestamps Blockchain Proof

This folder contains cryptographic proof that the Ternary Engine
source code existed on 2025-11-23 at 14:30:22 UTC.

Files:
- manifest_*.json: SHA512 hashes of all source files
- manifest_*.json.ots: Bitcoin blockchain proof
- timestamp_log_*.txt: Creation log
- verification_report_*.json: Verification results

The .ots file can be independently verified by anyone using:
  pip install opentimestamps-client
  ots verify manifest_*.json.ots

This provides cryptographic proof of prior art.
EOF

# 5. Create archive
tar -czf patent-evidence.tar.gz patent-evidence/

# 6. Provide to attorney
cp patent-evidence.tar.gz /path/to/attorney-documents/
```

---

## Scenario 8: Multiple Timestamp Comparison

**situation** - Want to see what changed between two timestamps

**steps**:
```bash
cd opentimestamps

# 1. List timestamps
python timestamp_verify.py

# Shows:
# 1. manifest_20251101_120000.json (Nov 1)
# 2. manifest_20251123_143022.json (Nov 23)

# 2. Extract file lists
python3 << 'EOF'
import json

# Load both manifests
with open('timestamps/manifest_20251101_120000.json') as f:
    old = json.load(f)
with open('timestamps/manifest_20251123_143022.json') as f:
    new = json.load(f)

old_files = set(old['files'].keys())
new_files = set(new['files'].keys())

print("New files added:")
for f in sorted(new_files - old_files):
    print(f"  + {f}")

print("\nFiles removed:")
for f in sorted(old_files - new_files):
    print(f"  - {f}")

print("\nFiles modified:")
for f in sorted(old_files & new_files):
    if old['files'][f]['sha512'] != new['files'][f]['sha512']:
        print(f"  M {f}")
EOF
```

---

## Scenario 9: Automated CI/CD Integration

**situation** - Want to automatically timestamp on major releases

**github_actions**:
```yaml
# .github/workflows/timestamp.yml
name: Create Timestamp

on:
  push:
    tags:
      - 'v*'

jobs:
  timestamp:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install OTS Client
        run: pip install opentimestamps-client

      - name: Create Timestamp
        run: |
          cd opentimestamps
          python timestamp_create.py

      - name: Commit Timestamp
        run: |
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add opentimestamps/timestamps/
          git add opentimestamps/logs/
          git commit -m "TIMESTAMP: Automated timestamp for ${{ github.ref_name }}"
          git push
```

---

## Scenario 10: Emergency IP Defense

**situation** - Someone claims prior invention, you need immediate proof

**steps**:
```bash
# 1. Find earliest timestamp
cd opentimestamps
ls -lt timestamps/manifest_*.json | tail -1

# Shows: manifest_20240601_100000.json (your earliest timestamp)

# 2. Verify timestamp still valid
python timestamp_verify.py manifest_20240601_100000.json

# 3. Verify blockchain proof
ots verify timestamps/manifest_20240601_100000.json.ots

# Output:
# Success! Bitcoin block 812345 attests existence as of 2024-06-01 10:00:00 UTC

# 4. Prepare evidence package
mkdir ip-defense
cp timestamps/manifest_20240601_100000.* ip-defense/
cp logs/timestamp_log_20240601_100000.txt ip-defense/

# 5. Create affidavit document
cat > ip-defense/PROOF_OF_PRIOR_ART.md << 'EOF'
# Proof of Prior Art - Ternary Engine

## Claim
Our Ternary Engine source code existed as of 2024-06-01 10:00:00 UTC,
as proven by Bitcoin blockchain transaction.

## Evidence
1. manifest_20240601_100000.json - SHA512 hashes of all source files
2. manifest_20240601_100000.json.ots - Bitcoin blockchain proof
3. Independent verification: Anyone can verify using OpenTimestamps

## Verification Steps
```bash
pip install opentimestamps-client
ots verify manifest_20240601_100000.json.ots
```

## Result
Bitcoin block 812345 permanently attests that our code existed
on 2024-06-01, establishing prior art.
EOF

# 6. Archive everything
zip -r ip-defense-evidence.zip ip-defense/
```

**key_point** - The blockchain proof is immutable and independently verifiable by anyone, making it very strong evidence.

---

## Best Practices Summary

**when_to_timestamp**:
- Before major releases
- After novel innovations
- Weekly/monthly snapshots
- Before public disclosure
- Before patent filing

**what_to_commit**:
- All `timestamps/` files
- All `logs/` files
- Keep in version control

**how_to_store**:
- Primary: Git repository
- Secondary: Cloud backup
- Tertiary: External drive
- Critical releases: Print hashes

**verification_schedule**:
- Verify new timestamps immediately
- Re-verify old timestamps periodically
- Verify before legal use

---

**For more information, see README.md and QUICK_REFERENCE.md**
