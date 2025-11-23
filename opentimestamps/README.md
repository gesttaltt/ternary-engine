# OpenTimestamps IP Protection System

**Doc-Type:** Technical Documentation · Version 1.0 · Updated 2025-11-23 · Ternary Engine

Cryptographic timestamping system using SHA512 and Bitcoin blockchain for intellectual property protection.

---

## Purpose & Overview

**what_this_does** - Creates cryptographically verifiable proof of when your code existed
**why_it_matters** - Protects intellectual property by establishing prior art on the Bitcoin blockchain
**how_it_works** - Computes SHA512 hashes of files, creates manifest, submits to blockchain via OpenTimestamps

**trust_model**:
- SHA512 cryptographic hashes (virtually impossible to forge)
- Bitcoin blockchain immutability (cannot be altered retroactively)
- Distributed calendar servers (no single point of failure)
- Open protocol (independently verifiable by anyone)

---

## Quick Start

### Create Your First Timestamp

**windows**:
```batch
cd opentimestamps
timestamp_now.bat
```

**linux_macos**:
```bash
cd opentimestamps
python timestamp_create.py
```

**what_happens**:
1. Scans your codebase for critical files
2. Computes SHA512 hash of each file
3. Creates manifest with all hashes
4. Submits manifest hash to Bitcoin blockchain
5. Saves .ots proof file

**output_files**:
- `timestamps/manifest_YYYYMMDD_HHMMSS.json` - File hashes
- `timestamps/manifest_YYYYMMDD_HHMMSS.sha512` - Manifest hash
- `timestamps/manifest_YYYYMMDD_HHMMSS.json.ots` - Blockchain proof
- `logs/timestamp_log_YYYYMMDD_HHMMSS.txt` - Creation log

---

## Verification

### Verify Existing Timestamp

**list_available**:
```bash
python timestamp_verify.py
```

**verify_specific**:
```bash
python timestamp_verify.py manifest_20251123_150000.json
```

**what_it_checks**:
1. Manifest integrity (hasn't been tampered with)
2. File hashes (files match original state)
3. Blockchain proof (timestamp is on Bitcoin blockchain)

**verification_results**:
- ✓ All verified - Files unchanged since timestamp
- ✗ Modified - Files have been changed
- ✗ Missing - Files have been deleted
- ✗ Errors - Cannot read files

---

## How It Works

### SHA512 Hash Function

**algorithm** - SHA512 (Secure Hash Algorithm 512-bit)
**properties**:
- Deterministic (same file = same hash)
- Unique (different files = different hashes)
- One-way (cannot reverse hash to get file)
- Collision-resistant (cannot find two files with same hash)

**example**:
```
File: hello.txt containing "Hello, World!"
SHA512: 374d794a95cdcfd8b35993185fef9ba368f160d8daf432d08ba9f1ed1e5abe6c...
```

If you change even one character, the hash completely changes.

### OpenTimestamps Protocol

**calendar_servers**:
- alice.btc.calendar.opentimestamps.org
- bob.btc.calendar.opentimestamps.org
- finney.calendar.eternitywall.com

**process**:
1. Submit hash to calendar servers
2. Servers aggregate multiple hashes
3. Merkle tree root committed to Bitcoin transaction
4. Proof returned showing your hash in the tree

**timeline**:
- Immediate: Pending timestamp created
- ~15 min to 24 hrs: Bitcoin confirmation
- Forever: Immutable proof on blockchain

**cost** - Free (servers batch multiple timestamps into single Bitcoin transaction)

---

## Configuration

### Files to Timestamp

Edit `config.json` to customize what gets timestamped:

**source_code** - C++ implementation files
**critical_scripts** - Python build and test scripts
**documentation** - Important markdown files
**datasets** - Training data and results
**build_configs** - Build system files

**exclude_patterns** - Temporary and generated files

### Example Configuration

```json
{
  "timestamp_config": {
    "hash_algorithm": "sha512"
  },
  "files_to_timestamp": {
    "source_code": [
      "ternary_core/**/*.cpp",
      "ternary_core/**/*.h"
    ]
  },
  "exclude_patterns": [
    "**/__pycache__/**",
    "**/*.pyc"
  ]
}
```

---

## Use Cases

### Before Major Release

**when** - Before publishing code publicly
**why** - Establish prior art before disclosure
**how**:
```bash
cd opentimestamps
python timestamp_create.py
git add timestamps/ logs/
git commit -m "TIMESTAMP: Pre-release IP protection"
git tag v1.0-timestamp
```

### Weekly Snapshots

**when** - Every Friday or end of sprint
**why** - Continuous IP protection during development
**how** - Add to CI/CD or run manually

### After Major Innovation

**when** - Implemented novel algorithm or approach
**why** - Protect specific innovation with timestamp
**how** - Create timestamp, archive files separately

### Before Patent Filing

**when** - Before filing provisional or full patent
**why** - Additional evidence of invention date
**how** - Create timestamp, include in patent documentation

### Legal Documentation

**when** - Need to prove code existed at specific time
**why** - Defend against IP claims or establish priority
**how** - Verify timestamp, export verification report

---

## Installation

### Required

**python** - Python 3.7 or higher (already installed)
**libraries** - Standard library only (hashlib, json, pathlib)

### Optional (for Blockchain Timestamping)

**opentimestamps_client**:
```bash
pip install opentimestamps-client
```

**verification**:
```bash
ots --version
```

**if_not_installed** - Scripts still work but skip blockchain submission

---

## File Structure

```
opentimestamps/
├── README.md                    # This file
├── config.json                  # Configuration
├── timestamp_create.py          # Creation script
├── timestamp_verify.py          # Verification script
├── timestamp_now.bat            # Windows quick-start
├── .gitignore                   # Git ignore rules
├── timestamps/                  # Timestamp proofs (commit to git)
│   ├── manifest_*.json          # File hashes
│   ├── manifest_*.sha512        # Manifest hashes
│   └── manifest_*.json.ots      # Blockchain proofs
└── logs/                        # Audit trail (commit to git)
    ├── timestamp_log_*.txt      # Creation logs
    └── verification_report_*.json # Verification reports
```

---

## Security Considerations

### What This Protects Against

**backdating_claims** - Blockchain proves when hash existed
**file_tampering** - SHA512 detects any modification
**repudiation** - Cannot deny timestamp on public blockchain
**loss_of_evidence** - Multiple copies on blockchain nodes

### What This Does NOT Protect Against

**code_theft** - Does not prevent unauthorized copying
**patent_trolls** - Not a substitute for patents
**copyright_registration** - Not a legal copyright
**trade_secrets** - Timestamps make existence public

### Best Practices

**commit_to_git** - Timestamp files should be in version control
**multiple_copies** - Keep timestamps in multiple locations
**regular_schedule** - Timestamp regularly, not just once
**document_intent** - Keep logs explaining why timestamps were created
**verify_periodically** - Confirm old timestamps still verify

---

## Legal Disclaimer

**not_legal_advice** - This is a technical tool, not legal protection
**consult_attorney** - Speak with IP attorney about your situation
**patent_priority** - Timestamps may help establish prior art but don't replace patents
**copyright_notice** - Use proper copyright notices in your code
**trade_secret** - Don't timestamp if keeping something secret

**evidence_value** - Timestamps are admissible evidence but not absolute proof

---

## Troubleshooting

### OTS Client Not Found

**symptom**:
```
Warning: opentimestamps-client not found
```

**solution**:
```bash
pip install opentimestamps-client
```

**workaround** - Scripts still create SHA512 hashes without blockchain proof

### Files Not Being Timestamped

**check_config** - Verify patterns in `config.json`
**check_exclusions** - Files might match exclude patterns
**check_paths** - Ensure files exist relative to project root

### Verification Fails

**modified_files** - Files have changed since timestamp
**missing_files** - Files have been deleted
**corrupt_manifest** - Manifest file has been tampered with

**expected_behavior** - Verification should fail if files changed

### Blockchain Verification Pending

**symptom**:
```
ots verify: Pending confirmation in Bitcoin blockchain
```

**solution** - Wait 15 minutes to 24 hours for Bitcoin confirmation
**check_later** - Run verification again after waiting

---

## Advanced Usage

### Manual OTS Commands

**create_timestamp**:
```bash
ots stamp file.txt
# Creates file.txt.ots
```

**verify_timestamp**:
```bash
ots verify file.txt.ots
```

**upgrade_timestamp** (after Bitcoin confirmation):
```bash
ots upgrade file.txt.ots
```

**info_about_timestamp**:
```bash
ots info file.txt.ots
```

### Timestamp Individual Files

```python
import hashlib

# Compute hash
with open('important.cpp', 'rb') as f:
    hash = hashlib.sha512(f.read()).hexdigest()

print(f"SHA512: {hash}")
```

### Compare Manifests

```bash
# List all manifests
python timestamp_verify.py

# Compare two timestamps to see what changed
diff timestamps/manifest_20251101_*.json timestamps/manifest_20251123_*.json
```

---

## Integration with Git

### Git Hooks

Add to `.git/hooks/pre-commit`:
```bash
#!/bin/bash
# Optionally create timestamp before important commits
if git log -1 --pretty=%B | grep -q "RELEASE"; then
    cd opentimestamps
    python timestamp_create.py
    cd ..
    git add opentimestamps/timestamps/
    git add opentimestamps/logs/
fi
```

### Git Tags

```bash
# Create timestamp
python timestamp_create.py

# Tag the commit with timestamp reference
git tag -a v1.0-timestamp -m "Timestamped release v1.0"
```

---

## Backup Strategy

### What to Back Up

**critical**:
- All `.ots` files (blockchain proofs)
- All `.json` manifest files
- All `.sha512` hash files

**important**:
- Log files (audit trail)
- `config.json` (configuration)

**optional**:
- Verification reports

### Where to Store

**version_control** - Commit to git (primary backup)
**cloud_storage** - Google Drive, Dropbox, etc.
**physical_media** - USB drives, external hard drives
**print** - Print manifest hashes for critical releases
**email** - Email yourself timestamp files

**redundancy** - 3-2-1 rule: 3 copies, 2 different media, 1 offsite

---

## FAQ

### Why SHA512 instead of SHA256?

**security_margin** - SHA512 offers larger security margin
**future_proofing** - Resistant to quantum computing threats
**standard** - Widely accepted cryptographic standard
**performance** - Negligible performance difference for this use case

### Can timestamps be faked?

**short_answer** - No, not after Bitcoin confirmation

**details**:
- Cannot backdate Bitcoin transactions
- Cannot forge SHA512 hashes
- Cannot modify blockchain after confirmation
- Multiple independent calendar servers

### Do I need to timestamp every commit?

**recommendation** - No, only significant milestones

**good_times_to_timestamp**:
- Major releases
- Novel innovations
- Before public disclosure
- Monthly/quarterly snapshots

**avoid_over_timestamping** - Timestamps are for protection, not version control

### What if I lose the .ots file?

**can_recover** - If you have the manifest, you can recreate the timestamp

**process**:
1. Compute SHA512 of manifest file
2. Query OpenTimestamps calendar servers
3. Retrieve proof if still available

**best_practice** - Keep multiple backups of .ots files

### Is this legally binding?

**evidence** - Can be used as evidence in legal proceedings
**not_conclusive** - Must be interpreted by court/judge
**supporting_documentation** - Combine with other evidence
**jurisdiction_dependent** - Laws vary by country

**consult_attorney** - For legal questions about IP protection

---

## Resources

### OpenTimestamps

**website** - https://opentimestamps.org/
**github** - https://github.com/opentimestamps
**specification** - https://github.com/opentimestamps/opentimestamps-client/blob/master/doc/specification.md
**python_client** - https://github.com/opentimestamps/opentimestamps-client

### Cryptography

**sha512** - https://en.wikipedia.org/wiki/SHA-2
**merkle_trees** - https://en.wikipedia.org/wiki/Merkle_tree
**bitcoin** - https://bitcoin.org/bitcoin.pdf

### Intellectual Property

**prior_art** - https://en.wikipedia.org/wiki/Prior_art
**patent_basics** - https://www.uspto.gov/patents/basics
**copyright** - https://www.copyright.gov/

---

## Support

**issues** - Create issue in project repository
**questions** - Contact project maintainers
**contributions** - Pull requests welcome

---

## Changelog

| Date       | Version | Changes                          |
|:-----------|:--------|:---------------------------------|
| 2025-11-23 | 1.0     | Initial IP protection system     |

---

**Remember:** This tool protects your intellectual property by establishing cryptographic proof of when your code existed. Use it regularly for important milestones.
