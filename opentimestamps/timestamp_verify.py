#!/usr/bin/env python3
"""
OpenTimestamps Verification Script for Ternary Engine
Verifies SHA512 hashes and OpenTimestamps proofs
"""

import hashlib
import json
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Configuration
SCRIPT_DIR = Path(__file__).parent
TIMESTAMP_DIR = SCRIPT_DIR / "timestamps"
LOG_DIR = SCRIPT_DIR / "logs"
PROJECT_ROOT = SCRIPT_DIR.parent


def compute_sha512(file_path: Path) -> str:
    """Compute SHA512 hash of a file"""
    sha512 = hashlib.sha512()

    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            sha512.update(chunk)

    return sha512.hexdigest()


def load_manifest(manifest_path: Path) -> Dict:
    """Load a timestamp manifest file"""
    with open(manifest_path, 'r') as f:
        return json.load(f)


def verify_manifest_integrity(manifest_path: Path) -> Tuple[bool, str]:
    """Verify the manifest file itself hasn't been tampered with"""
    # Look for corresponding .sha512 file
    sha512_file = manifest_path.with_suffix('.sha512')

    if not sha512_file.exists():
        return False, "No SHA512 hash file found for manifest"

    # Read expected hash
    with open(sha512_file, 'r') as f:
        expected_hash = f.read().strip().split()[0]

    # Compute actual hash
    actual_hash = compute_sha512(manifest_path)

    if actual_hash == expected_hash:
        return True, "Manifest integrity verified"
    else:
        return False, f"Manifest has been tampered with!\nExpected: {expected_hash}\nActual: {actual_hash}"


def verify_file_hashes(manifest: Dict) -> Dict[str, Dict]:
    """Verify all file hashes in the manifest"""
    results = {
        "verified": [],
        "modified": [],
        "missing": [],
        "errors": []
    }

    total_files = len(manifest["files"])
    print(f"\nVerifying {total_files} files...")

    for rel_path, file_info in manifest["files"].items():
        file_path = PROJECT_ROOT / rel_path

        try:
            if not file_path.exists():
                results["missing"].append(rel_path)
                print(f"  ✗ MISSING: {rel_path}")
                continue

            # Compute current hash
            current_hash = compute_sha512(file_path)
            expected_hash = file_info["sha512"]

            if current_hash == expected_hash:
                results["verified"].append(rel_path)
                print(f"  ✓ {rel_path}")
            else:
                results["modified"].append({
                    "path": rel_path,
                    "expected": expected_hash,
                    "current": current_hash
                })
                print(f"  ✗ MODIFIED: {rel_path}")

        except Exception as e:
            results["errors"].append({
                "path": rel_path,
                "error": str(e)
            })
            print(f"  ✗ ERROR: {rel_path} - {e}")

    return results


def verify_ots_proof(manifest_path: Path) -> Tuple[bool, str]:
    """Verify OpenTimestamps proof using ots client"""
    ots_file = Path(str(manifest_path) + ".ots")

    if not ots_file.exists():
        return False, "No .ots proof file found"

    try:
        # Check if ots client is installed
        result = subprocess.run(
            ['ots', '--version'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return False, "opentimestamps-client not installed"

        # Verify the timestamp
        result = subprocess.run(
            ['ots', 'verify', str(ots_file)],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            # Parse the output to extract timestamp info
            output = result.stdout + result.stderr
            return True, output
        else:
            return False, f"Verification failed: {result.stderr}"

    except FileNotFoundError:
        return False, "opentimestamps-client not installed"


def print_verification_summary(results: Dict, manifest: Dict):
    """Print a summary of verification results"""
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    total = len(manifest["files"])
    verified = len(results["verified"])
    modified = len(results["modified"])
    missing = len(results["missing"])
    errors = len(results["errors"])

    print(f"\nTotal files: {total}")
    print(f"  ✓ Verified: {verified}")
    print(f"  ✗ Modified: {modified}")
    print(f"  ✗ Missing: {missing}")
    print(f"  ✗ Errors: {errors}")

    if modified:
        print("\n" + "=" * 60)
        print("MODIFIED FILES (HASH MISMATCH)")
        print("=" * 60)
        for item in results["modified"]:
            print(f"\nFile: {item['path']}")
            print(f"  Expected: {item['expected']}")
            print(f"  Current:  {item['current']}")

    if missing:
        print("\n" + "=" * 60)
        print("MISSING FILES")
        print("=" * 60)
        for path in results["missing"]:
            print(f"  - {path}")

    if errors:
        print("\n" + "=" * 60)
        print("ERRORS")
        print("=" * 60)
        for item in results["errors"]:
            print(f"\nFile: {item['path']}")
            print(f"  Error: {item['error']}")


def list_available_manifests():
    """List all available timestamp manifests"""
    manifests = sorted(TIMESTAMP_DIR.glob("manifest_*.json"))

    if not manifests:
        print("No timestamp manifests found.")
        print(f"Create one with: python timestamp_create.py")
        return []

    print("\nAvailable timestamp manifests:")
    print("=" * 60)
    for i, manifest in enumerate(manifests, 1):
        # Extract timestamp from filename
        try:
            data = load_manifest(manifest)
            timestamp = data.get("timestamp_date", "Unknown")
            file_count = len(data.get("files", {}))

            print(f"{i}. {manifest.name}")
            print(f"   Date: {timestamp}")
            print(f"   Files: {file_count}")

            # Check if OTS proof exists
            if Path(str(manifest) + ".ots").exists():
                print(f"   OTS Proof: Yes")
            print()

        except Exception as e:
            print(f"{i}. {manifest.name} (Error reading: {e})")

    return manifests


def main():
    """Main verification function"""
    print("=" * 60)
    print("OpenTimestamps Verification - Ternary Engine")
    print("=" * 60)

    # Check if manifest was specified
    if len(sys.argv) < 2:
        manifests = list_available_manifests()

        if not manifests:
            sys.exit(1)

        print("Usage: python timestamp_verify.py <manifest_file>")
        print(f"Example: python timestamp_verify.py {manifests[-1].name}")
        sys.exit(0)

    # Load specified manifest
    manifest_name = sys.argv[1]
    manifest_path = TIMESTAMP_DIR / manifest_name

    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}")
        sys.exit(1)

    print(f"\nVerifying manifest: {manifest_name}\n")

    # Step 1: Verify manifest integrity
    print("Step 1: Verifying manifest integrity...")
    manifest_ok, manifest_msg = verify_manifest_integrity(manifest_path)

    if manifest_ok:
        print(f"  ✓ {manifest_msg}")
    else:
        print(f"  ✗ {manifest_msg}")
        print("\nWARNING: Manifest has been tampered with!")
        print("Cannot trust verification results.")
        sys.exit(1)

    # Step 2: Load manifest
    manifest = load_manifest(manifest_path)
    print(f"\nManifest created: {manifest['timestamp_date']}")
    print(f"Hash algorithm: {manifest['hash_algorithm']}")

    # Step 3: Verify file hashes
    print("\nStep 2: Verifying file hashes...")
    results = verify_file_hashes(manifest)

    # Step 4: Verify OTS proof (if available)
    print("\nStep 3: Verifying blockchain timestamp...")
    ots_ok, ots_msg = verify_ots_proof(manifest_path)

    if ots_ok:
        print(f"  ✓ OpenTimestamps proof verified!")
        print("\nBlockchain Timestamp Details:")
        print("-" * 60)
        print(ots_msg)
    else:
        print(f"  ⚠ {ots_msg}")

    # Print summary
    print_verification_summary(results, manifest)

    # Overall status
    print("\n" + "=" * 60)
    print("OVERALL STATUS")
    print("=" * 60)

    all_verified = (
        len(results["modified"]) == 0 and
        len(results["missing"]) == 0 and
        len(results["errors"]) == 0
    )

    if all_verified and manifest_ok:
        print("\n✓ ALL CHECKS PASSED")
        print("All files match their timestamp hashes.")
        if ots_ok:
            print("Blockchain timestamp proof verified.")
    else:
        print("\n✗ VERIFICATION FAILED")
        print("Some files have been modified, are missing, or had errors.")

    # Save verification report
    timestamp_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_file = LOG_DIR / f"verification_report_{timestamp_id}.json"

    report = {
        "verification_date": datetime.utcnow().isoformat() + "Z",
        "manifest_file": manifest_name,
        "manifest_date": manifest['timestamp_date'],
        "manifest_integrity": manifest_ok,
        "ots_verified": ots_ok,
        "results": {
            "total_files": len(manifest["files"]),
            "verified": len(results["verified"]),
            "modified": len(results["modified"]),
            "missing": len(results["missing"]),
            "errors": len(results["errors"])
        },
        "modified_files": results["modified"],
        "missing_files": results["missing"],
        "errors": results["errors"]
    }

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nVerification report saved: {report_file}\n")


if __name__ == "__main__":
    main()
