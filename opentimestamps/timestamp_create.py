#!/usr/bin/env python3
"""
OpenTimestamps Creation Script for Ternary Engine
Generates cryptographic timestamps using SHA512 for IP protection
"""

import hashlib
import json
import os
import sys
import glob
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set

# Configuration
SCRIPT_DIR = Path(__file__).parent
CONFIG_FILE = SCRIPT_DIR / "config.json"
TIMESTAMP_DIR = SCRIPT_DIR / "timestamps"
LOG_DIR = SCRIPT_DIR / "logs"
PROJECT_ROOT = SCRIPT_DIR.parent


def load_config() -> Dict:
    """Load configuration from config.json"""
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)


def should_exclude(file_path: str, exclude_patterns: List[str]) -> bool:
    """Check if file matches any exclude pattern"""
    for pattern in exclude_patterns:
        if glob.fnmatch.fnmatch(file_path, pattern):
            return True
    return False


def get_files_to_timestamp(config: Dict) -> Set[Path]:
    """Get all files that need timestamping based on config"""
    files = set()
    exclude_patterns = config.get("exclude_patterns", [])

    for category, patterns in config["files_to_timestamp"].items():
        for pattern in patterns:
            # Convert glob pattern to absolute path
            full_pattern = PROJECT_ROOT / pattern

            # Handle recursive glob patterns
            matching_files = glob.glob(str(full_pattern), recursive=True)

            for file_path in matching_files:
                rel_path = os.path.relpath(file_path, PROJECT_ROOT)

                # Skip excluded files
                if should_exclude(rel_path, exclude_patterns):
                    continue

                # Only include files (not directories)
                if os.path.isfile(file_path):
                    files.add(Path(file_path))

    return files


def compute_sha512(file_path: Path) -> str:
    """Compute SHA512 hash of a file"""
    sha512 = hashlib.sha512()

    with open(file_path, 'rb') as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(65536), b''):
            sha512.update(chunk)

    return sha512.hexdigest()


def create_manifest(files: Set[Path], output_file: Path) -> Dict:
    """Create a manifest of all files with their SHA512 hashes"""
    manifest = {
        "timestamp_date": datetime.utcnow().isoformat() + "Z",
        "hash_algorithm": "SHA512",
        "project": "Ternary Engine",
        "files": {}
    }

    print(f"Computing SHA512 hashes for {len(files)} files...")

    for file_path in sorted(files):
        try:
            rel_path = os.path.relpath(file_path, PROJECT_ROOT)
            hash_value = compute_sha512(file_path)

            manifest["files"][rel_path] = {
                "sha512": hash_value,
                "size_bytes": os.path.getsize(file_path),
                "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
            }

            print(f"  ✓ {rel_path}")
        except Exception as e:
            print(f"  ✗ {rel_path}: {e}", file=sys.stderr)

    # Save manifest
    with open(output_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved to: {output_file}")
    return manifest


def compute_manifest_hash(manifest_file: Path) -> str:
    """Compute SHA512 of the manifest file itself"""
    return compute_sha512(manifest_file)


def create_ots_timestamp(file_path: Path) -> bool:
    """Create OTS timestamp using opentimestamps-client"""
    try:
        # Check if ots client is installed
        result = subprocess.run(
            ['ots', '--version'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("\nWarning: opentimestamps-client not found.")
            print("Install with: pip install opentimestamps-client")
            print("Skipping blockchain timestamp submission.\n")
            return False

        # Stamp the file
        print(f"\nSubmitting to OpenTimestamps calendar servers...")
        result = subprocess.run(
            ['ots', 'stamp', str(file_path)],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"✓ Timestamp created: {file_path}.ots")
            return True
        else:
            print(f"✗ Timestamp creation failed: {result.stderr}")
            return False

    except FileNotFoundError:
        print("\nWarning: 'ots' command not found.")
        print("Install with: pip install opentimestamps-client")
        print("Skipping blockchain timestamp submission.\n")
        return False


def main():
    """Main execution function"""
    print("=" * 60)
    print("OpenTimestamps Creation - Ternary Engine IP Protection")
    print("=" * 60)

    # Ensure directories exist
    TIMESTAMP_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    # Load configuration
    config = load_config()
    print(f"\nUsing hash algorithm: {config['timestamp_config']['hash_algorithm'].upper()}")

    # Get files to timestamp
    files = get_files_to_timestamp(config)

    if not files:
        print("\nNo files found to timestamp. Check your configuration.")
        sys.exit(1)

    print(f"\nFound {len(files)} files to timestamp\n")

    # Create timestamp identifier
    timestamp_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    manifest_file = TIMESTAMP_DIR / f"manifest_{timestamp_id}.json"

    # Create manifest with all file hashes
    manifest = create_manifest(files, manifest_file)

    # Compute hash of the manifest itself
    manifest_hash = compute_manifest_hash(manifest_file)
    print(f"\nManifest SHA512: {manifest_hash}")

    # Save manifest hash to separate file
    hash_file = TIMESTAMP_DIR / f"manifest_{timestamp_id}.sha512"
    with open(hash_file, 'w') as f:
        f.write(f"{manifest_hash}  {manifest_file.name}\n")

    print(f"Hash file saved: {hash_file}")

    # Create OpenTimestamps proof (if client is installed)
    ots_created = create_ots_timestamp(manifest_file)

    # Create summary log
    log_file = LOG_DIR / f"timestamp_log_{timestamp_id}.txt"
    with open(log_file, 'w') as f:
        f.write(f"Timestamp Creation Log\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"Date: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"Files timestamped: {len(manifest['files'])}\n")
        f.write(f"Manifest: {manifest_file.name}\n")
        f.write(f"Manifest SHA512: {manifest_hash}\n")
        f.write(f"OTS Proof Created: {'Yes' if ots_created else 'No'}\n")
        f.write(f"\nFiles included:\n")
        for file_path in sorted(manifest['files'].keys()):
            f.write(f"  - {file_path}\n")

    print(f"\nLog saved: {log_file}")

    print("\n" + "=" * 60)
    print("IMPORTANT: Save these files securely!")
    print("=" * 60)
    print(f"1. Manifest: {manifest_file}")
    print(f"2. Hash: {hash_file}")
    if ots_created:
        print(f"3. OTS Proof: {manifest_file}.ots")
    print(f"4. Log: {log_file}")

    if ots_created:
        print("\n" + "=" * 60)
        print("Next Steps:")
        print("=" * 60)
        print("The timestamp has been submitted to Bitcoin blockchain.")
        print("It will be confirmed within ~15 minutes to 24 hours.")
        print(f"\nVerify later with:")
        print(f"  python timestamp_verify.py {manifest_file.name}")
        print(f"\nOr use OTS client:")
        print(f"  ots verify {manifest_file}.ots")
    else:
        print("\n" + "=" * 60)
        print("Note: Blockchain timestamp not submitted")
        print("=" * 60)
        print("You have SHA512 hashes for all files, but no blockchain proof.")
        print("Install opentimestamps-client to enable blockchain timestamping:")
        print("  pip install opentimestamps-client")

    print("\n✓ Timestamp creation complete!\n")


if __name__ == "__main__":
    main()
