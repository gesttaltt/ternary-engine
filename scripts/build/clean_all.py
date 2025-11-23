"""
clean_all.py - Comprehensive cleanup utility for Ternary Engine build artifacts

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Cleans all build artifacts, temporary files, cached data, and deprecated files
to ensure fresh builds and accurate benchmarks.

Cleanup targets:
- Build artifacts (build/artifacts/*)
- Setuptools temp directories (build/temp.*, build/lib.*)
- PGO profile data (all locations)
- Root compiled modules (*.pyd, *.so)
- Deprecated build scripts (build/scripts/)
- Python cache (__pycache__, *.pyc)
- Benchmark results (benchmarks/results/)
- External temp dirs (C:/Temp/ternary_*) with --deep

Usage:
    python scripts/build/clean_all.py                    # Clean all build artifacts
    python scripts/build/clean_all.py --keep-latest      # Keep latest builds
    python scripts/build/clean_all.py --keep-results 5   # Keep last 5 benchmark results
    python scripts/build/clean_all.py --dry-run          # Show what would be deleted
    python scripts/build/clean_all.py --deep             # Include external temp dirs
"""

import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Get project root (script is in scripts/build/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


class CleanupManager:
    """Manages cleanup of build artifacts and cached data"""

    def __init__(self, dry_run=False, verbose=True):
        self.dry_run = dry_run
        self.verbose = verbose
        self.deleted_count = 0
        self.deleted_size = 0

    def print_section(self, title):
        """Print section header"""
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  {title}")
            print('='*70)

    def remove_path(self, path: Path, description: str = None):
        """Remove a file or directory"""
        if not path.exists():
            return

        # Calculate size before deletion
        if path.is_file():
            size = path.stat().st_size
        else:
            size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())

        desc = description or str(path.relative_to(PROJECT_ROOT))

        if self.dry_run:
            if self.verbose:
                print(f"  [DRY RUN] Would delete: {desc} ({self._format_size(size)})")
            # Still track stats in dry-run mode for summary
            self.deleted_count += 1
            self.deleted_size += size
        else:
            try:
                if path.is_file():
                    path.unlink()
                else:
                    shutil.rmtree(path)
                if self.verbose:
                    print(f"  [DELETED] {desc} ({self._format_size(size)})")
                self.deleted_count += 1
                self.deleted_size += size
            except Exception as e:
                if self.verbose:
                    print(f"  [ERROR] Failed to delete {desc}: {e}")

    def _format_size(self, size_bytes):
        """Format size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def clean_build_artifacts(self, keep_latest=False):
        """Clean build artifacts directory"""
        self.print_section("Cleaning Build Artifacts")

        artifacts_dir = PROJECT_ROOT / "build" / "artifacts"
        if not artifacts_dir.exists():
            if self.verbose:
                print("  No build artifacts found")
            return

        # Clean each build type
        for build_type_dir in artifacts_dir.iterdir():
            if not build_type_dir.is_dir():
                continue

            # Skip pgo_data directory (handled separately)
            if build_type_dir.name == "pgo_data":
                continue

            if self.verbose:
                print(f"\n  Build type: {build_type_dir.name}")

            # Get all timestamped builds (sorted by name, which is timestamp)
            timestamped_builds = sorted([
                d for d in build_type_dir.iterdir()
                if d.is_dir() and d.name != "latest"
            ])

            if keep_latest and len(timestamped_builds) > 1:
                # Keep the most recent timestamped build
                for build_dir in timestamped_builds[:-1]:
                    self.remove_path(build_dir, f"{build_type_dir.name}/{build_dir.name}")
                if self.verbose:
                    print(f"  [KEPT] {build_type_dir.name}/{timestamped_builds[-1].name} (latest)")
            else:
                # Remove all timestamped builds
                for build_dir in timestamped_builds:
                    self.remove_path(build_dir, f"{build_type_dir.name}/{build_dir.name}")

            # Remove 'latest' symlink/directory
            if not keep_latest:
                latest_dir = build_type_dir / "latest"
                if latest_dir.exists():
                    self.remove_path(latest_dir, f"{build_type_dir.name}/latest")

    def clean_setuptools_dirs(self):
        """Clean setuptools build directories"""
        self.print_section("Cleaning Setuptools Directories")

        # Clean build/temp.* directories (can have very deep paths)
        build_dir = PROJECT_ROOT / "build"
        if build_dir.exists():
            for temp_dir in build_dir.glob("temp.*"):
                self.remove_path(temp_dir, f"build/{temp_dir.name}")

            for lib_dir in build_dir.glob("lib.*"):
                self.remove_path(lib_dir, f"build/{lib_dir.name}")
        else:
            if self.verbose:
                print("  No setuptools directories found")

    def clean_pgo_data(self):
        """Clean PGO profile data from both locations"""
        self.print_section("Cleaning PGO Data")

        cleaned = False

        # Location 1: build/artifacts/pgo/pgo_data (build_pgo.py)
        pgo_data_1 = PROJECT_ROOT / "build" / "artifacts" / "pgo" / "pgo_data"
        if pgo_data_1.exists():
            self.remove_path(pgo_data_1, "build/artifacts/pgo/pgo_data")
            cleaned = True

        # Location 2: build/artifacts/pgo_data (future unified location)
        pgo_data_2 = PROJECT_ROOT / "build" / "artifacts" / "pgo_data"
        if pgo_data_2.exists():
            self.remove_path(pgo_data_2, "build/artifacts/pgo_data")
            cleaned = True

        # Location 3: pgo_data/ in project root (build_pgo_unified.py)
        pgo_data_3 = PROJECT_ROOT / "pgo_data"
        if pgo_data_3.exists():
            self.remove_path(pgo_data_3, "pgo_data")
            cleaned = True

        if not cleaned and self.verbose:
            print("  No PGO data found")

    def clean_root_modules(self):
        """Clean compiled modules from project root"""
        self.print_section("Cleaning Root Module Files")

        found = False

        # Remove .pyd files (Windows)
        for pyd_file in PROJECT_ROOT.glob("*.pyd"):
            self.remove_path(pyd_file, pyd_file.name)
            found = True

        # Remove .so files (Linux/macOS)
        for so_file in PROJECT_ROOT.glob("*.so"):
            self.remove_path(so_file, so_file.name)
            found = True

        if not found and self.verbose:
            print("  No compiled modules in project root")

    def clean_external_temp_dirs(self):
        """Clean external temporary directories (Windows)"""
        self.print_section("Cleaning External Temp Directories")

        temp_dirs = [
            Path("C:/Temp/ternary_build"),
            Path("C:/Temp/ternary_pgo_inst"),
            Path("C:/Temp/ternary_pgo_opt"),
        ]

        found = False
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                self.remove_path(temp_dir, str(temp_dir))
                found = True

        if not found and self.verbose:
            print("  No external temp directories found")

    def clean_benchmark_results(self, keep_count=0):
        """Clean old benchmark results"""
        self.print_section(f"Cleaning Benchmark Results (keeping last {keep_count})")

        results_dir = PROJECT_ROOT / "benchmarks" / "results"
        if not results_dir.exists():
            if self.verbose:
                print("  No benchmark results found")
            return

        # Get all result files sorted by modification time
        json_files = sorted(
            results_dir.rglob("bench_results_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        csv_files = sorted(
            results_dir.rglob("bench_results_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        # Keep only the most recent results
        for i, json_file in enumerate(json_files):
            if i >= keep_count:
                self.remove_path(json_file, f"benchmarks/results/{json_file.name}")

        for i, csv_file in enumerate(csv_files):
            if i >= keep_count:
                self.remove_path(csv_file, f"benchmarks/results/{csv_file.name}")

        # Clean empty subdirectories
        for subdir in results_dir.iterdir():
            if subdir.is_dir() and not any(subdir.iterdir()):
                self.remove_path(subdir, f"benchmarks/results/{subdir.name}")

        if keep_count > 0 and self.verbose:
            kept_count = min(keep_count, len(json_files))
            if kept_count > 0:
                print(f"\n  [KEPT] {kept_count} most recent result file(s)")

    def clean_deprecated_scripts(self):
        """Clean deprecated build/scripts directory"""
        self.print_section("Cleaning Deprecated Build Scripts")

        # Old location of build scripts (now in scripts/build/)
        deprecated_dir = PROJECT_ROOT / "build" / "scripts"
        if deprecated_dir.exists():
            self.remove_path(deprecated_dir, "build/scripts (deprecated)")
        else:
            if self.verbose:
                print("  No deprecated build scripts found")

    def clean_python_cache(self):
        """Clean Python cache files and directories"""
        self.print_section("Cleaning Python Cache")

        found = False

        # Remove __pycache__ directories
        for pycache_dir in PROJECT_ROOT.rglob("__pycache__"):
            # Skip virtual environments
            if "venv" in str(pycache_dir) or ".venv" in str(pycache_dir):
                continue
            self.remove_path(pycache_dir, str(pycache_dir.relative_to(PROJECT_ROOT)))
            found = True

        # Remove .pyc files
        for pyc_file in PROJECT_ROOT.rglob("*.pyc"):
            # Skip virtual environments
            if "venv" in str(pyc_file) or ".venv" in str(pyc_file):
                continue
            self.remove_path(pyc_file, str(pyc_file.relative_to(PROJECT_ROOT)))
            found = True

        if not found and self.verbose:
            print("  No Python cache files found")

    def print_summary(self):
        """Print cleanup summary"""
        print(f"\n{'='*70}")
        print("  CLEANUP SUMMARY")
        print('='*70)
        if self.dry_run:
            print(f"\n  [DRY RUN] Would delete {self.deleted_count} items")
            print(f"  [DRY RUN] Would free ~{self._format_size(self.deleted_size)}")
        else:
            print(f"\n  Deleted: {self.deleted_count} items")
            print(f"  Freed:   {self._format_size(self.deleted_size)}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive cleanup utility for Ternary Engine'
    )
    parser.add_argument('--keep-latest', action='store_true',
                       help='Keep the latest timestamped build for each build type')
    parser.add_argument('--keep-results', type=int, default=0, metavar='N',
                       help='Keep the N most recent benchmark result files (default: 0)')
    parser.add_argument('--deep', action='store_true',
                       help='Include external temp directories (C:/Temp/ternary_*)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without actually deleting')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')

    args = parser.parse_args()

    # Create cleanup manager
    cleaner = CleanupManager(dry_run=args.dry_run, verbose=not args.quiet)

    # Print header
    if not args.quiet:
        print("\n" + "="*70)
        print("  TERNARY ENGINE - CLEANUP UTILITY")
        if args.dry_run:
            print("  [DRY RUN MODE - No files will be deleted]")
        print("="*70)
        print(f"\nProject root: {PROJECT_ROOT}")
        print(f"Timestamp: {datetime.now().isoformat()}")

    # Execute cleanup operations
    cleaner.clean_build_artifacts(keep_latest=args.keep_latest)
    cleaner.clean_setuptools_dirs()
    cleaner.clean_pgo_data()
    cleaner.clean_root_modules()
    cleaner.clean_deprecated_scripts()
    cleaner.clean_python_cache()
    cleaner.clean_benchmark_results(keep_count=args.keep_results)

    if args.deep:
        cleaner.clean_external_temp_dirs()

    # Print summary
    if not args.quiet:
        cleaner.print_summary()

    return 0


if __name__ == "__main__":
    sys.exit(main())
