#!/usr/bin/env python3
"""
TritNet Workflow Orchestration Script

Unified interface for TritNet training pipeline:
- Generate truth tables
- Train models
- Validate results
- Make Go/No-Go decisions

Usage:
    # Full pipeline
    python models/tritnet/run_tritnet.py --all

    # Individual phases
    python models/tritnet/run_tritnet.py --phase datasets
    python models/tritnet/run_tritnet.py --phase training
    python models/tritnet/run_tritnet.py --phase validation

    # Specific operations
    python models/tritnet/run_tritnet.py --train tnot
    python models/tritnet/run_tritnet.py --train-all

Example workflows:
    # Proof-of-concept (Phase 2A)
    python models/tritnet/run_tritnet.py --phase datasets --train tnot

    # Full training (Phase 2B)
    python models/tritnet/run_tritnet.py --all
"""

import argparse
import sys
import subprocess
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent


def run_command(cmd: list, description: str) -> int:
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with code {result.returncode}")
        return result.returncode

    print(f"\n✓ {description} completed successfully")
    return 0


def check_prerequisites():
    """Check if required dependencies are installed."""
    print("\n" + "="*70)
    print("Checking Prerequisites")
    print("="*70)

    errors = []

    # Check Python packages
    required_packages = ['torch', 'numpy']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"❌ {package} not installed")
            errors.append(f"pip install {package}")

    # Check if dense243 module is built
    try:
        import ternary_dense243_module as td
        print(f"✓ ternary_dense243_module built (version {td.__version__})")
    except ImportError:
        print("❌ ternary_dense243_module not built")
        errors.append("python build/build_dense243.py")

    if errors:
        print("\n" + "="*70)
        print("Missing Prerequisites")
        print("="*70)
        print("Run the following commands to fix:\n")
        for cmd in errors:
            print(f"  {cmd}")
        print()
        return False

    print("\n✓ All prerequisites satisfied")
    return True


def phase_datasets():
    """Generate truth table datasets."""
    script = PROJECT_ROOT / "models" / "tritnet" / "src" / "generate_truth_tables.py"

    if not script.exists():
        print(f"❌ Script not found: {script}")
        return 1

    # Check if datasets already exist
    datasets_dir = PROJECT_ROOT / "models" / "datasets" / "tritnet"
    summary_file = datasets_dir / "generation_summary.json"

    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)

        # Found 2026-08-12: this used to trust summary_file.exists() alone
        # as proof the actual *_truth_table.json files were present. On
        # this checkout they weren't (only generation_summary.json and
        # README.md exist in models/datasets/tritnet/) — presumably a
        # previous generation run wrote the summary but not, or no longer
        # has, the per-operation files. That silently skipped regeneration
        # and left training with nothing to load. Verify each file the
        # summary itself claims to have produced actually exists before
        # trusting it.
        missing = [
            op['operation'] for op in summary['operations']
            if not (datasets_dir / f"{op['operation']}_truth_table.json").exists()
        ]

        if not missing:
            print("\n" + "="*70)
            print("Truth Tables Already Generated")
            print("="*70)
            print(f"Total samples: {summary['total_samples']:,}")
            print(f"Total size: {summary['total_size_mb']:.2f} MB")
            print(f"Operations: {', '.join([op['operation'] for op in summary['operations']])}")
            print("\nSkipping regeneration. To regenerate, delete models/datasets/tritnet/ first.")
            return 0

        print(f"⚠ generation_summary.json exists but is missing truth tables for: {', '.join(missing)}")
        print("  Regenerating all truth tables.")

    return run_command(
        [sys.executable, str(script)],
        "Generating Truth Table Datasets"
    )


def phase_training(operation: str = None, train_all: bool = False, use_phase2b: bool = False):
    """Train TritNet models.

    use_phase2b=True runs models/tritnet/train_phase2b.py — the QAT
    pipeline that actually produced the documented Phase 2B GO results
    (tadd 100%, tmul 99.5%, tmin 99.9%, tmax 99.9%; see phase2b/*/result.json
    and CLAUDE.md). Found 2026-08-12: before this flag existed, this
    function only ever called src/train_tritnet.py, an older MSE-based
    pipeline with no QAT-during-training and no relationship to the actual
    GO decision — running `run_tritnet.py --all` (the documented top-level
    workflow) trained and validated a known-inferior model and never
    touched the real Phase 2B pipeline at all. train_phase2b.py takes no
    CLI arguments (it always trains all 4 binary ops, tadd/tmul/tmin/tmax,
    with its own resume-if-already-passed logic), so `operation`/`train_all`
    are ignored when use_phase2b is set.
    """
    if use_phase2b:
        script = PROJECT_ROOT / "models" / "tritnet" / "train_phase2b.py"
        if not script.exists():
            print(f"❌ Script not found: {script}")
            return 1
        return run_command([sys.executable, str(script)], "Training Phase 2B (QAT, all 4 binary ops)")

    script = PROJECT_ROOT / "models" / "tritnet" / "src" / "train_tritnet.py"

    if not script.exists():
        print(f"❌ Script not found: {script}")
        return 1

    # Check if datasets exist
    datasets_dir = PROJECT_ROOT / "models" / "datasets" / "tritnet"
    if not (datasets_dir / "generation_summary.json").exists():
        print("❌ Truth tables not generated. Run --phase datasets first.")
        return 1

    # Build command
    cmd = [sys.executable, str(script)]

    if train_all:
        cmd.append("--all")
        description = "Training All TritNet Models"
    elif operation:
        cmd.extend(["--operation", operation])
        description = f"Training TritNet Model ({operation})"
    else:
        print("❌ Must specify either --train <op> or --train-all")
        return 1

    return run_command(cmd, description)


def phase_validation():
    """Validate trained models and make Go/No-Go decision."""
    models_dir = PROJECT_ROOT / "models" / "tritnet"

    if not models_dir.exists():
        print("❌ No models found. Run training first.")
        return 1

    print("\n" + "="*70)
    print("TritNet Model Validation")
    print("="*70)

    # Prefer models/tritnet/phase2b/<op>/result.json — the real, structured
    # results from train_phase2b.py (the QAT pipeline that actually achieved
    # the documented Phase 2B GO decision) — over the legacy *_history.json
    # glob below. Found 2026-08-12: this function previously read ONLY the
    # legacy files, which are stale/from a different, inferior pipeline
    # (verified on disk: tritnet_tadd_history.json records 15.8% final
    # accuracy, while phase2b/tadd/result.json — the actual GO result —
    # records 100%). Silently produced a NO-GO verdict contradicting
    # CLAUDE.md's documented "Phase 2B GO" status.
    accuracy = {}   # op -> accuracy in [0, 1]
    source = {}     # op -> which file the accuracy came from, for the table

    phase2b_dir = models_dir / "phase2b"
    if phase2b_dir.exists():
        for result_file in sorted(phase2b_dir.glob("*/result.json")):
            with open(result_file) as f:
                data = json.load(f)
            op = data.get('op') or result_file.parent.name
            accuracy[op] = data['best_acc']
            source[op] = f"phase2b/{op}/result.json"

    # Legacy history files: only used for operations phase2b didn't cover
    # (e.g. tnot, validated separately in Phase 2A) — never to override a
    # phase2b result, which is the higher-confidence source.
    history_files = list(models_dir.glob("*_history.json"))
    for history_file in history_files:
        with open(history_file) as f:
            data = json.load(f)
        op = data['metadata']['operation']
        if op not in accuracy:
            accuracy[op] = data['metadata']['final_accuracy']
            source[op] = history_file.name

    if not accuracy:
        print("❌ No training histories or phase2b results found")
        return 1

    # Print results table
    print("\nOperation | Accuracy | Source")
    print("-" * 70)

    for operation in sorted(accuracy.keys()):
        acc = accuracy[operation]
        print(f"{operation:5s}     | {acc*100:6.2f}%  | {source[operation]}")

    # Go/No-Go decision
    print("\n" + "="*70)
    print("Go/No-Go Decision")
    print("="*70)

    successful = [op for op, acc in accuracy.items() if acc > 0.99]
    perfect = [op for op, acc in accuracy.items() if acc >= 0.9999]

    print(f"\nOperations with >99% accuracy: {len(successful)}/{len(accuracy)}")
    print(f"Operations with 100% accuracy: {len(perfect)}/{len(accuracy)}")

    if successful:
        print(f"  >99%: {', '.join(successful)}")
    if perfect:
        print(f"  100%: {', '.join(perfect)}")

    print("\n" + "="*70)

    if len(successful) >= 3 and len(perfect) >= 1:
        print("✅ GO: Criteria met!")
        print("   - At least 3 operations achieved >99% accuracy")
        print("   - At least 1 operation achieved 100% accuracy")
        print("   - TritNet proves exact arithmetic is learnable!")
        print("\nNext: Proceed to Phase 3 (C++ Integration)")
        return 0
    elif len(successful) >= 1:
        print("⚠️  PARTIAL SUCCESS")
        print("   - Some operations learned successfully")
        print("   - Consider architecture adjustments")
        print("\nNext: Analyze failure modes and adjust")
        return 2
    else:
        print("❌ NO-GO: Criteria not met")
        print("   - No operations achieved >99% accuracy")
        print("\nNext: Investigate why NNs cannot learn exact arithmetic")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="TritNet workflow orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python models/tritnet/run_tritnet.py --all

  # Individual phases
  python models/tritnet/run_tritnet.py --phase datasets
  python models/tritnet/run_tritnet.py --phase training --train-all
  python models/tritnet/run_tritnet.py --phase validation

  # Proof-of-concept (tnot only)
  python models/tritnet/run_tritnet.py --train tnot

  # Specific operation
  python models/tritnet/run_tritnet.py --train tadd
        """
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run full pipeline: datasets → training → validation"
    )

    parser.add_argument(
        "--phase",
        choices=["datasets", "training", "validation"],
        help="Run specific phase"
    )

    parser.add_argument(
        "--train",
        choices=["tnot", "tadd", "tmul", "tmin", "tmax"],
        help="Train specific operation"
    )

    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Train all operations"
    )

    parser.add_argument(
        "--phase2b",
        action="store_true",
        help=(
            "Use train_phase2b.py (QAT pipeline, the one that actually "
            "produced the documented Phase 2B GO results) instead of the "
            "legacy src/train_tritnet.py. Applies to --phase training and "
            "--all; ignores --train/--train-all since train_phase2b.py "
            "always trains all 4 binary ops itself."
        )
    )

    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip prerequisite checks"
    )

    args = parser.parse_args()

    # Print header
    print("\n" + "="*70)
    print("TritNet Workflow Orchestration")
    print("="*70)
    print(f"Project root: {PROJECT_ROOT}")

    # Check prerequisites
    if not args.skip_checks:
        if not check_prerequisites():
            print("\n❌ Prerequisites not satisfied. Fix errors above.")
            return 1

    # Determine what to run
    exit_code = 0

    if args.all:
        # Full pipeline
        print("\n" + "="*70)
        print("Running Full TritNet Pipeline")
        print("="*70)
        print("Phases: datasets → training → validation")

        exit_code = phase_datasets()
        if exit_code != 0:
            return exit_code

        exit_code = phase_training(train_all=True, use_phase2b=args.phase2b)
        if exit_code != 0:
            return exit_code

        exit_code = phase_validation()

    elif args.phase:
        # Specific phase
        if args.phase == "datasets":
            exit_code = phase_datasets()

        elif args.phase == "training":
            if args.phase2b:
                exit_code = phase_training(use_phase2b=True)
            elif args.train_all:
                exit_code = phase_training(train_all=True)
            elif args.train:
                exit_code = phase_training(operation=args.train)
            else:
                print("❌ Must specify --train <op>, --train-all, or --phase2b with --phase training")
                exit_code = 1

        elif args.phase == "validation":
            exit_code = phase_validation()

    elif args.train:
        # Train specific operation (generate datasets if needed)
        datasets_dir = PROJECT_ROOT / "models" / "datasets" / "tritnet"
        if not (datasets_dir / "generation_summary.json").exists():
            print("\n⚠️  Truth tables not found. Generating first...")
            exit_code = phase_datasets()
            if exit_code != 0:
                return exit_code

        exit_code = phase_training(operation=args.train)

    elif args.train_all:
        # Train all operations (generate datasets if needed)
        datasets_dir = PROJECT_ROOT / "models" / "datasets" / "tritnet"
        if not (datasets_dir / "generation_summary.json").exists():
            print("\n⚠️  Truth tables not found. Generating first...")
            exit_code = phase_datasets()
            if exit_code != 0:
                return exit_code

        exit_code = phase_training(train_all=True)

    else:
        parser.print_help()
        return 1

    # Summary
    print("\n" + "="*70)
    print("Workflow Complete")
    print("="*70)

    if exit_code == 0:
        print("✓ All steps completed successfully")
    elif exit_code == 2:
        print("⚠️  Partial success (see details above)")
    else:
        print("❌ Workflow failed (see errors above)")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
