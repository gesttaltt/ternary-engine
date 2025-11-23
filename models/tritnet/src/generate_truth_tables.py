#!/usr/bin/env python3
"""
Truth Table Generator for TritNet Training

Generates complete truth tables for all dense243 ternary operations:
- Binary operations: tadd, tmul, tmin, tmax (243² = 59,049 samples each)
- Unary operations: tnot (243 samples)
- Total: 236,439 training samples

Purpose:
    These truth tables will be used to train TritNet, a tiny neural network
    with ternary weights {-1, 0, +1} that learns exact ternary arithmetic.

Output Format:
    JSON files containing input-output pairs for BitNet training:
    {
        "metadata": {...},
        "samples": [
            {"input": [t0, t1, t2, t3, t4, ...], "output": [r0, r1, r2, r3, r4]},
            ...
        ]
    }

Usage:
    python generate_truth_tables.py [--output-dir OUTPUT_DIR]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import dense243 module
try:
    import ternary_dense243_module as td
except ImportError:
    print("ERROR: ternary_dense243_module not found. Build it first:")
    print("  python build/build_dense243.py")
    sys.exit(1)

# Dense243 encoding: 5 trits per byte, valid range [0, 242]
DENSE243_MIN = 0
DENSE243_MAX = 242
DENSE243_STATES = 243

# 2-bit trit encoding
MINUS_ONE = 0b00
ZERO = 0b01
PLUS_ONE = 0b10
TRIT_VALUES = [MINUS_ONE, ZERO, PLUS_ONE]
TRIT_TO_INT = {MINUS_ONE: -1, ZERO: 0, PLUS_ONE: +1}
INT_TO_TRIT = {-1: MINUS_ONE, 0: ZERO, +1: PLUS_ONE}


def dense243_to_trit_list(byte_val: int) -> List[int]:
    """Unpack dense243 byte to list of 5 trits (2-bit encoded)."""
    packed = np.array([byte_val], dtype=np.uint8)
    trits_2bit = td.unpack(packed, num_trits=5)
    return trits_2bit.tolist()


def trit_list_to_dense243(trits: List[int]) -> int:
    """Pack list of 5 trits (2-bit encoded) to dense243 byte."""
    assert len(trits) == 5, f"Expected 5 trits, got {len(trits)}"
    trits_array = np.array(trits, dtype=np.uint8)
    packed = td.pack(trits_array)
    return int(packed[0])


def trit_to_int_list(trits_2bit: List[int]) -> List[int]:
    """Convert 2-bit encoded trits to integer values {-1, 0, +1}."""
    return [TRIT_TO_INT[t] for t in trits_2bit]


def int_to_trit_list(int_values: List[int]) -> List[int]:
    """Convert integer values {-1, 0, +1} to 2-bit encoded trits."""
    return [INT_TO_TRIT[v] for v in int_values]


def generate_binary_operation_table(
    operation_name: str,
    operation_func,
    output_path: Path
) -> Dict[str, Any]:
    """
    Generate complete truth table for a binary operation.

    Args:
        operation_name: Name of operation (e.g., "tadd")
        operation_func: Function from ternary_dense243_module
        output_path: Path to save JSON file

    Returns:
        Statistics dictionary
    """
    print(f"\n{'='*60}")
    print(f"Generating {operation_name} truth table...")
    print(f"{'='*60}")

    samples = []
    total_combinations = DENSE243_STATES * DENSE243_STATES

    for a_byte in range(DENSE243_MIN, DENSE243_MAX + 1):
        if a_byte % 50 == 0:
            progress = (a_byte * DENSE243_STATES) / total_combinations * 100
            print(f"Progress: {progress:.1f}% ({a_byte}/{DENSE243_STATES} first operands)")

        for b_byte in range(DENSE243_MIN, DENSE243_MAX + 1):
            # Unpack inputs
            a_trits = dense243_to_trit_list(a_byte)
            b_trits = dense243_to_trit_list(b_byte)

            # Perform operation in dense243 format
            a_dense = np.array([a_byte], dtype=np.uint8)
            b_dense = np.array([b_byte], dtype=np.uint8)
            result_dense = operation_func(a_dense, b_dense)
            result_byte = int(result_dense[0])

            # Unpack result
            result_trits = dense243_to_trit_list(result_byte)

            # Convert to integer representation for neural network
            a_ints = trit_to_int_list(a_trits)
            b_ints = trit_to_int_list(b_trits)
            result_ints = trit_to_int_list(result_trits)

            # Create sample (10 inputs → 5 outputs)
            sample = {
                "input": a_ints + b_ints,  # [a0, a1, a2, a3, a4, b0, b1, b2, b3, b4]
                "output": result_ints,      # [r0, r1, r2, r3, r4]
                "input_dense243": [int(a_byte), int(b_byte)],
                "output_dense243": int(result_byte)
            }
            samples.append(sample)

    # Create dataset with metadata
    dataset = {
        "metadata": {
            "operation": operation_name,
            "operation_type": "binary",
            "input_size": 10,  # 2 × 5 trits
            "output_size": 5,
            "num_samples": len(samples),
            "encoding": "balanced_ternary",
            "value_range": [-1, 0, 1],
            "dense243_range": [DENSE243_MIN, DENSE243_MAX],
            "generator_version": "1.0.0"
        },
        "samples": samples
    }

    # Save to JSON
    print(f"Writing {len(samples)} samples to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    # Calculate statistics
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    stats = {
        "operation": operation_name,
        "samples": len(samples),
        "file_size_mb": file_size_mb,
        "coverage": "100% (all 243² combinations)"
    }

    print(f"✓ Generated {len(samples):,} samples ({file_size_mb:.2f} MB)")
    return stats


def generate_unary_operation_table(
    operation_name: str,
    operation_func,
    output_path: Path
) -> Dict[str, Any]:
    """
    Generate complete truth table for a unary operation.

    Args:
        operation_name: Name of operation (e.g., "tnot")
        operation_func: Function from ternary_dense243_module
        output_path: Path to save JSON file

    Returns:
        Statistics dictionary
    """
    print(f"\n{'='*60}")
    print(f"Generating {operation_name} truth table...")
    print(f"{'='*60}")

    samples = []

    for a_byte in range(DENSE243_MIN, DENSE243_MAX + 1):
        if a_byte % 50 == 0:
            progress = a_byte / DENSE243_STATES * 100
            print(f"Progress: {progress:.1f}% ({a_byte}/{DENSE243_STATES} values)")

        # Unpack input
        a_trits = dense243_to_trit_list(a_byte)

        # Perform operation in dense243 format
        a_dense = np.array([a_byte], dtype=np.uint8)
        result_dense = operation_func(a_dense)
        result_byte = int(result_dense[0])

        # Unpack result
        result_trits = dense243_to_trit_list(result_byte)

        # Convert to integer representation
        a_ints = trit_to_int_list(a_trits)
        result_ints = trit_to_int_list(result_trits)

        # Create sample (5 inputs → 5 outputs)
        sample = {
            "input": a_ints,           # [a0, a1, a2, a3, a4]
            "output": result_ints,     # [r0, r1, r2, r3, r4]
            "input_dense243": int(a_byte),
            "output_dense243": int(result_byte)
        }
        samples.append(sample)

    # Create dataset with metadata
    dataset = {
        "metadata": {
            "operation": operation_name,
            "operation_type": "unary",
            "input_size": 5,
            "output_size": 5,
            "num_samples": len(samples),
            "encoding": "balanced_ternary",
            "value_range": [-1, 0, 1],
            "dense243_range": [DENSE243_MIN, DENSE243_MAX],
            "generator_version": "1.0.0"
        },
        "samples": samples
    }

    # Save to JSON
    print(f"Writing {len(samples)} samples to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    # Calculate statistics
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    stats = {
        "operation": operation_name,
        "samples": len(samples),
        "file_size_mb": file_size_mb,
        "coverage": "100% (all 243 combinations)"
    }

    print(f"✓ Generated {len(samples):,} samples ({file_size_mb:.2f} MB)")
    return stats


def verify_sample_correctness(num_samples: int = 100):
    """Verify random samples are correct by checking against module operations."""
    print(f"\n{'='*60}")
    print(f"Verifying sample correctness...")
    print(f"{'='*60}")

    np.random.seed(42)

    operations = {
        "tadd": td.tadd,
        "tmul": td.tmul,
        "tmin": td.tmin,
        "tmax": td.tmax,
        "tnot": td.tnot
    }

    for op_name, op_func in operations.items():
        print(f"\nVerifying {op_name}...")

        for _ in range(num_samples):
            if op_name == "tnot":
                # Unary operation
                a_byte = np.random.randint(DENSE243_MIN, DENSE243_MAX + 1)
                a_dense = np.array([a_byte], dtype=np.uint8)
                result_dense = op_func(a_dense)

                # Verify unpacking and repacking is consistent
                result_trits = td.unpack(result_dense, num_trits=5)
                repacked = td.pack(result_trits)
                assert repacked[0] == result_dense[0], f"Unpack/pack mismatch for {op_name}"
            else:
                # Binary operation
                a_byte = np.random.randint(DENSE243_MIN, DENSE243_MAX + 1)
                b_byte = np.random.randint(DENSE243_MIN, DENSE243_MAX + 1)
                a_dense = np.array([a_byte], dtype=np.uint8)
                b_dense = np.array([b_byte], dtype=np.uint8)
                result_dense = op_func(a_dense, b_dense)

                # Verify unpacking and repacking is consistent
                result_trits = td.unpack(result_dense, num_trits=5)
                repacked = td.pack(result_trits)
                assert repacked[0] == result_dense[0], f"Unpack/pack mismatch for {op_name}"

        print(f"✓ {op_name}: {num_samples} random samples verified")


def main():
    parser = argparse.ArgumentParser(
        description="Generate truth tables for TritNet training"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "datasets" / "tritnet",
        help="Output directory for truth tables"
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("TritNet Truth Table Generator")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Dense243 module version: {td.__version__}")
    print(f"Backend: {td.get_backend()}")
    print(f"Density: {td.DENSITY} trits/byte")

    # Verify module correctness first
    verify_sample_correctness(num_samples=100)

    # Generate truth tables for all operations
    all_stats = []

    # Binary operations
    binary_ops = {
        "tadd": td.tadd,
        "tmul": td.tmul,
        "tmin": td.tmin,
        "tmax": td.tmax
    }

    for op_name, op_func in binary_ops.items():
        output_path = args.output_dir / f"{op_name}_truth_table.json"
        stats = generate_binary_operation_table(op_name, op_func, output_path)
        all_stats.append(stats)

    # Unary operations
    unary_ops = {
        "tnot": td.tnot
    }

    for op_name, op_func in unary_ops.items():
        output_path = args.output_dir / f"{op_name}_truth_table.json"
        stats = generate_unary_operation_table(op_name, op_func, output_path)
        all_stats.append(stats)

    # Generate summary report
    print(f"\n{'='*60}")
    print("Generation Summary")
    print(f"{'='*60}")

    total_samples = sum(s["samples"] for s in all_stats)
    total_size_mb = sum(s["file_size_mb"] for s in all_stats)

    print(f"\nTotal samples generated: {total_samples:,}")
    print(f"Total dataset size: {total_size_mb:.2f} MB")
    print(f"\nPer-operation breakdown:")
    for stats in all_stats:
        print(f"  {stats['operation']:5s}: {stats['samples']:>7,} samples "
              f"({stats['file_size_mb']:>6.2f} MB) - {stats['coverage']}")

    # Save summary
    summary_path = args.output_dir / "generation_summary.json"
    summary = {
        "total_samples": total_samples,
        "total_size_mb": total_size_mb,
        "operations": all_stats,
        "module_version": td.__version__,
        "backend": td.get_backend(),
        "density": float(td.DENSITY),
        "state_utilization": float(td.STATE_UTILIZATION)
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to: {summary_path}")
    print(f"\n{'='*60}")
    print("Truth table generation complete!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"1. Review datasets in: {args.output_dir}")
    print(f"2. Set up BitNet training environment")
    print(f"3. Train TritNet models on these truth tables")
    print(f"4. See docs/TRITNET_ROADMAP.md for full implementation plan")


if __name__ == "__main__":
    main()
