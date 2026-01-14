"""
data.py - Operation LUT Generation and Dataset for 3-vae-gemm-v1

Uses the SIMD engine to precompute all ternary operation results efficiently.
For 9-trit values: 19,683 values × 19,683 values × 4 operations = ~1.5B results
We sample strategically to make training tractable.

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import time
import sys

# Try to import SIMD engine
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    import ternary_simd_engine as simd
    HAS_SIMD = True
except ImportError:
    HAS_SIMD = False
    print("Warning: SIMD engine not available, using Python fallback")


# =============================================================================
# TERNARY OPERATIONS (Python fallback)
# =============================================================================

def index_to_trits(idx: int, num_trits: int = 9) -> np.ndarray:
    """Convert index to balanced ternary representation."""
    trits = np.zeros(num_trits, dtype=np.int8)
    for i in range(num_trits):
        trits[i] = (idx % 3) - 1
        idx //= 3
    return trits


def trits_to_index(trits: np.ndarray) -> int:
    """Convert balanced ternary to index."""
    idx = 0
    for i in range(len(trits) - 1, -1, -1):
        idx = idx * 3 + int(trits[i]) + 1
    return idx


def py_tadd(a: int, b: int, num_trits: int = 9) -> int:
    """Pure Python balanced ternary addition."""
    trits_a = index_to_trits(a, num_trits)
    trits_b = index_to_trits(b, num_trits)
    result = np.zeros(num_trits, dtype=np.int8)
    carry = 0
    for i in range(num_trits):
        s = int(trits_a[i]) + int(trits_b[i]) + carry
        if s > 1:
            result[i] = s - 3
            carry = 1
        elif s < -1:
            result[i] = s + 3
            carry = -1
        else:
            result[i] = s
            carry = 0
    return trits_to_index(result)


def py_tmul(a: int, b: int, num_trits: int = 9) -> int:
    """Pure Python element-wise ternary multiplication."""
    trits_a = index_to_trits(a, num_trits)
    trits_b = index_to_trits(b, num_trits)
    result = trits_a * trits_b
    return trits_to_index(result)


def py_tmin(a: int, b: int, num_trits: int = 9) -> int:
    """Pure Python element-wise ternary minimum."""
    trits_a = index_to_trits(a, num_trits)
    trits_b = index_to_trits(b, num_trits)
    result = np.minimum(trits_a, trits_b)
    return trits_to_index(result)


def py_tmax(a: int, b: int, num_trits: int = 9) -> int:
    """Pure Python element-wise ternary maximum."""
    trits_a = index_to_trits(a, num_trits)
    trits_b = index_to_trits(b, num_trits)
    result = np.maximum(trits_a, trits_b)
    return trits_to_index(result)


# =============================================================================
# SIMD-ACCELERATED OPERATIONS
# =============================================================================

def index_to_trits_unbalanced(idx: int, num_trits: int = 9) -> np.ndarray:
    """Convert index to unbalanced ternary {0, 1, 2} for SIMD engine."""
    trits = np.zeros(num_trits, dtype=np.uint8)
    for i in range(num_trits):
        trits[i] = idx % 3
        idx //= 3
    return trits


def trits_to_index_unbalanced(trits: np.ndarray) -> int:
    """Convert unbalanced ternary {0, 1, 2} to index."""
    idx = 0
    for i in range(len(trits) - 1, -1, -1):
        idx = idx * 3 + int(trits[i])
    return idx


def simd_batch_operation(
    indices_a: np.ndarray,
    indices_b: np.ndarray,
    operation: str,
    num_trits: int = 9
) -> np.ndarray:
    """
    Perform batch ternary operations using SIMD engine.

    Args:
        indices_a: Array of first operand indices
        indices_b: Array of second operand indices
        operation: One of 'add', 'mul', 'min', 'max'
        num_trits: Number of trits per value

    Returns:
        Array of result indices
    """
    n = len(indices_a)

    if HAS_SIMD:
        # Convert indices to UNBALANCED trit arrays {0, 1, 2} for SIMD engine
        # Note: SIMD engine expects uint8 with values 0, 1, 2
        trits_a = np.zeros((n, num_trits), dtype=np.uint8)
        trits_b = np.zeros((n, num_trits), dtype=np.uint8)

        for i in range(n):
            trits_a[i] = index_to_trits_unbalanced(indices_a[i], num_trits)
            trits_b[i] = index_to_trits_unbalanced(indices_b[i], num_trits)

        # Flatten for SIMD
        flat_a = trits_a.flatten()
        flat_b = trits_b.flatten()

        # Use SIMD engine (operates on unbalanced encoding)
        if operation == 'add':
            flat_result = simd.tadd(flat_a, flat_b)
        elif operation == 'mul':
            flat_result = simd.tmul(flat_a, flat_b)
        elif operation == 'min':
            flat_result = simd.tmin(flat_a, flat_b)
        elif operation == 'max':
            flat_result = simd.tmax(flat_a, flat_b)
        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Reshape and convert back to indices (result is also unbalanced)
        result_trits = flat_result.reshape(n, num_trits)
        results = np.array([trits_to_index_unbalanced(result_trits[i]) for i in range(n)])

    else:
        # Python fallback
        op_funcs = {
            'add': py_tadd,
            'mul': py_tmul,
            'min': py_tmin,
            'max': py_tmax,
        }
        op_func = op_funcs[operation]
        results = np.array([op_func(indices_a[i], indices_b[i], num_trits)
                          for i in range(n)])

    return results


# =============================================================================
# LUT GENERATION
# =============================================================================

def create_operation_luts(
    num_values: int = 19683,
    num_trits: int = 9,
    sample_size: Optional[int] = None,
    operations: List[str] = ['add', 'mul', 'min', 'max'],
    output_dir: Optional[Path] = None,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Create operation lookup tables using SIMD engine.

    For full LUT: 19683² = 387,420,489 pairs per operation (~1.5GB per op)
    For sampled LUT: sample_size pairs per operation

    Args:
        num_values: Number of ternary values (3^num_trits)
        num_trits: Number of trits per value
        sample_size: If set, sample this many pairs instead of full LUT
        operations: List of operations to compute
        output_dir: Directory to save LUTs
        seed: Random seed for sampling

    Returns:
        Dictionary mapping operation names to (indices_a, indices_b, results) tuples
    """
    print(f"Creating operation LUTs for {num_values} values...")
    print(f"  SIMD engine: {'available' if HAS_SIMD else 'NOT available (using Python)'}")

    np.random.seed(seed)

    if sample_size is None:
        # Full LUT - warning: very large!
        print(f"  Full LUT mode: {num_values}² = {num_values**2:,} pairs per operation")
        indices_a, indices_b = np.meshgrid(
            np.arange(num_values),
            np.arange(num_values)
        )
        indices_a = indices_a.flatten()
        indices_b = indices_b.flatten()
    else:
        # Sampled LUT
        print(f"  Sampled mode: {sample_size:,} pairs per operation")
        indices_a = np.random.randint(0, num_values, size=sample_size)
        indices_b = np.random.randint(0, num_values, size=sample_size)

    luts = {}

    for op in operations:
        print(f"\n  Computing {op.upper()}...")
        t0 = time.time()

        # Batch processing for efficiency
        batch_size = 100000
        n_batches = (len(indices_a) + batch_size - 1) // batch_size

        results = []
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(indices_a))

            batch_a = indices_a[start:end]
            batch_b = indices_b[start:end]

            batch_results = simd_batch_operation(batch_a, batch_b, op, num_trits)
            results.append(batch_results)

            if (batch_idx + 1) % 10 == 0:
                print(f"    Batch {batch_idx + 1}/{n_batches}...")

        results = np.concatenate(results)
        luts[op] = {
            'indices_a': indices_a.copy(),
            'indices_b': indices_b.copy(),
            'results': results
        }

        print(f"    Done in {time.time() - t0:.1f}s")

    # Save if output directory specified
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for op, data in luts.items():
            np.savez_compressed(
                output_dir / f"lut_{op}.npz",
                indices_a=data['indices_a'],
                indices_b=data['indices_b'],
                results=data['results']
            )
        print(f"\nSaved LUTs to {output_dir}")

    return luts


def load_operation_luts(lut_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Load precomputed operation LUTs."""
    luts = {}
    for lut_file in lut_dir.glob("lut_*.npz"):
        op = lut_file.stem.replace("lut_", "")
        data = np.load(lut_file)
        luts[op] = {
            'indices_a': data['indices_a'],
            'indices_b': data['indices_b'],
            'results': data['results']
        }
    return luts


# =============================================================================
# PYTORCH DATASET
# =============================================================================

class OperationLUTDataset(Dataset):
    """
    PyTorch Dataset for operation-aware VAE training.

    Each sample is (idx_a, idx_b, op_id, idx_result) where:
    - idx_a, idx_b: Input value indices
    - op_id: Operation identifier (0=add, 1=mul, 2=min, 3=max)
    - idx_result: Correct result index
    """

    OP_TO_ID = {'add': 0, 'mul': 1, 'min': 2, 'max': 3}
    ID_TO_OP = {0: 'add', 1: 'mul', 2: 'min', 3: 'max'}

    def __init__(
        self,
        luts: Dict[str, Dict[str, np.ndarray]],
        embeddings: Optional[np.ndarray] = None,
        transform=None
    ):
        """
        Args:
            luts: Operation LUTs from create_operation_luts or load_operation_luts
            embeddings: Optional precomputed embeddings [N, D]
            transform: Optional transform to apply
        """
        self.embeddings = embeddings
        self.transform = transform

        # Combine all operations into single arrays
        all_a = []
        all_b = []
        all_ops = []
        all_results = []

        for op_name, data in luts.items():
            n = len(data['indices_a'])
            all_a.append(data['indices_a'])
            all_b.append(data['indices_b'])
            all_ops.append(np.full(n, self.OP_TO_ID[op_name], dtype=np.int64))
            all_results.append(data['results'])

        self.indices_a = np.concatenate(all_a)
        self.indices_b = np.concatenate(all_b)
        self.op_ids = np.concatenate(all_ops)
        self.results = np.concatenate(all_results)

        print(f"OperationLUTDataset: {len(self):,} samples")

    def __len__(self) -> int:
        return len(self.indices_a)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            'idx_a': torch.tensor(self.indices_a[idx], dtype=torch.long),
            'idx_b': torch.tensor(self.indices_b[idx], dtype=torch.long),
            'op_id': torch.tensor(self.op_ids[idx], dtype=torch.long),
            'idx_result': torch.tensor(self.results[idx], dtype=torch.long),
        }

        if self.embeddings is not None:
            sample['emb_a'] = torch.tensor(self.embeddings[self.indices_a[idx]], dtype=torch.float32)
            sample['emb_b'] = torch.tensor(self.embeddings[self.indices_b[idx]], dtype=torch.float32)
            sample['emb_result'] = torch.tensor(self.embeddings[self.results[idx]], dtype=torch.float32)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class TripletDataset(Dataset):
    """
    Dataset for ultrametric triplet training.

    Each sample is (idx_a, idx_b, idx_c) for ultrametric constraint:
    d(a,c) <= max(d(a,b), d(b,c))
    """

    def __init__(
        self,
        num_values: int = 19683,
        num_samples: int = 100000,
        embeddings: Optional[np.ndarray] = None,
        seed: int = 42
    ):
        np.random.seed(seed)

        self.indices_a = np.random.randint(0, num_values, size=num_samples)
        self.indices_b = np.random.randint(0, num_values, size=num_samples)
        self.indices_c = np.random.randint(0, num_values, size=num_samples)
        self.embeddings = embeddings

        # Precompute valuations
        self.valuations = np.array([self._valuation(i) for i in range(num_values)])

    def _valuation(self, n: int, max_val: int = 9) -> int:
        if n == 0:
            return max_val
        v = 0
        while n % 3 == 0 and v < max_val:
            n //= 3
            v += 1
        return v

    def __len__(self) -> int:
        return len(self.indices_a)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            'idx_a': torch.tensor(self.indices_a[idx], dtype=torch.long),
            'idx_b': torch.tensor(self.indices_b[idx], dtype=torch.long),
            'idx_c': torch.tensor(self.indices_c[idx], dtype=torch.long),
            'val_a': torch.tensor(self.valuations[self.indices_a[idx]], dtype=torch.long),
            'val_b': torch.tensor(self.valuations[self.indices_b[idx]], dtype=torch.long),
            'val_c': torch.tensor(self.valuations[self.indices_c[idx]], dtype=torch.long),
        }

        if self.embeddings is not None:
            sample['emb_a'] = torch.tensor(self.embeddings[self.indices_a[idx]], dtype=torch.float32)
            sample['emb_b'] = torch.tensor(self.embeddings[self.indices_b[idx]], dtype=torch.float32)
            sample['emb_c'] = torch.tensor(self.embeddings[self.indices_c[idx]], dtype=torch.float32)

        return sample


# =============================================================================
# MAIN - Generate LUTs
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate Operation LUTs')
    parser.add_argument('--samples', type=int, default=1000000, help='Number of samples')
    parser.add_argument('--output', type=str, default='luts', help='Output directory')
    parser.add_argument('--full', action='store_true', help='Generate full LUT (WARNING: huge)')

    args = parser.parse_args()

    output_dir = Path(__file__).parent / args.output

    sample_size = None if args.full else args.samples

    luts = create_operation_luts(
        sample_size=sample_size,
        output_dir=output_dir
    )

    # Print stats
    print("\nLUT Statistics:")
    for op, data in luts.items():
        print(f"  {op.upper()}: {len(data['results']):,} samples")
