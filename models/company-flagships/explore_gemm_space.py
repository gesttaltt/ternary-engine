#!/usr/bin/env python3
"""
Soft GEMM Space Exploration - Mapping the Continuous Ternary Landscape

Hypothesis: The discrete ternary operations (tadd, tmul, etc.) are sparse
islands in a continuous embedding space. The gaps between them may form
ultrametric lattices that reveal deeper structure.

Strategy:
1. COVERAGE over resolution - sample the full combinatorial space
2. Use learned embeddings as "soft" continuous coordinates
3. Map operation neighborhoods in embedding space
4. Detect ultrametric/lattice patterns between operation clusters
5. Generate pre-computed "soft GEMM map" for later refinement

Copyright 2025 Ternary Engine Contributors
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import spearmanr
from collections import defaultdict
import struct
import time


# =============================================================================
# LOAD MODELS AND EMBEDDINGS
# =============================================================================

def load_embeddings(path):
    """Load pre-computed embeddings."""
    return np.load(path)


def compute_valuation(n: int) -> int:
    """3-adic valuation."""
    if n == 0:
        return 10
    v = 0
    while n % 3 == 0:
        n //= 3
        v += 1
    return min(v, 9)


def index_to_trits(index: int) -> np.ndarray:
    """Convert index to 9 balanced trits."""
    trits = np.zeros(9, dtype=np.int8)
    for i in range(9):
        trits[i] = (index % 3) - 1
        index //= 3
    return trits


def trits_to_index(trits: np.ndarray) -> int:
    """Convert balanced trits to index."""
    index = 0
    for i in range(len(trits)):
        index += (int(trits[i]) + 1) * (3 ** i)
    return index


# =============================================================================
# TERNARY OPERATIONS (Ground Truth)
# =============================================================================

def ternary_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Balanced ternary addition with carry."""
    result = np.zeros(9, dtype=np.int8)
    carry = 0
    for i in range(9):
        s = int(a[i]) + int(b[i]) + carry
        if s > 1:
            result[i] = s - 3
            carry = 1
        elif s < -1:
            result[i] = s + 3
            carry = -1
        else:
            result[i] = s
            carry = 0
    return result


def ternary_mul_elementwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise ternary multiplication."""
    return np.clip(a * b, -1, 1).astype(np.int8)


def ternary_min(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise ternary min."""
    return np.minimum(a, b).astype(np.int8)


def ternary_max(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise ternary max."""
    return np.maximum(a, b).astype(np.int8)


# =============================================================================
# SOFT GEMM SPACE MAPPING
# =============================================================================

class SoftGEMMExplorer:
    """Explores the continuous structure of ternary GEMM space."""

    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings  # [19683, 16]
        self.N = len(embeddings)
        self.dim = embeddings.shape[1]

        # Pre-compute valuations
        self.valuations = np.array([compute_valuation(i) for i in range(self.N)])

        # Pre-compute radii
        self.radii = np.linalg.norm(embeddings, axis=1)

    def soft_operation(self, a_idx: int, b_idx: int, alpha: float = 0.5) -> np.ndarray:
        """
        Soft interpolation between two ternary values in embedding space.

        alpha=0 -> embedding of a
        alpha=1 -> embedding of b
        alpha=0.5 -> midpoint (approximate sum/operation)
        """
        return (1 - alpha) * self.embeddings[a_idx] + alpha * self.embeddings[b_idx]

    def find_nearest(self, query: np.ndarray, exclude: set = None) -> int:
        """Find nearest discrete ternary value to continuous query."""
        distances = np.linalg.norm(self.embeddings - query, axis=1)
        if exclude:
            for idx in exclude:
                distances[idx] = np.inf
        return np.argmin(distances)

    def operation_neighborhood(self, a_idx: int, b_idx: int,
                                operation: str = 'add', k: int = 10) -> dict:
        """
        Map the neighborhood around an operation result in embedding space.

        Returns: {
            'discrete_result': exact ternary operation result,
            'soft_result': continuous embedding midpoint,
            'nearest_k': k nearest discrete values to soft result,
            'distance_to_discrete': how far soft result is from discrete,
            'ultrametric_candidates': values that form ultrametric triangles
        }
        """
        # Get trits
        a_trits = index_to_trits(a_idx)
        b_trits = index_to_trits(b_idx)

        # Compute discrete operation
        if operation == 'add':
            result_trits = ternary_add(a_trits, b_trits)
        elif operation == 'mul':
            result_trits = ternary_mul_elementwise(a_trits, b_trits)
        elif operation == 'min':
            result_trits = ternary_min(a_trits, b_trits)
        elif operation == 'max':
            result_trits = ternary_max(a_trits, b_trits)
        else:
            result_trits = a_trits  # Identity

        discrete_result = trits_to_index(result_trits)

        # Compute soft result (midpoint in embedding space)
        soft_result = self.soft_operation(a_idx, b_idx, alpha=0.5)

        # Find k nearest to soft result
        distances = np.linalg.norm(self.embeddings - soft_result, axis=1)
        nearest_k = np.argsort(distances)[:k]
        nearest_distances = distances[nearest_k]

        # Distance from soft to discrete result
        dist_to_discrete = np.linalg.norm(soft_result - self.embeddings[discrete_result])

        # Check for ultrametric violations/patterns
        # Ultrametric: d(x,z) <= max(d(x,y), d(y,z))
        ultrametric_candidates = []
        d_ab = np.linalg.norm(self.embeddings[a_idx] - self.embeddings[b_idx])
        d_ar = np.linalg.norm(self.embeddings[a_idx] - self.embeddings[discrete_result])
        d_br = np.linalg.norm(self.embeddings[b_idx] - self.embeddings[discrete_result])

        # Check ultrametric property
        is_ultrametric = (d_ar <= max(d_ab, d_br) + 1e-6) and \
                         (d_br <= max(d_ab, d_ar) + 1e-6) and \
                         (d_ab <= max(d_ar, d_br) + 1e-6)

        return {
            'a_idx': a_idx,
            'b_idx': b_idx,
            'operation': operation,
            'discrete_result': discrete_result,
            'soft_embedding': soft_result,
            'nearest_k_indices': nearest_k.tolist(),
            'nearest_k_distances': nearest_distances.tolist(),
            'dist_to_discrete': dist_to_discrete,
            'is_ultrametric': is_ultrametric,
            'd_ab': d_ab,
            'd_ar': d_ar,
            'd_br': d_br,
            'discrete_in_top_k': discrete_result in nearest_k
        }

    def explore_operation_space(self, operation: str,
                                 num_samples: int = 10000,
                                 seed: int = 42) -> dict:
        """
        Explore the full combinatorial space of an operation.

        Samples pairs uniformly and maps their neighborhoods.
        """
        np.random.seed(seed)

        results = {
            'operation': operation,
            'num_samples': num_samples,
            'ultrametric_rate': 0,
            'discrete_in_top1_rate': 0,
            'discrete_in_top5_rate': 0,
            'mean_dist_to_discrete': 0,
            'valuation_preservation': [],
            'lattice_patterns': [],
            'soft_gap_distribution': []
        }

        ultrametric_count = 0
        top1_count = 0
        top5_count = 0
        dist_sum = 0

        # Sample pairs
        for i in range(num_samples):
            a_idx = np.random.randint(self.N)
            b_idx = np.random.randint(self.N)

            neighborhood = self.operation_neighborhood(a_idx, b_idx, operation, k=10)

            if neighborhood['is_ultrametric']:
                ultrametric_count += 1

            if neighborhood['discrete_result'] == neighborhood['nearest_k_indices'][0]:
                top1_count += 1

            if neighborhood['discrete_result'] in neighborhood['nearest_k_indices'][:5]:
                top5_count += 1

            dist_sum += neighborhood['dist_to_discrete']

            # Track soft gap (distance from soft result to nearest discrete)
            results['soft_gap_distribution'].append(neighborhood['nearest_k_distances'][0])

            # Track valuation changes
            v_a = self.valuations[a_idx]
            v_b = self.valuations[b_idx]
            v_r = self.valuations[neighborhood['discrete_result']]
            results['valuation_preservation'].append({
                'v_a': v_a, 'v_b': v_b, 'v_result': v_r,
                'v_min_input': min(v_a, v_b)
            })

        results['ultrametric_rate'] = ultrametric_count / num_samples
        results['discrete_in_top1_rate'] = top1_count / num_samples
        results['discrete_in_top5_rate'] = top5_count / num_samples
        results['mean_dist_to_discrete'] = dist_sum / num_samples

        return results

    def map_lattice_structure(self, num_anchor_points: int = 100) -> dict:
        """
        Map lattice-like structure by finding repeating distance patterns.

        Hypothesis: Operations form lattices where certain distance ratios
        appear repeatedly, indicating algebraic structure.
        """
        np.random.seed(42)

        # Sample anchor points across valuation levels
        anchors = []
        for v in range(10):
            mask = self.valuations == v
            indices = np.where(mask)[0]
            if len(indices) > 0:
                sample_size = min(num_anchor_points // 10, len(indices))
                anchors.extend(np.random.choice(indices, sample_size, replace=False))

        anchors = np.array(anchors)

        # Compute pairwise distances between anchors
        anchor_embeddings = self.embeddings[anchors]
        pairwise_dist = squareform(pdist(anchor_embeddings))

        # Find common distance ratios (potential lattice spacing)
        distances = pairwise_dist[np.triu_indices_from(pairwise_dist, k=1)]
        distances = distances[distances > 0.01]  # Filter near-zero

        # Histogram of distances
        hist, bin_edges = np.histogram(distances, bins=100)
        peak_bins = np.argsort(hist)[-10:]  # Top 10 peaks
        peak_distances = [(bin_edges[b] + bin_edges[b+1])/2 for b in peak_bins]

        # Check for ratio patterns (lattice signature)
        ratios = []
        for i, d1 in enumerate(peak_distances):
            for d2 in peak_distances[i+1:]:
                if d2 > 0.01:
                    ratios.append(d1 / d2)

        return {
            'num_anchors': len(anchors),
            'distance_peaks': peak_distances,
            'distance_ratios': ratios,
            'mean_distance': float(np.mean(distances)),
            'std_distance': float(np.std(distances)),
            'distance_histogram': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        }

    def compute_ultrametric_matrix(self, sample_size: int = 500) -> dict:
        """
        Compute ultrametric distance matrix for a sample.

        Ultrametric distance: d(x,z) = max(d(x,y), d(y,z)) for all y on path
        This reveals hierarchical/tree structure.
        """
        np.random.seed(42)
        sample_indices = np.random.choice(self.N, sample_size, replace=False)
        sample_embeddings = self.embeddings[sample_indices]

        # Euclidean distances (condensed form)
        euclidean_condensed = pdist(sample_embeddings)

        # Compute ultrametric via single-linkage clustering
        linkage_matrix = linkage(euclidean_condensed, method='single')

        # cophenet returns (cophenetic_correlation, cophenetic_distances)
        from scipy.cluster.hierarchy import cophenet
        correlation, cophenetic_dists = cophenet(linkage_matrix, euclidean_condensed)

        return {
            'sample_indices': sample_indices.tolist(),
            'euclidean_mean': float(np.mean(euclidean_condensed)),
            'cophenetic_mean': float(np.mean(cophenetic_dists)),
            'correlation': float(correlation)
        }


# =============================================================================
# CONTINUOUS GEMM MAP GENERATION
# =============================================================================

def generate_soft_gemm_map(explorer: SoftGEMMExplorer,
                           resolution: int = 50,
                           operation: str = 'add') -> dict:
    """
    Generate a continuous map of GEMM space.

    Samples the space at given resolution and records:
    - Soft interpolation results
    - Nearest discrete values
    - Distance patterns
    """
    print(f"\nGenerating soft GEMM map for {operation}...")
    print(f"Resolution: {resolution}x{resolution} = {resolution**2} samples")

    # Sample indices uniformly across the space
    indices = np.linspace(0, explorer.N - 1, resolution, dtype=int)

    soft_map = np.zeros((resolution, resolution, explorer.dim))
    nearest_map = np.zeros((resolution, resolution), dtype=int)
    distance_map = np.zeros((resolution, resolution))
    discrete_match_map = np.zeros((resolution, resolution), dtype=bool)

    for i, a_idx in enumerate(indices):
        if i % 10 == 0:
            print(f"  Row {i}/{resolution}...")

        for j, b_idx in enumerate(indices):
            # Soft result
            soft = explorer.soft_operation(int(a_idx), int(b_idx), alpha=0.5)
            soft_map[i, j] = soft

            # Nearest discrete
            nearest = explorer.find_nearest(soft)
            nearest_map[i, j] = nearest

            # Distance to nearest
            distance_map[i, j] = np.linalg.norm(soft - explorer.embeddings[nearest])

            # Does nearest match discrete operation?
            a_trits = index_to_trits(int(a_idx))
            b_trits = index_to_trits(int(b_idx))
            if operation == 'add':
                result_trits = ternary_add(a_trits, b_trits)
            elif operation == 'mul':
                result_trits = ternary_mul_elementwise(a_trits, b_trits)
            else:
                result_trits = a_trits
            discrete_result = trits_to_index(result_trits)
            discrete_match_map[i, j] = (nearest == discrete_result)

    return {
        'operation': operation,
        'resolution': resolution,
        'indices': indices.tolist(),
        'soft_map': soft_map,
        'nearest_map': nearest_map,
        'distance_map': distance_map,
        'discrete_match_map': discrete_match_map,
        'match_rate': float(discrete_match_map.mean()),
        'mean_distance': float(distance_map.mean()),
        'max_distance': float(distance_map.max()),
        'distance_std': float(distance_map.std())
    }


# =============================================================================
# MAIN EXPLORATION
# =============================================================================

def main():
    base_dir = Path(__file__).parent
    output_dir = base_dir / "gemm_exploration"
    output_dir.mkdir(exist_ok=True)

    print("="*70)
    print("SOFT GEMM SPACE EXPLORATION")
    print("Mapping the Continuous Ternary Landscape")
    print("="*70)

    # Load embeddings
    print("\nLoading embeddings...")
    embeddings = load_embeddings(base_dir / "gemm_export" / "ternary_embeddings.npy")
    print(f"  Shape: {embeddings.shape}")

    # Create explorer
    explorer = SoftGEMMExplorer(embeddings)

    # ==========================================================================
    # 1. EXPLORE EACH OPERATION
    # ==========================================================================
    print("\n" + "="*70)
    print("1. OPERATION SPACE EXPLORATION")
    print("="*70)

    operations = ['add', 'mul', 'min', 'max']
    operation_results = {}

    for op in operations:
        print(f"\nExploring {op}...")
        t0 = time.time()
        results = explorer.explore_operation_space(op, num_samples=5000)
        t1 = time.time()

        operation_results[op] = results
        print(f"  Ultrametric rate: {results['ultrametric_rate']:.2%}")
        print(f"  Discrete in top-1: {results['discrete_in_top1_rate']:.2%}")
        print(f"  Discrete in top-5: {results['discrete_in_top5_rate']:.2%}")
        print(f"  Mean dist to discrete: {results['mean_dist_to_discrete']:.4f}")
        print(f"  Time: {t1-t0:.1f}s")

    # ==========================================================================
    # 2. MAP LATTICE STRUCTURE
    # ==========================================================================
    print("\n" + "="*70)
    print("2. LATTICE STRUCTURE ANALYSIS")
    print("="*70)

    lattice = explorer.map_lattice_structure(num_anchor_points=200)
    print(f"\nDistance peaks (potential lattice spacings):")
    for i, d in enumerate(sorted(lattice['distance_peaks'])[:5]):
        print(f"  {i+1}. {d:.4f}")
    print(f"\nMean distance: {lattice['mean_distance']:.4f}")
    print(f"Std distance: {lattice['std_distance']:.4f}")

    # ==========================================================================
    # 3. ULTRAMETRIC STRUCTURE
    # ==========================================================================
    print("\n" + "="*70)
    print("3. ULTRAMETRIC STRUCTURE ANALYSIS")
    print("="*70)

    ultrametric = explorer.compute_ultrametric_matrix(sample_size=500)
    print(f"\nEuclidean vs Ultrametric correlation: {ultrametric['correlation']:.4f}")
    print("(High correlation suggests tree-like/hierarchical structure)")

    # ==========================================================================
    # 4. GENERATE SOFT GEMM MAPS
    # ==========================================================================
    print("\n" + "="*70)
    print("4. SOFT GEMM MAP GENERATION")
    print("="*70)

    gemm_maps = {}
    for op in ['add', 'mul']:
        gemm_map = generate_soft_gemm_map(explorer, resolution=100, operation=op)
        gemm_maps[op] = gemm_map
        print(f"\n{op.upper()} Map:")
        print(f"  Match rate (soft -> discrete): {gemm_map['match_rate']:.2%}")
        print(f"  Mean distance to nearest: {gemm_map['mean_distance']:.4f}")
        print(f"  Max distance (gap size): {gemm_map['max_distance']:.4f}")

    # ==========================================================================
    # 5. SAVE RESULTS
    # ==========================================================================
    print("\n" + "="*70)
    print("5. SAVING RESULTS")
    print("="*70)

    # Save operation exploration results
    for op, results in operation_results.items():
        # Remove large arrays for JSON
        save_results = {k: v for k, v in results.items()
                       if not isinstance(v, list) or len(v) < 100}
        save_results['soft_gap_stats'] = {
            'mean': float(np.mean(results['soft_gap_distribution'])),
            'std': float(np.std(results['soft_gap_distribution'])),
            'min': float(np.min(results['soft_gap_distribution'])),
            'max': float(np.max(results['soft_gap_distribution']))
        }

    # Save comprehensive report (convert all numpy types to Python types for JSON)
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_ternary_values': int(explorer.N),
        'embedding_dim': int(explorer.dim),
        'operations': {op: {
            'ultrametric_rate': float(operation_results[op]['ultrametric_rate']),
            'discrete_in_top1': float(operation_results[op]['discrete_in_top1_rate']),
            'discrete_in_top5': float(operation_results[op]['discrete_in_top5_rate']),
            'mean_dist_to_discrete': float(operation_results[op]['mean_dist_to_discrete']),
            'soft_gap_mean': float(np.mean(operation_results[op]['soft_gap_distribution'])),
            'soft_gap_max': float(np.max(operation_results[op]['soft_gap_distribution']))
        } for op in operations},
        'lattice': {
            'distance_peaks': [float(p) for p in lattice['distance_peaks']],
            'mean_distance': float(lattice['mean_distance']),
            'std_distance': float(lattice['std_distance'])
        },
        'ultrametric': {
            'euclidean_ultrametric_correlation': float(ultrametric['correlation'])
        },
        'gemm_maps': {op: {
            'match_rate': float(gemm_maps[op]['match_rate']),
            'mean_distance': float(gemm_maps[op]['mean_distance']),
            'max_distance': float(gemm_maps[op]['max_distance'])
        } for op in gemm_maps}
    }

    with open(output_dir / 'exploration_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    # Save GEMM maps as numpy
    for op in gemm_maps:
        np.save(output_dir / f'soft_gemm_map_{op}.npy', gemm_maps[op]['soft_map'])
        np.save(output_dir / f'nearest_map_{op}.npy', gemm_maps[op]['nearest_map'])
        np.save(output_dir / f'distance_map_{op}.npy', gemm_maps[op]['distance_map'])

    print(f"\nSaved to: {output_dir}")
    print(f"  exploration_report.json")
    print(f"  soft_gemm_map_*.npy")
    print(f"  nearest_map_*.npy")
    print(f"  distance_map_*.npy")

    # ==========================================================================
    # 6. KEY INSIGHTS
    # ==========================================================================
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)

    print("\n1. OPERATION STRUCTURE:")
    for op in operations:
        r = operation_results[op]
        print(f"   {op.upper()}: {r['ultrametric_rate']:.1%} ultrametric, "
              f"{r['discrete_in_top1_rate']:.1%} top-1 match")

    print("\n2. SOFT GAP ANALYSIS:")
    print("   The 'soft gap' is the distance from continuous interpolation")
    print("   to the nearest discrete ternary value.")
    for op in ['add', 'mul']:
        gap_mean = np.mean(operation_results[op]['soft_gap_distribution'])
        gap_max = np.max(operation_results[op]['soft_gap_distribution'])
        print(f"   {op.upper()}: mean={gap_mean:.4f}, max={gap_max:.4f}")

    print("\n3. LATTICE HYPOTHESIS:")
    print(f"   Euclidean-Ultrametric correlation: {ultrametric['correlation']:.4f}")
    if ultrametric['correlation'] > 0.7:
        print("   -> Strong tree-like structure detected!")
    elif ultrametric['correlation'] > 0.4:
        print("   -> Moderate hierarchical structure")
    else:
        print("   -> Complex non-hierarchical structure")

    print("\n4. COVERAGE ACHIEVED:")
    print(f"   Soft GEMM maps: 100x100 = 10,000 operation samples")
    print(f"   Operation exploration: 5,000 random pairs per operation")
    print(f"   Total coverage: ~30,000 unique operation mappings")

    print("\n" + "="*70)
    print("EXPLORATION COMPLETE")
    print("="*70)
    print("\nNext: Use these soft maps to guide hybrid neural GEMM refinement")


if __name__ == "__main__":
    main()
