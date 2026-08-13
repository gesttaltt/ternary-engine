#!/usr/bin/env python3
"""
Extended GEMM Space Exploration - Deep Analysis

Performs extended exploration of the continuous ternary embedding space
with higher coverage, multi-scale analysis, and deeper pattern discovery.

Compared to explore_gemm_space.py:
- 10x more samples per operation (50K vs 5K)
- 4x higher resolution GEMM maps (200x200 vs 100x100)
- Multi-scale lattice analysis
- Cluster structure discovery
- Extended ultrametric analysis with hierarchy depth estimation

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

USAGE: python explore_gemm_extended.py [--quick]
OUTPUT: gemm_exploration/extended_*.json, extended_*.npy
"""

import argparse
import time
import json
import zlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, cophenet
from scipy.stats import spearmanr
from collections import defaultdict


def stable_str_hash(s: str) -> int:
    """
    Deterministic string hash for seeding, in [0, 2**32).

    Python's builtin hash() is randomized per-process for str/bytes
    (PYTHONHASHSEED) unless explicitly disabled, so `42 + hash(op) % 1000`
    was not actually a fixed seed across runs. Found 2026-08-13; violates
    CLAUDE.md's "reproducible - Fixed random seeds for deterministic
    tests" convention. zlib.crc32 is stable across processes and
    platforms. (Same fix applied to embedding_exactitude_score.py.)
    """
    return zlib.crc32(s.encode('utf-8'))


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class ExtendedGEMMExplorer:
    """Extended GEMM space exploration with deep analysis."""

    def __init__(self, embeddings: np.ndarray):
        """
        Initialize with pre-computed embeddings.

        Args:
            embeddings: [N, D] array of embeddings for all N ternary values
        """
        self.embeddings = embeddings
        self.N = embeddings.shape[0]  # 19683
        self.dim = embeddings.shape[1]  # 16

        # Precompute norms and normalized embeddings
        self.norms = np.linalg.norm(embeddings, axis=1)
        self.normalized = embeddings / (self.norms[:, None] + 1e-8)

        # Precompute valuations. `i` is the RAW encoding index, not the
        # decoded balanced-ternary value -- the true zero (all-zero trits)
        # lives at idx_offset = (N-1)//2, not at i=0 (which decodes to the
        # most negative representable value). self._valuation() must be
        # called on the DECODED value (i - idx_offset), same bug class
        # already found and fixed across models/ and
        # research/scripts/falsify.py. Found 2026-08-13.
        idx_offset = (self.N - 1) // 2
        self.valuations = np.array([self._valuation(i - idx_offset) for i in range(self.N)])

    def _valuation(self, n: int) -> int:
        """Compute 3-adic valuation."""
        if n == 0:
            return 9
        v = 0
        while n % 3 == 0 and v < 9:
            n //= 3
            v += 1
        return v

    def index_to_trits(self, idx: int) -> np.ndarray:
        """Convert index to balanced trits."""
        trits = np.zeros(9, dtype=np.float32)
        for i in range(9):
            trits[i] = (idx % 3) - 1
            idx //= 3
        return trits

    def trits_to_index(self, trits: np.ndarray) -> int:
        """Convert balanced trits to index."""
        idx = 0
        for i in range(8, -1, -1):
            idx = idx * 3 + int(trits[i]) + 1
        return idx

    def discrete_operation(self, a: int, b: int, op: str) -> int:
        """Perform discrete ternary operation."""
        trits_a = self.index_to_trits(a)
        trits_b = self.index_to_trits(b)

        if op == 'add':
            # Balanced ternary addition with carry
            result = np.zeros(9, dtype=np.float32)
            carry = 0
            for i in range(9):
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
        elif op == 'mul':
            # Element-wise multiplication across all 9 trits (not "first
            # trit only" as this comment previously and incorrectly said --
            # trits_a/trits_b are full 9-element arrays, so `*` here is a
            # full element-wise product, not a scalar/first-element
            # operation. Comment corrected 2026-08-13 to match the actual
            # code; the code itself is unchanged.
            result = trits_a * trits_b
        elif op == 'min':
            result = np.minimum(trits_a, trits_b)
        elif op == 'max':
            result = np.maximum(trits_a, trits_b)
        else:
            raise ValueError(f"Unknown operation: {op}")

        # Clamp to valid range
        result = np.clip(result, -1, 1).round().astype(np.float32)
        return self.trits_to_index(result)

    def soft_operation(self, a: int, b: int, alpha: float = 0.5) -> np.ndarray:
        """Compute soft operation via embedding interpolation."""
        emb_a = self.embeddings[a]
        emb_b = self.embeddings[b]
        return (1 - alpha) * emb_a + alpha * emb_b

    def nearest_discrete(self, embedding: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest discrete embeddings."""
        distances = np.linalg.norm(self.embeddings - embedding, axis=1)
        indices = np.argsort(distances)[:k]
        return indices, distances[indices]


def explore_operations_extended(
    explorer: ExtendedGEMMExplorer,
    operations: List[str],
    num_samples: int = 50000
) -> Dict[str, Dict[str, Any]]:
    """Extended operation space exploration with more samples."""
    results = {}

    for op in operations:
        print(f"\nExploring {op} with {num_samples:,} samples...")
        t0 = time.perf_counter()

        # Random pairs
        np.random.seed(42 + stable_str_hash(op) % 1000)
        pairs_a = np.random.randint(0, explorer.N, size=num_samples)
        pairs_b = np.random.randint(0, explorer.N, size=num_samples)

        ultrametric_count = 0
        top1_count = 0
        top5_count = 0
        distances_to_discrete = []
        soft_gaps = []
        valuation_correlations = []

        for i in range(num_samples):
            a, b = pairs_a[i], pairs_b[i]

            # Discrete operation
            discrete_result = explorer.discrete_operation(a, b, op)

            # Soft operation (midpoint)
            soft_emb = explorer.soft_operation(a, b, 0.5)

            # Find nearest discrete
            nearest_idx, nearest_dist = explorer.nearest_discrete(soft_emb, k=5)

            # Check ultrametric: d(a,c) <= max(d(a,b), d(b,c))
            d_ab = np.linalg.norm(explorer.embeddings[a] - explorer.embeddings[b])
            d_ac = np.linalg.norm(explorer.embeddings[a] - explorer.embeddings[discrete_result])
            d_bc = np.linalg.norm(explorer.embeddings[b] - explorer.embeddings[discrete_result])
            if d_ac <= max(d_ab, d_bc) + 1e-6:
                ultrametric_count += 1

            # Check if discrete result is in top-k
            if discrete_result == nearest_idx[0]:
                top1_count += 1
            if discrete_result in nearest_idx:
                top5_count += 1

            # Distance to discrete result
            dist_to_discrete = np.linalg.norm(soft_emb - explorer.embeddings[discrete_result])
            distances_to_discrete.append(dist_to_discrete)

            # Soft gap (distance to nearest discrete, regardless of operation)
            soft_gaps.append(nearest_dist[0])

            # Track valuation correlation
            result_val = explorer.valuations[discrete_result]
            input_max_val = max(explorer.valuations[a], explorer.valuations[b])
            valuation_correlations.append((result_val, input_max_val))

            if (i + 1) % 10000 == 0:
                print(f"  {i+1:,}/{num_samples:,}...")

        # Compute valuation correlation
        vals_result = [v[0] for v in valuation_correlations]
        vals_input = [v[1] for v in valuation_correlations]
        val_corr, _ = spearmanr(vals_result, vals_input)

        results[op] = {
            'num_samples': num_samples,
            'ultrametric_rate': ultrametric_count / num_samples,
            'discrete_in_top1_rate': top1_count / num_samples,
            'discrete_in_top5_rate': top5_count / num_samples,
            'mean_dist_to_discrete': float(np.mean(distances_to_discrete)),
            'std_dist_to_discrete': float(np.std(distances_to_discrete)),
            'soft_gap_distribution': soft_gaps,
            'valuation_correlation': float(val_corr) if not np.isnan(val_corr) else 0.0,
            'time': time.perf_counter() - t0
        }

        print(f"  Ultrametric rate: {results[op]['ultrametric_rate']:.2%}")
        print(f"  Top-1 match: {results[op]['discrete_in_top1_rate']:.2%}")
        print(f"  Top-5 match: {results[op]['discrete_in_top5_rate']:.2%}")
        print(f"  Valuation correlation: {results[op]['valuation_correlation']:.4f}")
        print(f"  Time: {results[op]['time']:.1f}s")

    return results


def analyze_multiscale_lattice(explorer: ExtendedGEMMExplorer, num_samples: int = 2000) -> Dict[str, Any]:
    """Multi-scale lattice analysis to find structure at different scales."""
    print("\nMulti-scale lattice analysis...")

    # Sample points
    np.random.seed(123)
    sample_idx = np.random.choice(explorer.N, size=min(num_samples, explorer.N), replace=False)
    sample_embeddings = explorer.embeddings[sample_idx]

    # Compute pairwise distances
    distances = pdist(sample_embeddings)

    # Analyze at multiple scales
    scales = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    scale_analysis = {}

    for scale in scales:
        # Count pairs within this scale
        pairs_in_scale = np.sum(distances < scale)
        total_pairs = len(distances)

        # Density at this scale
        density = pairs_in_scale / total_pairs

        scale_analysis[f'scale_{scale:.2f}'] = {
            'pairs_count': int(pairs_in_scale),
            'density': float(density)
        }

    # Find distance distribution peaks at multiple resolutions
    peaks_by_resolution = {}
    for num_bins in [50, 100, 200, 500]:
        hist, bin_edges = np.histogram(distances, bins=num_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Find local maxima
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.mean(hist):
                peaks.append(float(bin_centers[i]))

        peaks_by_resolution[num_bins] = peaks[:20]  # Top 20 peaks

    # Hierarchical clustering analysis
    print("  Hierarchical clustering...")
    Z = linkage(sample_embeddings, method='ward')

    # Find optimal number of clusters at different thresholds
    cluster_analysis = {}
    for n_clusters in [5, 10, 20, 50, 100]:
        labels = fcluster(Z, n_clusters, criterion='maxclust')

        # Compute cluster quality metrics
        cluster_sizes = [np.sum(labels == c) for c in range(1, n_clusters + 1)]
        non_empty = [s for s in cluster_sizes if s > 0]

        cluster_analysis[f'k_{n_clusters}'] = {
            'num_non_empty': len(non_empty),
            'mean_size': float(np.mean(non_empty)) if non_empty else 0,
            'std_size': float(np.std(non_empty)) if non_empty else 0,
            'max_size': max(non_empty) if non_empty else 0,
            'min_size': min(non_empty) if non_empty else 0
        }

    return {
        'num_samples': len(sample_idx),
        'distance_stats': {
            'mean': float(np.mean(distances)),
            'std': float(np.std(distances)),
            'min': float(np.min(distances)),
            'max': float(np.max(distances)),
            'median': float(np.median(distances))
        },
        'scale_analysis': scale_analysis,
        'peaks_by_resolution': peaks_by_resolution,
        'cluster_analysis': cluster_analysis
    }


def analyze_ultrametric_extended(explorer: ExtendedGEMMExplorer, num_triplets: int = 50000) -> Dict[str, Any]:
    """Extended ultrametric analysis with hierarchy depth estimation."""
    print("\nExtended ultrametric analysis...")

    np.random.seed(456)

    # Sample triplets
    triplets_a = np.random.randint(0, explorer.N, size=num_triplets)
    triplets_b = np.random.randint(0, explorer.N, size=num_triplets)
    triplets_c = np.random.randint(0, explorer.N, size=num_triplets)

    ultrametric_violations = 0
    violation_magnitudes = []
    hierarchy_depths = []

    for i in range(num_triplets):
        a, b, c = triplets_a[i], triplets_b[i], triplets_c[i]

        d_ab = np.linalg.norm(explorer.embeddings[a] - explorer.embeddings[b])
        d_bc = np.linalg.norm(explorer.embeddings[b] - explorer.embeddings[c])
        d_ac = np.linalg.norm(explorer.embeddings[a] - explorer.embeddings[c])

        # Check ultrametric inequality: d(a,c) <= max(d(a,b), d(b,c))
        max_side = max(d_ab, d_bc)
        if d_ac > max_side + 1e-6:
            ultrametric_violations += 1
            violation_magnitudes.append(d_ac - max_side)

        # Estimate hierarchy depth based on valuation
        depths = [explorer.valuations[a], explorer.valuations[b], explorer.valuations[c]]
        hierarchy_depths.append(max(depths) - min(depths))

    # Compute ultrametric distance matrix for sample
    sample_size = min(500, explorer.N)
    sample_idx = np.random.choice(explorer.N, size=sample_size, replace=False)
    sample_emb = explorer.embeddings[sample_idx]

    euclidean_condensed = pdist(sample_emb)

    # Compute ultrametric (cophenetic) distance via single-linkage
    # clustering. This used to call _get_merge_height() in a nested loop
    # over all C(sample_size, 2) pairs, each doing an O(sample_size) dict
    # scan per merge step (~O(n^4) total) -- scipy.cluster.hierarchy.cophenet
    # computes the exact same cophenetic distances in one vectorized call
    # (already used for this same purpose in the sibling file
    # explore_gemm_space.py). Verified byte-for-byte identical output to
    # the old per-pair loop before removing it. Found 2026-08-13.
    Z = linkage(sample_emb, method='single')
    _, ultrametric_flat = cophenet(Z, euclidean_condensed)

    # Correlation between Euclidean and ultrametric
    correlation, _ = spearmanr(euclidean_condensed, ultrametric_flat)

    return {
        'num_triplets': num_triplets,
        'violation_rate': ultrametric_violations / num_triplets,
        'mean_violation_magnitude': float(np.mean(violation_magnitudes)) if violation_magnitudes else 0,
        'max_violation_magnitude': float(np.max(violation_magnitudes)) if violation_magnitudes else 0,
        'mean_hierarchy_depth': float(np.mean(hierarchy_depths)),
        'euclidean_ultrametric_correlation': float(correlation),
        'sample_size_for_correlation': sample_size
    }


def generate_extended_gemm_map(
    explorer: ExtendedGEMMExplorer,
    resolution: int = 200,
    operation: str = 'add'
) -> Dict[str, Any]:
    """Generate high-resolution soft GEMM map."""
    print(f"\nGenerating {resolution}x{resolution} GEMM map for {operation}...")

    # Sample indices uniformly across the space
    indices = np.linspace(0, explorer.N - 1, resolution).astype(int)

    soft_map = np.zeros((resolution, resolution, explorer.dim), dtype=np.float32)
    nearest_map = np.zeros((resolution, resolution), dtype=np.int32)
    distance_map = np.zeros((resolution, resolution), dtype=np.float32)
    discrete_map = np.zeros((resolution, resolution), dtype=np.int32)

    matches = 0
    total = 0

    for i, idx_a in enumerate(indices):
        if i % 20 == 0:
            print(f"  Row {i}/{resolution}...")

        for j, idx_b in enumerate(indices):
            # Soft operation
            soft_emb = explorer.soft_operation(idx_a, idx_b, 0.5)
            soft_map[i, j] = soft_emb

            # Find nearest discrete
            nearest_idx, nearest_dist = explorer.nearest_discrete(soft_emb, k=1)
            nearest_map[i, j] = nearest_idx[0]
            distance_map[i, j] = nearest_dist[0]

            # Discrete operation
            discrete_result = explorer.discrete_operation(idx_a, idx_b, operation)
            discrete_map[i, j] = discrete_result

            if nearest_idx[0] == discrete_result:
                matches += 1
            total += 1

    return {
        'operation': operation,
        'resolution': resolution,
        'soft_map': soft_map,
        'nearest_map': nearest_map,
        'distance_map': distance_map,
        'discrete_map': discrete_map,
        'match_rate': matches / total,
        'mean_distance': float(np.mean(distance_map)),
        'max_distance': float(np.max(distance_map)),
        'std_distance': float(np.std(distance_map))
    }


def analyze_gap_structure(distance_maps: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Analyze the structure of gaps (regions far from discrete values)."""
    print("\nAnalyzing gap structure...")

    results = {}

    for op, dist_map in distance_maps.items():
        # Find high-gap regions (threshold: mean + 1.5*std)
        threshold = np.mean(dist_map) + 1.5 * np.std(dist_map)
        high_gap_mask = dist_map > threshold

        # Count and characterize gaps
        num_high_gap = np.sum(high_gap_mask)
        total = dist_map.size

        # Find connected regions
        from scipy.ndimage import label
        labeled, num_features = label(high_gap_mask)

        # Analyze gap sizes
        gap_sizes = []
        gap_max_distances = []
        for region_id in range(1, num_features + 1):
            region_mask = labeled == region_id
            gap_sizes.append(np.sum(region_mask))
            gap_max_distances.append(float(np.max(dist_map[region_mask])))

        results[op] = {
            'threshold': float(threshold),
            'high_gap_fraction': num_high_gap / total,
            'num_gap_regions': num_features,
            'gap_sizes': gap_sizes[:50],  # Top 50
            'gap_max_distances': gap_max_distances[:50],
            'mean_gap_size': float(np.mean(gap_sizes)) if gap_sizes else 0,
            'max_gap_size': max(gap_sizes) if gap_sizes else 0
        }

        print(f"  {op.upper()}: {num_features} gap regions, "
              f"{num_high_gap/total:.1%} high-gap area")

    return results


def main():
    parser = argparse.ArgumentParser(description='Extended GEMM Space Exploration')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer samples)')
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    output_dir = base_dir / "gemm_exploration"
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("EXTENDED GEMM SPACE EXPLORATION")
    print("Deep Analysis of Continuous Ternary Landscape")
    print("=" * 70)

    # Configuration
    if args.quick:
        num_samples = 10000
        map_resolution = 100
        triplets = 10000
        lattice_samples = 500
    else:
        num_samples = 50000
        map_resolution = 200
        triplets = 50000
        lattice_samples = 2000

    print(f"\nConfiguration:")
    print(f"  Samples per operation: {num_samples:,}")
    print(f"  GEMM map resolution: {map_resolution}x{map_resolution}")
    print(f"  Ultrametric triplets: {triplets:,}")

    # Load embeddings
    print("\nLoading embeddings...")
    embeddings_path = base_dir / "gemm_export" / "ternary_embeddings.npy"
    if not embeddings_path.exists():
        print(f"ERROR: {embeddings_path} not found. Run create_embedding_lut.py first.")
        return

    embeddings = np.load(embeddings_path)
    print(f"  Shape: {embeddings.shape}")

    explorer = ExtendedGEMMExplorer(embeddings)

    # ==========================================================================
    # 1. Extended Operation Exploration
    # ==========================================================================
    print("\n" + "=" * 70)
    print("1. EXTENDED OPERATION EXPLORATION")
    print("=" * 70)

    operations = ['add', 'mul', 'min', 'max']
    operation_results = explore_operations_extended(explorer, operations, num_samples)

    # ==========================================================================
    # 2. Multi-scale Lattice Analysis
    # ==========================================================================
    print("\n" + "=" * 70)
    print("2. MULTI-SCALE LATTICE ANALYSIS")
    print("=" * 70)

    lattice = analyze_multiscale_lattice(explorer, lattice_samples)

    print("\nDistance distribution peaks by resolution:")
    for res, peaks in lattice['peaks_by_resolution'].items():
        if peaks:
            print(f"  {res} bins: {peaks[:5]}...")

    print("\nCluster analysis:")
    for k, stats in lattice['cluster_analysis'].items():
        print(f"  {k}: {stats['num_non_empty']} clusters, "
              f"mean size={stats['mean_size']:.1f}")

    # ==========================================================================
    # 3. Extended Ultrametric Analysis
    # ==========================================================================
    print("\n" + "=" * 70)
    print("3. EXTENDED ULTRAMETRIC ANALYSIS")
    print("=" * 70)

    ultrametric = analyze_ultrametric_extended(explorer, triplets)

    print(f"\nUltrametric violation rate: {ultrametric['violation_rate']:.2%}")
    print(f"Mean violation magnitude: {ultrametric['mean_violation_magnitude']:.4f}")
    print(f"Euclidean-Ultrametric correlation: {ultrametric['euclidean_ultrametric_correlation']:.4f}")
    print(f"Mean hierarchy depth: {ultrametric['mean_hierarchy_depth']:.2f}")

    # ==========================================================================
    # 4. High-Resolution GEMM Maps
    # ==========================================================================
    print("\n" + "=" * 70)
    print("4. HIGH-RESOLUTION GEMM MAPS")
    print("=" * 70)

    gemm_maps = {}
    distance_maps = {}

    for op in ['add', 'mul']:
        gemm_map = generate_extended_gemm_map(explorer, map_resolution, op)
        gemm_maps[op] = gemm_map
        distance_maps[op] = gemm_map['distance_map']

        print(f"\n{op.upper()} Map ({map_resolution}x{map_resolution}):")
        print(f"  Match rate: {gemm_map['match_rate']:.2%}")
        print(f"  Mean distance: {gemm_map['mean_distance']:.4f}")
        print(f"  Max distance: {gemm_map['max_distance']:.4f}")
        print(f"  Std distance: {gemm_map['std_distance']:.4f}")

    # ==========================================================================
    # 5. Gap Structure Analysis
    # ==========================================================================
    print("\n" + "=" * 70)
    print("5. GAP STRUCTURE ANALYSIS")
    print("=" * 70)

    gap_analysis = analyze_gap_structure(distance_maps)

    # ==========================================================================
    # 6. Save Results
    # ==========================================================================
    print("\n" + "=" * 70)
    print("6. SAVING RESULTS")
    print("=" * 70)

    # Prepare report (convert numpy types)
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'num_ternary_values': int(explorer.N),
            'embedding_dim': int(explorer.dim),
            'num_samples_per_op': num_samples,
            'gemm_map_resolution': map_resolution,
            'num_ultrametric_triplets': triplets,
            'checkpoint_path': 'v5_11_homeostasis/epoch_20.pt'
        },
        'operations': {op: {
            'num_samples': int(results['num_samples']),
            'ultrametric_rate': float(results['ultrametric_rate']),
            'discrete_in_top1': float(results['discrete_in_top1_rate']),
            'discrete_in_top5': float(results['discrete_in_top5_rate']),
            'mean_dist_to_discrete': float(results['mean_dist_to_discrete']),
            'std_dist_to_discrete': float(results['std_dist_to_discrete']),
            'valuation_correlation': float(results['valuation_correlation']),
            'soft_gap_mean': float(np.mean(results['soft_gap_distribution'])),
            'soft_gap_max': float(np.max(results['soft_gap_distribution'])),
            'soft_gap_std': float(np.std(results['soft_gap_distribution']))
        } for op, results in operation_results.items()},
        'lattice': {
            'distance_stats': lattice['distance_stats'],
            'scale_analysis': lattice['scale_analysis'],
            'peaks_by_resolution': {
                str(k): v for k, v in lattice['peaks_by_resolution'].items()
            },
            'cluster_analysis': lattice['cluster_analysis']
        },
        'ultrametric': {
            'violation_rate': float(ultrametric['violation_rate']),
            'mean_violation_magnitude': float(ultrametric['mean_violation_magnitude']),
            'max_violation_magnitude': float(ultrametric['max_violation_magnitude']),
            'euclidean_ultrametric_correlation': float(ultrametric['euclidean_ultrametric_correlation']),
            'mean_hierarchy_depth': float(ultrametric['mean_hierarchy_depth'])
        },
        'gemm_maps': {op: {
            'resolution': int(data['resolution']),
            'match_rate': float(data['match_rate']),
            'mean_distance': float(data['mean_distance']),
            'max_distance': float(data['max_distance']),
            'std_distance': float(data['std_distance'])
        } for op, data in gemm_maps.items()},
        'gap_analysis': {op: {
            'threshold': float(data['threshold']),
            'high_gap_fraction': float(data['high_gap_fraction']),
            'num_gap_regions': int(data['num_gap_regions']),
            'mean_gap_size': float(data['mean_gap_size']),
            'max_gap_size': int(data['max_gap_size'])
        } for op, data in gap_analysis.items()}
    }

    # Save JSON report (convert all numpy types)
    report = convert_numpy_types(report)
    with open(output_dir / 'extended_exploration_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    # Save numpy arrays
    for op in gemm_maps:
        np.save(output_dir / f'extended_soft_map_{op}.npy', gemm_maps[op]['soft_map'])
        np.save(output_dir / f'extended_nearest_map_{op}.npy', gemm_maps[op]['nearest_map'])
        np.save(output_dir / f'extended_distance_map_{op}.npy', gemm_maps[op]['distance_map'])
        np.save(output_dir / f'extended_discrete_map_{op}.npy', gemm_maps[op]['discrete_map'])

    print(f"\nSaved to: {output_dir}")
    print(f"  extended_exploration_report.json")
    print(f"  extended_*_map_*.npy")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("EXTENDED EXPLORATION SUMMARY")
    print("=" * 70)

    print("\n1. OPERATION PATTERNS:")
    for op in operations:
        r = operation_results[op]
        print(f"   {op.upper()}: ultra={r['ultrametric_rate']:.1%}, "
              f"val_corr={r['valuation_correlation']:.3f}, "
              f"gap_mean={np.mean(r['soft_gap_distribution']):.4f}")

    print("\n2. HIERARCHICAL STRUCTURE:")
    print(f"   Ultrametric correlation: {ultrametric['euclidean_ultrametric_correlation']:.4f}")
    print(f"   Violation rate: {ultrametric['violation_rate']:.2%}")
    print(f"   Mean hierarchy depth: {ultrametric['mean_hierarchy_depth']:.2f}")

    print("\n3. LATTICE CHARACTERISTICS:")
    print(f"   Mean distance: {lattice['distance_stats']['mean']:.4f}")
    print(f"   Distance std: {lattice['distance_stats']['std']:.4f}")
    print(f"   Optimal clusters (k=20): {lattice['cluster_analysis']['k_20']['num_non_empty']} non-empty")

    print("\n4. GAP REGIONS (neural GEMM targets):")
    for op, data in gap_analysis.items():
        print(f"   {op.upper()}: {data['num_gap_regions']} regions, "
              f"{data['high_gap_fraction']:.1%} high-gap area")

    total_coverage = num_samples * len(operations) + map_resolution**2 * 2
    print(f"\n5. TOTAL COVERAGE: {total_coverage:,} samples")

    print("\n" + "=" * 70)
    print("EXPLORATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
