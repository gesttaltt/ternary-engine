#!/usr/bin/env python3
"""
Embedding Exactitude Score (EES) - Comprehensive Ternary Embedding Evaluation

Evaluates any ternary VAE embedding for its quality in capturing ternary operation
structure. Provides a single composite score plus detailed breakdown.

METRICS:
1. Operation Reconstruction Accuracy (ORA) - Does nearest neighbor match discrete op?
2. Ultrametric Preservation (UP) - Triplet inequality satisfaction rate
3. Valuation-Radius Correlation (VRC) - Does p-adic depth map to embedding geometry?
4. Gap Sparsity Score (GSS) - Coverage of operation space (fewer gaps = better)
5. Hierarchical Consistency (HC) - Does clustering match valuation levels?
6. Operation Linearity (OL) - How linear are operations in embedding space?

COMPOSITE SCORE:
    EES = w1*ORA + w2*UP + w3*VRC + w4*GSS + w5*HC + w6*OL

Default weights prioritize operation accuracy and ultrametric structure.

USAGE:
    python embedding_exactitude_score.py <embeddings.npy> [--samples N] [--output FILE]

OUTPUT:
    - Console summary with all metrics
    - Optional JSON file with full breakdown

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class OperationAccuracy:
    """Accuracy metrics for a single operation."""
    operation: str
    top1_accuracy: float      # Nearest neighbor matches discrete result
    top5_accuracy: float      # Discrete result in top-5 neighbors
    top10_accuracy: float     # Discrete result in top-10 neighbors
    mean_rank: float          # Mean rank of correct answer
    mean_distance: float      # Mean distance to correct discrete result
    num_samples: int


@dataclass
class UltrametricMetrics:
    """Ultrametric structure preservation metrics."""
    triplet_satisfaction: float    # Fraction satisfying d(a,c) <= max(d(a,b), d(b,c))
    mean_violation: float          # Mean violation magnitude when violated
    max_violation: float           # Maximum violation magnitude
    strong_ultrametric: float      # Fraction with d(a,c) = max(d(a,b), d(b,c))
    num_triplets: int


@dataclass
class ValuationMetrics:
    """Valuation-geometry correlation metrics."""
    radius_correlation: float      # Spearman(valuation, norm)
    radius_p_value: float
    layer_separation: float        # Mean distance between valuation layers
    layer_overlap: float           # Fraction of overlapping radius ranges
    radii_by_valuation: Dict[int, Dict[str, float]]  # Stats per valuation level


@dataclass
class GapMetrics:
    """Gap/coverage metrics for embedding space."""
    coverage_score: float          # 1 - (gap_volume / total_volume estimate)
    mean_nearest_distance: float   # Mean distance to nearest discrete point
    max_nearest_distance: float    # Maximum gap size
    gap_fraction: float            # Fraction of samples in high-gap regions
    num_gap_regions: int           # Number of distinct gap regions
    gap_size_distribution: List[float]  # Sizes of gap regions


@dataclass
class HierarchyMetrics:
    """Hierarchical clustering consistency metrics."""
    cluster_valuation_ami: float   # Adjusted mutual info: clusters vs valuation
    dendrogram_correlation: float  # Correlation between merge height and valuation diff
    level_purity: float            # Mean purity of valuation levels in clusters


@dataclass
class LinearityMetrics:
    """Operation linearity in embedding space."""
    add_linearity: float           # R² of linear fit for addition
    mul_linearity: float           # R² of linear fit for multiplication
    min_linearity: float           # R² of linear fit for min
    max_linearity: float           # R² of linear fit for max
    mean_linearity: float          # Average across operations


@dataclass
class EmbeddingExactitudeScore:
    """Complete embedding evaluation results."""
    # Identification
    embedding_path: str
    timestamp: str
    num_values: int
    embedding_dim: int

    # Component metrics
    operation_accuracy: Dict[str, OperationAccuracy]
    ultrametric: UltrametricMetrics
    valuation: ValuationMetrics
    gaps: GapMetrics
    hierarchy: HierarchyMetrics
    linearity: LinearityMetrics

    # Composite scores
    ora_score: float       # Operation Reconstruction Accuracy (0-1)
    up_score: float        # Ultrametric Preservation (0-1)
    vrc_score: float       # Valuation-Radius Correlation (0-1)
    gss_score: float       # Gap Sparsity Score (0-1)
    hc_score: float        # Hierarchical Consistency (0-1)
    ol_score: float        # Operation Linearity (0-1)

    # Final composite
    ees_composite: float   # Weighted combination (0-1)
    ees_grade: str         # Letter grade A-F

    # Weights used
    weights: Dict[str, float]


# =============================================================================
# TERNARY OPERATIONS
# =============================================================================

def valuation_3adic(n: int, max_val: int = 9) -> int:
    """Compute 3-adic valuation of n."""
    if n == 0:
        return max_val
    v = 0
    while n % 3 == 0 and v < max_val:
        n //= 3
        v += 1
    return v


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


def discrete_tadd(a: int, b: int, num_trits: int = 9) -> int:
    """Balanced ternary addition with carry."""
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


def discrete_tmul(a: int, b: int, num_trits: int = 9) -> int:
    """Element-wise ternary multiplication."""
    trits_a = index_to_trits(a, num_trits)
    trits_b = index_to_trits(b, num_trits)
    result = trits_a * trits_b
    return trits_to_index(result)


def discrete_tmin(a: int, b: int, num_trits: int = 9) -> int:
    """Element-wise ternary minimum."""
    trits_a = index_to_trits(a, num_trits)
    trits_b = index_to_trits(b, num_trits)
    result = np.minimum(trits_a, trits_b)
    return trits_to_index(result)


def discrete_tmax(a: int, b: int, num_trits: int = 9) -> int:
    """Element-wise ternary maximum."""
    trits_a = index_to_trits(a, num_trits)
    trits_b = index_to_trits(b, num_trits)
    result = np.maximum(trits_a, trits_b)
    return trits_to_index(result)


OPERATIONS = {
    'add': discrete_tadd,
    'mul': discrete_tmul,
    'min': discrete_tmin,
    'max': discrete_tmax,
}


# =============================================================================
# METRIC COMPUTATION
# =============================================================================

def compute_operation_accuracy(
    embeddings: np.ndarray,
    num_samples: int = 10000,
    seed: int = 42
) -> Dict[str, OperationAccuracy]:
    """Compute operation reconstruction accuracy for all operations."""
    N = embeddings.shape[0]
    num_trits = int(np.log(N) / np.log(3) + 0.5)

    results = {}

    for op_name, op_func in OPERATIONS.items():
        np.random.seed(seed + hash(op_name) % 10000)

        pairs_a = np.random.randint(0, N, size=num_samples)
        pairs_b = np.random.randint(0, N, size=num_samples)

        top1_correct = 0
        top5_correct = 0
        top10_correct = 0
        ranks = []
        distances = []

        for i in range(num_samples):
            a, b = pairs_a[i], pairs_b[i]

            # Discrete operation result
            discrete_result = op_func(a, b, num_trits)

            # Soft operation: midpoint in embedding space
            soft_emb = (embeddings[a] + embeddings[b]) / 2

            # Find distances to all discrete points
            dists = np.linalg.norm(embeddings - soft_emb, axis=1)
            sorted_indices = np.argsort(dists)

            # Check accuracy
            rank = np.where(sorted_indices == discrete_result)[0][0]
            ranks.append(rank)
            distances.append(dists[discrete_result])

            if rank == 0:
                top1_correct += 1
            if rank < 5:
                top5_correct += 1
            if rank < 10:
                top10_correct += 1

        results[op_name] = OperationAccuracy(
            operation=op_name,
            top1_accuracy=top1_correct / num_samples,
            top5_accuracy=top5_correct / num_samples,
            top10_accuracy=top10_correct / num_samples,
            mean_rank=float(np.mean(ranks)),
            mean_distance=float(np.mean(distances)),
            num_samples=num_samples
        )

    return results


def compute_ultrametric_metrics(
    embeddings: np.ndarray,
    num_triplets: int = 50000,
    seed: int = 123
) -> UltrametricMetrics:
    """Compute ultrametric preservation metrics."""
    N = embeddings.shape[0]
    np.random.seed(seed)

    triplets_a = np.random.randint(0, N, size=num_triplets)
    triplets_b = np.random.randint(0, N, size=num_triplets)
    triplets_c = np.random.randint(0, N, size=num_triplets)

    satisfied = 0
    strong_satisfied = 0
    violations = []

    for i in range(num_triplets):
        a, b, c = triplets_a[i], triplets_b[i], triplets_c[i]

        d_ab = np.linalg.norm(embeddings[a] - embeddings[b])
        d_bc = np.linalg.norm(embeddings[b] - embeddings[c])
        d_ac = np.linalg.norm(embeddings[a] - embeddings[c])

        max_side = max(d_ab, d_bc)

        # Check ultrametric inequality
        if d_ac <= max_side + 1e-6:
            satisfied += 1
            # Check strong ultrametric (equality)
            if abs(d_ac - max_side) < 1e-4:
                strong_satisfied += 1
        else:
            violations.append(d_ac - max_side)

    return UltrametricMetrics(
        triplet_satisfaction=satisfied / num_triplets,
        mean_violation=float(np.mean(violations)) if violations else 0.0,
        max_violation=float(np.max(violations)) if violations else 0.0,
        strong_ultrametric=strong_satisfied / num_triplets,
        num_triplets=num_triplets
    )


def compute_valuation_metrics(embeddings: np.ndarray) -> ValuationMetrics:
    """Compute valuation-geometry correlation metrics."""
    N = embeddings.shape[0]
    num_trits = int(np.log(N) / np.log(3) + 0.5)

    # Compute valuations and radii
    valuations = np.array([valuation_3adic(i, num_trits) for i in range(N)])
    radii = np.linalg.norm(embeddings, axis=1)

    # Correlation
    corr, p_value = spearmanr(valuations, radii)

    # Layer statistics
    radii_by_val = {}
    for v in range(num_trits + 1):
        mask = valuations == v
        if np.sum(mask) > 0:
            r = radii[mask]
            radii_by_val[v] = {
                'count': int(np.sum(mask)),
                'mean': float(np.mean(r)),
                'std': float(np.std(r)),
                'min': float(np.min(r)),
                'max': float(np.max(r))
            }

    # Layer separation: mean distance between consecutive layer means
    layer_means = [radii_by_val[v]['mean'] for v in sorted(radii_by_val.keys())]
    if len(layer_means) > 1:
        separations = [abs(layer_means[i+1] - layer_means[i]) for i in range(len(layer_means)-1)]
        layer_separation = float(np.mean(separations))
    else:
        layer_separation = 0.0

    # Layer overlap: fraction of layers with overlapping radius ranges
    overlaps = 0
    total_pairs = 0
    sorted_vals = sorted(radii_by_val.keys())
    for i in range(len(sorted_vals) - 1):
        v1, v2 = sorted_vals[i], sorted_vals[i+1]
        r1_max = radii_by_val[v1]['max']
        r2_min = radii_by_val[v2]['min']
        if r1_max > r2_min:
            overlaps += 1
        total_pairs += 1

    layer_overlap = overlaps / total_pairs if total_pairs > 0 else 0.0

    return ValuationMetrics(
        radius_correlation=float(corr) if not np.isnan(corr) else 0.0,
        radius_p_value=float(p_value) if not np.isnan(p_value) else 1.0,
        layer_separation=layer_separation,
        layer_overlap=layer_overlap,
        radii_by_valuation=radii_by_val
    )


def compute_gap_metrics(
    embeddings: np.ndarray,
    num_samples: int = 10000,
    seed: int = 456
) -> GapMetrics:
    """Compute gap/coverage metrics."""
    N, D = embeddings.shape
    np.random.seed(seed)

    # Sample random points in embedding space
    # Use convex hull approximation: random convex combinations
    emb_min = embeddings.min(axis=0)
    emb_max = embeddings.max(axis=0)

    # Random points in bounding box
    random_points = np.random.uniform(emb_min, emb_max, size=(num_samples, D))

    # Also sample interpolations between discrete points
    pairs_a = np.random.randint(0, N, size=num_samples)
    pairs_b = np.random.randint(0, N, size=num_samples)
    alphas = np.random.uniform(0, 1, size=num_samples)
    interp_points = (1 - alphas[:, None]) * embeddings[pairs_a] + alphas[:, None] * embeddings[pairs_b]

    # Combine samples
    all_samples = np.vstack([random_points, interp_points])

    # Find nearest discrete point for each sample
    nearest_distances = []
    for i in range(len(all_samples)):
        dists = np.linalg.norm(embeddings - all_samples[i], axis=1)
        nearest_distances.append(np.min(dists))

    nearest_distances = np.array(nearest_distances)

    # Gap threshold: mean + 1.5*std
    threshold = np.mean(nearest_distances) + 1.5 * np.std(nearest_distances)
    high_gap_mask = nearest_distances > threshold
    gap_fraction = np.mean(high_gap_mask)

    # Estimate coverage score (inverse of normalized gap)
    max_possible_dist = np.linalg.norm(emb_max - emb_min)
    coverage_score = 1.0 - (np.mean(nearest_distances) / (max_possible_dist + 1e-8))
    coverage_score = max(0.0, min(1.0, coverage_score))

    # Count gap regions (simplified: just count high-gap samples)
    num_gap_regions = int(np.sum(high_gap_mask))

    return GapMetrics(
        coverage_score=coverage_score,
        mean_nearest_distance=float(np.mean(nearest_distances)),
        max_nearest_distance=float(np.max(nearest_distances)),
        gap_fraction=gap_fraction,
        num_gap_regions=num_gap_regions,
        gap_size_distribution=nearest_distances[high_gap_mask].tolist()[:100]  # Top 100
    )


def compute_hierarchy_metrics(
    embeddings: np.ndarray,
    sample_size: int = 1000,
    seed: int = 789
) -> HierarchyMetrics:
    """Compute hierarchical clustering consistency metrics."""
    N = embeddings.shape[0]
    num_trits = int(np.log(N) / np.log(3) + 0.5)

    np.random.seed(seed)
    sample_idx = np.random.choice(N, size=min(sample_size, N), replace=False)
    sample_emb = embeddings[sample_idx]
    sample_vals = np.array([valuation_3adic(i, num_trits) for i in sample_idx])

    # Hierarchical clustering
    Z = linkage(sample_emb, method='ward')

    # Cluster at different levels and check valuation purity
    purities = []
    for n_clusters in [5, 10, 20, 50]:
        labels = fcluster(Z, n_clusters, criterion='maxclust')

        # Compute purity: for each cluster, fraction of dominant valuation
        cluster_purities = []
        for c in range(1, n_clusters + 1):
            mask = labels == c
            if np.sum(mask) > 0:
                vals_in_cluster = sample_vals[mask]
                most_common = np.bincount(vals_in_cluster).argmax()
                purity = np.mean(vals_in_cluster == most_common)
                cluster_purities.append(purity)

        if cluster_purities:
            purities.append(np.mean(cluster_purities))

    level_purity = float(np.mean(purities)) if purities else 0.0

    # Dendrogram correlation: merge height vs valuation difference
    # For each merge, compute valuation difference between merged clusters
    merge_heights = Z[:, 2]
    val_diffs = []

    # Simplified: correlation between pairwise distances and valuation differences
    if sample_size <= 500:
        pairwise_dists = pdist(sample_emb)
        val_diff_matrix = []
        for i in range(len(sample_idx)):
            for j in range(i + 1, len(sample_idx)):
                val_diff_matrix.append(abs(sample_vals[i] - sample_vals[j]))
        val_diff_matrix = np.array(val_diff_matrix)

        dendrogram_corr, _ = spearmanr(pairwise_dists, val_diff_matrix)
        dendrogram_corr = float(dendrogram_corr) if not np.isnan(dendrogram_corr) else 0.0
    else:
        dendrogram_corr = 0.0

    # AMI between clusters and valuation (using k=10 clusters)
    try:
        from sklearn.metrics import adjusted_mutual_info_score
        labels_10 = fcluster(Z, 10, criterion='maxclust')
        ami = adjusted_mutual_info_score(sample_vals, labels_10)
    except ImportError:
        ami = 0.0

    return HierarchyMetrics(
        cluster_valuation_ami=float(ami),
        dendrogram_correlation=dendrogram_corr,
        level_purity=level_purity
    )


def compute_linearity_metrics(
    embeddings: np.ndarray,
    num_samples: int = 5000,
    seed: int = 321
) -> LinearityMetrics:
    """Compute operation linearity in embedding space."""
    N = embeddings.shape[0]
    num_trits = int(np.log(N) / np.log(3) + 0.5)

    results = {}

    for op_name, op_func in OPERATIONS.items():
        np.random.seed(seed + hash(op_name) % 10000)

        pairs_a = np.random.randint(0, N, size=num_samples)
        pairs_b = np.random.randint(0, N, size=num_samples)

        # Input: concatenated embeddings
        X = np.hstack([embeddings[pairs_a], embeddings[pairs_b]])

        # Output: embedding of discrete result
        discrete_results = [op_func(pairs_a[i], pairs_b[i], num_trits) for i in range(num_samples)]
        Y = embeddings[discrete_results]

        # Linear regression: Y = X @ W + b
        # Use least squares to find best linear approximation
        X_aug = np.hstack([X, np.ones((num_samples, 1))])  # Add bias term

        try:
            W, residuals, rank, s = np.linalg.lstsq(X_aug, Y, rcond=None)
            Y_pred = X_aug @ W

            # R² score
            ss_res = np.sum((Y - Y_pred) ** 2)
            ss_tot = np.sum((Y - np.mean(Y, axis=0)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            r2 = max(0.0, min(1.0, r2))
        except:
            r2 = 0.0

        results[op_name] = r2

    return LinearityMetrics(
        add_linearity=results.get('add', 0.0),
        mul_linearity=results.get('mul', 0.0),
        min_linearity=results.get('min', 0.0),
        max_linearity=results.get('max', 0.0),
        mean_linearity=float(np.mean(list(results.values())))
    )


# =============================================================================
# COMPOSITE SCORE COMPUTATION
# =============================================================================

def compute_composite_scores(
    op_accuracy: Dict[str, OperationAccuracy],
    ultrametric: UltrametricMetrics,
    valuation: ValuationMetrics,
    gaps: GapMetrics,
    hierarchy: HierarchyMetrics,
    linearity: LinearityMetrics,
    weights: Optional[Dict[str, float]] = None
) -> Tuple[Dict[str, float], float, str]:
    """Compute component scores and final composite."""

    if weights is None:
        weights = {
            'ora': 0.25,   # Operation Reconstruction Accuracy
            'up': 0.20,    # Ultrametric Preservation
            'vrc': 0.20,   # Valuation-Radius Correlation
            'gss': 0.15,   # Gap Sparsity Score
            'hc': 0.10,    # Hierarchical Consistency
            'ol': 0.10,    # Operation Linearity
        }

    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    # ORA: Average top-10 accuracy across operations (more lenient than top-1)
    ora_score = np.mean([op.top10_accuracy for op in op_accuracy.values()])

    # UP: Triplet satisfaction rate
    up_score = ultrametric.triplet_satisfaction

    # VRC: Absolute correlation (we want strong correlation, positive or negative)
    vrc_score = abs(valuation.radius_correlation)

    # GSS: Coverage score (already 0-1)
    gss_score = gaps.coverage_score

    # HC: Combination of AMI and purity
    hc_score = 0.5 * (hierarchy.cluster_valuation_ami + 1) / 2 + 0.5 * hierarchy.level_purity
    hc_score = max(0.0, min(1.0, hc_score))

    # OL: Mean linearity
    ol_score = linearity.mean_linearity

    scores = {
        'ora': ora_score,
        'up': up_score,
        'vrc': vrc_score,
        'gss': gss_score,
        'hc': hc_score,
        'ol': ol_score,
    }

    # Composite score
    ees = sum(weights[k] * scores[k] for k in weights)

    # Grade
    if ees >= 0.9:
        grade = 'A+'
    elif ees >= 0.85:
        grade = 'A'
    elif ees >= 0.80:
        grade = 'A-'
    elif ees >= 0.75:
        grade = 'B+'
    elif ees >= 0.70:
        grade = 'B'
    elif ees >= 0.65:
        grade = 'B-'
    elif ees >= 0.60:
        grade = 'C+'
    elif ees >= 0.55:
        grade = 'C'
    elif ees >= 0.50:
        grade = 'C-'
    elif ees >= 0.45:
        grade = 'D+'
    elif ees >= 0.40:
        grade = 'D'
    else:
        grade = 'F'

    return scores, ees, grade


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate_embeddings(
    embeddings_path: str,
    num_samples: int = 10000,
    weights: Optional[Dict[str, float]] = None,
    verbose: bool = True
) -> EmbeddingExactitudeScore:
    """Run complete embedding evaluation."""

    t_start = time.time()

    # Load embeddings
    if verbose:
        print(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    N, D = embeddings.shape

    if verbose:
        print(f"  Shape: {N} values x {D} dimensions")
        print(f"  Memory: {embeddings.nbytes / 1024 / 1024:.2f} MB")

    # Compute all metrics
    if verbose:
        print("\n" + "=" * 70)
        print("COMPUTING METRICS")
        print("=" * 70)

    # 1. Operation Accuracy
    if verbose:
        print("\n[1/6] Operation Reconstruction Accuracy...")
    t0 = time.time()
    op_accuracy = compute_operation_accuracy(embeddings, num_samples)
    if verbose:
        print(f"  Done in {time.time() - t0:.1f}s")
        for op, acc in op_accuracy.items():
            print(f"    {op.upper()}: top1={acc.top1_accuracy:.2%}, top10={acc.top10_accuracy:.2%}")

    # 2. Ultrametric Preservation
    if verbose:
        print("\n[2/6] Ultrametric Preservation...")
    t0 = time.time()
    ultrametric = compute_ultrametric_metrics(embeddings, num_samples * 5)
    if verbose:
        print(f"  Done in {time.time() - t0:.1f}s")
        print(f"    Satisfaction: {ultrametric.triplet_satisfaction:.2%}")
        print(f"    Strong ultrametric: {ultrametric.strong_ultrametric:.2%}")

    # 3. Valuation-Radius Correlation
    if verbose:
        print("\n[3/6] Valuation-Radius Correlation...")
    t0 = time.time()
    valuation = compute_valuation_metrics(embeddings)
    if verbose:
        print(f"  Done in {time.time() - t0:.1f}s")
        print(f"    Correlation: {valuation.radius_correlation:.4f}")
        print(f"    Layer separation: {valuation.layer_separation:.4f}")
        print(f"    Layer overlap: {valuation.layer_overlap:.2%}")

    # 4. Gap Metrics
    if verbose:
        print("\n[4/6] Gap/Coverage Analysis...")
    t0 = time.time()
    gaps = compute_gap_metrics(embeddings, num_samples)
    if verbose:
        print(f"  Done in {time.time() - t0:.1f}s")
        print(f"    Coverage: {gaps.coverage_score:.2%}")
        print(f"    Mean gap: {gaps.mean_nearest_distance:.4f}")
        print(f"    Gap fraction: {gaps.gap_fraction:.2%}")

    # 5. Hierarchy Metrics
    if verbose:
        print("\n[5/6] Hierarchical Consistency...")
    t0 = time.time()
    hierarchy = compute_hierarchy_metrics(embeddings, min(1000, N))
    if verbose:
        print(f"  Done in {time.time() - t0:.1f}s")
        print(f"    Cluster-valuation AMI: {hierarchy.cluster_valuation_ami:.4f}")
        print(f"    Level purity: {hierarchy.level_purity:.2%}")

    # 6. Linearity Metrics
    if verbose:
        print("\n[6/6] Operation Linearity...")
    t0 = time.time()
    linearity = compute_linearity_metrics(embeddings, num_samples // 2)
    if verbose:
        print(f"  Done in {time.time() - t0:.1f}s")
        print(f"    Mean linearity (R²): {linearity.mean_linearity:.4f}")

    # Compute composite scores
    scores, ees, grade = compute_composite_scores(
        op_accuracy, ultrametric, valuation, gaps, hierarchy, linearity, weights
    )

    # Build result object
    result = EmbeddingExactitudeScore(
        embedding_path=str(embeddings_path),
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        num_values=N,
        embedding_dim=D,
        operation_accuracy=op_accuracy,
        ultrametric=ultrametric,
        valuation=valuation,
        gaps=gaps,
        hierarchy=hierarchy,
        linearity=linearity,
        ora_score=scores['ora'],
        up_score=scores['up'],
        vrc_score=scores['vrc'],
        gss_score=scores['gss'],
        hc_score=scores['hc'],
        ol_score=scores['ol'],
        ees_composite=ees,
        ees_grade=grade,
        weights=weights or {
            'ora': 0.25, 'up': 0.20, 'vrc': 0.20,
            'gss': 0.15, 'hc': 0.10, 'ol': 0.10
        }
    )

    if verbose:
        print(f"\nTotal evaluation time: {time.time() - t_start:.1f}s")

    return result


def print_report(result: EmbeddingExactitudeScore):
    """Print comprehensive evaluation report."""
    print("\n" + "=" * 70)
    print("EMBEDDING EXACTITUDE SCORE (EES) REPORT")
    print("=" * 70)

    print(f"\nEmbedding: {result.embedding_path}")
    print(f"Timestamp: {result.timestamp}")
    print(f"Values: {result.num_values:,} | Dimensions: {result.embedding_dim}")

    print("\n" + "-" * 70)
    print("COMPONENT SCORES")
    print("-" * 70)

    components = [
        ('ORA', 'Operation Reconstruction Accuracy', result.ora_score, result.weights.get('ora', 0.25)),
        ('UP', 'Ultrametric Preservation', result.up_score, result.weights.get('up', 0.20)),
        ('VRC', 'Valuation-Radius Correlation', result.vrc_score, result.weights.get('vrc', 0.20)),
        ('GSS', 'Gap Sparsity Score', result.gss_score, result.weights.get('gss', 0.15)),
        ('HC', 'Hierarchical Consistency', result.hc_score, result.weights.get('hc', 0.10)),
        ('OL', 'Operation Linearity', result.ol_score, result.weights.get('ol', 0.10)),
    ]

    print(f"\n{'Component':<6} {'Description':<35} {'Score':>8} {'Weight':>8} {'Contrib':>8}")
    print("-" * 70)
    for abbr, desc, score, weight in components:
        contrib = score * weight
        bar = '█' * int(score * 20) + '░' * (20 - int(score * 20))
        print(f"{abbr:<6} {desc:<35} {score:>7.2%} {weight:>7.0%} {contrib:>7.2%}")

    print("-" * 70)
    print(f"{'TOTAL':<6} {'Embedding Exactitude Score':<35} {result.ees_composite:>7.2%}   {'100%':>7} {result.ees_composite:>7.2%}")

    print("\n" + "-" * 70)
    print("FINAL GRADE")
    print("-" * 70)

    grade_colors = {
        'A+': '🌟', 'A': '✨', 'A-': '⭐',
        'B+': '👍', 'B': '👌', 'B-': '🆗',
        'C+': '⚠️', 'C': '⚡', 'C-': '⬇️',
        'D+': '❗', 'D': '❌', 'F': '🚫'
    }

    print(f"\n   {grade_colors.get(result.ees_grade, '')} GRADE: {result.ees_grade}  (EES = {result.ees_composite:.4f})")

    print("\n" + "-" * 70)
    print("DETAILED METRICS")
    print("-" * 70)

    print("\n[Operation Accuracy]")
    for op, acc in result.operation_accuracy.items():
        print(f"  {op.upper()}: top1={acc.top1_accuracy:.2%}, top5={acc.top5_accuracy:.2%}, "
              f"top10={acc.top10_accuracy:.2%}, mean_rank={acc.mean_rank:.1f}")

    print("\n[Ultrametric Structure]")
    print(f"  Triplet satisfaction: {result.ultrametric.triplet_satisfaction:.2%}")
    print(f"  Strong ultrametric: {result.ultrametric.strong_ultrametric:.2%}")
    print(f"  Mean violation: {result.ultrametric.mean_violation:.4f}")
    print(f"  Max violation: {result.ultrametric.max_violation:.4f}")

    print("\n[Valuation-Geometry]")
    print(f"  Radius correlation: {result.valuation.radius_correlation:.4f}")
    print(f"  Layer separation: {result.valuation.layer_separation:.4f}")
    print(f"  Layer overlap: {result.valuation.layer_overlap:.2%}")

    print("\n[Coverage/Gaps]")
    print(f"  Coverage score: {result.gaps.coverage_score:.2%}")
    print(f"  Mean nearest distance: {result.gaps.mean_nearest_distance:.4f}")
    print(f"  Max gap: {result.gaps.max_nearest_distance:.4f}")
    print(f"  Gap fraction: {result.gaps.gap_fraction:.2%}")

    print("\n[Hierarchy]")
    print(f"  Cluster-valuation AMI: {result.hierarchy.cluster_valuation_ami:.4f}")
    print(f"  Dendrogram correlation: {result.hierarchy.dendrogram_correlation:.4f}")
    print(f"  Level purity: {result.hierarchy.level_purity:.2%}")

    print("\n[Linearity]")
    print(f"  ADD: {result.linearity.add_linearity:.4f}")
    print(f"  MUL: {result.linearity.mul_linearity:.4f}")
    print(f"  MIN: {result.linearity.min_linearity:.4f}")
    print(f"  MAX: {result.linearity.max_linearity:.4f}")

    print("\n" + "=" * 70)


def save_report(result: EmbeddingExactitudeScore, output_path: str):
    """Save evaluation results to JSON."""

    def to_serializable(obj):
        if hasattr(obj, '__dict__'):
            return {k: to_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    data = to_serializable(result)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved detailed report to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Embedding Exactitude Score (EES) - Evaluate Ternary VAE Embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python embedding_exactitude_score.py embeddings.npy
  python embedding_exactitude_score.py embeddings.npy --samples 50000 --output ees_report.json
  python embedding_exactitude_score.py embeddings.npy --quick

Metrics computed:
  ORA  - Operation Reconstruction Accuracy
  UP   - Ultrametric Preservation
  VRC  - Valuation-Radius Correlation
  GSS  - Gap Sparsity Score
  HC   - Hierarchical Consistency
  OL   - Operation Linearity
"""
    )

    parser.add_argument('embeddings', type=str, help='Path to embeddings .npy file')
    parser.add_argument('--samples', type=int, default=10000, help='Samples per metric (default: 10000)')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer samples)')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')

    args = parser.parse_args()

    if args.quick:
        args.samples = 2000

    # Check file exists
    if not Path(args.embeddings).exists():
        print(f"ERROR: File not found: {args.embeddings}")
        return 1

    # Run evaluation
    result = evaluate_embeddings(
        args.embeddings,
        num_samples=args.samples,
        verbose=not args.quiet
    )

    # Print report
    if not args.quiet:
        print_report(result)
    else:
        print(f"\nEES: {result.ees_composite:.4f} (Grade: {result.ees_grade})")

    # Save if requested
    if args.output:
        save_report(result, args.output)
    else:
        # Default output path
        embeddings_path = Path(args.embeddings)
        default_output = embeddings_path.parent / f"ees_report_{embeddings_path.stem}.json"
        save_report(result, str(default_output))

    return 0


if __name__ == "__main__":
    exit(main())
