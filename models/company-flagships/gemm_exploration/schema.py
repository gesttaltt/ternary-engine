#!/usr/bin/env python3
"""
GEMM Exploration Results Schema and Utilities

Provides structured data classes and I/O utilities for GEMM space exploration
results. Enables programmatic access and analysis across exploration runs.

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

USAGE:
    from gemm_exploration.schema import ExplorationResults, load_results
    results = load_results("exploration_report.json")
    print(results.ultrametric.correlation)
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path
import json
import numpy as np


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class OperationMetrics:
    """Metrics for a single ternary operation (add, mul, min, max)."""
    name: str
    ultrametric_rate: float          # Fraction of pairs satisfying ultrametric inequality
    discrete_in_top1: float          # Fraction where discrete result is nearest neighbor
    discrete_in_top5: float          # Fraction where discrete result is in top-5 neighbors
    mean_dist_to_discrete: float     # Mean distance from midpoint to discrete result
    soft_gap_mean: float             # Mean distance from continuous to nearest discrete
    soft_gap_max: float              # Max distance (largest "gap" in space)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, name: str, d: Dict[str, Any]) -> 'OperationMetrics':
        return cls(name=name, **d)


@dataclass
class LatticeStructure:
    """Lattice-like structure analysis results."""
    distance_peaks: List[float]      # Peak distances (potential lattice spacings)
    mean_distance: float             # Mean pairwise distance
    std_distance: float              # Std of pairwise distances
    num_peaks: int = field(init=False)
    peak_spacing_mean: float = field(init=False)
    peak_spacing_std: float = field(init=False)

    def __post_init__(self):
        self.num_peaks = len(self.distance_peaks)
        if self.num_peaks > 1:
            spacings = np.diff(sorted(self.distance_peaks))
            self.peak_spacing_mean = float(np.mean(spacings))
            self.peak_spacing_std = float(np.std(spacings))
        else:
            self.peak_spacing_mean = 0.0
            self.peak_spacing_std = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'distance_peaks': self.distance_peaks,
            'mean_distance': self.mean_distance,
            'std_distance': self.std_distance,
            'num_peaks': self.num_peaks,
            'peak_spacing_mean': self.peak_spacing_mean,
            'peak_spacing_std': self.peak_spacing_std
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LatticeStructure':
        return cls(
            distance_peaks=d['distance_peaks'],
            mean_distance=d['mean_distance'],
            std_distance=d['std_distance']
        )


@dataclass
class UltrametricStructure:
    """Ultrametric structure analysis results."""
    correlation: float               # Euclidean vs ultrametric distance correlation
    is_strong: bool = field(init=False)      # correlation > 0.7
    is_moderate: bool = field(init=False)    # 0.4 < correlation <= 0.7
    structure_type: str = field(init=False)  # 'tree-like', 'moderate', 'complex'

    def __post_init__(self):
        self.is_strong = self.correlation > 0.7
        self.is_moderate = 0.4 < self.correlation <= 0.7
        if self.is_strong:
            self.structure_type = 'tree-like'
        elif self.is_moderate:
            self.structure_type = 'moderate-hierarchical'
        else:
            self.structure_type = 'complex-non-hierarchical'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'correlation': self.correlation,
            'is_strong': self.is_strong,
            'is_moderate': self.is_moderate,
            'structure_type': self.structure_type
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'UltrametricStructure':
        return cls(correlation=d.get('euclidean_ultrametric_correlation', d.get('correlation')))


@dataclass
class GEMMMapMetrics:
    """Metrics for a soft GEMM map."""
    operation: str
    resolution: int                  # Grid resolution (e.g., 100 for 100x100)
    match_rate: float                # Rate where soft matches discrete
    mean_distance: float             # Mean distance to nearest discrete
    max_distance: float              # Max distance (largest gap)
    coverage: int = field(init=False)  # Total samples (resolution^2)

    def __post_init__(self):
        self.coverage = self.resolution * self.resolution

    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation': self.operation,
            'resolution': self.resolution,
            'match_rate': self.match_rate,
            'mean_distance': self.mean_distance,
            'max_distance': self.max_distance,
            'coverage': self.coverage
        }

    @classmethod
    def from_dict(cls, op: str, d: Dict[str, Any], resolution: int = 100) -> 'GEMMMapMetrics':
        return cls(
            operation=op,
            resolution=resolution,
            match_rate=d['match_rate'],
            mean_distance=d['mean_distance'],
            max_distance=d['max_distance']
        )


@dataclass
class ExplorationConfig:
    """Configuration for an exploration run."""
    num_ternary_values: int          # 19683 for 9-trit values
    embedding_dim: int               # 16 for our VAE
    num_samples_per_op: int          # Random pairs sampled per operation
    gemm_map_resolution: int         # Grid resolution for GEMM maps
    checkpoint_path: str             # Source checkpoint

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExplorationConfig':
        return cls(**d)


@dataclass
class ExplorationResults:
    """Complete results from a GEMM space exploration run."""
    timestamp: str
    config: ExplorationConfig
    operations: Dict[str, OperationMetrics]
    lattice: LatticeStructure
    ultrametric: UltrametricStructure
    gemm_maps: Dict[str, GEMMMapMetrics]

    # Derived summary metrics
    total_coverage: int = field(init=False)
    best_ultrametric_op: str = field(init=False)
    smallest_gap_op: str = field(init=False)

    def __post_init__(self):
        # Calculate total coverage
        op_coverage = self.config.num_samples_per_op * len(self.operations)
        map_coverage = sum(m.coverage for m in self.gemm_maps.values())
        self.total_coverage = op_coverage + map_coverage

        # Find best operations
        self.best_ultrametric_op = max(
            self.operations.keys(),
            key=lambda op: self.operations[op].ultrametric_rate
        )
        self.smallest_gap_op = min(
            self.operations.keys(),
            key=lambda op: self.operations[op].soft_gap_mean
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'config': self.config.to_dict(),
            'operations': {op: m.to_dict() for op, m in self.operations.items()},
            'lattice': self.lattice.to_dict(),
            'ultrametric': self.ultrametric.to_dict(),
            'gemm_maps': {op: m.to_dict() for op, m in self.gemm_maps.items()},
            'summary': {
                'total_coverage': self.total_coverage,
                'best_ultrametric_op': self.best_ultrametric_op,
                'smallest_gap_op': self.smallest_gap_op
            }
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "GEMM SPACE EXPLORATION RESULTS",
            "=" * 70,
            f"Timestamp: {self.timestamp}",
            f"Ternary values: {self.config.num_ternary_values}",
            f"Embedding dim: {self.config.embedding_dim}",
            f"Total coverage: {self.total_coverage:,} samples",
            "",
            "ULTRAMETRIC STRUCTURE:",
            f"  Correlation: {self.ultrametric.correlation:.4f}",
            f"  Type: {self.ultrametric.structure_type}",
            "",
            "LATTICE STRUCTURE:",
            f"  Peaks found: {self.lattice.num_peaks}",
            f"  Peak spacing: {self.lattice.peak_spacing_mean:.4f} +/- {self.lattice.peak_spacing_std:.4f}",
            f"  Mean distance: {self.lattice.mean_distance:.4f}",
            "",
            "OPERATIONS:",
        ]
        for op, m in self.operations.items():
            lines.append(f"  {op.upper()}: ultrametric={m.ultrametric_rate:.1%}, "
                        f"gap_mean={m.soft_gap_mean:.4f}, gap_max={m.soft_gap_max:.4f}")
        lines.extend([
            "",
            "GEMM MAPS:",
        ])
        for op, m in self.gemm_maps.items():
            lines.append(f"  {op.upper()}: match={m.match_rate:.2%}, "
                        f"mean_dist={m.mean_distance:.4f}, max_dist={m.max_distance:.4f}")
        lines.append("=" * 70)
        return "\n".join(lines)


# =============================================================================
# I/O UTILITIES
# =============================================================================

def load_results(path: Path | str) -> ExplorationResults:
    """Load exploration results from JSON file."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    # Handle both old format (from explore_gemm_space.py) and new format
    if 'config' in data:
        config = ExplorationConfig.from_dict(data['config'])
    else:
        # Infer config from old format
        config = ExplorationConfig(
            num_ternary_values=data.get('num_ternary_values', 19683),
            embedding_dim=data.get('embedding_dim', 16),
            num_samples_per_op=5000,  # Default from explore_gemm_space.py
            gemm_map_resolution=100,  # Default
            checkpoint_path='v5_11_homeostasis/epoch_20.pt'
        )

    operations = {
        op: OperationMetrics.from_dict(op, metrics)
        for op, metrics in data['operations'].items()
    }

    lattice = LatticeStructure.from_dict(data['lattice'])
    ultrametric = UltrametricStructure.from_dict(data['ultrametric'])

    gemm_maps = {
        op: GEMMMapMetrics.from_dict(op, metrics, config.gemm_map_resolution)
        for op, metrics in data.get('gemm_maps', {}).items()
    }

    return ExplorationResults(
        timestamp=data['timestamp'],
        config=config,
        operations=operations,
        lattice=lattice,
        ultrametric=ultrametric,
        gemm_maps=gemm_maps
    )


def save_results(results: ExplorationResults, path: Path | str):
    """Save exploration results to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results.to_dict(), f, indent=2)


def load_gemm_map(base_dir: Path | str, operation: str) -> Dict[str, np.ndarray]:
    """Load GEMM map arrays for an operation."""
    base_dir = Path(base_dir)
    return {
        'soft_map': np.load(base_dir / f'soft_gemm_map_{operation}.npy'),
        'nearest_map': np.load(base_dir / f'nearest_map_{operation}.npy'),
        'distance_map': np.load(base_dir / f'distance_map_{operation}.npy')
    }


def compare_runs(results_list: List[ExplorationResults]) -> Dict[str, Any]:
    """Compare multiple exploration runs."""
    comparison = {
        'num_runs': len(results_list),
        'timestamps': [r.timestamp for r in results_list],
        'ultrametric_correlations': [r.ultrametric.correlation for r in results_list],
        'total_coverages': [r.total_coverage for r in results_list],
        'operations': {}
    }

    # Aggregate per-operation metrics
    all_ops = set()
    for r in results_list:
        all_ops.update(r.operations.keys())

    for op in all_ops:
        comparison['operations'][op] = {
            'ultrametric_rates': [
                r.operations[op].ultrametric_rate
                for r in results_list if op in r.operations
            ],
            'gap_means': [
                r.operations[op].soft_gap_mean
                for r in results_list if op in r.operations
            ]
        }

    return comparison


# =============================================================================
# MAIN - Demo/Test
# =============================================================================

if __name__ == "__main__":
    # Demo: Load existing results and print summary
    base_dir = Path(__file__).parent
    report_path = base_dir / "exploration_report.json"

    if report_path.exists():
        results = load_results(report_path)
        print(results.summary())

        # Show programmatic access
        print("\nProgrammatic access examples:")
        print(f"  results.ultrametric.correlation = {results.ultrametric.correlation}")
        print(f"  results.lattice.distance_peaks = {results.lattice.distance_peaks[:3]}...")
        print(f"  results.operations['add'].ultrametric_rate = {results.operations['add'].ultrametric_rate}")
        print(f"  results.total_coverage = {results.total_coverage:,}")
    else:
        print(f"No results found at {report_path}")
        print("Run explore_gemm_space.py first to generate results.")
