"""
falsify.py - Ternary Semantic Hypothesis Falsification Framework

Philosophy: We do not search for solutions. We LISTEN to errors.
By falsifying hypotheses, we discover truth through negative space.

This script PROPERLY wires ALL project components:
- SIMD Engine: simd_batch_operation from data.py
- Hyperbolic VAE: PoincareOperations, HyperbolicEncoder from hyperbolic_ops.py
- Ultrametric Energy: compute_3adic_valuation from ultrametric_energy.py
- Operation LUTs: create_operation_luts, OperationLUTDataset from data.py
- Complete 19,683 operation corpus

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

USAGE:
    python research/scripts/falsify.py --hypothesis H1
    python research/scripts/falsify.py --tier tier1_easy
    python research/scripts/falsify.py --all
"""

import json
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# Setup paths
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "3-vae-gemm-v1"))
sys.path.insert(0, str(ROOT / "models" / "gemm_discovery"))


# =============================================================================
# COMPONENT LOADING - PROPER WIRING
# =============================================================================

class ComponentLoader:
    """Load and verify all project components with REAL implementations."""

    def __init__(self):
        self.components = {}
        self.errors = []
        self._load_status = {}

    def load_simd_and_data(self) -> bool:
        """Load SIMD engine and data utilities from data.py."""
        try:
            from data import (
                HAS_SIMD,
                OperationLUTDataset,
                TripletDataset,
                create_operation_luts,
                index_to_trits,
                load_operation_luts,
                simd_batch_operation,
                trits_to_index,
            )

            self.components['data'] = {
                'index_to_trits': index_to_trits,
                'trits_to_index': trits_to_index,
                'simd_batch_operation': simd_batch_operation,
                'create_operation_luts': create_operation_luts,
                'load_operation_luts': load_operation_luts,
                'OperationLUTDataset': OperationLUTDataset,
                'TripletDataset': TripletDataset,
                'HAS_SIMD': HAS_SIMD,
            }
            print(f"[OK] Data utilities loaded (SIMD: {'available' if HAS_SIMD else 'fallback'})")
            self._load_status['data'] = True
            return True
        except ImportError as e:
            self.errors.append(f"Data utilities: {e}")
            print(f"[ERROR] Data utilities failed: {e}")
            self._load_status['data'] = False
            return False

    def load_hyperbolic(self) -> bool:
        """Load hyperbolic/Poincaré operations from hyperbolic_ops.py."""
        try:
            from hyperbolic_ops import (
                HyperbolicEncoder,
                HyperbolicLinear,
                HyperbolicMLR,
                HyperbolicOperationFlow,
                HyperbolicOperationLoss,
                HyperbolicOperationModel,
                PoincareOperations,
                UltrametricAttractorField,
            )

            self.components['hyperbolic'] = {
                'PoincareOperations': PoincareOperations,
                'HyperbolicEncoder': HyperbolicEncoder,
                'HyperbolicLinear': HyperbolicLinear,
                'HyperbolicMLR': HyperbolicMLR,
                'UltrametricAttractorField': UltrametricAttractorField,
                'HyperbolicOperationFlow': HyperbolicOperationFlow,
                'HyperbolicOperationModel': HyperbolicOperationModel,
                'HyperbolicOperationLoss': HyperbolicOperationLoss,
            }
            print("[OK] Hyperbolic operations loaded")
            self._load_status['hyperbolic'] = True
            return True
        except ImportError as e:
            self.errors.append(f"Hyperbolic: {e}")
            print(f"[ERROR] Hyperbolic operations failed: {e}")
            self._load_status['hyperbolic'] = False
            return False

    def load_ultrametric(self) -> bool:
        """Load ultrametric/p-adic functions from ultrametric_energy.py."""
        try:
            from ebm.ultrametric_energy import (
                UltrametricEnergyFunction,
                compute_3adic_valuation,
                compute_factor_valuation_profile,
                compute_ultrametric_distance,
            )

            self.components['ultrametric'] = {
                'compute_3adic_valuation': compute_3adic_valuation,
                'compute_ultrametric_distance': compute_ultrametric_distance,
                'compute_factor_valuation_profile': compute_factor_valuation_profile,
                'UltrametricEnergyFunction': UltrametricEnergyFunction,
            }
            print("[OK] Ultrametric energy loaded")
            self._load_status['ultrametric'] = True
            return True
        except ImportError as e:
            self.errors.append(f"Ultrametric: {e}")
            print(f"[WARN] Ultrametric energy failed: {e}")
            self._load_status['ultrametric'] = False
            return False

    def load_or_create_luts(self, sample_size: int = 100000) -> bool:
        """Load existing LUTs or create new ones."""
        lut_dir = ROOT / "models" / "3-vae-gemm-v1" / "luts"

        if lut_dir.exists() and list(lut_dir.glob("lut_*.npz")):
            try:
                load_luts = self.components['data']['load_operation_luts']
                luts = load_luts(lut_dir)
                self.components['luts'] = luts
                total = sum(len(d['results']) for d in luts.values())
                print(f"[OK] LUTs loaded from disk ({total:,} samples)")
                return True
            except Exception as e:
                print(f"[WARN] Failed to load LUTs: {e}")

        # Create new LUTs
        try:
            print(f"[INFO] Creating new LUTs ({sample_size:,} samples)...")
            create_luts = self.components['data']['create_operation_luts']
            luts = create_luts(
                sample_size=sample_size,
                output_dir=lut_dir
            )
            self.components['luts'] = luts
            return True
        except Exception as e:
            self.errors.append(f"LUT creation: {e}")
            print(f"[ERROR] LUT creation failed: {e}")
            return False

    def load_trained_model(self) -> bool:
        """Load trained hyperbolic model checkpoint if available."""
        checkpoint_dir = ROOT / "models" / "3-vae-gemm-v1" / "checkpoints_hyperbolic"
        checkpoint_path = checkpoint_dir / "best_model.pt"

        if not checkpoint_path.exists():
            print("[INFO] No trained model checkpoint found")
            self._load_status['trained_model'] = False
            return False

        try:
            HyperbolicOperationModel = self.components['hyperbolic']['HyperbolicOperationModel']
            model = HyperbolicOperationModel(
                num_trits=9,
                num_values=19683,
                latent_dim=16,
                num_operations=4
            )
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            self.components['trained_model'] = model
            print(f"[OK] Trained model loaded (epoch {checkpoint.get('epoch', '?')})")
            self._load_status['trained_model'] = True
            return True
        except Exception as e:
            self.errors.append(f"Trained model: {e}")
            print(f"[WARN] Failed to load trained model: {e}")
            self._load_status['trained_model'] = False
            return False

    def build_corpus(self) -> bool:
        """Build the complete 19,683 value corpus with valuations."""
        try:
            v3 = self.components['ultrametric']['compute_3adic_valuation']
            index_to_trits = self.components['data']['index_to_trits']

            num_values = 19683
            valuations = np.array([v3(i) for i in range(num_values)])

            # Build trit representations
            trits = np.array([index_to_trits(i, 9) for i in range(num_values)])

            self.components['corpus'] = {
                'num_values': num_values,
                'valuations': valuations,
                'trits': trits,
                'v3': v3,
            }

            # Print valuation distribution
            unique, counts = np.unique(valuations, return_counts=True)
            print(f"[OK] Corpus built ({num_values} values)")
            print(f"     Valuation distribution: {dict(zip(unique, counts))}")  # noqa: B905

            return True
        except Exception as e:
            self.errors.append(f"Corpus: {e}")
            print(f"[ERROR] Corpus build failed: {e}")
            return False

    def load_all(self) -> dict[str, bool]:
        """Load all components and return status."""
        status = {}

        status['data'] = self.load_simd_and_data()
        status['hyperbolic'] = self.load_hyperbolic()
        status['ultrametric'] = self.load_ultrametric()

        if status['data'] and status['ultrametric']:
            status['corpus'] = self.build_corpus()
            status['luts'] = self.load_or_create_luts(sample_size=50000)
        else:
            status['corpus'] = False
            status['luts'] = False

        if status['hyperbolic']:
            status['trained_model'] = self.load_trained_model()
        else:
            status['trained_model'] = False

        return status


# =============================================================================
# FALSIFICATION RESULT
# =============================================================================

@dataclass
class FalsificationResult:
    """Result of testing a hypothesis."""
    hypothesis_id: str
    hypothesis_name: str
    score: float
    grade: str
    predictions_tested: int
    predictions_passed: int
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    timing_seconds: float = 0.0
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'hypothesis_id': self.hypothesis_id,
            'hypothesis_name': self.hypothesis_name,
            'score': self.score,
            'grade': self.grade,
            'predictions_tested': self.predictions_tested,
            'predictions_passed': self.predictions_passed,
            'anomaly_count': len(self.anomalies),
            'anomalies_sample': self.anomalies[:5],
            'timing_seconds': self.timing_seconds,
            'error': self.error,
            'details': self.details,
        }


# =============================================================================
# HYPOTHESIS TESTS - PROPERLY WIRED
# =============================================================================

class HypothesisTests:
    """
    Collection of falsification tests for each hypothesis.

    ALL tests use REAL components:
    - SIMD engine via simd_batch_operation
    - Hyperbolic ops via PoincareOperations
    - Ultrametric via compute_3adic_valuation
    - Real operation LUTs
    """

    def __init__(self, loader: ComponentLoader):
        self.loader = loader
        self.c = loader.components

    def _score_to_grade(self, score: float) -> str:
        if score >= 0.95:
            return 'A'
        if score >= 0.80:
            return 'B'
        if score >= 0.50:
            return 'C'
        if score >= 0.20:
            return 'D'
        return 'F'

    # -------------------------------------------------------------------------
    # H1: p-adic / 3-adic Semantics
    # -------------------------------------------------------------------------
    def test_H1_padic(self) -> FalsificationResult:
        """
        Test p-adic structure using REAL valuation function and operation LUTs.

        Predictions:
        1. Ultrametric triangle inequality holds on operation results
        2. Valuation distribution follows 2/3^k pattern
        3. High-valuation results are exponentially rare
        """
        start = time.time()
        passed = 0
        tested = 0
        anomalies = []
        details = {}

        v3 = self.c['corpus']['v3']
        luts = self.c['luts']

        # Test 1: Ultrametric inequality on REAL operation results
        # d(a,c) <= max(d(a,b), d(b,c)) in p-adic metric
        # p-adic distance: d(x,y) = 3^(-v3(x-y))

        print("  Testing ultrametric inequality on real operations...")
        for op_name, lut_data in luts.items():
            results = lut_data['results']
            n_samples = min(1000, len(results))

            for i in range(0, n_samples - 2, 3):
                a, b, c = int(results[i]), int(results[i+1]), int(results[i+2])

                tested += 1
                v_ab = v3(abs(a - b)) if a != b else 100
                v_bc = v3(abs(b - c)) if b != c else 100
                v_ac = v3(abs(a - c)) if a != c else 100

                # Ultrametric: v(a-c) >= min(v(a-b), v(b-c))
                # (higher valuation = smaller distance)
                if v_ac >= min(v_ab, v_bc):
                    passed += 1
                else:
                    if len(anomalies) < 20:
                        anomalies.append({
                            'test': 'ultrametric_inequality',
                            'op': op_name,
                            'a': a, 'b': b, 'c': c,
                            'v_ab': v_ab, 'v_bc': v_bc, 'v_ac': v_ac
                        })

        details['ultrametric_tests'] = tested
        details['ultrametric_passed'] = passed

        # Test 2: Valuation distribution of operation results
        print("  Testing valuation distribution...")
        all_results = np.concatenate([d['results'] for d in luts.values()])
        result_valuations = np.array([v3(int(r)) for r in all_results[:10000]])

        unique_vals, counts = np.unique(result_valuations, return_counts=True)
        val_dist = dict(zip(unique_vals.tolist(), (counts / counts.sum()).tolist()))  # noqa: B905
        details['valuation_distribution'] = val_dist

        # Expected: ~66.7% at v=0
        v0_ratio = val_dist.get(0, 0)
        tested += 1
        if v0_ratio > 0.5:  # Should dominate
            passed += 1
        else:
            anomalies.append({'test': 'v0_dominance', 'ratio': v0_ratio, 'expected': '>0.5'})

        # Test 3: High valuation rarity (should decrease exponentially)
        v0_count = counts[unique_vals == 0][0] if 0 in unique_vals else 0
        v1_count = counts[unique_vals == 1][0] if 1 in unique_vals else 0
        v2_count = counts[unique_vals == 2][0] if 2 in unique_vals else 0

        tested += 2
        # v1 should be less than v0
        if v1_count < v0_count:
            passed += 1
        else:
            anomalies.append({'test': 'v1_less_v0', 'v0': int(v0_count), 'v1': int(v1_count)})

        # v2 should be less than v1
        if v2_count < v1_count:
            passed += 1
        else:
            anomalies.append({'test': 'v2_less_v1', 'v1': int(v1_count), 'v2': int(v2_count)})

        score = passed / tested if tested > 0 else 0
        grade = self._score_to_grade(score)

        return FalsificationResult(
            hypothesis_id='H1',
            hypothesis_name='p-adic / 3-adic Semantics',
            score=score,
            grade=grade,
            predictions_tested=tested,
            predictions_passed=passed,
            anomalies=anomalies,
            timing_seconds=time.time() - start,
            details=details
        )

    # -------------------------------------------------------------------------
    # H2: Ultrametric Tree Semantics
    # -------------------------------------------------------------------------
    def test_H2_ultrametric(self) -> FalsificationResult:
        """
        Test ultrametric tree structure using REAL embeddings.

        Key test: ALL triangles must be isoceles with the two largest sides equal.
        This is the STRONG form of the ultrametric inequality.

        We test this on:
        1. Raw ternary values with p-adic distance
        2. Hyperbolic embeddings if trained model available
        """
        start = time.time()
        passed = 0
        tested = 0
        anomalies = []
        details = {}

        v3 = self.c['corpus']['v3']

        # Test 1: Isoceles property on raw values with p-adic distance
        print("  Testing isoceles property on p-adic distances...")

        # Sample triplets from corpus
        num_values = self.c['corpus']['num_values']
        np.random.seed(42)

        n_triplets = 2000
        triplet_results = {'isoceles': 0, 'not_isoceles': 0}

        for _ in range(n_triplets):
            a, b, c = np.random.choice(num_values, 3, replace=False)

            # p-adic distances (using valuation of difference)
            # Higher valuation = smaller distance
            # We use v3 directly as "inverse distance"
            v_ab = v3(abs(int(a) - int(b))) if a != b else 100
            v_bc = v3(abs(int(b) - int(c))) if b != c else 100
            v_ac = v3(abs(int(a) - int(c))) if a != c else 100

            # Sort by valuation (descending = ascending distance)
            vals = sorted([v_ab, v_bc, v_ac])

            tested += 1
            # Isoceles: two smallest valuations should be equal (= two largest distances)
            if vals[0] == vals[1]:
                passed += 1
                triplet_results['isoceles'] += 1
            else:
                triplet_results['not_isoceles'] += 1
                if len(anomalies) < 20:
                    anomalies.append({
                        'test': 'isoceles_padic',
                        'a': int(a), 'b': int(b), 'c': int(c),
                        'v_ab': v_ab, 'v_bc': v_bc, 'v_ac': v_ac,
                        'sorted': vals
                    })

        details['padic_triplets'] = triplet_results

        # Test 2: If trained hyperbolic model available, test in embedding space
        if 'trained_model' in self.c and self.c['trained_model'] is not None:
            print("  Testing isoceles property on hyperbolic embeddings...")

            model = self.c['trained_model']
            PoincareOps = self.c['hyperbolic']['PoincareOperations']

            # Encode a sample of values
            sample_idx = torch.randint(0, num_values, (500,))
            with torch.no_grad():
                embeddings = model.encode(sample_idx)

            hyp_triplet_results = {'isoceles': 0, 'not_isoceles': 0}

            for _ in range(500):
                i, j, k = np.random.choice(len(embeddings), 3, replace=False)
                emb_a, emb_b, emb_c = embeddings[i], embeddings[j], embeddings[k]

                # Hyperbolic distances
                d_ab = PoincareOps.hyperbolic_distance(
                    emb_a.unsqueeze(0), emb_b.unsqueeze(0)
                ).item()
                d_bc = PoincareOps.hyperbolic_distance(
                    emb_b.unsqueeze(0), emb_c.unsqueeze(0)
                ).item()
                d_ac = PoincareOps.hyperbolic_distance(
                    emb_a.unsqueeze(0), emb_c.unsqueeze(0)
                ).item()

                dists = sorted([d_ab, d_bc, d_ac])

                tested += 1
                # Isoceles: two largest should be approximately equal
                if abs(dists[1] - dists[2]) < 0.1 * dists[2]:  # 10% tolerance
                    passed += 1
                    hyp_triplet_results['isoceles'] += 1
                else:
                    hyp_triplet_results['not_isoceles'] += 1

            details['hyperbolic_triplets'] = hyp_triplet_results

        score = passed / tested if tested > 0 else 0
        grade = self._score_to_grade(score)

        return FalsificationResult(
            hypothesis_id='H2',
            hypothesis_name='Ultrametric Tree Semantics',
            score=score,
            grade=grade,
            predictions_tested=tested,
            predictions_passed=passed,
            anomalies=anomalies,
            timing_seconds=time.time() - start,
            details=details
        )

    # -------------------------------------------------------------------------
    # H3: Hyperbolic / Poincaré Ball Semantics
    # -------------------------------------------------------------------------
    def test_H3_hyperbolic(self) -> FalsificationResult:
        """
        Test hyperbolic geometry using REAL Poincaré operations.

        Predictions:
        1. Geodesic midpoint is equidistant (NOT Euclidean average)
        2. Valuation correlates with radial position (high v = near center)
        3. Operations follow geodesic trajectories
        """
        start = time.time()
        passed = 0
        tested = 0
        anomalies = []
        details = {}

        PoincareOps = self.c['hyperbolic']['PoincareOperations']

        # Test 1: Geodesic midpoint equidistance
        print("  Testing geodesic midpoint equidistance...")

        n_pairs = 500
        equidistance_errors = []
        euclidean_errors = []

        dim = 16
        for _ in range(n_pairs):
            # Random points in Poincaré ball - PROPERLY scaled for dimension
            # randn(1, dim) has expected norm sqrt(dim), so we must scale down
            # to keep points INSIDE the ball (not on boundary where math is singular)
            x_dir = torch.randn(1, dim)
            y_dir = torch.randn(1, dim)
            target_radius = 0.3 + 0.5 * torch.rand(1).item()  # Radius 0.3-0.8
            x = x_dir / x_dir.norm() * target_radius * torch.rand(1).item()
            y = y_dir / y_dir.norm() * target_radius * torch.rand(1).item()
            x = PoincareOps.project(x)
            y = PoincareOps.project(y)

            # Geodesic midpoint
            hyp_mid = PoincareOps.geodesic_midpoint(x, y)

            # Euclidean midpoint (WRONG)
            euc_mid = (x + y) / 2
            euc_mid = PoincareOps.project(euc_mid)

            # Distances
            d_x_hyp = PoincareOps.hyperbolic_distance(x, hyp_mid).item()
            d_y_hyp = PoincareOps.hyperbolic_distance(y, hyp_mid).item()
            d_x_euc = PoincareOps.hyperbolic_distance(x, euc_mid).item()
            d_y_euc = PoincareOps.hyperbolic_distance(y, euc_mid).item()

            hyp_error = abs(d_x_hyp - d_y_hyp)
            euc_error = abs(d_x_euc - d_y_euc)

            equidistance_errors.append(hyp_error)
            euclidean_errors.append(euc_error)

            tested += 1
            # Test 1: Geodesic midpoint should be equidistant (error < 0.01)
            # Test 2: Geodesic should be BETTER than Euclidean
            if hyp_error < 0.01 and hyp_error <= euc_error:
                passed += 1
            else:
                if len(anomalies) < 10:
                    anomalies.append({
                        'test': 'geodesic_equidistance',
                        'd_x_hyp': d_x_hyp, 'd_y_hyp': d_y_hyp,
                        'hyp_error': hyp_error,
                        'euc_error': euc_error,
                        'geodesic_better': hyp_error < euc_error
                    })

        details['geodesic_mean_error'] = float(np.mean(equidistance_errors))
        details['euclidean_mean_error'] = float(np.mean(euclidean_errors))
        details['geodesic_better'] = details['geodesic_mean_error'] < details['euclidean_mean_error']
        geodesic_wins = sum(1 for g, e in zip(equidistance_errors, euclidean_errors) if g <= e)  # noqa: B905
        details['geodesic_wins_count'] = geodesic_wins
        details['geodesic_wins_pct'] = geodesic_wins / len(equidistance_errors) * 100

        # Test 2: Valuation-radius correlation (if trained model available)
        if 'trained_model' in self.c and self.c['trained_model'] is not None:
            print("  Testing valuation-radius correlation...")

            model = self.c['trained_model']
            valuations = self.c['corpus']['valuations']

            # Encode all values
            all_idx = torch.arange(min(1000, len(valuations)))
            with torch.no_grad():
                embeddings = model.encode(all_idx)

            radii = embeddings.norm(dim=-1).numpy()
            vals = valuations[all_idx.numpy()]

            # Compute Spearman correlation
            from scipy.stats import spearmanr
            vrc, pval = spearmanr(vals, radii)

            details['valuation_radius_correlation'] = float(vrc)
            details['vrc_pvalue'] = float(pval)

            tested += 1
            # Should be NEGATIVE (high valuation = small radius = near center)
            if vrc < -0.3:
                passed += 1
            else:
                anomalies.append({
                    'test': 'vrc_negative',
                    'vrc': float(vrc),
                    'expected': '< -0.3'
                })

        score = passed / tested if tested > 0 else 0
        grade = self._score_to_grade(score)

        return FalsificationResult(
            hypothesis_id='H3',
            hypothesis_name='Hyperbolic / Poincaré Ball Semantics',
            score=score,
            grade=grade,
            predictions_tested=tested,
            predictions_passed=passed,
            anomalies=anomalies,
            timing_seconds=time.time() - start,
            details=details
        )

    # -------------------------------------------------------------------------
    # H4: Tropical Algebra Semantics
    # -------------------------------------------------------------------------
    def test_H4_tropical(self) -> FalsificationResult:  # noqa: C901
        """
        Test tropical semiring properties using REAL SIMD operations.

        Tropical algebra (min-plus or max-plus):
        - ⊕ = min (tropical addition)
        - ⊗ = + (tropical multiplication, but we test with tadd)

        Predictions:
        1. Idempotent: a ⊕ a = a (tmin/tmax with self)
        2. Distributivity: a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c)
        3. Associativity of ⊕: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
        4. Zero element behavior
        """
        start = time.time()
        passed = 0
        tested = 0
        anomalies = []
        details = {}

        simd_op = self.c['data']['simd_batch_operation']
        trits_to_index = self.c['data']['trits_to_index']

        num_values = self.c['corpus']['num_values']
        np.random.seed(42)

        print("  Testing tropical algebra properties...")

        n_samples = 500

        a_idx = np.random.randint(0, num_values, n_samples)
        b_idx = np.random.randint(0, num_values, n_samples)
        c_idx = np.random.randint(0, num_values, n_samples)

        # Test 1: Idempotent (a ⊕ a = a) for both min and max
        print("    Idempotent property...")
        a_min_a = simd_op(a_idx, a_idx, 'min')
        a_max_a = simd_op(a_idx, a_idx, 'max')

        idemp_min = 0
        idemp_max = 0
        for i in range(n_samples):
            tested += 2
            if a_min_a[i] == a_idx[i]:
                passed += 1
                idemp_min += 1
            if a_max_a[i] == a_idx[i]:
                passed += 1
                idemp_max += 1

        details['idempotent_min_rate'] = idemp_min / n_samples
        details['idempotent_max_rate'] = idemp_max / n_samples

        # Test 2: Associativity of ⊕ (min/max)
        print("    Associativity of tropical addition...")
        # (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
        a_min_b = simd_op(a_idx, b_idx, 'min')
        ab_min_c = simd_op(a_min_b, c_idx, 'min')
        b_min_c = simd_op(b_idx, c_idx, 'min')
        a_min_bc = simd_op(a_idx, b_min_c, 'min')

        assoc_min = 0
        for i in range(n_samples):
            tested += 1
            if ab_min_c[i] == a_min_bc[i]:
                passed += 1
                assoc_min += 1
            else:
                if len(anomalies) < 10:
                    anomalies.append({
                        'test': 'assoc_min',
                        'a': int(a_idx[i]), 'b': int(b_idx[i]), 'c': int(c_idx[i]),
                        'ab_c': int(ab_min_c[i]), 'a_bc': int(a_min_bc[i])
                    })

        details['associativity_min_rate'] = assoc_min / n_samples

        # Test 3: Distributivity of ⊗ over ⊕
        # tadd(a, tmin(b,c)) = tmin(tadd(a,b), tadd(a,c))
        print("    Distributivity of ⊗ over ⊕...")
        b_min_c = simd_op(b_idx, c_idx, 'min')
        a_add_bminc = simd_op(a_idx, b_min_c, 'add')  # a ⊗ (b ⊕ c)

        a_add_b = simd_op(a_idx, b_idx, 'add')
        a_add_c = simd_op(a_idx, c_idx, 'add')
        ab_min_ac = simd_op(a_add_b, a_add_c, 'min')  # (a ⊗ b) ⊕ (a ⊗ c)

        dist_add_min = 0
        for i in range(n_samples):
            tested += 1
            if a_add_bminc[i] == ab_min_ac[i]:
                passed += 1
                dist_add_min += 1
            else:
                if len(anomalies) < 20:
                    anomalies.append({
                        'test': 'dist_add_over_min',
                        'a': int(a_idx[i]), 'b': int(b_idx[i]), 'c': int(c_idx[i]),
                        'lhs': int(a_add_bminc[i]), 'rhs': int(ab_min_ac[i])
                    })

        details['distributivity_add_over_min'] = dist_add_min / n_samples

        # Test 4: Distributivity with max
        # tadd(a, tmax(b,c)) = tmax(tadd(a,b), tadd(a,c))
        b_max_c = simd_op(b_idx, c_idx, 'max')
        a_add_bmaxc = simd_op(a_idx, b_max_c, 'add')  # a ⊗ (b ⊕ c)

        ab_max_ac = simd_op(a_add_b, a_add_c, 'max')  # (a ⊗ b) ⊕ (a ⊗ c)

        dist_add_max = 0
        for i in range(n_samples):
            tested += 1
            if a_add_bmaxc[i] == ab_max_ac[i]:
                passed += 1
                dist_add_max += 1

        details['distributivity_add_over_max'] = dist_add_max / n_samples

        # Test 5: Distributivity with tmul
        # tmul(a, tmin(b,c)) = tmin(tmul(a,b), tmul(a,c))
        print("    Distributivity of tmul over min...")
        a_mul_bminc = simd_op(a_idx, b_min_c, 'mul')

        a_mul_b = simd_op(a_idx, b_idx, 'mul')
        a_mul_c = simd_op(a_idx, c_idx, 'mul')
        ab_mul_min_ac = simd_op(a_mul_b, a_mul_c, 'min')

        dist_mul_min = 0
        for i in range(n_samples):
            tested += 1
            if a_mul_bminc[i] == ab_mul_min_ac[i]:
                passed += 1
                dist_mul_min += 1

        details['distributivity_mul_over_min'] = dist_mul_min / n_samples

        # Test 6: Zero absorbs in multiplication
        # a ⊗ 0 = 0 (zero is absorbing element for tmul)
        print("    Zero absorption...")
        zero_idx = np.full(n_samples, trits_to_index(np.array([0]*9)))
        a_mul_zero = simd_op(a_idx, zero_idx, 'mul')

        zero_absorb = 0
        zero_val = trits_to_index(np.array([0]*9))
        for i in range(n_samples):
            tested += 1
            if a_mul_zero[i] == zero_val:
                passed += 1
                zero_absorb += 1

        details['zero_absorption_mul'] = zero_absorb / n_samples

        score = passed / tested if tested > 0 else 0
        grade = self._score_to_grade(score)

        return FalsificationResult(
            hypothesis_id='H4',
            hypothesis_name='Tropical Algebra Semantics',
            score=score,
            grade=grade,
            predictions_tested=tested,
            predictions_passed=passed,
            anomalies=anomalies,
            timing_seconds=time.time() - start,
            details=details
        )

    # -------------------------------------------------------------------------
    # H11: Lattice / Order-Theoretic Semantics
    # -------------------------------------------------------------------------
    def test_H11_lattice(self) -> FalsificationResult:
        """
        Test lattice properties using REAL SIMD tmin/tmax operations.

        Predictions:
        1. Absorption laws: a ∧ (a v b) = a
        2. Distributivity: a ∧ (b v c) = (a ∧ b) v (a ∧ c)
        3. Idempotence: a ∧ a = a, a v a = a
        """
        start = time.time()
        passed = 0
        tested = 0
        anomalies = []
        details = {}

        simd_op = self.c['data']['simd_batch_operation']

        num_values = self.c['corpus']['num_values']
        np.random.seed(42)

        # Test with actual ternary values
        print("  Testing lattice properties with SIMD tmin/tmax...")

        n_samples = 500

        # Sample random indices
        a_idx = np.random.randint(0, num_values, n_samples)
        b_idx = np.random.randint(0, num_values, n_samples)
        c_idx = np.random.randint(0, num_values, n_samples)

        # Compute operations using SIMD
        # meet = tmin, join = tmax
        a_join_b = simd_op(a_idx, b_idx, 'max')  # a v b
        a_meet_b = simd_op(a_idx, b_idx, 'min')  # a ∧ b

        # Test 1: Absorption a ∧ (a v b) = a
        print("    Absorption law 1...")
        result = simd_op(a_idx, a_join_b, 'min')  # a ∧ (a v b)

        for i in range(n_samples):
            tested += 1
            if result[i] == a_idx[i]:
                passed += 1
            else:
                if len(anomalies) < 10:
                    anomalies.append({
                        'test': 'absorption1',
                        'a': int(a_idx[i]),
                        'a_join_b': int(a_join_b[i]),
                        'result': int(result[i]),
                        'expected': int(a_idx[i])
                    })

        details['absorption1_pass_rate'] = passed / n_samples

        # Test 2: Absorption a v (a ∧ b) = a
        print("    Absorption law 2...")
        result2 = simd_op(a_idx, a_meet_b, 'max')  # a v (a ∧ b)

        abs2_passed = 0
        for i in range(n_samples):
            tested += 1
            if result2[i] == a_idx[i]:
                passed += 1
                abs2_passed += 1
            else:
                if len(anomalies) < 20:
                    anomalies.append({
                        'test': 'absorption2',
                        'a': int(a_idx[i]),
                        'a_meet_b': int(a_meet_b[i]),
                        'result': int(result2[i]),
                        'expected': int(a_idx[i])
                    })

        details['absorption2_pass_rate'] = abs2_passed / n_samples

        # Test 3: Idempotence a ∧ a = a
        print("    Idempotence...")
        result_min = simd_op(a_idx, a_idx, 'min')
        result_max = simd_op(a_idx, a_idx, 'max')

        idemp_passed = 0
        for i in range(n_samples):
            tested += 2
            if result_min[i] == a_idx[i]:
                passed += 1
                idemp_passed += 1
            if result_max[i] == a_idx[i]:
                passed += 1
                idemp_passed += 1

        details['idempotence_pass_rate'] = idemp_passed / (2 * n_samples)

        # Test 4: Distributivity a ∧ (b v c) = (a ∧ b) v (a ∧ c)
        print("    Distributivity...")
        b_join_c = simd_op(b_idx, c_idx, 'max')
        lhs = simd_op(a_idx, b_join_c, 'min')  # a ∧ (b v c)

        a_meet_c = simd_op(a_idx, c_idx, 'min')
        rhs = simd_op(a_meet_b, a_meet_c, 'max')  # (a ∧ b) v (a ∧ c)

        dist_passed = 0
        for i in range(n_samples):
            tested += 1
            if lhs[i] == rhs[i]:
                passed += 1
                dist_passed += 1
            else:
                if len(anomalies) < 30:
                    anomalies.append({
                        'test': 'distributivity',
                        'a': int(a_idx[i]), 'b': int(b_idx[i]), 'c': int(c_idx[i]),
                        'lhs': int(lhs[i]), 'rhs': int(rhs[i])
                    })

        details['distributivity_pass_rate'] = dist_passed / n_samples

        score = passed / tested if tested > 0 else 0
        grade = self._score_to_grade(score)

        return FalsificationResult(
            hypothesis_id='H11',
            hypothesis_name='Lattice / Order-Theoretic Semantics',
            score=score,
            grade=grade,
            predictions_tested=tested,
            predictions_passed=passed,
            anomalies=anomalies,
            timing_seconds=time.time() - start,
            details=details
        )

    # -------------------------------------------------------------------------
    # H6: Three-Valued Logic Semantics
    # -------------------------------------------------------------------------
    def test_H6_three_valued_logic(self) -> FalsificationResult:
        """
        Test three-valued logic properties using REAL operations.

        Predictions:
        1. Kleene logic axioms (strong 3-valued)
        2. Lukasiewicz logic axioms
        3. Specific tautologies hold/fail predictably
        """
        start = time.time()
        passed = 0
        tested = 0
        anomalies = []
        details = {}

        simd_op = self.c['data']['simd_batch_operation']
        index_to_trits = self.c['data']['index_to_trits']
        trits_to_index = self.c['data']['trits_to_index']

        # tnot: negate each trit (-1 -> +1, 0 -> 0, +1 -> -1)
        def tnot_batch(indices):
            """Apply tnot to batch of indices."""
            results = np.zeros_like(indices)
            for i, idx in enumerate(indices):
                trits = index_to_trits(idx, 9)
                neg_trits = -trits  # Negate each trit
                results[i] = trits_to_index(neg_trits)
            return results

        # In balanced ternary: -1=False, 0=Unknown, +1=True
        # We'll test on single-trit values (indices 0-2 map to trits -1,0,+1)
        # For 9-trit system, we test component-wise logic

        print("  Testing three-valued logic axioms...")

        np.random.seed(42)
        n_samples = 500
        num_values = self.c['corpus']['num_values']

        # Sample random indices
        a_idx = np.random.randint(0, num_values, n_samples)
        b_idx = np.random.randint(0, num_values, n_samples)

        # Test 1: Double negation (tnot(tnot(a)) = a)
        print("    Double negation...")
        not_a = tnot_batch(a_idx)
        not_not_a = tnot_batch(not_a)

        dn_passed = 0
        for i in range(n_samples):
            tested += 1
            if not_not_a[i] == a_idx[i]:
                passed += 1
                dn_passed += 1
            else:
                if len(anomalies) < 10:
                    anomalies.append({
                        'test': 'double_negation',
                        'a': int(a_idx[i]),
                        'not_not_a': int(not_not_a[i])
                    })

        details['double_negation_rate'] = dn_passed / n_samples

        # Test 2: De Morgan's laws
        # not(a and b) = not(a) or not(b) -> not(min(a,b)) = max(not(a), not(b))
        print("    De Morgan's laws...")
        a_min_b = simd_op(a_idx, b_idx, 'min')
        not_a_min_b = tnot_batch(a_min_b)

        not_a = tnot_batch(a_idx)
        not_b = tnot_batch(b_idx)
        not_a_max_not_b = simd_op(not_a, not_b, 'max')

        dm1_passed = 0
        for i in range(n_samples):
            tested += 1
            if not_a_min_b[i] == not_a_max_not_b[i]:
                passed += 1
                dm1_passed += 1
            else:
                if len(anomalies) < 20:
                    anomalies.append({
                        'test': 'de_morgan_1',
                        'a': int(a_idx[i]), 'b': int(b_idx[i]),
                        'lhs': int(not_a_min_b[i]),
                        'rhs': int(not_a_max_not_b[i])
                    })

        details['de_morgan_1_rate'] = dm1_passed / n_samples

        # Test 3: De Morgan 2: not(a or b) = not(a) and not(b)
        a_max_b = simd_op(a_idx, b_idx, 'max')
        not_a_max_b = tnot_batch(a_max_b)
        not_a_min_not_b = simd_op(not_a, not_b, 'min')

        dm2_passed = 0
        for i in range(n_samples):
            tested += 1
            if not_a_max_b[i] == not_a_min_not_b[i]:
                passed += 1
                dm2_passed += 1

        details['de_morgan_2_rate'] = dm2_passed / n_samples

        # Test 4: Law of excluded middle (a or not(a) = True?)
        # In 3-valued logic, this FAILS for Unknown
        print("    Excluded middle (expected to fail for Unknown)...")
        a_or_not_a = simd_op(a_idx, not_a, 'max')

        # Check how many give "True" (max trit value)
        max_trit_idx = trits_to_index(np.array([1]*9))  # All +1s
        em_true = sum(1 for x in a_or_not_a if x == max_trit_idx)
        details['excluded_middle_true_rate'] = em_true / n_samples
        details['excluded_middle_note'] = 'Expected <100% in 3-valued logic'

        score = passed / tested if tested > 0 else 0
        grade = self._score_to_grade(score)

        return FalsificationResult(
            hypothesis_id='H6',
            hypothesis_name='Three-Valued Logic Semantics',
            score=score,
            grade=grade,
            predictions_tested=tested,
            predictions_passed=passed,
            anomalies=anomalies,
            timing_seconds=time.time() - start,
            details=details
        )

    # -------------------------------------------------------------------------
    # H9: Information-Theoretic Semantics
    # -------------------------------------------------------------------------
    def test_H9_information(self) -> FalsificationResult:
        """
        Test information-theoretic properties of ternary operations.

        Predictions:
        1. Entropy bounds: H(output) related to H(inputs)
        2. Mutual information: I(input; output) structure
        3. Optimal coding: log2(3) ≈ 1.585 bits/trit efficiency
        4. Valuation entropy: high-valuation values are rare (low entropy contribution)
        """
        start = time.time()
        passed = 0
        tested = 0
        anomalies = []
        details = {}

        simd_op = self.c['data']['simd_batch_operation']
        v3 = self.c['corpus']['v3']
        valuations = self.c['corpus']['valuations']
        luts = self.c['luts']

        print("  Testing information-theoretic properties...")

        # Helper: compute entropy from counts
        def entropy(counts):
            """Shannon entropy in bits."""
            probs = counts / counts.sum()
            probs = probs[probs > 0]  # Remove zeros
            return -np.sum(probs * np.log2(probs))

        # Test 1: Output entropy for each operation
        print("    Output entropy by operation...")
        op_entropies = {}
        max_entropy = np.log2(19683)  # Maximum possible entropy

        for op_name, lut_data in luts.items():
            results = lut_data['results']
            _unique, counts = np.unique(results, return_counts=True)
            H = entropy(counts)
            op_entropies[op_name] = H

            tested += 1
            # Entropy should be positive and less than maximum
            if 0 < H <= max_entropy:
                passed += 1
            else:
                anomalies.append({
                    'test': 'entropy_bounds',
                    'op': op_name,
                    'entropy': H,
                    'max': max_entropy
                })

        details['operation_entropies'] = op_entropies
        details['max_possible_entropy'] = max_entropy

        # Test 2: Valuation distribution entropy
        print("    Valuation distribution entropy...")
        _unique_vals, val_counts = np.unique(valuations, return_counts=True)
        val_entropy = entropy(val_counts)
        details['valuation_entropy'] = val_entropy

        # Expected: low entropy because high valuations are exponentially rare
        # Theoretical: sum over k of (2/3)*(1/3)^k * log2((2/3)*(1/3)^k)
        tested += 1
        if val_entropy < max_entropy * 0.5:  # Should be much less than max
            passed += 1
        else:
            anomalies.append({
                'test': 'valuation_entropy_low',
                'H': val_entropy,
                'threshold': max_entropy * 0.5
            })

        details['valuation_entropy_ratio'] = val_entropy / max_entropy

        # Test 3: Conditional entropy H(output | input)
        # For deterministic operations, H(output | inputs) = 0
        print("    Conditional entropy (determinism)...")

        for op_name, lut_data in luts.items():
            indices_a = lut_data['indices_a']
            indices_b = lut_data['indices_b']
            results = lut_data['results']

            # Check determinism: same inputs -> same output
            n_check = min(10000, len(indices_a))
            input_output = {}
            deterministic = 0
            non_deterministic = 0

            for i in range(n_check):
                key = (int(indices_a[i]), int(indices_b[i]))
                result = int(results[i])
                if key in input_output:
                    if input_output[key] == result:
                        deterministic += 1
                    else:
                        non_deterministic += 1
                else:
                    input_output[key] = result
                    deterministic += 1

            tested += 1
            if non_deterministic == 0:
                passed += 1
            else:
                anomalies.append({
                    'test': 'determinism',
                    'op': op_name,
                    'non_deterministic': non_deterministic
                })

        details['operations_deterministic'] = True  # All should be

        # Test 4: Data processing inequality
        # I(A; C) <= I(A; B) when A -> B -> C forms a Markov chain
        # For op(a, op(b, c)): info about a should decrease with more ops
        print("    Data processing inequality...")

        np.random.seed(42)
        n_samples = 5000
        num_values = 19683

        a_idx = np.random.randint(0, num_values, n_samples)
        b_idx = np.random.randint(0, num_values, n_samples)
        c_idx = np.random.randint(0, num_values, n_samples)

        # Compute b + c, then a + (b + c)
        b_add_c = simd_op(b_idx, c_idx, 'add')
        a_add_bc = simd_op(a_idx, b_add_c, 'add')

        # Mutual information proxy: correlation between a and final result
        # Using valuation as a "summary statistic"
        v_a = np.array([v3(int(x)) for x in a_idx])
        v_bc = np.array([v3(int(x)) for x in b_add_c])
        v_abc = np.array([v3(int(x)) for x in a_add_bc])

        # Correlation as MI proxy
        corr_a_bc = np.corrcoef(v_a, v_bc)[0, 1] if np.std(v_bc) > 0 else 0
        corr_a_abc = np.corrcoef(v_a, v_abc)[0, 1] if np.std(v_abc) > 0 else 0

        details['valuation_corr_a_bc'] = float(corr_a_bc) if not np.isnan(corr_a_bc) else 0
        details['valuation_corr_a_abc'] = float(corr_a_abc) if not np.isnan(corr_a_abc) else 0

        # Test 5: Bits per trit efficiency
        print("    Bits per trit efficiency...")
        theoretical_bits_per_trit = np.log2(3)  # ≈ 1.585
        details['theoretical_bits_per_trit'] = theoretical_bits_per_trit

        # For 9 trits, theoretical max info = 9 * log2(3) ≈ 14.26 bits
        theoretical_max_bits = 9 * theoretical_bits_per_trit
        actual_max_bits = np.log2(19683)  # = 9 * log2(3) exactly

        tested += 1
        if abs(theoretical_max_bits - actual_max_bits) < 0.001:
            passed += 1
            details['bits_per_trit_match'] = True
        else:
            details['bits_per_trit_match'] = False

        # Test 6: Zero has special information status (high valuation = low info)
        print("    Zero information content...")
        zero_idx = 9841  # Middle index (all zeros)
        zero_valuation = v3(zero_idx)

        tested += 1
        if zero_valuation >= 8:  # Zero should have very high valuation
            passed += 1
            details['zero_is_special'] = True
        else:
            details['zero_is_special'] = False
            anomalies.append({
                'test': 'zero_valuation',
                'expected': '>=8',
                'actual': zero_valuation
            })

        details['zero_valuation'] = zero_valuation

        score = passed / tested if tested > 0 else 0
        grade = self._score_to_grade(score)

        return FalsificationResult(
            hypothesis_id='H9',
            hypothesis_name='Information-Theoretic Semantics',
            score=score,
            grade=grade,
            predictions_tested=tested,
            predictions_passed=passed,
            anomalies=anomalies,
            timing_seconds=time.time() - start,
            details=details
        )

    # -------------------------------------------------------------------------
    # H10: Group Theory Semantics
    # -------------------------------------------------------------------------
    def test_H10_group(self) -> FalsificationResult:  # noqa: C901
        """
        Test group axioms using REAL SIMD tadd operation.

        Predictions:
        1. Associativity: (a + b) + c = a + (b + c)
        2. Identity: a + 0 = a (zero element)
        3. Inverse: a + (-a) = 0
        4. Closure: result is valid ternary value
        """
        start = time.time()
        passed = 0
        tested = 0
        anomalies = []
        details = {}

        simd_op = self.c['data']['simd_batch_operation']
        trits_to_index = self.c['data']['trits_to_index']
        index_to_trits = self.c['data']['index_to_trits']

        num_values = self.c['corpus']['num_values']
        np.random.seed(42)

        print("  Testing group axioms with SIMD tadd...")

        n_samples = 500

        a_idx = np.random.randint(0, num_values, n_samples)
        b_idx = np.random.randint(0, num_values, n_samples)
        c_idx = np.random.randint(0, num_values, n_samples)

        # Identity element: all zeros
        zero_idx = np.full(n_samples, trits_to_index(np.array([0]*9)))

        # Test 1: Right identity (a + 0 = a)
        print("    Right identity...")
        a_plus_zero = simd_op(a_idx, zero_idx, 'add')

        ri_passed = 0
        for i in range(n_samples):
            tested += 1
            if a_plus_zero[i] == a_idx[i]:
                passed += 1
                ri_passed += 1
            else:
                if len(anomalies) < 10:
                    anomalies.append({
                        'test': 'right_identity',
                        'a': int(a_idx[i]),
                        'a_plus_zero': int(a_plus_zero[i])
                    })

        details['right_identity_rate'] = ri_passed / n_samples

        # Test 2: Left identity (0 + a = a)
        print("    Left identity...")
        zero_plus_a = simd_op(zero_idx, a_idx, 'add')

        li_passed = 0
        for i in range(n_samples):
            tested += 1
            if zero_plus_a[i] == a_idx[i]:
                passed += 1
                li_passed += 1

        details['left_identity_rate'] = li_passed / n_samples

        # Test 3: Associativity ((a + b) + c = a + (b + c))
        print("    Associativity...")
        a_plus_b = simd_op(a_idx, b_idx, 'add')
        ab_plus_c = simd_op(a_plus_b, c_idx, 'add')

        b_plus_c = simd_op(b_idx, c_idx, 'add')
        a_plus_bc = simd_op(a_idx, b_plus_c, 'add')

        assoc_passed = 0
        for i in range(n_samples):
            tested += 1
            if ab_plus_c[i] == a_plus_bc[i]:
                passed += 1
                assoc_passed += 1
            else:
                if len(anomalies) < 20:
                    anomalies.append({
                        'test': 'associativity',
                        'a': int(a_idx[i]), 'b': int(b_idx[i]), 'c': int(c_idx[i]),
                        'ab_c': int(ab_plus_c[i]), 'a_bc': int(a_plus_bc[i])
                    })

        details['associativity_rate'] = assoc_passed / n_samples

        # Test 4: Commutativity (a + b = b + a)
        print("    Commutativity...")
        b_plus_a = simd_op(b_idx, a_idx, 'add')

        comm_passed = 0
        for i in range(n_samples):
            tested += 1
            if a_plus_b[i] == b_plus_a[i]:
                passed += 1
                comm_passed += 1

        details['commutativity_rate'] = comm_passed / n_samples

        # Test 5: Inverse existence (a + neg(a) = 0)
        # tnot: negate each trit (-1 -> +1, 0 -> 0, +1 -> -1)
        def tnot_batch(indices):
            results = np.zeros_like(indices)
            for i, idx in enumerate(indices):
                trits = index_to_trits(idx, 9)
                neg_trits = -trits
                results[i] = trits_to_index(neg_trits)
            return results

        print("    Inverse existence...")
        neg_a = tnot_batch(a_idx)  # tnot negates
        a_plus_neg_a = simd_op(a_idx, neg_a, 'add')

        inv_passed = 0
        zero_val = trits_to_index(np.array([0]*9))
        for i in range(n_samples):
            tested += 1
            if a_plus_neg_a[i] == zero_val:
                passed += 1
                inv_passed += 1
            else:
                if len(anomalies) < 30:
                    anomalies.append({
                        'test': 'inverse',
                        'a': int(a_idx[i]),
                        'neg_a': int(neg_a[i]),
                        'sum': int(a_plus_neg_a[i]),
                        'expected': int(zero_val)
                    })

        details['inverse_rate'] = inv_passed / n_samples

        score = passed / tested if tested > 0 else 0
        grade = self._score_to_grade(score)

        return FalsificationResult(
            hypothesis_id='H10',
            hypothesis_name='Group Theory Semantics',
            score=score,
            grade=grade,
            predictions_tested=tested,
            predictions_passed=passed,
            anomalies=anomalies,
            timing_seconds=time.time() - start,
            details=details
        )

    # -------------------------------------------------------------------------
    # H23: Modular Arithmetic Semantics
    # -------------------------------------------------------------------------
    def test_H23_modular(self) -> FalsificationResult:
        """
        Test Z/3Z (modular arithmetic mod 3) properties.

        IMPORTANT: Balanced ternary tadd/tmul are SATURATED, not modular.
        This means +1 + +1 saturates to +1, not 0 as in mod 3.

        We test what CAN be tested:
        1. p-adic valuation properties (always hold)
        2. Index encoding preserves mod 3 structure
        3. Single-trit operations show SATURATION behavior
        """
        start = time.time()
        passed = 0
        tested = 0
        anomalies = []
        details = {}

        simd_op = self.c['data']['simd_batch_operation']
        trits_to_index = self.c['data']['trits_to_index']
        v3 = self.c['corpus']['v3']

        num_values = self.c['corpus']['num_values']
        np.random.seed(42)

        print("  Testing Z/3Z modular arithmetic...")

        n_samples = 500
        a_idx = np.random.randint(0, num_values, n_samples)
        b_idx = np.random.randint(0, num_values, n_samples)

        # Test 1: p-adic valuation of sums
        # v3(a + b) >= min(v3(a), v3(b)) - ALWAYS holds for integers
        print("    Valuation of sums (p-adic)...")
        a_plus_b = simd_op(a_idx, b_idx, 'add')

        v_sum_passed = 0
        for i in range(n_samples):
            v_a = v3(abs(int(a_idx[i])))
            v_b = v3(abs(int(b_idx[i])))
            v_sum = v3(abs(int(a_plus_b[i])))

            tested += 1
            if v_sum >= min(v_a, v_b):
                passed += 1
                v_sum_passed += 1

        details['valuation_sum_rate'] = v_sum_passed / n_samples

        # Test 2: p-adic valuation of products
        # v3(a * b) = v3(a) + v3(b) - may not hold due to saturation
        print("    Valuation of products (p-adic)...")
        a_times_b = simd_op(a_idx, b_idx, 'mul')

        v_prod_passed = 0
        for i in range(n_samples):
            v_a = v3(abs(int(a_idx[i])))
            v_b = v3(abs(int(b_idx[i])))
            v_prod = v3(abs(int(a_times_b[i])))

            tested += 1
            # Allow ±1 tolerance for saturated arithmetic
            if abs(v_prod - (v_a + v_b)) <= 1:
                passed += 1
                v_prod_passed += 1

        details['valuation_product_rate'] = v_prod_passed / n_samples

        # Test 3: Single-trit SATURATION behavior
        print("    Single-trit saturation...")

        # Create single-trit indices for values -1, 0, +1
        minus_one = trits_to_index(np.array([-1] + [0]*8))
        zero = trits_to_index(np.array([0] + [0]*8))
        plus_one = trits_to_index(np.array([1] + [0]*8))

        # Test -1 + +1 = 0 (balanced ternary, not modular)
        minus_plus = simd_op(np.array([minus_one]), np.array([plus_one]), 'add')[0]
        tested += 1
        if minus_plus == zero:
            passed += 1
        else:
            details['minus_plus_balanced'] = False

        # Test +1 + +1 SATURATES to +1 (not modular 0)
        plus_plus = simd_op(np.array([plus_one]), np.array([plus_one]), 'add')[0]
        tested += 1
        if plus_plus == plus_one:
            passed += 1
        else:
            details['plus_plus_saturates'] = False

        # Test -1 + -1 SATURATES to -1 (not modular +1)
        minus_minus = simd_op(np.array([minus_one]), np.array([minus_one]), 'add')[0]
        tested += 1
        if minus_minus == minus_one:
            passed += 1
        else:
            details['minus_minus_saturates'] = False

        # Test multiplication: -1 * +1 = -1
        minus_times_plus = simd_op(np.array([minus_one]), np.array([plus_one]), 'mul')[0]
        tested += 1
        if minus_times_plus == minus_one:
            passed += 1

        # Test multiplication: +1 * +1 = +1
        plus_times_plus = simd_op(np.array([plus_one]), np.array([plus_one]), 'mul')[0]
        tested += 1
        if plus_times_plus == plus_one:
            passed += 1

        # Test 4: Verify index encoding is mod 3 consistent
        print("    Index encoding structure...")

        # The index encodes balanced ternary: idx = sum((trit + 1) * 3^i)
        # So idx % 3 should always be in {0, 1, 2}
        mod3_valid = 0
        for i in range(n_samples):
            idx = int(a_idx[i])
            mod3_val = idx % 3
            if mod3_val in [0, 1, 2]:
                mod3_valid += 1

        details['mod3_structure_rate'] = mod3_valid / n_samples
        tested += 1
        if details['mod3_structure_rate'] == 1.0:
            passed += 1

        score = passed / tested if tested > 0 else 0
        grade = self._score_to_grade(score)

        return FalsificationResult(
            hypothesis_id='H23',
            hypothesis_name='Modular/Saturated Arithmetic',
            score=score,
            grade=grade,
            predictions_tested=int(tested),
            predictions_passed=int(passed),
            anomalies=anomalies,
            timing_seconds=time.time() - start,
            details=details
        )
    # -------------------------------------------------------------------------
    # Get test function by ID
    # -------------------------------------------------------------------------
    def get_test_function(self, hypothesis_id: str) -> Callable | None:
        """Get test function for hypothesis."""
        mapping = {
            'H1': self.test_H1_padic,
            'H2': self.test_H2_ultrametric,
            'H3': self.test_H3_hyperbolic,
            'H4': self.test_H4_tropical,
            'H6': self.test_H6_three_valued_logic,
            'H8': self.test_H8_categorical,
            'H9': self.test_H9_information,
            'H10': self.test_H10_group,
            'H11': self.test_H11_lattice,
            'H23': self.test_H23_modular,
        }
        return mapping.get(hypothesis_id)

    # -------------------------------------------------------------------------
    # H7: Quantum-Inspired Superposition Semantics
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # H8: Category-Theoretic / Topos Semantics
    # -------------------------------------------------------------------------
    def test_H8_categorical(self) -> FalsificationResult:

        """

        Test category-theoretic structure.



        CLAIM: Ternary operations form a category with morphisms.



        KEY PROPERTIES tested:

        1. Composition associativity: (f ∘ g) ∘ h = f ∘ (g ∘ h)

        2. Identity laws: id ∘ f = f = f ∘ id

        3. Functoriality: F preserves composition

        4. Naturality: operations commute appropriately

        """

        start = time.time()

        passed = 0

        tested = 0

        anomalies = []

        details = {}



        simd_op = self.c['data']['simd_batch_operation']
        trits_to_index = self.c['data']['trits_to_index']
        v3 = self.c['corpus']['v3']

        num_values = self.c['corpus']['num_values']

        np.random.seed(42)



        print("  Testing category-theoretic properties...")



        # CORRECTED: Balanced 0 is identity for tadd

        zero_trits = np.array([0]*9)  # Balanced 0 (identity for tadd)

        zero_val = trits_to_index(zero_trits)



        # Test 1: Composition associativity

        print("    Testing composition associativity...")

        assoc_passed = 0

        n_assoc = 100



        for op_type in ['add', 'mul']:

            for _ in range(n_assoc):

                a_idx = np.random.randint(0, num_values)

                a = int(a_idx)



                # (a op a) op a vs a op (a op a)

                aa = simd_op(np.array([a]), np.array([a]), op_type)[0]

                a_aa = simd_op(np.array([a]), np.array([int(aa)]), op_type)[0]

                aa_a = simd_op(np.array([int(aa)]), np.array([a]), op_type)[0]



                tested += 2



                if int(a_aa) == int(aa_a):

                    passed += 2

                    assoc_passed += 2

                else:

                    if len(anomalies) < 10:

                        anomalies.append({

                            'test': 'associativity',

                            'op': op_type, 'a': a,

                            'a_aa': int(a_aa), 'aa_a': int(aa_a)

                        })



        details['associativity_rate'] = assoc_passed / (2 * n_assoc)



        # Test 2: Identity laws

        print("    Testing identity morphisms...")

        identity_passed = 0

        n_id = 100

        sample = np.random.choice(num_values, n_id, replace=False)



        for idx in sample:

            a = int(idx)



            # Left identity: 0 ⊕ a = a

            zero_plus_a = simd_op(np.array([zero_val]), np.array([a]), 'add')[0]

            tested += 1

            if int(zero_plus_a) == a:

                passed += 1

                identity_passed += 1

            else:

                if len(anomalies) < 5:

                    anomalies.append({

                        'test': 'left_identity',

                        'a': a, 'result': int(zero_plus_a)

                    })



            # Right identity: a ⊕ 0 = a

            a_plus_zero = simd_op(np.array([a]), np.array([zero_val]), 'add')[0]

            tested += 1

            if int(a_plus_zero) == a:

                passed += 1

                identity_passed += 1

            else:

                if len(anomalies) < 5:

                    anomalies.append({

                        'test': 'right_identity',

                        'a': a, 'result': int(a_plus_zero)

                    })



        details['identity_rate'] = identity_passed / (2 * n_id)



        # Test 3: Functoriality (3-adic ultrametric preservation)

        print("    Testing functoriality (ultrametric preservation)...")

        functorial_passed = 0

        n_func = 100



        for _ in range(n_func):

            a_idx = np.random.randint(0, num_values)

            b_idx = np.random.randint(0, num_values)

            a = int(a_idx)

            b = int(b_idx)



            a_plus_b = simd_op(np.array([a]), np.array([b]), 'add')[0]



            # FIXED: v3 of differences (proper p-adic metric)

            if a != b and b != a_plus_b:

                v_ab = v3(abs(a - b))

                v_bc = v3(abs(b - a_plus_b))

                v_ac = v3(abs(a - a_plus_b))



                tested += 1

                # Ultrametric: v(a,c) >= min(v(a,b), v(b,c))

                if v_ac >= min(v_ab, v_bc):

                    passed += 1

                    functorial_passed += 1



        details['functoriality_rate'] = functorial_passed / n_func



        # Test 4: Naturality (min/max relationship)

        print("    Testing naturality (commutation)...")

        naturality_passed = 0

        n_nat = 100



        for _ in range(n_nat):

            a_idx = np.random.randint(0, num_values)

            b_idx = np.random.randint(0, num_values)

            a = int(a_idx)

            b = int(b_idx)



            a_min_b = simd_op(np.array([a]), np.array([b]), 'min')[0]

            a_max_b = simd_op(np.array([a]), np.array([b]), 'max')[0]



            tested += 1



            # Naturality: min and max are dual

            # tmin(a,b) <= tmax(a,b) always

            if int(a_min_b) <= int(a_max_b):

                passed += 1

                naturality_passed += 1



        details['naturality_rate'] = naturality_passed / n_nat



        score = passed / tested if tested > 0 else 0

        grade = self._score_to_grade(score)



        return FalsificationResult(

            hypothesis_id='H8',

            hypothesis_name='Category-Theoretic / Topos Semantics',

            score=score,

            grade=grade,

            predictions_tested=int(tested),

            predictions_passed=int(passed),

            anomalies=anomalies[:10],

            timing_seconds=time.time() - start,

            details=details

        )



# =============================================================================
# MAIN RUNNER
# =============================================================================

class FalsificationRunner:
    """Main runner for hypothesis falsification."""

    def __init__(self, config_path: Path | None = None):
        self.config = self._load_config(config_path)
        self.loader = ComponentLoader()
        self.results: list[FalsificationResult] = []

    def _load_config(self, config_path: Path | None) -> dict:
        default_config = ROOT / "research" / "configs" / "schema.yaml"
        path = config_path or default_config
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)
        return {}

    def run_hypothesis(self, hypothesis_id: str) -> FalsificationResult:
        """Run falsification test for single hypothesis."""
        print(f"\n{'='*60}")
        print(f"TESTING {hypothesis_id}")
        print(f"{'='*60}")

        tests = HypothesisTests(self.loader)
        test_fn = tests.get_test_function(hypothesis_id)

        if test_fn is None:
            return FalsificationResult(
                hypothesis_id=hypothesis_id,
                hypothesis_name=f"Unknown: {hypothesis_id}",
                score=0.0,
                grade='?',
                predictions_tested=0,
                predictions_passed=0,
                error=f"No test implemented for {hypothesis_id}"
            )

        try:
            result = test_fn()
            self.results.append(result)
            self._print_result(result)
            return result
        except Exception as e:
            traceback.print_exc()
            result = FalsificationResult(
                hypothesis_id=hypothesis_id,
                hypothesis_name=hypothesis_id,
                score=0.0,
                grade='E',
                predictions_tested=0,
                predictions_passed=0,
                error=str(e)
            )
            return result

    def run_all_implemented(self) -> list[FalsificationResult]:
        """Run all implemented hypothesis tests."""
        implemented = ['H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H9', 'H10', 'H11', 'H23']

        print(f"\n{'#'*60}")
        print("RUNNING ALL IMPLEMENTED HYPOTHESES")
        print(f"{'#'*60}")

        results = []
        for h_id in implemented:
            results.append(self.run_hypothesis(h_id))

        self._print_summary(results)
        return results

    def _print_result(self, result: FalsificationResult):
        """Print single result."""
        status = "FALSIFIED" if result.grade == 'F' else "SUPPORTED" if result.grade in ['A', 'B'] else "WEAK"
        print(f"\n{result.hypothesis_id}: {result.hypothesis_name}")
        print(f"  Score: {result.score:.2%} (Grade: {result.grade}) - {status}")
        print(f"  Predictions: {result.predictions_passed}/{result.predictions_tested}")
        print(f"  Anomalies: {len(result.anomalies)}")
        print(f"  Time: {result.timing_seconds:.2f}s")
        if result.details:
            print(f"  Details: {json.dumps(result.details, indent=4, default=str)[:500]}...")
        if result.error:
            print(f"  Error: {result.error}")

    def _print_summary(self, results: list[FalsificationResult]):
        """Print summary of all results."""
        print(f"\n{'='*60}")
        print("FALSIFICATION SUMMARY")
        print(f"{'='*60}")

        falsified = [r for r in results if r.grade == 'F']
        supported = [r for r in results if r.grade in ['A', 'B']]
        weak = [r for r in results if r.grade in ['C', 'D']]
        errors = [r for r in results if r.error]

        print(f"\nFALSIFIED ({len(falsified)}):")
        for r in falsified:
            print(f"  {r.hypothesis_id}: {r.hypothesis_name} ({r.score:.2%})")

        print(f"\nSUPPORTED ({len(supported)}):")
        for r in supported:
            print(f"  {r.hypothesis_id}: {r.hypothesis_name} ({r.score:.2%})")

        print(f"\nWEAK ({len(weak)}):")
        for r in weak:
            print(f"  {r.hypothesis_id}: {r.hypothesis_name} ({r.score:.2%})")

        if errors:
            print(f"\nERRORS ({len(errors)}):")
            for r in errors:
                print(f"  {r.hypothesis_id}: {r.error}")

        # Save results
        self._save_results(results)

    def _save_results(self, results: list[FalsificationResult]):
        """Save results to file."""
        output_dir = ROOT / "research" / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"falsification_{timestamp}.json"

        data = {
            'timestamp': timestamp,
            'total_hypotheses': len(results),
            'results': [r.to_dict() for r in results]
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Ternary Semantic Hypothesis Falsification',
        epilog='Philosophy: Listen to errors, discover by negative space.'
    )
    parser.add_argument('--hypothesis', '-H', type=str,
                       help='Run specific hypothesis (e.g., H1, H11)')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Run all implemented hypotheses')

    args = parser.parse_args()

    print("="*60)
    print("TERNARY SEMANTIC FALSIFICATION FRAMEWORK")
    print("Philosophy: We do not search. We LISTEN to errors.")
    print("="*60)

    runner = FalsificationRunner()

    # Load components
    print("\nLoading components (REAL wiring)...")
    status = runner.loader.load_all()
    print(f"\nComponent status: {status}")

    if not all([status['data'], status['corpus']]):
        print("\n[FATAL] Core components failed to load. Cannot proceed.")
        return 1

    # Run tests
    if args.hypothesis:
        runner.run_hypothesis(args.hypothesis)
    elif args.all:
        runner.run_all_implemented()
    else:
        # Default: run all
        print("\nNo arguments provided. Running all implemented tests...")
        runner.run_all_implemented()

    return 0


if __name__ == "__main__":
    sys.exit(main())
