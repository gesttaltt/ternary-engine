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

import sys
import json
import yaml
import time
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime
from itertools import combinations
import traceback

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
                index_to_trits,
                trits_to_index,
                simd_batch_operation,
                create_operation_luts,
                load_operation_luts,
                OperationLUTDataset,
                TripletDataset,
                HAS_SIMD
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
                PoincareOperations,
                HyperbolicEncoder,
                HyperbolicLinear,
                HyperbolicMLR,
                UltrametricAttractorField,
                HyperbolicOperationFlow,
                HyperbolicOperationModel,
                HyperbolicOperationLoss,
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
                compute_3adic_valuation,
                compute_ultrametric_distance,
                compute_factor_valuation_profile,
                UltrametricEnergyFunction,
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
            print(f"     Valuation distribution: {dict(zip(unique, counts))}")

            return True
        except Exception as e:
            self.errors.append(f"Corpus: {e}")
            print(f"[ERROR] Corpus build failed: {e}")
            return False

    def load_all(self) -> Dict[str, bool]:
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
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    timing_seconds: float = 0.0
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
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
        if score >= 0.95: return 'A'
        if score >= 0.80: return 'B'
        if score >= 0.50: return 'C'
        if score >= 0.20: return 'D'
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
        valuations = self.c['corpus']['valuations']
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
        val_dist = dict(zip(unique_vals.tolist(), (counts / counts.sum()).tolist()))
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
        geodesic_wins = sum(1 for g, e in zip(equidistance_errors, euclidean_errors) if g <= e)
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
    # H11: Lattice / Order-Theoretic Semantics
    # -------------------------------------------------------------------------
    def test_H11_lattice(self) -> FalsificationResult:
        """
        Test lattice properties using REAL SIMD tmin/tmax operations.

        Predictions:
        1. Absorption laws: a ∧ (a ∨ b) = a
        2. Distributivity: a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c)
        3. Idempotence: a ∧ a = a, a ∨ a = a
        """
        start = time.time()
        passed = 0
        tested = 0
        anomalies = []
        details = {}

        simd_op = self.c['data']['simd_batch_operation']
        index_to_trits = self.c['data']['index_to_trits']
        trits_to_index = self.c['data']['trits_to_index']

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
        a_join_b = simd_op(a_idx, b_idx, 'max')  # a ∨ b
        a_meet_b = simd_op(a_idx, b_idx, 'min')  # a ∧ b

        # Test 1: Absorption a ∧ (a ∨ b) = a
        print("    Absorption law 1...")
        result = simd_op(a_idx, a_join_b, 'min')  # a ∧ (a ∨ b)

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

        # Test 2: Absorption a ∨ (a ∧ b) = a
        print("    Absorption law 2...")
        result2 = simd_op(a_idx, a_meet_b, 'max')  # a ∨ (a ∧ b)

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

        # Test 4: Distributivity a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c)
        print("    Distributivity...")
        b_join_c = simd_op(b_idx, c_idx, 'max')
        lhs = simd_op(a_idx, b_join_c, 'min')  # a ∧ (b ∨ c)

        a_meet_c = simd_op(a_idx, c_idx, 'min')
        rhs = simd_op(a_meet_b, a_meet_c, 'max')  # (a ∧ b) ∨ (a ∧ c)

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
    # Get test function by ID
    # -------------------------------------------------------------------------
    def get_test_function(self, hypothesis_id: str) -> Optional[Callable]:
        """Get test function for hypothesis."""
        mapping = {
            'H1': self.test_H1_padic,
            'H2': self.test_H2_ultrametric,
            'H3': self.test_H3_hyperbolic,
            'H11': self.test_H11_lattice,
        }
        return mapping.get(hypothesis_id)


# =============================================================================
# MAIN RUNNER
# =============================================================================

class FalsificationRunner:
    """Main runner for hypothesis falsification."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.loader = ComponentLoader()
        self.results: List[FalsificationResult] = []

    def _load_config(self, config_path: Optional[Path]) -> Dict:
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

    def run_all_implemented(self) -> List[FalsificationResult]:
        """Run all implemented hypothesis tests."""
        implemented = ['H1', 'H2', 'H3', 'H11']

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

    def _print_summary(self, results: List[FalsificationResult]):
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

    def _save_results(self, results: List[FalsificationResult]):
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
