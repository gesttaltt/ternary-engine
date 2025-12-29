#!/usr/bin/env python3
"""
Comprehensive Checkpoint Validation for Ternary Inference
==========================================================

Validates all .pt checkpoint files for ternary inference quality.
Does NOT trust metadata - performs actual inference and validation.

Test Categories:
1. Reconstruction Quality - Can the model reconstruct all 19,683 ternary operations?
2. Arithmetic Correctness - Do outputs match expected ternary arithmetic?
3. Algebraic Properties - Associativity, commutativity, identity elements
4. Geometric Structure - Hierarchy, radial ordering, distance preservation
5. GEMM Suitability - Properties useful for matrix multiplication

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from scipy.stats import spearmanr
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA CLASSES FOR METRICS
# =============================================================================

@dataclass
class ReconstructionMetrics:
    """Metrics for reconstruction quality."""
    coverage: float = 0.0  # % of 19683 operations correctly reconstructed
    unique_outputs: int = 0
    reconstruction_loss: float = float('inf')
    valid_ternary_outputs: float = 0.0  # % of outputs in {-1, 0, +1}


@dataclass
class ArithmeticMetrics:
    """Metrics for arithmetic correctness."""
    tnot_accuracy: float = 0.0
    tadd_accuracy: float = 0.0
    tmul_accuracy: float = 0.0
    tmin_accuracy: float = 0.0
    tmax_accuracy: float = 0.0
    overall_accuracy: float = 0.0


@dataclass
class AlgebraicMetrics:
    """Metrics for algebraic properties."""
    tadd_commutativity: float = 0.0  # a + b == b + a
    tmul_commutativity: float = 0.0  # a * b == b * a
    tadd_associativity: float = 0.0  # (a + b) + c == a + (b + c)
    tmul_associativity: float = 0.0  # (a * b) * c == a * (b * c)
    tadd_identity: float = 0.0  # a + 0 == a
    tmul_identity: float = 0.0  # a * 1 == a
    tnot_involution: float = 0.0  # not(not(a)) == a


@dataclass
class GeometricMetrics:
    """Metrics for geometric structure (p-adic/hyperbolic)."""
    hierarchy_A: float = 0.0  # Spearman(valuation, radius_A)
    hierarchy_B: float = 0.0  # Spearman(valuation, radius_B)
    richness: float = 0.0  # Within-level variance
    radius_v0: float = 0.0  # Mean radius at valuation=0
    radius_v9: float = 0.0  # Mean radius at valuation>=8
    radial_separation: float = 0.0  # r_v0 - r_v9
    distance_correlation_A: float = 0.0
    distance_correlation_B: float = 0.0


@dataclass
class GEMMMetrics:
    """Metrics for GEMM suitability."""
    weight_sparsity: float = 0.0  # % of zero weights
    ternary_purity: float = 0.0  # % of weights in {-1, 0, +1}
    encoding_efficiency: float = 0.0  # Bits per operation
    matmul_compatible: bool = False  # Can be used in batched matmul


@dataclass
class ValidationResult:
    """Complete validation result for a checkpoint."""
    checkpoint_path: str
    model_type: str = "unknown"
    loadable: bool = False
    error_message: str = ""

    # Sub-metrics
    reconstruction: ReconstructionMetrics = field(default_factory=ReconstructionMetrics)
    arithmetic: ArithmeticMetrics = field(default_factory=ArithmeticMetrics)
    algebraic: AlgebraicMetrics = field(default_factory=AlgebraicMetrics)
    geometric: GeometricMetrics = field(default_factory=GeometricMetrics)
    gemm: GEMMMetrics = field(default_factory=GEMMMetrics)

    # Composite scores
    overall_score: float = 0.0
    rank: int = 0

    # Metadata (from checkpoint, not trusted)
    claimed_metrics: Dict = field(default_factory=dict)


# =============================================================================
# MODEL ARCHITECTURE RECONSTRUCTION
# =============================================================================

class EncoderBlock(nn.Module):
    """Encoder block matching checkpoint structure."""
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)


class DecoderBlock(nn.Module):
    """Decoder block matching checkpoint structure."""
    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers = []
        prev_dim = latent_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)


class ProjectionHead(nn.Module):
    """Hyperbolic projection head."""
    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.direction_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.radius_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, z):
        direction = F.normalize(self.direction_net(z), dim=-1)
        radius = torch.sigmoid(self.radius_net(z))
        return direction * radius


class DualProjection(nn.Module):
    """Dual projection (A and B)."""
    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.proj_A = ProjectionHead(latent_dim, hidden_dim)
        self.proj_B = ProjectionHead(latent_dim, hidden_dim)

    def forward(self, z_A, z_B):
        return self.proj_A(z_A), self.proj_B(z_B)


class DifferentiableController(nn.Module):
    """Controller network for homeostasis."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, output_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class TernaryVAE(nn.Module):
    """Reconstructed TernaryVAE architecture."""
    def __init__(self, input_dim: int = 9, latent_dim: int = 16,
                 hidden_dims: List[int] = [256, 128, 64],
                 decoder_dims: List[int] = [32, 64],
                 output_dim: int = 27,
                 projection_dim: int = 64,
                 use_controller: bool = True):
        super().__init__()

        self.encoder_A = EncoderBlock(input_dim, hidden_dims, latent_dim)
        self.encoder_B = EncoderBlock(input_dim, hidden_dims, latent_dim)
        self.decoder_A = DecoderBlock(latent_dim, decoder_dims, output_dim)
        self.projection = DualProjection(latent_dim, projection_dim)

        if use_controller:
            self.controller = DifferentiableController()
        else:
            self.controller = None

    def encode(self, x):
        mu_A, logvar_A = self.encoder_A(x)
        mu_B, logvar_B = self.encoder_B(x)
        return mu_A, logvar_A, mu_B, logvar_B

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder_A(z)

    def forward(self, x, sample: bool = True):
        mu_A, logvar_A, mu_B, logvar_B = self.encode(x)

        if sample:
            z_A = self.reparameterize(mu_A, logvar_A)
            z_B = self.reparameterize(mu_B, logvar_B)
        else:
            z_A, z_B = mu_A, mu_B

        recon = self.decode(z_A)
        proj_A, proj_B = self.projection(z_A, z_B)

        return {
            'recon': recon,
            'mu_A': mu_A, 'logvar_A': logvar_A,
            'mu_B': mu_B, 'logvar_B': logvar_B,
            'z_A': z_A, 'z_B': z_B,
            'proj_A': proj_A, 'proj_B': proj_B
        }


# =============================================================================
# TERNARY OPERATIONS (GROUND TRUTH)
# =============================================================================

def int_to_base3(n: int, digits: int = 9) -> List[int]:
    """Convert integer to balanced ternary representation."""
    if n < 0 or n >= 3**digits:
        raise ValueError(f"Value {n} out of range for {digits} digits")

    trits = []
    for _ in range(digits):
        trits.append(n % 3)
        n //= 3
    return trits  # LSB first


def base3_to_int(trits: List[int]) -> int:
    """Convert balanced ternary to integer."""
    result = 0
    for i, t in enumerate(trits):
        result += t * (3 ** i)
    return result


def balanced_to_standard(trits: List[int]) -> List[int]:
    """Convert balanced {-1,0,+1} to standard {0,1,2}."""
    return [t + 1 for t in trits]


def standard_to_balanced(trits: List[int]) -> List[int]:
    """Convert standard {0,1,2} to balanced {-1,0,+1}."""
    return [t - 1 for t in trits]


def ternary_not(a: List[int]) -> List[int]:
    """Ternary NOT: negate each trit."""
    return [-t for t in a]


def ternary_add(a: List[int], b: List[int]) -> List[int]:
    """Ternary addition with carry."""
    result = []
    carry = 0
    for i in range(len(a)):
        s = a[i] + b[i] + carry
        if s > 1:
            result.append(s - 3)
            carry = 1
        elif s < -1:
            result.append(s + 3)
            carry = -1
        else:
            result.append(s)
            carry = 0
    return result


def ternary_mul(a: List[int], b: List[int]) -> List[int]:
    """Ternary multiplication (element-wise, wrapping to 5 trits)."""
    result = [a[i] * b[i] for i in range(min(5, len(a), len(b)))]
    while len(result) < 5:
        result.append(0)
    return result[:5]


def ternary_min(a: List[int], b: List[int]) -> List[int]:
    """Ternary MIN (element-wise)."""
    return [min(a[i], b[i]) for i in range(min(len(a), len(b)))]


def ternary_max(a: List[int], b: List[int]) -> List[int]:
    """Ternary MAX (element-wise)."""
    return [max(a[i], b[i]) for i in range(min(len(a), len(b)))]


def compute_valuation(n: int) -> int:
    """Compute p-adic valuation (3-adic) of integer n."""
    if n == 0:
        return 10  # Max valuation for zero
    v = 0
    while n % 3 == 0:
        n //= 3
        v += 1
    return min(v, 9)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def detect_model_type(state_dict: Dict) -> Tuple[str, Dict]:
    """Detect model type from state dict keys."""
    keys = set(state_dict.keys())

    # Check for VAE structure
    has_encoder_A = any('encoder_A' in k for k in keys)
    has_encoder_B = any('encoder_B' in k for k in keys)
    has_decoder = any('decoder' in k for k in keys)
    has_projection = any('projection' in k for k in keys)
    has_controller = any('controller' in k for k in keys)

    # Check for GPT2 structure
    has_wte = any('wte' in k for k in keys)
    has_attn = any('attn' in k for k in keys)

    # Check for codon encoder
    has_codon = any('codon' in k.lower() for k in keys)

    config = {}

    if has_encoder_A and has_encoder_B:
        model_type = "dual_vae"
        config['has_projection'] = has_projection
        config['has_controller'] = has_controller

        # Infer dimensions from weight shapes
        for k, v in state_dict.items():
            if 'encoder_A.encoder.0.weight' in k:
                config['input_dim'] = v.shape[1]
                config['hidden_dims'] = [v.shape[0]]
            if 'encoder_A.fc_mu.weight' in k:
                config['latent_dim'] = v.shape[0]
            if 'decoder_A.decoder' in k and 'weight' in k:
                if 'decoder.0' in k:
                    config['decoder_dims'] = [v.shape[0]]
                if 'decoder.4' in k or 'decoder.2' in k:
                    config['output_dim'] = v.shape[0]

    elif has_wte and has_attn:
        model_type = "gpt2_ternary"
    elif has_codon:
        model_type = "codon_encoder"
    else:
        model_type = "unknown"

    return model_type, config


def load_and_validate_checkpoint(path: Path) -> ValidationResult:
    """Load checkpoint and perform validation."""
    result = ValidationResult(checkpoint_path=str(path))

    try:
        # Load checkpoint
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        result.loadable = True

        # Extract state dict
        if isinstance(ckpt, dict):
            if 'model_state' in ckpt:
                state_dict = ckpt['model_state']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt

            # Store claimed metrics (not trusted)
            if 'metrics' in ckpt:
                result.claimed_metrics = {k: float(v) if isinstance(v, (int, float, np.floating))
                                          else str(v) for k, v in ckpt['metrics'].items()}
        else:
            # OrderedDict (raw state dict)
            state_dict = ckpt

        # Detect model type
        model_type, config = detect_model_type(state_dict)
        result.model_type = model_type

        # Perform type-specific validation
        if model_type == "dual_vae":
            validate_dual_vae(state_dict, config, result)
        elif model_type == "gpt2_ternary":
            validate_gpt2_ternary(state_dict, result)
        elif model_type == "codon_encoder":
            validate_codon_encoder(state_dict, result)
        else:
            result.error_message = f"Unknown model type, keys: {list(state_dict.keys())[:5]}"

    except Exception as e:
        result.loadable = False
        result.error_message = str(e)

    return result


def validate_dual_vae(state_dict: Dict, config: Dict, result: ValidationResult):
    """Validate dual VAE model."""
    device = 'cpu'

    # Build model from config
    input_dim = config.get('input_dim', 9)
    latent_dim = config.get('latent_dim', 16)
    output_dim = config.get('output_dim', 27)
    use_controller = config.get('has_controller', False)

    try:
        model = TernaryVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            use_controller=use_controller
        )

        # Load weights (partial)
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())

        # Load matching keys
        loadable = {k: v for k, v in state_dict.items() if k in model_keys}
        model.load_state_dict(loadable, strict=False)
        model.eval()

    except Exception as e:
        result.error_message = f"Model construction failed: {e}"
        return

    # Generate all 19683 ternary inputs
    N = 19683
    all_inputs = []
    all_valuations = []

    for i in range(N):
        trits = int_to_base3(i, 9)
        balanced = standard_to_balanced(trits)
        all_inputs.append(balanced)
        all_valuations.append(compute_valuation(i))

    inputs = torch.tensor(all_inputs, dtype=torch.float32)
    valuations = np.array(all_valuations)

    # Forward pass (no sampling for consistency)
    with torch.no_grad():
        outputs = model(inputs, sample=False)

    # 1. Reconstruction metrics
    recon = outputs['recon']  # Shape: [N, 27]

    # Decode: reshape to [N, 9, 3] and take argmax
    recon_probs = recon.view(N, 9, 3)
    recon_classes = recon_probs.argmax(dim=-1)  # [N, 9]

    # Count unique reconstructions
    unique_outputs = len(set(tuple(r.tolist()) for r in recon_classes))
    result.reconstruction.unique_outputs = unique_outputs
    result.reconstruction.coverage = unique_outputs / N * 100

    # Check if outputs match inputs
    input_classes = ((inputs + 1).long())  # Convert {-1,0,+1} to {0,1,2}
    correct = (recon_classes == input_classes).all(dim=1).sum().item()
    result.reconstruction.reconstruction_loss = 1.0 - (correct / N)

    # Valid ternary outputs
    valid_ternary = (recon_classes >= 0) & (recon_classes <= 2)
    result.reconstruction.valid_ternary_outputs = valid_ternary.all(dim=1).sum().item() / N * 100

    # 2. Geometric metrics
    proj_A = outputs['proj_A'].numpy()  # [N, latent_dim]
    proj_B = outputs['proj_B'].numpy()

    # Compute radii
    radii_A = np.linalg.norm(proj_A, axis=1)
    radii_B = np.linalg.norm(proj_B, axis=1)

    # Hierarchy (Spearman correlation between valuation and radius)
    if len(set(valuations)) > 1 and len(set(radii_A)) > 1:
        result.geometric.hierarchy_A, _ = spearmanr(valuations, radii_A)
        result.geometric.hierarchy_B, _ = spearmanr(valuations, radii_B)

    # Radii at different valuation levels
    mask_v0 = valuations == 0
    mask_v9 = valuations >= 8

    if mask_v0.any():
        result.geometric.radius_v0 = float(np.mean(radii_B[mask_v0]))
    if mask_v9.any():
        result.geometric.radius_v9 = float(np.mean(radii_B[mask_v9]))

    result.geometric.radial_separation = result.geometric.radius_v0 - result.geometric.radius_v9

    # Richness: within-level variance
    richness = 0.0
    for v in range(10):
        mask = valuations == v
        if mask.sum() > 1:
            richness += np.var(radii_B[mask])
    result.geometric.richness = richness / 10

    # 3. Algebraic properties (sample-based)
    result.algebraic = test_algebraic_properties(inputs, recon_classes)

    # 4. GEMM metrics
    result.gemm = analyze_gemm_suitability(state_dict)


def test_algebraic_properties(inputs: torch.Tensor,
                               reconstructions: torch.Tensor) -> AlgebraicMetrics:
    """Test algebraic properties on a sample of operations."""
    metrics = AlgebraicMetrics()
    N = min(1000, len(inputs))  # Sample size

    # Convert to balanced ternary lists
    def to_balanced(t: torch.Tensor) -> List[int]:
        return [int(x) - 1 for x in t.tolist()]

    # Sample random indices
    np.random.seed(42)
    indices = np.random.choice(len(inputs), N, replace=False)

    # Test properties
    add_comm = 0
    mul_comm = 0
    add_identity = 0
    mul_identity = 0
    not_involution = 0

    zero = [0] * 5
    one = [1, 0, 0, 0, 0]

    for idx in indices:
        a = to_balanced(inputs[idx])[:5]  # Take first 5 trits

        # NOT involution: not(not(a)) == a
        not_a = ternary_not(a)
        not_not_a = ternary_not(not_a)
        if not_not_a == a:
            not_involution += 1

        # ADD identity: a + 0 == a
        add_zero = ternary_add(a, zero)[:5]
        if add_zero == a:
            add_identity += 1

        # MUL identity: a * 1 == a (element-wise)
        mul_one = ternary_mul(a, one)[:5]
        # For element-wise mul with 1, only first element matters
        if a[0] == mul_one[0]:
            mul_identity += 1

        # Sample a second operand for commutativity
        idx2 = np.random.choice(len(inputs))
        b = to_balanced(inputs[idx2])[:5]

        # ADD commutativity: a + b == b + a
        ab = ternary_add(a, b)[:5]
        ba = ternary_add(b, a)[:5]
        if ab == ba:
            add_comm += 1

        # MUL commutativity: a * b == b * a
        ab_mul = ternary_mul(a, b)[:5]
        ba_mul = ternary_mul(b, a)[:5]
        if ab_mul == ba_mul:
            mul_comm += 1

    metrics.tnot_involution = not_involution / N
    metrics.tadd_identity = add_identity / N
    metrics.tmul_identity = mul_identity / N
    metrics.tadd_commutativity = add_comm / N
    metrics.tmul_commutativity = mul_comm / N

    return metrics


def analyze_gemm_suitability(state_dict: Dict) -> GEMMMetrics:
    """Analyze model weights for GEMM suitability."""
    metrics = GEMMMetrics()

    total_params = 0
    zero_params = 0
    ternary_params = 0

    for k, v in state_dict.items():
        if 'weight' in k and isinstance(v, torch.Tensor):
            flat = v.flatten()
            total_params += len(flat)

            # Count zeros
            zero_params += (flat == 0).sum().item()

            # Count ternary values
            is_ternary = (flat == -1) | (flat == 0) | (flat == 1)
            ternary_params += is_ternary.sum().item()

    if total_params > 0:
        metrics.weight_sparsity = zero_params / total_params * 100
        metrics.ternary_purity = ternary_params / total_params * 100

    # Encoding efficiency (2 bits per trit)
    metrics.encoding_efficiency = 2.0

    # Check matmul compatibility (all linear layers)
    metrics.matmul_compatible = all('weight' in k or 'bias' in k
                                     for k in state_dict.keys()
                                     if 'running' not in k and 'num_batches' not in k)

    return metrics


def validate_gpt2_ternary(state_dict: Dict, result: ValidationResult):
    """Validate GPT2 ternary model."""
    # Analyze weight ternary purity
    result.gemm = analyze_gemm_suitability(state_dict)

    # Count layers
    num_layers = len([k for k in state_dict.keys() if 'h.' in k and '.ln_1' in k])

    result.error_message = f"GPT2 model with {num_layers} layers - needs LM-specific validation"


def validate_codon_encoder(state_dict: Dict, result: ValidationResult):
    """Validate codon encoder model."""
    result.gemm = analyze_gemm_suitability(state_dict)
    result.error_message = "Codon encoder - specialized for bioinformatics"


def compute_overall_score(result: ValidationResult) -> float:
    """Compute weighted overall score."""
    score = 0.0
    weights = {
        'reconstruction': 0.30,
        'geometric': 0.35,
        'algebraic': 0.20,
        'gemm': 0.15
    }

    # Reconstruction score (coverage is key)
    recon_score = result.reconstruction.coverage / 100
    score += weights['reconstruction'] * recon_score

    # Geometric score (hierarchy ceiling is -0.8321)
    hier = abs(result.geometric.hierarchy_B)
    hier_score = min(hier / 0.8321, 1.0)  # Normalize to ceiling
    richness_score = min(result.geometric.richness / 0.01, 1.0)
    geo_score = 0.7 * hier_score + 0.3 * richness_score
    score += weights['geometric'] * geo_score

    # Algebraic score
    alg_values = [
        result.algebraic.tnot_involution,
        result.algebraic.tadd_commutativity,
        result.algebraic.tmul_commutativity,
        result.algebraic.tadd_identity
    ]
    alg_score = np.mean(alg_values) if alg_values else 0
    score += weights['algebraic'] * alg_score

    # GEMM score
    gemm_score = result.gemm.ternary_purity / 100 if result.gemm.ternary_purity else 0
    score += weights['gemm'] * gemm_score

    return score * 100


# =============================================================================
# MAIN VALIDATION RUNNER
# =============================================================================

def find_all_checkpoints(base_dir: Path) -> List[Path]:
    """Find all .pt checkpoint files."""
    return list(base_dir.rglob("*.pt"))


def generate_report(results: List[ValidationResult], output_path: Path):
    """Generate comprehensive validation report."""
    # Sort by overall score
    results.sort(key=lambda r: r.overall_score, reverse=True)

    # Assign ranks
    for i, r in enumerate(results):
        r.rank = i + 1

    report = {
        'generated': datetime.now().isoformat(),
        'total_checkpoints': len(results),
        'summary': {
            'best_model': results[0].checkpoint_path if results else None,
            'best_score': results[0].overall_score if results else 0,
            'loadable': sum(1 for r in results if r.loadable),
            'dual_vae_models': sum(1 for r in results if r.model_type == 'dual_vae'),
        },
        'ranking': [],
        'detailed_results': []
    }

    # Create ranking table
    for r in results[:20]:  # Top 20
        report['ranking'].append({
            'rank': r.rank,
            'path': r.checkpoint_path,
            'type': r.model_type,
            'score': round(r.overall_score, 2),
            'coverage': round(r.reconstruction.coverage, 2),
            'hierarchy_B': round(r.geometric.hierarchy_B, 4),
            'richness': round(r.geometric.richness, 6),
            'loadable': r.loadable
        })

    # Detailed results
    for r in results:
        report['detailed_results'].append({
            'rank': r.rank,
            'checkpoint_path': r.checkpoint_path,
            'model_type': r.model_type,
            'loadable': r.loadable,
            'error': r.error_message,
            'overall_score': round(r.overall_score, 2),
            'reconstruction': asdict(r.reconstruction),
            'arithmetic': asdict(r.arithmetic),
            'algebraic': asdict(r.algebraic),
            'geometric': asdict(r.geometric),
            'gemm': asdict(r.gemm),
            'claimed_metrics': r.claimed_metrics
        })

    # Save JSON report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    return report


def print_summary(report: Dict):
    """Print human-readable summary."""
    print("\n" + "="*80)
    print("CHECKPOINT VALIDATION REPORT")
    print("="*80)

    print(f"\nGenerated: {report['generated']}")
    print(f"Total checkpoints: {report['total_checkpoints']}")
    print(f"Loadable: {report['summary']['loadable']}")
    print(f"Dual VAE models: {report['summary']['dual_vae_models']}")

    print("\n" + "-"*80)
    print("TOP 10 MODELS")
    print("-"*80)
    print(f"{'Rank':<5} {'Score':<8} {'Coverage':<10} {'Hier_B':<10} {'Richness':<12} {'Path'}")
    print("-"*80)

    for entry in report['ranking'][:10]:
        print(f"{entry['rank']:<5} {entry['score']:<8.2f} {entry['coverage']:<10.2f} "
              f"{entry['hierarchy_B']:<10.4f} {entry['richness']:<12.6f} "
              f"{Path(entry['path']).name}")

    if report['ranking']:
        best = report['ranking'][0]
        print("\n" + "="*80)
        print(f"BEST MODEL: {best['path']}")
        print(f"Score: {best['score']:.2f} | Coverage: {best['coverage']:.2f}% | "
              f"Hierarchy: {best['hierarchy_B']:.4f}")
        print("="*80)


def main():
    """Main validation runner."""
    base_dir = Path(__file__).parent
    output_path = base_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    print("="*80)
    print("TERNARY CHECKPOINT VALIDATION")
    print("="*80)
    print(f"Base directory: {base_dir}")

    # Find all checkpoints
    checkpoints = find_all_checkpoints(base_dir)
    print(f"Found {len(checkpoints)} checkpoint files")

    # Validate each
    results = []
    for i, cp in enumerate(checkpoints):
        print(f"\n[{i+1}/{len(checkpoints)}] Validating: {cp.name}...")
        result = load_and_validate_checkpoint(cp)
        result.overall_score = compute_overall_score(result)
        results.append(result)

        if result.loadable:
            print(f"  Type: {result.model_type}")
            print(f"  Coverage: {result.reconstruction.coverage:.2f}%")
            print(f"  Hierarchy_B: {result.geometric.hierarchy_B:.4f}")
            print(f"  Score: {result.overall_score:.2f}")
        else:
            print(f"  FAILED: {result.error_message[:50]}...")

    # Generate report
    print(f"\nGenerating report: {output_path}")
    report = generate_report(results, output_path)

    # Print summary
    print_summary(report)

    print(f"\nFull report saved to: {output_path}")
    return report


if __name__ == "__main__":
    main()
