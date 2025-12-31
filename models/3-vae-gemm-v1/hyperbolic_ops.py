"""
hyperbolic_ops.py - Hyperbolic Operations for Non-Euclidean Embedding Space

The embedding space is NOT Euclidean. It is:
- p-adic (3-adic): Distance determined by divisibility by 3
- Non-Archimedean: |a + b| ≤ max(|a|, |b|) (ultrametric inequality)
- Hyperbolic: Negative curvature, tree-like hierarchy
- Ultrametric: All triangles isoceles with equal sides being longest

Key insight: Operations don't "classify" into bins - they create TRAJECTORIES
that flow along geodesics to natural ATTRACTOR BASINS.

Mathematical foundations:
- Poincaré ball model: B^n = {x ∈ R^n : ||x|| < 1}
- Metric: ds² = 4(dx₁² + ... + dxₙ²) / (1 - ||x||²)²
- Möbius addition: x ⊕ y = ((1 + 2⟨x,y⟩ + ||y||²)x + (1 - ||x||²)y) / (1 + 2⟨x,y⟩ + ||x||²||y||²)
- Exponential map: exp_x(v) = x ⊕ (tanh(||v||/2) * v/||v||)
- Logarithmic map: log_x(y) = artanh(||−x⊕y||) * (−x⊕y)/||−x⊕y||

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


# =============================================================================
# POINCARÉ BALL OPERATIONS
# =============================================================================

class PoincareOperations:
    """
    Operations in the Poincaré ball model of hyperbolic space.

    The Poincaré ball B^n has:
    - Points: {x ∈ R^n : ||x|| < 1}
    - Curvature: -c (we use c=1 by default)
    - Boundary at infinity: ||x|| → 1

    In p-adic terms:
    - Center (||x|| ≈ 0): High valuation (divisible by high powers of 3)
    - Boundary (||x|| → 1): Low valuation (not divisible by 3)
    """

    EPS = 1e-5
    MAX_NORM = 1.0 - 1e-5

    @staticmethod
    def project(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """Project points to Poincaré ball (ensure ||x|| < 1/√c)."""
        max_norm = (1.0 / math.sqrt(c)) - PoincareOperations.EPS
        norm = x.norm(dim=-1, keepdim=True).clamp(min=PoincareOperations.EPS)
        cond = norm > max_norm
        projected = x / norm * max_norm
        return torch.where(cond, projected, x)

    @staticmethod
    def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """
        Möbius addition in Poincaré ball.

        x ⊕_c y = ((1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y) /
                   (1 + 2c⟨x,y⟩ + c²||x||²||y||²)

        This is the hyperbolic analog of vector addition.
        """
        x_sq = (x * x).sum(dim=-1, keepdim=True)
        y_sq = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)

        num = (1 + 2*c*xy + c*y_sq) * x + (1 - c*x_sq) * y
        denom = 1 + 2*c*xy + c*c*x_sq*y_sq

        result = num / denom.clamp(min=PoincareOperations.EPS)
        return PoincareOperations.project(result, c)

    @staticmethod
    def mobius_scalar(r: torch.Tensor, x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """
        Möbius scalar multiplication.

        r ⊗_c x = tanh(r * artanh(√c||x||)) * x / (√c||x||)
        """
        x_norm = x.norm(dim=-1, keepdim=True).clamp(min=PoincareOperations.EPS)
        sqrt_c = math.sqrt(c)

        # artanh(√c||x||)
        scaled_norm = (sqrt_c * x_norm).clamp(max=1.0 - PoincareOperations.EPS)
        artanh_norm = 0.5 * torch.log((1 + scaled_norm) / (1 - scaled_norm))

        # tanh(r * artanh(...))
        scaled = torch.tanh(r * artanh_norm)

        result = scaled * x / (sqrt_c * x_norm)
        return PoincareOperations.project(result, c)

    @staticmethod
    def geodesic_midpoint(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """
        Compute geodesic midpoint in Poincaré ball.

        This is NOT (x + y) / 2 !!!

        The geodesic midpoint lies on the geodesic connecting x and y,
        equidistant (in hyperbolic metric) from both points.

        Midpoint = x ⊕ (0.5 ⊗ (−x ⊕ y))
        """
        # -x in Möbius sense is just -x (negation works)
        neg_x = -x

        # -x ⊕ y (transport y to origin-centered view)
        diff = PoincareOperations.mobius_add(neg_x, y, c)

        # 0.5 ⊗ diff (half the distance along geodesic)
        half = torch.full_like(diff[..., :1], 0.5)
        half_diff = PoincareOperations.mobius_scalar(half, diff, c)

        # x ⊕ half_diff (transport back)
        midpoint = PoincareOperations.mobius_add(x, half_diff, c)

        return midpoint

    @staticmethod
    def hyperbolic_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """
        Hyperbolic distance in Poincaré ball.

        d_c(x, y) = (2/√c) * artanh(√c ||−x ⊕ y||)
        """
        diff = PoincareOperations.mobius_add(-x, y, c)
        diff_norm = diff.norm(dim=-1, keepdim=True).clamp(min=PoincareOperations.EPS)
        sqrt_c = math.sqrt(c)

        scaled_norm = (sqrt_c * diff_norm).clamp(max=1.0 - PoincareOperations.EPS)
        dist = (2 / sqrt_c) * 0.5 * torch.log((1 + scaled_norm) / (1 - scaled_norm))

        return dist.squeeze(-1)

    @staticmethod
    def exp_map(x: torch.Tensor, v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """
        Exponential map: move from x in direction v (tangent vector).

        exp_x(v) = x ⊕ (tanh(√c||v||/2) * v / (√c||v||))
        """
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=PoincareOperations.EPS)
        sqrt_c = math.sqrt(c)

        # Direction
        v_unit = v / v_norm

        # Distance to move (in hyperbolic terms)
        coef = torch.tanh(sqrt_c * v_norm / 2) / sqrt_c

        # Apply Möbius addition
        result = PoincareOperations.mobius_add(x, coef * v_unit, c)
        return result

    @staticmethod
    def log_map(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """
        Logarithmic map: tangent vector from x pointing to y.

        log_x(y) = (2 / (√c * λ_x)) * artanh(√c||−x⊕y||) * (−x⊕y) / ||−x⊕y||

        where λ_x = 2 / (1 - c||x||²) is the conformal factor.
        """
        diff = PoincareOperations.mobius_add(-x, y, c)
        diff_norm = diff.norm(dim=-1, keepdim=True).clamp(min=PoincareOperations.EPS)
        sqrt_c = math.sqrt(c)

        x_sq = (x * x).sum(dim=-1, keepdim=True)
        lambda_x = 2 / (1 - c * x_sq).clamp(min=PoincareOperations.EPS)

        scaled_norm = (sqrt_c * diff_norm).clamp(max=1.0 - PoincareOperations.EPS)
        artanh_norm = 0.5 * torch.log((1 + scaled_norm) / (1 - scaled_norm))

        result = (2 / (sqrt_c * lambda_x)) * artanh_norm * diff / diff_norm
        return result


# =============================================================================
# HYPERBOLIC LAYERS
# =============================================================================

class HyperbolicLinear(nn.Module):
    """
    Linear layer in hyperbolic space.

    Instead of W @ x + b (Euclidean), we use:
    1. Log map to tangent space at origin
    2. Linear transform in tangent space
    3. Exp map back to hyperbolic space
    """

    def __init__(self, in_features: int, out_features: int, c: float = 1.0, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c

        # Weight operates in tangent space
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is in Poincaré ball
        # Map to tangent space at origin
        v = PoincareOperations.log_map(torch.zeros_like(x), x, self.c)

        # Linear transform in tangent space (Euclidean)
        v_transformed = F.linear(v, self.weight, self.bias)

        # Map back to Poincaré ball
        y = PoincareOperations.exp_map(torch.zeros_like(v_transformed), v_transformed, self.c)

        return y


class HyperbolicMLR(nn.Module):
    """
    Hyperbolic Multinomial Logistic Regression.

    For classification in hyperbolic space, we don't use Euclidean softmax.
    Instead, we measure hyperbolic distances to class "centroids" (attractors).

    P(y = k | x) ∝ exp(-d_hyp(x, μ_k)² / τ)

    where μ_k are learnable hyperbolic centroids and τ is temperature.
    """

    def __init__(self, embedding_dim: int, num_classes: int, c: float = 1.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.c = c

        # Learnable class centroids in Poincaré ball
        # Initialize near origin (high valuation region)
        self.centroids = nn.Parameter(torch.randn(num_classes, embedding_dim) * 0.1)

        # Temperature for softmax sharpness
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log-probabilities based on hyperbolic distances to centroids.

        Args:
            x: [B, D] embeddings in Poincaré ball

        Returns:
            log_probs: [B, num_classes]
        """
        # Project centroids to valid Poincaré ball
        centroids = PoincareOperations.project(self.centroids, self.c)

        # Compute hyperbolic distance from each x to each centroid
        # x: [B, D], centroids: [K, D] -> distances: [B, K]
        x_expanded = x.unsqueeze(1)  # [B, 1, D]
        centroids_expanded = centroids.unsqueeze(0)  # [1, K, D]

        distances = PoincareOperations.hyperbolic_distance(
            x_expanded.expand(-1, self.num_classes, -1),
            centroids_expanded.expand(x.shape[0], -1, -1),
            self.c
        )  # [B, K]

        # Convert distances to log-probabilities
        # Closer = higher probability
        log_probs = F.log_softmax(-distances.pow(2) / self.temperature.abs(), dim=-1)

        return log_probs

    def get_attractor_for_class(self, class_idx: int) -> torch.Tensor:
        """Get the hyperbolic centroid (attractor) for a class."""
        return PoincareOperations.project(self.centroids[class_idx], self.c)


# =============================================================================
# ULTRAMETRIC ATTRACTOR DYNAMICS
# =============================================================================

class UltrametricAttractorField(nn.Module):
    """
    Models the attractor field in ultrametric embedding space.

    Key insight: Each ternary value defines an attractor basin.
    Operations create trajectories that flow to these basins.

    The field respects ultrametric structure:
    - Values with same valuation are equidistant from center
    - Transitions between valuation levels have discrete jumps
    - The strong triangle inequality d(a,c) ≤ max(d(a,b), d(b,c))
    """

    def __init__(
        self,
        num_values: int = 19683,
        embedding_dim: int = 16,
        num_trits: int = 9,
        c: float = 1.0
    ):
        super().__init__()
        self.num_values = num_values
        self.embedding_dim = embedding_dim
        self.num_trits = num_trits
        self.c = c

        # Learnable attractor positions in Poincaré ball
        # Initialize with radial structure based on valuation
        self._init_attractors()

        # Per-valuation radial targets (0 = boundary, num_trits = center)
        self.valuation_radii = nn.Parameter(
            torch.linspace(0.9, 0.1, num_trits + 1)
        )

    def _init_attractors(self):
        """Initialize attractors with p-adic structure."""
        attractors = torch.zeros(self.num_values, self.embedding_dim)

        for idx in range(self.num_values):
            # Compute valuation
            val = self._valuation(idx)

            # Target radius: high valuation → near center
            target_r = 0.9 - (val / self.num_trits) * 0.8

            # Random direction
            direction = torch.randn(self.embedding_dim)
            direction = direction / direction.norm()

            # Place at target radius
            attractors[idx] = direction * target_r

        self.attractors = nn.Parameter(attractors)

    def _valuation(self, n: int) -> int:
        """3-adic valuation."""
        if n == 0:
            return self.num_trits
        v = 0
        while n % 3 == 0 and v < self.num_trits:
            n //= 3
            v += 1
        return v

    def get_attractor(self, idx: torch.Tensor) -> torch.Tensor:
        """Get attractor position for given indices."""
        return PoincareOperations.project(self.attractors[idx], self.c)

    def _batch_hyperbolic_distance(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise hyperbolic distances efficiently.

        Args:
            x: [B, D] query points
            y: [N, D] attractor points

        Returns:
            distances: [B, N] pairwise distances
        """
        # Expand for broadcasting: x [B, 1, D], y [1, N, D]
        x_exp = x.unsqueeze(1)  # [B, 1, D]
        y_exp = y.unsqueeze(0)  # [1, N, D]

        # Möbius addition: -x ⊕ y (vectorized)
        neg_x = -x_exp  # [B, 1, D]

        x_sq = (neg_x * neg_x).sum(dim=-1, keepdim=True)  # [B, 1, 1]
        y_sq = (y_exp * y_exp).sum(dim=-1, keepdim=True)  # [1, N, 1]
        xy = (neg_x * y_exp).sum(dim=-1, keepdim=True)    # [B, N, 1]

        c = self.c
        num = (1 + 2*c*xy + c*y_sq) * neg_x + (1 - c*x_sq) * y_exp
        denom = 1 + 2*c*xy + c*c*x_sq*y_sq

        diff = num / denom.clamp(min=1e-5)  # [B, N, D]

        # Hyperbolic distance from diff norm
        diff_norm = diff.norm(dim=-1)  # [B, N]
        sqrt_c = math.sqrt(c)
        scaled_norm = (sqrt_c * diff_norm).clamp(max=1.0 - 1e-5)
        distances = (2 / sqrt_c) * 0.5 * torch.log((1 + scaled_norm) / (1 - scaled_norm))

        return distances

    def find_nearest_attractor(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find nearest attractor using efficient batched computation.

        Args:
            x: [B, D] or [D] query points

        Returns:
            nearest_attractor: [B, D] or [D] attractor positions
            attractor_idx: [B] or scalar indices
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        # Project attractors to valid Poincaré ball
        attractors = PoincareOperations.project(self.attractors, self.c)

        # Compute all pairwise distances at once
        distances = self._batch_hyperbolic_distance(x, attractors)  # [B, N]

        # Find nearest
        attractor_idx = distances.argmin(dim=-1)  # [B]
        nearest_attractor = attractors[attractor_idx]  # [B, D]

        if squeeze:
            return nearest_attractor.squeeze(0), attractor_idx.squeeze(0)
        return nearest_attractor, attractor_idx

    def flow_to_nearest_attractor(
        self,
        x: torch.Tensor,
        steps: int = 3,
        step_size: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Flow a point to its nearest attractor via gradient descent on hyperbolic distance.

        Uses efficient batched operations for O(B*N) instead of O(B*N*steps) complexity.

        Returns:
            final_position: Position after flow
            attractor_idx: Index of nearest attractor
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        current = x.clone()
        attractors = PoincareOperations.project(self.attractors, self.c)

        for _ in range(steps):
            # Find nearest attractor (batched)
            distances = self._batch_hyperbolic_distance(current, attractors)
            nearest_idx = distances.argmin(dim=-1)
            nearest_attractor = attractors[nearest_idx]

            # Move toward nearest attractor along geodesic
            direction = PoincareOperations.log_map(current, nearest_attractor, self.c)

            # Step toward attractor
            step = direction * step_size
            current = PoincareOperations.exp_map(current, step, self.c)

        # Final nearest attractor
        _, attractor_idx = self.find_nearest_attractor(current)

        if squeeze:
            return current.squeeze(0), attractor_idx
        return current, attractor_idx


class HyperbolicOperationFlow(nn.Module):
    """
    Models how operations create FLOWS in hyperbolic space.

    Instead of: MLP(emb_a, emb_b, op) → emb_result (WRONG - Euclidean)

    We use: Operation creates a FLOW FIELD that moves the geodesic midpoint
    toward the correct attractor basin.

    The key insight: The operation doesn't "classify" - it creates a
    TRAJECTORY in the non-Euclidean space that naturally flows to the
    correct attractor.
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        num_operations: int = 4,
        num_values: int = 19683,
        c: float = 1.0,
        flow_steps: int = 5
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_operations = num_operations
        self.num_values = num_values
        self.c = c
        self.flow_steps = flow_steps

        # Operation embeddings - these modulate the flow field
        self.op_embedding = nn.Embedding(num_operations, embedding_dim)

        # Flow field generator: given operation, generates tangent vectors
        # that guide the flow from midpoint to result
        self.flow_generator = nn.Sequential(
            nn.Linear(embedding_dim * 3, 64),  # (emb_a, emb_b, op_emb)
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, embedding_dim * flow_steps)  # Generate flow vectors
        )

        # Attractor field
        self.attractor_field = UltrametricAttractorField(
            num_values=num_values,
            embedding_dim=embedding_dim,
            c=c
        )

    def compute_trajectory(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        op_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the trajectory from geodesic midpoint to result attractor.

        Args:
            emb_a: [B, D] embedding of first operand
            emb_b: [B, D] embedding of second operand
            op_id: [B] operation indices

        Returns:
            trajectory: [B, steps+1, D] positions along flow
            final_position: [B, D] final position
        """
        batch_size = emb_a.shape[0]

        # Start at geodesic midpoint (NOT Euclidean average!)
        midpoint = PoincareOperations.geodesic_midpoint(emb_a, emb_b, self.c)

        # Get operation embedding
        op_emb = self.op_embedding(op_id)  # [B, D]

        # Generate flow vectors
        flow_input = torch.cat([emb_a, emb_b, op_emb], dim=-1)  # [B, 3D]
        flow_vectors = self.flow_generator(flow_input)  # [B, D * steps]
        flow_vectors = flow_vectors.view(batch_size, self.flow_steps, self.embedding_dim)

        # Execute flow trajectory
        trajectory = [midpoint]
        current = midpoint

        for step in range(self.flow_steps):
            # Flow vector for this step (tangent vector at current position)
            v = flow_vectors[:, step, :]

            # Move along geodesic in direction v
            current = PoincareOperations.exp_map(current, v, self.c)
            trajectory.append(current)

        trajectory = torch.stack(trajectory, dim=1)  # [B, steps+1, D]
        final_position = current

        return trajectory, final_position

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        op_id: torch.Tensor,
        return_trajectory: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute operation result via hyperbolic flow.

        Returns:
            Dictionary containing:
            - predicted_emb: Final position after flow
            - predicted_idx: Nearest attractor index (predicted result)
            - trajectory: (optional) Full trajectory
        """
        trajectory, final_pos = self.compute_trajectory(emb_a, emb_b, op_id)

        # Find nearest attractor
        _, predicted_idx = self.attractor_field.flow_to_nearest_attractor(
            final_pos, steps=3, step_size=0.2
        )

        result = {
            'predicted_emb': final_pos,
            'predicted_idx': predicted_idx,
            'midpoint': trajectory[:, 0],  # Geodesic midpoint
        }

        if return_trajectory:
            result['trajectory'] = trajectory

        return result


# =============================================================================
# HYPERBOLIC VAE ENCODER
# =============================================================================

class HyperbolicEncoder(nn.Module):
    """
    Encoder that maps ternary values to Poincaré ball.

    Key property: Valuation determines radial position.
    - High valuation (divisible by 3^k) → near center (small ||x||)
    - Low valuation → near boundary (large ||x||)
    """

    def __init__(
        self,
        num_trits: int = 9,
        hidden_dims: list = [64, 32],
        latent_dim: int = 16,
        c: float = 1.0
    ):
        super().__init__()
        self.num_trits = num_trits
        self.latent_dim = latent_dim
        self.c = c

        # Euclidean encoder (operates in tangent space)
        layers = []
        in_dim = num_trits
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
            ])
            in_dim = h_dim

        self.encoder = nn.Sequential(*layers)

        # Output mean in tangent space (will be mapped to Poincaré ball)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)

        # Output controls radial position (related to valuation)
        self.fc_radius = nn.Linear(hidden_dims[-1], 1)

    def forward(self, trits: torch.Tensor) -> torch.Tensor:
        """
        Encode trit sequence to Poincaré ball.

        Args:
            trits: [B, num_trits] values in {-1, 0, +1}

        Returns:
            embedding: [B, latent_dim] in Poincaré ball
        """
        h = self.encoder(trits.float())

        # Direction in tangent space
        direction = self.fc_mu(h)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp(min=1e-5)

        # Radius (constrained to Poincaré ball)
        radius = torch.sigmoid(self.fc_radius(h)) * 0.95  # Max radius < 1

        # Combine direction and radius
        embedding = direction * radius

        return PoincareOperations.project(embedding, self.c)


# =============================================================================
# FULL HYPERBOLIC OPERATION MODEL
# =============================================================================

class HyperbolicOperationModel(nn.Module):
    """
    Full model for operation-aware embeddings in hyperbolic space.

    Architecture:
    1. HyperbolicEncoder: trits → Poincaré ball (respects p-adic structure)
    2. HyperbolicOperationFlow: (emb_a, emb_b, op) → trajectory → result_emb
    3. UltrametricAttractorField: result_emb → attractor basin → result_idx

    Training:
    - Encoder learns to place values with correct radial structure
    - Flow learns to create trajectories that land in correct attractors
    - Attractors self-organize to respect ultrametric constraints
    """

    def __init__(
        self,
        num_trits: int = 9,
        num_values: int = 19683,
        latent_dim: int = 16,
        num_operations: int = 4,
        c: float = 1.0
    ):
        super().__init__()
        self.num_trits = num_trits
        self.num_values = num_values
        self.latent_dim = latent_dim
        self.c = c

        # Encoder
        self.encoder = HyperbolicEncoder(
            num_trits=num_trits,
            latent_dim=latent_dim,
            c=c
        )

        # Operation flow
        self.operation_flow = HyperbolicOperationFlow(
            embedding_dim=latent_dim,
            num_operations=num_operations,
            num_values=num_values,
            c=c
        )

        # Trit table for index → trit conversion
        self._build_trit_table()

        # Valuation table
        self._build_valuation_table()

    def _build_trit_table(self):
        """Precompute index → trit conversion."""
        table = torch.zeros(self.num_values, self.num_trits, dtype=torch.float32)
        for idx in range(self.num_values):
            n = idx
            for i in range(self.num_trits):
                table[idx, i] = (n % 3) - 1
                n //= 3
        self.register_buffer('trit_table', table)

    def _build_valuation_table(self):
        """Precompute valuations."""
        valuations = torch.zeros(self.num_values, dtype=torch.long)
        for idx in range(self.num_values):
            if idx == 0:
                valuations[idx] = self.num_trits
            else:
                n = idx
                v = 0
                while n % 3 == 0 and v < self.num_trits:
                    n //= 3
                    v += 1
                valuations[idx] = v
        self.register_buffer('valuations', valuations)

    def encode(self, indices: torch.Tensor) -> torch.Tensor:
        """Encode indices to Poincaré ball."""
        trits = self.trit_table[indices]
        return self.encoder(trits)

    def forward_operation(
        self,
        idx_a: torch.Tensor,
        idx_b: torch.Tensor,
        op_id: torch.Tensor,
        idx_result: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for operation prediction.

        Returns predicted result embedding and index via hyperbolic flow.
        """
        # Encode operands
        emb_a = self.encode(idx_a)
        emb_b = self.encode(idx_b)

        # Compute operation flow
        result = self.operation_flow(emb_a, emb_b, op_id, return_trajectory=True)

        # Add target info if provided
        if idx_result is not None:
            result['target_emb'] = self.encode(idx_result)
            result['target_idx'] = idx_result

            # Get target attractor position
            result['target_attractor'] = self.operation_flow.attractor_field.get_attractor(idx_result)

        return result

    def get_all_attractors(self) -> torch.Tensor:
        """Get all attractor positions."""
        return PoincareOperations.project(
            self.operation_flow.attractor_field.attractors,
            self.c
        )


# =============================================================================
# LOSS FUNCTIONS FOR HYPERBOLIC SPACE
# =============================================================================

class HyperbolicOperationLoss(nn.Module):
    """
    Loss function for training hyperbolic operation model.

    Components:
    1. Trajectory loss: Final position should be near target attractor
    2. Flow smoothness: Trajectory should follow geodesics smoothly
    3. Attractor structure: Attractors should respect ultrametric constraints
    4. Radial alignment: Valuation should correlate with radius
    """

    def __init__(self, c: float = 1.0):
        super().__init__()
        self.c = c

    def trajectory_loss(
        self,
        final_pos: torch.Tensor,
        target_attractor: torch.Tensor
    ) -> torch.Tensor:
        """Loss: final position should be near target attractor."""
        return PoincareOperations.hyperbolic_distance(final_pos, target_attractor, self.c).mean()

    def flow_smoothness_loss(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Loss: trajectory should be smooth (no sudden jumps)."""
        # Compute velocity at each step
        velocities = []
        for t in range(trajectory.shape[1] - 1):
            v = PoincareOperations.log_map(trajectory[:, t], trajectory[:, t+1], self.c)
            velocities.append(v.norm(dim=-1))

        velocities = torch.stack(velocities, dim=1)  # [B, steps]

        # Penalize variance in velocities (should be constant for geodesic)
        return velocities.var(dim=1).mean()

    def attractor_ultrametric_loss(
        self,
        attractors: torch.Tensor,
        valuations: torch.Tensor
    ) -> torch.Tensor:
        """
        Loss: attractors should satisfy ultrametric constraints.

        For triplets (a, b, c):
        d(a, c) ≤ max(d(a, b), d(b, c))
        """
        # Sample random triplets
        n = min(100, attractors.shape[0])
        idx = torch.randperm(attractors.shape[0])[:n]

        loss = 0.0
        count = 0

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = attractors[idx[i]], attractors[idx[j]], attractors[idx[k]]

                    d_ab = PoincareOperations.hyperbolic_distance(a.unsqueeze(0), b.unsqueeze(0), self.c)
                    d_bc = PoincareOperations.hyperbolic_distance(b.unsqueeze(0), c.unsqueeze(0), self.c)
                    d_ac = PoincareOperations.hyperbolic_distance(a.unsqueeze(0), c.unsqueeze(0), self.c)

                    # Ultrametric violation: d(a,c) > max(d(a,b), d(b,c))
                    violation = F.relu(d_ac - torch.max(d_ab, d_bc))
                    loss += violation
                    count += 1

        return loss / max(count, 1)

    def radial_alignment_loss(
        self,
        embeddings: torch.Tensor,
        valuations: torch.Tensor
    ) -> torch.Tensor:
        """
        Loss: radius should inversely correlate with valuation.

        High valuation → small radius (near center)
        Low valuation → large radius (near boundary)
        """
        radii = embeddings.norm(dim=-1)

        # Target radius: high valuation → small radius
        max_val = valuations.max().float()
        target_radii = 0.9 - (valuations.float() / max_val) * 0.8

        return F.mse_loss(radii, target_radii)

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        valuations: torch.Tensor,
        attractors: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses."""
        losses = {}

        # Trajectory loss (primary) - weight: 1.0
        if 'predicted_emb' in model_output and 'target_attractor' in model_output:
            losses['trajectory'] = self.trajectory_loss(
                model_output['predicted_emb'],
                model_output['target_attractor']
            )

        # Flow smoothness - weight: 0.1
        if 'trajectory' in model_output:
            losses['smoothness'] = 0.1 * self.flow_smoothness_loss(model_output['trajectory'])

        # Radial alignment (CRITICAL for H2) - weight: 0.5
        # High valuation values should have smaller radius (nearer to center)
        if 'predicted_emb' in model_output:
            losses['radial'] = 0.5 * self.radial_alignment_loss(
                model_output['predicted_emb'],
                valuations
            )

        # Also enforce radial structure on target embeddings
        if 'target_emb' in model_output:
            losses['radial_target'] = 0.5 * self.radial_alignment_loss(
                model_output['target_emb'],
                valuations
            )

        # Ultrametric structure on predicted embeddings - weight: 0.3
        if 'predicted_emb' in model_output and len(model_output['predicted_emb']) >= 10:
            losses['ultrametric'] = 0.3 * self.attractor_ultrametric_loss(
                model_output['predicted_emb'],
                valuations
            )

        # If attractors provided, also enforce ultrametric on them
        if attractors is not None:
            losses['ultrametric_attractors'] = 0.3 * self.attractor_ultrametric_loss(
                attractors, valuations
            )

        # Total
        losses['total'] = sum(losses.values())

        return losses


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Hyperbolic Operations...")

    # Test Poincaré operations
    x = torch.tensor([[0.3, 0.2, 0.1]])
    y = torch.tensor([[0.1, 0.4, 0.2]])

    print(f"\nx = {x}")
    print(f"y = {y}")

    # Euclidean midpoint (WRONG)
    euc_mid = (x + y) / 2
    print(f"\nEuclidean midpoint: {euc_mid}")
    print(f"  ||euc_mid|| = {euc_mid.norm():.4f}")

    # Geodesic midpoint (CORRECT)
    hyp_mid = PoincareOperations.geodesic_midpoint(x, y)
    print(f"\nGeodesic midpoint: {hyp_mid}")
    print(f"  ||hyp_mid|| = {hyp_mid.norm():.4f}")

    # Distances
    d_xy = PoincareOperations.hyperbolic_distance(x, y)
    d_x_euc = PoincareOperations.hyperbolic_distance(x, euc_mid)
    d_y_euc = PoincareOperations.hyperbolic_distance(y, euc_mid)
    d_x_hyp = PoincareOperations.hyperbolic_distance(x, hyp_mid)
    d_y_hyp = PoincareOperations.hyperbolic_distance(y, hyp_mid)

    print(f"\nHyperbolic distances:")
    print(f"  d(x, y) = {d_xy.item():.4f}")
    print(f"  d(x, euc_mid) = {d_x_euc.item():.4f}")
    print(f"  d(y, euc_mid) = {d_y_euc.item():.4f}")
    print(f"  d(x, hyp_mid) = {d_x_hyp.item():.4f}")
    print(f"  d(y, hyp_mid) = {d_y_hyp.item():.4f}")

    print(f"\nGeodesic midpoint is equidistant: {abs(d_x_hyp - d_y_hyp).item() < 0.01}")
    print(f"Euclidean midpoint is NOT equidistant: {abs(d_x_euc - d_y_euc).item():.4f} difference")

    # Test full model
    print("\n" + "="*60)
    print("Testing HyperbolicOperationModel...")

    model = HyperbolicOperationModel(
        num_trits=9,
        num_values=19683,
        latent_dim=16,
        num_operations=4
    )

    batch_size = 32
    idx_a = torch.randint(0, 19683, (batch_size,))
    idx_b = torch.randint(0, 19683, (batch_size,))
    op_id = torch.randint(0, 4, (batch_size,))
    idx_result = torch.randint(0, 19683, (batch_size,))

    output = model.forward_operation(idx_a, idx_b, op_id, idx_result)

    print(f"\nBatch size: {batch_size}")
    print(f"Predicted embedding shape: {output['predicted_emb'].shape}")
    print(f"Predicted indices shape: {output['predicted_idx'].shape}")
    print(f"Trajectory shape: {output['trajectory'].shape}")

    # Check predictions
    correct = (output['predicted_idx'] == idx_result).float().mean()
    print(f"Accuracy (untrained): {correct.item()*100:.1f}%")

    # Check radii
    emb = model.encode(torch.arange(100))
    radii = emb.norm(dim=-1)
    vals = model.valuations[:100]

    import numpy as np
    from scipy.stats import spearmanr

    vrc = spearmanr(vals.numpy(), radii.detach().numpy())[0]
    print(f"\nValuation-Radius Correlation (untrained): {vrc:.4f}")
    print("(Should be strongly NEGATIVE after training)")

    print("\nHyperbolic operations test complete!")
