"""
loss.py - EES-Based Loss Functions for 3-vae-gemm-v1

Training objective based on Embedding Exactitude Score (EES):
    L = λ₁·ORA_loss + λ₂·UP_loss + λ₃·VRC_loss + λ₄·Recon_loss + λ₅·KL_loss

Where:
    ORA_loss: Operation Reconstruction Accuracy - embed(a⊕b) ≈ midpoint(embed(a),embed(b))
    UP_loss: Ultrametric Preservation - d(a,c) <= max(d(a,b), d(b,c))
    VRC_loss: Valuation-Radius Correlation - higher valuation → smaller radius

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class ORALoss(nn.Module):
    """
    Operation Reconstruction Accuracy Loss.

    Objective: embed(a ⊕ b) ≈ midpoint(embed(a), embed(b))

    We train the operation head to predict the result embedding from
    the operand embeddings and operation type.
    """

    def __init__(
        self,
        margin: float = 0.1,
        use_cosine: bool = False,
        temperature: float = 0.1
    ):
        super().__init__()
        self.margin = margin
        self.use_cosine = use_cosine
        self.temperature = temperature

    def forward(
        self,
        emb_result_pred: torch.Tensor,
        emb_result_target: torch.Tensor,
        all_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            emb_result_pred: [B, D] predicted result embeddings
            emb_result_target: [B, D] target result embeddings (from frozen encoder)
            all_embeddings: [N, D] all value embeddings for ranking loss

        Returns:
            Dictionary with loss components
        """
        # Primary loss: MSE between predicted and target
        mse_loss = F.mse_loss(emb_result_pred, emb_result_target)

        # Cosine similarity loss (optional)
        if self.use_cosine:
            cos_sim = F.cosine_similarity(emb_result_pred, emb_result_target, dim=-1)
            cosine_loss = 1 - cos_sim.mean()
        else:
            cosine_loss = torch.tensor(0.0, device=mse_loss.device)

        # Ranking loss: predicted should be closer to target than to random negatives
        if all_embeddings is not None:
            ranking_loss = self._ranking_loss(emb_result_pred, emb_result_target, all_embeddings)
        else:
            ranking_loss = torch.tensor(0.0, device=mse_loss.device)

        # Combined loss
        total_loss = mse_loss + 0.1 * cosine_loss + 0.5 * ranking_loss

        return {
            'ora_loss': total_loss,
            'ora_mse': mse_loss,
            'ora_cosine': cosine_loss,
            'ora_ranking': ranking_loss,
        }

    def _ranking_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        all_embeddings: torch.Tensor,
        num_negatives: int = 100
    ) -> torch.Tensor:
        """Contrastive ranking loss."""
        B, D = pred.shape
        N = all_embeddings.shape[0]

        # Sample negative indices
        neg_indices = torch.randint(0, N, (B, num_negatives), device=pred.device)
        negatives = all_embeddings[neg_indices]  # [B, num_neg, D]

        # Distance to target
        pos_dist = torch.norm(pred - target, dim=-1)  # [B]

        # Distance to negatives
        neg_dist = torch.norm(pred.unsqueeze(1) - negatives, dim=-1)  # [B, num_neg]

        # Margin ranking loss: pos_dist + margin < neg_dist
        losses = F.relu(pos_dist.unsqueeze(1) + self.margin - neg_dist)
        return losses.mean()


class UltrametricLoss(nn.Module):
    """
    Ultrametric Preservation Loss.

    Objective: For all triplets (a, b, c), enforce d(a,c) <= max(d(a,b), d(b,c))

    This encourages tree-like structure in embedding space.
    """

    def __init__(self, margin: float = 0.01, soft: bool = True):
        super().__init__()
        self.margin = margin
        self.soft = soft

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        emb_c: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            emb_a: [B, D] embeddings of first values
            emb_b: [B, D] embeddings of second values
            emb_c: [B, D] embeddings of third values

        Returns:
            Dictionary with ultrametric loss
        """
        # Compute pairwise distances
        d_ab = torch.norm(emb_a - emb_b, dim=-1)
        d_bc = torch.norm(emb_b - emb_c, dim=-1)
        d_ac = torch.norm(emb_a - emb_c, dim=-1)

        # Ultrametric constraint: d(a,c) <= max(d(a,b), d(b,c))
        max_side = torch.max(d_ab, d_bc)
        violation = F.relu(d_ac - max_side - self.margin)

        if self.soft:
            # Soft violation with log-sum-exp
            loss = torch.log1p(violation).mean()
        else:
            # Hard violation
            loss = violation.mean()

        # Metrics
        satisfaction_rate = (d_ac <= max_side + 1e-6).float().mean()

        return {
            'up_loss': loss,
            'up_satisfaction': satisfaction_rate,
            'up_mean_violation': violation.mean(),
        }


class VRCLoss(nn.Module):
    """
    Valuation-Radius Correlation Loss.

    Objective: Higher 3-adic valuation → smaller embedding radius

    This encourages p-adic hierarchical structure where:
    - v=0 values (most of them) are at the outer shell
    - v=9 (zero) is at the center
    """

    def __init__(
        self,
        target_correlation: float = -0.9,
        radius_scale: float = 1.0
    ):
        super().__init__()
        self.target_correlation = target_correlation
        self.radius_scale = radius_scale

    def forward(
        self,
        embeddings: torch.Tensor,
        valuations: torch.Tensor,
        max_valuation: int = 9
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            embeddings: [B, D] embeddings
            valuations: [B] valuation levels (0 to max_valuation)
            max_valuation: Maximum valuation value

        Returns:
            Dictionary with VRC loss
        """
        # Compute radii
        radii = torch.norm(embeddings, dim=-1)

        # Target radii: higher valuation → smaller radius
        # Linear mapping: v=0 → radius=1, v=max → radius=0.1
        target_radii = self.radius_scale * (1.0 - 0.9 * valuations.float() / max_valuation)

        # MSE loss on radii
        radius_loss = F.mse_loss(radii, target_radii)

        # Differentiable correlation approximation
        # We want radii and valuations to have negative correlation
        radii_centered = radii - radii.mean()
        vals_centered = valuations.float() - valuations.float().mean()

        cov = (radii_centered * vals_centered).mean()
        std_r = radii_centered.std() + 1e-6
        std_v = vals_centered.std() + 1e-6

        correlation = cov / (std_r * std_v)

        # Loss: push correlation toward target
        corr_loss = (correlation - self.target_correlation) ** 2

        # Combined loss
        total_loss = radius_loss + 0.1 * corr_loss

        return {
            'vrc_loss': total_loss,
            'vrc_radius_loss': radius_loss,
            'vrc_correlation': correlation,
            'vrc_mean_radius': radii.mean(),
        }


class ReconstructionLoss(nn.Module):
    """
    Standard VAE reconstruction loss for trit sequences.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        recon_logits: torch.Tensor,
        target_trits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            recon_logits: [B, num_trits, 3] logits for each trit position
            target_trits: [B, num_trits] target trit values in {-1, 0, 1}

        Returns:
            Dictionary with reconstruction loss
        """
        # Convert target trits from {-1, 0, 1} to {0, 1, 2} for cross-entropy
        target_classes = (target_trits + 1).long()

        # Cross-entropy loss
        B, T, C = recon_logits.shape
        loss = F.cross_entropy(
            recon_logits.view(B * T, C),
            target_classes.view(B * T)
        )

        # Accuracy
        pred_classes = recon_logits.argmax(dim=-1)
        accuracy = (pred_classes == target_classes).float().mean()

        return {
            'recon_loss': loss,
            'recon_accuracy': accuracy,
        }


class KLLoss(nn.Module):
    """
    KL divergence loss for VAE.
    """

    def __init__(self, free_bits: float = 0.0):
        super().__init__()
        self.free_bits = free_bits

    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            mu: [B, D] mean of latent distribution
            logvar: [B, D] log variance of latent distribution

        Returns:
            Dictionary with KL loss
        """
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        # Free bits: minimum KL per dimension
        if self.free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)

        kl_loss = kl_per_dim.sum(dim=-1).mean()

        return {
            'kl_loss': kl_loss,
            'kl_per_dim': kl_per_dim.mean(dim=0),
        }


class EESLoss(nn.Module):
    """
    Combined EES-based loss for 3-vae-gemm-v1.

    L = λ_ora * ORA_loss + λ_up * UP_loss + λ_vrc * VRC_loss
      + λ_recon * Recon_loss + λ_kl * KL_loss
    """

    def __init__(
        self,
        ora_weight: float = 1.0,
        up_weight: float = 0.5,
        vrc_weight: float = 0.3,
        recon_weight: float = 0.5,
        kl_weight: float = 0.001,
        ora_margin: float = 0.1,
        up_margin: float = 0.01,
        vrc_target_correlation: float = -0.9,
    ):
        super().__init__()

        self.ora_weight = ora_weight
        self.up_weight = up_weight
        self.vrc_weight = vrc_weight
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight

        self.ora_loss = ORALoss(margin=ora_margin)
        self.up_loss = UltrametricLoss(margin=up_margin)
        self.vrc_loss = VRCLoss(target_correlation=vrc_target_correlation)
        self.recon_loss = ReconstructionLoss()
        self.kl_loss = KLLoss()

    def forward(
        self,
        # Operation prediction
        emb_result_pred: Optional[torch.Tensor] = None,
        emb_result_target: Optional[torch.Tensor] = None,
        all_embeddings: Optional[torch.Tensor] = None,
        # Ultrametric triplets
        emb_triplet_a: Optional[torch.Tensor] = None,
        emb_triplet_b: Optional[torch.Tensor] = None,
        emb_triplet_c: Optional[torch.Tensor] = None,
        # VRC
        embeddings: Optional[torch.Tensor] = None,
        valuations: Optional[torch.Tensor] = None,
        # Reconstruction
        recon_logits: Optional[torch.Tensor] = None,
        target_trits: Optional[torch.Tensor] = None,
        # KL
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined EES loss.

        Returns:
            Dictionary with all loss components and total loss
        """
        total_loss = torch.tensor(0.0, device=self._get_device(
            emb_result_pred, emb_triplet_a, embeddings, mu
        ))
        metrics = {}

        # ORA Loss
        if emb_result_pred is not None and emb_result_target is not None:
            ora_metrics = self.ora_loss(emb_result_pred, emb_result_target, all_embeddings)
            total_loss = total_loss + self.ora_weight * ora_metrics['ora_loss']
            metrics.update(ora_metrics)

        # UP Loss
        if emb_triplet_a is not None:
            up_metrics = self.up_loss(emb_triplet_a, emb_triplet_b, emb_triplet_c)
            total_loss = total_loss + self.up_weight * up_metrics['up_loss']
            metrics.update(up_metrics)

        # VRC Loss
        if embeddings is not None and valuations is not None:
            vrc_metrics = self.vrc_loss(embeddings, valuations)
            total_loss = total_loss + self.vrc_weight * vrc_metrics['vrc_loss']
            metrics.update(vrc_metrics)

        # Reconstruction Loss
        if recon_logits is not None and target_trits is not None:
            recon_metrics = self.recon_loss(recon_logits, target_trits)
            total_loss = total_loss + self.recon_weight * recon_metrics['recon_loss']
            metrics.update(recon_metrics)

        # KL Loss
        if mu is not None and logvar is not None:
            kl_metrics = self.kl_loss(mu, logvar)
            total_loss = total_loss + self.kl_weight * kl_metrics['kl_loss']
            metrics.update(kl_metrics)

        metrics['total_loss'] = total_loss
        return metrics

    def _get_device(self, *tensors):
        """Get device from first non-None tensor."""
        for t in tensors:
            if t is not None:
                return t.device
        return 'cpu'


# =============================================================================
# METRICS FOR MONITORING
# =============================================================================

def compute_ora_accuracy(
    model: nn.Module,
    embeddings: torch.Tensor,
    idx_a: torch.Tensor,
    idx_b: torch.Tensor,
    op_id: torch.Tensor,
    idx_result: torch.Tensor,
    k: int = 10
) -> Dict[str, float]:
    """
    Compute ORA accuracy metrics.

    Returns:
        Dictionary with top-1, top-5, top-10 accuracy
    """
    with torch.no_grad():
        # Get predicted embeddings
        emb_a = embeddings[idx_a]
        emb_b = embeddings[idx_b]

        if hasattr(model, 'operation_head'):
            emb_pred = model.operation_head(emb_a, emb_b, op_id)
        else:
            emb_pred = (emb_a + emb_b) / 2  # Midpoint baseline

        # Find nearest neighbors
        distances = torch.cdist(emb_pred, embeddings)  # [B, N]
        _, nearest = distances.topk(k, largest=False, dim=-1)

        # Check accuracy
        target = idx_result.unsqueeze(-1)
        top1 = (nearest[:, :1] == target).any(dim=-1).float().mean()
        top5 = (nearest[:, :5] == target).any(dim=-1).float().mean()
        top10 = (nearest[:, :k] == target).any(dim=-1).float().mean()

        return {
            'ora_top1': top1.item(),
            'ora_top5': top5.item(),
            'ora_top10': top10.item(),
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Test losses
    B, D = 32, 16

    # Test ORA loss
    ora = ORALoss()
    pred = torch.randn(B, D)
    target = torch.randn(B, D)
    ora_out = ora(pred, target)
    print(f"ORA loss: {ora_out['ora_loss'].item():.4f}")

    # Test UP loss
    up = UltrametricLoss()
    emb_a = torch.randn(B, D)
    emb_b = torch.randn(B, D)
    emb_c = torch.randn(B, D)
    up_out = up(emb_a, emb_b, emb_c)
    print(f"UP loss: {up_out['up_loss'].item():.4f}")
    print(f"UP satisfaction: {up_out['up_satisfaction'].item():.2%}")

    # Test VRC loss
    vrc = VRCLoss()
    embeddings = torch.randn(B, D)
    valuations = torch.randint(0, 10, (B,))
    vrc_out = vrc(embeddings, valuations)
    print(f"VRC loss: {vrc_out['vrc_loss'].item():.4f}")
    print(f"VRC correlation: {vrc_out['vrc_correlation'].item():.4f}")

    # Test combined EES loss
    ees = EESLoss()
    ees_out = ees(
        emb_result_pred=pred,
        emb_result_target=target,
        emb_triplet_a=emb_a,
        emb_triplet_b=emb_b,
        emb_triplet_c=emb_c,
        embeddings=embeddings,
        valuations=valuations,
    )
    print(f"\nTotal EES loss: {ees_out['total_loss'].item():.4f}")

    print("\nLoss test passed!")
