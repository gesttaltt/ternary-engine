"""
model.py - 3-vae-gemm-v1 Architecture

Operation-aware VAE that learns both p-adic structure AND operation semantics.

Key Innovation:
    embed(a ⊕ b) ≈ midpoint(embed(a), embed(b))

Architecture:
    - Encoder: Trit sequence → Latent space (inherited from v5.11)
    - Decoder: Latent space → Trit sequence (inherited from v5.11)
    - OperationHead: (emb_a, emb_b, op_id) → predicted_emb_result

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import numpy as np


@dataclass
class VAEGemmV1Config:
    """Configuration for 3-vae-gemm-v1."""
    # Input/Output
    num_trits: int = 9
    num_values: int = 19683  # 3^9

    # Encoder/Decoder (inherited from v5.11)
    input_dim: int = 9  # Trit sequence length
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    latent_dim: int = 16

    # Operation Head
    op_hidden_dim: int = 64
    num_operations: int = 4  # add, mul, min, max
    op_embedding_dim: int = 8

    # Training
    kl_weight: float = 0.001
    ora_weight: float = 1.0      # Operation Reconstruction Accuracy
    up_weight: float = 0.5       # Ultrametric Preservation
    vrc_weight: float = 0.3      # Valuation-Radius Correlation

    # Checkpoint
    pretrained_path: Optional[str] = None


class Encoder(nn.Module):
    """Encoder: Trit sequence → (mu, logvar)."""

    def __init__(self, config: VAEGemmV1Config):
        super().__init__()
        self.config = config

        # Build encoder layers
        layers = []
        in_dim = config.input_dim
        for h_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
            ])
            in_dim = h_dim

        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(config.hidden_dims[-1], config.latent_dim)
        self.fc_logvar = nn.Linear(config.hidden_dims[-1], config.latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, num_trits] trit values in {-1, 0, 1}

        Returns:
            mu: [B, latent_dim]
            logvar: [B, latent_dim]
        """
        h = self.encoder(x.float())
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """Decoder: Latent → Trit sequence."""

    def __init__(self, config: VAEGemmV1Config):
        super().__init__()
        self.config = config

        # Build decoder layers (reverse of encoder)
        layers = []
        in_dim = config.latent_dim
        for h_dim in reversed(config.hidden_dims):
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
            ])
            in_dim = h_dim

        self.decoder = nn.Sequential(*layers)
        # Output logits for 3 classes per trit
        self.fc_out = nn.Linear(config.hidden_dims[0], config.input_dim * 3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, latent_dim]

        Returns:
            logits: [B, num_trits, 3] logits for each trit value
        """
        h = self.decoder(z)
        logits = self.fc_out(h)
        return logits.view(-1, self.config.input_dim, 3)


class OperationHead(nn.Module):
    """
    Operation Head: (emb_a, emb_b, op_id) → predicted_emb_result

    This is the key innovation: learns how operations transform embeddings.
    """

    def __init__(self, config: VAEGemmV1Config):
        super().__init__()
        self.config = config

        # Operation embedding
        self.op_embedding = nn.Embedding(config.num_operations, config.op_embedding_dim)

        # MLP: concat(emb_a, emb_b, op_emb) → predicted_emb_result
        input_dim = config.latent_dim * 2 + config.op_embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, config.op_hidden_dim),
            nn.LayerNorm(config.op_hidden_dim),
            nn.GELU(),
            nn.Linear(config.op_hidden_dim, config.op_hidden_dim),
            nn.LayerNorm(config.op_hidden_dim),
            nn.GELU(),
            nn.Linear(config.op_hidden_dim, config.latent_dim),
        )

        # Residual connection: midpoint + learned_delta
        self.use_residual = True

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        op_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            emb_a: [B, latent_dim] embedding of first operand
            emb_b: [B, latent_dim] embedding of second operand
            op_id: [B] operation identifiers

        Returns:
            predicted_emb: [B, latent_dim] predicted embedding of result
        """
        # Get operation embedding
        op_emb = self.op_embedding(op_id)  # [B, op_embedding_dim]

        # Concatenate inputs
        x = torch.cat([emb_a, emb_b, op_emb], dim=-1)

        # Predict delta from midpoint
        delta = self.mlp(x)

        if self.use_residual:
            # Residual: midpoint + learned_delta
            midpoint = (emb_a + emb_b) / 2
            predicted = midpoint + delta
        else:
            predicted = delta

        return predicted


class VAEGemmV1(nn.Module):
    """
    3-vae-gemm-v1: Operation-Aware Ternary VAE

    Combines:
    - Standard VAE (encoder/decoder) for reconstruction
    - Operation head for operation-aware embeddings
    - Valuation-aware loss for p-adic structure
    """

    def __init__(self, config: VAEGemmV1Config):
        super().__init__()
        self.config = config

        # Core VAE components
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        # Operation head
        self.operation_head = OperationHead(config)

        # Precompute index → trit conversion
        self._build_trit_table()

        # Precompute valuations
        self._build_valuation_table()

    def _build_trit_table(self):
        """Precompute index → trit conversion table."""
        table = torch.zeros(self.config.num_values, self.config.num_trits, dtype=torch.float32)
        for idx in range(self.config.num_values):
            n = idx
            for i in range(self.config.num_trits):
                table[idx, i] = (n % 3) - 1
                n //= 3
        self.register_buffer('trit_table', table)

    def _build_valuation_table(self):
        """Precompute valuation table."""
        # `idx` here is the RAW encoding index (idx = sum((trit_k+1)*3^k)),
        # not the decoded balanced-ternary value -- the true zero (all-zero
        # trits) lives at idx_offset = (num_values-1)//2, not at idx=0
        # (which decodes to the most negative representable value). This
        # used to special-case idx==0 as the max-valuation "zero" case and
        # compute v3(idx) directly on the raw index otherwise, inverting
        # which point was treated as p-adically near zero -- same bug class
        # already found and fixed in research/scripts/falsify.py's
        # build_corpus(). Found 2026-08-13.
        idx_offset = (self.config.num_values - 1) // 2
        valuations = torch.zeros(self.config.num_values, dtype=torch.long)
        for idx in range(self.config.num_values):
            n = idx - idx_offset
            if n == 0:
                valuations[idx] = self.config.num_trits
            else:
                v = 0
                while n % 3 == 0 and v < self.config.num_trits:
                    n //= 3
                    v += 1
                valuations[idx] = v
        self.register_buffer('valuations', valuations)

    def index_to_trits(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert indices to trit sequences using precomputed table."""
        return self.trit_table[indices]

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode indices to latent distribution."""
        trits = self.index_to_trits(indices)
        return self.encoder(trits)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to trit logits."""
        return self.decoder(z)

    def get_embeddings(self, indices: torch.Tensor) -> torch.Tensor:
        """Get embeddings (mu) for given indices."""
        mu, _ = self.encode(indices)
        return mu

    def forward(
        self,
        indices: torch.Tensor,
        compute_recon: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for reconstruction.

        Args:
            indices: [B] value indices
            compute_recon: Whether to compute reconstruction

        Returns:
            Dictionary with mu, logvar, z, and optionally recon_logits
        """
        mu, logvar = self.encode(indices)
        z = self.reparameterize(mu, logvar)

        output = {
            'mu': mu,
            'logvar': logvar,
            'z': z,
        }

        if compute_recon:
            output['recon_logits'] = self.decode(z)
            output['target_trits'] = self.index_to_trits(indices)

        return output

    def forward_operation(
        self,
        idx_a: torch.Tensor,
        idx_b: torch.Tensor,
        op_id: torch.Tensor,
        idx_result: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for operation prediction.

        Args:
            idx_a: [B] first operand indices
            idx_b: [B] second operand indices
            op_id: [B] operation identifiers
            idx_result: [B] correct result indices

        Returns:
            Dictionary with embeddings and predictions
        """
        # Get embeddings for operands
        emb_a = self.get_embeddings(idx_a)
        emb_b = self.get_embeddings(idx_b)
        emb_result_target = self.get_embeddings(idx_result)

        # Predict result embedding
        emb_result_pred = self.operation_head(emb_a, emb_b, op_id)

        # Midpoint for comparison
        midpoint = (emb_a + emb_b) / 2

        return {
            'emb_a': emb_a,
            'emb_b': emb_b,
            'emb_result_target': emb_result_target,
            'emb_result_pred': emb_result_pred,
            'midpoint': midpoint,
            'op_id': op_id,
        }

    def get_all_embeddings(self) -> torch.Tensor:
        """Get embeddings for all values."""
        indices = torch.arange(self.config.num_values, device=self.trit_table.device)
        with torch.no_grad():
            embeddings = self.get_embeddings(indices)
        return embeddings

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        config: Optional[VAEGemmV1Config] = None
    ) -> 'VAEGemmV1':
        """
        Load model from pretrained checkpoint.

        Handles loading from v5.11 format and initializing new operation head.
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Determine config
        if config is None:
            config = VAEGemmV1Config()

        # Create model
        model = cls(config)

        # Load pretrained weights (may be partial)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Map v5.11 keys to our keys if needed
        key_mapping = {
            # Add any key mappings needed for v5.11 compatibility
        }

        # Load matching keys
        model_state = model.state_dict()
        loaded_keys = []
        for key, value in state_dict.items():
            mapped_key = key_mapping.get(key, key)
            if mapped_key in model_state and model_state[mapped_key].shape == value.shape:
                model_state[mapped_key] = value
                loaded_keys.append(mapped_key)

        model.load_state_dict(model_state, strict=False)
        print(f"Loaded {len(loaded_keys)} pretrained weights from {checkpoint_path}")

        return model

    def save(self, path: str, optimizer=None, epoch: int = 0, metrics: Dict = None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'epoch': epoch,
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if metrics is not None:
            checkpoint['metrics'] = metrics
        torch.save(checkpoint, path)


# =============================================================================
# EMBEDDING LOOKUP TABLE
# =============================================================================

class EmbeddingLUT(nn.Module):
    """
    Learnable embedding lookup table for all ternary values.

    Alternative to encoder - directly learns embeddings.
    Can be initialized from pretrained VAE embeddings.
    """

    def __init__(
        self,
        num_values: int = 19683,
        embedding_dim: int = 16,
        init_embeddings: Optional[torch.Tensor] = None
    ):
        super().__init__()

        self.num_values = num_values
        self.embedding_dim = embedding_dim

        # Learnable embeddings
        self.embeddings = nn.Parameter(torch.randn(num_values, embedding_dim) * 0.1)

        if init_embeddings is not None:
            with torch.no_grad():
                self.embeddings.copy_(init_embeddings)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Get embeddings for given indices."""
        return self.embeddings[indices]

    def get_all(self) -> torch.Tensor:
        """Get all embeddings."""
        return self.embeddings

    @classmethod
    def from_numpy(cls, embeddings: np.ndarray) -> 'EmbeddingLUT':
        """Initialize from numpy embeddings."""
        num_values, embedding_dim = embeddings.shape
        lut = cls(num_values, embedding_dim)
        with torch.no_grad():
            lut.embeddings.copy_(torch.from_numpy(embeddings))
        return lut


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Test model
    config = VAEGemmV1Config()
    model = VAEGemmV1(config)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 32
    indices = torch.randint(0, config.num_values, (batch_size,))

    output = model(indices)
    print(f"mu shape: {output['mu'].shape}")
    print(f"recon_logits shape: {output['recon_logits'].shape}")

    # Test operation forward
    idx_a = torch.randint(0, config.num_values, (batch_size,))
    idx_b = torch.randint(0, config.num_values, (batch_size,))
    op_id = torch.randint(0, 4, (batch_size,))
    idx_result = torch.randint(0, config.num_values, (batch_size,))

    op_output = model.forward_operation(idx_a, idx_b, op_id, idx_result)
    print(f"emb_result_pred shape: {op_output['emb_result_pred'].shape}")

    print("\nModel test passed!")
