#!/usr/bin/env python3
"""
Create Pre-computed Embedding LUT for Ternary Engine

Pre-computes all 19,683 embeddings from the VAE encoder_B (best hierarchy)
and exports as a binary lookup table for O(1) access.

Use cases:
1. Geometric nearest-neighbor search in ternary space
2. Hierarchical clustering respecting p-adic valuation
3. Approximate ternary operations via embedding arithmetic
4. Similarity search for ternary values

Output: ternary_embeddings.bin (1.26 MB)
  - Header: magic(4) + version(4) + num_entries(4) + embed_dim(4)
  - Data: [19683, 16] float32 embeddings

Copyright 2025 Ternary Engine Contributors
"""

import struct
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr


class Encoder(nn.Module):
    """Encoder matching checkpoint structure."""
    def __init__(self, state_dict, prefix):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(9, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, 16)

        # Load weights
        self.encoder[0].weight.data = state_dict[f'{prefix}.encoder.0.weight']
        self.encoder[0].bias.data = state_dict[f'{prefix}.encoder.0.bias']
        self.encoder[2].weight.data = state_dict[f'{prefix}.encoder.2.weight']
        self.encoder[2].bias.data = state_dict[f'{prefix}.encoder.2.bias']
        self.encoder[4].weight.data = state_dict[f'{prefix}.encoder.4.weight']
        self.encoder[4].bias.data = state_dict[f'{prefix}.encoder.4.bias']
        self.fc_mu.weight.data = state_dict[f'{prefix}.fc_mu.weight']
        self.fc_mu.bias.data = state_dict[f'{prefix}.fc_mu.bias']

    def forward(self, x):
        return self.fc_mu(self.encoder(x))


class HyperbolicProjection(nn.Module):
    """Projection head for hyperbolic embeddings."""
    def __init__(self, state_dict, prefix):
        super().__init__()
        self.direction_net = nn.Sequential(
            nn.Linear(16, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16)
        )
        self.radius_net = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

        # Load weights
        self.direction_net[0].weight.data = state_dict[f'{prefix}.direction_net.0.weight']
        self.direction_net[0].bias.data = state_dict[f'{prefix}.direction_net.0.bias']
        self.direction_net[1].weight.data = state_dict[f'{prefix}.direction_net.1.weight']
        self.direction_net[1].bias.data = state_dict[f'{prefix}.direction_net.1.bias']
        self.direction_net[4].weight.data = state_dict[f'{prefix}.direction_net.4.weight']
        self.direction_net[4].bias.data = state_dict[f'{prefix}.direction_net.4.bias']
        self.radius_net[0].weight.data = state_dict[f'{prefix}.radius_net.0.weight']
        self.radius_net[0].bias.data = state_dict[f'{prefix}.radius_net.0.bias']
        self.radius_net[3].weight.data = state_dict[f'{prefix}.radius_net.3.weight']
        self.radius_net[3].bias.data = state_dict[f'{prefix}.radius_net.3.bias']

    def forward(self, z):
        direction = torch.nn.functional.normalize(self.direction_net(z), dim=-1)
        radius = torch.sigmoid(self.radius_net(z))
        return direction * radius


def compute_valuation(n: int) -> int:
    """Compute 3-adic valuation of integer n."""
    if n == 0:
        return 10
    v = 0
    while n % 3 == 0:
        n //= 3
        v += 1
    return min(v, 9)


def main():
    base_dir = Path(__file__).parent
    # CLAUDE.md documents this checkpoint's path as
    # v5_11_homeostasis/best.pt, not epoch_20.pt -- prefer best.pt (the
    # documented source of truth) and fall back to epoch_20.pt so this
    # still works if the checkpoint was only saved under its per-epoch
    # name. Same mismatch existed identically in export_vae_to_gemm.py and
    # test_vae_gemm.py; fixed in all three. Found 2026-08-13.
    checkpoint_dir = base_dir / "v5_11_homeostasis"
    checkpoint_path = checkpoint_dir / "best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = checkpoint_dir / "epoch_20.pt"
    output_dir = base_dir / "gemm_export"

    print("="*70)
    print("Creating Ternary Embedding LUT")
    print("="*70)

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path.name}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state']

    # Create encoder B (best hierarchy) and projection
    encoder_B = Encoder(state_dict, 'encoder_B')
    projection_B = HyperbolicProjection(state_dict, 'projection.proj_B')
    encoder_B.eval()
    projection_B.eval()

    # Generate all 19683 ternary inputs
    N = 19683
    print(f"\nGenerating embeddings for {N} ternary values...")

    all_inputs = []
    valuations = []
    for i in range(N):
        trits = []
        val = i
        for _ in range(9):
            trits.append((val % 3) - 1)
            val //= 3
        all_inputs.append(trits)
        valuations.append(compute_valuation(i))

    inputs = torch.tensor(all_inputs, dtype=torch.float32)
    valuations = np.array(valuations)

    # Compute embeddings
    with torch.no_grad():
        mu = encoder_B(inputs)  # [N, 16] latent
        embeddings = projection_B(mu)  # [N, 16] hyperbolic

    embeddings_np = embeddings.numpy()
    mu_np = mu.numpy()

    # Compute radii and verify hierarchy
    radii = np.linalg.norm(embeddings_np, axis=1)
    hierarchy, _ = spearmanr(valuations, radii)

    print(f"\nEmbedding statistics:")
    print(f"  Shape: {embeddings_np.shape}")
    print(f"  Hierarchy (Spearman): {hierarchy:.4f}")
    print(f"  Mean radius: {radii.mean():.4f}")
    print(f"  Radius range: [{radii.min():.4f}, {radii.max():.4f}]")

    # Radius by valuation
    print(f"\n  Radius by valuation:")
    for v in range(10):
        mask = valuations == v
        if mask.sum() > 0:
            print(f"    v={v}: r={radii[mask].mean():.4f} (n={mask.sum()})")

    # Save binary LUT
    lut_path = output_dir / "ternary_embeddings.bin"
    print(f"\nSaving to: {lut_path}")

    with open(lut_path, 'wb') as f:
        # Header
        f.write(struct.pack('<4s', b'TLUT'))  # Magic
        f.write(struct.pack('<I', 1))          # Version
        f.write(struct.pack('<I', N))          # Num entries
        f.write(struct.pack('<I', 16))         # Embed dim

        # Embeddings [N, 16] as float32
        f.write(embeddings_np.astype(np.float32).tobytes())

    # Also save latent (mu) for encoder-only use
    mu_path = output_dir / "ternary_latents.bin"
    with open(mu_path, 'wb') as f:
        f.write(struct.pack('<4s', b'TLAT'))
        f.write(struct.pack('<I', 1))
        f.write(struct.pack('<I', N))
        f.write(struct.pack('<I', 16))
        f.write(mu_np.astype(np.float32).tobytes())

    # Save as NumPy for easy Python access
    np.save(output_dir / "ternary_embeddings.npy", embeddings_np)
    np.save(output_dir / "ternary_latents.npy", mu_np)

    print(f"\nOutput files:")
    print(f"  {lut_path.name}: {lut_path.stat().st_size:,} bytes (hyperbolic)")
    print(f"  {mu_path.name}: {mu_path.stat().st_size:,} bytes (latent)")
    print(f"  ternary_embeddings.npy: {(output_dir / 'ternary_embeddings.npy').stat().st_size:,} bytes")

    # Quick demo of use cases
    print("\n" + "="*70)
    print("DEMO: Use Cases")
    print("="*70)

    # 1. Nearest neighbor search
    print("\n1. Nearest Neighbor Search")
    query_idx = 12345
    query = embeddings_np[query_idx]
    distances = np.linalg.norm(embeddings_np - query, axis=1)
    nearest = np.argsort(distances)[1:6]  # Top 5 (excluding self)
    print(f"   Query: {query_idx} (valuation={valuations[query_idx]})")
    print(f"   Nearest neighbors:")
    for nn_idx in nearest:
        print(f"     {nn_idx}: dist={distances[nn_idx]:.4f}, v={valuations[nn_idx]}")

    # 2. Hierarchical clustering preview
    print("\n2. Valuation-based Clustering")
    for v in [0, 5, 9]:
        mask = valuations == v
        if mask.sum() > 0:
            cluster_radii = radii[mask]
            print(f"   v={v}: {mask.sum()} items, mean_r={cluster_radii.mean():.4f}")

    # 3. Approximate arithmetic via embedding interpolation
    print("\n3. Embedding Arithmetic Example")
    a, b = 100, 200
    embed_a = embeddings_np[a]
    embed_b = embeddings_np[b]
    embed_sum = (embed_a + embed_b) / 2  # Simple average

    # Find nearest to interpolated
    dist_to_sum = np.linalg.norm(embeddings_np - embed_sum, axis=1)
    approx_result = np.argmin(dist_to_sum)
    print(f"   midpoint({a}, {b}) ≈ {approx_result}")

    print("\n" + "="*70)
    print("LUT Creation Complete")
    print("="*70)


if __name__ == "__main__":
    main()
