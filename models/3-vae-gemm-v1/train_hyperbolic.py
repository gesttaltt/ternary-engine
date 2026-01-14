"""
train_hyperbolic.py - Training for Hyperbolic Operation Model

This trains operations via NATURAL FLOWS in non-Euclidean space.
NOT classification - but attractor basin dynamics.

Key differences from Euclidean training:
1. Loss is hyperbolic distance, not Euclidean MSE
2. Operations create trajectories that flow to attractors
3. Ultrametric constraints (strong triangle inequality)
4. Radial structure: valuation → radius mapping

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0
"""

import sys
# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import time
import json
from typing import Dict, Optional
import argparse

from hyperbolic_ops import (
    HyperbolicOperationModel,
    HyperbolicOperationLoss,
    PoincareOperations
)
from data import create_operation_luts, OperationLUTDataset


def compute_metrics(
    model: HyperbolicOperationModel,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 50
) -> Dict[str, float]:
    """Compute evaluation metrics on validation set."""
    model.eval()

    total_correct = 0
    total_samples = 0
    total_hyp_dist = 0.0
    trajectory_lengths = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            idx_a = batch['idx_a'].to(device)
            idx_b = batch['idx_b'].to(device)
            op_id = batch['op_id'].to(device)
            idx_result = batch['idx_result'].to(device)

            output = model.forward_operation(idx_a, idx_b, op_id, idx_result)

            # Accuracy: predicted index matches target
            predicted_idx = output['predicted_idx']
            correct = (predicted_idx == idx_result).sum().item()
            total_correct += correct
            total_samples += len(idx_result)

            # Hyperbolic distance to target attractor
            hyp_dist = PoincareOperations.hyperbolic_distance(
                output['predicted_emb'],
                output['target_attractor'],
                model.c
            ).mean().item()
            total_hyp_dist += hyp_dist

            # Trajectory length
            trajectory = output['trajectory']
            traj_len = 0.0
            for t in range(trajectory.shape[1] - 1):
                step_dist = PoincareOperations.hyperbolic_distance(
                    trajectory[:, t],
                    trajectory[:, t+1],
                    model.c
                ).mean().item()
                traj_len += step_dist
            trajectory_lengths.append(traj_len)

    metrics = {
        'accuracy': total_correct / total_samples,
        'mean_hyp_dist': total_hyp_dist / min(batch_idx + 1, max_batches),
        'mean_traj_length': np.mean(trajectory_lengths),
    }

    return metrics


def compute_radial_structure(
    model: HyperbolicOperationModel,
    device: torch.device
) -> Dict[str, float]:
    """Compute valuation-radius correlation."""
    model.eval()

    # Sample across valuation levels
    indices_by_val = {v: [] for v in range(10)}
    for idx in range(model.num_values):
        val = model.valuations[idx].item()
        if val <= 9:
            indices_by_val[val].append(idx)

    radii_by_val = {}
    with torch.no_grad():
        for val, indices in indices_by_val.items():
            if len(indices) > 0:
                sample_idx = torch.tensor(indices[:100], device=device)
                emb = model.encode(sample_idx)
                radii = emb.norm(dim=-1)
                radii_by_val[val] = radii.mean().item()

    # Compute correlation
    vals = list(radii_by_val.keys())
    radii = [radii_by_val[v] for v in vals]

    from scipy.stats import spearmanr
    if len(vals) >= 2:
        vrc, _ = spearmanr(vals, radii)
    else:
        vrc = 0.0

    return {
        'vrc': vrc,
        'radii_by_valuation': radii_by_val
    }


def train_epoch(
    model: HyperbolicOperationModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: HyperbolicOperationLoss,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_traj_loss = 0.0
    total_smooth_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        idx_a = batch['idx_a'].to(device)
        idx_b = batch['idx_b'].to(device)
        op_id = batch['op_id'].to(device)
        idx_result = batch['idx_result'].to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model.forward_operation(idx_a, idx_b, op_id, idx_result)

        # Compute losses
        valuations = model.valuations[idx_result]
        losses = loss_fn(output, valuations)

        # Backprop
        losses['total'].backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += losses['total'].item()
        if 'trajectory' in losses:
            total_traj_loss += losses['trajectory'].item()
        if 'smoothness' in losses:
            total_smooth_loss += losses['smoothness'].item()
        num_batches += 1

        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)} | Loss: {losses['total'].item():.4f}")

    return {
        'total_loss': total_loss / num_batches,
        'trajectory_loss': total_traj_loss / num_batches,
        'smoothness_loss': total_smooth_loss / num_batches,
    }


def main():
    parser = argparse.ArgumentParser(description='Train Hyperbolic Operation Model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--lut-samples', type=int, default=200000, help='LUT samples')
    parser.add_argument('--latent-dim', type=int, default=16, help='Latent dimension')
    parser.add_argument('--curvature', type=float, default=1.0, help='Hyperbolic curvature')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path(__file__).parent / "checkpoints_hyperbolic"
    output_dir.mkdir(exist_ok=True)

    # Create/load LUTs
    lut_dir = Path(__file__).parent / "luts"
    if lut_dir.exists() and list(lut_dir.glob("lut_*.npz")):
        print("Loading existing LUTs...")
        from data import load_operation_luts
        luts = load_operation_luts(lut_dir)
    else:
        print(f"Generating LUTs ({args.lut_samples:,} samples)...")
        luts = create_operation_luts(
            sample_size=args.lut_samples,
            output_dir=lut_dir
        )

    # Create dataset
    print("Creating datasets...")
    full_dataset = OperationLUTDataset(luts)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Val samples: {len(val_dataset):,}")

    # Create model
    print("\nCreating HyperbolicOperationModel...")
    model = HyperbolicOperationModel(
        num_trits=9,
        num_values=19683,
        latent_dim=args.latent_dim,
        num_operations=4,
        c=args.curvature
    )
    model = model.to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    loss_fn = HyperbolicOperationLoss(c=args.curvature)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training history
    history = {
        'train_loss': [],
        'val_accuracy': [],
        'val_hyp_dist': [],
        'vrc': [],
    }

    best_accuracy = 0.0

    print("\n" + "="*60)
    print("HYPERBOLIC TRAINING")
    print("="*60)
    print(f"Training with GEODESIC flows, not Euclidean heuristics")
    print(f"Operations create TRAJECTORIES that flow to ATTRACTOR BASINS")
    print("="*60 + "\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch
        )

        # Validate
        val_metrics = compute_metrics(model, val_loader, device)

        # Radial structure
        radial = compute_radial_structure(model, device)

        # Update scheduler
        scheduler.step()

        # Log
        elapsed = time.time() - t0
        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.1f}s)")
        print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
        print(f"  Val Accuracy: {val_metrics['accuracy']*100:.2f}%")
        print(f"  Val Hyp Dist: {val_metrics['mean_hyp_dist']:.4f}")
        print(f"  VRC: {radial['vrc']:.4f}")
        print(f"  Trajectory Length: {val_metrics['mean_traj_length']:.4f}")

        # Save history
        history['train_loss'].append(train_metrics['total_loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_hyp_dist'].append(val_metrics['mean_hyp_dist'])
        history['vrc'].append(radial['vrc'])

        # Save best model
        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            print(f"  [NEW BEST] Accuracy: {best_accuracy*100:.2f}%")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
                'vrc': radial['vrc'],
                'radii_by_valuation': radial['radii_by_valuation'],
            }, output_dir / "best_model.pt")

    # Save final history
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Accuracy: {best_accuracy*100:.2f}%")
    print(f"Final VRC: {radial['vrc']:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
