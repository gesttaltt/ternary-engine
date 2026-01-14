#!/usr/bin/env python3
"""
train.py - Training Pipeline for 3-vae-gemm-v1

Trains operation-aware VAE with EES-based objective:
    L = λ₁·ORA_loss + λ₂·UP_loss + λ₃·VRC_loss + λ₄·Recon_loss + λ₅·KL_loss

Training targets:
    - EES >= 0.95
    - ORA >= 95% (operation reconstruction accuracy)
    - Natural ultrametric emergence (UP > 90%)

USAGE:
    python train.py --epochs 100 --lr 1e-4 --checkpoint path/to/v5.11/checkpoint.pt

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0
"""

import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import sys

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model import VAEGemmV1, VAEGemmV1Config, EmbeddingLUT
from loss import EESLoss, compute_ora_accuracy
from data import (
    OperationLUTDataset,
    TripletDataset,
    create_operation_luts,
    load_operation_luts
)


# =============================================================================
# TRAINING LOOP
# =============================================================================

class Trainer:
    """Training manager for 3-vae-gemm-v1."""

    def __init__(
        self,
        model: nn.Module,
        config: VAEGemmV1Config,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: Optional[Path] = None
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = output_dir or Path(__file__).parent / 'checkpoints'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Loss function
        self.criterion = EESLoss(
            ora_weight=config.ora_weight,
            up_weight=config.up_weight,
            vrc_weight=config.vrc_weight,
        )

        # Optimizer
        self.optimizer = None
        self.scheduler = None

        # Metrics history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'ora_top1': [],
            'ora_top10': [],
            'up_satisfaction': [],
            'vrc_correlation': [],
            'ees_score': [],
        }

        # Best model tracking
        self.best_ees = 0.0
        self.best_epoch = 0

        # Frozen embeddings for operation training
        self.frozen_embeddings = None

    def setup_optimizer(self, lr: float = 1e-4, weight_decay: float = 1e-5):
        """Setup optimizer and scheduler."""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )

    def freeze_encoder(self):
        """Freeze encoder weights (only train operation head)."""
        for name, param in self.model.named_parameters():
            if 'operation_head' not in name:
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Frozen encoder: {trainable:,}/{total:,} trainable params")

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True

    def compute_frozen_embeddings(self):
        """Compute and cache frozen embeddings from encoder."""
        self.model.eval()
        with torch.no_grad():
            all_indices = torch.arange(self.config.num_values, device=self.device)
            self.frozen_embeddings = self.model.get_embeddings(all_indices)
        print(f"Cached frozen embeddings: {self.frozen_embeddings.shape}")

    def train_epoch(
        self,
        op_loader: DataLoader,
        triplet_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()

        total_loss = 0
        total_ora = 0
        total_up = 0
        total_vrc = 0
        num_batches = 0

        # Ensure we have frozen embeddings
        if self.frozen_embeddings is None:
            self.compute_frozen_embeddings()

        # Zip operation and triplet loaders
        op_iter = iter(op_loader)
        triplet_iter = iter(triplet_loader)

        for batch_idx in range(min(len(op_loader), len(triplet_loader))):
            try:
                op_batch = next(op_iter)
                triplet_batch = next(triplet_iter)
            except StopIteration:
                break

            # Move to device
            idx_a = op_batch['idx_a'].to(self.device)
            idx_b = op_batch['idx_b'].to(self.device)
            op_id = op_batch['op_id'].to(self.device)
            idx_result = op_batch['idx_result'].to(self.device)

            triplet_a = triplet_batch['idx_a'].to(self.device)
            triplet_b = triplet_batch['idx_b'].to(self.device)
            triplet_c = triplet_batch['idx_c'].to(self.device)
            triplet_vals = triplet_batch['val_a'].to(self.device)

            # Forward pass for operations
            op_output = self.model.forward_operation(idx_a, idx_b, op_id, idx_result)

            # Get triplet embeddings
            emb_trip_a = self.model.get_embeddings(triplet_a)
            emb_trip_b = self.model.get_embeddings(triplet_b)
            emb_trip_c = self.model.get_embeddings(triplet_c)

            # Compute loss
            loss_dict = self.criterion(
                emb_result_pred=op_output['emb_result_pred'],
                emb_result_target=op_output['emb_result_target'].detach(),  # Frozen target
                all_embeddings=self.frozen_embeddings,
                emb_triplet_a=emb_trip_a,
                emb_triplet_b=emb_trip_b,
                emb_triplet_c=emb_trip_c,
                embeddings=emb_trip_a,
                valuations=triplet_vals,
            )

            loss = loss_dict['total_loss']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            if 'ora_loss' in loss_dict:
                total_ora += loss_dict['ora_loss'].item()
            if 'up_satisfaction' in loss_dict:
                total_up += loss_dict['up_satisfaction'].item()
            if 'vrc_correlation' in loss_dict:
                total_vrc += loss_dict['vrc_correlation'].item()
            num_batches += 1

            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(op_loader)} | Loss: {loss.item():.4f}")

        # Step scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        return {
            'loss': total_loss / max(num_batches, 1),
            'ora_loss': total_ora / max(num_batches, 1),
            'up_satisfaction': total_up / max(num_batches, 1),
            'vrc_correlation': total_vrc / max(num_batches, 1),
        }

    @torch.no_grad()
    def validate(
        self,
        op_loader: DataLoader,
        triplet_loader: DataLoader
    ) -> Dict[str, float]:
        """Validation pass."""
        self.model.eval()

        total_loss = 0
        ora_top1_total = 0
        ora_top10_total = 0
        up_sat_total = 0
        vrc_corr_total = 0
        num_batches = 0

        # Get current embeddings
        all_embeddings = self.model.get_all_embeddings()

        op_iter = iter(op_loader)
        triplet_iter = iter(triplet_loader)

        for batch_idx in range(min(len(op_loader), len(triplet_loader))):
            try:
                op_batch = next(op_iter)
                triplet_batch = next(triplet_iter)
            except StopIteration:
                break

            # Move to device
            idx_a = op_batch['idx_a'].to(self.device)
            idx_b = op_batch['idx_b'].to(self.device)
            op_id = op_batch['op_id'].to(self.device)
            idx_result = op_batch['idx_result'].to(self.device)

            triplet_a = triplet_batch['idx_a'].to(self.device)
            triplet_b = triplet_batch['idx_b'].to(self.device)
            triplet_c = triplet_batch['idx_c'].to(self.device)
            triplet_vals = triplet_batch['val_a'].to(self.device)

            # Forward pass
            op_output = self.model.forward_operation(idx_a, idx_b, op_id, idx_result)

            # Get triplet embeddings
            emb_trip_a = self.model.get_embeddings(triplet_a)
            emb_trip_b = self.model.get_embeddings(triplet_b)
            emb_trip_c = self.model.get_embeddings(triplet_c)

            # Compute loss
            loss_dict = self.criterion(
                emb_result_pred=op_output['emb_result_pred'],
                emb_result_target=op_output['emb_result_target'],
                emb_triplet_a=emb_trip_a,
                emb_triplet_b=emb_trip_b,
                emb_triplet_c=emb_trip_c,
                embeddings=emb_trip_a,
                valuations=triplet_vals,
            )

            # Compute ORA accuracy
            ora_metrics = compute_ora_accuracy(
                self.model, all_embeddings,
                idx_a, idx_b, op_id, idx_result
            )

            total_loss += loss_dict['total_loss'].item()
            ora_top1_total += ora_metrics['ora_top1']
            ora_top10_total += ora_metrics['ora_top10']
            if 'up_satisfaction' in loss_dict:
                up_sat_total += loss_dict['up_satisfaction'].item()
            if 'vrc_correlation' in loss_dict:
                vrc_corr_total += loss_dict['vrc_correlation'].item()
            num_batches += 1

        return {
            'loss': total_loss / max(num_batches, 1),
            'ora_top1': ora_top1_total / max(num_batches, 1),
            'ora_top10': ora_top10_total / max(num_batches, 1),
            'up_satisfaction': up_sat_total / max(num_batches, 1),
            'vrc_correlation': vrc_corr_total / max(num_batches, 1),
        }

    def compute_ees(self, val_metrics: Dict[str, float]) -> float:
        """Compute EES from validation metrics."""
        # Weights matching embedding_exactitude_score.py
        ora = val_metrics.get('ora_top10', 0)
        up = val_metrics.get('up_satisfaction', 0)
        vrc = abs(val_metrics.get('vrc_correlation', 0))

        # Simplified EES (matching the 3 main components)
        ees = 0.4 * ora + 0.3 * up + 0.3 * vrc
        return ees

    def train(
        self,
        train_op_loader: DataLoader,
        train_triplet_loader: DataLoader,
        val_op_loader: DataLoader,
        val_triplet_loader: DataLoader,
        epochs: int = 100,
        lr: float = 1e-4,
        freeze_encoder_epochs: int = 10,
        early_stop_patience: int = 20,
        target_ees: float = 0.95
    ):
        """Full training loop."""
        print("=" * 70)
        print("3-VAE-GEMM-V1 TRAINING")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {lr}")
        print(f"Target EES: {target_ees}")
        print(f"Freeze encoder epochs: {freeze_encoder_epochs}")
        print()

        self.setup_optimizer(lr)

        # Phase 1: Train only operation head with frozen encoder
        if freeze_encoder_epochs > 0:
            print("Phase 1: Training operation head only (frozen encoder)")
            self.freeze_encoder()
            self.compute_frozen_embeddings()

        no_improve_count = 0

        for epoch in range(epochs):
            t0 = time.time()

            # Unfreeze after initial phase
            if epoch == freeze_encoder_epochs and freeze_encoder_epochs > 0:
                print("\nPhase 2: Fine-tuning all parameters")
                self.unfreeze_all()
                self.setup_optimizer(lr * 0.1)  # Lower LR for fine-tuning

            # Train
            train_metrics = self.train_epoch(
                train_op_loader, train_triplet_loader, epoch
            )

            # Validate
            val_metrics = self.validate(val_op_loader, val_triplet_loader)

            # Compute EES
            ees = self.compute_ees(val_metrics)

            # Log
            epoch_time = time.time() - t0
            print(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  ORA Top-1: {val_metrics['ora_top1']:.2%}")
            print(f"  ORA Top-10: {val_metrics['ora_top10']:.2%}")
            print(f"  UP Satisfaction: {val_metrics['up_satisfaction']:.2%}")
            print(f"  VRC Correlation: {val_metrics['vrc_correlation']:.4f}")
            print(f"  EES: {ees:.4f}")

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['ora_top1'].append(val_metrics['ora_top1'])
            self.history['ora_top10'].append(val_metrics['ora_top10'])
            self.history['up_satisfaction'].append(val_metrics['up_satisfaction'])
            self.history['vrc_correlation'].append(val_metrics['vrc_correlation'])
            self.history['ees_score'].append(ees)

            # Save best model
            if ees > self.best_ees:
                self.best_ees = ees
                self.best_epoch = epoch + 1
                no_improve_count = 0

                self.save_checkpoint(
                    epoch + 1, val_metrics, ees,
                    self.output_dir / 'best_model.pt'
                )
                print(f"  [NEW BEST] EES: {ees:.4f}")
            else:
                no_improve_count += 1

            # Check target reached
            if ees >= target_ees:
                print(f"\n{'='*70}")
                print(f"TARGET EES {target_ees} REACHED!")
                print(f"{'='*70}")
                break

            # Early stopping
            if no_improve_count >= early_stop_patience:
                print(f"\nEarly stopping after {early_stop_patience} epochs without improvement")
                break

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    epoch + 1, val_metrics, ees,
                    self.output_dir / f'checkpoint_epoch_{epoch+1}.pt'
                )

        # Final summary
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Best EES: {self.best_ees:.4f} (epoch {self.best_epoch})")

        # Save final model and history
        self.save_history()

        return self.best_ees

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        ees: float,
        path: Path
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'config': self.config,
            'metrics': metrics,
            'ees': ees,
            'history': self.history,
        }
        torch.save(checkpoint, path)

    def save_history(self):
        """Save training history."""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved training history to {history_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train 3-vae-gemm-v1')

    # Data
    parser.add_argument('--lut-dir', type=str, default=None,
                       help='Directory with precomputed LUTs')
    parser.add_argument('--lut-samples', type=int, default=500000,
                       help='Samples for LUT generation')
    parser.add_argument('--triplet-samples', type=int, default=100000,
                       help='Triplet samples for ultrametric training')

    # Model
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Pretrained checkpoint path')
    parser.add_argument('--latent-dim', type=int, default=16,
                       help='Latent dimension')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--freeze-epochs', type=int, default=10,
                       help='Epochs with frozen encoder')
    parser.add_argument('--target-ees', type=float, default=0.95,
                       help='Target EES score')

    # Output
    parser.add_argument('--output', type=str, default='checkpoints',
                       help='Output directory')

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent
    output_dir = base_dir / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("3-VAE-GEMM-V1 SETUP")
    print("=" * 70)

    # Create or load LUTs
    if args.lut_dir and Path(args.lut_dir).exists():
        print(f"\nLoading LUTs from {args.lut_dir}...")
        luts = load_operation_luts(Path(args.lut_dir))
    else:
        print(f"\nGenerating LUTs ({args.lut_samples:,} samples)...")
        lut_dir = base_dir / 'luts'
        luts = create_operation_luts(
            sample_size=args.lut_samples,
            output_dir=lut_dir
        )

    # Create datasets
    print("\nCreating datasets...")
    op_dataset = OperationLUTDataset(luts)

    # Split operation dataset
    train_size = int(0.9 * len(op_dataset))
    val_size = len(op_dataset) - train_size
    train_op_dataset, val_op_dataset = random_split(op_dataset, [train_size, val_size])

    # Create triplet datasets
    train_triplet_dataset = TripletDataset(num_samples=args.triplet_samples)
    val_triplet_dataset = TripletDataset(num_samples=args.triplet_samples // 10, seed=123)

    # Create data loaders
    train_op_loader = DataLoader(
        train_op_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_op_loader = DataLoader(
        val_op_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    train_triplet_loader = DataLoader(
        train_triplet_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_triplet_loader = DataLoader(
        val_triplet_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    print(f"  Train operation samples: {len(train_op_dataset):,}")
    print(f"  Val operation samples: {len(val_op_dataset):,}")
    print(f"  Train triplet samples: {len(train_triplet_dataset):,}")
    print(f"  Val triplet samples: {len(val_triplet_dataset):,}")

    # Create model
    print("\nCreating model...")
    config = VAEGemmV1Config(latent_dim=args.latent_dim)

    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading pretrained weights from {args.checkpoint}")
        model = VAEGemmV1.from_pretrained(args.checkpoint, config)
    else:
        print("Initializing new model")
        model = VAEGemmV1(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = Trainer(model, config, output_dir=output_dir)

    # Train
    best_ees = trainer.train(
        train_op_loader=train_op_loader,
        train_triplet_loader=train_triplet_loader,
        val_op_loader=val_op_loader,
        val_triplet_loader=val_triplet_loader,
        epochs=args.epochs,
        lr=args.lr,
        freeze_encoder_epochs=args.freeze_epochs,
        target_ees=args.target_ees
    )

    print(f"\nFinal Best EES: {best_ees:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
