#!/usr/bin/env python3
"""
TritNet Training Script

Train TritNet models on balanced ternary arithmetic truth tables.

Usage:
    # Train tnot model (proof-of-concept)
    python train_tritnet.py --operation tnot --hidden-size 8

    # Train binary operation
    python train_tritnet.py --operation tadd --hidden-size 16

    # Train all operations
    python train_tritnet.py --all

Arguments:
    --operation: Operation to train (tnot, tadd, tmul, tmin, tmax, or --all)
    --hidden-size: Number of hidden neurons (default: 8 for unary, 16 for binary)
    --learning-rate: Learning rate for Adam optimizer (default: 0.001)
    --max-epochs: Maximum training epochs (default: 2000)
    --threshold: Ternary quantization threshold (default: 0.5)
    --seed: Random seed for reproducibility (default: 42)
    --output-dir: Directory to save trained models (default: models/tritnet/)
"""

import argparse
import json
import sys
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "models" / "tritnet" / "src"))

from ternary_layers import count_parameters, count_ternary_parameters
from tritnet_model import TritNetUnary, TritNetUnaryDeep, TritNetBinary, save_tritnet_model


def load_truth_table(operation: str, data_dir: Path) -> tuple:
    """
    Load truth table dataset for an operation.

    Args:
        operation: Operation name (tnot, tadd, tmul, tmin, tmax)
        data_dir: Directory containing truth table JSON files

    Returns:
        Tuple of (X, Y) as torch tensors
        - X: Input tensor [num_samples, input_size]
        - Y: Output tensor [num_samples, 5]
    """
    truth_table_file = data_dir / f"{operation}_truth_table.json"

    if not truth_table_file.exists():
        raise FileNotFoundError(f"Truth table not found: {truth_table_file}")

    print(f"Loading truth table: {truth_table_file}")

    with open(truth_table_file) as f:
        data = json.load(f)

    metadata = data['metadata']
    samples = data['samples']

    print(f"  Operation: {metadata['operation']}")
    print(f"  Type: {metadata['operation_type']}")
    print(f"  Samples: {metadata['num_samples']}")

    # Extract inputs and outputs
    X = []
    Y = []

    for sample in samples:
        X.append(sample['input'])
        Y.append(sample['output'])

    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    print(f"  Input shape: {X.shape}, Output shape: {Y.shape}")

    return X, Y


def ternary_to_class_indices(values: torch.Tensor) -> torch.Tensor:
    """
    Convert ternary values {-1, 0, +1} to class indices {0, 1, 2}.

    Args:
        values: Ternary values [batch_size, 5] with values in {-1, 0, +1}

    Returns:
        Class indices [batch_size, 5] with values in {0, 1, 2}
    """
    # Map: -1 → 0, 0 → 1, +1 → 2
    return (values + 1).long()


def class_indices_to_ternary(indices: torch.Tensor) -> torch.Tensor:
    """
    Convert class indices {0, 1, 2} to ternary values {-1, 0, +1}.

    Args:
        indices: Class indices [batch_size, 5] with values in {0, 1, 2}

    Returns:
        Ternary values [batch_size, 5] with values in {-1, 0, +1}
    """
    # Map: 0 → -1, 1 → 0, 2 → +1
    return indices.float() - 1


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute exact match accuracy (all 5 trits must match).

    Args:
        predictions: Predicted values [batch_size, 5] (continuous during training)
        targets: Target ternary values [batch_size, 5]

    Returns:
        Accuracy as fraction in [0, 1]
    """
    # Quantize predictions to ternary values {-1, 0, +1}
    # During training, predictions are continuous, so we need to discretize them
    pred_ternary = torch.sign(predictions)

    # Check if all 5 trits match for each sample
    exact_matches = (pred_ternary == targets).all(dim=1)

    # Return fraction of exact matches
    return exact_matches.float().mean().item()


def compute_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute exact match accuracy from cross-entropy logits.

    Args:
        logits: Predicted logits [batch_size, 5, 3] (3 classes per trit)
        targets: Target ternary values [batch_size, 5]

    Returns:
        Accuracy as fraction in [0, 1]
    """
    # Get predicted class for each trit position
    pred_classes = torch.argmax(logits, dim=2)  # [batch_size, 5]

    # Convert to ternary values
    pred_ternary = class_indices_to_ternary(pred_classes)

    # Check if all 5 trits match for each sample
    exact_matches = (pred_ternary == targets).all(dim=1)

    # Return fraction of exact matches
    return exact_matches.float().mean().item()


def train_tritnet(
    operation: str,
    hidden_size: int,
    learning_rate: float,
    max_epochs: int,
    threshold: float,
    seed: int,
    data_dir: Path,
    output_dir: Path,
    architecture: str = 'shallow',
    loss_type: str = 'mse'
) -> dict:
    """
    Train TritNet model on a single operation.

    Args:
        operation: Operation to train (tnot, tadd, tmul, tmin, tmax)
        hidden_size: Number of hidden neurons
        learning_rate: Learning rate for optimizer
        max_epochs: Maximum training epochs
        threshold: Ternary quantization threshold
        seed: Random seed for reproducibility
        data_dir: Directory containing truth tables
        output_dir: Directory to save trained model
        architecture: Architecture type ('shallow' or 'deep')
        loss_type: Loss function ('mse' or 'crossentropy')

    Returns:
        Dictionary with training results
    """
    print("\n" + "="*70)
    print(f"Training TritNet for operation: {operation}")
    print("="*70)

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load truth table
    X_train, Y_train = load_truth_table(operation, data_dir)
    num_samples = X_train.shape[0]

    # Determine model type
    is_unary = (operation == 'tnot')

    # Create model
    if is_unary:
        if architecture == 'deep':
            model = TritNetUnaryDeep(hidden_size=hidden_size, threshold=threshold)
            model_type = 'TritNetUnaryDeep'
        else:
            model = TritNetUnary(hidden_size=hidden_size, threshold=threshold)
            model_type = 'TritNetUnary'
    else:
        model = TritNetBinary(hidden_size=hidden_size, threshold=threshold)
        model_type = 'TritNetBinary'

    print(f"\nModel architecture:")
    print(f"  Type: {model_type}")
    print(f"  Architecture: {architecture}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Total parameters: {count_parameters(model)}")
    print(f"  Quantization threshold: {threshold}")
    print(f"  Loss function: {loss_type}")

    # Setup training
    if loss_type == 'mse':
        criterion = nn.MSELoss()
        use_crossentropy = False
    elif loss_type == 'crossentropy':
        # TODO: Cross-entropy requires model output reshape to [batch, 5, 3]
        # For now, using MSE. Cross-entropy is future work.
        print("  WARNING: Cross-entropy not yet implemented, using MSE instead")
        criterion = nn.MSELoss()
        use_crossentropy = False
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"\nTraining for up to {max_epochs} epochs...")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Dataset size: {num_samples}")

    start_time = time.time()
    best_accuracy = 0.0
    best_epoch = 0
    training_history = []

    for epoch in range(max_epochs):
        # Forward pass
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)

        # Compute loss
        loss = criterion(outputs, Y_train)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Evaluate accuracy
        model.eval()
        with torch.no_grad():
            predictions = model(X_train)
            accuracy = compute_accuracy(predictions, Y_train)

        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch

        # Record history
        training_history.append({
            'epoch': epoch,
            'loss': loss.item(),
            'accuracy': accuracy
        })

        # Print progress
        if epoch % 100 == 0 or accuracy >= 0.9999:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:4d}: Loss {loss.item():.6f}, "
                  f"Accuracy {accuracy*100:.2f}%, "
                  f"Time {elapsed:.1f}s")

        # Early stopping if 100% accuracy achieved
        if accuracy >= 0.9999:
            print(f"\n✓ Achieved 100% accuracy at epoch {epoch}!")
            break

    # Final evaluation
    total_time = time.time() - start_time
    model.eval()
    with torch.no_grad():
        final_predictions = model(X_train)
        final_accuracy = compute_accuracy(final_predictions, Y_train)

    print(f"\nTraining complete:")
    print(f"  Final accuracy: {final_accuracy*100:.2f}%")
    print(f"  Best accuracy: {best_accuracy*100:.2f}% (epoch {best_epoch})")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Epochs: {epoch+1}")

    # Analyze weight distribution
    ternary_counts = count_ternary_parameters(model)
    total_weights = sum(ternary_counts.values())
    print(f"\nWeight distribution:")
    print(f"  -1: {ternary_counts['minus_one']} ({ternary_counts['minus_one']/total_weights*100:.1f}%)")
    print(f"   0: {ternary_counts['zero']} ({ternary_counts['zero']/total_weights*100:.1f}%)")
    print(f"  +1: {ternary_counts['plus_one']} ({ternary_counts['plus_one']/total_weights*100:.1f}%)")

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"tritnet_{operation}.tritnet"

    metadata = {
        'operation': operation,
        'final_accuracy': final_accuracy,
        'best_accuracy': best_accuracy,
        'epochs_trained': epoch + 1,
        'training_time_seconds': total_time,
        'hidden_size': hidden_size,
        'learning_rate': learning_rate,
        'threshold': threshold,
        'seed': seed,
        'num_samples': num_samples,
        'ternary_counts': ternary_counts,
        'architecture': architecture,
        'loss_type': loss_type,
        'model_type': model_type,
    }

    save_tritnet_model(model, model_path, metadata)

    # Save training history
    history_path = output_dir / f"tritnet_{operation}_history.json"
    with open(history_path, 'w') as f:
        json.dump({
            'metadata': metadata,
            'history': training_history
        }, f, indent=2)

    print(f"✓ Training history saved to: {history_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Train TritNet models on ternary arithmetic operations"
    )
    parser.add_argument(
        "--operation",
        type=str,
        choices=['tnot', 'tadd', 'tmul', 'tmin', 'tmax'],
        help="Operation to train"
    )
    parser.add_argument(
        "--all",
        action='store_true',
        help="Train all operations"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        help="Number of hidden neurons (default: 8 for unary, 16 for binary)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer (default: 0.001)"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=2000,
        help="Maximum training epochs (default: 2000)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Ternary quantization threshold (default: 0.5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "datasets" / "tritnet",
        help="Directory containing truth tables"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "tritnet",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=['shallow', 'deep'],
        default='shallow',
        help="Network architecture (shallow: 2 hidden layers, deep: 4 hidden layers with skip connections)"
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=['mse', 'crossentropy'],
        default='mse',
        help="Loss function (mse: regression loss, crossentropy: classification loss)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.operation:
        parser.error("Must specify either --operation or --all")

    # Determine operations to train
    if args.all:
        operations = ['tnot', 'tadd', 'tmul', 'tmin', 'tmax']
    else:
        operations = [args.operation]

    print("\n" + "="*70)
    print("TritNet Training Pipeline")
    print("="*70)
    print(f"Operations to train: {', '.join(operations)}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Random seed: {args.seed}")

    # Train each operation
    results = {}

    for operation in operations:
        # Determine hidden size
        if args.hidden_size:
            hidden_size = args.hidden_size
        else:
            hidden_size = 8 if operation == 'tnot' else 16

        # Train model
        try:
            metadata = train_tritnet(
                operation=operation,
                hidden_size=hidden_size,
                learning_rate=args.learning_rate,
                max_epochs=args.max_epochs,
                threshold=args.threshold,
                seed=args.seed,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                architecture=args.architecture,
                loss_type=args.loss
            )
            results[operation] = metadata
        except Exception as e:
            print(f"\n❌ Error training {operation}: {e}")
            import traceback
            traceback.print_exc()
            results[operation] = {'error': str(e)}

    # Summary
    print("\n" + "="*70)
    print("Training Summary")
    print("="*70)

    for operation, metadata in results.items():
        if 'error' in metadata:
            print(f"  {operation:5s}: ERROR - {metadata['error']}")
        else:
            accuracy = metadata['final_accuracy']
            epochs = metadata['epochs_trained']
            time_sec = metadata['training_time_seconds']
            print(f"  {operation:5s}: {accuracy*100:.2f}% accuracy "
                  f"({epochs} epochs, {time_sec:.1f}s)")

    # Go/No-Go decision
    print("\n" + "="*70)
    print("Go/No-Go Decision Criteria")
    print("="*70)

    successful_operations = [
        op for op, meta in results.items()
        if 'error' not in meta and meta['final_accuracy'] > 0.99
    ]
    perfect_operations = [
        op for op, meta in results.items()
        if 'error' not in meta and meta['final_accuracy'] >= 0.9999
    ]

    print(f"Operations with >99% accuracy: {len(successful_operations)}/{ len(results)}")
    print(f"Operations with 100% accuracy: {len(perfect_operations)}/{len(results)}")

    if successful_operations:
        print(f"  >99%: {', '.join(successful_operations)}")
    if perfect_operations:
        print(f"  100%: {', '.join(perfect_operations)}")

    # Decision
    if len(successful_operations) >= 3 and len(perfect_operations) >= 1:
        print("\n✅ GO: Criteria met! Proceed to Phase 3 (C++ Integration)")
        print("   - At least 3 operations achieved >99% accuracy")
        print("   - At least 1 operation achieved 100% accuracy")
        print("   - TritNet proves exact arithmetic is learnable!")
    elif len(successful_operations) >= 1:
        print("\n⚠️  PARTIAL SUCCESS: Some operations learned successfully")
        print("   - Consider architecture adjustments for failed operations")
        print("   - Analyze what makes certain operations easier to learn")
        print("   - Possible pivot to approximate arithmetic research")
    else:
        print("\n❌ NO-GO: Criteria not met")
        print("   - No operations achieved >99% accuracy")
        print("   - Investigate why NNs cannot learn exact arithmetic")
        print("   - Consider publishing negative results as research contribution")


if __name__ == "__main__":
    main()
