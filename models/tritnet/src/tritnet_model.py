"""
TritNet Model Definitions

Neural network models for learning balanced ternary arithmetic operations.

Models:
- TritNetUnary: For unary operations (tnot)
- TritNetBinary: For binary operations (tadd, tmul, tmin, tmax)

Architecture:
- All weights quantized to {-1, 0, +1}
- Sign activation for ternary outputs
- 2 hidden layers with configurable size

Usage:
    from tritnet_model import TritNetUnary, TritNetBinary
    import torch

    # Unary operation model
    model = TritNetUnary(hidden_size=8)
    x = torch.tensor([[-1, 0, 1, -1, 1]], dtype=torch.float32)
    y = model(x)  # Returns ternary outputs {-1, 0, +1}

    # Binary operation model
    model = TritNetBinary(hidden_size=16)
    x = torch.tensor([[-1, 0, 1, -1, 1, 1, 0, -1, 1, 0]], dtype=torch.float32)
    y = model(x)  # Returns ternary outputs {-1, 0, +1}
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any

from ternary_layers import TernaryLinear, TernaryActivation, count_parameters


class TritNetUnary(nn.Module):
    """
    TritNet model for unary ternary operations (e.g., tnot).

    Architecture:
        Input: 5 trits {-1, 0, +1}
        Hidden Layer 1: hidden_size neurons, ternary weights
        Hidden Layer 2: hidden_size neurons, ternary weights
        Output: 5 trits {-1, 0, +1}

    Args:
        hidden_size: Number of neurons in hidden layers (default: 8)
        threshold: Quantization threshold for ternary weights (default: 0.5)

    Attributes:
        layer1: First hidden layer [5 → hidden_size]
        layer2: Second hidden layer [hidden_size → hidden_size]
        layer3: Output layer [hidden_size → 5]
        activation: Ternary sign activation
    """

    def __init__(self, hidden_size: int = 8, threshold: float = 0.5):
        super().__init__()

        self.hidden_size = hidden_size
        self.threshold = threshold
        self.apply_activation = False  # Don't apply during training

        # Network layers
        self.layer1 = TernaryLinear(5, hidden_size, bias=False, threshold=threshold)
        self.layer2 = TernaryLinear(hidden_size, hidden_size, bias=False, threshold=threshold)
        self.layer3 = TernaryLinear(hidden_size, 5, bias=False, threshold=threshold)

        # Ternary activation (only for inference)
        self.activation = TernaryActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Args:
            x: Input tensor [batch_size, 5] with values in {-1, 0, +1}

        Returns:
            Output tensor [batch_size, 5] with values in {-1, 0, +1}
        """
        # Layer 1 (NO activation - allow gradients to flow)
        x = self.layer1(x)

        # Layer 2 (NO activation - allow gradients to flow)
        x = self.layer2(x)

        # Output layer
        x = self.layer3(x)

        # Only apply activation during inference, not training
        if self.apply_activation:
            x = self.activation(x)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for saving."""
        return {
            'model_type': 'TritNetUnary',
            'hidden_size': self.hidden_size,
            'threshold': self.threshold,
            'num_parameters': count_parameters(self),
        }


class TritNetUnaryDeep(nn.Module):
    """
    Improved TritNet model for unary operations with deeper architecture.

    Architecture (Phase 2A-v2):
        Input: 5 trits {-1, 0, +1}
        Hidden Layer 1: hidden_size neurons, ternary weights
        Hidden Layer 2: hidden_size neurons, ternary weights + skip connection
        Hidden Layer 3: hidden_size neurons, ternary weights
        Hidden Layer 4: hidden_size neurons, ternary weights + skip connection
        Output: 5 trits {-1, 0, +1}

    Improvements over TritNetUnary:
    - 4 hidden layers instead of 2 (more complex decision boundaries)
    - Skip connections every 2 layers (ResNet-style, helps gradient flow)
    - Larger default hidden size (16 vs 8)

    Args:
        hidden_size: Number of neurons in hidden layers (default: 16)
        threshold: Quantization threshold for ternary weights (default: 0.5)

    Attributes:
        layer1-5: Network layers
        projection: Optional projection for skip connections if dimensions mismatch
        activation: Ternary sign activation (inference only)
    """

    def __init__(self, hidden_size: int = 16, threshold: float = 0.5):
        super().__init__()

        self.hidden_size = hidden_size
        self.threshold = threshold
        self.apply_activation = False  # Don't apply during training

        # Network layers with gradually increasing then decreasing size
        self.layer1 = TernaryLinear(5, hidden_size, bias=False, threshold=threshold)
        self.layer2 = TernaryLinear(hidden_size, hidden_size, bias=False, threshold=threshold)
        self.layer3 = TernaryLinear(hidden_size, hidden_size, bias=False, threshold=threshold)
        self.layer4 = TernaryLinear(hidden_size, hidden_size, bias=False, threshold=threshold)
        self.layer5 = TernaryLinear(hidden_size, 5, bias=False, threshold=threshold)

        # Projection layer for first skip connection (5 → hidden_size)
        self.projection = TernaryLinear(5, hidden_size, bias=False, threshold=threshold)

        # Ternary activation (only for inference)
        self.activation = TernaryActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connections.

        Args:
            x: Input tensor [batch_size, 5] with values in {-1, 0, +1}

        Returns:
            Output tensor [batch_size, 5] with values in {-1, 0, +1}
        """
        # Save input for first skip connection
        identity = self.projection(x)

        # Layer 1
        x = self.layer1(x)

        # Layer 2 + skip connection from input
        x = self.layer2(x) + identity

        # Save for second skip connection
        identity2 = x

        # Layer 3
        x = self.layer3(x)

        # Layer 4 + skip connection
        x = self.layer4(x) + identity2

        # Output layer
        x = self.layer5(x)

        # Only apply activation during inference, not training
        if self.apply_activation:
            x = self.activation(x)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for saving."""
        return {
            'model_type': 'TritNetUnaryDeep',
            'hidden_size': self.hidden_size,
            'threshold': self.threshold,
            'num_parameters': count_parameters(self),
        }


class TritNetBinary(nn.Module):
    """
    TritNet model for binary ternary operations (e.g., tadd, tmul, tmin, tmax).

    Architecture:
        Input: 10 trits (5 from A, 5 from B) {-1, 0, +1}
        Hidden Layer 1: hidden_size neurons, ternary weights
        Hidden Layer 2: hidden_size neurons, ternary weights
        Output: 5 trits {-1, 0, +1}

    Args:
        hidden_size: Number of neurons in hidden layers (default: 16)
        threshold: Quantization threshold for ternary weights (default: 0.5)

    Attributes:
        layer1: First hidden layer [10 → hidden_size]
        layer2: Second hidden layer [hidden_size → hidden_size]
        layer3: Output layer [hidden_size → 5]
        activation: Ternary sign activation
    """

    def __init__(self, hidden_size: int = 16, threshold: float = 0.5):
        super().__init__()

        self.hidden_size = hidden_size
        self.threshold = threshold
        self.apply_activation = False  # Don't apply during training

        # Network layers
        self.layer1 = TernaryLinear(10, hidden_size, bias=False, threshold=threshold)
        self.layer2 = TernaryLinear(hidden_size, hidden_size, bias=False, threshold=threshold)
        self.layer3 = TernaryLinear(hidden_size, 5, bias=False, threshold=threshold)

        # Ternary activation (only for inference)
        self.activation = TernaryActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Args:
            x: Input tensor [batch_size, 10] with values in {-1, 0, +1}
               First 5 values from operand A, next 5 from operand B

        Returns:
            Output tensor [batch_size, 5] with values in {-1, 0, +1}
        """
        # Layer 1 (NO activation - allow gradients to flow)
        x = self.layer1(x)

        # Layer 2 (NO activation - allow gradients to flow)
        x = self.layer2(x)

        # Output layer
        x = self.layer3(x)

        # Only apply activation during inference, not training
        if self.apply_activation:
            x = self.activation(x)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for saving."""
        return {
            'model_type': 'TritNetBinary',
            'hidden_size': self.hidden_size,
            'threshold': self.threshold,
            'num_parameters': count_parameters(self),
        }


def save_tritnet_model(
    model: nn.Module,
    filepath: Path,
    metadata: Dict[str, Any] = None
):
    """
    Save TritNet model to .tritnet format.

    File format:
        - PyTorch .pth checkpoint (full model state)
        - Quantized weights as numpy arrays
        - Metadata (operation, accuracy, etc.)

    Args:
        model: Trained TritNet model
        filepath: Path to save model (with .tritnet extension)
        metadata: Additional metadata to store (accuracy, operation name, etc.)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Extract quantized weights
    quantized_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            w_ternary = module.get_quantized_weights()
            quantized_weights[name] = w_ternary.cpu().numpy()

    # Create checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'quantized_weights': quantized_weights,
        'config': model.get_config(),
        'metadata': metadata or {},
    }

    # Save as PyTorch checkpoint
    torch.save(checkpoint, filepath)

    print(f"✓ Model saved to: {filepath}")
    print(f"  Size: {filepath.stat().st_size / 1024:.2f} KB")
    print(f"  Parameters: {checkpoint['config']['num_parameters']}")


def load_tritnet_model(filepath: Path) -> tuple:
    """
    Load TritNet model from .tritnet file.

    Args:
        filepath: Path to .tritnet model file

    Returns:
        Tuple of (model, metadata)
    """
    filepath = Path(filepath)

    # Load checkpoint
    checkpoint = torch.load(filepath, map_location='cpu')

    # Reconstruct model
    config = checkpoint['config']
    if config['model_type'] == 'TritNetUnary':
        model = TritNetUnary(
            hidden_size=config['hidden_size'],
            threshold=config['threshold']
        )
    elif config['model_type'] == 'TritNetUnaryDeep':
        model = TritNetUnaryDeep(
            hidden_size=config['hidden_size'],
            threshold=config['threshold']
        )
    elif config['model_type'] == 'TritNetBinary':
        model = TritNetBinary(
            hidden_size=config['hidden_size'],
            threshold=config['threshold']
        )
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    metadata = checkpoint.get('metadata', {})

    print(f"✓ Model loaded from: {filepath}")
    print(f"  Type: {config['model_type']}")
    print(f"  Parameters: {config['num_parameters']}")

    return model, metadata


def export_weights_to_numpy(
    model: nn.Module,
    filepath: Path
):
    """
    Export quantized ternary weights as NumPy arrays.

    Saves weights in format suitable for C++ integration:
        - W1.npy: [in_features, hidden_size]
        - W2.npy: [hidden_size, hidden_size]
        - W3.npy: [hidden_size, out_features]

    Args:
        model: Trained TritNet model
        filepath: Directory to save weight arrays
    """
    filepath = Path(filepath)
    filepath.mkdir(parents=True, exist_ok=True)

    layer_names = ['layer1', 'layer2', 'layer3']
    weight_names = ['W1', 'W2', 'W3']

    for layer_name, weight_name in zip(layer_names, weight_names):
        layer = getattr(model, layer_name)
        if isinstance(layer, TernaryLinear):
            w_ternary = layer.get_quantized_weights().cpu().numpy()

            # Transpose for C++ row-major layout
            w_ternary_t = w_ternary.T

            # Save as int8 array
            w_int8 = w_ternary_t.astype(np.int8)
            output_file = filepath / f"{weight_name}.npy"
            np.save(output_file, w_int8)

            print(f"✓ Exported {weight_name}: {w_int8.shape} → {output_file}")


if __name__ == "__main__":
    print("Testing TritNet models...")

    # Test unary model
    print("\n1. Testing TritNetUnary...")
    model_unary = TritNetUnary(hidden_size=8)
    print(f"   Architecture: {model_unary}")
    print(f"   Parameters: {count_parameters(model_unary)}")

    x_unary = torch.tensor([[-1, 0, 1, -1, 1]], dtype=torch.float32)
    y_unary = model_unary(x_unary)
    print(f"   Input shape: {x_unary.shape}, Output shape: {y_unary.shape}")
    print(f"   Output values: {y_unary[0].tolist()}")

    # Verify ternary outputs
    unique_values = torch.unique(y_unary)
    assert set(unique_values.tolist()).issubset({-1.0, 0.0, 1.0}), "Outputs not ternary!"
    print(f"   ✓ Outputs are ternary: {unique_values.tolist()}")

    # Test binary model
    print("\n2. Testing TritNetBinary...")
    model_binary = TritNetBinary(hidden_size=16)
    print(f"   Architecture: {model_binary}")
    print(f"   Parameters: {count_parameters(model_binary)}")

    x_binary = torch.tensor([[-1, 0, 1, -1, 1, 1, 0, -1, 1, 0]], dtype=torch.float32)
    y_binary = model_binary(x_binary)
    print(f"   Input shape: {x_binary.shape}, Output shape: {y_binary.shape}")
    print(f"   Output values: {y_binary[0].tolist()}")

    # Verify ternary outputs
    unique_values = torch.unique(y_binary)
    assert set(unique_values.tolist()).issubset({-1.0, 0.0, 1.0}), "Outputs not ternary!"
    print(f"   ✓ Outputs are ternary: {unique_values.tolist()}")

    # Test save/load
    print("\n3. Testing save/load...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.tritnet"

        # Save model
        save_tritnet_model(
            model_unary,
            model_path,
            metadata={'operation': 'tnot', 'test': True}
        )

        # Load model
        loaded_model, metadata = load_tritnet_model(model_path)
        print(f"   Loaded metadata: {metadata}")

        # Verify same outputs
        y_loaded = loaded_model(x_unary)
        assert torch.allclose(y_unary, y_loaded), "Loaded model outputs differ!"
        print(f"   ✓ Loaded model produces identical outputs")

    print("\n✓ All tests passed!")
