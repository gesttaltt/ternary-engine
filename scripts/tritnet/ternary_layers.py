"""
Ternary Neural Network Layers

Implements neural network layers with ternary weights {-1, 0, +1} for TritNet.

Key components:
- TernaryLinear: Linear layer with ternary weight quantization
- Straight-through estimator (STE) for gradient flow
- Ternary activation functions

Usage:
    import torch
    from ternary_layers import TernaryLinear, ternary_sign

    layer = TernaryLinear(10, 16)  # 10 inputs → 16 outputs
    x = torch.randn(batch_size, 10)
    y = ternary_sign(layer(x))  # Apply ternary activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def quantize_ternary(
    weights: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Quantize weights to ternary values {-1, 0, +1}.

    Args:
        weights: Full-precision weight tensor
        threshold: Threshold for zero region (weights with |w| < threshold → 0)

    Returns:
        Ternary weights with values in {-1, 0, +1}

    Method:
        w_ternary = sign(w) if |w| > threshold else 0
    """
    sign = torch.sign(weights)
    mask = (torch.abs(weights) > threshold).float()
    return sign * mask


def ternary_sign(x: torch.Tensor) -> torch.Tensor:
    """
    Ternary activation function using sign.

    Args:
        x: Input tensor

    Returns:
        Ternary output with values in {-1, 0, +1}

    Note:
        torch.sign(x) returns:
        - -1 if x < 0
        - 0 if x == 0
        - +1 if x > 0
    """
    return torch.sign(x)


class StraightThroughEstimator(torch.autograd.Function):
    """
    Straight-through estimator for gradient flow through quantization.

    Forward: Quantize to ternary
    Backward: Pass gradients straight through (identity function)

    This allows gradients to flow back to full-precision weights during training.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Quantize input to ternary values."""
        return quantize_ternary(input, threshold)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """Pass gradients straight through."""
        # Return gradients for input and None for threshold (no gradient)
        return grad_output, None


class TernaryLinear(nn.Module):
    """
    Linear layer with ternary weight quantization.

    During forward pass:
    - Weights are quantized to {-1, 0, +1}
    - Standard linear transformation: y = xW^T + b

    During backward pass:
    - Gradients flow to full-precision weights via STE
    - Weights updated with standard optimizers (Adam, SGD, etc.)

    Args:
        in_features: Number of input features
        out_features: Number of output features
        bias: If True, add learnable bias (default: False for ternary networks)
        threshold: Quantization threshold for zero region (default: 0.5)

    Attributes:
        weight: Full-precision weight parameter [out_features, in_features]
        bias: Bias parameter (if enabled)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        threshold: float = 0.5
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold

        # Full-precision weights for training
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights with small random values."""
        # Use smaller initialization for ternary weights
        nn.init.normal_(self.weight, mean=0.0, std=0.1)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with ternary weight quantization.

        Args:
            input: Input tensor [batch_size, in_features]

        Returns:
            Output tensor [batch_size, out_features]
        """
        # Quantize weights to ternary using STE
        weight_ternary = StraightThroughEstimator.apply(self.weight, self.threshold)

        # Standard linear transformation
        return F.linear(input, weight_ternary, self.bias)

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'bias={self.bias is not None}, threshold={self.threshold}'

    def get_quantized_weights(self) -> torch.Tensor:
        """
        Get quantized ternary weights (for inspection/export).

        Returns:
            Ternary weights with values in {-1, 0, +1}
        """
        with torch.no_grad():
            return quantize_ternary(self.weight, self.threshold)

    def count_ternary_values(self) -> dict:
        """
        Count distribution of ternary values in quantized weights.

        Returns:
            Dictionary with counts of {-1, 0, +1} values
        """
        with torch.no_grad():
            w_ternary = self.get_quantized_weights()
            return {
                'minus_one': (w_ternary == -1).sum().item(),
                'zero': (w_ternary == 0).sum().item(),
                'plus_one': (w_ternary == 1).sum().item(),
            }


class TernaryActivation(nn.Module):
    """
    Ternary activation function wrapper.

    Applies sign activation to produce ternary outputs {-1, 0, +1}.

    Args:
        threshold: Optional threshold for ternary classification
                  (default: None, uses standard sign)
    """

    def __init__(self, threshold: Optional[float] = None):
        super().__init__()
        self.threshold = threshold

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply ternary activation.

        Args:
            input: Input tensor

        Returns:
            Ternary output with values in {-1, 0, +1}
        """
        if self.threshold is not None:
            # Threshold-based ternary activation
            sign = torch.sign(input)
            mask = (torch.abs(input) > self.threshold).float()
            return sign * mask
        else:
            # Standard sign activation
            return torch.sign(input)

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f'threshold={self.threshold}' if self.threshold else 'sign'


# Convenience functions
def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_ternary_parameters(model: nn.Module) -> dict:
    """
    Count ternary value distribution across all TernaryLinear layers.

    Args:
        model: Neural network model

    Returns:
        Dictionary with total counts of {-1, 0, +1} values
    """
    total_counts = {'minus_one': 0, 'zero': 0, 'plus_one': 0}

    for module in model.modules():
        if isinstance(module, TernaryLinear):
            counts = module.count_ternary_values()
            total_counts['minus_one'] += counts['minus_one']
            total_counts['zero'] += counts['zero']
            total_counts['plus_one'] += counts['plus_one']

    return total_counts


if __name__ == "__main__":
    # Test ternary layers
    print("Testing TernaryLinear layer...")

    # Create test layer
    layer = TernaryLinear(5, 8, bias=False, threshold=0.5)
    print(f"Layer: {layer}")

    # Test forward pass
    x = torch.randn(10, 5)  # Batch of 10 samples
    y = layer(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")

    # Check quantized weights
    w_ternary = layer.get_quantized_weights()
    unique_values = torch.unique(w_ternary)
    print(f"Quantized weight values: {unique_values.tolist()}")
    assert set(unique_values.tolist()).issubset({-1.0, 0.0, 1.0}), "Weights not ternary!"

    # Count ternary distribution
    counts = layer.count_ternary_values()
    print(f"Ternary distribution: {counts}")

    # Test activation
    activation = TernaryActivation()
    y_activated = activation(y)
    unique_activated = torch.unique(y_activated)
    print(f"Activated values: {unique_activated.tolist()}")

    # Test gradient flow
    loss = y_activated.sum()
    loss.backward()
    print(f"Gradients computed: {layer.weight.grad is not None}")

    print("\n✓ All tests passed!")
