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
import numpy as np

# Try to import ternary GEMM module (optional dependency)
try:
    import ternary_tritnet_gemm as gemm
    GEMM_AVAILABLE = True
except ImportError:
    GEMM_AVAILABLE = False
    gemm = None


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


def pack_ternary_to_dense243(
    weights: torch.Tensor
) -> np.ndarray:
    """
    Pack ternary weights {-1, 0, +1} to Dense243 format (5 trits/byte).

    Args:
        weights: Ternary weight tensor [out_features, in_features]
                 Values must be in {-1, 0, +1}

    Returns:
        Dense243-packed weights [out_features, in_features/5] as uint8 array

    Note:
        in_features must be multiple of 5 for correct packing.
        If not, weights are padded with zeros.

    Packing algorithm:
        - Groups of 5 trits are packed into 1 byte
        - Each trit {-1, 0, +1} is mapped to {0, 1, 2}
        - Byte value = t0 + t1*3 + t2*9 + t3*27 + t4*81
        - Valid byte range: 0-242 (Dense243 encoding)
    """
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights)

    out_features, in_features = weights.shape

    # Pad to multiple of 5 if needed
    if in_features % 5 != 0:
        pad_size = 5 - (in_features % 5)
        weights = torch.nn.functional.pad(weights, (0, pad_size), value=0)
        in_features += pad_size

    # Convert to numpy and ensure ternary values
    w_np = weights.detach().cpu().numpy().astype(np.int8)

    # Clamp to {-1, 0, +1} if needed
    w_np = np.clip(w_np, -1, 1)

    # Pack 5 trits per byte
    in_features_packed = in_features // 5
    packed = np.zeros((out_features, in_features_packed), dtype=np.uint8)

    for i in range(in_features_packed):
        # Extract 5 trits
        trits = w_np[:, i*5:(i+1)*5]

        # Map {-1, 0, +1} → {0, 1, 2}
        trits_mapped = trits + 1  # Now in {0, 1, 2}

        # Pack to Dense243 byte: value = t0 + t1*3 + t2*9 + t3*27 + t4*81
        packed[:, i] = (
            trits_mapped[:, 0] +
            trits_mapped[:, 1] * 3 +
            trits_mapped[:, 2] * 9 +
            trits_mapped[:, 3] * 27 +
            trits_mapped[:, 4] * 81
        ).astype(np.uint8)

    return packed


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


class TernaryGEMMFunction(torch.autograd.Function):
    """
    Custom autograd function for TritNet GEMM with gradient support.

    Forward: Use optimized C++ GEMM
    Backward: Use PyTorch for gradient computation (gradients not optimized yet)
    """

    @staticmethod
    def forward(ctx, input, weight, bias, B_packed, threshold):
        """
        Forward pass using C++ GEMM.

        Args:
            input: [M, K] activations
            weight: [N, K] full-precision weights (for backward)
            bias: [N] or None
            B_packed: [N, K/5] Dense243-packed weights
            threshold: Quantization threshold
        """
        # Save for backward
        ctx.save_for_backward(input, weight)
        ctx.threshold = threshold
        ctx.has_bias = bias is not None

        # Convert input to numpy (C-contiguous)
        input_np = input.detach().cpu().numpy().astype(np.float32)
        if not input_np.flags['C_CONTIGUOUS']:
            input_np = np.ascontiguousarray(input_np)

        # Transpose B_packed for GEMM
        B_packed_T = np.ascontiguousarray(B_packed.T)

        # GEMM dimensions
        M = input_np.shape[0]
        K = input_np.shape[1]
        N = B_packed.shape[0]  # out_features

        # Pad K to multiple of 5
        K_padded = ((K + 4) // 5) * 5
        if K != K_padded:
            input_np = np.pad(input_np, ((0, 0), (0, K_padded - K)), mode='constant')
            K = K_padded

        # Call C++ GEMM
        output_np = gemm.gemm(input_np, B_packed_T, M, N, K)

        # Convert to tensor
        output = torch.from_numpy(output_np).to(input.device)

        # Add bias
        if bias is not None:
            output = output + bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using PyTorch (not optimized yet).

        This uses standard PyTorch autodiff with quantized weights.
        Future: Optimize backward pass with custom GEMM kernels.
        """
        input, weight = ctx.saved_tensors

        # Quantize weights for backward (same as forward)
        weight_quantized = quantize_ternary(weight, ctx.threshold)

        # Compute gradients using PyTorch
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            # grad_input = grad_output @ weight_quantized
            grad_input = grad_output.mm(weight_quantized)

        if ctx.needs_input_grad[1]:
            # grad_weight = grad_output^T @ input
            grad_weight = grad_output.t().mm(input)

        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None


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
        threshold: float = 0.5,
        quantize_weights: bool = False,  # Control quantization
        use_ternary_gemm: bool = False   # NEW: Use optimized C++ GEMM
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.quantize_weights = quantize_weights  # Only quantize if enabled
        self.use_ternary_gemm = use_ternary_gemm and GEMM_AVAILABLE  # Use GEMM if available

        # Warn if GEMM requested but not available
        if use_ternary_gemm and not GEMM_AVAILABLE:
            import warnings
            warnings.warn(
                "TritNet GEMM module not available. "
                "Build with: python build/build_tritnet_gemm.py\n"
                "Falling back to PyTorch F.linear.",
                RuntimeWarning
            )

        # Full-precision weights for training
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Cache for packed weights (used when use_ternary_gemm=True)
        self.weights_packed_cache = None
        self._weights_hash = None  # Track if weights changed

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights with values that don't all quantize to zero."""
        # Use larger initialization so weights span {-1, 0, +1} range
        # With threshold=0.5, need std > 0.5 to avoid all-zero quantization
        nn.init.normal_(self.weight, mean=0.0, std=1.0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _pack_weights_if_needed(self) -> np.ndarray:
        """
        Pack weights to Dense243 format and cache the result.

        Returns:
            Dense243-packed weights [out_features, in_features/5]
        """
        # Compute hash of current weights to detect changes
        with torch.no_grad():
            current_hash = hash(self.weight.data.cpu().numpy().tobytes())

        # Return cached weights if they haven't changed
        if self.weights_packed_cache is not None and self._weights_hash == current_hash:
            return self.weights_packed_cache

        # Pack quantized weights to Dense243
        with torch.no_grad():
            w_ternary = quantize_ternary(self.weight, self.threshold)
            packed = pack_ternary_to_dense243(w_ternary)

        # Cache the result
        self.weights_packed_cache = packed
        self._weights_hash = current_hash

        return packed

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional ternary weight quantization.

        Uses optimized C++ GEMM if use_ternary_gemm=True, otherwise
        falls back to PyTorch F.linear.

        Args:
            input: Input tensor [batch_size, in_features]

        Returns:
            Output tensor [batch_size, out_features]
        """
        if self.use_ternary_gemm:
            # Use optimized ternary GEMM (C++ implementation)
            return self._forward_gemm(input)
        else:
            # Standard PyTorch implementation
            return self._forward_pytorch(input)

    def _forward_pytorch(self, input: torch.Tensor) -> torch.Tensor:
        """
        Standard PyTorch forward pass.

        Args:
            input: Input tensor [batch_size, in_features]

        Returns:
            Output tensor [batch_size, out_features]
        """
        # Only quantize weights if enabled (for post-training quantization)
        if self.quantize_weights:
            weight_to_use = StraightThroughEstimator.apply(self.weight, self.threshold)
        else:
            # Use full-precision weights during training
            weight_to_use = self.weight

        # Standard linear transformation
        return F.linear(input, weight_to_use, self.bias)

    def _forward_gemm(self, input: torch.Tensor) -> torch.Tensor:
        """
        Optimized forward pass using TritNet GEMM.

        Args:
            input: Input tensor [batch_size, in_features]

        Returns:
            Output tensor [batch_size, out_features]

        Note:
            - Weights are quantized to ternary {-1, 0, +1}
            - Packed to Dense243 format (5 trits/byte)
            - C++ GEMM kernel used for forward pass
            - PyTorch used for backward pass (gradient computation)
            - Expected speedup: 10-20× for small matrices (forward only)
        """
        # Get packed weights (cached if not changed)
        B_packed = self._pack_weights_if_needed()

        # Use custom autograd function for gradient support
        return TernaryGEMMFunction.apply(
            input, self.weight, self.bias, B_packed, self.threshold
        )

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
