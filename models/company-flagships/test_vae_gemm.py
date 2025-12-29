#!/usr/bin/env python3
"""
Test VAE GEMM Inference Pipeline

Validates the exported Dense243 weights by:
1. Loading binary files and performing inference in Python
2. Comparing against original PyTorch model
3. Measuring speedup from ternary quantization

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0
"""

import os
import sys
import json
import struct
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


# =============================================================================
# Dense243 Unpacking (Python reference)
# =============================================================================

def unpack_dense243_byte(packed: int) -> np.ndarray:
    """Unpack a Dense243 byte to 5 trits {-1, 0, +1}."""
    trits = np.zeros(5, dtype=np.float32)  # Use float32 to avoid overflow
    val = int(packed)
    # Invalid bytes (>242) map to all zeros
    if val >= 243:
        return trits
    for i in range(5):
        trits[i] = float((val % 3) - 1)  # {0,1,2} -> {-1,0,+1}
        val //= 3
    return trits


def unpack_dense243_array(packed: np.ndarray, K: int) -> np.ndarray:
    """Unpack Dense243 array to ternary weights."""
    out_features, K_packed = packed.shape
    weights = np.zeros((out_features, K), dtype=np.float32)

    for n in range(out_features):
        for k_pack in range(K_packed):
            trits = unpack_dense243_byte(packed[n, k_pack])
            k_base = k_pack * 5
            for t in range(5):
                if k_base + t < K:
                    weights[n, k_base + t] = trits[t]

    return weights


# =============================================================================
# Binary File Loading
# =============================================================================

def load_layer_file(path: Path):
    """Load encoder/decoder binary file."""
    with open(path, 'rb') as f:
        # Header
        magic = f.read(4).decode('ascii', errors='ignore')
        version, = struct.unpack('<I', f.read(4))
        num_layers, = struct.unpack('<I', f.read(4))
        weight_bytes, = struct.unpack('<I', f.read(4))
        bias_bytes, = struct.unpack('<I', f.read(4))

        # Layer configs
        layers = []
        for _ in range(num_layers):
            name = f.read(32).rstrip(b'\x00').decode('utf-8')
            in_feat, = struct.unpack('<I', f.read(4))
            out_feat, = struct.unpack('<I', f.read(4))
            w_offset, = struct.unpack('<I', f.read(4))
            w_size, = struct.unpack('<I', f.read(4))
            b_offset, = struct.unpack('<I', f.read(4))
            b_size, = struct.unpack('<I', f.read(4))
            has_act, = struct.unpack('<?', f.read(1))

            layers.append({
                'name': name,
                'in_features': in_feat,
                'out_features': out_feat,
                'weight_offset': w_offset,
                'weight_size': w_size,
                'bias_offset': b_offset,
                'bias_size': b_size,
                'has_activation': has_act
            })

        # Weights (Dense243 packed)
        weights_data = f.read(weight_bytes)

        # Biases (float32)
        biases_data = f.read(bias_bytes)

    return {
        'magic': magic,
        'version': version,
        'layers': layers,
        'weights_raw': weights_data,
        'biases_raw': biases_data
    }


def get_layer_weights(data: dict, layer_idx: int):
    """Extract unpacked weights and biases for a layer."""
    layer = data['layers'][layer_idx]

    # Get packed weights
    K = layer['in_features']
    K_packed = (K + 4) // 5
    N = layer['out_features']

    w_start = layer['weight_offset']
    w_end = w_start + layer['weight_size']
    packed = np.frombuffer(data['weights_raw'][w_start:w_end], dtype=np.uint8)
    packed = packed.reshape(N, K_packed)

    # Unpack to ternary (already float32)
    weights = unpack_dense243_array(packed, K)

    # Get bias
    b_start = layer['bias_offset']
    b_end = b_start + layer['bias_size']
    bias = np.frombuffer(data['biases_raw'][b_start:b_end], dtype=np.float32).copy()

    return weights, bias


# =============================================================================
# Ternary Linear Layer (Python reference)
# =============================================================================

def ternary_linear(x: np.ndarray, W: np.ndarray, b: np.ndarray,
                   apply_relu: bool = True) -> np.ndarray:
    """
    Ternary linear layer: y = xW^T + b

    Args:
        x: [batch, in_features]
        W: [out_features, in_features] ternary weights
        b: [out_features] bias

    Returns:
        y: [batch, out_features]
    """
    y = x @ W.T + b
    if apply_relu:
        y = np.maximum(y, 0)
    return y


def encoder_forward(x: np.ndarray, data: dict) -> np.ndarray:
    """Run encoder forward pass."""
    current = x

    for i, layer in enumerate(data['layers']):
        W, b = get_layer_weights(data, i)

        # Skip logvar for inference
        if layer['name'] == 'fc_logvar':
            break

        current = ternary_linear(current, W, b, layer['has_activation'])

        if layer['name'] == 'fc_mu':
            return current

    return current


def decoder_forward(z: np.ndarray, data: dict) -> np.ndarray:
    """Run decoder forward pass."""
    current = z

    for i, layer in enumerate(data['layers']):
        W, b = get_layer_weights(data, i)
        current = ternary_linear(current, W, b, layer['has_activation'])

    return current


# =============================================================================
# Original PyTorch Model (for comparison)
# =============================================================================

def quantize_to_ternary(weights: torch.Tensor, threshold: float) -> torch.Tensor:
    """Quantize weights to ternary {-1, 0, +1}."""
    sign = torch.sign(weights)
    mask = (torch.abs(weights) > threshold).float()
    return sign * mask


class OriginalEncoder(nn.Module):
    def __init__(self, state_dict, prefix, quantize=False, threshold=0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(9, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, 16)

        # Load weights (optionally quantized)
        for idx, layer_idx in enumerate([0, 2, 4]):
            w = state_dict[f'{prefix}.encoder.{layer_idx}.weight']
            b = state_dict[f'{prefix}.encoder.{layer_idx}.bias']
            if quantize:
                w = quantize_to_ternary(w, threshold)
            self.encoder[layer_idx].weight.data = w
            self.encoder[layer_idx].bias.data = b

        w_mu = state_dict[f'{prefix}.fc_mu.weight']
        if quantize:
            w_mu = quantize_to_ternary(w_mu, threshold)
        self.fc_mu.weight.data = w_mu
        self.fc_mu.bias.data = state_dict[f'{prefix}.fc_mu.bias']

    def forward(self, x):
        h = self.encoder(x)
        return self.fc_mu(h)


class OriginalDecoder(nn.Module):
    def __init__(self, state_dict, quantize=False, threshold=0.0):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 27)
        )

        for layer_idx in [0, 2, 4]:
            w = state_dict[f'decoder_A.decoder.{layer_idx}.weight']
            b = state_dict[f'decoder_A.decoder.{layer_idx}.bias']
            if quantize:
                w = quantize_to_ternary(w, threshold)
            self.decoder[layer_idx].weight.data = w
            self.decoder[layer_idx].bias.data = b

    def forward(self, z):
        return self.decoder(z)


# =============================================================================
# Main Test
# =============================================================================

def main():
    base_dir = Path(__file__).parent
    export_dir = base_dir / "gemm_export"
    checkpoint_path = base_dir / "v5_11_homeostasis" / "epoch_20.pt"

    print("="*70)
    print("VAE GEMM Inference Test")
    print("="*70)

    # Load config
    with open(export_dir / "vae_config.json") as f:
        config = json.load(f)

    print(f"Quantization threshold: {config['quantization_threshold']:.4f}")
    print(f"Weight distribution: {config['weight_distribution']}")

    # Load binary files
    print("\nLoading Dense243 binary files...")
    enc_a_data = load_layer_file(export_dir / "vae_encoder_A.bin")
    enc_b_data = load_layer_file(export_dir / "vae_encoder_B.bin")
    dec_data = load_layer_file(export_dir / "vae_decoder_A.bin")

    print(f"  Encoder A: {len(enc_a_data['layers'])} layers")
    print(f"  Encoder B: {len(enc_b_data['layers'])} layers")
    print(f"  Decoder: {len(dec_data['layers'])} layers")

    # Load original checkpoint
    print("\nLoading original PyTorch checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state']

    # Create quantized models (same quantization as export)
    threshold = config['quantization_threshold']
    quant_enc_a = OriginalEncoder(state_dict, 'encoder_A', quantize=True, threshold=threshold)
    quant_enc_b = OriginalEncoder(state_dict, 'encoder_B', quantize=True, threshold=threshold)
    quant_dec = OriginalDecoder(state_dict, quantize=True, threshold=threshold)

    # Also create original (float32) models for comparison
    orig_enc_a = OriginalEncoder(state_dict, 'encoder_A', quantize=False)
    orig_dec = OriginalDecoder(state_dict, quantize=False)

    # Test data: all 19683 ternary operations
    print("\nGenerating test data (all 19683 ternary operations)...")
    N = 19683
    test_inputs = []
    for i in range(N):
        trits = []
        val = i
        for _ in range(9):
            trits.append((val % 3) - 1)
            val //= 3
        test_inputs.append(trits)

    test_inputs = np.array(test_inputs, dtype=np.float32)
    test_inputs_torch = torch.from_numpy(test_inputs)

    # Run ternary GEMM inference
    print("\nRunning ternary GEMM inference...")
    t0 = time.perf_counter()
    mu_ternary = encoder_forward(test_inputs, enc_a_data)
    recon_ternary = decoder_forward(mu_ternary, dec_data)
    t_ternary = time.perf_counter() - t0
    print(f"  Time: {t_ternary*1000:.2f} ms")

    # Run quantized PyTorch inference (should match ternary GEMM exactly)
    print("Running quantized PyTorch inference...")
    with torch.no_grad():
        t0 = time.perf_counter()
        mu_quant = quant_enc_a(test_inputs_torch).numpy()
        recon_quant = quant_dec(torch.from_numpy(mu_quant)).numpy()
        t_quant = time.perf_counter() - t0
    print(f"  Time: {t_quant*1000:.2f} ms")

    # Run original (float32) PyTorch inference
    print("Running original (float32) PyTorch inference...")
    with torch.no_grad():
        t0 = time.perf_counter()
        mu_orig = orig_enc_a(test_inputs_torch).numpy()
        recon_orig = orig_dec(torch.from_numpy(mu_orig)).numpy()
        t_orig = time.perf_counter() - t0
    print(f"  Time: {t_orig*1000:.2f} ms")

    # Compare ternary GEMM vs quantized PyTorch (should be exact match)
    print("\n" + "-"*70)
    print("COMPARISON 1: Ternary GEMM vs Quantized PyTorch (expect exact match)")
    print("-"*70)

    mu_diff = np.abs(mu_ternary - mu_quant)
    print(f"\nLatent (mu) difference:")
    print(f"  Max: {mu_diff.max():.6f}")
    print(f"  Mean: {mu_diff.mean():.6f}")

    recon_diff = np.abs(recon_ternary - recon_quant)
    print(f"\nReconstruction difference:")
    print(f"  Max: {recon_diff.max():.6f}")
    print(f"  Mean: {recon_diff.mean():.6f}")

    argmax_ternary = recon_ternary.reshape(N, 9, 3).argmax(axis=-1)
    argmax_quant = recon_quant.reshape(N, 9, 3).argmax(axis=-1)
    agreement_quant = (argmax_ternary == argmax_quant).all(axis=1).mean()
    print(f"\nClassification agreement: {agreement_quant*100:.2f}%")

    # Compare quantized vs original (shows quantization impact)
    print("\n" + "-"*70)
    print("COMPARISON 2: Quantized vs Original (shows quantization impact)")
    print("-"*70)

    argmax_orig = recon_orig.reshape(N, 9, 3).argmax(axis=-1)
    agreement_orig = (argmax_quant == argmax_orig).all(axis=1).mean()
    print(f"Classification agreement: {agreement_orig*100:.2f}%")

    # Coverage (unique outputs)
    unique_ternary = len(set(tuple(r.flatten().tolist()) for r in argmax_ternary))
    unique_orig = len(set(tuple(r.flatten().tolist()) for r in argmax_orig))
    print(f"\nUnique outputs:")
    print(f"  Ternary GEMM: {unique_ternary}/{N} ({unique_ternary/N*100:.2f}%)")
    print(f"  Original: {unique_orig}/{N} ({unique_orig/N*100:.2f}%)")

    # Memory comparison
    print("\n" + "-"*70)
    print("MEMORY ANALYSIS")
    print("-"*70)

    orig_params = sum(p.numel() * 4 for p in orig_enc_a.parameters())  # float32
    orig_params += sum(p.numel() * 4 for p in orig_dec.parameters())
    ternary_bytes = config['total_bytes']['encoder_A'] + config['total_bytes']['decoder_A']

    print(f"Original model (float32): {orig_params:,} bytes")
    print(f"Ternary GEMM (Dense243): {ternary_bytes:,} bytes")
    print(f"Compression ratio: {orig_params/ternary_bytes:.2f}x")

    # Performance
    print("\n" + "-"*70)
    print("PERFORMANCE")
    print("-"*70)
    print(f"Original PyTorch: {t_orig*1000:.2f} ms")
    print(f"Ternary GEMM (Python): {t_ternary*1000:.2f} ms")
    print(f"Speedup: {t_orig/t_ternary:.2f}x (Python reference, C++ will be faster)")

    ops_per_inference = 2 * (9*256 + 256*128 + 128*64 + 64*16 + 16*32 + 32*64 + 64*27)
    total_ops = ops_per_inference * N
    gops = total_ops / t_ternary / 1e9
    print(f"Throughput: {gops:.2f} Gops/s (Python reference)")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

    # Determine pass/fail - GEMM should match quantized PyTorch exactly
    if agreement_quant >= 0.99:
        print("RESULT: PASS - Ternary GEMM matches quantized PyTorch")
        print(f"  GEMM vs Quantized: {agreement_quant*100:.1f}% agreement")
        print(f"  Quantized vs Original: {agreement_orig*100:.1f}% agreement")
    else:
        print(f"RESULT: FAIL - GEMM vs Quantized agreement {agreement_quant*100:.1f}% below 99%")


if __name__ == "__main__":
    main()
