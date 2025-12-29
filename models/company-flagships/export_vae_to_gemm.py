#!/usr/bin/env python3
"""
Export VAE Encoder Weights to Dense243 Binary Format for C++ GEMM

Converts the best dual-VAE checkpoint to Dense243-packed binary files
that can be loaded directly by C++ GEMM kernels.

Output files:
  - vae_encoder_A.bin: Encoder A weights (all layers concatenated)
  - vae_encoder_B.bin: Encoder B weights (all layers concatenated)
  - vae_decoder_A.bin: Decoder A weights
  - vae_projection.bin: Projection head weights (float32, small)
  - vae_config.json: Layer dimensions and offsets

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0
"""

import os
import sys
import json
import struct
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional


@dataclass
class LayerConfig:
    """Configuration for a single linear layer."""
    name: str
    in_features: int
    out_features: int
    weight_offset: int  # Byte offset in binary file
    weight_size: int    # Size in bytes (Dense243 packed)
    bias_offset: int    # Byte offset for bias (float32)
    bias_size: int      # Size in bytes
    has_activation: bool = True  # ReLU after layer


@dataclass
class VAEConfig:
    """Complete VAE configuration for C++ loader."""
    input_dim: int
    latent_dim: int
    encoder_layers: List[LayerConfig]
    decoder_layers: List[LayerConfig]
    total_weight_bytes: int
    total_bias_bytes: int
    quantization_threshold: float


def quantize_to_ternary(weights: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Quantize weights to ternary {-1, 0, +1} using threshold."""
    sign = torch.sign(weights)
    mask = (torch.abs(weights) > threshold).float()
    return sign * mask


def find_optimal_threshold(weights: torch.Tensor, target_sparsity: float = 0.3) -> float:
    """Find threshold that achieves target sparsity (% of zeros)."""
    abs_weights = torch.abs(weights).flatten()
    sorted_weights, _ = torch.sort(abs_weights)
    idx = int(len(sorted_weights) * target_sparsity)
    return float(sorted_weights[idx])


def pack_ternary_to_dense243(weights: np.ndarray) -> np.ndarray:
    """
    Pack ternary weights {-1, 0, +1} to Dense243 format (5 trits/byte).

    Args:
        weights: Ternary weight array [out_features, in_features]
                 Values must be in {-1, 0, +1}

    Returns:
        Dense243-packed weights [out_features, ceil(in_features/5)] as uint8
    """
    out_features, in_features = weights.shape

    # Pad to multiple of 5
    pad_size = (5 - (in_features % 5)) % 5
    if pad_size > 0:
        weights = np.pad(weights, ((0, 0), (0, pad_size)), constant_values=0)
        in_features += pad_size

    # Ensure ternary values
    weights = np.clip(weights, -1, 1).astype(np.int8)

    # Pack 5 trits per byte
    in_features_packed = in_features // 5
    packed = np.zeros((out_features, in_features_packed), dtype=np.uint8)

    for i in range(in_features_packed):
        trits = weights[:, i*5:(i+1)*5]
        trits_mapped = trits + 1  # {-1,0,+1} -> {0,1,2}

        packed[:, i] = (
            trits_mapped[:, 0] +
            trits_mapped[:, 1] * 3 +
            trits_mapped[:, 2] * 9 +
            trits_mapped[:, 3] * 27 +
            trits_mapped[:, 4] * 81
        ).astype(np.uint8)

    return packed


def export_encoder_weights(
    state_dict: Dict,
    encoder_prefix: str,
    threshold: float,
    output_path: Path
) -> Tuple[List[LayerConfig], int]:
    """Export encoder weights to Dense243 binary file."""

    layers = []
    weight_offset = 0
    bias_offset = 0

    # Encoder layers: encoder.0, encoder.2, encoder.4 (Linear + ReLU)
    layer_indices = [0, 2, 4]

    all_weights = []
    all_biases = []

    for idx in layer_indices:
        weight_key = f"{encoder_prefix}.encoder.{idx}.weight"
        bias_key = f"{encoder_prefix}.encoder.{idx}.bias"

        if weight_key not in state_dict:
            continue

        weight = state_dict[weight_key]  # [out, in]
        bias = state_dict[bias_key]      # [out]

        out_features, in_features = weight.shape

        # Quantize weights to ternary
        ternary_weight = quantize_to_ternary(weight, threshold)
        weight_np = ternary_weight.detach().cpu().numpy().astype(np.int8)

        # Pack to Dense243
        packed = pack_ternary_to_dense243(weight_np)
        packed_size = packed.nbytes

        # Bias as float32
        bias_np = bias.detach().cpu().numpy().astype(np.float32)
        bias_size = bias_np.nbytes

        layers.append(LayerConfig(
            name=f"encoder.{idx}",
            in_features=in_features,
            out_features=out_features,
            weight_offset=weight_offset,
            weight_size=packed_size,
            bias_offset=bias_offset,
            bias_size=bias_size,
            has_activation=True
        ))

        all_weights.append(packed.tobytes())
        all_biases.append(bias_np.tobytes())

        weight_offset += packed_size
        bias_offset += bias_size

    # fc_mu and fc_logvar (VAE latent projection)
    for fc_name in ['fc_mu', 'fc_logvar']:
        weight_key = f"{encoder_prefix}.{fc_name}.weight"
        bias_key = f"{encoder_prefix}.{fc_name}.bias"

        weight = state_dict[weight_key]
        bias = state_dict[bias_key]

        out_features, in_features = weight.shape

        ternary_weight = quantize_to_ternary(weight, threshold)
        weight_np = ternary_weight.detach().cpu().numpy().astype(np.int8)
        packed = pack_ternary_to_dense243(weight_np)
        packed_size = packed.nbytes

        bias_np = bias.detach().cpu().numpy().astype(np.float32)
        bias_size = bias_np.nbytes

        layers.append(LayerConfig(
            name=fc_name,
            in_features=in_features,
            out_features=out_features,
            weight_offset=weight_offset,
            weight_size=packed_size,
            bias_offset=bias_offset,
            bias_size=bias_size,
            has_activation=False  # No activation on latent projection
        ))

        all_weights.append(packed.tobytes())
        all_biases.append(bias_np.tobytes())

        weight_offset += packed_size
        bias_offset += bias_size

    # Write binary file
    with open(output_path, 'wb') as f:
        # Header: magic + version + num_layers + total_weight_bytes + total_bias_bytes
        f.write(struct.pack('<4s', b'TVAE'))  # Magic
        f.write(struct.pack('<I', 1))          # Version
        f.write(struct.pack('<I', len(layers)))
        f.write(struct.pack('<I', weight_offset))
        f.write(struct.pack('<I', bias_offset))

        # Layer configs
        for layer in layers:
            f.write(struct.pack('<32s', layer.name.encode('utf-8').ljust(32, b'\x00')))
            f.write(struct.pack('<IIIIII?',
                layer.in_features,
                layer.out_features,
                layer.weight_offset,
                layer.weight_size,
                layer.bias_offset,
                layer.bias_size,
                layer.has_activation
            ))

        # Weight data
        for w in all_weights:
            f.write(w)

        # Bias data
        for b in all_biases:
            f.write(b)

    return layers, weight_offset + bias_offset


def export_decoder_weights(
    state_dict: Dict,
    threshold: float,
    output_path: Path
) -> Tuple[List[LayerConfig], int]:
    """Export decoder weights to binary file."""

    layers = []
    weight_offset = 0
    bias_offset = 0

    all_weights = []
    all_biases = []

    # Decoder layers: decoder.0, decoder.2, decoder.4
    for idx in [0, 2, 4]:
        weight_key = f"decoder_A.decoder.{idx}.weight"
        bias_key = f"decoder_A.decoder.{idx}.bias"

        if weight_key not in state_dict:
            continue

        weight = state_dict[weight_key]
        bias = state_dict[bias_key]

        out_features, in_features = weight.shape

        # Quantize to ternary
        ternary_weight = quantize_to_ternary(weight, threshold)
        weight_np = ternary_weight.detach().cpu().numpy().astype(np.int8)
        packed = pack_ternary_to_dense243(weight_np)
        packed_size = packed.nbytes

        bias_np = bias.detach().cpu().numpy().astype(np.float32)
        bias_size = bias_np.nbytes

        # Last layer (idx=4) outputs logits, no activation
        has_activation = (idx != 4)

        layers.append(LayerConfig(
            name=f"decoder.{idx}",
            in_features=in_features,
            out_features=out_features,
            weight_offset=weight_offset,
            weight_size=packed_size,
            bias_offset=bias_offset,
            bias_size=bias_size,
            has_activation=has_activation
        ))

        all_weights.append(packed.tobytes())
        all_biases.append(bias_np.tobytes())

        weight_offset += packed_size
        bias_offset += bias_size

    # Write binary file
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<4s', b'TDEC'))
        f.write(struct.pack('<I', 1))
        f.write(struct.pack('<I', len(layers)))
        f.write(struct.pack('<I', weight_offset))
        f.write(struct.pack('<I', bias_offset))

        for layer in layers:
            f.write(struct.pack('<32s', layer.name.encode('utf-8').ljust(32, b'\x00')))
            f.write(struct.pack('<IIIIII?',
                layer.in_features,
                layer.out_features,
                layer.weight_offset,
                layer.weight_size,
                layer.bias_offset,
                layer.bias_size,
                layer.has_activation
            ))

        for w in all_weights:
            f.write(w)
        for b in all_biases:
            f.write(b)

    return layers, weight_offset + bias_offset


def export_projection_weights(
    state_dict: Dict,
    output_path: Path
) -> int:
    """Export projection head weights as float32 (small, keep precision)."""

    proj_keys = [k for k in state_dict.keys() if 'projection' in k]

    with open(output_path, 'wb') as f:
        f.write(struct.pack('<4s', b'TPRJ'))
        f.write(struct.pack('<I', 1))
        f.write(struct.pack('<I', len(proj_keys)))

        # Write all projection weights as float32
        total_bytes = 0
        for key in sorted(proj_keys):
            tensor = state_dict[key]
            data = tensor.detach().cpu().numpy().astype(np.float32).tobytes()

            # Key name + shape + data
            f.write(struct.pack('<64s', key.encode('utf-8').ljust(64, b'\x00')))
            f.write(struct.pack('<I', len(tensor.shape)))
            for dim in tensor.shape:
                f.write(struct.pack('<I', dim))
            f.write(struct.pack('<I', len(data)))
            f.write(data)

            total_bytes += len(data)

    return total_bytes


def main():
    """Export best VAE checkpoint to Dense243 binary format."""

    base_dir = Path(__file__).parent
    output_dir = base_dir / "gemm_export"
    output_dir.mkdir(exist_ok=True)

    # Best checkpoint from validation
    checkpoint_path = base_dir / "v5_11_homeostasis" / "epoch_20.pt"

    print("="*70)
    print("VAE to Dense243 GEMM Export")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {output_dir}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state']

    # Find optimal quantization threshold (target 30% sparsity)
    all_weights = []
    for k, v in state_dict.items():
        if 'weight' in k and 'norm' not in k.lower():
            all_weights.append(v.flatten())

    all_weights_tensor = torch.cat(all_weights)
    threshold = find_optimal_threshold(all_weights_tensor, target_sparsity=0.3)
    print(f"Quantization threshold: {threshold:.4f}")

    # Calculate sparsity and ternary distribution
    ternary = quantize_to_ternary(all_weights_tensor, threshold)
    sparsity = (ternary == 0).float().mean().item()
    neg_ratio = (ternary == -1).float().mean().item()
    pos_ratio = (ternary == 1).float().mean().item()
    print(f"Weight distribution: -1={neg_ratio:.1%}, 0={sparsity:.1%}, +1={pos_ratio:.1%}")

    # Export encoder A
    print("\nExporting encoder A...")
    enc_a_layers, enc_a_bytes = export_encoder_weights(
        state_dict, "encoder_A", threshold,
        output_dir / "vae_encoder_A.bin"
    )
    print(f"  Layers: {len(enc_a_layers)}, Size: {enc_a_bytes:,} bytes")

    # Export encoder B
    print("Exporting encoder B...")
    enc_b_layers, enc_b_bytes = export_encoder_weights(
        state_dict, "encoder_B", threshold,
        output_dir / "vae_encoder_B.bin"
    )
    print(f"  Layers: {len(enc_b_layers)}, Size: {enc_b_bytes:,} bytes")

    # Export decoder A
    print("Exporting decoder A...")
    dec_layers, dec_bytes = export_decoder_weights(
        state_dict, threshold,
        output_dir / "vae_decoder_A.bin"
    )
    print(f"  Layers: {len(dec_layers)}, Size: {dec_bytes:,} bytes")

    # Export projection (float32)
    print("Exporting projection (float32)...")
    proj_bytes = export_projection_weights(
        state_dict,
        output_dir / "vae_projection.bin"
    )
    print(f"  Size: {proj_bytes:,} bytes")

    # Save config JSON
    config = {
        "source_checkpoint": str(checkpoint_path),
        "quantization_threshold": threshold,
        "input_dim": 9,
        "latent_dim": 16,
        "output_dim": 27,
        "encoder_A": [asdict(l) for l in enc_a_layers],
        "encoder_B": [asdict(l) for l in enc_b_layers],
        "decoder_A": [asdict(l) for l in dec_layers],
        "total_bytes": {
            "encoder_A": enc_a_bytes,
            "encoder_B": enc_b_bytes,
            "decoder_A": dec_bytes,
            "projection": proj_bytes,
            "total": enc_a_bytes + enc_b_bytes + dec_bytes + proj_bytes
        },
        "weight_distribution": {
            "minus_one": neg_ratio,
            "zero": sparsity,
            "plus_one": pos_ratio
        }
    }

    with open(output_dir / "vae_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "="*70)
    print("Export Summary")
    print("="*70)
    print(f"Total size: {config['total_bytes']['total']:,} bytes")
    print(f"  Encoder A: {enc_a_bytes:,} bytes ({len(enc_a_layers)} layers)")
    print(f"  Encoder B: {enc_b_bytes:,} bytes ({len(enc_b_layers)} layers)")
    print(f"  Decoder A: {dec_bytes:,} bytes ({len(dec_layers)} layers)")
    print(f"  Projection: {proj_bytes:,} bytes (float32)")
    print(f"\nOutput files:")
    for f in output_dir.glob("*"):
        print(f"  {f.name}: {f.stat().st_size:,} bytes")

    print(f"\nConfig saved to: {output_dir / 'vae_config.json'}")
    print("="*70)


if __name__ == "__main__":
    main()
