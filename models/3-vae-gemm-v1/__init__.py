"""
3-vae-gemm-v1: Operation-Aware Ternary VAE for GEMM Discovery

A VAE that learns both p-adic structure AND operation semantics,
enabling neural GEMM algorithm discovery via embedding space traversal.

Key Innovation:
    embed(a ⊕ b) ≈ midpoint(embed(a), embed(b))

Training Objective:
    EES >= 0.95 with ORA >= 95% and natural ultrametric emergence

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0
"""

from .model import VAEGemmV1, VAEGemmV1Config
from .loss import EESLoss, ORALoss, UltrametricLoss
from .data import OperationLUTDataset, create_operation_luts

__all__ = [
    'VAEGemmV1',
    'VAEGemmV1Config',
    'EESLoss',
    'ORALoss',
    'UltrametricLoss',
    'OperationLUTDataset',
    'create_operation_luts',
]
