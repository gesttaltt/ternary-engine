"""
test_hyperbolic.py - Quick smoke test for hyperbolic operations

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Sanity-checks PoincareOperations (geodesic midpoint vs. the discredited
Euclidean midpoint) and a freshly-initialized (untrained) HyperbolicOperationModel
forward pass. This is a smoke test, not a correctness fixture: an untrained
model's specific accuracy/VRC values aren't meaningful to assert on, so checks
here are structural (no crash, correct shapes, finite values, no NaN) rather
than value-outcome assertions.

Rewritten 2026-08-13: previously had zero assert statements and wrapped its
entire body in a bare try/except that only printed a traceback and always
exited 0 -- a regression (e.g. the valuation-table bug fixed the same day in
model.py/hyperbolic_ops.py/data.py) could break this script's own claims
with no indication beyond eyeballing printed numbers. Also added the missing
copyright header, `if __name__ == "__main__":` guard, and a fixed random
seed, per CLAUDE.md conventions.

USAGE: python test_hyperbolic.py
OUTPUT: Diagnostic printout of geodesic vs. Euclidean midpoint behavior and
an untrained HyperbolicOperationModel forward pass; exit code 0 on success,
1 on any assertion failure or exception.
"""
import sys
import time

import torch

from hyperbolic_ops import PoincareOperations, HyperbolicOperationModel


def main() -> int:
    torch.manual_seed(42)

    print("Starting test...")
    sys.stdout.flush()

    print("PyTorch imported")
    print("Hyperbolic ops imported")
    sys.stdout.flush()

    # Test Poincare operations
    x = torch.tensor([[0.3, 0.2, 0.1]])
    y = torch.tensor([[0.1, 0.4, 0.2]])

    print(f"\nx = {x}")
    print(f"y = {y}")

    # Euclidean midpoint (WRONG -- kept only as a contrast baseline)
    euc_mid = (x + y) / 2
    print(f"\nEuclidean midpoint: {euc_mid}")
    print(f"  ||euc_mid|| = {euc_mid.norm():.4f}")

    # Geodesic midpoint (CORRECT)
    hyp_mid = PoincareOperations.geodesic_midpoint(x, y)
    print(f"\nGeodesic midpoint: {hyp_mid}")
    print(f"  ||hyp_mid|| = {hyp_mid.norm():.4f}")
    assert torch.isfinite(hyp_mid).all(), "geodesic_midpoint produced non-finite values"

    # Distances
    d_xy = PoincareOperations.hyperbolic_distance(x, y)
    d_x_euc = PoincareOperations.hyperbolic_distance(x, euc_mid)
    d_y_euc = PoincareOperations.hyperbolic_distance(y, euc_mid)
    d_x_hyp = PoincareOperations.hyperbolic_distance(x, hyp_mid)
    d_y_hyp = PoincareOperations.hyperbolic_distance(y, hyp_mid)

    for name, d in [('d_xy', d_xy), ('d_x_euc', d_x_euc), ('d_y_euc', d_y_euc),
                     ('d_x_hyp', d_x_hyp), ('d_y_hyp', d_y_hyp)]:
        assert torch.isfinite(d).all(), f"hyperbolic_distance {name} is not finite: {d}"

    print(f"\nHyperbolic distances:")
    print(f"  d(x, y) = {d_xy.item():.4f}")
    print(f"  d(x, euc_mid) = {d_x_euc.item():.4f}")
    print(f"  d(y, euc_mid) = {d_y_euc.item():.4f}")
    print(f"  d(x, hyp_mid) = {d_x_hyp.item():.4f}")
    print(f"  d(y, hyp_mid) = {d_y_hyp.item():.4f}")

    geodesic_equidistant = abs(d_x_hyp - d_y_hyp).item() < 0.01
    print(f"\nGeodesic midpoint is equidistant: {geodesic_equidistant}")
    print(f"Euclidean midpoint is NOT equidistant: {abs(d_x_euc - d_y_euc).item():.4f} difference")
    # This is the specific structural claim this file exists to check
    # (CLAUDE.md's "Hyperbolic GEMM Research" geodesic-vs-Euclidean-midpoint
    # proof) -- worth asserting, unlike the untrained-model checks below.
    assert geodesic_equidistant, (
        f"geodesic_midpoint should be equidistant from x and y (within 0.01), "
        f"got d(x,mid)={d_x_hyp.item():.4f}, d(y,mid)={d_y_hyp.item():.4f}"
    )

    print("\n" + "="*60)
    print("Testing HyperbolicOperationModel...")

    model = HyperbolicOperationModel(
        num_trits=9,
        num_values=19683,
        latent_dim=16,
        num_operations=4
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    batch_size = 32
    idx_a = torch.randint(0, 19683, (batch_size,))
    idx_b = torch.randint(0, 19683, (batch_size,))
    op_id = torch.randint(0, 4, (batch_size,))
    idx_result = torch.randint(0, 19683, (batch_size,))

    print(f"\nRunning forward_operation with batch_size={batch_size}...")
    t0 = time.time()
    output = model.forward_operation(idx_a, idx_b, op_id, idx_result)
    t1 = time.time()
    print(f"Forward pass completed in {t1-t0:.2f}s")

    print(f"\nBatch size: {batch_size}")
    print(f"Predicted embedding shape: {output['predicted_emb'].shape}")
    print(f"Predicted indices shape: {output['predicted_idx'].shape}")
    print(f"Trajectory shape: {output['trajectory'].shape}")

    assert output['predicted_emb'].shape[0] == batch_size, \
        f"predicted_emb batch dim mismatch: {output['predicted_emb'].shape}"
    assert output['predicted_idx'].shape == (batch_size,), \
        f"predicted_idx shape mismatch: {output['predicted_idx'].shape}"
    assert torch.isfinite(output['predicted_emb']).all(), \
        "predicted_emb contains non-finite values"

    # Check predictions (informational only -- an untrained, randomly
    # initialized model has no reason to predict correctly)
    correct = (output['predicted_idx'] == idx_result).float().mean()
    print(f"Accuracy (untrained): {correct.item()*100:.1f}%")

    # Check radial structure
    print("\n" + "="*60)
    print("Testing radial p-adic structure...")

    # Sample values with different valuations: powers of 3 have valuations
    # 0,1,2,3,... by construction (v3(3^k) = k), the clearest possible
    # values to sanity-check the valuation table against. These are DECODED
    # balanced-ternary values, not raw corpus indices -- idx_offset must be
    # added to get the raw index model.valuations expects (idx = decoded +
    # idx_offset, idx_offset = (num_values-1)//2 = 9841 for num_values=19683
    # -- see the valuation-table fix in model.py/hyperbolic_ops.py/data.py).
    # Previously this used the powers of 3 directly AS raw indices (i.e.
    # assumed idx_offset=0), which happened to look sensible only because
    # the valuation table itself had the matching bug at the time. Found
    # and fixed together 2026-08-13.
    idx_offset = (model.valuations.shape[0] - 1) // 2
    powers_of_3 = [0, 1, 3, 9, 27, 81, 243, 729, 2187, 6561]
    sample_indices = torch.tensor([p + idx_offset for p in powers_of_3])
    embeddings = model.encode(sample_indices)
    radii = embeddings.norm(dim=-1)
    valuations = model.valuations[sample_indices]

    assert torch.isfinite(embeddings).all(), "model.encode() produced non-finite embeddings"
    assert torch.isfinite(radii).all(), "embedding radii are not finite"

    # Deterministic, training-independent check: v3(3^k) = k exactly, by
    # definition of 3-adic valuation (and v3(0) = max_val = num_trits, this
    # codebase's convention for the true-zero special case). Unlike the
    # model-behavior checks elsewhere in this file, this doesn't depend on
    # the model being trained -- it directly validates the valuation table
    # is wired up correctly against ground truth. powers_of_3[0] is 0
    # itself, not 3^0, hence the leading num_trits before the 0..8 run.
    expected_valuations = [model.num_trits] + list(range(len(powers_of_3) - 1))
    actual_valuations = valuations.tolist()
    assert actual_valuations == expected_valuations, (
        f"valuation table incorrect for powers of 3: expected {expected_valuations}, "
        f"got {actual_valuations}"
    )

    print(f"\nIndex | Valuation | Radius")
    print("-" * 30)
    for i, idx in enumerate(sample_indices):
        print(f"{idx.item():5d} | {valuations[i].item():9d} | {radii[i].item():.4f}")

    from scipy.stats import spearmanr
    vrc = spearmanr(valuations.numpy(), radii.detach().numpy())[0]
    print(f"\nValuation-Radius Correlation: {vrc:.4f}")
    print("(Target: strongly NEGATIVE - high valuation = small radius; not")
    print(" asserted here since this model is untrained and has no reason to")
    print(" show this correlation yet -- see train_hyperbolic.py.)")

    print("\nTest complete!")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)
