"""Quick test for hyperbolic operations."""
import sys
import time
print("Starting test...")
sys.stdout.flush()

try:
    import torch
    print("PyTorch imported")
    sys.stdout.flush()

    from hyperbolic_ops import PoincareOperations, HyperbolicOperationModel
    print("Hyperbolic ops imported")
    sys.stdout.flush()

    # Test Poincaré operations
    x = torch.tensor([[0.3, 0.2, 0.1]])
    y = torch.tensor([[0.1, 0.4, 0.2]])

    print(f"\nx = {x}")
    print(f"y = {y}")

    # Euclidean midpoint (WRONG)
    euc_mid = (x + y) / 2
    print(f"\nEuclidean midpoint: {euc_mid}")
    print(f"  ||euc_mid|| = {euc_mid.norm():.4f}")

    # Geodesic midpoint (CORRECT)
    hyp_mid = PoincareOperations.geodesic_midpoint(x, y)
    print(f"\nGeodesic midpoint: {hyp_mid}")
    print(f"  ||hyp_mid|| = {hyp_mid.norm():.4f}")

    # Distances
    d_xy = PoincareOperations.hyperbolic_distance(x, y)
    d_x_euc = PoincareOperations.hyperbolic_distance(x, euc_mid)
    d_y_euc = PoincareOperations.hyperbolic_distance(y, euc_mid)
    d_x_hyp = PoincareOperations.hyperbolic_distance(x, hyp_mid)
    d_y_hyp = PoincareOperations.hyperbolic_distance(y, hyp_mid)

    print(f"\nHyperbolic distances:")
    print(f"  d(x, y) = {d_xy.item():.4f}")
    print(f"  d(x, euc_mid) = {d_x_euc.item():.4f}")
    print(f"  d(y, euc_mid) = {d_y_euc.item():.4f}")
    print(f"  d(x, hyp_mid) = {d_x_hyp.item():.4f}")
    print(f"  d(y, hyp_mid) = {d_y_hyp.item():.4f}")

    print(f"\nGeodesic midpoint is equidistant: {abs(d_x_hyp - d_y_hyp).item() < 0.01}")
    print(f"Euclidean midpoint is NOT equidistant: {abs(d_x_euc - d_y_euc).item():.4f} difference")

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

    # Check predictions
    correct = (output['predicted_idx'] == idx_result).float().mean()
    print(f"Accuracy (untrained): {correct.item()*100:.1f}%")

    # Check radial structure
    print("\n" + "="*60)
    print("Testing radial p-adic structure...")

    # Sample values with different valuations
    sample_indices = [0, 1, 3, 9, 27, 81, 243, 729, 2187, 6561]  # Powers of 3
    sample_indices = torch.tensor(sample_indices)
    embeddings = model.encode(sample_indices)
    radii = embeddings.norm(dim=-1)
    valuations = model.valuations[sample_indices]

    print(f"\nIndex | Valuation | Radius")
    print("-" * 30)
    for i, idx in enumerate(sample_indices):
        print(f"{idx.item():5d} | {valuations[i].item():9d} | {radii[i].item():.4f}")

    from scipy.stats import spearmanr
    vrc = spearmanr(valuations.numpy(), radii.detach().numpy())[0]
    print(f"\nValuation-Radius Correlation: {vrc:.4f}")
    print("(Target: strongly NEGATIVE - high valuation = small radius)")

    print("\nTest complete!")

except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
