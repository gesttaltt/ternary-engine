#!/usr/bin/env python3
"""Quick test of extended exploration pipeline."""

import time
import json
import numpy as np
from pathlib import Path
from explore_gemm_extended import (
    ExtendedGEMMExplorer,
    explore_operations_extended,
    analyze_multiscale_lattice,
    analyze_ultrametric_extended,
    generate_extended_gemm_map,
    analyze_gap_structure
)

def main():
    print("=" * 70)
    print("QUICK PIPELINE TEST")
    print("=" * 70)

    # Load embeddings
    base_dir = Path(__file__).parent
    print("\nLoading embeddings...")
    embeddings = np.load(base_dir / "gemm_export" / "ternary_embeddings.npy")
    print(f"  Shape: {embeddings.shape}")

    explorer = ExtendedGEMMExplorer(embeddings)
    print(f"  Explorer ready: N={explorer.N}, dim={explorer.dim}")

    # Test 1: Operations (1K samples)
    print("\n1. Testing operations (1K samples)...")
    t0 = time.time()
    op_results = explore_operations_extended(explorer, ['add', 'mul'], num_samples=1000)
    print(f"   Time: {time.time()-t0:.1f}s")
    print(f"   ADD ultrametric: {op_results['add']['ultrametric_rate']:.2%}")
    print(f"   MUL ultrametric: {op_results['mul']['ultrametric_rate']:.2%}")

    # Test 2: Lattice (500 samples)
    print("\n2. Testing lattice analysis...")
    t0 = time.time()
    lattice = analyze_multiscale_lattice(explorer, num_samples=500)
    print(f"   Time: {time.time()-t0:.1f}s")
    print(f"   Mean distance: {lattice['distance_stats']['mean']:.4f}")

    # Test 3: Ultrametric (1K triplets)
    print("\n3. Testing ultrametric analysis...")
    t0 = time.time()
    ultra = analyze_ultrametric_extended(explorer, num_triplets=1000)
    print(f"   Time: {time.time()-t0:.1f}s")
    print(f"   Correlation: {ultra['euclidean_ultrametric_correlation']:.4f}")

    # Test 4: GEMM map (50x50)
    print("\n4. Testing GEMM map generation (50x50)...")
    t0 = time.time()
    gemm_map = generate_extended_gemm_map(explorer, resolution=50, operation='add')
    print(f"   Time: {time.time()-t0:.1f}s")
    print(f"   Match rate: {gemm_map['match_rate']:.2%}")
    print(f"   Mean distance: {gemm_map['mean_distance']:.4f}")

    # Test 5: Gap analysis
    print("\n5. Testing gap analysis...")
    t0 = time.time()
    gap = analyze_gap_structure({'add': gemm_map['distance_map']})
    print(f"   Time: {time.time()-t0:.1f}s")
    print(f"   Gap regions: {gap['add']['num_gap_regions']}")

    print("\n" + "=" * 70)
    print("ALL PIPELINE TESTS PASSED")
    print("=" * 70)

    # Save mini results
    output_dir = base_dir / "gemm_exploration"
    mini_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_mode': True,
        'operations': {op: {
            'ultrametric_rate': float(r['ultrametric_rate']),
            'mean_dist': float(r['mean_dist_to_discrete'])
        } for op, r in op_results.items()},
        'lattice_mean_dist': float(lattice['distance_stats']['mean']),
        'ultrametric_corr': float(ultra['euclidean_ultrametric_correlation']),
        'gemm_match_rate': float(gemm_map['match_rate']),
        'gap_regions': int(gap['add']['num_gap_regions'])
    }

    with open(output_dir / 'mini_test_results.json', 'w') as f:
        json.dump(mini_results, f, indent=2)

    print(f"\nSaved mini results to {output_dir / 'mini_test_results.json'}")

if __name__ == "__main__":
    main()
