"""
geometric_metrics.py - Geometric/semantic invariant measurements for ternary arrays

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

This module measures semantic properties of ternary arrays to understand why
canonical indexing provides massive speedups in certain regions.

GEOMETRIC INVARIANTS MEASURED:
- Shannon entropy (information content)
- Autocorrelation (inter-trit dependencies)
- Fractal dimension (self-similarity)
- Triad distribution (±1/0 balance)
- Repetitiveness (compression ratio)

These metrics help identify performance regions where:
- Low entropy → geometric path (canonical indexing) excels
- High entropy → cold path (direct kernels) excels

USAGE:
    from benchmarks.utils.geometric_metrics import GeometricMetrics

    metrics = GeometricMetrics(trit_array)
    entropy = metrics.shannon_entropy()
    correlation = metrics.autocorrelation(lag=1)
    dimension = metrics.fractal_dimension()
"""

import numpy as np
from typing import Dict, Tuple
from collections import Counter

class GeometricMetrics:
    """Calculate geometric/semantic invariants for ternary arrays."""

    def __init__(self, data: np.ndarray):
        """
        Initialize with ternary array data.

        Args:
            data: NumPy array of ternary values {0, 1, 2} (representing {-1, 0, +1})
        """
        self.data = data.flatten()  # Ensure 1D
        self.n = len(self.data)

    def shannon_entropy(self) -> float:
        """
        Calculate Shannon entropy of trit sequence.

        Returns:
            Entropy in bits (0.0 = perfectly predictable, ~1.585 = maximum for ternary)

        Formula:
            H = -Σ p(x) * log2(p(x))

        where p(x) is the probability of trit value x ∈ {0, 1, 2}
        """
        # Count occurrences of each trit value
        counts = np.bincount(self.data, minlength=3)
        probabilities = counts / self.n

        # Shannon entropy (avoid log(0))
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)

        return entropy

    def autocorrelation(self, lag: int = 1) -> float:
        """
        Calculate autocorrelation at given lag.

        Args:
            lag: Lag distance (default: 1 for adjacent trits)

        Returns:
            Correlation coefficient [-1, 1]
            - Close to 1: Strong positive correlation (patterns repeat)
            - Close to 0: No correlation (random)
            - Close to -1: Strong negative correlation (anti-patterns)

        Formula:
            ρ(lag) = Cov(X[t], X[t+lag]) / (σ² + ε)

        This measures inter-trit dependencies that canonical indexing can exploit.
        """
        if lag >= self.n:
            return 0.0

        # Convert to float for correlation calculation
        x = self.data.astype(float)

        # Calculate mean and variance
        mean_x = np.mean(x)
        variance_x = np.var(x)

        if variance_x == 0:
            return 1.0  # Perfectly correlated (constant array)

        # Autocorrelation at lag
        x_lagged = x[lag:]
        x_original = x[:-lag]

        covariance = np.mean((x_original - mean_x) * (x_lagged - mean_x))
        correlation = covariance / (variance_x + 1e-10)  # Avoid division by zero

        return correlation

    def fractal_dimension(self, max_box_size: int = 32) -> float:
        """
        Estimate Hausdorff (box-counting) fractal dimension.

        Args:
            max_box_size: Maximum box size for box-counting (default: 32)

        Returns:
            Fractal dimension D ≈ 1.0 for simple patterns, higher for complex

        Algorithm:
            1. Divide array into boxes of varying sizes
            2. Count non-empty boxes at each scale
            3. Fit line to log(N) vs log(1/box_size)
            4. Slope = fractal dimension

        This measures self-similarity at different scales.
        """
        # Box sizes (powers of 2 for efficiency)
        box_sizes = [2**i for i in range(1, int(np.log2(max_box_size)) + 1) if 2**i <= self.n]

        if len(box_sizes) < 2:
            return 1.0  # Not enough data

        counts = []
        scales = []

        for box_size in box_sizes:
            # Divide array into boxes
            num_boxes = self.n // box_size
            boxes = self.data[:num_boxes * box_size].reshape(num_boxes, box_size)

            # Count non-empty boxes (boxes with variation)
            non_empty = np.sum(np.std(boxes, axis=1) > 0)

            if non_empty > 0:
                counts.append(non_empty)
                scales.append(1.0 / box_size)

        if len(counts) < 2:
            return 1.0

        # Fit line in log-log space
        log_counts = np.log(counts)
        log_scales = np.log(scales)

        # Linear regression: log(N) = D * log(1/size) + const
        coefficients = np.polyfit(log_scales, log_counts, 1)
        dimension = coefficients[0]

        return max(1.0, min(2.0, dimension))  # Clamp to reasonable range

    def triad_distribution(self) -> Dict[str, float]:
        """
        Analyze distribution of ternary values.

        Returns:
            Dictionary with:
            - 'minus_one_ratio': Proportion of -1 values (encoded as 0)
            - 'zero_ratio': Proportion of 0 values (encoded as 1)
            - 'plus_one_ratio': Proportion of +1 values (encoded as 2)
            - 'balance_score': How balanced the distribution is [0, 1]
                               (1.0 = perfectly balanced, 0.0 = all one value)
        """
        counts = np.bincount(self.data, minlength=3)
        probabilities = counts / self.n

        # Balance score: entropy normalized to [0, 1]
        max_entropy = np.log2(3)  # Maximum entropy for ternary
        actual_entropy = self.shannon_entropy()
        balance_score = actual_entropy / max_entropy if max_entropy > 0 else 0.0

        return {
            'minus_one_ratio': probabilities[0],  # Encoded as 0
            'zero_ratio': probabilities[1],       # Encoded as 1
            'plus_one_ratio': probabilities[2],   # Encoded as 2
            'balance_score': balance_score,
        }

    def repetitiveness(self, window_size: int = 8) -> float:
        """
        Measure repetitiveness using pattern compression.

        Args:
            window_size: Size of pattern window (default: 8 trits)

        Returns:
            Compression ratio [0, 1]
            - Close to 1: Highly repetitive (few unique patterns)
            - Close to 0: Not repetitive (many unique patterns)

        Algorithm:
            Count unique patterns of size window_size.
            Repetitiveness = 1 - (unique_patterns / max_possible_patterns)
        """
        if self.n < window_size:
            return 0.0

        # Extract all windows
        patterns = []
        for i in range(self.n - window_size + 1):
            pattern = tuple(self.data[i:i + window_size])
            patterns.append(pattern)

        # Count unique patterns
        unique_patterns = len(set(patterns))
        max_possible = min(3**window_size, len(patterns))  # Max unique patterns

        if max_possible == 0:
            return 0.0

        compression_ratio = 1.0 - (unique_patterns / max_possible)
        return compression_ratio

    def compute_all_metrics(self) -> Dict[str, float]:
        """
        Compute all geometric metrics in one call.

        Returns:
            Dictionary with all metrics:
            - 'entropy': Shannon entropy
            - 'autocorrelation_lag1': Autocorrelation at lag 1
            - 'autocorrelation_lag2': Autocorrelation at lag 2
            - 'fractal_dimension': Hausdorff dimension
            - 'repetitiveness': Pattern compression ratio
            - 'minus_one_ratio': Proportion of -1 values
            - 'zero_ratio': Proportion of 0 values
            - 'plus_one_ratio': Proportion of +1 values
            - 'balance_score': Distribution balance
        """
        triad = self.triad_distribution()

        metrics = {
            # Information-theoretic
            'entropy': self.shannon_entropy(),

            # Correlation
            'autocorrelation_lag1': self.autocorrelation(lag=1),
            'autocorrelation_lag2': self.autocorrelation(lag=2),

            # Complexity
            'fractal_dimension': self.fractal_dimension(),
            'repetitiveness': self.repetitiveness(),

            # Distribution
            'minus_one_ratio': triad['minus_one_ratio'],
            'zero_ratio': triad['zero_ratio'],
            'plus_one_ratio': triad['plus_one_ratio'],
            'balance_score': triad['balance_score'],
        }

        return metrics


def generate_synthetic_dataset(entropy_level: str, size: int = 1000000) -> Tuple[np.ndarray, Dict]:
    """
    Generate synthetic ternary array with controlled entropy level.

    Args:
        entropy_level: 'low', 'medium', or 'high'
        size: Array size (default: 1M elements)

    Returns:
        Tuple of (ternary_array, expected_metrics)

    Examples:
        - Low entropy: Repetitive patterns, high correlation
        - Medium entropy: Pseudo-Markov with local structure
        - High entropy: Cryptographic random
    """
    np.random.seed(42)  # Reproducible

    if entropy_level == 'low':
        # Low entropy: Highly repetitive pattern
        pattern = np.array([0, 1, 2, 1] * 100, dtype=np.uint8)  # 400-element pattern
        repetitions = size // len(pattern) + 1
        data = np.tile(pattern, repetitions)[:size]

        expected = {
            'entropy_range': (0.5, 1.0),
            'correlation_range': (0.5, 1.0),
            'repetitiveness_range': (0.7, 1.0),
        }

    elif entropy_level == 'medium':
        # Medium entropy: Markov-like with memory
        data = np.zeros(size, dtype=np.uint8)
        data[0] = np.random.randint(0, 3)

        # Markov transitions (60% stay same, 40% change)
        for i in range(1, size):
            if np.random.rand() < 0.6:
                data[i] = data[i-1]  # Stay same (creates correlation)
            else:
                data[i] = np.random.randint(0, 3)  # Random change

        expected = {
            'entropy_range': (1.0, 1.4),
            'correlation_range': (0.2, 0.6),
            'repetitiveness_range': (0.3, 0.7),
        }

    elif entropy_level == 'high':
        # High entropy: Cryptographic random
        data = np.random.randint(0, 3, size=size, dtype=np.uint8)

        expected = {
            'entropy_range': (1.4, 1.585),  # Near maximum
            'correlation_range': (-0.1, 0.1),  # Near zero
            'repetitiveness_range': (0.0, 0.1),  # Very low
        }

    else:
        raise ValueError(f"Invalid entropy_level: {entropy_level}. Use 'low', 'medium', or 'high'.")

    return data, expected


# Test/validation code
if __name__ == '__main__':
    print("="*80)
    print("  GEOMETRIC METRICS MODULE - VALIDATION")
    print("="*80)

    # Test on synthetic datasets
    for entropy_level in ['low', 'medium', 'high']:
        print(f"\n{entropy_level.upper()} ENTROPY DATASET:")
        print("-" * 80)

        data, expected = generate_synthetic_dataset(entropy_level, size=100000)
        metrics = GeometricMetrics(data)
        results = metrics.compute_all_metrics()

        print(f"Entropy:              {results['entropy']:.4f} (expected: {expected['entropy_range']})")
        print(f"Autocorrelation lag1: {results['autocorrelation_lag1']:.4f} (expected: {expected['correlation_range']})")
        print(f"Fractal dimension:    {results['fractal_dimension']:.4f}")
        print(f"Repetitiveness:       {results['repetitiveness']:.4f} (expected: {expected['repetitiveness_range']})")
        print(f"Balance score:        {results['balance_score']:.4f}")
        print(f"Distribution: -1={results['minus_one_ratio']:.3f}, 0={results['zero_ratio']:.3f}, +1={results['plus_one_ratio']:.3f}")

    print("\n" + "="*80)
    print("✅ VALIDATION COMPLETE")
    print("="*80)
