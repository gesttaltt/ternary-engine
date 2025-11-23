"""
Model Quantization Benchmark - Phase 5

Quantize real neural network models to ternary and measure:
- Accuracy degradation
- Inference speed
- Memory footprint
- Token generation throughput

This is the PROOF - if ternary-quantized models maintain reasonable
accuracy and run faster, we have a commercial product.

Usage:
    python bench_model_quantization.py --model TinyLlama-1.1B
    python bench_model_quantization.py --model all
"""

import numpy as np
import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import sys

# Check for optional dependencies
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available - model quantization disabled")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available - using mock implementation")


class TernaryQuantizer:
    """
    Quantize neural network weights to ternary {-1, 0, +1}

    Strategies:
    1. Threshold-based (simple)
    2. Learned thresholds (better accuracy)
    3. Per-layer thresholds (adaptive)
    """

    def __init__(self, strategy: str = "threshold"):
        """
        Initialize quantizer

        Args:
            strategy: "threshold", "learned", or "adaptive"
        """
        self.strategy = strategy

    def quantize_tensor(
        self,
        tensor: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Quantize a tensor to ternary values

        Args:
            tensor: Input tensor (numpy array)
            threshold: Optional threshold value. If None, use tensor statistics

        Returns:
            (quantized_tensor, threshold_used)
        """
        if threshold is None:
            if self.strategy == "threshold":
                # Simple: use mean absolute value
                threshold = np.abs(tensor).mean()
            elif self.strategy == "learned":
                # Better: use percentile
                threshold = np.percentile(np.abs(tensor), 75)
            elif self.strategy == "adaptive":
                # Adaptive: use standard deviation
                threshold = 0.5 * np.std(tensor)
            else:
                threshold = np.abs(tensor).mean()

        # Quantize
        quantized = np.zeros_like(tensor, dtype=np.int8)
        quantized[tensor > threshold] = 1
        quantized[tensor < -threshold] = -1
        # Values in [-threshold, threshold] remain 0

        return quantized, threshold

    def quantize_model_weights(self, model, verbose: bool = True) -> Dict[str, Any]:
        """
        Quantize all weights in a model to ternary

        Args:
            model: PyTorch model
            verbose: Print quantization statistics

        Returns:
            Dictionary with quantization statistics
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for model quantization")

        stats = {
            'layers': [],
            'total_params': 0,
            'quantized_params': 0,
            'sparsity': []
        }

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                original_weight = module.weight.data.cpu().numpy()
                quantized_weight, threshold = self.quantize_tensor(original_weight)

                # Calculate statistics
                neg_count = (quantized_weight == -1).sum()
                zero_count = (quantized_weight == 0).sum()
                pos_count = (quantized_weight == 1).sum()
                total = quantized_weight.size

                sparsity = zero_count / total

                layer_stats = {
                    'name': name,
                    'shape': original_weight.shape,
                    'original_range': (float(original_weight.min()), float(original_weight.max())),
                    'threshold': float(threshold),
                    'neg_count': int(neg_count),
                    'zero_count': int(zero_count),
                    'pos_count': int(pos_count),
                    'sparsity': float(sparsity)
                }

                stats['layers'].append(layer_stats)
                stats['total_params'] += total
                stats['quantized_params'] += (neg_count + pos_count)
                stats['sparsity'].append(sparsity)

                if verbose:
                    print(f"Layer {name}:")
                    print(f"  Shape: {original_weight.shape}")
                    print(f"  Original range: [{original_weight.min():.3f}, {original_weight.max():.3f}]")
                    print(f"  Threshold: {threshold:.3f}")
                    print(f"  Distribution: {neg_count} neg, {zero_count} zero, {pos_count} pos")
                    print(f"  Sparsity: {sparsity*100:.1f}%")

                # Replace weights with quantized version
                module.weight.data = torch.from_numpy(quantized_weight).float().to(module.weight.device)

        stats['avg_sparsity'] = float(np.mean(stats['sparsity']))

        return stats


class ModelQuantizationBenchmark:
    """
    Benchmark suite for model quantization to ternary

    Tests:
    1. Quantization statistics
    2. Accuracy degradation
    3. Inference speed
    4. Memory footprint
    5. Text generation quality
    """

    def __init__(self, output_dir: str = None):
        # Default to benchmarks/results/quantization/
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, "results", "quantization")

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'pytorch_available': TORCH_AVAILABLE,
                'transformers_available': TRANSFORMERS_AVAILABLE
            },
            'models': {}
        }

    def benchmark_model(
        self,
        model_name: str,
        quantization_strategy: str = "threshold"
    ) -> Dict[str, Any]:
        """
        Benchmark a specific model

        Args:
            model_name: HuggingFace model name
            quantization_strategy: Quantization strategy

        Returns:
            Benchmark results
        """
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            print("Error: PyTorch and Transformers required for model benchmarking")
            return self._mock_benchmark(model_name)

        print("\n" + "=" * 80)
        print(f"BENCHMARKING: {model_name}")
        print("=" * 80)

        results = {
            'model_name': model_name,
            'strategy': quantization_strategy,
            'stages': {}
        }

        # Load model
        print("\n[1/5] Loading model...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"  ✓ Model loaded: {model_name}")
        except Exception as e:
            print(f"  ✗ Failed to load model: {e}")
            return results

        # Measure original model
        print("\n[2/5] Measuring original model...")
        original_stats = self._measure_model(model, tokenizer)
        results['stages']['original'] = original_stats

        # Quantize model
        print("\n[3/5] Quantizing to ternary...")
        quantizer = TernaryQuantizer(strategy=quantization_strategy)
        quant_stats = quantizer.quantize_model_weights(model, verbose=True)
        results['stages']['quantization'] = quant_stats

        # Measure quantized model
        print("\n[4/5] Measuring quantized model...")
        quantized_stats = self._measure_model(model, tokenizer)
        results['stages']['quantized'] = quantized_stats

        # Test text generation
        print("\n[5/5] Testing text generation...")
        generation_test = self._test_generation(model, tokenizer)
        results['stages']['generation'] = generation_test

        # Calculate metrics
        results['metrics'] = self._calculate_metrics(original_stats, quantized_stats)

        return results

    def _measure_model(self, model, tokenizer) -> Dict[str, Any]:
        """Measure model statistics"""
        stats = {}

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        stats['total_params'] = total_params

        # Estimate memory (rough)
        # Each parameter stored as float32 = 4 bytes
        memory_bytes = total_params * 4
        stats['memory_mb'] = memory_bytes / (1024 * 1024)

        # Measure inference speed (simple forward pass)
        prompt = "Hello, how are you?"
        inputs = tokenizer(prompt, return_tensors="pt")

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model.generate(**inputs, max_length=20)

        # Measure
        import time
        iterations = 10
        start = time.perf_counter()
        for _ in range(iterations):
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=20)
        elapsed = time.perf_counter() - start

        stats['inference_time_ms'] = (elapsed / iterations) * 1000

        return stats

    def _test_generation(self, model, tokenizer) -> Dict[str, Any]:
        """Test text generation quality"""
        test_prompts = [
            "Once upon a time",
            "The capital of France is",
            "To be or not to be",
        ]

        results = []

        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt")

            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=50)

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            results.append({
                'prompt': prompt,
                'generated': generated_text
            })

            print(f"\n  Prompt: {prompt}")
            print(f"  Generated: {generated_text}")

        return {'samples': results}

    def _calculate_metrics(
        self,
        original_stats: Dict,
        quantized_stats: Dict
    ) -> Dict[str, float]:
        """Calculate comparison metrics"""
        metrics = {}

        # Inference speedup
        if 'inference_time_ms' in original_stats and 'inference_time_ms' in quantized_stats:
            metrics['inference_speedup'] = (
                original_stats['inference_time_ms'] /
                quantized_stats['inference_time_ms']
            )

        # Memory reduction
        if 'memory_mb' in original_stats and 'memory_mb' in quantized_stats:
            metrics['memory_reduction'] = (
                original_stats['memory_mb'] /
                quantized_stats['memory_mb']
            )

        return metrics

    def _mock_benchmark(self, model_name: str) -> Dict[str, Any]:
        """Mock benchmark when dependencies not available"""
        print(f"\nMock benchmark for {model_name}")
        print("(Install PyTorch and Transformers for real benchmarking)")

        return {
            'model_name': model_name,
            'status': 'mock',
            'note': 'Install pytorch and transformers for real benchmarking'
        }

    def save_results(self):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(
            self.output_dir,
            f"model_quantization_{timestamp}.json"
        )

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Results saved to {filename}")
        return filename

    def print_summary(self):
        """Print summary of all benchmarks"""
        print("\n" + "=" * 80)
        print("MODEL QUANTIZATION BENCHMARK SUMMARY")
        print("=" * 80)

        for model_name, result in self.results['models'].items():
            print(f"\n{model_name}:")

            if 'metrics' in result:
                metrics = result['metrics']
                if 'inference_speedup' in metrics:
                    print(f"  Inference speedup: {metrics['inference_speedup']:.2f}x")
                if 'memory_reduction' in metrics:
                    print(f"  Memory reduction:  {metrics['memory_reduction']:.2f}x")

            if 'stages' in result and 'quantization' in result['stages']:
                quant = result['stages']['quantization']
                if 'avg_sparsity' in quant:
                    print(f"  Average sparsity:  {quant['avg_sparsity']*100:.1f}%")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Model Quantization Benchmark'
    )
    parser.add_argument(
        '--model',
        default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        help='Model name or "all" for predefined list'
    )
    parser.add_argument(
        '--strategy',
        choices=['threshold', 'learned', 'adaptive'],
        default='threshold',
        help='Quantization strategy'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output directory (default: benchmarks/results/quantization/)'
    )

    args = parser.parse_args()

    benchmark = ModelQuantizationBenchmark(output_dir=args.output)

    if args.model == 'all':
        # Predefined model list
        models = [
            'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            'microsoft/phi-2',
        ]
    else:
        models = [args.model]

    for model_name in models:
        result = benchmark.benchmark_model(model_name, args.strategy)
        benchmark.results['models'][model_name] = result

    benchmark.save_results()
    benchmark.print_summary()


if __name__ == "__main__":
    main()
