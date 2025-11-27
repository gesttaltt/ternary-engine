"""
Download models for quantization benchmarking

Downloads and caches models from HuggingFace for testing:
- TinyLlama-1.1B (primary test model)
- Phi-2 (optional)
- Gemma-2B (optional)

Usage:
    python download_models.py
    python download_models.py --model phi-2
    python download_models.py --all
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    print("Error: PyTorch and Transformers required")
    print("Install with: pip install torch transformers")
    sys.exit(1)


# Model configurations
MODELS = {
    'tinyllama': {
        'name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'size': '1.1B',
        'description': 'Small chat model, good for testing',
        'recommended': True
    },
    'phi-2': {
        'name': 'microsoft/phi-2',
        'size': '2.7B',
        'description': 'Microsoft small but capable model',
        'recommended': False
    },
    'gemma-2b': {
        'name': 'google/gemma-2b',
        'size': '2B',
        'description': 'Google small model',
        'recommended': False
    }
}


def download_model(model_key: str, cache_dir: str = None):
    """
    Download and cache a model

    Args:
        model_key: Key from MODELS dict
        cache_dir: Optional cache directory
    """
    if model_key not in MODELS:
        print(f"Error: Unknown model '{model_key}'")
        print(f"Available models: {', '.join(MODELS.keys())}")
        return False

    model_info = MODELS[model_key]
    model_name = model_info['name']

    print("=" * 80)
    print(f"DOWNLOADING: {model_name}")
    print("=" * 80)
    print(f"Size: {model_info['size']}")
    print(f"Description: {model_info['description']}")
    print()

    try:
        # Set cache directory if specified
        if cache_dir:
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            print(f"Cache directory: {cache_dir}")

        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✓ Tokenizer downloaded ({len(tokenizer)} tokens)")

        # Download model
        print("\nDownloading model...")
        print("This may take a few minutes...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model downloaded ({total_params:,} parameters)")

        # Quick test
        print("\nTesting model...")
        test_input = tokenizer("Hello, how are you?", return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(**test_input, max_length=20)

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test generation: {generated}")

        print("\n" + "=" * 80)
        print(f"SUCCESS: {model_name} ready for benchmarking")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\nError downloading model: {e}")
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Download models for quantization benchmarking'
    )
    parser.add_argument(
        '--model',
        choices=list(MODELS.keys()) + ['all'],
        default='tinyllama',
        help='Model to download (default: tinyllama)'
    )
    parser.add_argument(
        '--cache-dir',
        help='Cache directory for models (default: ~/.cache/huggingface)'
    )

    args = parser.parse_args()

    # Set up cache directory
    if args.cache_dir:
        cache_dir = os.path.abspath(args.cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
    else:
        cache_dir = None

    print("=" * 80)
    print("MODEL DOWNLOADER FOR TERNARY QUANTIZATION BENCHMARKS")
    print("=" * 80)
    print()

    # Download models
    if args.model == 'all':
        print("Downloading all models...")
        print()
        success_count = 0
        for model_key in MODELS.keys():
            if download_model(model_key, cache_dir):
                success_count += 1
            print()

        print(f"\nDownloaded {success_count}/{len(MODELS)} models successfully")

    else:
        download_model(args.model, cache_dir)

    print("\nNext steps:")
    print("  1. Run quantization benchmark:")
    print(f"     python bench_model_quantization.py --model {MODELS[args.model if args.model != 'all' else 'tinyllama']['name']}")
    print("  2. Check results in results/quantization/")


if __name__ == "__main__":
    main()
