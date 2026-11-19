#!/usr/bin/env python
"""
End-to-end test on a real HuggingFace model.

Tests the full pipeline: load -> quantize -> save -> load -> evaluate.
"""

import torch
import tempfile
import sys
from pathlib import Path


def test_real_model_weights():
    """Test quantization on real model weights (without full inference)."""
    print("=" * 60)
    print("Testing on Real Model Weights")
    print("=" * 60)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pgrq.core.config import PGRQConfig
    from pgrq.core.quantizer import PGRQuantizer
    
    model_name = "facebook/opt-125m"
    print(f"Loading model: {model_name}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for testing
        device_map=None,
    )
    
    # Extract linear layer weights
    weights = []
    weight_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if "embed" not in name.lower() and "lm_head" not in name.lower():
                weights.append(module.weight.data.clone())
                weight_names.append(name)
    
    print(f"Found {len(weights)} linear layers to quantize")
    total_params = sum(w.numel() for w in weights)
    print(f"Total parameters: {total_params:,}")
    
    # Configure quantizer
    config = PGRQConfig(
        block_size=64,
        num_groups=4,
        base_codebook_size=256,
        num_residual_levels=2,
        residual_codebook_size=256,
        joint_refinement=True,
        kmeans_iters=50,
        entropy_coding=True,
    )
    
    print("\nTraining codebooks...")
    quantizer = PGRQuantizer(config)
    # Use subset for faster training
    train_weights = weights[:10] if len(weights) > 10 else weights
    quantizer.train_codebooks(train_weights, verbose=True)
    
    print("\nQuantizing weights...")
    quantized_weights = []
    total_compressed = 0
    total_mse = 0.0
    
    for i, (name, w) in enumerate(zip(weight_names, weights)):
        qt = quantizer.quantize(w, verbose=False)
        quantized_weights.append(qt)
        
        stats = quantizer.get_compression_stats(qt)
        total_compressed += stats["total_bytes"]
        
        # Check reconstruction
        w_rec = quantizer.dequantize(qt)
        mse = ((w - w_rec) ** 2).mean().item()
        total_mse += mse * w.numel()
        
        if i < 3:
            print(f"  {name}: {w.shape}, MSE={mse:.6f}")
    
    avg_mse = total_mse / total_params
    bits_per_weight = total_compressed * 8 / total_params
    original_bytes = total_params * 4  # float32
    compression_ratio = original_bytes / total_compressed
    
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"Total parameters: {total_params:,}")
    print(f"Original size: {original_bytes / 1e6:.2f} MB")
    print(f"Compressed size: {total_compressed / 1e6:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Bits per weight: {bits_per_weight:.3f}")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Average RMSE: {avg_mse**0.5:.6f}")
    
    # Verify reconstruction is reasonable
    assert avg_mse < 0.5, f"MSE too high: {avg_mse}"
    assert bits_per_weight < 5.0, f"Bits too high: {bits_per_weight}"
    
    print("\nTest PASSED!")
    return True


def test_full_pipeline():
    """Test full quantize -> save -> load -> evaluate pipeline."""
    print("\n" + "=" * 60)
    print("Testing Full Pipeline")
    print("=" * 60)
    
    from pgrq.hf.integration import quantize_model, load_quantized_model
    
    model_name = "facebook/opt-125m"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "quantized"
        
        print(f"Quantizing {model_name}...")
        stats = quantize_model(
            model_name,
            str(output_dir),
            calibration_samples=8,  # Small for speed
            verbose=True,
        )
        
        print(f"\nQuantization stats:")
        print(f"  Bits per weight: {stats['bits_per_weight']:.3f}")
        print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
        
        print(f"\nLoading quantized model...")
        model, tokenizer = load_quantized_model(str(output_dir), verbose=True)
        
        print(f"\nTesting inference...")
        inputs = tokenizer("Hello, my name is", return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=30,
                do_sample=False,
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated}")
        
        print("\nFull pipeline test PASSED!")
        return True


if __name__ == "__main__":
    try:
        # Test weights only (faster)
        test_real_model_weights()
        
        # Optional: full pipeline test (slower)
        if "--full" in sys.argv:
            test_full_pipeline()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
