"""
HuggingFace model integration for PGR-Q.

Provides functions to quantize and load HuggingFace models.
"""

import torch
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import os
from tqdm import tqdm

from pgrq.core.config import PGRQConfig
from pgrq.core.quantizer import PGRQuantizer, QuantizedTensor
from pgrq.hf.calibration import (
    collect_calibration_data,
    compute_activation_stats,
    get_linear_layers,
    should_quantize_layer,
)


def quantize_model(
    model_name_or_path: str,
    output_dir: str,
    config: Optional[PGRQConfig] = None,
    calibration_samples: int = 128,
    calibration_dataset: str = "wikitext",
    device: str = "auto",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Quantize a HuggingFace model using PGR-Q.
    
    Args:
        model_name_or_path: HuggingFace model ID or path.
        output_dir: Directory to save quantized model.
        config: PGRQConfig (uses defaults if None).
        calibration_samples: Number of calibration sequences.
        calibration_dataset: "wikitext" or "c4".
        device: Target device.
        verbose: Print progress.
        
    Returns:
        stats: Dict with quantization statistics.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    if config is None:
        config = PGRQConfig()
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Loading model: {model_name_or_path}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )
    
    if device == "cpu":
        model = model.to(device)
    
    if verbose:
        print(f"Model loaded on {device}")
    
    # Collect calibration data
    if verbose:
        print(f"Collecting {calibration_samples} calibration samples from {calibration_dataset}")
    
    calibration_data = collect_calibration_data(
        model,
        tokenizer,
        dataset_name=calibration_dataset,
        num_samples=calibration_samples,
        seq_length=config.calibration_seq_length,
        device=device,
        seed=config.seed,
    )
    
    # Compute activation statistics
    if config.activation_aware_importance:
        if verbose:
            print("Computing activation statistics...")
        activation_stats = compute_activation_stats(
            model, calibration_data, device=device
        )
    else:
        activation_stats = {}
    
    # Get layers to quantize
    linear_layers = get_linear_layers(model)
    layers_to_quantize = {
        name: module for name, module in linear_layers.items()
        if should_quantize_layer(name, module, config)
    }
    
    if verbose:
        print(f"Found {len(layers_to_quantize)} layers to quantize")
    
    # Collect all weights for codebook training
    if verbose:
        print("Training codebooks...")
    
    weights = [m.weight.data.float() for m in layers_to_quantize.values()]
    
    # Initialize and train quantizer
    quantizer = PGRQuantizer(config)
    quantizer.train_codebooks(weights, verbose=verbose)
    
    # Quantize each layer
    quantized_layers = {}
    total_stats = {
        "original_bytes": 0,
        "compressed_bytes": 0,
        "codebook_bytes": 0,
        "num_weights": 0,
        "layers": {},
    }
    
    if verbose:
        print("Quantizing layers...")
    
    for name, module in tqdm(layers_to_quantize.items(), desc="Quantizing", disable=not verbose):
        weight = module.weight.data.float()
        
        # Get activation stats for this layer
        act_stats = activation_stats.get(name)
        
        # Quantize
        quantized = quantizer.quantize(weight, activation_stats=act_stats, verbose=False)
        quantized_layers[name] = quantized
        
        # Collect stats
        layer_stats = quantizer.get_compression_stats(quantized)
        total_stats["layers"][name] = layer_stats
        total_stats["original_bytes"] += layer_stats["original_bytes"]
        total_stats["compressed_bytes"] += layer_stats["compressed_bytes"]
        total_stats["codebook_bytes"] += layer_stats["codebook_bytes"]
        total_stats["num_weights"] += layer_stats["num_weights"]
    
    # Compute overall stats
    total_bytes = total_stats["compressed_bytes"] + total_stats["codebook_bytes"]
    total_stats["total_bytes"] = total_bytes
    total_stats["compression_ratio"] = total_stats["original_bytes"] / total_bytes
    total_stats["bits_per_weight"] = total_bytes * 8 / total_stats["num_weights"]
    
    if verbose:
        print(f"\nQuantization complete:")
        print(f"  Original size: {total_stats['original_bytes'] / 1e6:.2f} MB")
        print(f"  Compressed size: {total_bytes / 1e6:.2f} MB")
        print(f"  Compression ratio: {total_stats['compression_ratio']:.2f}x")
        print(f"  Bits per weight: {total_stats['bits_per_weight']:.3f}")
    
    # Save quantized model
    if verbose:
        print(f"Saving to {output_dir}...")
    
    # Save config
    with open(output_path / "config.json", "w") as f:
        json.dump({
            "model_name_or_path": model_name_or_path,
            "pgrq_config": config.to_dict(),
            "stats": total_stats,
        }, f, indent=2)
    
    # Save quantized weights
    quantized_state = {}
    for name, qt in quantized_layers.items():
        quantized_state[name] = {
            "packed_data": qt.packed_data,
            "shape_info": qt.shape_info,
            "config": qt.config,
            "base_codebook_state": qt.base_codebook_state,
            "rvq_codebook_state": qt.rvq_codebook_state,
            "phase_corrector_state": qt.phase_corrector_state,
            "scalar_quantizer_state": qt.scalar_quantizer_state,
        }
    
    torch.save(quantized_state, output_path / "quantized_weights.pt")
    
    # Save non-quantized parts of the model
    non_quantized_state = {}
    for name, param in model.named_parameters():
        # Check if this is a weight we quantized
        is_quantized = False
        for qname in quantized_layers.keys():
            if name == qname + ".weight":
                is_quantized = True
                break
        
        if not is_quantized:
            non_quantized_state[name] = param.data.cpu()
    
    torch.save(non_quantized_state, output_path / "non_quantized_weights.pt")
    
    # Save tokenizer
    tokenizer.save_pretrained(output_path)
    
    return total_stats


def load_quantized_model(
    quantized_dir: str,
    device: str = "auto",
    verbose: bool = True,
):
    """
    Load a quantized model.
    
    Args:
        quantized_dir: Directory with quantized model.
        device: Target device.
        verbose: Print progress.
        
    Returns:
        model: Reconstructed model.
        tokenizer: Tokenizer.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    quantized_path = Path(quantized_dir)
    
    # Load config
    with open(quantized_path / "config.json", "r") as f:
        config_data = json.load(f)
    
    model_name = config_data["model_name_or_path"]
    pgrq_config = PGRQConfig.from_dict(config_data["pgrq_config"])
    
    if verbose:
        print(f"Loading quantized model from {quantized_dir}")
        print(f"Original model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(quantized_path, trust_remote_code=True)
    
    # Load model config and create empty model
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    # Create model with random weights (we'll replace them)
    with torch.no_grad():
        model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
        model = model.to(torch.float16).to(device)
    
    # Load non-quantized weights
    non_quantized = torch.load(quantized_path / "non_quantized_weights.pt", map_location=device)
    
    # Load and set non-quantized parameters
    state_dict = model.state_dict()
    for name, param in non_quantized.items():
        if name in state_dict:
            state_dict[name].copy_(param.to(device))
    
    # Load quantized weights
    quantized_state = torch.load(quantized_path / "quantized_weights.pt", map_location="cpu")
    
    # Initialize quantizer for dequantization
    quantizer = PGRQuantizer(pgrq_config)
    
    # Dequantize each layer
    if verbose:
        print(f"Dequantizing {len(quantized_state)} layers...")
    
    for name, qt_data in tqdm(quantized_state.items(), desc="Dequantizing", disable=not verbose):
        # Reconstruct QuantizedTensor
        quantized = QuantizedTensor(
            packed_data=qt_data["packed_data"],
            shape_info=qt_data["shape_info"],
            config=qt_data["config"],
            base_codebook_state=qt_data["base_codebook_state"],
            rvq_codebook_state=qt_data["rvq_codebook_state"],
            phase_corrector_state=qt_data["phase_corrector_state"],
            scalar_quantizer_state=qt_data["scalar_quantizer_state"],
        )
        
        # Dequantize
        weight = quantizer.dequantize(quantized, device=device)
        
        # Set in state dict
        weight_name = name + ".weight"
        if weight_name in state_dict:
            state_dict[weight_name].copy_(weight.to(torch.float16))
    
    model.load_state_dict(state_dict)
    
    if verbose:
        print("Model loaded successfully")
    
    return model, tokenizer


class PGRQModelWrapper:
    """
    Wrapper for quantized models that supports efficient inference.
    
    Keeps weights in quantized form and dequantizes on-the-fly during inference.
    This reduces memory usage but increases compute.
    """
    
    def __init__(
        self,
        quantized_dir: str,
        device: str = "auto",
    ):
        """
        Initialize wrapper.
        
        Args:
            quantized_dir: Directory with quantized model.
            device: Target device.
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.quantized_path = Path(quantized_dir)
        
        # Load config
        with open(self.quantized_path / "config.json", "r") as f:
            config_data = json.load(f)
        
        self.model_name = config_data["model_name_or_path"]
        self.pgrq_config = PGRQConfig.from_dict(config_data["pgrq_config"])
        
        # Load quantized state
        self.quantized_state = torch.load(
            self.quantized_path / "quantized_weights.pt", 
            map_location="cpu"
        )
        
        # Initialize quantizer
        self.quantizer = PGRQuantizer(self.pgrq_config)
        
        # Dequantized weight cache (optional, for repeated access)
        self._weight_cache = {}
    
    def get_weight(self, name: str, use_cache: bool = True) -> torch.Tensor:
        """
        Get dequantized weight for a layer.
        
        Args:
            name: Layer name.
            use_cache: Cache dequantized weight.
            
        Returns:
            weight: Dequantized weight tensor.
        """
        if use_cache and name in self._weight_cache:
            return self._weight_cache[name]
        
        qt_data = self.quantized_state[name]
        
        quantized = QuantizedTensor(
            packed_data=qt_data["packed_data"],
            shape_info=qt_data["shape_info"],
            config=qt_data["config"],
            base_codebook_state=qt_data["base_codebook_state"],
            rvq_codebook_state=qt_data["rvq_codebook_state"],
            phase_corrector_state=qt_data["phase_corrector_state"],
            scalar_quantizer_state=qt_data["scalar_quantizer_state"],
        )
        
        weight = self.quantizer.dequantize(quantized, device=self.device)
        
        if use_cache:
            self._weight_cache[name] = weight
        
        return weight
    
    def clear_cache(self):
        """Clear dequantized weight cache."""
        self._weight_cache = {}
