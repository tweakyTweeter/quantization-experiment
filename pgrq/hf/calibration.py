"""
Calibration data collection for activation-aware quantization.

Collects activation statistics from a calibration dataset (WikiText-2 or C4).
"""

import torch
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


def collect_calibration_data(
    model,
    tokenizer,
    dataset_name: str = "wikitext",
    num_samples: int = 128,
    seq_length: int = 2048,
    device: str = "cuda",
    seed: int = 42,
) -> List[torch.Tensor]:
    """
    Collect calibration sequences from a dataset.
    
    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        dataset_name: "wikitext" or "c4".
        num_samples: Number of calibration sequences.
        seq_length: Sequence length.
        device: Device for data.
        seed: Random seed for sampling.
        
    Returns:
        calibration_data: List of input_ids tensors.
    """
    from datasets import load_dataset
    
    torch.manual_seed(seed)
    
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text_column = "text"
    elif dataset_name == "c4":
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
        text_column = "text"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    calibration_data = []
    
    # Collect text samples
    texts = []
    if dataset_name == "c4":
        # Streaming dataset
        for i, example in enumerate(dataset):
            if len(texts) >= num_samples * 2:  # Collect extra to filter
                break
            if len(example[text_column]) > 100:  # Skip very short texts
                texts.append(example[text_column])
    else:
        # Regular dataset
        for example in dataset:
            if len(example[text_column]) > 100:
                texts.append(example[text_column])
            if len(texts) >= num_samples * 2:
                break
    
    # Tokenize and create sequences
    for text in tqdm(texts[:num_samples], desc="Preparing calibration data"):
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=seq_length,
            padding=False,
        )
        
        input_ids = tokens["input_ids"]
        
        if input_ids.shape[1] >= seq_length // 2:  # At least half the target length
            # Pad if needed
            if input_ids.shape[1] < seq_length:
                padding = torch.full(
                    (1, seq_length - input_ids.shape[1]),
                    tokenizer.pad_token_id or tokenizer.eos_token_id,
                    dtype=input_ids.dtype,
                )
                input_ids = torch.cat([input_ids, padding], dim=1)
            else:
                input_ids = input_ids[:, :seq_length]
            
            calibration_data.append(input_ids.to(device))
        
        if len(calibration_data) >= num_samples:
            break
    
    return calibration_data


def compute_activation_stats(
    model,
    calibration_data: List[torch.Tensor],
    layer_names: Optional[List[str]] = None,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Compute activation statistics (E[a^2]) for each layer.
    
    Args:
        model: HuggingFace model.
        calibration_data: List of input_ids tensors.
        layer_names: Optional list of layer names to track.
        device: Device for computation.
        
    Returns:
        activation_stats: Dict mapping layer name to E[a^2] tensor.
    """
    activation_stats = {}
    activation_counts = {}
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            # Track input activations to linear layers
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input
            
            if x is None:
                return
            
            # Compute E[a^2] over batch and sequence dimensions
            # Shape: (batch, seq_len, hidden_dim) -> (hidden_dim,)
            x_sq = x.float().pow(2)
            mean_sq = x_sq.mean(dim=(0, 1))  # Average over batch and seq
            
            if name not in activation_stats:
                activation_stats[name] = mean_sq
                activation_counts[name] = 1
            else:
                # Running average
                activation_stats[name] = (
                    activation_stats[name] * activation_counts[name] + mean_sq
                ) / (activation_counts[name] + 1)
                activation_counts[name] += 1
        
        return hook
    
    # Register hooks on linear layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if layer_names is None or name in layer_names:
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)
    
    # Run forward passes on calibration data
    model.eval()
    with torch.no_grad():
        for input_ids in tqdm(calibration_data, desc="Computing activation stats"):
            try:
                _ = model(input_ids)
            except Exception as e:
                print(f"Warning: Forward pass failed: {e}")
                continue
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activation_stats


def get_linear_layers(model) -> Dict[str, torch.nn.Linear]:
    """
    Get all linear layers from a model.
    
    Args:
        model: HuggingFace model.
        
    Returns:
        layers: Dict mapping layer name to Linear module.
    """
    layers = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            layers[name] = module
    return layers


def should_quantize_layer(
    name: str,
    module: torch.nn.Module,
    config,
) -> bool:
    """
    Determine if a layer should be quantized based on config.
    
    Args:
        name: Layer name.
        module: Layer module.
        config: PGRQConfig.
        
    Returns:
        should_quantize: True if layer should be quantized.
    """
    # Check if it's a linear layer
    if not isinstance(module, torch.nn.Linear):
        return False
    
    # Check for embedding layers
    if "embed" in name.lower() or "wte" in name.lower() or "wpe" in name.lower():
        return config.quantize_embeddings
    
    # Check for LM head
    if "lm_head" in name.lower() or name.endswith(".head"):
        return config.quantize_lm_head
    
    # Check for layernorm-related (some models have linear layers for LN)
    if "layernorm" in name.lower() or "layer_norm" in name.lower() or "ln_" in name.lower():
        return config.quantize_layernorms
    
    # Default: quantize
    return True
