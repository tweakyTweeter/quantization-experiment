"""
AWQ implementation (Lin et al., 2023).

Core algorithm: Activation-Aware Weight Quantization.
1. Find salient weight channels via activation magnitudes
2. Apply per-channel scaling to protect salient weights
3. Quantize with standard RTN after scaling

The key insight is that some weight channels are much more important
(have larger activation magnitudes) and should be quantized more carefully.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, Tuple


class AWQ:
    """
    AWQ quantizer for a single linear layer.
    
    Steps:
    1. Compute channel importance from activation statistics
    2. Search for optimal per-channel scales that minimize quantization error
    3. Apply scales, quantize, then undo scales in the output
    """
    
    def __init__(
        self,
        layer: nn.Linear,
        bits: int = 4,
        group_size: int = 128,
        n_grid: int = 20,
        sym: bool = True,
    ):
        self.layer = layer
        self.bits = bits
        self.group_size = group_size
        self.n_grid = n_grid
        self.sym = sym
        
        self.rows = layer.weight.shape[0]
        self.cols = layer.weight.shape[1]
        
        # Activation statistics accumulator
        self.act_sum = torch.zeros(self.cols, dtype=torch.float32)
        self.nsamples = 0
    
    def add_batch(self, inp: torch.Tensor):
        """Add batch of inputs for activation statistics."""
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.float()
        self.act_sum += inp.abs().sum(dim=0)
        self.nsamples += inp.shape[0]
    
    def quantize(self) -> Tuple[torch.Tensor, Dict]:
        """
        Run AWQ quantization.
        
        Returns:
            quantized_weight: Dequantized weight tensor  
            info: Dict with metadata
        """
        W = self.layer.weight.data.float().clone()
        
        # Average activation magnitude per channel
        act_scales = self.act_sum / self.nsamples  # (cols,)
        
        # Find optimal scaling factors via grid search
        best_scales = torch.ones(self.cols, dtype=torch.float32)
        best_error = float('inf')
        
        # Process in groups
        Q_best = None
        
        for ratio in torch.linspace(0, 1, self.n_grid + 1)[1:]:  # skip 0
            # Scale = act_scales ^ ratio (protects high-activation channels)
            scales = act_scales.pow(ratio).clamp(min=1e-4)
            scales = scales / scales.mean()  # Normalize to avoid changing overall magnitude
            
            # Apply scaling: W_scaled = W * diag(scales)
            W_scaled = W * scales.unsqueeze(0)
            
            # Quantize with RTN (per group)
            Q_scaled = self._quantize_rtn(W_scaled)
            
            # Undo scaling: Q = Q_scaled / diag(scales)
            Q = Q_scaled / scales.unsqueeze(0)
            
            # Compute error weighted by activations
            err = ((W - Q) ** 2)
            # Weight by activation importance
            weighted_err = (err * act_scales.unsqueeze(0)).sum().item()
            
            if weighted_err < best_error:
                best_error = weighted_err
                best_scales = scales.clone()
                Q_best = Q.clone()
        
        if Q_best is None:
            Q_best = self._quantize_rtn(W)
        
        mse = ((W - Q_best) ** 2).mean().item()
        
        return Q_best, {
            'bits': self.bits,
            'group_size': self.group_size,
            'mse': mse,
            'best_ratio': None,
        }
    
    def _quantize_rtn(self, W: torch.Tensor) -> torch.Tensor:
        """Standard round-to-nearest quantization with per-group scales."""
        Q = torch.zeros_like(W)
        
        for g_start in range(0, self.cols, self.group_size):
            g_end = min(g_start + self.group_size, self.cols)
            group = W[:, g_start:g_end]
            
            if self.sym:
                maxval = group.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
                scale = maxval / (2**(self.bits-1) - 1)
                q = torch.clamp(
                    torch.round(group / scale),
                    -(2**(self.bits-1) - 1),
                    2**(self.bits-1) - 1
                )
                Q[:, g_start:g_end] = q * scale
            else:
                minval = group.min(dim=1, keepdim=True).values
                maxval = group.max(dim=1, keepdim=True).values
                scale = (maxval - minval).clamp(min=1e-8) / (2**self.bits - 1)
                zero = torch.round(-minval / scale)
                q = torch.clamp(
                    torch.round(group / scale + zero),
                    0, 2**self.bits - 1
                )
                Q[:, g_start:g_end] = (q - zero) * scale
        
        return Q


def quantize_model_awq(
    model,
    tokenizer,
    bits: int = 4,
    group_size: int = 128,
    n_samples: int = 128,
    seq_len: int = 512,
    verbose: bool = True,
) -> Dict:
    """
    Quantize a HuggingFace model using AWQ.
    
    Args:
        model: HuggingFace causal LM
        tokenizer: Tokenizer
        bits: Target bits
        group_size: Group size
        n_samples: Calibration samples
        seq_len: Sequence length
        verbose: Print progress
        
    Returns:
        stats: Quantization statistics
    """
    from datasets import load_dataset
    from tqdm import tqdm
    
    model.eval()
    device = next(model.parameters()).device
    
    # Calibration data
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in dataset["text"] if len(t.strip()) > 100][:n_samples]
    
    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=seq_len,
    )
    cal_input_ids = encodings.input_ids.to(device)
    
    # Collect layers
    layers_to_quantize = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if 'embed' not in name.lower() and 'lm_head' not in name.lower() and 'norm' not in name.lower():
                layers_to_quantize.append((name, module))
    
    if verbose:
        print(f"AWQ: Quantizing {len(layers_to_quantize)} layers at {bits}-bit (group_size={group_size})")
    
    # Collect activations
    layer_inputs = {}
    hooks = []
    
    def make_hook(name):
        def hook_fn(module, input, output):
            if name not in layer_inputs:
                layer_inputs[name] = []
            layer_inputs[name].append(input[0].detach().float())
        return hook_fn
    
    for name, module in layers_to_quantize:
        h = module.register_forward_hook(make_hook(name))
        hooks.append(h)
    
    with torch.no_grad():
        batch_size = 4
        for i in range(0, min(n_samples, cal_input_ids.shape[0]), batch_size):
            batch = cal_input_ids[i:i+batch_size]
            try:
                model(batch)
            except:
                pass
    
    for h in hooks:
        h.remove()
    
    # Quantize each layer
    total_mse = 0.0
    total_params = 0
    
    for name, module in tqdm(layers_to_quantize, desc="AWQ", disable=not verbose):
        if name not in layer_inputs or len(layer_inputs[name]) == 0:
            continue
        
        awq = AWQ(module, bits=bits, group_size=group_size)
        
        for inp in layer_inputs[name]:
            awq.add_batch(inp)
        del layer_inputs[name]
        
        Q, info = awq.quantize()
        module.weight.data = Q.to(module.weight.dtype)
        
        total_mse += info['mse'] * module.weight.numel()
        total_params += module.weight.numel()
    
    # BPW calculation
    num_groups = total_params / group_size
    total_bits = total_params * bits + num_groups * 16  # weights + scales
    bpw = total_bits / total_params
    
    stats = {
        'method': f'AWQ-{bits}b',
        'bits': bits,
        'group_size': group_size,
        'bpw': bpw,
        'mse': total_mse / total_params if total_params > 0 else 0,
        'total_params': total_params,
    }
    
    if verbose:
        print(f"  AWQ-{bits}b: BPW={bpw:.3f}, MSE={stats['mse']:.6f}")
    
    return stats
